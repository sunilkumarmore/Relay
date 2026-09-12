"""The whole mechanism, end to end.

A provider that inflates its token counts should be caught by the consumer,
disputed, ruled against, slashed, and dropped from selection — without anyone
telling the system which provider was the dishonest one.
"""

from __future__ import annotations

import yaml

from relay.consumer.market import Directory, Requirements
from relay.disputes import DEFAULT_MIN_STAKE, adjudicate_open, set_stake, staked_providers
from relay.identity import Identity
from relay.inference.backends import FakeBackend
from relay.ledger import Ledger
from relay.reputation import DEFAULT_MIN_REPUTATION, recompute, reputations

THRESHOLD = DEFAULT_MIN_REPUTATION


class InflatingBackend(FakeBackend):
    def __init__(self, multiplier: int = 80, **kwargs):
        super().__init__(**kwargs)
        self.multiplier = multiplier

    def complete(self, prompt, max_tokens=1200, model=None):
        text, tokens_in, tokens_out = super().complete(prompt, max_tokens, model)
        return text, tokens_in, tokens_out * self.multiplier


def write_task(tmp_path, steps, requirements=None, name="task.yaml"):
    path = tmp_path / name
    body = {
        "goal": "Marketplace end to end",
        "steps": [{"topic": f"T{i}", "prompt": f"Prompt {i}"} for i in range(1, steps + 1)],
    }
    if requirements:
        body["requirements"] = requirements
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return str(path)


def fund(store, tmp_path, worker_id="worker-alpha", amount=500.0):
    identity = Identity.load_or_create(tmp_path / f"{worker_id}.key")
    Ledger(store).deposit(identity.node_id, amount)
    return identity


def test_a_lying_provider_is_caught_slashed_and_dropped(store, market, spawn_worker, tmp_path):
    ledger = Ledger(store)

    liar = market(
        store,
        name="liar",
        price_out=0.01,  # cheapest, so the consumer picks it
        backend=InflatingBackend(available_models=["llama3"]),
    )
    honest = market(store, name="honest", price_out=0.50)

    for provider in (liar, honest):
        ledger.deposit(provider.provider_node_id, 100.0)
        set_stake(store, ledger, provider.provider_node_id, 10.0)

    consumer = fund(store, tmp_path)

    worker = spawn_worker(
        store=store,
        registry_url="",
        session_id="job-e2e",
        task_path=write_task(tmp_path, 3, {"budget_credits": 300.0}),
    )
    assert worker.proc.wait(timeout=120) == 0, worker.proc.stdout.read()

    # 1. The consumer refused to countersign, and said why.
    receipts = store.list_receipts(job_id="job-e2e")
    assert receipts and all(r["status"] == "disputed" for r in receipts)

    # 2. It raised the objective kind of dispute, on its own.
    disputes = store.list_disputes()
    assert disputes, "the consumer never formally disputed anything"
    assert all(d["reason"] == "token_overclaim" for d in disputes)
    assert all(d["opened_by"] == consumer.node_id for d in disputes)

    # 3. Adjudication rules against the provider, from the signed evidence.
    balance_before = ledger.balance(liar.provider_node_id)
    resolved = adjudicate_open(store, ledger)
    assert resolved and all(d.status == "upheld" for d in resolved)
    assert ledger.balance(liar.provider_node_id) < balance_before, "not slashed"
    assert ledger.balance(liar.provider_node_id) >= 0
    ledger.check_invariant()

    # 4. One recompute is enough to price it out.
    recompute(store)
    scores = reputations(store)
    assert scores[liar.provider_node_id] < THRESHOLD
    assert scores[honest.provider_node_id] >= THRESHOLD

    # 5. And selection now skips it, without anyone naming it.
    offers = Directory(store).find_offers(Requirements(min_reputation=THRESHOLD))
    assert [o.provider_node_id for o in offers] == [honest.provider_node_id]


def test_an_honest_provider_keeps_its_place(store, market, spawn_worker, tmp_path):
    """The mechanism has to be safe for the people it is not aimed at."""
    ledger = Ledger(store)
    honest = market(store, name="honest", price_out=0.05)
    ledger.deposit(honest.provider_node_id, 100.0)
    set_stake(store, ledger, honest.provider_node_id, 10.0)
    fund(store, tmp_path)

    worker = spawn_worker(
        store=store,
        registry_url="",
        session_id="job-honest-e2e",
        task_path=write_task(tmp_path, 4, {"budget_credits": 100.0}),
    )
    assert worker.proc.wait(timeout=120) == 0, worker.proc.stdout.read()

    assert all(r["status"] == "acknowledged" for r in store.list_receipts(job_id="job-honest-e2e"))
    assert store.list_disputes() == []

    recompute(store)
    assert reputations(store)[honest.provider_node_id] >= THRESHOLD
    assert Directory(store).find_offers(Requirements(min_reputation=THRESHOLD))


def test_a_slashed_provider_can_fall_out_of_the_staked_set(store, market, spawn_worker, tmp_path):
    ledger = Ledger(store)
    liar = market(
        store, name="liar", price_out=0.20, backend=InflatingBackend(available_models=["llama3"])
    )
    # Barely staked: one upheld dispute should be enough to unlist it.
    ledger.deposit(liar.provider_node_id, DEFAULT_MIN_STAKE)
    set_stake(store, ledger, liar.provider_node_id, DEFAULT_MIN_STAKE)
    fund(store, tmp_path)

    assert liar.provider_node_id in staked_providers(store, DEFAULT_MIN_STAKE)

    worker = spawn_worker(
        store=store,
        registry_url="",
        session_id="job-stake",
        task_path=write_task(tmp_path, 2, {"budget_credits": 300.0}),
    )
    assert worker.proc.wait(timeout=120) == 0

    adjudicate_open(store, ledger)
    assert liar.provider_node_id not in staked_providers(store, DEFAULT_MIN_STAKE)
    assert Directory(store, min_stake=DEFAULT_MIN_STAKE).find_offers(Requirements()) == []
    ledger.check_invariant()


def test_sampled_verification_compares_two_providers(store, market, spawn_worker, tmp_path):
    """Sampling looks for patterns, so it records a finding rather than a verdict."""
    ledger = Ledger(store)
    first = market(store, name="p1", price_out=0.01)
    second = market(store, name="p2", price_out=0.50)
    for provider in (first, second):
        ledger.deposit(provider.provider_node_id, 100.0)
        set_stake(store, ledger, provider.provider_node_id, 10.0)
    fund(store, tmp_path)

    worker = spawn_worker(
        store=store,
        registry_url="",
        session_id="job-sample",
        task_path=write_task(tmp_path, 3, {"budget_credits": 100.0}),
        verify_sample_rate=1.0,
    )
    assert worker.proc.wait(timeout=120) == 0, worker.proc.stdout.read()

    findings = store.list_provider_events(kind="cross_checked")
    assert findings, "nothing was cross-checked"
    detail = findings[0]["detail"]
    # Both run the same deterministic backend, so the answers should agree.
    assert detail["identical"] is True
    assert detail["plausible"] is True
    assert detail["compared_with"] != detail["provider_node_id"]


def test_sampling_notices_a_provider_whose_output_does_not_match(store, market, spawn_worker, tmp_path):
    ledger = Ledger(store)
    liar = market(
        store, name="liar", price_out=0.01, backend=InflatingBackend(available_models=["llama3"])
    )
    honest = market(store, name="honest", price_out=0.50)
    for provider in (liar, honest):
        ledger.deposit(provider.provider_node_id, 100.0)
        set_stake(store, ledger, provider.provider_node_id, 10.0)
    fund(store, tmp_path)

    worker = spawn_worker(
        store=store,
        registry_url="",
        session_id="job-sample-liar",
        task_path=write_task(tmp_path, 2, {"budget_credits": 300.0}),
        verify_sample_rate=1.0,
    )
    assert worker.proc.wait(timeout=120) == 0

    findings = store.list_provider_events(kind="cross_checked")
    assert findings
    # The text matches (same deterministic backend) but the token count does not.
    assert any(f["detail"]["plausible"] is False for f in findings)
