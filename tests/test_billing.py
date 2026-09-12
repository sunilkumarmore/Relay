"""Money, end to end: a real job, real receipts, real settlement.

The point of these tests is that the numbers add up on both sides afterwards —
the provider is paid exactly the sum of the receipts the consumer countersigned,
and the ledger still sums to zero.
"""

from __future__ import annotations

import yaml

from relay.identity import Identity
from relay.inference.backends import FakeBackend
from relay.ledger import Ledger
from relay.receipts import Receipt


def write_task(tmp_path, steps: int, requirements: dict | None = None, name="task.yaml") -> str:
    path = tmp_path / name
    body: dict = {
        "goal": "Billing test",
        "steps": [{"topic": f"T{i}", "prompt": f"Prompt number {i}"} for i in range(1, steps + 1)],
    }
    if requirements:
        body["requirements"] = requirements
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return str(path)


def consumer_identity(tmp_path, worker_id="worker-alpha") -> Identity:
    # Same path the spawned worker will use, so we can fund it in advance.
    return Identity.load_or_create(tmp_path / f"{worker_id}.key")


class InflatingBackend(FakeBackend):
    """A provider that bills for more tokens than it produced."""

    def __init__(self, multiplier: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.multiplier = multiplier

    def complete(self, prompt, max_tokens=1200, model=None):
        text, tokens_in, tokens_out = super().complete(prompt, max_tokens, model)
        return text, tokens_in, tokens_out * self.multiplier


def test_a_job_produces_one_acknowledged_receipt_per_step(store, market, spawn_worker, tmp_path):
    provider = market(store, name="p1", price_out=0.10, price_in=0.02)
    consumer = consumer_identity(tmp_path)
    Ledger(store).deposit(consumer.node_id, 100.0)

    worker = spawn_worker(
        store=store,
        registry_url="",
        session_id="job-bill",
        task_path=write_task(tmp_path, 3, {"budget_credits": 10.0}),
    )
    assert worker.proc.wait(timeout=90) == 0, worker.proc.stdout.read()

    receipts = store.list_receipts(job_id="job-bill")
    assert len(receipts) == 3
    assert {r["step_number"] for r in receipts} == {1, 2, 3}
    assert all(r["status"] == "acknowledged" for r in receipts)

    for row in receipts:
        receipt = Receipt.from_row(row)
        assert receipt.provider_signature_is_valid(), "provider signature"
        assert receipt.consumer_signature_is_valid(), "consumer countersignature"
        assert receipt.provider_node_id == provider.provider_node_id
        assert receipt.consumer_node_id == consumer.node_id


def test_the_provider_is_paid_exactly_the_sum_of_its_receipts(store, market, spawn_worker, tmp_path):
    provider = market(store, name="p1", price_out=0.10, price_in=0.02)
    consumer = consumer_identity(tmp_path)
    ledger = Ledger(store)
    ledger.deposit(consumer.node_id, 100.0)

    worker = spawn_worker(
        store=store,
        registry_url="",
        session_id="job-pay",
        task_path=write_task(tmp_path, 4, {"budget_credits": 10.0}),
    )
    assert worker.proc.wait(timeout=90) == 0, worker.proc.stdout.read()

    billed = sum(float(r["amount_credits"]) for r in store.list_receipts(job_id="job-pay"))
    assert billed > 0, "the work was not billed at all"
    assert ledger.balance(provider.provider_node_id) == round(billed, 6)
    assert ledger.balance(consumer.node_id) == round(100.0 - billed, 6)
    assert ledger.held_balance(consumer.node_id) == 0.0, "the unspent budget came back"
    ledger.check_invariant()


def test_a_job_with_no_budget_cannot_start(store, market, spawn_worker, tmp_path):
    market(store, name="p1")
    consumer_identity(tmp_path)  # funded with nothing

    worker = spawn_worker(
        store=store,
        registry_url="",
        session_id="job-broke",
        task_path=write_task(tmp_path, 2, {"budget_credits": 5.0}),
    )
    assert worker.proc.wait(timeout=90) == 1
    output = worker.proc.stdout.read()
    assert "credits" in output
    assert store.get_checkpoints("job-broke") == [], "no provider capacity was consumed"


def test_a_provider_inflating_tokens_is_disputed_and_not_paid(store, market, spawn_worker, tmp_path):
    """The consumer counts for itself, and refuses to countersign a claim far
    outside what it measured."""
    liar = market(
        store,
        name="liar",
        price_out=0.10,
        backend=InflatingBackend(multiplier=50, available_models=["llama3"]),
    )
    consumer = consumer_identity(tmp_path)
    ledger = Ledger(store)
    ledger.deposit(consumer.node_id, 100.0)

    worker = spawn_worker(
        store=store,
        registry_url="",
        session_id="job-liar",
        task_path=write_task(tmp_path, 2, {"budget_credits": 10.0}),
    )
    assert worker.proc.wait(timeout=90) == 0, worker.proc.stdout.read()

    receipts = store.list_receipts(job_id="job-liar")
    assert receipts, "the work still happened and was still billed for"
    assert all(r["status"] == "disputed" for r in receipts)
    assert any("output tokens claimed" in r["dispute_reason"] for r in receipts)

    assert ledger.balance(liar.provider_node_id) == 0.0, "a disputed receipt is not paid"
    ledger.check_invariant()


def test_an_honest_provider_is_not_disputed(store, market, spawn_worker, tmp_path):
    """The bound has to be wide enough not to accuse honest providers."""
    market(store, name="honest", price_out=0.10)
    consumer = consumer_identity(tmp_path)
    Ledger(store).deposit(consumer.node_id, 100.0)

    worker = spawn_worker(
        store=store,
        registry_url="",
        session_id="job-honest",
        task_path=write_task(tmp_path, 4, {"budget_credits": 10.0}),
    )
    assert worker.proc.wait(timeout=90) == 0

    receipts = store.list_receipts(job_id="job-honest")
    assert len(receipts) == 4
    assert [r["status"] for r in receipts] == ["acknowledged"] * 4


def test_a_pinned_endpoint_with_no_offer_bills_nothing(store, registry_server, spawn_worker, task_file):
    """The pre-market arrangement still works, and still charges nothing."""
    server = registry_server(store)
    worker = spawn_worker(
        store=store, registry_url=server.url, session_id="job-free", task_path=task_file(2)
    )
    assert worker.proc.wait(timeout=90) == 0
    assert store.list_receipts(job_id="job-free") == []
    Ledger(store).check_invariant()


def test_receipts_survive_eviction_and_resume(store, market, spawn_worker, tmp_path):
    import signal

    provider = market(store, name="p1", price_out=0.10, latency_ms=400)
    consumer = consumer_identity(tmp_path)
    ledger = Ledger(store)
    ledger.deposit(consumer.node_id, 100.0)
    task = write_task(tmp_path, 5, {"budget_credits": 10.0})

    worker = spawn_worker(store=store, registry_url="", session_id="job-evict", task_path=task)
    assert worker.wait_for_checkpoints(1)
    worker.proc.send_signal(signal.SIGTERM)
    worker.proc.wait(timeout=30)

    paid_so_far = ledger.balance(provider.provider_node_id)
    assert paid_so_far > 0, "work done before eviction was still paid for"

    # The same consumer identity resumes the job.
    resumed = spawn_worker(
        store=store, registry_url="", session_id="job-evict", task_path=task
    )
    assert resumed.proc.wait(timeout=90) == 0, resumed.proc.stdout.read()

    receipts = store.list_receipts(job_id="job-evict")
    assert len({r["step_number"] for r in receipts}) == 5, "one receipt per step, no gaps"
    billed = sum(float(r["amount_credits"]) for r in receipts if r["status"] == "acknowledged")
    assert ledger.balance(provider.provider_node_id) == round(billed, 6)
    ledger.check_invariant()


def test_provider_can_see_what_it_earned(store, market, spawn_worker, tmp_path):
    provider = market(store, name="p1", price_out=0.10)
    consumer = consumer_identity(tmp_path)
    Ledger(store).deposit(consumer.node_id, 100.0)

    worker = spawn_worker(
        store=store,
        registry_url="",
        session_id="job-earn",
        task_path=write_task(tmp_path, 3, {"budget_credits": 10.0}),
    )
    assert worker.proc.wait(timeout=90) == 0

    earnings = provider.registry.earnings()
    assert earnings["receipts"] == 3
    assert earnings["acknowledged"] > 0
    assert earnings["disputed"] == 0
