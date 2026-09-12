"""Disputes: the two a program can rule on, and the many it cannot."""

from __future__ import annotations

import pytest

from relay.disputes import (
    DEFAULT_MIN_STAKE,
    REASON_HASH_MISMATCH,
    REASON_OTHER,
    REASON_TOKEN_OVERCLAIM,
    REJECTED,
    SLASH_MULTIPLE,
    UNADJUDICATED,
    UPHELD,
    Dispute,
    adjudicate,
    adjudicate_open,
    open_dispute,
    set_stake,
    staked_providers,
)
from relay.identity import Identity
from relay.ledger import Ledger
from relay.provider.offers import build_offer
from relay.receipts import build_receipt, request_fingerprint
from relay.store import MemoryStore

PROMPT = "What is the smallest positive integer satisfying these congruences?"
SOLUTION = "The answer is 23, by the Chinese Remainder Theorem."
MAX_TOKENS = 512


@pytest.fixture
def world():
    """A completed step: a checkpoint, a settled receipt, and funded parties."""
    store = MemoryStore()
    ledger = Ledger(store)
    provider, consumer = Identity.generate(), Identity.generate()

    ledger.deposit(consumer.node_id, 100.0)
    ledger.deposit(provider.node_id, 50.0)
    ledger.hold(consumer.node_id, 90.0, "job-1")
    set_stake(store, ledger, provider.node_id, 10.0)

    offer = build_offer(
        provider,
        endpoint_url="http://p:8765",
        model="llama3",
        context_window=8192,
        price_in_per_1k=0.05,
        price_out_per_1k=0.15,
    )
    store.upsert_offer(offer.to_row())
    store.insert_checkpoint(
        session_id="job-1",
        worker_id="w1",
        step_number=1,
        problem=PROMPT,
        solution=SOLUTION,
        reasoning=SOLUTION,
        machine_id="m1",
        inference_node="n1",
        inference_latency_ms=100,
        tokens_used=30,
    )
    return store, ledger, provider, consumer, offer


def issue(world, **overrides):
    store, ledger, provider, consumer, offer = world
    from relay import tokens as tokenizer

    defaults = {
        "tokens_in": tokenizer.estimate(PROMPT).tokens,
        "tokens_out": tokenizer.estimate(SOLUTION).tokens,
    }
    defaults.update(overrides)
    receipt = build_receipt(
        provider,
        job_id="job-1",
        step_number=1,
        consumer_node_id=consumer.node_id,
        offer=offer,
        request_hash=overrides.pop("request_hash", None)
        or request_fingerprint(PROMPT, MAX_TOKENS, "llama3"),
        response_text=overrides.pop("response_text", SOLUTION),
        tokens_in=defaults["tokens_in"],
        tokens_out=defaults["tokens_out"],
        latency_ms=100,
    )
    acknowledged = receipt.acknowledged_by(consumer)
    store.upsert_receipt(acknowledged.to_row())
    ledger.settle(
        consumer.node_id,
        provider.node_id,
        acknowledged.amount_credits,
        receipt_id=acknowledged.receipt_id,
        job_id="job-1",
    )
    return acknowledged


# -- objective: token over-claim ------------------------------------------


def test_an_inflated_token_claim_is_upheld_refunded_and_slashed(world):
    store, ledger, provider, consumer, _ = world
    # Small enough that the provider can cover both the refund and the penalty.
    receipt = issue(world, tokens_out=50_000)
    before_provider = ledger.balance(provider.node_id)
    before_consumer = ledger.balance(consumer.node_id)

    dispute = open_dispute(
        store,
        receipt_id=receipt.receipt_id,
        opened_by=consumer.node_id,
        reason=REASON_TOKEN_OVERCLAIM,
        evidence={"max_tokens": MAX_TOKENS},
    )
    resolved = adjudicate(store, ledger, dispute)

    assert resolved.status == UPHELD
    assert resolved.evidence["output"]["claimed"] == 50_000

    refunded = receipt.amount_credits
    penalty = round(refunded * SLASH_MULTIPLE, 6)
    assert ledger.balance(consumer.node_id) == round(before_consumer + refunded, 6)
    assert ledger.balance(provider.node_id) == round(before_provider - refunded - penalty, 6)
    ledger.check_invariant()


def test_an_honest_token_claim_is_rejected(world):
    """Over-claiming must be caught without punishing honest counts."""
    store, ledger, provider, consumer, _ = world
    receipt = issue(world)
    before = ledger.balance(provider.node_id)

    dispute = open_dispute(
        store,
        receipt_id=receipt.receipt_id,
        opened_by=consumer.node_id,
        reason=REASON_TOKEN_OVERCLAIM,
        evidence={"max_tokens": MAX_TOKENS},
    )
    resolved = adjudicate(store, ledger, dispute)

    assert resolved.status == REJECTED
    assert ledger.balance(provider.node_id) == before, "the bill stands"
    ledger.check_invariant()


def test_a_rejected_dispute_restores_the_receipt(world):
    store, ledger, provider, consumer, _ = world
    receipt = issue(world)
    store.upsert_receipt({**receipt.to_row(), "status": "disputed", "dispute_reason": "wrong"})

    dispute = open_dispute(
        store,
        receipt_id=receipt.receipt_id,
        opened_by=consumer.node_id,
        reason=REASON_TOKEN_OVERCLAIM,
        evidence={"max_tokens": MAX_TOKENS},
    )
    adjudicate(store, ledger, dispute)
    assert store.get_receipt(receipt.receipt_id)["status"] != "disputed"


# -- objective: hash mismatch ---------------------------------------------


def test_a_receipt_for_a_response_we_did_not_get_is_upheld(world):
    store, ledger, provider, consumer, _ = world
    receipt = issue(world, response_text="a completely different answer")

    dispute = open_dispute(
        store,
        receipt_id=receipt.receipt_id,
        opened_by=consumer.node_id,
        reason=REASON_HASH_MISMATCH,
        evidence={"max_tokens": MAX_TOKENS},
    )
    resolved = adjudicate(store, ledger, dispute)

    assert resolved.status == UPHELD
    assert "response_hash" in resolved.evidence
    ledger.check_invariant()


def test_matching_hashes_are_rejected(world):
    store, ledger, provider, consumer, _ = world
    receipt = issue(world)

    dispute = open_dispute(
        store,
        receipt_id=receipt.receipt_id,
        opened_by=consumer.node_id,
        reason=REASON_HASH_MISMATCH,
        evidence={"max_tokens": MAX_TOKENS},
    )
    assert adjudicate(store, ledger, dispute).status == REJECTED


# -- what is deliberately not ruled on ------------------------------------


def test_a_subjective_dispute_is_recorded_not_ruled_on(world):
    """"The answer was poor" is not something a program should decide."""
    store, ledger, provider, consumer, _ = world
    receipt = issue(world)
    before = ledger.balance(provider.node_id)

    dispute = open_dispute(
        store,
        receipt_id=receipt.receipt_id,
        opened_by=consumer.node_id,
        reason=REASON_OTHER,
        evidence={"complaint": "the answer was unhelpful"},
    )
    resolved = adjudicate(store, ledger, dispute)

    assert resolved.status == UNADJUDICATED
    assert ledger.balance(provider.node_id) == before, "no credits moved on a subjective claim"


def test_a_dispute_against_an_unknown_receipt_is_rejected(world):
    store, ledger, *_ = world
    dispute = Dispute(receipt_id="nope", opened_by="x", reason=REASON_TOKEN_OVERCLAIM)
    assert adjudicate(store, ledger, dispute).status == REJECTED


def test_a_dispute_with_no_checkpoint_cannot_be_ruled_on(world):
    store, ledger, provider, consumer, _ = world
    receipt = issue(world)
    store.reset_session("job-1")
    store.upsert_receipt(receipt.to_row())

    dispute = open_dispute(
        store,
        receipt_id=receipt.receipt_id,
        opened_by=consumer.node_id,
        reason=REASON_TOKEN_OVERCLAIM,
    )
    resolved = adjudicate(store, ledger, dispute)
    assert resolved.status == REJECTED
    assert "no checkpoint" in resolved.evidence["finding"]


def test_adjudicating_everything_open_is_safe_to_repeat(world):
    store, ledger, provider, consumer, _ = world
    receipt = issue(world, tokens_out=500_000)
    open_dispute(
        store,
        receipt_id=receipt.receipt_id,
        opened_by=consumer.node_id,
        reason=REASON_TOKEN_OVERCLAIM,
        evidence={"max_tokens": MAX_TOKENS},
    )

    first = adjudicate_open(store, ledger)
    balance_after = ledger.balance(provider.node_id)
    second = adjudicate_open(store, ledger)

    assert len(first) == 1 and first[0].status == UPHELD
    assert second == [], "resolved disputes are not reopened"
    assert ledger.balance(provider.node_id) == balance_after, "not slashed twice"
    ledger.check_invariant()


# -- stake -----------------------------------------------------------------


def test_stake_cannot_exceed_what_you_hold():
    store = MemoryStore()
    ledger = Ledger(store)
    node = "n" * 64
    ledger.deposit(node, 5.0)
    assert set_stake(store, ledger, node, 100.0) == 5.0


def test_only_staked_providers_are_listed():
    store = MemoryStore()
    ledger = Ledger(store)
    rich, poor = "r" * 64, "p" * 64
    ledger.deposit(rich, 10.0)
    ledger.deposit(poor, 10.0)
    set_stake(store, ledger, rich, DEFAULT_MIN_STAKE)
    set_stake(store, ledger, poor, DEFAULT_MIN_STAKE / 10)

    listed = staked_providers(store, DEFAULT_MIN_STAKE)
    assert rich in listed
    assert poor not in listed


def test_a_slash_takes_a_provider_off_the_market(world):
    """The penalty has to be able to cost a provider its place in the market."""
    store, ledger, provider, consumer, _ = world
    set_stake(store, ledger, provider.node_id, DEFAULT_MIN_STAKE)
    receipt = issue(world, tokens_out=500_000)

    dispute = open_dispute(
        store,
        receipt_id=receipt.receipt_id,
        opened_by=consumer.node_id,
        reason=REASON_TOKEN_OVERCLAIM,
        evidence={"max_tokens": MAX_TOKENS},
    )
    adjudicate(store, ledger, dispute)

    assert ledger.balance(provider.node_id) >= 0, "a slash must not create a debt"
    account = store.get_account(provider.node_id)
    assert float(account["stake"]) <= ledger.balance(provider.node_id) + 1e-9
    assert provider.node_id not in staked_providers(store, DEFAULT_MIN_STAKE)
    ledger.check_invariant()


def test_a_provider_that_cannot_cover_the_refund_leaves_a_recorded_shortfall(world):
    """Stake exists because this case is possible. It must be visible, not
    silently papered over by inventing credits."""
    store, ledger, provider, consumer, _ = world
    receipt = issue(world, tokens_out=500_000)
    # By the time the dispute is heard, the provider has spent its earnings.
    ledger.slash(provider.node_id, ledger.balance(provider.node_id) - 1.0, reason="setup")

    dispute = open_dispute(
        store,
        receipt_id=receipt.receipt_id,
        opened_by=consumer.node_id,
        reason=REASON_TOKEN_OVERCLAIM,
        evidence={"max_tokens": MAX_TOKENS},
    )
    resolved = adjudicate(store, ledger, dispute)

    assert resolved.status == UPHELD
    assert resolved.evidence["shortfall"] > 0
    assert resolved.evidence["refunded"] < receipt.amount_credits
    assert ledger.balance(provider.node_id) >= 0
    ledger.check_invariant()


def test_an_understaked_provider_is_not_discoverable():
    from relay.consumer.market import Directory, Requirements

    store = MemoryStore()
    ledger = Ledger(store)
    identity = Identity.generate()
    store.upsert_offer(
        build_offer(
            identity,
            endpoint_url="http://x",
            model="llama3",
            context_window=8192,
            price_in_per_1k=0.01,
            price_out_per_1k=0.02,
        ).to_row()
    )

    assert Directory(store).find_offers(Requirements()), "listed with no stake requirement"
    assert Directory(store, min_stake=DEFAULT_MIN_STAKE).find_offers(Requirements()) == []

    ledger.deposit(identity.node_id, 10.0)
    set_stake(store, ledger, identity.node_id, DEFAULT_MIN_STAKE)
    assert Directory(store, min_stake=DEFAULT_MIN_STAKE).find_offers(Requirements())
