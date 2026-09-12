"""Receipts: what a consumer checks before paying."""

from __future__ import annotations

import pytest

from relay.identity import Identity
from relay.provider.offers import build_offer
from relay.receipts import (
    STATUS_ACKNOWLEDGED,
    STATUS_DISPUTED,
    Receipt,
    build_receipt,
    hash_text,
    price_for,
    request_fingerprint,
    verify_receipt,
)

PROMPT = "Explain the Chinese Remainder Theorem."
RESPONSE = "Here is the explanation."
MAX_TOKENS = 512


@pytest.fixture
def parties():
    return Identity.generate(), Identity.generate()  # provider, consumer


@pytest.fixture
def offer(parties):
    provider, _ = parties
    return build_offer(
        provider,
        endpoint_url="http://10.0.0.5:8765",
        model="llama3",
        context_window=8192,
        price_in_per_1k=0.05,
        price_out_per_1k=0.15,
    )


@pytest.fixture
def receipt(parties, offer):
    provider, consumer = parties
    return build_receipt(
        provider,
        job_id="job-1",
        step_number=3,
        consumer_node_id=consumer.node_id,
        offer=offer,
        request_hash=request_fingerprint(PROMPT, MAX_TOKENS, "llama3"),
        response_text=RESPONSE,
        tokens_in=2000,
        tokens_out=1000,
        latency_ms=1234,
    )


def check(receipt, offer, consumer, **overrides):
    kwargs = {
        "offer": offer,
        "consumer_node_id": consumer.node_id,
        "request_hash": request_fingerprint(PROMPT, MAX_TOKENS, "llama3"),
        "response_text": RESPONSE,
        "job_id": "job-1",
        "step_number": 3,
    }
    kwargs.update(overrides)
    return verify_receipt(receipt, **kwargs)


# -- the happy path --------------------------------------------------------


def test_a_well_formed_receipt_has_no_objections(receipt, offer, parties):
    _, consumer = parties
    assert check(receipt, offer, consumer) == []
    assert receipt.provider_signature_is_valid()


def test_amount_is_priced_from_the_offer(receipt):
    # 2000 in at 0.05 + 1000 out at 0.15
    assert receipt.amount_credits == pytest.approx(0.25)
    assert receipt.amount_credits == price_for(0.05, 0.15, 2000, 1000)


def test_countersigning_acknowledges_it(receipt, parties):
    _, consumer = parties
    acknowledged = receipt.acknowledged_by(consumer)
    assert acknowledged.status == STATUS_ACKNOWLEDGED
    assert acknowledged.consumer_signature_is_valid()
    assert acknowledged.is_acknowledged
    # Countersigning does not disturb the provider's signature.
    assert acknowledged.provider_signature_is_valid()


def test_a_receipt_starts_unacknowledged(receipt):
    assert receipt.is_acknowledged is False


def test_only_the_named_parties_can_sign(receipt, parties):
    provider, _ = parties
    stranger = Identity.generate()
    with pytest.raises(ValueError, match="acknowledged by the consumer"):
        receipt.acknowledged_by(stranger)
    with pytest.raises(ValueError, match="issued by the provider"):
        receipt.model_copy().issued_by(stranger)


# -- what the check catches ------------------------------------------------


def test_an_unsigned_receipt_is_rejected(receipt, offer, parties):
    _, consumer = parties
    forged = receipt.model_copy(update={"provider_signature": ""})
    assert "provider signature is not valid" in check(forged, offer, consumer)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tokens_in", 20_000),
        ("tokens_out", 50_000),
        ("amount_credits", 9.99),
        ("price_out_per_1k", 1.50),
        ("latency_ms", 1),
        ("response_hash", hash_text("something else")),
    ],
)
def test_editing_any_number_breaks_the_signature(receipt, offer, parties, field, value):
    _, consumer = parties
    tampered = receipt.model_copy(update={field: value})
    assert "provider signature is not valid" in check(tampered, offer, consumer)


def test_inflated_tokens_are_caught_even_when_resigned(receipt, offer, parties):
    """A provider can always sign its own lie — so the check is arithmetic, not
    just cryptography."""
    provider, consumer = parties
    inflated = receipt.model_copy(update={"tokens_out": 100_000}).issued_by(provider)
    problems = check(inflated, offer, consumer)
    assert inflated.provider_signature_is_valid(), "authentically signed"
    assert any("does not match" in p for p in problems), "but the arithmetic gives it away"


def test_overcharging_at_honest_token_counts_is_caught(receipt, offer, parties):
    provider, consumer = parties
    overcharged = receipt.model_copy(update={"amount_credits": 2.50}).issued_by(provider)
    assert any("does not match" in p for p in check(overcharged, offer, consumer))


def test_a_price_that_does_not_match_the_offer_is_caught(receipt, offer, parties):
    provider, consumer = parties
    gouged = receipt.model_copy(
        update={"price_out_per_1k": 1.50, "amount_credits": price_for(0.05, 1.50, 2000, 1000)}
    ).issued_by(provider)
    problems = check(gouged, offer, consumer)
    assert "output price does not match the offer" in problems


def test_a_response_we_did_not_receive_is_caught(receipt, offer, parties):
    _, consumer = parties
    problems = check(receipt, offer, consumer, response_text="a different answer")
    assert "response hash does not match what we received" in problems


def test_a_request_we_did_not_send_is_caught(receipt, offer, parties):
    _, consumer = parties
    problems = check(
        receipt, offer, consumer, request_hash=request_fingerprint("other prompt", MAX_TOKENS, "llama3")
    )
    assert "request hash does not match what we sent" in problems


def test_changing_max_tokens_changes_the_request_hash():
    a = request_fingerprint(PROMPT, 512, "llama3")
    b = request_fingerprint(PROMPT, 4096, "llama3")
    assert a != b, "billing must be tied to what was actually asked for"


def test_a_receipt_for_another_step_is_caught(receipt, offer, parties):
    _, consumer = parties
    assert "receipt is for a different step" in check(receipt, offer, consumer, step_number=4)


def test_a_receipt_naming_another_consumer_is_caught(receipt, offer, parties):
    _, consumer = parties
    assert "receipt names a different consumer" in check(
        receipt, offer, Identity.generate()
    ) or True
    problems = verify_receipt(
        receipt,
        offer=offer,
        consumer_node_id=Identity.generate().node_id,
        request_hash=request_fingerprint(PROMPT, MAX_TOKENS, "llama3"),
        response_text=RESPONSE,
        job_id="job-1",
        step_number=3,
    )
    assert "receipt names a different consumer" in problems


def test_a_receipt_citing_another_offer_is_caught(receipt, parties):
    provider, consumer = parties
    other_offer = build_offer(
        provider,
        endpoint_url="http://10.0.0.5:8765",
        model="llama3",
        context_window=8192,
        price_in_per_1k=0.05,
        price_out_per_1k=0.15,
    )
    problems = check(receipt, other_offer, consumer)
    assert "receipt cites a different offer" in problems


def test_negative_tokens_are_rejected(receipt, offer, parties):
    provider, consumer = parties
    weird = receipt.model_copy(
        update={"tokens_in": -5, "amount_credits": price_for(0.05, 0.15, -5, 1000)}
    ).issued_by(provider)
    assert "token counts cannot be negative" in check(weird, offer, consumer)


# -- storage ---------------------------------------------------------------


def test_round_trips_through_a_store_row(receipt, parties):
    _, consumer = parties
    acknowledged = receipt.acknowledged_by(consumer)
    restored = Receipt.from_row(acknowledged.to_row())
    assert restored == acknowledged
    assert restored.provider_signature_is_valid()
    assert restored.consumer_signature_is_valid()


def test_disputing_records_the_reason(receipt):
    disputed = receipt.disputed("output tokens claimed 99999")
    assert disputed.status == STATUS_DISPUTED
    assert "99999" in disputed.dispute_reason
    assert disputed.is_acknowledged is False
