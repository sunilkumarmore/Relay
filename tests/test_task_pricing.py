"""Offers and receipts for work that is not inference.

The load-bearing property here is that a receipt signed before tasks existed
still verifies. Offers expire in five minutes so their signed field list can
simply change; a receipt is a debt, has to be checkable years later, and so its
field list is versioned instead of edited.
"""

from __future__ import annotations

import pytest

from relay.identity import Identity
from relay.provider.offers import build_offer
from relay.receipts import (
    SCHEMA_V1,
    SCHEMA_V2,
    SIGNED_FIELDS_V1,
    SIGNED_FIELDS_V2,
    WORK_TASK,
    Receipt,
    task_fingerprint,
)


def inference_receipt(provider: Identity, consumer: Identity, **overrides) -> Receipt:
    fields = {
        "job_id": "job",
        "step_number": 1,
        "consumer_node_id": consumer.node_id,
        "provider_node_id": provider.node_id,
        "offer_id": "offer",
        "model": "llama",
        "request_hash": "a" * 64,
        "response_hash": "b" * 64,
        "tokens_in": 100,
        "tokens_out": 50,
        "latency_ms": 12,
        "price_in_per_1k": 0.1,
        "price_out_per_1k": 0.2,
        "amount_credits": 0.02,
    }
    fields.update(overrides)
    return Receipt(**fields).issued_by(provider)


def task_receipt(provider: Identity, consumer: Identity, **overrides) -> Receipt:
    return inference_receipt(
        provider,
        consumer,
        schema_version=SCHEMA_V2,
        work_kind=WORK_TASK,
        work_units=2_400_000,
        task_id="task-1",
        model="",
        step_number=0,
        tokens_in=0,
        tokens_out=0,
        price_in_per_1k=0.0,
        price_out_per_1k=0.0,
        amount_credits=1.2,
        **overrides,
    )


# -- offers -----------------------------------------------------------------


def test_a_node_with_no_model_can_still_make_an_offer():
    """A phone has no model on it. It can still sell arithmetic."""
    offer = build_offer(
        Identity.generate(),
        endpoint_url="",
        task_types=("matmul_block",),
        price_per_mega_unit=0.5,
    )
    assert offer.signature_is_valid()
    assert offer.model == ""
    assert offer.context_window == 0
    assert offer.serves_task_type("matmul_block")
    assert not offer.serves_task_type("python_exec")


def test_an_inference_offer_is_unchanged_by_the_pivot():
    offer = build_offer(
        Identity.generate(),
        endpoint_url="http://node",
        model="llama",
        context_window=8192,
        price_in_per_1k=0.1,
        price_out_per_1k=0.2,
    )
    assert offer.signature_is_valid()
    assert offer.price(1000, 500) == pytest.approx(0.2)
    assert offer.task_types == ()


def test_task_price_is_a_function_of_the_signed_quantity():
    """work_units is fixed by the task and re-derived from the payload, so
    neither side can argue about the quantity after the fact."""
    offer = build_offer(
        Identity.generate(),
        endpoint_url="",
        task_types=("matmul_block",),
        price_per_task=0.01,
        price_per_mega_unit=0.5,
    )
    assert offer.task_price(2_000_000) == pytest.approx(1.01)
    assert offer.task_price(0) == pytest.approx(0.01)


def test_changing_a_term_invalidates_the_offer():
    offer = build_offer(
        Identity.generate(), endpoint_url="", task_types=("matmul_block",), price_per_mega_unit=0.5
    )
    assert not offer.model_copy(update={"price_per_mega_unit": 0.01}).signature_is_valid()
    assert not offer.model_copy(update={"task_types": ("python_exec",)}).signature_is_valid()


# -- receipt versioning -----------------------------------------------------


def test_a_v1_receipt_still_verifies_after_the_schema_grew():
    """The whole point of versioning rather than editing. A receipt that stops
    verifying is a debt nobody can prove."""
    receipt = inference_receipt(Identity.generate(), Identity.generate())
    assert receipt.schema_version == SCHEMA_V1
    assert receipt.provider_signature_is_valid()
    assert Receipt.from_row(receipt.to_row()).provider_signature_is_valid()


def test_v1_receipts_are_signed_over_exactly_the_fields_they_always_were():
    assert SIGNED_FIELDS_V2[: len(SIGNED_FIELDS_V1)] == SIGNED_FIELDS_V1
    assert "work_units" not in SIGNED_FIELDS_V1
    assert "work_units" in SIGNED_FIELDS_V2


def test_a_task_receipt_verifies_and_is_countersigned():
    provider, consumer = Identity.generate(), Identity.generate()
    receipt = task_receipt(provider, consumer)
    assert receipt.provider_signature_is_valid()
    acknowledged = receipt.acknowledged_by(consumer)
    assert acknowledged.is_acknowledged


def test_editing_the_billed_work_units_breaks_the_signature():
    receipt = task_receipt(Identity.generate(), Identity.generate())
    assert not receipt.model_copy(update={"work_units": 1}).provider_signature_is_valid()


def test_a_v1_and_a_v2_receipt_never_hash_the_same():
    """Otherwise a v1 signature could be replayed onto a v2 receipt carrying
    whatever work_units the holder liked."""
    provider, consumer = Identity.generate(), Identity.generate()
    assert (
        inference_receipt(provider, consumer).canonical_bytes()
        != task_receipt(provider, consumer).canonical_bytes()
    )


def test_a_receipt_claiming_an_unknown_schema_verifies_as_nothing():
    receipt = task_receipt(Identity.generate(), Identity.generate())
    assert not receipt.model_copy(update={"schema_version": 99}).provider_signature_is_valid()


def test_the_task_fingerprint_ties_a_receipt_to_the_order_that_was_signed():
    first = task_fingerprint("t1", "p" * 64, 100)
    assert first == task_fingerprint("t1", "p" * 64, 100)
    assert first != task_fingerprint("t1", "p" * 64, 101)
    assert first != task_fingerprint("t2", "p" * 64, 100)
