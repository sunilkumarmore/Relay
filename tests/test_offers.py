"""Offers: signed, expiring statements of terms."""

from __future__ import annotations

from datetime import timedelta

import pytest

from relay.identity import Identity
from relay.provider.offers import Offer, build_offer, now_utc

TERMS = {
    "endpoint_url": "http://10.0.0.5:8765",
    "model": "llama3",
    "context_window": 8192,
    "price_in_per_1k": 0.05,
    "price_out_per_1k": 0.15,
    "max_concurrency": 2,
    "region": "home-lab",
}


def test_offer_is_signed_by_its_provider():
    identity = Identity.generate()
    offer = build_offer(identity, **TERMS)
    assert offer.provider_node_id == identity.node_id
    assert offer.signature_is_valid() is True
    assert offer.is_usable() is True


def test_unsigned_offer_is_not_usable():
    identity = Identity.generate()
    offer = build_offer(identity, **TERMS).model_copy(update={"signature": ""})
    assert offer.signature_is_valid() is False
    assert offer.is_usable() is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("price_out_per_1k", 0.001),   # undercut the advertised price
        ("price_in_per_1k", 0.0),
        ("context_window", 999_999),   # claim a bigger window than agreed
        ("endpoint_url", "http://attacker.example"),
        ("model", "some-other-model"),
        ("max_concurrency", 1000),
        ("expires_at", (now_utc() + timedelta(days=365)).isoformat()),
    ],
)
def test_every_term_is_covered_by_the_signature(field, value):
    """If a term could be edited in the directory without breaking the
    signature, it would not be a commitment."""
    offer = build_offer(Identity.generate(), **TERMS)
    assert offer.model_copy(update={field: value}).signature_is_valid() is False


def test_a_provider_cannot_sign_for_another_node():
    alice, mallory = Identity.generate(), Identity.generate()
    offer = build_offer(alice, **TERMS)
    with pytest.raises(ValueError, match="signed by the node that makes it"):
        offer.signed_by(mallory)


def test_forged_signature_from_another_key_fails():
    alice, mallory = Identity.generate(), Identity.generate()
    offer = build_offer(alice, **TERMS)
    forged = offer.model_copy(
        update={"price_out_per_1k": 0.0, "signature": mallory.sign(offer.canonical_bytes()).hex()}
    )
    assert forged.signature_is_valid() is False


def test_offers_expire():
    identity = Identity.generate()
    past = now_utc() - timedelta(hours=1)
    offer = build_offer(identity, **TERMS, ttl_seconds=60, at=past)
    assert offer.signature_is_valid() is True, "still authentic"
    assert offer.is_expired() is True, "but no longer a live commitment"
    assert offer.is_usable() is False


def test_offer_with_no_expiry_is_treated_as_expired():
    offer = Offer(provider_node_id="x", **TERMS)
    assert offer.is_expired() is True


def test_price_is_per_thousand_tokens_and_splits_in_from_out():
    offer = build_offer(Identity.generate(), **TERMS)
    # 2000 in at 0.05, 1000 out at 0.15
    assert offer.price(2000, 1000) == pytest.approx(0.25)
    assert offer.price(0, 0) == 0.0
    # Output costs more than input, so the same token count is not the same price.
    assert offer.price(1000, 0) < offer.price(0, 1000)


def test_price_is_stable_to_the_millicredit():
    """Both sides must arrive at the same number from the same inputs."""
    offer = build_offer(Identity.generate(), **TERMS)
    assert offer.price(333, 777) == offer.price(333, 777)
    assert isinstance(offer.price(1, 1), float)


def test_round_trips_through_a_store_row():
    offer = build_offer(Identity.generate(), **TERMS)
    restored = Offer.from_row(offer.to_row())
    assert restored == offer
    assert restored.signature_is_valid() is True


def test_from_row_ignores_columns_it_does_not_know():
    offer = build_offer(Identity.generate(), **TERMS)
    row = {**offer.to_row(), "id": "db-uuid", "inserted_at": "whenever"}
    assert Offer.from_row(row).signature_is_valid() is True
