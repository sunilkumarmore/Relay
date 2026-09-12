"""Reputation: deterministic, checkable, and fair to newcomers."""

from __future__ import annotations

from datetime import timedelta

from relay.consumer.market import Directory, Requirements
from relay.identity import Identity
from relay.provider.offers import build_offer, now_utc
from relay.reputation import (
    CONSUMER,
    DEFAULT_WINDOW,
    PROVIDER,
    recompute,
    reputations,
    score_provider,
)
from relay.store import MemoryStore

# Real keypairs: an offer only counts as advertising if its signature verifies,
# so a made-up node id cannot stand in here.
GOOD_ID = Identity.generate()
BAD_ID = Identity.generate()
GOOD = GOOD_ID.node_id
BAD = BAD_ID.node_id
IDENTITIES = {GOOD: GOOD_ID, BAD: BAD_ID}
CONSUMER_ID = "c" * 64


def receipt_row(provider, status, *, step=1, issued_at=None, consumer=CONSUMER_ID):
    return {
        "receipt_id": f"{provider[:4]}-{status}-{step}",
        "job_id": "job-1",
        "step_number": step,
        "consumer_node_id": consumer,
        "provider_node_id": provider,
        "offer_id": "o1",
        "amount_credits": 1.0,
        "status": status,
        "issued_at": (issued_at or now_utc()).isoformat(),
    }


def health_row(provider, ok, latency=100, observed_at=None):
    return {
        "observer_node_id": "obs",
        "provider_node_id": provider,
        "ok": ok,
        "latency_ms": latency,
        "observed_at": (observed_at or now_utc()).isoformat(),
    }


def advertise(store, provider, identity=None):
    """Publish a live, validly signed offer.

    The score counts whether a node is currently advertising, so a test about
    anything else has to hold that constant.
    """
    offer = build_offer(
        identity or IDENTITIES[provider],
        endpoint_url="http://x",
        model="llama3",
        context_window=8192,
        price_in_per_1k=0.01,
        price_out_per_1k=0.02,
    )
    store.upsert_offer(offer.to_row())


def seed(
    store,
    provider,
    *,
    acknowledged=0,
    disputed=0,
    ok=0,
    failed=0,
    latency=100,
    offer=True,
    identity=None,
):
    if offer:
        advertise(store, provider, identity)
    for i in range(acknowledged):
        store.upsert_receipt(receipt_row(provider, "acknowledged", step=i + 1))
    for i in range(disputed):
        store.upsert_receipt(receipt_row(provider, "disputed", step=100 + i))
    for _ in range(ok):
        store.insert_provider_health("obs", provider, True, latency)
    for _ in range(failed):
        store.insert_provider_health("obs", provider, False, None, "boom")


def test_no_history_scores_at_the_prior_not_at_zero():
    """Being new is not the same as being bad."""
    score = score_provider(
        GOOD, receipts=[], health=[], offers=[], market_median_latency=None, at=now_utc()
    )
    assert 0.3 < score.score < 0.7


def test_a_clean_record_scores_high():
    store = MemoryStore()
    seed(store, GOOD, acknowledged=40, ok=40)
    scores = {s.node_id: s.score for s in recompute(store) if s.role == PROVIDER}
    assert scores[GOOD] > 0.75


def test_a_disputed_record_scores_low():
    store = MemoryStore()
    seed(store, GOOD, acknowledged=40, ok=40)
    # Available and answering, but its bills keep getting refused.
    seed(store, BAD, acknowledged=2, disputed=30, ok=30)
    scores = {s.node_id: s.score for s in recompute(store) if s.role == PROVIDER}
    assert scores[BAD] < scores[GOOD]
    assert scores[BAD] < 0.4


def test_the_score_is_deterministic():
    """A reputation nobody can recompute is one you have to take on faith."""
    store = MemoryStore()
    seed(store, GOOD, acknowledged=12, disputed=3, ok=20, failed=2)
    at = now_utc()

    first = recompute(store, at=at)
    second = recompute(store, at=at)
    assert [s.score for s in first] == [s.score for s in second]
    assert [s.components for s in first] == [s.components for s in second]


def test_components_show_the_working():
    store = MemoryStore()
    seed(store, GOOD, acknowledged=10, disputed=2, ok=15, failed=1)
    score = next(s for s in recompute(store) if s.node_id == GOOD and s.role == PROVIDER)

    assert score.components["integrity"]["acknowledged"] == 10
    assert score.components["integrity"]["disputed"] == 2
    assert score.components["reliability"]["observations"] == 16
    assert score.components["advertising"]["value"] in (0.0, 1.0)


def test_old_evidence_falls_out_of_the_window():
    store = MemoryStore()
    stale = now_utc() - DEFAULT_WINDOW - timedelta(days=1)
    for i in range(30):
        store.upsert_receipt(receipt_row(BAD, "disputed", step=i, issued_at=stale))

    score = next(s for s in recompute(store) if s.node_id == BAD and s.role == PROVIDER)
    assert score.components["integrity"]["disputed"] == 0, "ancient history is not the present"


def test_smoothing_stops_two_good_calls_outranking_two_hundred():
    store = MemoryStore()
    seed(store, GOOD, acknowledged=200, ok=200)
    seed(store, BAD, acknowledged=2, ok=2)
    scores = {s.node_id: s.score for s in recompute(store) if s.role == PROVIDER}
    assert scores[GOOD] > scores[BAD]


def test_billing_fraud_is_not_redeemed_by_good_service():
    """Being quick and always up while overcharging is worse, not better."""
    store = MemoryStore()
    seed(store, GOOD, acknowledged=30, ok=30, latency=50)
    seed(store, BAD, acknowledged=0, disputed=30, ok=30, latency=50)

    scores = {s.node_id: s for s in recompute(store) if s.role == PROVIDER}
    assert scores[BAD].components["service"]["value"] > 0.5, "its service really was fine"
    assert scores[BAD].score < 0.3, "and that does not save it"
    assert scores[GOOD].score > scores[BAD].score * 2


def test_one_bad_receipt_among_many_barely_registers():
    store = MemoryStore()
    seed(store, GOOD, acknowledged=50, disputed=1, ok=50)
    score = next(s for s in recompute(store) if s.node_id == GOOD and s.role == PROVIDER)
    assert score.components["integrity"]["value"] > 0.95


def test_latency_is_scored_relative_to_the_market():
    store = MemoryStore()
    seed(store, GOOD, acknowledged=10, ok=10, latency=50)
    seed(store, BAD, acknowledged=10, ok=10, latency=5000)
    scores = {s.node_id: s for s in recompute(store) if s.role == PROVIDER}

    assert scores[GOOD].components["latency"]["value"] > scores[BAD].components["latency"]["value"]


def test_latency_is_skipped_rather_than_penalised_when_unmeasured():
    """A component we cannot measure is dropped from the mean, not scored zero."""
    measured = MemoryStore()
    seed(measured, GOOD, acknowledged=10, ok=10)
    with_latency = next(s for s in recompute(measured) if s.node_id == GOOD and s.role == PROVIDER)

    unmeasured = MemoryStore()
    seed(unmeasured, GOOD, acknowledged=10)
    without = next(s for s in recompute(unmeasured) if s.node_id == GOOD and s.role == PROVIDER)

    assert without.components["latency"]["value"] is None
    assert with_latency.components["latency"]["value"] is not None
    assert without.score > 0.5, "a missing signal must not drag the score down"


def test_advertising_counts_for_a_live_offer():
    store = MemoryStore()
    identity = Identity.generate()
    offer = build_offer(
        identity,
        endpoint_url="http://x",
        model="llama3",
        context_window=8192,
        price_in_per_1k=0.01,
        price_out_per_1k=0.02,
    )
    store.upsert_offer(offer.to_row())

    score = next(s for s in recompute(store) if s.node_id == identity.node_id and s.role == PROVIDER)
    assert score.components["advertising"]["value"] == 1.0


def test_consumers_are_scored_too():
    store = MemoryStore()
    store.upsert_receipt(receipt_row(GOOD, "acknowledged", step=1))
    store.upsert_receipt(receipt_row(GOOD, "acknowledged", step=2))
    scores = {s.node_id: s.score for s in recompute(store) if s.role == CONSUMER}
    assert CONSUMER_ID in scores
    assert scores[CONSUMER_ID] > 0.4


def test_a_consumer_whose_disputes_keep_failing_scores_lower():
    store = MemoryStore()
    for i in range(6):
        store.upsert_receipt(receipt_row(GOOD, "acknowledged", step=i))
    honest = recompute(store)
    honest_score = next(s.score for s in honest if s.role == CONSUMER)

    for i in range(6):
        store.upsert_dispute(
            {
                "dispute_id": f"d{i}",
                "receipt_id": f"{GOOD[:4]}-acknowledged-{i}",
                "opened_by": CONSUMER_ID,
                "reason": "token_overclaim",
                "status": "rejected",
                "evidence": {},
                "opened_at": now_utc().isoformat(),
            }
        )
    crier = next(s.score for s in recompute(store) if s.role == CONSUMER)
    assert crier < honest_score, "crying wolf should cost something"


def test_scores_are_persisted_for_selection():
    store = MemoryStore()
    seed(store, GOOD, acknowledged=20, ok=20)
    recompute(store)
    assert reputations(store)[GOOD] > 0.5
    assert store.get_reputation(GOOD, PROVIDER) is not None


def test_selection_skips_a_provider_with_a_bad_record():
    store = MemoryStore()
    good_id = Identity.generate()
    bad_id = Identity.generate()
    for identity in (good_id, bad_id):
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
    seed(store, good_id.node_id, acknowledged=40, ok=40, offer=False)
    seed(store, bad_id.node_id, acknowledged=1, disputed=40, ok=40, offer=False)
    recompute(store)

    found = Directory(store).find_offers(Requirements(min_reputation=0.45))
    assert [o.provider_node_id for o in found] == [good_id.node_id]
