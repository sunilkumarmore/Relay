"""Discovery and selection: who should I ask?"""

from __future__ import annotations

from datetime import timedelta

import pytest

from relay.consumer.market import (
    CHEAPEST,
    FASTEST,
    PINNED,
    ROUND_ROBIN,
    Bench,
    Directory,
    Requirements,
    Selector,
)
from relay.identity import Identity
from relay.provider.offers import build_offer, now_utc
from relay.store import MemoryStore


def publish(store, *, price_out=0.10, price_in=0.02, model="llama3", context_window=8192,
            region="lab", ttl=300, at=None, identity=None, endpoint=None):
    identity = identity or Identity.generate()
    offer = build_offer(
        identity,
        endpoint_url=endpoint or f"http://{identity.node_id[:8]}:8765",
        model=model,
        context_window=context_window,
        price_in_per_1k=price_in,
        price_out_per_1k=price_out,
        region=region,
        ttl_seconds=ttl,
        at=at,
    )
    store.upsert_offer(offer.to_row())
    return offer


# -- the directory ---------------------------------------------------------


def test_finds_published_offers():
    store = MemoryStore()
    publish(store)
    publish(store)
    assert len(Directory(store).all_offers()) == 2


def test_expired_offers_are_invisible():
    store = MemoryStore()
    live = publish(store)
    publish(store, ttl=60, at=now_utc() - timedelta(hours=1))

    found = Directory(store).all_offers()
    assert [o.offer_id for o in found] == [live.offer_id]


def test_offers_with_a_broken_signature_are_ignored():
    """A directory row anyone could have edited is not a commitment."""
    store = MemoryStore()
    good = publish(store)
    tampered = publish(store)
    row = next(r for r in store.list_offers() if r["offer_id"] == tampered.offer_id)
    row["price_out_per_1k"] = 0.0001
    store.upsert_offer(row)

    found = Directory(store).all_offers()
    assert [o.offer_id for o in found] == [good.offer_id]


def test_unparseable_rows_do_not_break_discovery():
    store = MemoryStore()
    good = publish(store)
    store.upsert_offer({"offer_id": "junk", "provider_node_id": "x"})
    assert [o.offer_id for o in Directory(store).all_offers()] == [good.offer_id]


def test_no_store_means_no_offers():
    assert Directory(None).all_offers() == []


# -- requirements ----------------------------------------------------------


def test_filters_by_model():
    store = MemoryStore()
    publish(store, model="llama3")
    publish(store, model="mistral")
    found = Directory(store).find_offers(Requirements(model="mistral"))
    assert [o.model for o in found] == ["mistral"]


def test_filters_by_price_ceiling():
    store = MemoryStore()
    publish(store, price_out=0.05)
    publish(store, price_out=0.50)
    found = Directory(store).find_offers(Requirements(max_price_out_per_1k=0.10))
    assert [o.price_out_per_1k for o in found] == [0.05]


def test_filters_by_input_price_separately():
    store = MemoryStore()
    publish(store, price_in=0.01, price_out=0.90)
    found = Directory(store).find_offers(Requirements(max_price_in_per_1k=0.02))
    assert len(found) == 1, "a cheap input price should qualify on its own terms"


def test_filters_by_context_window():
    store = MemoryStore()
    publish(store, context_window=4096)
    big = publish(store, context_window=32768)
    found = Directory(store).find_offers(Requirements(min_context_window=8192))
    assert [o.offer_id for o in found] == [big.offer_id]


def test_region_is_a_preference_not_a_filter():
    """A quiet region should not make the job fail."""
    store = MemoryStore()
    publish(store, region="eu")
    found = Directory(store).find_offers(Requirements(region="us"))
    assert len(found) == 1


def test_region_orders_results():
    store = MemoryStore()
    publish(store, region="eu", price_out=0.01)
    near = publish(store, region="us", price_out=0.50)
    ranked = Selector(CHEAPEST).order(
        Directory(store).find_offers(Requirements(region="us")), prefer_region="us"
    )
    assert ranked[0].offer_id == near.offer_id, "local first, even at a higher price"


def test_min_reputation_excludes_unproven_providers():
    store = MemoryStore()
    trusted = publish(store)
    publish(store)
    found = Directory(store).find_offers(
        Requirements(min_reputation=0.5), reputations={trusted.provider_node_id: 0.9}
    )
    assert [o.offer_id for o in found] == [trusted.offer_id]


def test_requirements_from_yaml_dict():
    req = Requirements.from_dict(
        {"model": "llama3", "max_price_out_per_1k": 0.2, "min_context_window": 8192, "budget_credits": 5}
    )
    assert req.model == "llama3"
    assert req.max_price_out_per_1k == 0.2
    assert req.budget_credits == 5.0


def test_empty_requirements_accept_everything():
    store = MemoryStore()
    publish(store, price_out=99.0, context_window=1)
    assert len(Directory(store).find_offers(Requirements.from_dict(None))) == 1


# -- policies --------------------------------------------------------------


def test_cheapest_picks_the_lowest_output_price():
    store = MemoryStore()
    publish(store, price_out=0.50)
    cheap = publish(store, price_out=0.01)
    publish(store, price_out=0.20)
    assert Selector(CHEAPEST).select(Directory(store).all_offers()).offer_id == cheap.offer_id


def test_cheapest_breaks_ties_on_input_price():
    store = MemoryStore()
    publish(store, price_out=0.10, price_in=0.09)
    better = publish(store, price_out=0.10, price_in=0.01)
    assert Selector(CHEAPEST).select(Directory(store).all_offers()).offer_id == better.offer_id


def test_fastest_uses_observed_latency():
    store = MemoryStore()
    slow = publish(store, price_out=0.01)
    quick = publish(store, price_out=0.99)
    for _ in range(3):
        store.insert_provider_health("obs", slow.provider_node_id, True, 900)
        store.insert_provider_health("obs", quick.provider_node_id, True, 40)

    directory = Directory(store)
    chosen = Selector(FASTEST).select(
        directory.all_offers(), latencies=directory.latency_by_provider()
    )
    assert chosen.offer_id == quick.offer_id, "fastest should outrank cheapest under this policy"


def test_fastest_ignores_failed_observations():
    store = MemoryStore()
    a = publish(store)
    store.insert_provider_health("obs", a.provider_node_id, False, 5, "boom")
    assert Directory(store).latency_by_provider() == {}


def test_fastest_gives_unmeasured_providers_a_chance():
    """A new provider with no history should not be frozen out."""
    store = MemoryStore()
    slow = publish(store)
    newcomer = publish(store)
    for _ in range(3):
        store.insert_provider_health("obs", slow.provider_node_id, True, 5000)

    directory = Directory(store)
    ranked = Selector(FASTEST).order(directory.all_offers(), latencies=directory.latency_by_provider())
    assert ranked[0].offer_id == newcomer.offer_id


def test_round_robin_spreads_across_providers():
    store = MemoryStore()
    for _ in range(3):
        publish(store)
    offers = Directory(store).all_offers()

    selector = Selector(ROUND_ROBIN)
    picked = [selector.select(offers).offer_id for _ in range(6)]
    assert len(set(picked)) == 3, "round robin should not keep picking the same node"
    assert picked[:3] == picked[3:], "and it should cycle"


def test_pinned_selects_only_the_named_node():
    store = MemoryStore()
    publish(store, price_out=0.001)
    wanted = publish(store, price_out=0.99)
    chosen = Selector(PINNED, pinned_node_id=wanted.provider_node_id).select(
        Directory(store).all_offers()
    )
    assert chosen.offer_id == wanted.offer_id, "pinned beats cheap, that is the point"


def test_pinned_to_an_absent_node_selects_nothing():
    store = MemoryStore()
    publish(store)
    assert Selector(PINNED, pinned_node_id="nobody").select(Directory(store).all_offers()) is None


def test_unknown_policy_is_rejected_at_construction():
    with pytest.raises(ValueError, match="Unknown policy"):
        Selector("vibes")


def test_selecting_from_nothing_returns_none():
    assert Selector(CHEAPEST).select([]) is None


# -- the bench -------------------------------------------------------------


def test_a_failing_provider_is_benched_then_returns():
    bench = Bench(cooldown_seconds=60)
    bench.penalize("p1")
    assert bench.is_benched("p1") is True
    assert bench.is_benched("p2") is False
    # Cooldown elapses.
    assert bench.is_benched("p1", at=now_utc() + timedelta(seconds=61)) is False


def test_bench_filters_the_candidate_list():
    store = MemoryStore()
    bad = publish(store)
    good = publish(store)
    bench = Bench()
    bench.penalize(bad.provider_node_id)
    remaining = bench.filter(Directory(store).all_offers())
    assert [o.offer_id for o in remaining] == [good.offer_id]
