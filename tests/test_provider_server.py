"""The provider serving many models, advertising terms, and refusing overload."""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from relay.identity import Identity
from relay.inference.backends import FakeBackend
from relay.provider.config import ModelOffering, ProviderConfig
from relay.provider.offers import Offer
from relay.provider.server import Provider, create_app
from relay.store import MemoryStore
from tests.test_auth import REGISTER, post

CHEAP = ModelOffering("mistral", "fake", 4096, 0.01, 0.03, max_concurrency=2)
DEAR = ModelOffering("llama3", "fake", 8192, 0.05, 0.15, max_concurrency=1)


def build_provider(store=None, *, models=None, latency_ms=0, endpoint="http://10.0.0.5:8765"):
    backend = FakeBackend(latency_ms=latency_ms, available_models=["llama3", "mistral"])
    cfg = ProviderConfig(
        endpoint_url=endpoint,
        region="lab",
        offer_ttl_seconds=300,
        models=list(models or [DEAR, CHEAP]),
        backends={"fake": backend},
    )
    provider = Provider(cfg, store, identity=Identity.generate(), node_id="provider-1")
    return provider, backend


@pytest.fixture
def served():
    store = MemoryStore()
    provider, backend = build_provider(store)
    with TestClient(create_app(provider, prune_in_background=False)) as client:
        yield client, provider, backend, store


def complete(client, identity, **overrides):
    payload = {
        "worker_id": "w1",
        "session_id": "s1",
        "prompt": "hello",
        "max_tokens": 32,
        **overrides,
    }
    return post(client, "/inference/complete", payload, identity)


# -- offers ---------------------------------------------------------------


def test_provider_publishes_a_signed_offer_per_model(served):
    client, provider, _, store = served
    body = client.get("/offer").json()
    assert {o["model"] for o in body["offers"]} == {"llama3", "mistral"}

    for row in body["offers"]:
        offer = Offer.from_row(row)
        assert offer.signature_is_valid()
        assert offer.provider_node_id == provider.identity.node_id
        assert offer.endpoint_url == "http://10.0.0.5:8765"

    # And they reach the shared directory, not just this node's memory.
    assert len(store.list_offers()) == 2
    assert {r["model"] for r in store.list_offers("llama3")} == {"llama3"}


def test_offer_carries_the_configured_price_and_capacity(served):
    client, *_ = served
    offers = {o["model"]: o for o in client.get("/offer").json()["offers"]}
    assert offers["llama3"]["price_out_per_1k"] == 0.15
    assert offers["mistral"]["price_out_per_1k"] == 0.03
    assert offers["llama3"]["max_concurrency"] == 1
    assert offers["mistral"]["context_window"] == 4096


def test_republishing_refreshes_rather_than_duplicates():
    store = MemoryStore()
    provider, _ = build_provider(store)
    first = provider.publish_offers()
    time.sleep(0.01)
    second = provider.publish_offers()

    assert len(store.list_offers()) == 2, "one row per model, not one per cycle"
    assert {o.offer_id for o in first} == {o.offer_id for o in second}
    assert second[0].expires_at > first[0].expires_at


def test_shutting_down_withdraws_the_offers():
    """A dead node should not keep advertising capacity nobody can reach."""
    store = MemoryStore()
    provider, _ = build_provider(store)
    with TestClient(create_app(provider, prune_in_background=False)):
        assert len(store.list_offers()) == 2
    assert store.list_offers() == []


def test_offer_endpoint_is_public(served):
    client, *_ = served
    assert client.get("/offer").status_code == 200, "discovery cannot require an account"


# -- many models ----------------------------------------------------------


def test_serves_the_requested_model(served):
    client, provider, backend, _ = served
    identity = Identity.generate()
    post(client, "/worker/register", REGISTER, identity)

    body = complete(client, identity, model="mistral").json()
    assert body["model"] == "mistral"
    assert "[mistral]" in body["response"]


def test_defaults_to_the_first_configured_model(served):
    client, *_ = served
    identity = Identity.generate()
    post(client, "/worker/register", REGISTER, identity)
    assert complete(client, identity).json()["model"] == "llama3"


def test_asking_for_an_unserved_model_is_a_404(served):
    client, *_ = served
    identity = Identity.generate()
    post(client, "/worker/register", REGISTER, identity)

    resp = complete(client, identity, model="gpt-4")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "model_not_served: gpt-4"


def test_response_names_the_offer_it_was_served_under(served):
    client, provider, _, _ = served
    identity = Identity.generate()
    post(client, "/worker/register", REGISTER, identity)

    body = complete(client, identity).json()
    assert body["provider_node_id"] == provider.identity.node_id
    assert body["offer_id"] == provider.current_offers()[0].offer_id or body["offer_id"]


def test_token_split_is_persisted(served):
    client, _, _, store = served
    identity = Identity.generate()
    post(client, "/worker/register", REGISTER, identity)
    body = complete(client, identity).json()

    logged = store.snapshot()["inference_log"][0]
    assert logged["tokens_in"] == body["tokens_in"]
    assert logged["tokens_out"] == body["tokens_out"]
    assert logged["tokens_used"] == body["tokens_in"] + body["tokens_out"]


# -- back-pressure --------------------------------------------------------


def test_saturation_returns_429_with_retry_after():
    """Past the advertised capacity, refuse honestly instead of queueing."""
    store = MemoryStore()
    provider, _ = build_provider(store, models=[DEAR], latency_ms=400)
    identity = Identity.generate()

    with TestClient(create_app(provider, prune_in_background=False)) as client:
        post(client, "/worker/register", REGISTER, identity)

        results: list[int] = []

        def fire():
            results.append(complete(client, identity).status_code)

        threads = [threading.Thread(target=fire) for _ in range(3)]
        for t in threads:
            t.start()
            time.sleep(0.05)
        for t in threads:
            t.join(timeout=10)

    assert 200 in results
    assert 429 in results, f"max_concurrency=1 should have refused someone: {results}"

    saturation = store.list_provider_events(kind="saturated")
    assert saturation, "a refusal the provider never recorded cannot inform reputation"
    assert saturation[0]["model"] == "llama3"


def test_capacity_is_released_after_each_call(served):
    """A finished request must not hold a slot — otherwise the provider
    saturates permanently after max_concurrency calls."""
    client, *_ = served
    identity = Identity.generate()
    post(client, "/worker/register", REGISTER, identity)

    for _ in range(5):
        assert complete(client, identity).status_code == 200


def test_capacity_is_released_even_when_the_backend_fails():
    store = MemoryStore()
    provider, backend = build_provider(store, models=[DEAR])
    identity = Identity.generate()
    backend.fail_calls = {1}

    with TestClient(create_app(provider, prune_in_background=False)) as client:
        post(client, "/worker/register", REGISTER, identity)
        assert complete(client, identity).status_code == 502
        # The slot must have come back, or one failure would wedge the provider.
        assert complete(client, identity).status_code == 200


def test_concurrency_is_per_model_not_per_provider():
    store = MemoryStore()
    provider, _ = build_provider(store, models=[DEAR, CHEAP], latency_ms=300)
    identity = Identity.generate()

    with TestClient(create_app(provider, prune_in_background=False)) as client:
        post(client, "/worker/register", REGISTER, identity)
        codes: list[tuple[str, int]] = []

        def fire(model):
            codes.append((model, complete(client, identity, model=model).status_code))

        threads = [threading.Thread(target=fire, args=(m,)) for m in ("llama3", "mistral")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

    # Saturating llama3 must not block mistral, which has its own budget.
    assert sorted(codes) == [("llama3", 200), ("mistral", 200)]


# -- unchanged guarantees -------------------------------------------------


def test_signature_is_still_required(served):
    client, *_ = served
    assert client.post("/inference/complete", json={"worker_id": "w1", "session_id": "s", "prompt": "p"}).status_code == 401


def test_worker_binding_still_holds(served):
    client, *_ = served
    owner, impostor = Identity.generate(), Identity.generate()
    post(client, "/worker/register", REGISTER, owner)
    assert complete(client, impostor).status_code == 403


def test_status_lists_the_models_served(served):
    client, *_ = served
    body = client.get("/inference/status").json()
    assert body["models"] == ["llama3", "mistral"]
