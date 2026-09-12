from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from relay.inference.backends import BackendError, FakeBackend
from relay.inference.registry import Registry, create_app, now_utc
from relay.store import MemoryStore

REGISTER = {"worker_id": "w1", "session_id": "s1", "machine_id": "m1"}
COMPLETE = {"worker_id": "w1", "session_id": "s1", "prompt": "hello", "max_tokens": 32}


@pytest.fixture
def parts():
    backend = FakeBackend()
    store = MemoryStore()
    registry = Registry(backend, store, node_id="node-1")
    with TestClient(create_app(registry, prune_in_background=False, require_signatures=False)) as client:
        yield client, registry, backend, store


def test_health_reports_backend(parts):
    client, *_ = parts
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["backend"] == "fake"
    assert body["ollama_connected"] is True
    assert body["backend_connected"] is True


def test_register_returns_node_id_and_tracks_worker(parts):
    client, registry, *_ = parts
    body = client.post("/worker/register", json=REGISTER).json()
    assert body == {"accepted": True, "inference_node_id": "node-1"}
    assert "w1" in registry.active_workers


def test_re_register_preserves_progress_and_registered_at(parts):
    client, registry, *_ = parts
    client.post("/worker/register", json=REGISTER)
    client.post("/worker/heartbeat", json={"worker_id": "w1", "last_checkpoint": 3, "steps_completed": 3})
    first_seen = registry.active_workers["w1"].registered_at

    client.post("/worker/register", json=REGISTER)

    info = registry.active_workers["w1"]
    assert info.steps_completed == 3, "dashboard would flash back to 0"
    assert info.registered_at == first_seen


def test_heartbeat_for_unknown_worker_is_a_no_op(parts):
    client, registry, *_ = parts
    resp = client.post(
        "/worker/heartbeat", json={"worker_id": "ghost", "last_checkpoint": 0, "steps_completed": 0}
    )
    assert resp.json() == {"ok": True}
    assert registry.active_workers == {}


def test_deregister_removes_worker_and_is_idempotent(parts):
    client, registry, *_ = parts
    client.post("/worker/register", json=REGISTER)
    for _ in range(2):
        assert client.post("/worker/deregister", json={"worker_id": "w1"}).json() == {"ok": True}
    assert registry.active_workers == {}


def test_prune_drops_only_stale_workers(parts):
    client, registry, *_ = parts
    client.post("/worker/register", json=REGISTER)
    client.post("/worker/register", json={**REGISTER, "worker_id": "w2"})
    registry.active_workers["w1"].last_heartbeat = now_utc() - timedelta(seconds=120)

    assert registry.prune_stale_workers() == ["w1"]
    assert set(registry.active_workers) == {"w2"}


def test_complete_requires_registration(parts):
    client, *_ = parts
    resp = client.post("/inference/complete", json=COMPLETE)
    assert resp.status_code == 401
    assert resp.json()["detail"] == "worker_not_registered"


def test_complete_returns_token_split_and_logs_telemetry(parts):
    client, registry, backend, store = parts
    client.post("/worker/register", json=REGISTER)

    body = client.post("/inference/complete", json=COMPLETE).json()

    assert body["tokens_used"] == body["tokens_in"] + body["tokens_out"]
    assert body["inference_node_id"] == "node-1"
    assert backend.prompts == ["hello"]

    logged = store.snapshot()["inference_log"]
    assert len(logged) == 1
    assert logged[0]["tokens_used"] == body["tokens_used"]
    assert logged[0]["success"] is True


def test_backend_failure_is_a_502(parts):
    client, registry, backend, _ = parts
    backend.fail_calls = {1}
    client.post("/worker/register", json=REGISTER)

    resp = client.post("/inference/complete", json=COMPLETE)
    assert resp.status_code == 502
    assert "backend_error" in resp.json()["detail"]


def test_unexpected_backend_exception_is_also_a_502(parts):
    client, registry, backend, _ = parts
    client.post("/worker/register", json=REGISTER)

    def boom(prompt, max_tokens=1200):
        raise ValueError("something the backend did not anticipate")

    backend.complete = boom
    assert client.post("/inference/complete", json=COMPLETE).status_code == 502


def test_status_summarizes_recent_calls(parts):
    client, *_ = parts
    client.post("/worker/register", json=REGISTER)
    for _ in range(3):
        client.post("/inference/complete", json=COMPLETE)

    body = client.get("/inference/status").json()
    assert body["total_requests"] == 3
    assert body["requests_last_minute"] == 3
    assert len(body["active_workers"]) == 1
    assert len(body["recent_calls"]) == 3


def test_registry_runs_without_a_store():
    """Telemetry is best-effort; a node with no store still serves inference."""
    registry = Registry(FakeBackend(), None, node_id="n")
    with TestClient(create_app(registry, prune_in_background=False, require_signatures=False)) as client:
        client.post("/worker/register", json=REGISTER)
        assert client.post("/inference/complete", json=COMPLETE).status_code == 200


def test_store_failure_does_not_fail_the_request():
    class ExplodingStore(MemoryStore):
        def insert_inference_log(self, **kwargs):
            raise RuntimeError("supabase is down")

    registry = Registry(FakeBackend(), ExplodingStore(), node_id="n")
    with TestClient(create_app(registry, prune_in_background=False, require_signatures=False)) as client:
        client.post("/worker/register", json=REGISTER)
        assert client.post("/inference/complete", json=COMPLETE).status_code == 200


def test_health_check_is_cached():
    calls = {"n": 0}

    class CountingBackend(FakeBackend):
        def health(self):
            calls["n"] += 1
            return True

    registry = Registry(CountingBackend(), None, node_id="n")
    with TestClient(create_app(registry, prune_in_background=False, require_signatures=False)) as client:
        for _ in range(5):
            client.get("/health")
    assert calls["n"] == 1, "each /health poll should not hit the backend"


def test_backend_error_is_importable_from_backends():
    assert issubclass(BackendError, RuntimeError)
