"""A registry that restarts must not forget who is working.

Before this, registrations lived in a process dictionary: restarting the registry
401'd every worker mid-job and reset the dashboard to zero. The workers recovered
by re-registering, but the telemetry did not, and the window was real.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from relay.identity import Identity
from relay.inference.backends import FakeBackend
from relay.inference.registry import Registry, create_app
from relay.store import FileStore, MemoryStore
from tests.helpers import COMPLETE, REGISTER, post


def test_registrations_survive_a_restart():
    store = MemoryStore()
    identity = Identity.generate()

    first = Registry(FakeBackend(), store, node_id="node-1")
    with TestClient(create_app(first, prune_in_background=False)) as client:
        post(client, "/worker/register", REGISTER, identity)
        post(client, "/worker/heartbeat", {"worker_id": "w1", "last_checkpoint": 2, "steps_completed": 2}, identity)

    # The process dies. A new one starts against the same store.
    second = Registry(FakeBackend(), store, node_id="node-1")
    assert "w1" in second.active_workers
    info = second.active_workers["w1"]
    assert info.node_id == identity.node_id
    assert info.steps_completed == 2

    with TestClient(create_app(second, prune_in_background=False)) as client:
        # No re-registration needed: the worker is still known, and still bound
        # to the same node.
        assert post(client, "/inference/complete", COMPLETE, identity).status_code == 200


def test_restart_does_not_hand_a_worker_id_to_a_new_node():
    store = MemoryStore()
    owner, impostor = Identity.generate(), Identity.generate()

    first = Registry(FakeBackend(), store, node_id="node-1")
    with TestClient(create_app(first, prune_in_background=False)) as client:
        post(client, "/worker/register", REGISTER, owner)

    second = Registry(FakeBackend(), store, node_id="node-1")
    with TestClient(create_app(second, prune_in_background=False)) as client:
        assert post(client, "/inference/complete", COMPLETE, impostor).status_code == 403


def test_request_log_survives_a_restart():
    store = MemoryStore()
    identity = Identity.generate()

    first = Registry(FakeBackend(), store, node_id="node-1")
    with TestClient(create_app(first, prune_in_background=False)) as client:
        post(client, "/worker/register", REGISTER, identity)
        for _ in range(3):
            post(client, "/inference/complete", COMPLETE, identity)

    second = Registry(FakeBackend(), store, node_id="node-1")
    assert second.status()["total_requests"] == 3
    assert len(second.status()["recent_calls"]) == 3


def test_deregistering_clears_the_durable_row():
    store = MemoryStore()
    identity = Identity.generate()
    registry = Registry(FakeBackend(), store, node_id="node-1")

    with TestClient(create_app(registry, prune_in_background=False)) as client:
        post(client, "/worker/register", REGISTER, identity)
        assert store.get_node("w1") is not None
        post(client, "/worker/deregister", {"worker_id": "w1"}, identity)

    assert store.get_node("w1") is None
    assert Registry(FakeBackend(), store, node_id="node-1").active_workers == {}


def test_pruning_clears_the_durable_row():
    from datetime import timedelta

    from relay.inference.registry import now_utc

    store = MemoryStore()
    identity = Identity.generate()
    registry = Registry(FakeBackend(), store, node_id="node-1")

    with TestClient(create_app(registry, prune_in_background=False)) as client:
        post(client, "/worker/register", REGISTER, identity)

    registry.active_workers["w1"].last_heartbeat = now_utc() - timedelta(seconds=120)
    assert registry.prune_stale_workers() == ["w1"]
    assert store.get_node("w1") is None


def test_two_registries_on_one_store_keep_separate_rosters(tmp_path):
    """Phase 2 runs many providers against one directory; they must not collide."""
    store = FileStore(tmp_path / "s.json")
    identity = Identity.generate()

    node_a = Registry(FakeBackend(), store, node_id="provider-a")
    node_b = Registry(FakeBackend(), store, node_id="provider-b")

    with TestClient(create_app(node_a, prune_in_background=False)) as client:
        post(client, "/worker/register", REGISTER, identity)

    assert "w1" in Registry(FakeBackend(), store, node_id="provider-a").active_workers
    assert Registry(FakeBackend(), store, node_id="provider-b").active_workers == {}
    assert node_b.active_workers == {}


def test_failed_inference_is_logged_as_a_failure():
    store = MemoryStore()
    identity = Identity.generate()
    backend = FakeBackend(fail_calls={1})
    registry = Registry(backend, store, node_id="node-1")

    with TestClient(create_app(registry, prune_in_background=False)) as client:
        post(client, "/worker/register", REGISTER, identity)
        assert post(client, "/inference/complete", COMPLETE, identity).status_code == 502

    logged = store.list_registry_requests("node-1")
    assert len(logged) == 1
    assert logged[0]["success"] is False
