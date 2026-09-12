"""Signed requests: what the signature proves, and what it does not."""

from __future__ import annotations

import time

import pytest
import requests
from fastapi import HTTPException
from fastapi.testclient import TestClient

from relay.auth import (
    NODE_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    RelayAuth,
    verify_headers,
)
from relay.identity import Identity
from relay.inference.backends import FakeBackend
from relay.inference.registry import Registry, create_app
from relay.store import MemoryStore
from tests.helpers import COMPLETE, REGISTER, post, sign_headers

# -- the primitive ---------------------------------------------------------


def test_valid_signature_verifies():
    identity = Identity.generate()
    headers = sign_headers(identity, "POST", "/x", b'{"a":1}')
    signed = verify_headers(
        "POST", "/x", b'{"a":1}', headers[NODE_HEADER], headers[TIMESTAMP_HEADER], headers[SIGNATURE_HEADER]
    )
    assert signed.node_id == identity.node_id


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("GET", "/x", b'{"a":1}'),      # method swapped
        ("POST", "/y", b'{"a":1}'),     # signature replayed onto another endpoint
        ("POST", "/x", b'{"a":2}'),     # body altered in flight
    ],
)
def test_signature_does_not_transfer(method, path, body):
    identity = Identity.generate()
    headers = sign_headers(identity, "POST", "/x", b'{"a":1}')
    with pytest.raises(HTTPException) as exc:
        verify_headers(
            method, path, body, headers[NODE_HEADER], headers[TIMESTAMP_HEADER], headers[SIGNATURE_HEADER]
        )
    assert exc.value.status_code == 401


def test_missing_headers_are_rejected():
    with pytest.raises(HTTPException) as exc:
        verify_headers("POST", "/x", b"", None, None, None)
    assert exc.value.detail == "request_not_signed"


def test_old_request_cannot_be_replayed():
    identity = Identity.generate()
    old = int(time.time()) - 3600
    headers = sign_headers(identity, "POST", "/x", b"", timestamp=old)
    with pytest.raises(HTTPException) as exc:
        verify_headers(
            "POST", "/x", b"", headers[NODE_HEADER], headers[TIMESTAMP_HEADER], headers[SIGNATURE_HEADER]
        )
    assert exc.value.detail == "stale_timestamp"


def test_future_timestamp_is_also_rejected():
    identity = Identity.generate()
    headers = sign_headers(identity, "POST", "/x", b"", timestamp=int(time.time()) + 3600)
    with pytest.raises(HTTPException) as exc:
        verify_headers(
            "POST", "/x", b"", headers[NODE_HEADER], headers[TIMESTAMP_HEADER], headers[SIGNATURE_HEADER]
        )
    assert exc.value.detail == "stale_timestamp"


def test_malformed_values_are_rejected_cleanly():
    identity = Identity.generate()
    stamp = str(int(time.time()))
    with pytest.raises(HTTPException) as exc:
        verify_headers("POST", "/x", b"", identity.node_id, "not-a-number", "ab")
    assert exc.value.detail == "bad_timestamp"
    with pytest.raises(HTTPException) as exc:
        verify_headers("POST", "/x", b"", identity.node_id, stamp, "nothex")
    assert exc.value.detail == "bad_signature"


def test_requests_auth_adapter_signs_a_real_request():
    identity = Identity.generate()
    prepared = requests.Request(
        "POST", "http://example.com/inference/complete", json={"a": 1}
    ).prepare()
    RelayAuth(identity)(prepared)

    signed = verify_headers(
        "POST",
        "/inference/complete",
        prepared.body,
        prepared.headers[NODE_HEADER],
        prepared.headers[TIMESTAMP_HEADER],
        prepared.headers[SIGNATURE_HEADER],
    )
    assert signed.node_id == identity.node_id


# -- the registry enforcing it --------------------------------------------


@pytest.fixture
def signed_registry():
    registry = Registry(FakeBackend(), MemoryStore(), node_id="node-1")
    with TestClient(create_app(registry, prune_in_background=False)) as client:
        yield client, registry


def test_unsigned_inference_is_rejected(signed_registry):
    client, _ = signed_registry
    resp = client.post("/inference/complete", json=COMPLETE)
    assert resp.status_code == 401
    assert resp.json()["detail"] == "request_not_signed"


def test_health_stays_public(signed_registry):
    client, _ = signed_registry
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["requires_signature"] is True


def test_signed_worker_can_register_and_infer(signed_registry):
    client, registry = signed_registry
    identity = Identity.generate()

    assert post(client, "/worker/register", REGISTER, identity).status_code == 200
    assert registry.active_workers["w1"].node_id == identity.node_id

    resp = post(client, "/inference/complete", COMPLETE, identity)
    assert resp.status_code == 200
    assert resp.json()["tokens_in"] > 0


def test_another_node_cannot_use_a_registered_worker_id(signed_registry):
    """The whole point: a worker_id is a label, not a credential."""
    client, _ = signed_registry
    owner, impostor = Identity.generate(), Identity.generate()
    post(client, "/worker/register", REGISTER, owner)

    resp = post(client, "/inference/complete", COMPLETE, impostor)
    assert resp.status_code == 403
    assert resp.json()["detail"] == "worker_belongs_to_another_node"


def test_another_node_cannot_claim_a_taken_worker_id(signed_registry):
    client, _ = signed_registry
    owner, impostor = Identity.generate(), Identity.generate()
    post(client, "/worker/register", REGISTER, owner)

    resp = post(client, "/worker/register", REGISTER, impostor)
    assert resp.status_code == 403
    assert resp.json()["detail"] == "worker_id_taken"


def test_another_node_cannot_deregister_your_worker(signed_registry):
    client, registry = signed_registry
    owner, impostor = Identity.generate(), Identity.generate()
    post(client, "/worker/register", REGISTER, owner)

    assert post(client, "/worker/deregister", {"worker_id": "w1"}, impostor).status_code == 403
    assert "w1" in registry.active_workers


def test_another_nodes_heartbeat_does_not_move_your_progress(signed_registry):
    client, registry = signed_registry
    owner, impostor = Identity.generate(), Identity.generate()
    post(client, "/worker/register", REGISTER, owner)
    post(client, "/worker/heartbeat", {"worker_id": "w1", "last_checkpoint": 9, "steps_completed": 9}, owner)

    post(
        client,
        "/worker/heartbeat",
        {"worker_id": "w1", "last_checkpoint": 0, "steps_completed": 0},
        impostor,
    )
    assert registry.active_workers["w1"].steps_completed == 9


def test_tampering_with_the_prompt_invalidates_the_signature(signed_registry):
    client, _ = signed_registry
    identity = Identity.generate()
    post(client, "/worker/register", REGISTER, identity)

    import json

    body = json.dumps(COMPLETE).encode()
    headers = sign_headers(identity, "POST", "/inference/complete", body)
    tampered = json.dumps({**COMPLETE, "prompt": "something else entirely"}).encode()

    resp = client.post(
        "/inference/complete",
        content=tampered,
        headers={"content-type": "application/json", **headers},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "bad_signature"
