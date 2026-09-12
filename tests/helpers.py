"""Shared test helpers.

These live here rather than in a test module so that importing them does not
mean importing somebody else's tests.
"""

from __future__ import annotations

import json
import time

from relay.auth import NODE_HEADER, SIGNATURE_HEADER, TIMESTAMP_HEADER, canonical_message
from relay.identity import Identity

REGISTER = {"worker_id": "w1", "session_id": "s1", "machine_id": "m1"}
COMPLETE = {"worker_id": "w1", "session_id": "s1", "prompt": "hello", "max_tokens": 32}


def sign_headers(
    identity: Identity, method: str, path: str, body: bytes, *, timestamp: int | None = None
) -> dict[str, str]:
    """The three headers a provider checks, for one request."""
    stamp = str(timestamp if timestamp is not None else int(time.time()))
    message = canonical_message(method, path, stamp, body)
    return {
        NODE_HEADER: identity.node_id,
        TIMESTAMP_HEADER: stamp,
        SIGNATURE_HEADER: identity.sign(message).hex(),
    }


def post(client, path: str, payload: dict, identity: Identity):
    """POST a signed JSON body through a TestClient.

    Built by hand rather than with `json=` so the bytes that are signed are
    exactly the bytes that are sent — which is the property under test.
    """
    body = json.dumps(payload).encode()
    return client.post(
        path,
        content=body,
        headers={"content-type": "application/json", **sign_headers(identity, "POST", path, body)},
    )
