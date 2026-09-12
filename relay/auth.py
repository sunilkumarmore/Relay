"""Signed requests between Relay nodes.

A request is signed over a canonical string built from the method, the path, a
timestamp, and a hash of the body. Signing the body hash means a proxy cannot
alter the prompt; including the timestamp means a captured request cannot be
replayed later; including the path means a signature for one endpoint is not
valid on another.

The signature proves *which node* sent the request. What that node is allowed to
do is a separate question, answered by the registry.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import requests
from fastapi import HTTPException, Request

from relay import identity as ident
from relay.identity import Identity

NODE_HEADER = "X-Relay-Node"
TIMESTAMP_HEADER = "X-Relay-Timestamp"
SIGNATURE_HEADER = "X-Relay-Signature"

# How far out of step a clock may be before a request is refused.
MAX_CLOCK_SKEW_SECONDS = 60


def body_hash(body: bytes) -> str:
    return hashlib.sha256(body or b"").hexdigest()


def canonical_message(method: str, path: str, timestamp: str, body: bytes) -> bytes:
    """The exact bytes both sides sign. Any disagreement here fails closed."""
    return "\n".join([method.upper(), path, str(timestamp), body_hash(body)]).encode("utf-8")


@dataclass(frozen=True)
class SignedBy:
    node_id: str
    timestamp: int


class RelayAuth(requests.auth.AuthBase):
    """``requests`` adapter that signs every outgoing call."""

    def __init__(self, identity: Identity) -> None:
        self.identity = identity

    def __call__(self, request):  # noqa: ANN001 - requests' interface
        timestamp = str(int(time.time()))
        path = requests.utils.urlparse(request.url).path or "/"
        body = request.body or b""
        if isinstance(body, str):
            body = body.encode("utf-8")
        message = canonical_message(request.method, path, timestamp, body)
        request.headers[NODE_HEADER] = self.identity.node_id
        request.headers[TIMESTAMP_HEADER] = timestamp
        request.headers[SIGNATURE_HEADER] = self.identity.sign(message).hex()
        return request


def verify_headers(
    method: str,
    path: str,
    body: bytes,
    node_id: str | None,
    timestamp: str | None,
    signature: str | None,
    *,
    now: float | None = None,
    max_skew: int = MAX_CLOCK_SKEW_SECONDS,
) -> SignedBy:
    """Verify one signed request, or raise 401. Never returns a partial result."""
    if not node_id or not timestamp or not signature:
        raise HTTPException(status_code=401, detail="request_not_signed")

    try:
        stamp = int(timestamp)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="bad_timestamp") from exc

    current = int(now if now is not None else time.time())
    if abs(current - stamp) > max_skew:
        # Either a replay of an old request, or a clock far enough out that we
        # cannot tell the difference. Both are refused.
        raise HTTPException(status_code=401, detail="stale_timestamp")

    try:
        raw_signature = bytes.fromhex(signature)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="bad_signature") from exc

    if not ident.verify(node_id, canonical_message(method, path, timestamp, body), raw_signature):
        raise HTTPException(status_code=401, detail="bad_signature")

    return SignedBy(node_id=node_id, timestamp=stamp)


async def require_signature(request: Request) -> SignedBy:
    """FastAPI dependency. Attaches the verified caller to ``request.state``."""
    body = await request.body()
    signed = verify_headers(
        request.method,
        request.url.path,
        body,
        request.headers.get(NODE_HEADER),
        request.headers.get(TIMESTAMP_HEADER),
        request.headers.get(SIGNATURE_HEADER),
    )
    request.state.signed_by = signed
    return signed
