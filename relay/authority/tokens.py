"""Minting the tokens PostgREST will accept.

Relay nodes prove who they are with Ed25519. PostgREST checks a JWT signed with
the project's own secret. Something has to bridge the two, and that something is
the only component in the system that needs the database's signing secret —
which is exactly why it is a separate service and not a library every node
links against.

The claim that matters is ``relay_node_id``. Every RLS policy written in
migration 002 is expressed in terms of it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import jwt

ALGORITHM = "HS256"

# PostgREST maps this claim to the database role it runs the request as.
# `authenticated` is the Supabase role that RLS policies are written against;
# it is emphatically not `service_role`, which bypasses RLS entirely.
DEFAULT_ROLE = "authenticated"
DEFAULT_AUDIENCE = "authenticated"
ISSUER = "relay-authority"

# Short by design. A leaked token is a bearer credential for one node, and the
# only thing limiting the damage is how soon it expires.
DEFAULT_TTL_SECONDS = 3600


class TokenError(RuntimeError):
    pass


@dataclass
class IssuedToken:
    access_token: str
    node_id: str
    expires_at: int

    @property
    def expires_in(self) -> int:
        return max(0, self.expires_at - int(time.time()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "token_type": "bearer",
            "node_id": self.node_id,
            "expires_at": self.expires_at,
            "expires_in": self.expires_in,
        }


def mint(
    node_id: str,
    secret: str,
    *,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    role: str = DEFAULT_ROLE,
    audience: str = DEFAULT_AUDIENCE,
    now: int | None = None,
) -> IssuedToken:
    """Sign a JWT carrying this node's identity."""
    if not secret:
        raise TokenError(
            "No JWT secret configured. The authority needs RELAY_JWT_SECRET set to the "
            "Supabase project's JWT secret — it is the one component that does."
        )
    issued_at = int(now if now is not None else time.time())
    expires_at = issued_at + ttl_seconds
    payload = {
        "role": role,
        "aud": audience,
        "iss": ISSUER,
        # `sub` and the custom claim carry the same value; policies read the
        # custom one so that nothing depends on how Supabase treats `sub`.
        "sub": node_id,
        "relay_node_id": node_id,
        "iat": issued_at,
        "exp": expires_at,
    }
    return IssuedToken(
        access_token=jwt.encode(payload, secret, algorithm=ALGORITHM),
        node_id=node_id,
        expires_at=expires_at,
    )


def read(token: str, secret: str, *, audience: str = DEFAULT_AUDIENCE) -> dict[str, Any]:
    """Verify and decode. Used by tests and by anyone auditing a token."""
    try:
        return jwt.decode(token, secret, algorithms=[ALGORITHM], audience=audience)
    except jwt.PyJWTError as exc:
        raise TokenError(str(exc)) from exc


def node_id_of(token: str) -> str:
    """The claimed node id, without verifying. For logging only."""
    try:
        claims = jwt.decode(token, options={"verify_signature": False})
    except jwt.PyJWTError:
        return ""
    return str(claims.get("relay_node_id", ""))
