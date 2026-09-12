"""The token authority.

A node proves it holds a private key; it gets back a short-lived JWT that
PostgREST accepts and that RLS policies can reason about.

The exchange is a two-round challenge, not a self-signed timestamp like
``relay/auth.py`` uses for provider calls. The difference matters: a provider
request replayed inside the clock-skew window merely repeats that request,
whereas a token request replayed inside the window hands the replayer a *bearer
credential* for the victim's node. So the server picks the nonce, remembers it,
and accepts it exactly once.

Anyone with a keypair can obtain a token. That is deliberate — it is an open
market, and the token only ever says *which* node you are. What that node may
then do is the database's business, not this service's.
"""

from __future__ import annotations

import secrets
import socket
import time
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from relay import config
from relay import identity as ident
from relay.authority.tokens import DEFAULT_TTL_SECONDS, TokenError, mint

# Long enough for a round trip on a bad connection, short enough that an
# unclaimed challenge is not a standing invitation.
CHALLENGE_TTL_SECONDS = 120

# Binds a signature to this exchange. Without it, a signature gathered for one
# purpose could be presented as proof for another.
SIGNING_CONTEXT = "relay-authority-v1"


class ChallengeRequest(BaseModel):
    node_id: str


class TokenRequest(BaseModel):
    node_id: str
    nonce: str
    signature: str


@dataclass
class Challenge:
    node_id: str
    expires_at: float


def signing_message(node_id: str, nonce: str, audience: str) -> bytes:
    """Exactly what a node signs. Both sides build it the same way or nothing works."""
    return "\n".join([SIGNING_CONTEXT, node_id, nonce, audience]).encode("utf-8")


class Authority:
    def __init__(
        self,
        secret: str,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        challenge_ttl_seconds: int = CHALLENGE_TTL_SECONDS,
        audience: str = "authenticated",
    ) -> None:
        self.secret = secret
        self.ttl_seconds = ttl_seconds
        self.challenge_ttl_seconds = challenge_ttl_seconds
        self.audience = audience
        # In memory on purpose: a restart invalidates outstanding challenges,
        # which costs a client one retry and keeps this service stateless enough
        # to run anywhere.
        self._challenges: dict[str, Challenge] = {}
        self.issued = 0

    def _prune(self, now: float) -> None:
        for nonce in [n for n, c in self._challenges.items() if c.expires_at <= now]:
            del self._challenges[nonce]

    def challenge(self, node_id: str, *, now: float | None = None) -> dict[str, Any]:
        moment = now if now is not None else time.time()
        self._prune(moment)

        if not node_id:
            raise HTTPException(status_code=400, detail="node_id_required")
        try:
            ident.public_key_from_node_id(node_id)
        except ident.IdentityError as exc:
            # Fail here rather than at verification, so a malformed id gets a
            # useful answer instead of a signature mismatch.
            raise HTTPException(status_code=400, detail="not_a_node_id") from exc

        nonce = secrets.token_urlsafe(32)
        self._challenges[nonce] = Challenge(node_id=node_id, expires_at=moment + self.challenge_ttl_seconds)
        return {
            "nonce": nonce,
            "expires_at": int(moment + self.challenge_ttl_seconds),
            "audience": self.audience,
            "context": SIGNING_CONTEXT,
        }

    def token(self, req: TokenRequest, *, now: float | None = None) -> dict[str, Any]:
        moment = now if now is not None else time.time()
        self._prune(moment)

        # Consumed whether or not it verifies: a nonce is one attempt, so a
        # wrong guess cannot be retried against the same challenge.
        challenge = self._challenges.pop(req.nonce, None)
        if challenge is None:
            raise HTTPException(status_code=401, detail="unknown_or_used_challenge")
        if challenge.expires_at <= moment:
            raise HTTPException(status_code=401, detail="challenge_expired")
        if challenge.node_id != req.node_id:
            raise HTTPException(status_code=401, detail="challenge_belongs_to_another_node")

        try:
            signature = bytes.fromhex(req.signature)
        except ValueError as exc:
            raise HTTPException(status_code=401, detail="bad_signature") from exc

        message = signing_message(req.node_id, req.nonce, self.audience)
        if not ident.verify(req.node_id, message, signature):
            raise HTTPException(status_code=401, detail="bad_signature")

        try:
            issued = mint(req.node_id, self.secret, ttl_seconds=self.ttl_seconds, audience=self.audience)
        except TokenError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        self.issued += 1
        return issued.to_dict()

    @property
    def outstanding_challenges(self) -> int:
        self._prune(time.time())
        return len(self._challenges)


def create_app(authority: Authority) -> FastAPI:
    app = FastAPI(title="Relay Token Authority")
    app.state.authority = authority

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "configured": bool(authority.secret),
            "token_ttl_seconds": authority.ttl_seconds,
            "audience": authority.audience,
        }

    @app.post("/auth/challenge")
    def challenge(req: ChallengeRequest) -> dict[str, Any]:
        return authority.challenge(req.node_id)

    @app.post("/auth/token")
    def token(req: TokenRequest) -> dict[str, Any]:
        return authority.token(req)

    return app


def authority_from_env() -> Authority:
    config.load_env()
    secret = config.get("RELAY_JWT_SECRET")
    if not secret:
        raise SystemExit(
            "RELAY_JWT_SECRET is not set.\n"
            " This is the Supabase project's JWT secret (Settings > API > JWT Settings).\n"
            " Only the authority needs it — never put it on a worker or a provider."
        )
    return Authority(secret, ttl_seconds=config.get_int("RELAY_TOKEN_TTL", DEFAULT_TTL_SECONDS))


def app_from_env() -> FastAPI:
    return create_app(authority_from_env())


def __getattr__(name: str):
    if name == "app":
        return app_from_env()
    raise AttributeError(name)


def run() -> None:
    import uvicorn

    config.load_env()
    host = config.get("RELAY_AUTHORITY_HOST", "0.0.0.0") or "0.0.0.0"
    port = config.get_int("RELAY_AUTHORITY_PORT", 8790)
    print(f"Relay token authority on {host}:{port} (node: {socket.gethostname()})")
    uvicorn.run(app_from_env(), host=host, port=port)


if __name__ == "__main__":
    run()
