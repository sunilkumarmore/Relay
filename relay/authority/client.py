"""Getting and keeping a database token.

A node holds one of these for the life of the process. It fetches a token when
first needed and replaces it before it expires — there is no background thread,
because the only moment freshness matters is just before a request.
"""

from __future__ import annotations

import threading
import time

import requests

from relay import config
from relay.authority.server import signing_message
from relay.authority.tokens import IssuedToken
from relay.identity import Identity

REQUEST_TIMEOUT_SECONDS = 15

# Replace the token this long before it actually expires, so a request never
# leaves with a credential that dies in flight.
REFRESH_SKEW_SECONDS = 120


class AuthorityError(RuntimeError):
    pass


class TokenProvider:
    """Exchanges this node's key for a PostgREST token, and keeps it fresh."""

    def __init__(
        self,
        identity: Identity,
        authority_url: str,
        *,
        refresh_skew_seconds: int = REFRESH_SKEW_SECONDS,
        session: requests.Session | None = None,
    ) -> None:
        self.identity = identity
        self.authority_url = authority_url.rstrip("/")
        self.refresh_skew = refresh_skew_seconds
        self.session = session or requests.Session()
        self._token: IssuedToken | None = None
        self._lock = threading.Lock()
        self.exchanges = 0

    # -- lifetime ---------------------------------------------------------
    def _is_fresh(self, now: float) -> bool:
        return self._token is not None and self._token.expires_at - self.refresh_skew > now

    def token(self, *, now: float | None = None) -> str:
        moment = now if now is not None else time.time()
        # Held across the exchange so a burst of calls performs one, not twenty.
        with self._lock:
            if not self._is_fresh(moment):
                self._token = self._exchange()
            assert self._token is not None
            return self._token.access_token

    def invalidate(self) -> None:
        """Drop the current token — used when the database rejects it."""
        with self._lock:
            self._token = None

    # -- the exchange -----------------------------------------------------
    def _exchange(self) -> IssuedToken:
        node_id = self.identity.node_id
        try:
            challenge = self.session.post(
                f"{self.authority_url}/auth/challenge",
                json={"node_id": node_id},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            challenge.raise_for_status()
            body = challenge.json()
        except requests.RequestException as exc:
            raise AuthorityError(
                f"Could not reach the token authority at {self.authority_url}: {exc}\n"
                " Start it with: python -m relay.authority"
            ) from exc

        nonce = str(body.get("nonce", ""))
        audience = str(body.get("audience", "authenticated"))
        if not nonce:
            raise AuthorityError("Authority returned no challenge")

        signature = self.identity.sign(signing_message(node_id, nonce, audience)).hex()

        try:
            issued = self.session.post(
                f"{self.authority_url}/auth/token",
                json={"node_id": node_id, "nonce": nonce, "signature": signature},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            issued.raise_for_status()
            payload = issued.json()
        except requests.HTTPError as exc:
            detail = ""
            if exc.response is not None:
                try:
                    detail = str(exc.response.json().get("detail", ""))
                except Exception:
                    detail = exc.response.text[:200]
            raise AuthorityError(f"The authority refused this node: {detail or exc}") from exc
        except requests.RequestException as exc:
            raise AuthorityError(f"Token exchange failed: {exc}") from exc

        self.exchanges += 1
        return IssuedToken(
            access_token=str(payload["access_token"]),
            node_id=str(payload.get("node_id", node_id)),
            expires_at=int(payload["expires_at"]),
        )


def provider_from_env(identity: Identity | None = None) -> TokenProvider | None:
    """Build a provider if an authority is configured, otherwise None.

    None means "connect with whatever key is in SUPABASE_KEY", which is how
    Relay worked before RLS and how a single-operator deployment can still run.
    """
    config.load_env()
    url = config.get("RELAY_AUTHORITY_URL")
    if not url:
        return None

    from relay.identity import identity_from_env

    return TokenProvider(
        identity or identity_from_env(),
        url,
        refresh_skew_seconds=config.get_int("RELAY_TOKEN_REFRESH_SKEW", REFRESH_SKEW_SECONDS),
    )
