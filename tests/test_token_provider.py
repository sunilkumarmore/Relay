"""The node side: getting a token, keeping it fresh, attaching it."""

from __future__ import annotations

import time

import pytest
import requests
from fastapi.testclient import TestClient

from relay.authority.client import AuthorityError, TokenProvider
from relay.authority.server import Authority, create_app
from relay.authority.tokens import read
from relay.identity import Identity

SECRET = "a-supabase-jwt-secret"


class ClientSession:
    """Routes the provider's requests into a TestClient, so the real two-round
    exchange runs without a socket."""

    def __init__(self, client, fail_with=None):
        self.client = client
        self.fail_with = fail_with
        self.calls: list[str] = []

    def post(self, url, json=None, timeout=None):
        if self.fail_with:
            raise self.fail_with
        path = url.split("://", 1)[-1].split("/", 1)[-1]
        self.calls.append("/" + path)
        response = self.client.post("/" + path, json=json)
        response.raise_for_status = _raise_for_status(response)
        return response


def _raise_for_status(response):
    def raise_for_status():
        if response.status_code >= 400:
            raise requests.HTTPError(str(response.status_code), response=response)

    return raise_for_status


@pytest.fixture
def served():
    authority = Authority(SECRET, ttl_seconds=3600)
    with TestClient(create_app(authority)) as client:
        yield client, authority


def build(served, identity=None, **kwargs):
    client, _ = served
    return TokenProvider(
        identity or Identity.generate(), "http://authority", session=ClientSession(client), **kwargs
    )


def test_a_provider_exchanges_its_key_for_a_token(served):
    identity = Identity.generate()
    provider = build(served, identity)

    token = provider.token()
    assert read(token, SECRET)["relay_node_id"] == identity.node_id
    assert provider.exchanges == 1


def test_the_token_is_reused_while_it_is_fresh(served):
    provider = build(served)
    first = provider.token()

    for _ in range(10):
        assert provider.token() == first
    assert provider.exchanges == 1, "one exchange, not eleven"


def test_the_token_is_replaced_before_it_expires(served):
    """A request must never leave carrying a credential that dies in flight."""
    provider = build(served, refresh_skew_seconds=300)
    provider.token()
    assert provider.exchanges == 1

    # Now inside the refresh window, though the token is still technically valid.
    replaced = provider.token(now=time.time() + 3600 - 200)
    assert provider.exchanges == 2, "the provider should have gone back for a new one"
    # Two tokens minted in the same second are byte-identical, so the string is
    # not the evidence — the exchange is. What matters is that what comes back
    # is valid and names this node.
    assert read(replaced, SECRET)["relay_node_id"] == provider.identity.node_id


def test_invalidating_forces_a_fresh_exchange(served):
    provider = build(served)
    provider.token()
    provider.invalidate()
    provider.token()
    assert provider.exchanges == 2


def test_a_concurrent_burst_performs_one_exchange(served):
    """Twenty store calls at once should not mean twenty round trips."""
    import threading

    provider = build(served)
    tokens: list[str] = []

    threads = [threading.Thread(target=lambda: tokens.append(provider.token())) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    assert len(set(tokens)) == 1
    assert provider.exchanges == 1


def test_an_unreachable_authority_says_how_to_start_it():
    provider = TokenProvider(
        Identity.generate(),
        "http://nowhere",
        session=ClientSession(None, fail_with=requests.ConnectionError("refused")),
    )
    with pytest.raises(AuthorityError, match="python -m relay.authority"):
        provider.token()


def test_a_refusal_surfaces_the_reason(served):
    client, _ = served
    identity = Identity.generate()
    provider = TokenProvider(identity, "http://authority", session=ClientSession(client))

    # Sign with the wrong key by swapping the identity after the challenge.
    provider.identity = Identity.generate()
    original = provider.identity
    provider.identity = type(
        "Mismatched",
        (),
        {"node_id": identity.node_id, "sign": staticmethod(lambda m: original.sign(m))},
    )()

    with pytest.raises(AuthorityError, match="bad_signature"):
        provider.token()


# -- the store attaches it -------------------------------------------------


class FakePostgrest:
    def __init__(self):
        self.tokens: list[str] = []

    def auth(self, token):
        self.tokens.append(token)


class StubProvider:
    """Hands out tokens on demand, so the store's caching is tested on its own
    rather than through JWT timestamp resolution."""

    def __init__(self, tokens):
        self.queue = list(tokens)
        self.last = None

    def token(self):
        if self.queue:
            self.last = self.queue.pop(0)
        return self.last


class FakeSupabaseStore:
    """Exercises SupabaseStore._db() without a Supabase project."""

    def __init__(self, provider):
        from relay.store import SupabaseStore

        self.postgrest = FakePostgrest()
        self._db = SupabaseStore._db.__get__(self)
        self.client = self
        self.token_provider = provider
        self._attached_token = None


def test_the_store_attaches_the_token_before_a_call(served):
    provider = build(served)
    store = FakeSupabaseStore(provider)

    store._db()
    assert len(store.postgrest.tokens) == 1
    assert read(store.postgrest.tokens[0], SECRET)["relay_node_id"] == provider.identity.node_id


def test_the_store_does_not_reattach_an_unchanged_token(served):
    provider = build(served)
    store = FakeSupabaseStore(provider)

    for _ in range(5):
        store._db()
    assert len(store.postgrest.tokens) == 1


def test_the_store_reattaches_when_the_token_changes(served):
    store = FakeSupabaseStore(StubProvider(["token-one", "token-two"]))

    store._db()
    store._db()

    assert store.postgrest.tokens == ["token-one", "token-two"]


def test_the_store_skips_reattaching_an_unchanged_token(served):
    """Re-attaching the same credential on every one of 47 call sites would be
    pure overhead."""
    store = FakeSupabaseStore(StubProvider(["token-one"]))

    for _ in range(5):
        store._db()

    assert store.postgrest.tokens == ["token-one"]


def test_without_an_authority_the_store_uses_the_configured_key(served):
    """How a single-operator deployment runs before migration 002 is applied."""
    store = FakeSupabaseStore(None)
    assert store._db() is store.client
    assert store.postgrest.tokens == []
