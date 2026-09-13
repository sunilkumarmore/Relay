"""Retry policy against one provider.

This lives on MarketSession now — the worker no longer owns a URL, so it no
longer owns the retry loop either. The policy is unchanged: transient failures
back off, our own mistakes do not.
"""

from __future__ import annotations

import pytest
import requests

from relay.consumer import session as session_mod
from relay.consumer.session import Binding, MarketSession, ProviderUnavailable
from relay.identity import Identity
from relay.store import MemoryStore

BINDING = Binding(
    endpoint_url="http://provider", provider_node_id="p1", inference_node_id="n1", offer=None
)


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, headers: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)


@pytest.fixture
def session():
    return MarketSession(
        Identity.generate(),
        MemoryStore(),
        worker_id="w1",
        session_id="s1",
        machine_id="m1",
    )


@pytest.fixture
def no_sleep(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr(session_mod.time, "sleep", waits.append)
    return waits


def test_succeeds_first_try(session, monkeypatch, no_sleep):
    monkeypatch.setattr(
        requests, "post", lambda *a, **k: FakeResponse(200, {"response": "ok", "latency_ms": 5})
    )
    assert session._attempt(BINDING, "p", 32)["response"] == "ok"
    assert no_sleep == []


def test_client_error_does_not_retry(session, monkeypatch, no_sleep):
    calls = {"n": 0}

    def post(*args, **kwargs):
        calls["n"] += 1
        return FakeResponse(400)

    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(requests.HTTPError):
        session._attempt(BINDING, "p", 32)
    assert calls["n"] == 1, "a 4xx is our fault; another provider would reject it identically"
    assert no_sleep == []


def test_server_error_retries_with_exponential_backoff(session, monkeypatch, no_sleep):
    calls = {"n": 0}

    def post(*args, **kwargs):
        calls["n"] += 1
        return FakeResponse(503)

    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(ProviderUnavailable, match="failed after 4 attempts"):
        session._attempt(BINDING, "p", 32)
    assert calls["n"] == 4
    assert no_sleep == [1, 2, 4]


def test_timeout_retries_then_succeeds(session, monkeypatch, no_sleep):
    calls = {"n": 0}

    def post(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.Timeout("slow")
        return FakeResponse(200, {"response": "eventually"})

    monkeypatch.setattr(requests, "post", post)
    assert session._attempt(BINDING, "p", 32)["response"] == "eventually"
    assert no_sleep == [1, 2]


def test_connection_error_retries(session, monkeypatch, no_sleep):
    calls = {"n": 0}

    def post(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.ConnectionError("refused")
        return FakeResponse(200, {"response": "ok"})

    monkeypatch.setattr(requests, "post", post)
    assert session._attempt(BINDING, "p", 32)["response"] == "ok"
    assert no_sleep == [1]


def test_429_fails_over_immediately_without_burning_retries(session, monkeypatch, no_sleep):
    """Advertised capacity is gone. Waiting here helps nobody — someone else
    may have capacity right now."""
    calls = {"n": 0}

    def post(*args, **kwargs):
        calls["n"] += 1
        return FakeResponse(429, headers={"Retry-After": "5"})

    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(ProviderUnavailable, match="saturated"):
        session._attempt(BINDING, "p", 32)
    assert calls["n"] == 1
    assert no_sleep == []


def test_401_re_registers_and_retries_in_place(session, monkeypatch, no_sleep):
    posts: list[str] = []

    def post(url, json=None, timeout=None, auth=None):
        posts.append(url)
        if url.endswith("/worker/register"):
            return FakeResponse(200, {"inference_node_id": "n1"})
        return FakeResponse(401) if posts.count(url) == 1 else FakeResponse(200, {"response": "ok"})

    monkeypatch.setattr(requests, "post", post)
    assert session._attempt(BINDING, "p", 32)["response"] == "ok"
    assert posts == [
        "http://provider/inference/complete",
        "http://provider/worker/register",
        "http://provider/inference/complete",
    ]
    assert no_sleep == [], "re-registering is not a backoff-worthy failure"


def test_max_retries_is_configurable(monkeypatch, no_sleep):
    session = MarketSession(
        Identity.generate(), None, worker_id="w", session_id="s", machine_id="m", max_retries=1
    )
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(500))
    with pytest.raises(ProviderUnavailable, match="after 2 attempts"):
        session._attempt(BINDING, "p", 32)
    assert no_sleep == [1]


def test_the_model_is_named_when_an_offer_backs_the_binding(monkeypatch, no_sleep):
    from relay.identity import Identity as Ident
    from relay.provider.offers import build_offer

    offer = build_offer(
        Ident.generate(),
        endpoint_url="http://provider",
        model="mistral",
        context_window=4096,
        price_in_per_1k=0.01,
        price_out_per_1k=0.03,
    )
    session = MarketSession(
        Identity.generate(), None, worker_id="w", session_id="s", machine_id="m"
    )
    sent: dict = {}

    def post(url, json=None, timeout=None, auth=None):
        sent.update(json)
        return FakeResponse(200, {"response": "ok"})

    monkeypatch.setattr(requests, "post", post)
    binding = Binding("http://provider", offer.provider_node_id, "n1", offer)
    session._attempt(binding, "p", 32)
    assert sent["model"] == "mistral", "buy what you agreed to pay for"
