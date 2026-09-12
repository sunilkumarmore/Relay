"""call_inference retry policy, with the HTTP layer mocked."""

from __future__ import annotations

import pytest
import requests

from relay.identity import Identity
from relay.worker import daemon
from relay.worker.daemon import Runtime, WorkerConfig, call_inference

CFG = WorkerConfig(
    inference_registry="http://registry",
    worker_id="w1",
    session_id="s1",
    machine_id="m1",
    identity=Identity.generate(),
)
RUNTIME = Runtime(
    worker_id="w1",
    session_id="s1",
    machine_id="m1",
    inference_node="n1",
    steps_completed=0,
    next_step_number=1,
    next_problem="p",
)


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)


@pytest.fixture
def no_sleep(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr(daemon.time, "sleep", waits.append)
    return waits


def test_succeeds_first_try(monkeypatch, no_sleep):
    monkeypatch.setattr(
        requests, "post", lambda *a, **k: FakeResponse(200, {"response": "ok", "latency_ms": 5})
    )
    assert call_inference(CFG, RUNTIME, "p", 32)["response"] == "ok"
    assert no_sleep == []


def test_client_error_does_not_retry(monkeypatch, no_sleep):
    calls = {"n": 0}

    def post(*args, **kwargs):
        calls["n"] += 1
        return FakeResponse(400)

    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(requests.HTTPError):
        call_inference(CFG, RUNTIME, "p", 32)
    assert calls["n"] == 1, "a 4xx is our fault; retrying just wastes the provider's time"
    assert no_sleep == []


def test_server_error_retries_with_exponential_backoff(monkeypatch, no_sleep):
    calls = {"n": 0}

    def post(*args, **kwargs):
        calls["n"] += 1
        return FakeResponse(503)

    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(RuntimeError, match="Inference failed after 4 attempts"):
        call_inference(CFG, RUNTIME, "p", 32)
    assert calls["n"] == 4
    assert no_sleep == [1, 2, 4]


def test_timeout_retries_then_succeeds(monkeypatch, no_sleep):
    calls = {"n": 0}

    def post(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.Timeout("slow")
        return FakeResponse(200, {"response": "eventually"})

    monkeypatch.setattr(requests, "post", post)
    assert call_inference(CFG, RUNTIME, "p", 32)["response"] == "eventually"
    assert no_sleep == [1, 2]


def test_connection_error_retries(monkeypatch, no_sleep):
    calls = {"n": 0}

    def post(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.ConnectionError("refused")
        return FakeResponse(200, {"response": "ok"})

    monkeypatch.setattr(requests, "post", post)
    assert call_inference(CFG, RUNTIME, "p", 32)["response"] == "ok"
    assert no_sleep == [1]


def test_401_re_registers_and_retries_within_the_same_attempt(monkeypatch, no_sleep):
    """The registry restarted or pruned us as stale — say hello again."""
    posts: list[str] = []

    def post(url, json=None, timeout=None, auth=None):
        posts.append(url)
        if url.endswith("/worker/register"):
            return FakeResponse(200, {"inference_node_id": "n1"})
        return FakeResponse(401) if posts.count(url) == 1 else FakeResponse(200, {"response": "ok"})

    monkeypatch.setattr(requests, "post", post)
    assert call_inference(CFG, RUNTIME, "p", 32)["response"] == "ok"
    assert posts == [
        "http://registry/inference/complete",
        "http://registry/worker/register",
        "http://registry/inference/complete",
    ]
    assert no_sleep == [], "re-registering is not a backoff-worthy failure"


def test_max_retries_is_configurable(monkeypatch, no_sleep):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(500))
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        call_inference(CFG, RUNTIME, "p", 32, max_retries=1)
    assert no_sleep == [1]
