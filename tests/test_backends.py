from __future__ import annotations

import httpx
import pytest

from relay.inference.backends import BackendError, FakeBackend, InferenceBackend, OllamaBackend


def test_fake_backend_implements_protocol():
    assert isinstance(FakeBackend(), InferenceBackend)


def test_fake_backend_is_deterministic():
    """A resumed run must be comparable to an uninterrupted one."""
    a, b = FakeBackend(), FakeBackend()
    assert a.complete("same prompt") == b.complete("same prompt")
    assert a.complete("one")[0] != a.complete("two")[0]


def test_fake_backend_reports_a_token_split():
    _, tokens_in, tokens_out = FakeBackend().complete("a prompt of some length")
    assert tokens_in > 0
    assert tokens_out > 0


def test_fake_backend_failure_injection():
    backend = FakeBackend(fail_calls={2})
    assert backend.complete("first")
    with pytest.raises(BackendError):
        backend.complete("second")
    assert backend.complete("third")
    assert backend.calls == 3


def test_fake_backend_records_prompts():
    backend = FakeBackend()
    backend.complete("p1")
    backend.complete("p2")
    assert backend.prompts == ["p1", "p2"]


def test_ollama_backend_splits_prompt_and_eval_counts(monkeypatch):
    def fake_post(url, json, timeout):  # noqa: A002
        assert json["options"]["num_predict"] == 64
        return httpx.Response(
            200,
            json={"response": "  hello  ", "prompt_eval_count": 11, "eval_count": 7},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    text, tokens_in, tokens_out = OllamaBackend("http://h", "m").complete("p", 64)
    assert (text, tokens_in, tokens_out) == ("hello", 11, 7)


def test_ollama_backend_missing_counts_default_to_zero(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, json, timeout: httpx.Response(  # noqa: A002
            200, json={"response": "x"}, request=httpx.Request("POST", url)
        ),
    )
    assert OllamaBackend("http://h", "m").complete("p") == ("x", 0, 0)


def test_ollama_backend_wraps_transport_errors(monkeypatch):
    def boom(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", boom)
    with pytest.raises(BackendError):
        OllamaBackend("http://h", "m").complete("p")


def test_ollama_health_is_false_when_unreachable(monkeypatch):
    def boom(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", boom)
    assert OllamaBackend("http://h", "m").health() is False
