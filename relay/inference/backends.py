"""Inference backends.

A backend is anything that can turn a prompt into text and report whether it is
alive. Keeping this behind a protocol is what lets the whole system be tested
without a GPU, an LLM, or a network.

``complete`` returns input and output tokens separately. The current Supabase
schema only stores a combined ``tokens_used``, so callers sum them — but priced
inference bills the two at different rates, and the split has to be measured at
the point of generation, not reconstructed later.
"""

from __future__ import annotations

import hashlib
import time
from typing import Protocol, runtime_checkable

import httpx

from relay import config


class BackendError(RuntimeError):
    pass


# Kept under the old name for anything still catching it.
OllamaClientError = BackendError


@runtime_checkable
class InferenceBackend(Protocol):
    name: str
    model: str

    def complete(self, prompt: str, max_tokens: int = 1200) -> tuple[str, int, int]:
        """Return ``(text, tokens_in, tokens_out)`` or raise ``BackendError``."""
        ...

    def health(self) -> bool: ...


class OllamaBackend:
    name = "ollama"

    def __init__(self, host: str, model: str, timeout: float = 180.0) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def health(self) -> bool:
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=10.0)
            response.raise_for_status()
            return True
        except Exception:
            return False

    def complete(self, prompt: str, max_tokens: int = 1200) -> tuple[str, int, int]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        try:
            response = httpx.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            raise BackendError(str(exc)) from exc

        text = str(body.get("response", "")).strip()
        tokens_in = int(body.get("prompt_eval_count") or 0)
        tokens_out = int(body.get("eval_count") or 0)
        return text, tokens_in, tokens_out


class FakeBackend:
    """Deterministic backend for tests and local demos.

    The same prompt always produces the same text, so a run that is evicted and
    resumed can be compared byte-for-byte against one that ran straight through.
    """

    name = "fake"

    def __init__(
        self,
        model: str = "fake-model",
        *,
        latency_ms: int = 0,
        fail_calls: set[int] | None = None,
        healthy: bool = True,
    ) -> None:
        self.model = model
        self.latency_ms = latency_ms
        self.fail_calls = set(fail_calls or ())
        self.healthy = healthy
        self.calls = 0
        self.prompts: list[str] = []

    def health(self) -> bool:
        return self.healthy

    def complete(self, prompt: str, max_tokens: int = 1200) -> tuple[str, int, int]:
        self.calls += 1
        self.prompts.append(prompt)
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000.0)
        if self.calls in self.fail_calls:
            raise BackendError(f"injected failure on call {self.calls}")

        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        text = f"[{self.model}] answer {digest[:16]}"
        return text, max(1, len(prompt) // 4), max(1, len(text) // 4)


def backend_from_env(env_path: str | None = None) -> InferenceBackend:
    """Build the backend named by ``RELAY_BACKEND`` (default ``ollama``)."""
    config.load_env(env_path)
    kind = config.get("RELAY_BACKEND", "ollama").lower()

    if kind == "fake":
        raw = config.get("RELAY_FAKE_FAIL_CALLS")
        fail_calls = {int(x) for x in raw.split(",") if x.strip()} if raw else set()
        return FakeBackend(
            model=config.get("OLLAMA_MODEL", "fake-model") or "fake-model",
            latency_ms=config.get_int("RELAY_FAKE_LATENCY_MS", 0),
            fail_calls=fail_calls,
        )
    if kind != "ollama":
        raise BackendError(f"Unknown RELAY_BACKEND: {kind}")

    return OllamaBackend(
        config.get("OLLAMA_HOST", "http://localhost:11434") or "http://localhost:11434",
        config.get("OLLAMA_MODEL", "llama3") or "llama3",
        timeout=config.get_float("OLLAMA_TIMEOUT", 180.0),
    )
