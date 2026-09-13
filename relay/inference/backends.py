"""Inference backends.

A backend is anything that can turn a prompt into text, say which models it
serves, and report whether it is alive. Keeping this behind a protocol is what
lets the whole system be tested without a GPU, an LLM, or a network.

``complete`` returns input and output tokens separately. A marketplace bills the
two at different rates, and the split has to be measured where generation
happens — it cannot be reconstructed afterwards from a single total.
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

    def complete(
        self, prompt: str, max_tokens: int = 1200, model: str | None = None
    ) -> tuple[str, int, int]:
        """Return ``(text, tokens_in, tokens_out)`` or raise ``BackendError``."""
        ...

    def models(self) -> list[str]:
        """Models this backend can actually serve right now."""
        ...

    def health(self) -> bool: ...


class OllamaBackend:
    name = "ollama"

    def __init__(self, host: str, model: str = "", timeout: float = 180.0) -> None:
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

    def models(self) -> list[str]:
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=10.0)
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            raise BackendError(str(exc)) from exc
        names = []
        for entry in body.get("models") or []:
            name = str(entry.get("name", ""))
            if name:
                names.append(name)
                # Ollama reports "llama3:latest"; accept the bare name too.
                if ":" in name:
                    names.append(name.split(":", 1)[0])
        return sorted(set(names))

    def complete(
        self, prompt: str, max_tokens: int = 1200, model: str | None = None
    ) -> tuple[str, int, int]:
        payload = {
            "model": model or self.model,
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


class OpenAICompatBackend:
    """Anything speaking the OpenAI completions API: vLLM, llama.cpp, LM Studio.

    Most self-hosted servers expose this, so supporting it roughly multiplies the
    hardware that can join the market without Relay caring what runs underneath.
    """

    name = "openai"

    def __init__(
        self, base_url: str, model: str = "", *, api_key: str = "", timeout: float = 180.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def health(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/models", headers=self._headers(), timeout=10.0)
            response.raise_for_status()
            return True
        except Exception:
            return False

    def models(self) -> list[str]:
        try:
            response = httpx.get(f"{self.base_url}/models", headers=self._headers(), timeout=10.0)
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            raise BackendError(str(exc)) from exc
        return sorted({str(entry.get("id", "")) for entry in body.get("data") or []} - {""})

    def complete(
        self, prompt: str, max_tokens: int = 1200, model: str | None = None
    ) -> tuple[str, int, int]:
        payload = {
            "model": model or self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            raise BackendError(str(exc)) from exc

        choices = body.get("choices") or []
        text = str(choices[0].get("message", {}).get("content", "")).strip() if choices else ""
        usage = body.get("usage") or {}
        return text, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


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
        available_models: list[str] | None = None,
    ) -> None:
        self.model = model
        self.latency_ms = latency_ms
        self.fail_calls = set(fail_calls or ())
        self.healthy = healthy
        self.available_models = available_models if available_models is not None else None
        self.calls = 0
        self.prompts: list[str] = []

    def health(self) -> bool:
        return self.healthy

    def models(self) -> list[str]:
        if self.available_models is not None:
            return list(self.available_models)
        return [self.model] if self.model else []

    def complete(
        self, prompt: str, max_tokens: int = 1200, model: str | None = None
    ) -> tuple[str, int, int]:
        self.calls += 1
        self.prompts.append(prompt)
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000.0)
        if self.calls in self.fail_calls:
            raise BackendError(f"injected failure on call {self.calls}")

        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        label = model or self.model
        # Mirror the response contract a real provider is asked for, so the
        # parsing path is exercised rather than stubbed around.
        text = (
            f"<reasoning>\n[{label}] working {digest[16:32]}\n</reasoning>\n"
            f"<solution>\n[{label}] answer {digest[:16]}\n</solution>"
        )
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
    if kind in {"openai", "openai-compatible", "vllm"}:
        return OpenAICompatBackend(
            config.get("OPENAI_BASE_URL", "http://localhost:8000/v1") or "http://localhost:8000/v1",
            config.get("OLLAMA_MODEL", "") or "",
            api_key=config.get("OPENAI_API_KEY"),
            timeout=config.get_float("OLLAMA_TIMEOUT", 180.0),
        )
    if kind != "ollama":
        raise BackendError(f"Unknown RELAY_BACKEND: {kind}")

    return OllamaBackend(
        config.get("OLLAMA_HOST", "http://localhost:11434") or "http://localhost:11434",
        config.get("OLLAMA_MODEL", "llama3") or "llama3",
        timeout=config.get_float("OLLAMA_TIMEOUT", 180.0),
    )
