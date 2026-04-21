from __future__ import annotations

import httpx


class OllamaClientError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, host: str, model: str) -> None:
        self.host = host.rstrip("/")
        self.model = model

    def health(self) -> bool:
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=10.0)
            response.raise_for_status()
            return True
        except Exception:
            return False

    def complete(self, prompt: str, max_tokens: int = 1200) -> tuple[str, int]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
            },
        }
        try:
            response = httpx.post(f"{self.host}/api/generate", json=payload, timeout=180.0)
            response.raise_for_status()
            body = response.json()
            text = str(body.get("response", "")).strip()
            tokens_used = int(body.get("eval_count") or 0) + int(body.get("prompt_eval_count") or 0)
            return text, tokens_used
        except Exception as exc:
            raise OllamaClientError(str(exc)) from exc
