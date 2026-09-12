"""Provider configuration.

A provider's terms are a document, not a scatter of environment variables: it
declares which backends it runs and, per model, the context window and the price
it will honour. Those are commitments a consumer signs against, so they belong
somewhere reviewable and diffable.

The file is validated on start, and a provider refuses to start if it advertises
a model its backend does not actually serve — the cheapest possible way to avoid
selling something you cannot deliver.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from relay.inference.backends import (
    BackendError,
    FakeBackend,
    InferenceBackend,
    OllamaBackend,
    OpenAICompatBackend,
)

DEFAULT_CONFIG_PATH = "relay-provider.yaml"


class ProviderConfigError(RuntimeError):
    pass


@dataclass
class ModelOffering:
    model: str
    backend: str
    context_window: int
    price_in_per_1k: float
    price_out_per_1k: float
    max_concurrency: int = 1


@dataclass
class ProviderConfig:
    endpoint_url: str
    region: str = "unknown"
    offer_ttl_seconds: int = 300
    models: list[ModelOffering] = field(default_factory=list)
    backends: dict[str, InferenceBackend] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)

    def backend_for(self, model: str) -> InferenceBackend:
        for offering in self.models:
            if offering.model == model:
                return self.backends[offering.backend]
        raise ProviderConfigError(f"This provider does not serve {model!r}")

    def offering_for(self, model: str) -> ModelOffering:
        for offering in self.models:
            if offering.model == model:
                return offering
        raise ProviderConfigError(f"This provider does not serve {model!r}")

    @property
    def default_model(self) -> str:
        if not self.models:
            raise ProviderConfigError("Provider serves no models")
        return self.models[0].model

    def validate(self, *, check_backends: bool = True) -> None:
        if not self.endpoint_url:
            raise ProviderConfigError("endpoint_url is required — consumers need somewhere to call")
        if not self.models:
            raise ProviderConfigError("A provider must offer at least one model")

        seen: set[str] = set()
        for offering in self.models:
            if offering.model in seen:
                raise ProviderConfigError(f"Duplicate model in config: {offering.model}")
            seen.add(offering.model)

            if offering.backend not in self.backends:
                raise ProviderConfigError(
                    f"Model {offering.model!r} names backend {offering.backend!r}, "
                    f"which is not defined. Known: {sorted(self.backends) or 'none'}"
                )
            if offering.context_window <= 0:
                raise ProviderConfigError(f"{offering.model}: context_window must be positive")
            if offering.price_in_per_1k < 0 or offering.price_out_per_1k < 0:
                raise ProviderConfigError(f"{offering.model}: prices cannot be negative")
            if offering.max_concurrency <= 0:
                raise ProviderConfigError(f"{offering.model}: max_concurrency must be at least 1")

        if not check_backends:
            return

        for offering in self.models:
            backend = self.backends[offering.backend]
            try:
                available = backend.models()
            except BackendError:
                # A backend that is down at start-up is an operational problem,
                # not a misconfiguration. Offers simply will not publish until
                # the health check passes.
                continue
            if available and offering.model not in available:
                raise ProviderConfigError(
                    f"Backend {offering.backend!r} does not serve {offering.model!r}. "
                    f"It has: {', '.join(sorted(available)) or 'nothing'}"
                )


def build_backend(name: str, spec: dict[str, Any]) -> InferenceBackend:
    kind = str(spec.get("kind", "ollama")).lower()
    timeout = float(spec.get("timeout", 180))

    if kind == "ollama":
        return OllamaBackend(str(spec.get("host", "http://localhost:11434")), "", timeout=timeout)
    if kind in {"openai", "openai-compatible", "vllm", "llama.cpp"}:
        api_key = ""
        if spec.get("api_key_env"):
            api_key = os.getenv(str(spec["api_key_env"]), "")
        return OpenAICompatBackend(
            str(spec.get("base_url", "http://localhost:8000/v1")),
            "",
            api_key=api_key or str(spec.get("api_key", "")),
            timeout=timeout,
        )
    if kind == "fake":
        return FakeBackend(
            model="",
            latency_ms=int(spec.get("latency_ms", 0)),
            available_models=list(spec.get("models", [])) or None,
        )
    raise ProviderConfigError(f"Unknown backend kind {kind!r} for backend {name!r}")


def load_provider_config(path: str | os.PathLike[str] | None = None) -> ProviderConfig:
    config_path = Path(path or DEFAULT_CONFIG_PATH)
    if not config_path.exists():
        raise ProviderConfigError(
            f"No provider config at {config_path}. "
            "Copy relay-provider.example.yaml and set your endpoint and prices."
        )
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ProviderConfigError(f"{config_path} is not valid YAML: {exc}") from exc

    node = data.get("node") or {}
    backends = {name: build_backend(name, spec or {}) for name, spec in (data.get("backends") or {}).items()}

    models = []
    for entry in data.get("models") or []:
        if "model" not in entry:
            raise ProviderConfigError("Every entry under `models:` needs a `model:` name")
        models.append(
            ModelOffering(
                model=str(entry["model"]),
                backend=str(entry.get("backend", next(iter(backends), ""))),
                context_window=int(entry.get("context_window", 8192)),
                price_in_per_1k=float(entry.get("price_in_per_1k", 0.0)),
                price_out_per_1k=float(entry.get("price_out_per_1k", 0.0)),
                max_concurrency=int(entry.get("max_concurrency", 1)),
            )
        )

    config = ProviderConfig(
        endpoint_url=str(node.get("endpoint_url", "")),
        region=str(node.get("region", "unknown")),
        offer_ttl_seconds=int(node.get("offer_ttl_seconds", 300)),
        models=models,
        backends=backends,
        capabilities=dict(node.get("capabilities") or {}),
    )
    config.validate()
    return config
