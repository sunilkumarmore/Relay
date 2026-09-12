"""Provider config: refuse to advertise what you cannot serve."""

from __future__ import annotations

import pytest
import yaml

from relay.inference.backends import FakeBackend, OllamaBackend, OpenAICompatBackend
from relay.provider.config import (
    ModelOffering,
    ProviderConfig,
    ProviderConfigError,
    load_provider_config,
)

BASE = {
    "node": {"endpoint_url": "http://10.0.0.5:8765", "region": "lab"},
    "backends": {"fake": {"kind": "fake", "models": ["llama3", "mistral"]}},
    "models": [
        {
            "model": "llama3",
            "backend": "fake",
            "context_window": 8192,
            "price_in_per_1k": 0.05,
            "price_out_per_1k": 0.15,
            "max_concurrency": 2,
        }
    ],
}


def write(tmp_path, data):
    path = tmp_path / "relay-provider.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_loads_a_valid_config(tmp_path):
    cfg = load_provider_config(write(tmp_path, BASE))
    assert cfg.endpoint_url == "http://10.0.0.5:8765"
    assert cfg.region == "lab"
    assert cfg.default_model == "llama3"
    assert cfg.offering_for("llama3").price_out_per_1k == 0.15
    assert isinstance(cfg.backend_for("llama3"), FakeBackend)


def test_refuses_a_model_the_backend_does_not_serve(tmp_path):
    """The cheapest possible way to avoid selling what you cannot deliver."""
    data = {**BASE, "models": [{**BASE["models"][0], "model": "gpt-4"}]}
    with pytest.raises(ProviderConfigError, match="does not serve 'gpt-4'"):
        load_provider_config(write(tmp_path, data))


def test_refuses_a_model_naming_an_undefined_backend(tmp_path):
    data = {**BASE, "models": [{**BASE["models"][0], "backend": "nope"}]}
    with pytest.raises(ProviderConfigError, match="which is not defined"):
        load_provider_config(write(tmp_path, data))


def test_requires_an_endpoint_consumers_can_reach(tmp_path):
    data = {**BASE, "node": {"region": "lab"}}
    with pytest.raises(ProviderConfigError, match="endpoint_url is required"):
        load_provider_config(write(tmp_path, data))


def test_requires_at_least_one_model(tmp_path):
    with pytest.raises(ProviderConfigError, match="at least one model"):
        load_provider_config(write(tmp_path, {**BASE, "models": []}))


def test_rejects_duplicate_models(tmp_path):
    data = {**BASE, "models": [BASE["models"][0], BASE["models"][0]]}
    with pytest.raises(ProviderConfigError, match="Duplicate model"):
        load_provider_config(write(tmp_path, data))


@pytest.mark.parametrize(
    "override,message",
    [
        ({"context_window": 0}, "context_window must be positive"),
        ({"price_in_per_1k": -1}, "prices cannot be negative"),
        ({"price_out_per_1k": -0.5}, "prices cannot be negative"),
        ({"max_concurrency": 0}, "max_concurrency must be at least 1"),
    ],
)
def test_rejects_nonsense_terms(tmp_path, override, message):
    data = {**BASE, "models": [{**BASE["models"][0], **override}]}
    with pytest.raises(ProviderConfigError, match=message):
        load_provider_config(write(tmp_path, data))


def test_missing_file_says_what_to_do(tmp_path):
    with pytest.raises(ProviderConfigError, match="Copy relay-provider.example.yaml"):
        load_provider_config(tmp_path / "absent.yaml")


def test_invalid_yaml_is_a_clear_error(tmp_path):
    path = tmp_path / "relay-provider.yaml"
    path.write_text("node: [unclosed\n", encoding="utf-8")
    with pytest.raises(ProviderConfigError, match="not valid YAML"):
        load_provider_config(path)


def test_builds_an_openai_compatible_backend(tmp_path):
    data = {
        **BASE,
        "backends": {"vllm": {"kind": "openai", "base_url": "http://localhost:8000/v1"}},
        "models": [{**BASE["models"][0], "backend": "vllm"}],
    }
    # The backend is unreachable, so model validation is skipped rather than
    # failing start-up: a backend being down is operational, not a config error.
    cfg = load_provider_config(write(tmp_path, data))
    assert isinstance(cfg.backend_for("llama3"), OpenAICompatBackend)


def test_builds_an_ollama_backend(tmp_path):
    data = {
        **BASE,
        "backends": {"ollama": {"kind": "ollama", "host": "http://localhost:11434"}},
        "models": [{**BASE["models"][0], "backend": "ollama"}],
    }
    assert isinstance(load_provider_config(write(tmp_path, data)).backend_for("llama3"), OllamaBackend)


def test_unknown_backend_kind_is_rejected(tmp_path):
    data = {**BASE, "backends": {"weird": {"kind": "quantum"}}}
    with pytest.raises(ProviderConfigError, match="Unknown backend kind"):
        load_provider_config(write(tmp_path, data))


def test_serves_several_models_with_different_prices(tmp_path):
    data = {
        **BASE,
        "models": [
            BASE["models"][0],
            {
                "model": "mistral",
                "backend": "fake",
                "context_window": 4096,
                "price_in_per_1k": 0.01,
                "price_out_per_1k": 0.03,
                "max_concurrency": 5,
            },
        ],
    }
    cfg = load_provider_config(write(tmp_path, data))
    assert [o.model for o in cfg.models] == ["llama3", "mistral"]
    assert cfg.offering_for("mistral").max_concurrency == 5
    assert cfg.default_model == "llama3"


def test_asking_for_an_unserved_model_is_an_error():
    cfg = ProviderConfig(
        endpoint_url="http://x",
        models=[ModelOffering("a", "b", 10, 0.0, 0.0)],
        backends={"b": FakeBackend()},
    )
    with pytest.raises(ProviderConfigError, match="does not serve 'zzz'"):
        cfg.offering_for("zzz")
