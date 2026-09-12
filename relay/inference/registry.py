"""Compatibility surface for the inference registry.

The registry became :mod:`relay.provider.server` in Phase 2 — it no longer just
fronts one backend, it publishes terms and serves whoever can pay under them.
Everything here re-exports that module so existing imports and the two-machine
demo keep working.
"""

from __future__ import annotations

import socket

from fastapi import FastAPI

from relay import config
from relay.identity import identity_from_env
from relay.inference.backends import backend_from_env
from relay.provider.server import (  # noqa: F401
    ANONYMOUS,
    HEALTH_TTL,
    PRUNE_INTERVAL_SECONDS,
    STALE_AFTER,
    DeregisterRequest,
    HeartbeatRequest,
    InferenceRequest,
    Provider,
    RegisterRequest,
    Registry,
    Saturated,
    WorkerInfo,
    create_app,
    now_utc,
    require_signatures_from_env,
)


def registry_from_env() -> Registry:
    """A single-backend provider built from environment variables."""
    config.load_env()
    return Registry(
        backend_from_env(),
        __import__("relay.store", fromlist=["store_from_env"]).store_from_env(optional=True),
        node_id=config.get("MACHINE_ID") or socket.gethostname(),
        max_tokens_default=config.get_int("MAX_TOKENS", 1200),
        identity=identity_from_env(),
    )


def app_from_env() -> FastAPI:
    return create_app(registry_from_env(), require_signatures=require_signatures_from_env())


def __getattr__(name: str):
    if name == "app":
        return app_from_env()
    raise AttributeError(name)


def run() -> None:
    import uvicorn

    config.load_env()
    uvicorn.run(app_from_env(), host="0.0.0.0", port=config.get_int("REGISTRY_PORT", 8765))


if __name__ == "__main__":
    run()
