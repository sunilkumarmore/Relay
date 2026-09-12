"""The inference registry: an HTTP front door for one inference backend.

Workers register, heartbeat, and send completion requests here. The registry
records telemetry to the store when one is available, and keeps a small in-memory
view for ``/inference/status`` and the dashboard.
"""

from __future__ import annotations

import asyncio
import socket
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from relay import config
from relay.inference.backends import BackendError, InferenceBackend, backend_from_env
from relay.store import Store, store_from_env

STALE_AFTER = timedelta(seconds=60)
PRUNE_INTERVAL_SECONDS = 30
HEALTH_TTL = timedelta(seconds=5)


class RegisterRequest(BaseModel):
    worker_id: str
    session_id: str
    machine_id: str


class HeartbeatRequest(BaseModel):
    worker_id: str
    last_checkpoint: int
    steps_completed: int


class DeregisterRequest(BaseModel):
    worker_id: str


class InferenceRequest(BaseModel):
    worker_id: str
    session_id: str
    prompt: str
    max_tokens: int = 1200


@dataclass
class WorkerInfo:
    worker_id: str
    session_id: str
    machine_id: str
    registered_at: datetime
    last_heartbeat: datetime
    steps_completed: int


def now_utc() -> datetime:
    return datetime.now(UTC)


class Registry:
    """Registry state and behavior, independent of the HTTP layer."""

    def __init__(
        self,
        backend: InferenceBackend,
        store: Store | None = None,
        *,
        node_id: str = "",
        max_tokens_default: int = 1200,
    ) -> None:
        self.backend = backend
        self.store = store
        self.node_id = node_id or socket.gethostname()
        self.max_tokens_default = max_tokens_default
        self.active_workers: dict[str, WorkerInfo] = {}
        self.request_log: deque[dict[str, Any]] = deque(maxlen=100)
        self.total_requests = 0
        self._health_cache: tuple[bool, datetime] | None = None

    def backend_healthy(self) -> bool:
        now = now_utc()
        if self._health_cache is not None:
            result, ts = self._health_cache
            if now - ts < HEALTH_TTL:
                return result
        result = self.backend.health()
        self._health_cache = (result, now)
        return result

    def prune_stale_workers(self) -> list[str]:
        cutoff = now_utc() - STALE_AFTER
        stale = [wid for wid, info in self.active_workers.items() if info.last_heartbeat < cutoff]
        for wid in stale:
            del self.active_workers[wid]
        return stale

    def register(self, req: RegisterRequest) -> str:
        ts = now_utc()
        existing = self.active_workers.get(req.worker_id)
        self.active_workers[req.worker_id] = WorkerInfo(
            worker_id=req.worker_id,
            session_id=req.session_id,
            machine_id=req.machine_id,
            registered_at=existing.registered_at if existing else ts,
            last_heartbeat=ts,
            # Preserve progress so the dashboard doesn't flash back to 0 on re-register.
            steps_completed=existing.steps_completed if existing else 0,
        )
        return self.node_id

    def heartbeat(self, req: HeartbeatRequest) -> None:
        info = self.active_workers.get(req.worker_id)
        if info:
            info.last_heartbeat = now_utc()
            info.steps_completed = req.steps_completed

    def deregister(self, worker_id: str) -> None:
        self.active_workers.pop(worker_id, None)

    def complete(self, req: InferenceRequest) -> dict[str, Any]:
        if req.worker_id not in self.active_workers:
            raise HTTPException(status_code=401, detail="worker_not_registered")

        started = time.perf_counter()
        try:
            text, tokens_in, tokens_out = self.backend.complete(req.prompt, req.max_tokens)
        except BackendError as exc:
            raise HTTPException(status_code=502, detail=f"backend_error: {exc}") from exc
        except Exception as exc:  # a backend that raises something unexpected is still a bad gateway
            raise HTTPException(status_code=502, detail=f"backend_error: {exc}") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        # The schema stores one combined figure; the split is reported to the caller.
        tokens_used = tokens_in + tokens_out
        self.total_requests += 1
        self.active_workers[req.worker_id].last_heartbeat = now_utc()
        self.request_log.append(
            {
                "ts": now_utc().isoformat(),
                "worker_id": req.worker_id,
                "session_id": req.session_id,
                "latency_ms": latency_ms,
                "success": True,
            }
        )

        if self.store is not None:
            try:
                self.store.insert_inference_log(
                    worker_id=req.worker_id,
                    session_id=req.session_id,
                    inference_node=self.node_id,
                    latency_ms=latency_ms,
                    tokens_used=tokens_used,
                    success=True,
                )
            except Exception:
                pass

        return {
            "response": text,
            "tokens_used": tokens_used,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency_ms,
            "inference_node_id": self.node_id,
        }

    def status(self) -> dict[str, Any]:
        one_minute_ago = now_utc() - timedelta(minutes=1)
        recent = [r for r in self.request_log if datetime.fromisoformat(r["ts"]) >= one_minute_ago]
        avg_latency = int(sum(r["latency_ms"] for r in recent) / len(recent)) if recent else 0
        return {
            "active_workers": [
                {
                    "worker_id": info.worker_id,
                    "session_id": info.session_id,
                    "machine_id": info.machine_id,
                    "steps_completed": info.steps_completed,
                    "last_heartbeat": info.last_heartbeat.isoformat(),
                }
                for info in self.active_workers.values()
            ],
            "requests_last_minute": len(recent),
            "avg_latency_ms": avg_latency,
            "total_requests": self.total_requests,
            "inference_node_id": self.node_id,
            "recent_calls": list(self.request_log)[-10:],
        }


def create_app(registry: Registry, *, prune_in_background: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = None
        if prune_in_background:

            async def prune_loop() -> None:
                while True:
                    await asyncio.sleep(PRUNE_INTERVAL_SECONDS)
                    registry.prune_stale_workers()

            task = asyncio.create_task(prune_loop())
        yield
        if task is not None:
            task.cancel()

    app = FastAPI(title="Relay Inference Registry", lifespan=lifespan)
    app.state.registry = registry

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": registry.backend.model,
            "backend": registry.backend.name,
            # Old name kept so an existing worker build still reads this correctly.
            "ollama_connected": registry.backend_healthy(),
            "backend_connected": registry.backend_healthy(),
        }

    @app.post("/worker/register")
    def worker_register(req: RegisterRequest) -> dict[str, Any]:
        return {"accepted": True, "inference_node_id": registry.register(req)}

    @app.post("/worker/heartbeat")
    def worker_heartbeat(req: HeartbeatRequest) -> dict[str, Any]:
        registry.heartbeat(req)
        return {"ok": True}

    @app.post("/worker/deregister")
    def worker_deregister(req: DeregisterRequest) -> dict[str, Any]:
        registry.deregister(req.worker_id)
        return {"ok": True}

    @app.post("/inference/complete")
    def inference_complete(req: InferenceRequest) -> dict[str, Any]:
        return registry.complete(req)

    @app.get("/inference/status")
    def inference_status() -> dict[str, Any]:
        return registry.status()

    return app


def registry_from_env() -> Registry:
    config.load_env()
    return Registry(
        backend_from_env(),
        store_from_env(optional=True),
        node_id=config.get("MACHINE_ID") or socket.gethostname(),
        max_tokens_default=config.get_int("MAX_TOKENS", 1200),
    )


def app_from_env() -> FastAPI:
    return create_app(registry_from_env())


def __getattr__(name: str):
    # Supports `uvicorn relay.inference.registry:app` without building a registry
    # (and reaching for Supabase) at import time.
    if name == "app":
        return app_from_env()
    raise AttributeError(name)


def run() -> None:
    import uvicorn

    config.load_env()
    uvicorn.run(app_from_env(), host="0.0.0.0", port=config.get_int("REGISTRY_PORT", 8765))


if __name__ == "__main__":
    run()
