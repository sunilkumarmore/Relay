"""The inference registry: an HTTP front door for one inference backend.

Two things changed the trust model here from "everyone is honest":

Registration binds a ``worker_id`` to the Ed25519 node that claimed it, and
every later call for that worker must be signed by the same node. A label is no
longer a credential.

Registrations and the request log live in the store, so a registry that restarts
does not forget who is working. The in-memory copy is a cache, not the record.
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

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

from relay import config
from relay.auth import SignedBy, require_signature
from relay.inference.backends import InferenceBackend, backend_from_env
from relay.store import Store, store_from_env

STALE_AFTER = timedelta(seconds=60)
PRUNE_INTERVAL_SECONDS = 30
HEALTH_TTL = timedelta(seconds=5)

ANONYMOUS = SignedBy(node_id="anonymous", timestamp=0)


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
    node_id: str
    session_id: str
    machine_id: str
    registered_at: datetime
    last_heartbeat: datetime
    steps_completed: int


def now_utc() -> datetime:
    return datetime.now(UTC)


def _parse(ts: str | None, fallback: datetime) -> datetime:
    if not ts:
        return fallback
    try:
        parsed = datetime.fromisoformat(ts)
    except ValueError:
        return fallback
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


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
        self.hydrate()

    # -- durability -------------------------------------------------------
    def hydrate(self) -> None:
        """Reload registrations written before this process started."""
        if self.store is None:
            return
        try:
            rows = self.store.list_nodes(self.node_id)
            logged = self.store.list_registry_requests(self.node_id, limit=self.request_log.maxlen)
        except Exception:
            return

        now = now_utc()
        for row in rows:
            self.active_workers[row["worker_id"]] = WorkerInfo(
                worker_id=row["worker_id"],
                node_id=row.get("node_id", ""),
                session_id=row.get("session_id", ""),
                machine_id=row.get("machine_id", ""),
                registered_at=_parse(row.get("registered_at"), now),
                last_heartbeat=_parse(row.get("last_heartbeat"), now),
                steps_completed=int(row.get("steps_completed") or 0),
            )
        for row in logged:
            self.request_log.append(
                {
                    "ts": row.get("requested_at", now.isoformat()),
                    "worker_id": row.get("worker_id", ""),
                    "session_id": row.get("session_id", ""),
                    "latency_ms": int(row.get("latency_ms") or 0),
                    "success": bool(row.get("success", True)),
                }
            )
        self.total_requests = len(self.request_log)

    def _persist(self, info: WorkerInfo) -> None:
        if self.store is None:
            return
        try:
            self.store.upsert_node(
                worker_id=info.worker_id,
                node_id=info.node_id,
                session_id=info.session_id,
                machine_id=info.machine_id,
                inference_node=self.node_id,
                steps_completed=info.steps_completed,
                last_heartbeat=info.last_heartbeat.isoformat(),
                registered_at=info.registered_at.isoformat(),
            )
        except Exception:
            pass

    def _forget(self, worker_id: str) -> None:
        if self.store is None:
            return
        try:
            self.store.delete_node(worker_id)
        except Exception:
            pass

    # -- health -----------------------------------------------------------
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
            self._forget(wid)
        return stale

    # -- authorization ----------------------------------------------------
    def _authorize(self, worker_id: str, caller: SignedBy) -> WorkerInfo:
        """A worker may only be driven by the node that registered it."""
        info = self.active_workers.get(worker_id)
        if info is None:
            raise HTTPException(status_code=401, detail="worker_not_registered")
        if info.node_id != caller.node_id:
            raise HTTPException(status_code=403, detail="worker_belongs_to_another_node")
        return info

    # -- operations -------------------------------------------------------
    def register(self, req: RegisterRequest, caller: SignedBy = ANONYMOUS) -> str:
        ts = now_utc()
        existing = self.active_workers.get(req.worker_id)
        if existing is not None and existing.node_id != caller.node_id:
            # Someone else already holds this label on this provider.
            raise HTTPException(status_code=403, detail="worker_id_taken")

        info = WorkerInfo(
            worker_id=req.worker_id,
            node_id=caller.node_id,
            session_id=req.session_id,
            machine_id=req.machine_id,
            registered_at=existing.registered_at if existing else ts,
            last_heartbeat=ts,
            # Preserve progress so the dashboard doesn't flash back to 0 on re-register.
            steps_completed=existing.steps_completed if existing else 0,
        )
        self.active_workers[req.worker_id] = info
        self._persist(info)
        return self.node_id

    def heartbeat(self, req: HeartbeatRequest, caller: SignedBy = ANONYMOUS) -> None:
        info = self.active_workers.get(req.worker_id)
        if info is None or info.node_id != caller.node_id:
            # An unknown or mismatched heartbeat tells us nothing; ignore it.
            return
        info.last_heartbeat = now_utc()
        info.steps_completed = req.steps_completed
        self._persist(info)

    def deregister(self, worker_id: str, caller: SignedBy = ANONYMOUS) -> None:
        info = self.active_workers.get(worker_id)
        if info is not None and info.node_id != caller.node_id:
            raise HTTPException(status_code=403, detail="worker_belongs_to_another_node")
        self.active_workers.pop(worker_id, None)
        self._forget(worker_id)

    def complete(self, req: InferenceRequest, caller: SignedBy = ANONYMOUS) -> dict[str, Any]:
        info = self._authorize(req.worker_id, caller)

        started = time.perf_counter()
        try:
            text, tokens_in, tokens_out = self.backend.complete(req.prompt, req.max_tokens)
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            self._log_request(info, req.session_id, latency_ms, success=False)
            detail = f"backend_error: {exc}"
            raise HTTPException(status_code=502, detail=detail) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        # The schema stores one combined figure; the split is reported to the caller.
        tokens_used = tokens_in + tokens_out
        info.last_heartbeat = now_utc()
        self._log_request(info, req.session_id, latency_ms, success=True)

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

    def _log_request(self, info: WorkerInfo, session_id: str, latency_ms: int, *, success: bool) -> None:
        self.total_requests += 1
        self.request_log.append(
            {
                "ts": now_utc().isoformat(),
                "worker_id": info.worker_id,
                "session_id": session_id,
                "latency_ms": latency_ms,
                "success": success,
            }
        )
        if self.store is None:
            return
        try:
            self.store.insert_registry_request(
                worker_id=info.worker_id,
                node_id=info.node_id,
                session_id=session_id,
                inference_node=self.node_id,
                latency_ms=latency_ms,
                success=success,
            )
        except Exception:
            pass

    def status(self) -> dict[str, Any]:
        one_minute_ago = now_utc() - timedelta(minutes=1)
        recent = [r for r in self.request_log if _parse(r["ts"], one_minute_ago) >= one_minute_ago]
        avg_latency = int(sum(r["latency_ms"] for r in recent) / len(recent)) if recent else 0
        return {
            "active_workers": [
                {
                    "worker_id": info.worker_id,
                    "node_id": info.node_id,
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


def create_app(
    registry: Registry,
    *,
    prune_in_background: bool = True,
    require_signatures: bool = True,
) -> FastAPI:
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

    async def caller(request: Request) -> SignedBy:
        if not require_signatures:
            return ANONYMOUS
        return await require_signature(request)

    @app.get("/health")
    def health() -> dict[str, Any]:
        # Public: a liveness probe must work before anyone has an identity.
        return {
            "status": "ok",
            "model": registry.backend.model,
            "backend": registry.backend.name,
            "node_id": registry.node_id,
            "requires_signature": require_signatures,
            # Old name kept so an existing worker build still reads this correctly.
            "ollama_connected": registry.backend_healthy(),
            "backend_connected": registry.backend_healthy(),
        }

    @app.post("/worker/register")
    def worker_register(req: RegisterRequest, who: SignedBy = Depends(caller)) -> dict[str, Any]:
        return {"accepted": True, "inference_node_id": registry.register(req, who)}

    @app.post("/worker/heartbeat")
    def worker_heartbeat(req: HeartbeatRequest, who: SignedBy = Depends(caller)) -> dict[str, Any]:
        registry.heartbeat(req, who)
        return {"ok": True}

    @app.post("/worker/deregister")
    def worker_deregister(req: DeregisterRequest, who: SignedBy = Depends(caller)) -> dict[str, Any]:
        registry.deregister(req.worker_id, who)
        return {"ok": True}

    @app.post("/inference/complete")
    def inference_complete(req: InferenceRequest, who: SignedBy = Depends(caller)) -> dict[str, Any]:
        return registry.complete(req, who)

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


def require_signatures_from_env() -> bool:
    return config.get("RELAY_REQUIRE_SIGNATURES", "1").lower() not in {"0", "false", "no"}


def app_from_env() -> FastAPI:
    return create_app(registry_from_env(), require_signatures=require_signatures_from_env())


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
