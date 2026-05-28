from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
import socket
import time
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from .ollama_client import OllamaClient
except ImportError:
    from ollama_client import OllamaClient

try:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT / "worker") not in sys.path:
        sys.path.append(str(ROOT / "worker"))
    from checkpoint_client import RelayCheckpointClient, RelayCheckpointError
except Exception:
    RelayCheckpointClient = None  # type: ignore[assignment]
    RelayCheckpointError = RuntimeError  # type: ignore[assignment]


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


load_dotenv(os.getenv("ENV_FILE", ".env"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3").strip() or "llama3"
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "180"))
INFERENCE_NODE_ID = os.getenv("MACHINE_ID", socket.gethostname()).strip() or socket.gethostname()
REGISTRY_PORT = int(os.getenv("REGISTRY_PORT", "8765"))
MAX_TOKENS_DEFAULT = int(os.getenv("MAX_TOKENS", "1200"))

ollama = OllamaClient(OLLAMA_HOST, OLLAMA_MODEL, timeout=OLLAMA_TIMEOUT)

supabase: RelayCheckpointClient | None = None
if RelayCheckpointClient is not None:
    try:
        supabase = RelayCheckpointClient.from_env(os.getenv("ENV_FILE", ".env"))
        supabase.verify_connection()
    except Exception:
        supabase = None

active_workers: dict[str, WorkerInfo] = {}
request_log: deque[dict[str, Any]] = deque(maxlen=100)
total_requests = 0

# Cache for the Ollama health check — avoids a live HTTP call on every /health request.
_ollama_health_cache: tuple[bool, datetime] | None = None
_OLLAMA_HEALTH_TTL = timedelta(seconds=5)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def ollama_healthy() -> bool:
    global _ollama_health_cache
    now = now_utc()
    if _ollama_health_cache is not None:
        result, ts = _ollama_health_cache
        if now - ts < _OLLAMA_HEALTH_TTL:
            return result
    result = ollama.health()
    _ollama_health_cache = (result, now)
    return result


def prune_stale_workers() -> None:
    cutoff = now_utc() - timedelta(seconds=60)
    stale = [wid for wid, info in active_workers.items() if info.last_heartbeat < cutoff]
    for wid in stale:
        del active_workers[wid]


async def _prune_loop() -> None:
    while True:
        await asyncio.sleep(30)
        prune_stale_workers()


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    task = asyncio.create_task(_prune_loop())
    yield
    task.cancel()


app = FastAPI(title="Relay Inference Registry", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": OLLAMA_MODEL,
        "ollama_connected": ollama_healthy(),
    }


@app.post("/worker/register")
def worker_register(req: RegisterRequest) -> dict[str, Any]:
    ts = now_utc()
    existing = active_workers.get(req.worker_id)
    active_workers[req.worker_id] = WorkerInfo(
        worker_id=req.worker_id,
        session_id=req.session_id,
        machine_id=req.machine_id,
        registered_at=existing.registered_at if existing else ts,
        last_heartbeat=ts,
        # Preserve progress so the dashboard doesn't flash back to 0 on re-register.
        steps_completed=existing.steps_completed if existing else 0,
    )
    return {"accepted": True, "inference_node_id": INFERENCE_NODE_ID}


@app.post("/worker/heartbeat")
def worker_heartbeat(req: HeartbeatRequest) -> dict[str, Any]:
    info = active_workers.get(req.worker_id)
    if info:
        info.last_heartbeat = now_utc()
        info.steps_completed = req.steps_completed
    return {"ok": True}


@app.post("/worker/deregister")
def worker_deregister(req: DeregisterRequest) -> dict[str, Any]:
    active_workers.pop(req.worker_id, None)
    return {"ok": True}


@app.post("/inference/complete")
def inference_complete(req: InferenceRequest) -> dict[str, Any]:
    global total_requests

    if req.worker_id not in active_workers:
        raise HTTPException(status_code=401, detail="worker_not_registered")

    started = time.perf_counter()
    success = False
    tokens_used = 0
    try:
        response_text, tokens_used = ollama.complete(req.prompt, req.max_tokens)
        success = True
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ollama_error: {exc}") from exc
    finally:
        latency_ms = int((time.perf_counter() - started) * 1000)

    total_requests += 1
    active_workers[req.worker_id].last_heartbeat = now_utc()
    request_log.append(
        {
            "ts": now_utc().isoformat(),
            "worker_id": req.worker_id,
            "session_id": req.session_id,
            "latency_ms": latency_ms,
            "success": success,
        }
    )

    if supabase is not None:
        try:
            supabase.insert_inference_log(
                worker_id=req.worker_id,
                session_id=req.session_id,
                inference_node=INFERENCE_NODE_ID,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                success=success,
            )
        except Exception:
            pass

    return {
        "response": response_text,
        "tokens_used": tokens_used,
        "latency_ms": latency_ms,
        "inference_node_id": INFERENCE_NODE_ID,
    }


@app.get("/inference/status")
def inference_status() -> dict[str, Any]:
    one_minute_ago = now_utc() - timedelta(minutes=1)
    recent = [row for row in request_log if datetime.fromisoformat(row["ts"]) >= one_minute_ago]
    avg_latency = int(sum(row["latency_ms"] for row in recent) / len(recent)) if recent else 0

    return {
        "active_workers": [
            {
                "worker_id": info.worker_id,
                "session_id": info.session_id,
                "machine_id": info.machine_id,
                "steps_completed": info.steps_completed,
                "last_heartbeat": info.last_heartbeat.isoformat(),
            }
            for info in active_workers.values()
        ],
        "requests_last_minute": len(recent),
        "avg_latency_ms": avg_latency,
        "total_requests": total_requests,
        "inference_node_id": INFERENCE_NODE_ID,
        "recent_calls": list(request_log)[-10:],
    }


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=REGISTRY_PORT)


if __name__ == "__main__":
    main()
