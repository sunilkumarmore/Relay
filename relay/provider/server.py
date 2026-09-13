"""The provider: a node that sells inference.

This was the "inference registry". The change of name is the change of role —
it no longer just fronts one Ollama for a known set of workers, it publishes
terms and serves whoever can pay under them.

Three things are new here:

*Offers.* The provider advertises each model it serves with a price, a context
window and a capacity, signed, into the shared directory, and republishes before
the advertisement expires.

*Models.* One provider can serve several, each with its own price and its own
concurrency limit.

*Back-pressure.* Capacity is finite and now stated. Past the limit the provider
returns 429 with Retry-After rather than queueing silently and blowing every
consumer's latency — an honest refusal is worth more than a slow yes.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

from relay import config as env
from relay.auth import SignedBy, require_signature
from relay.identity import Identity, identity_from_env
from relay.inference.backends import InferenceBackend
from relay.provider.config import (
    ModelOffering,
    ProviderConfig,
    ProviderConfigError,
    load_provider_config,
)
from relay.provider.offers import Offer, build_offer
from relay.receipts import build_receipt, request_fingerprint
from relay.store import Store, store_from_env

STALE_AFTER = timedelta(seconds=60)
PRUNE_INTERVAL_SECONDS = 30
HEALTH_TTL = timedelta(seconds=5)
RETRY_AFTER_SECONDS = 5

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
    model: str | None = None
    # Which step of the job this is. Receipts are per step, so that a consumer
    # can tie what it was charged to what it got.
    step_number: int = 0


@dataclass
class WorkerInfo:
    worker_id: str
    node_id: str
    session_id: str
    machine_id: str
    registered_at: datetime
    last_heartbeat: datetime
    steps_completed: int


class Saturated(Exception):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.model = model


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


class Provider:
    """Provider state and behavior, independent of the HTTP layer."""

    def __init__(
        self,
        provider_config: ProviderConfig,
        store: Store | None = None,
        *,
        identity: Identity | None = None,
        node_id: str = "",
        min_stake: float = 0.0,
    ) -> None:
        self.config = provider_config
        # Credits this node must have at risk before its offers are listed.
        self.min_stake = min_stake
        self.store = store
        self.identity = identity or Identity.generate()
        # The human label for this provider, used in telemetry and the dashboard.
        # The signing key remains the real identity.
        self.node_id = node_id or socket.gethostname()
        self.active_workers: dict[str, WorkerInfo] = {}
        self.request_log: deque[dict[str, Any]] = deque(maxlen=100)
        self.total_requests = 0
        self._health_cache: dict[str, tuple[bool, datetime]] = {}
        self._semaphores = {
            offering.model: threading.BoundedSemaphore(offering.max_concurrency)
            for offering in provider_config.models
        }
        self._offers: dict[str, Offer] = {}
        # A legacy single-backend registry has no real terms to advertise.
        self.publishes_offers = True
        self.hydrate()

    # -- offers -----------------------------------------------------------
    def build_offers(self, at: datetime | None = None) -> list[Offer]:
        return [
            build_offer(
                self.identity,
                endpoint_url=self.config.endpoint_url,
                model=offering.model,
                context_window=offering.context_window,
                price_in_per_1k=offering.price_in_per_1k,
                price_out_per_1k=offering.price_out_per_1k,
                max_concurrency=offering.max_concurrency,
                region=self.config.region,
                capabilities=self.config.capabilities,
                ttl_seconds=self.config.offer_ttl_seconds,
                # Stable per (provider, model) so republishing refreshes the row
                # instead of piling up a new one every cycle.
                offer_id=f"{self.identity.node_id[:16]}:{offering.model}",
                at=at,
            )
            for offering in self.config.models
        ]

    def stake(self) -> float:
        if self.store is None:
            return 0.0
        try:
            account = self.store.get_account(self.identity.node_id) or {}
        except Exception:
            return 0.0
        return float(account.get("stake") or 0.0)

    def is_staked(self) -> bool:
        return self.min_stake <= 0 or self.stake() >= self.min_stake

    def publish_offers(self, at: datetime | None = None) -> list[Offer]:
        """Refresh this provider's advertisements in the directory."""
        if not self.is_staked():
            # Nothing at risk, nothing listed. A slash has to be able to cost
            # something, or the penalty is theatre.
            print(
                f"Not advertising: stake {self.stake()} is below the {self.min_stake} required. "
                "Top up with: python -m relay.controller wallet stake --amount N"
            )
            self.withdraw_offers()
            self._record_event("understaked", "", {"stake": self.stake(), "required": self.min_stake})
            return []

        offers = self.build_offers(at)
        self._offers = {offer.model: offer for offer in offers}
        if self.store is not None:
            for offer in offers:
                try:
                    self.store.upsert_offer(offer.to_row())
                except Exception:
                    pass
        return offers

    def withdraw_offers(self) -> None:
        """Stop advertising. A provider shutting down cleanly should not leave
        capacity listed that nobody can reach."""
        self._offers = {}
        if self.store is not None:
            try:
                self.store.delete_offers_for(self.identity.node_id)
            except Exception:
                pass

    def current_offers(self) -> list[Offer]:
        return list(self._offers.values())

    # -- durability -------------------------------------------------------
    def hydrate(self) -> None:
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

    def _record_event(self, kind: str, model: str, detail: dict[str, Any] | None = None) -> None:
        if self.store is None:
            return
        try:
            self.store.insert_provider_event(self.identity.node_id, kind, model, detail)
        except Exception:
            pass

    # -- health -----------------------------------------------------------
    @property
    def backend(self) -> InferenceBackend:
        return self.config.backend_for(self.config.default_model)

    def backend_healthy(self, model: str | None = None) -> bool:
        name = model or self.config.default_model
        now = now_utc()
        cached = self._health_cache.get(name)
        if cached is not None and now - cached[1] < HEALTH_TTL:
            return cached[0]
        try:
            result = self.config.backend_for(name).health()
        except ProviderConfigError:
            result = False
        self._health_cache[name] = (result, now)
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
            raise HTTPException(status_code=403, detail="worker_id_taken")

        info = WorkerInfo(
            worker_id=req.worker_id,
            node_id=caller.node_id,
            session_id=req.session_id,
            machine_id=req.machine_id,
            registered_at=existing.registered_at if existing else ts,
            last_heartbeat=ts,
            steps_completed=existing.steps_completed if existing else 0,
        )
        self.active_workers[req.worker_id] = info
        self._persist(info)
        return self.node_id

    def heartbeat(self, req: HeartbeatRequest, caller: SignedBy = ANONYMOUS) -> None:
        info = self.active_workers.get(req.worker_id)
        if info is None or info.node_id != caller.node_id:
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

    def resolve_model(self, requested: str | None) -> ModelOffering:
        if requested is None:
            return self.config.offering_for(self.config.default_model)
        try:
            return self.config.offering_for(requested)
        except ProviderConfigError as exc:
            # Asking for something we never advertised is the caller's mistake.
            raise HTTPException(status_code=404, detail=f"model_not_served: {requested}") from exc

    def complete(self, req: InferenceRequest, caller: SignedBy = ANONYMOUS) -> dict[str, Any]:
        info = self._authorize(req.worker_id, caller)
        offering = self.resolve_model(req.model)
        semaphore = self._semaphores[offering.model]

        if not semaphore.acquire(blocking=False):
            # Refuse now rather than queue: the consumer can try another provider
            # in less time than it would spend waiting behind our backlog.
            self._record_event(
                "saturated",
                offering.model,
                {"worker_id": req.worker_id, "max_concurrency": offering.max_concurrency},
            )
            raise Saturated(offering.model)

        started = time.perf_counter()
        try:
            text, tokens_in, tokens_out = self.config.backend_for(offering.model).complete(
                req.prompt, req.max_tokens, offering.model
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            self._log_request(info, req.session_id, latency_ms, success=False)
            raise HTTPException(status_code=502, detail=f"backend_error: {exc}") from exc
        finally:
            semaphore.release()

        latency_ms = int((time.perf_counter() - started) * 1000)
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
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                )
            except Exception:
                pass

        offer = self._offers.get(offering.model)
        payload: dict[str, Any] = {
            "response": text,
            "tokens_used": tokens_used,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency_ms,
            "model": offering.model,
            "inference_node_id": self.node_id,
            "provider_node_id": self.identity.node_id,
            "offer_id": offer.offer_id if offer else "",
        }

        if offer is not None and caller.node_id != ANONYMOUS.node_id:
            # Bill for the work. The receipt is signed here, at the point the
            # numbers are actually known — reconstructing it later would be
            # reconstructing the provider's own claim about itself.
            receipt = build_receipt(
                self.identity,
                job_id=req.session_id,
                step_number=req.step_number,
                consumer_node_id=caller.node_id,
                offer=offer,
                request_hash=request_fingerprint(req.prompt, req.max_tokens, offering.model),
                response_text=text,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
            )
            payload["receipt"] = receipt.model_dump()
            if self.store is not None:
                try:
                    # Record it unacknowledged; the consumer countersigns its copy.
                    self.store.upsert_receipt(receipt.to_row())
                except Exception:
                    pass

        return payload

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

    def earnings(self) -> dict[str, Any]:
        if self.store is None:
            return {"acknowledged": 0.0, "unacknowledged": 0.0, "disputed": 0.0, "receipts": 0}
        rows = self.store.list_receipts(provider_node_id=self.identity.node_id)
        totals = {"acknowledged": 0.0, "unacknowledged": 0.0, "disputed": 0.0}
        for row in rows:
            bucket = str(row.get("status", "unacknowledged"))
            if bucket in totals:
                totals[bucket] += float(row.get("amount_credits") or 0)
        return {**{k: round(v, 6) for k, v in totals.items()}, "receipts": len(rows)}

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
            "provider_node_id": self.identity.node_id,
            "models": [o.model for o in self.config.models],
            "recent_calls": list(self.request_log)[-10:],
        }


def create_app(
    provider: Provider,
    *,
    prune_in_background: bool = True,
    require_signatures: bool = True,
    publish_offers: bool = True,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        tasks = []
        advertising = publish_offers and provider.publishes_offers
        if advertising:
            provider.publish_offers()

            async def republish_loop() -> None:
                # Refresh at half the TTL: one missed cycle should not drop us
                # out of the directory.
                interval = max(5, provider.config.offer_ttl_seconds // 2)
                while True:
                    await asyncio.sleep(interval)
                    provider.publish_offers()

            tasks.append(asyncio.create_task(republish_loop()))

        if prune_in_background:

            async def prune_loop() -> None:
                while True:
                    await asyncio.sleep(PRUNE_INTERVAL_SECONDS)
                    provider.prune_stale_workers()

            tasks.append(asyncio.create_task(prune_loop()))
        yield
        for task in tasks:
            task.cancel()
        if advertising:
            provider.withdraw_offers()

    app = FastAPI(title="Relay Provider", lifespan=lifespan)
    app.state.provider = provider

    async def caller(request: Request) -> SignedBy:
        if not require_signatures:
            return ANONYMOUS
        return await require_signature(request)

    @app.get("/health")
    def health() -> dict[str, Any]:
        # Public: a liveness probe must work before anyone has an identity.
        return {
            "status": "ok",
            "model": provider.config.default_model,
            "models": [o.model for o in provider.config.models],
            "backend": provider.backend.name,
            "node_id": provider.node_id,
            "provider_node_id": provider.identity.node_id,
            "requires_signature": require_signatures,
            "ollama_connected": provider.backend_healthy(),
            "backend_connected": provider.backend_healthy(),
        }

    @app.get("/offer")
    def offers() -> dict[str, Any]:
        # Public: the directory is how consumers find this node at all.
        return {"offers": [o.model_dump() for o in provider.current_offers()]}

    @app.post("/worker/register")
    def worker_register(req: RegisterRequest, who: SignedBy = Depends(caller)) -> dict[str, Any]:
        return {"accepted": True, "inference_node_id": provider.register(req, who)}

    @app.post("/worker/heartbeat")
    def worker_heartbeat(req: HeartbeatRequest, who: SignedBy = Depends(caller)) -> dict[str, Any]:
        provider.heartbeat(req, who)
        return {"ok": True}

    @app.post("/worker/deregister")
    def worker_deregister(req: DeregisterRequest, who: SignedBy = Depends(caller)) -> dict[str, Any]:
        provider.deregister(req.worker_id, who)
        return {"ok": True}

    @app.post("/inference/complete")
    def inference_complete(
        req: InferenceRequest, response: Response, who: SignedBy = Depends(caller)
    ) -> Any:
        try:
            return provider.complete(req, who)
        except Saturated as exc:
            return Response(
                content=f'{{"detail":"saturated: {exc.model}"}}',
                status_code=429,
                headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
                media_type="application/json",
            )

    @app.get("/inference/status")
    def inference_status() -> dict[str, Any]:
        return provider.status()

    @app.get("/earnings")
    def earnings(who: SignedBy = Depends(caller)) -> dict[str, Any]:
        """What this provider has earned, and what is still unacknowledged."""
        if who.node_id != provider.identity.node_id:
            raise HTTPException(status_code=403, detail="not_your_earnings")
        return provider.earnings()

    return app


class Registry(Provider):
    """Single-backend provider, as the inference registry used to be.

    Kept so the two-machine demo and everything written against the old shape
    keep working while the config-driven provider becomes the real entrypoint.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        store: Store | None = None,
        *,
        node_id: str = "",
        max_tokens_default: int = 1200,
        identity: Identity | None = None,
        endpoint_url: str = "http://localhost:8765",
    ) -> None:
        model = backend.model or "default"
        provider_config = ProviderConfig(
            endpoint_url=endpoint_url,
            models=[
                ModelOffering(
                    model=model,
                    backend="default",
                    context_window=8192,
                    price_in_per_1k=0.0,
                    price_out_per_1k=0.0,
                    # Unlimited in spirit: the legacy registry never refused work.
                    max_concurrency=1024,
                )
            ],
            backends={"default": backend},
        )
        super().__init__(provider_config, store, identity=identity, node_id=node_id)
        self.max_tokens_default = max_tokens_default
        self._backend = backend
        self.publishes_offers = False

    @property
    def backend(self) -> InferenceBackend:
        return self._backend


def provider_from_env() -> Provider:
    env.load_env()
    return Provider(
        load_provider_config(env.get("RELAY_PROVIDER_CONFIG") or None),
        store_from_env(optional=True),
        identity=identity_from_env(),
        node_id=env.get("MACHINE_ID") or socket.gethostname(),
        min_stake=env.get_float("RELAY_MIN_STAKE", 0.0),
    )


def require_signatures_from_env() -> bool:
    return env.get("RELAY_REQUIRE_SIGNATURES", "1").lower() not in {"0", "false", "no"}


def app_from_env() -> FastAPI:
    return create_app(provider_from_env(), require_signatures=require_signatures_from_env())


def __getattr__(name: str):
    if name == "app":
        return app_from_env()
    raise AttributeError(name)


def run() -> None:
    import uvicorn

    env.load_env()
    uvicorn.run(app_from_env(), host="0.0.0.0", port=env.get_int("REGISTRY_PORT", 8765))


if __name__ == "__main__":
    run()
