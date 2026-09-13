"""Offers — what a provider advertises, and what it commits to.

An offer is a signed statement: *this node, at this endpoint, will do this kind
of work, at this price, up to this concurrency, until this time.* Signing matters
because the offer is also the price quote a consumer later checks its receipts
against (Phase 4). An unsigned offer is a rumour.

Two kinds of work fit through the same structure. An inference offer names a
model, a context window, and a price per thousand tokens. A task offer names the
task types it will run and a price per million work units. A node may advertise
both, one, or — for a phone with no model on it — only tasks. The inference
fields default to empty rather than being required, which is the whole change
the task pivot needed here.

Offers expire. A provider that stops republishing drops out of the directory on
its own, so a dead node cannot keep advertising capacity it no longer has.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from relay import identity as ident
from relay.identity import Identity

DEFAULT_OFFER_TTL_SECONDS = 300

# Fields covered by the signature. `signature` itself is excluded, and so is
# nothing else: every term a consumer relies on is signed.
SIGNED_FIELDS = (
    "offer_id",
    "provider_node_id",
    "endpoint_url",
    "model",
    "context_window",
    "price_in_per_1k",
    "price_out_per_1k",
    "task_types",
    "price_per_mega_unit",
    "price_per_task",
    "max_payload_bytes",
    "max_concurrency",
    "region",
    "capabilities",
    "published_at",
    "expires_at",
)


def now_utc() -> datetime:
    return datetime.now(UTC)


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class Offer(BaseModel):
    offer_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    provider_node_id: str
    endpoint_url: str
    model: str = ""
    context_window: int = 0
    price_in_per_1k: float = 0.0
    price_out_per_1k: float = 0.0
    # Task work. `task_types` is the capability gate; the two prices are the
    # quote. Per-unit is the honest one — work units are fixed by the task, so
    # neither side can argue about the quantity afterwards.
    task_types: tuple[str, ...] = ()
    price_per_mega_unit: float = 0.0
    price_per_task: float = 0.0
    max_payload_bytes: int = 0
    max_concurrency: int = 1
    region: str = "unknown"
    capabilities: dict[str, Any] = Field(default_factory=dict)
    published_at: str = ""
    expires_at: str = ""
    signature: str = ""

    # -- signing ----------------------------------------------------------
    def canonical_bytes(self) -> bytes:
        """Stable bytes both sides hash. Sorted keys, no whitespace drift."""
        payload = {field: getattr(self, field) for field in SIGNED_FIELDS}
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def signed_by(self, identity: Identity) -> Offer:
        if identity.node_id != self.provider_node_id:
            raise ValueError("an offer must be signed by the node that makes it")
        return self.model_copy(update={"signature": identity.sign(self.canonical_bytes()).hex()})

    def signature_is_valid(self) -> bool:
        if not self.signature:
            return False
        try:
            raw = bytes.fromhex(self.signature)
        except ValueError:
            return False
        return ident.verify(self.provider_node_id, self.canonical_bytes(), raw)

    # -- lifetime ---------------------------------------------------------
    def is_expired(self, at: datetime | None = None) -> bool:
        expires = parse_time(self.expires_at)
        if expires is None:
            return True
        return expires <= (at or now_utc())

    def is_usable(self, at: datetime | None = None) -> bool:
        return self.signature_is_valid() and not self.is_expired(at)

    # -- pricing ----------------------------------------------------------
    def price(self, tokens_in: int, tokens_out: int) -> float:
        """Cost in credits for one inference call. Rounded to a millicredit so
        both sides arrive at the same number from the same inputs."""
        raw = (tokens_in / 1000.0) * self.price_in_per_1k + (tokens_out / 1000.0) * self.price_out_per_1k
        return round(raw, 6)

    def task_price(self, work_units: int) -> float:
        """Cost in credits for one task.

        `work_units` comes off the task, where the consumer signed it and the
        queue re-derived it from the payload. So unlike a token count, it is not
        something the provider reports and nobody can check — which is why there
        is no task equivalent of the token over-claim dispute.
        """
        raw = self.price_per_task + (work_units / 1_000_000.0) * self.price_per_mega_unit
        return round(raw, 6)

    def serves_task_type(self, task_type: str) -> bool:
        return task_type in self.task_types

    def to_row(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Offer:
        known = {k: v for k, v in row.items() if k in cls.model_fields}
        return cls(**known)


def build_offer(
    identity: Identity,
    *,
    endpoint_url: str,
    model: str = "",
    context_window: int = 0,
    price_in_per_1k: float = 0.0,
    price_out_per_1k: float = 0.0,
    task_types: tuple[str, ...] | list[str] = (),
    price_per_mega_unit: float = 0.0,
    price_per_task: float = 0.0,
    max_payload_bytes: int = 0,
    max_concurrency: int = 1,
    region: str = "unknown",
    capabilities: dict[str, Any] | None = None,
    ttl_seconds: int = DEFAULT_OFFER_TTL_SECONDS,
    offer_id: str | None = None,
    at: datetime | None = None,
) -> Offer:
    published = at or now_utc()
    offer = Offer(
        offer_id=offer_id or str(uuid.uuid4()),
        provider_node_id=identity.node_id,
        endpoint_url=endpoint_url,
        model=model,
        context_window=context_window,
        price_in_per_1k=price_in_per_1k,
        price_out_per_1k=price_out_per_1k,
        task_types=tuple(task_types),
        price_per_mega_unit=price_per_mega_unit,
        price_per_task=price_per_task,
        max_payload_bytes=max_payload_bytes,
        max_concurrency=max_concurrency,
        region=region,
        capabilities=capabilities or {},
        published_at=published.isoformat(),
        expires_at=(published + timedelta(seconds=ttl_seconds)).isoformat(),
    )
    return offer.signed_by(identity)
