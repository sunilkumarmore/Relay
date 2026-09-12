"""Receipts — the unit of settlement.

A receipt is what turns "the provider says it did some work" into something both
sides can check and neither can quietly revise. The provider issues and signs it;
the consumer verifies it against the offer it selected and the request it
actually sent, then countersigns.

The verification is the substance. A signature alone only proves the provider
wrote the numbers; checking the hashes proves they describe *this* request and
*this* response, and checking the prices proves they match the terms advertised.
A receipt the consumer has not countersigned is unacknowledged: recorded, but not
agreed, and Phase 5 decides who was right.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from pydantic import BaseModel, Field

from relay import identity as ident
from relay.identity import Identity
from relay.provider.offers import Offer, now_utc

STATUS_UNACKNOWLEDGED = "unacknowledged"
STATUS_ACKNOWLEDGED = "acknowledged"
STATUS_DISPUTED = "disputed"

SIGNED_FIELDS = (
    "receipt_id",
    "job_id",
    "step_number",
    "consumer_node_id",
    "provider_node_id",
    "offer_id",
    "model",
    "request_hash",
    "response_hash",
    "tokens_in",
    "tokens_out",
    "latency_ms",
    "price_in_per_1k",
    "price_out_per_1k",
    "amount_credits",
    "issued_at",
)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_fingerprint(prompt: str, max_tokens: int, model: str) -> str:
    """Covers everything that determines what was asked for, so a provider
    cannot bill for a cheaper request than it received (or vice versa)."""
    payload = {"prompt": prompt, "max_tokens": max_tokens, "model": model}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def price_for(price_in_per_1k: float, price_out_per_1k: float, tokens_in: int, tokens_out: int) -> float:
    raw = (tokens_in / 1000.0) * price_in_per_1k + (tokens_out / 1000.0) * price_out_per_1k
    return round(raw, 6)


class Receipt(BaseModel):
    receipt_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str
    step_number: int
    consumer_node_id: str
    provider_node_id: str
    offer_id: str
    model: str = ""
    request_hash: str
    response_hash: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    price_in_per_1k: float
    price_out_per_1k: float
    amount_credits: float
    issued_at: str = ""
    provider_signature: str = ""
    consumer_signature: str = ""
    status: str = STATUS_UNACKNOWLEDGED
    dispute_reason: str = ""

    # -- signing ----------------------------------------------------------
    def canonical_bytes(self) -> bytes:
        payload = {field: getattr(self, field) for field in SIGNED_FIELDS}
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def issued_by(self, identity: Identity) -> Receipt:
        if identity.node_id != self.provider_node_id:
            raise ValueError("a receipt must be issued by the provider that earned it")
        return self.model_copy(
            update={"provider_signature": identity.sign(self.canonical_bytes()).hex()}
        )

    def acknowledged_by(self, identity: Identity) -> Receipt:
        if identity.node_id != self.consumer_node_id:
            raise ValueError("a receipt must be acknowledged by the consumer that owes it")
        return self.model_copy(
            update={
                "consumer_signature": identity.sign(self.canonical_bytes()).hex(),
                "status": STATUS_ACKNOWLEDGED,
            }
        )

    def disputed(self, reason: str) -> Receipt:
        return self.model_copy(update={"status": STATUS_DISPUTED, "dispute_reason": reason[:500]})

    def _valid(self, node_id: str, signature: str) -> bool:
        if not signature:
            return False
        try:
            raw = bytes.fromhex(signature)
        except ValueError:
            return False
        return ident.verify(node_id, self.canonical_bytes(), raw)

    def provider_signature_is_valid(self) -> bool:
        return self._valid(self.provider_node_id, self.provider_signature)

    def consumer_signature_is_valid(self) -> bool:
        return self._valid(self.consumer_node_id, self.consumer_signature)

    @property
    def is_acknowledged(self) -> bool:
        return self.status == STATUS_ACKNOWLEDGED and self.consumer_signature_is_valid()

    def to_row(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Receipt:
        return cls(**{k: v for k, v in row.items() if k in cls.model_fields})


def build_receipt(
    identity: Identity,
    *,
    job_id: str,
    step_number: int,
    consumer_node_id: str,
    offer: Offer,
    request_hash: str,
    response_text: str,
    tokens_in: int,
    tokens_out: int,
    latency_ms: int,
) -> Receipt:
    """Issue and sign a receipt for one completed call."""
    receipt = Receipt(
        job_id=job_id,
        step_number=step_number,
        consumer_node_id=consumer_node_id,
        provider_node_id=identity.node_id,
        offer_id=offer.offer_id,
        model=offer.model,
        request_hash=request_hash,
        response_hash=hash_text(response_text),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
        price_in_per_1k=offer.price_in_per_1k,
        price_out_per_1k=offer.price_out_per_1k,
        amount_credits=offer.price(tokens_in, tokens_out),
        issued_at=now_utc().isoformat(),
    )
    return receipt.issued_by(identity)


def verify_receipt(
    receipt: Receipt,
    *,
    offer: Offer,
    consumer_node_id: str,
    request_hash: str,
    response_text: str,
    job_id: str,
    step_number: int,
) -> list[str]:
    """Check a receipt before paying it. Returns the reasons not to.

    An empty list means: this receipt is authentic, it describes the request we
    actually made and the response we actually received, and it charges the
    price that was advertised.
    """
    problems: list[str] = []

    if not receipt.provider_signature_is_valid():
        problems.append("provider signature is not valid")
    if receipt.provider_node_id != offer.provider_node_id:
        problems.append("receipt is from a different provider than the offer")
    if receipt.consumer_node_id != consumer_node_id:
        problems.append("receipt names a different consumer")
    if receipt.offer_id != offer.offer_id:
        problems.append("receipt cites a different offer")
    if receipt.job_id != job_id or receipt.step_number != step_number:
        problems.append("receipt is for a different step")

    if receipt.request_hash != request_hash:
        problems.append("request hash does not match what we sent")
    if receipt.response_hash != hash_text(response_text):
        problems.append("response hash does not match what we received")

    if receipt.price_in_per_1k != offer.price_in_per_1k:
        problems.append("input price does not match the offer")
    if receipt.price_out_per_1k != offer.price_out_per_1k:
        problems.append("output price does not match the offer")

    if receipt.tokens_in < 0 or receipt.tokens_out < 0:
        problems.append("token counts cannot be negative")

    expected = price_for(
        receipt.price_in_per_1k, receipt.price_out_per_1k, receipt.tokens_in, receipt.tokens_out
    )
    if abs(receipt.amount_credits - expected) > 1e-9:
        problems.append(f"amount {receipt.amount_credits} does not match {expected} at these prices")

    return problems
