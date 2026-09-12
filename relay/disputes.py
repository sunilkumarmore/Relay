"""Disputes, and the two that can be settled without a judge.

Most disagreements about AI output are subjective — "the answer was poor" is not
a thing a program can rule on, and pretending otherwise would make the mechanism
worse than useless. So this module adjudicates exactly two kinds, both decidable
from evidence both parties already signed:

*Hash mismatch.* The receipt claims a request or response that is not the one
recorded in the checkpoint. Decidable by recomputing the hashes.

*Token over-claim.* The receipt bills for more tokens than the recorded text can
account for. Decidable by re-counting.

Everything else is recorded, surfaced, and left to the humans. An upheld dispute
refunds the consumer and slashes the provider; a rejected one means the consumer
pays after all — disputing has to cost something, or it is free to cry wolf.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from relay import tokens as tokenizer
from relay.ledger import Ledger
from relay.provider.offers import now_utc
from relay.receipts import Receipt, hash_text, request_fingerprint
from relay.store import Store

OPEN = "open"
UPHELD = "upheld"
REJECTED = "rejected"
UNADJUDICATED = "unadjudicated"

REASON_HASH_MISMATCH = "hash_mismatch"
REASON_TOKEN_OVERCLAIM = "token_overclaim"
REASON_OTHER = "other"

OBJECTIVE_REASONS = (REASON_HASH_MISMATCH, REASON_TOKEN_OVERCLAIM)

# What an upheld dispute costs the provider on top of the refund, as a multiple
# of the disputed amount. Small enough not to be ruinous, large enough that
# over-claiming does not pay even when it usually goes unchallenged.
SLASH_MULTIPLE = 2.0

# A provider must hold at least this much before its offers are listed. Stake is
# what a slash draws from: a provider with nothing at risk has nothing to lose.
DEFAULT_MIN_STAKE = 1.0


@dataclass
class Dispute:
    receipt_id: str
    opened_by: str
    reason: str
    dispute_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = OPEN
    evidence: dict[str, Any] = field(default_factory=dict)
    opened_at: str = field(default_factory=lambda: now_utc().isoformat())
    resolved_at: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "dispute_id": self.dispute_id,
            "receipt_id": self.receipt_id,
            "opened_by": self.opened_by,
            "reason": self.reason,
            "status": self.status,
            "evidence": self.evidence,
            "opened_at": self.opened_at,
            "resolved_at": self.resolved_at,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Dispute:
        return cls(
            receipt_id=str(row["receipt_id"]),
            opened_by=str(row.get("opened_by", "")),
            reason=str(row.get("reason", REASON_OTHER)),
            dispute_id=str(row.get("dispute_id", uuid.uuid4())),
            status=str(row.get("status", OPEN)),
            evidence=dict(row.get("evidence") or {}),
            opened_at=str(row.get("opened_at", "")),
            resolved_at=str(row.get("resolved_at", "")),
        )


def open_dispute(
    store: Store, *, receipt_id: str, opened_by: str, reason: str, evidence: dict[str, Any] | None = None
) -> Dispute:
    dispute = Dispute(
        receipt_id=receipt_id, opened_by=opened_by, reason=reason, evidence=evidence or {}
    )
    store.upsert_dispute(dispute.to_row())
    return dispute


def _recorded_step(store: Store, receipt: Receipt) -> dict[str, Any] | None:
    """The checkpoint for this step — the consumer's record of what it sent and
    what it got. Both sides can read it, so neither can rewrite the argument."""
    for row in store.get_checkpoints(receipt.job_id):
        if int(row.get("step_number") or 0) == receipt.step_number:
            return row
    return None


def adjudicate(store: Store, ledger: Ledger, dispute: Dispute) -> Dispute:
    """Rule on a dispute, and move the credits that follow from the ruling."""
    row = store.get_receipt(dispute.receipt_id)
    if row is None:
        return _resolve(store, dispute, REJECTED, {"finding": "no such receipt"})

    receipt = Receipt.from_row(row)

    if dispute.reason not in OBJECTIVE_REASONS:
        # "The answer was bad" is not something a program should rule on.
        return _resolve(
            store,
            dispute,
            UNADJUDICATED,
            {"finding": "subjective disputes are recorded, not adjudicated"},
        )

    if not receipt.provider_signature_is_valid():
        return _uphold(store, ledger, dispute, receipt, {"finding": "provider signature invalid"})

    step = _recorded_step(store, receipt)
    if step is None:
        return _resolve(
            store, dispute, REJECTED, {"finding": "no checkpoint recorded for this step"}
        )

    prompt = str(step.get("problem", ""))
    solution = str(step.get("solution", ""))

    if dispute.reason == REASON_HASH_MISMATCH:
        expected_request = request_fingerprint(
            prompt, dispute.evidence.get("max_tokens", 1200), receipt.model
        )
        expected_response = hash_text(solution)
        findings = {}
        if receipt.response_hash != expected_response:
            findings["response_hash"] = {"claimed": receipt.response_hash, "actual": expected_response}
        if dispute.evidence.get("max_tokens") and receipt.request_hash != expected_request:
            findings["request_hash"] = {"claimed": receipt.request_hash, "actual": expected_request}
        if findings:
            return _uphold(store, ledger, dispute, receipt, findings)
        return _resolve(store, dispute, REJECTED, {"finding": "hashes match the recorded step"})

    # Token over-claim: re-count with the canonical tokenizer.
    findings = {}
    for label, text, claimed in (
        ("input", prompt, receipt.tokens_in),
        ("output", solution, receipt.tokens_out),
    ):
        measured = tokenizer.estimate(text, receipt.model)
        if not measured.permits(claimed):
            findings[label] = {
                "claimed": claimed,
                "measured": measured.tokens,
                "tolerance": measured.tolerance,
                "exact": measured.exact,
            }
    if findings:
        return _uphold(store, ledger, dispute, receipt, findings)
    return _resolve(store, dispute, REJECTED, {"finding": "token counts are within tolerance"})


def _uphold(
    store: Store, ledger: Ledger, dispute: Dispute, receipt: Receipt, findings: dict[str, Any]
) -> Dispute:
    amount = float(receipt.amount_credits)
    outcome: dict[str, Any] = {**findings}

    # The consumer gets its money back — as far as the provider can cover it.
    # A provider that cannot is exactly why stake is required to be listed.
    settled = store.list_ledger_entries(kind="settle", ref_receipt_id=receipt.receipt_id)
    refunded = 0.0
    if settled:
        refunded = ledger.recoverable(receipt.provider_node_id, amount)
        ledger.refund(
            receipt.consumer_node_id,
            receipt.provider_node_id,
            refunded,
            receipt_id=receipt.receipt_id,
            job_id=receipt.job_id,
        )
        if refunded < amount:
            outcome["shortfall"] = round(amount - refunded, 6)

    # The penalty is what makes over-claiming unprofitable even when it usually
    # goes unchallenged. It too can only take what is there.
    penalty = ledger.recoverable(receipt.provider_node_id, round(amount * SLASH_MULTIPLE, 6))
    if penalty > 0:
        ledger.slash(
            receipt.provider_node_id,
            penalty,
            reason=dispute.reason,
            ref_receipt_id=receipt.receipt_id,
        )
    _adjust_stake(store, ledger, receipt.provider_node_id)

    store.upsert_receipt({**receipt.to_row(), "status": "disputed", "dispute_reason": dispute.reason})
    return _resolve(store, dispute, UPHELD, {**outcome, "refunded": refunded, "slashed": penalty})


def _resolve(store: Store, dispute: Dispute, status: str, evidence: dict[str, Any]) -> Dispute:
    resolved = Dispute(
        receipt_id=dispute.receipt_id,
        opened_by=dispute.opened_by,
        reason=dispute.reason,
        dispute_id=dispute.dispute_id,
        status=status,
        evidence={**dispute.evidence, **evidence},
        opened_at=dispute.opened_at,
        resolved_at=now_utc().isoformat(),
    )
    store.upsert_dispute(resolved.to_row())

    if status == REJECTED:
        # The dispute failed, so the bill stands.
        row = store.get_receipt(dispute.receipt_id)
        if row is not None and row.get("status") == "disputed":
            store.upsert_receipt({**row, "status": "unacknowledged", "dispute_reason": ""})
    return resolved


def _adjust_stake(store: Store, ledger: Ledger, node_id: str) -> None:
    """Stake tracks the balance: a slashed provider may fall below the floor and
    drop out of the directory until it tops up."""
    balance = ledger.balance(node_id)
    account = store.get_account(node_id) or {}
    stake = min(float(account.get("stake") or 0.0), max(balance, 0.0))
    store.upsert_account(node_id, balance, stake)


def set_stake(store: Store, ledger: Ledger, node_id: str, amount: float) -> float:
    """Put credits at risk. A provider cannot stake more than it holds."""
    balance = ledger.balance(node_id)
    staked = max(0.0, min(amount, balance))
    store.upsert_account(node_id, balance, staked)
    return staked


def staked_providers(store: Store, min_stake: float = DEFAULT_MIN_STAKE) -> set[str]:
    try:
        rows = store.list_accounts()
    except Exception:
        return set()
    return {str(r["node_id"]) for r in rows if float(r.get("stake") or 0.0) >= min_stake}


def adjudicate_open(store: Store, ledger: Ledger) -> list[Dispute]:
    """Rule on everything outstanding. Safe to run repeatedly."""
    return [
        adjudicate(store, ledger, Dispute.from_row(row)) for row in store.list_disputes(status=OPEN)
    ]
