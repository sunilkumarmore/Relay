"""Paying for a task, and what happens when two providers disagree.

The economic loop, closed. A consumer holds credits against a job before any
work is queued; a provider that completes a tile issues a signed receipt; the
consumer checks it against the offer it accepted and the task it signed, then
countersigns and settles out of the hold. Whatever is left when the job finishes
is released.

The check on the way in is the point. A receipt is only paid if the work it
describes is the work that was ordered — same task, same payload hash, same
work units, same price as the published offer — and every one of those facts was
signed by somebody before the money moved.

## Why this needs no over-claim dispute

An inference receipt bills for tokens the provider counted, so the consumer has
to keep its own estimate and argue when the two disagree. A task receipt bills
for `work_units`, which is fixed by the task's own payload and re-derived by the
queue before the work is handed out. There is nothing for a provider to inflate,
so the quantity is never in dispute — only the answer is.
"""

from __future__ import annotations

from typing import Any

from relay.identity import Identity
from relay.ledger import InsufficientFunds, Ledger
from relay.provider.offers import Offer
from relay.receipts import (
    CURRENT_TASK_SCHEMA,
    WORK_TASK,
    Receipt,
    hash_text,
    task_fingerprint,
)
from relay.store import Store
from relay.tasks import model as task_model
from relay.tasks import verify
from relay.tasks.model import Task, TaskResult

REASON_OUTPUT_DIVERGENCE = verify.REASON_OUTPUT_DIVERGENCE


class BillingError(RuntimeError):
    """A receipt that should not be paid."""


def free_offer(identity: Identity) -> Offer:
    """A zero-priced offer, for running the market with the money switched off.

    Useful for a demo and for a fleet of a person's own devices, where the
    ledger is bookkeeping rather than commerce. It is a real signed offer, so
    the same verification path runs either way.
    """
    from relay.provider.offers import build_offer
    from relay.tasks import kernels

    return build_offer(
        identity,
        endpoint_url="",
        task_types=kernels.known_types(),
        price_per_task=0.0,
        price_per_mega_unit=0.0,
    )


def issue_receipt(
    identity: Identity,
    *,
    task: Task,
    result: TaskResult,
    offer: Offer,
) -> Receipt:
    """The provider's bill for one completed tile."""
    if identity.node_id != result.provider_node_id:
        raise BillingError("a receipt must be issued by the provider that did the work")
    if result.status != task_model.RESULT_OK:
        raise BillingError("a failed task is not billable")

    receipt = Receipt(
        schema_version=CURRENT_TASK_SCHEMA,
        work_kind=WORK_TASK,
        work_units=task.work_units,
        task_id=task.task_id,
        job_id=task.job_id,
        step_number=0,
        consumer_node_id=task.consumer_node_id,
        provider_node_id=identity.node_id,
        offer_id=offer.offer_id,
        model="",
        request_hash=task_fingerprint(task.task_id, task.payload_hash, task.work_units),
        response_hash=hash_text(result.output_hash),
        tokens_in=0,
        tokens_out=0,
        latency_ms=result.duration_ms,
        price_in_per_1k=0.0,
        price_out_per_1k=0.0,
        amount_credits=offer.task_price(task.work_units),
        issued_at=result.finished_at,
    )
    return receipt.issued_by(identity)


def verify_receipt(
    receipt: Receipt,
    *,
    task: Task,
    result: TaskResult,
    offer: Offer,
) -> list[str]:
    """Everything the consumer checks before paying. Returns the reasons not to."""
    problems: list[str] = []

    if receipt.schema_version != CURRENT_TASK_SCHEMA or receipt.work_kind != WORK_TASK:
        problems.append("receipt is not a task receipt")
        return problems
    if not receipt.provider_signature_is_valid():
        problems.append("provider signature is not valid")
    if receipt.provider_node_id != result.provider_node_id:
        problems.append("receipt is from a different provider than the result")
    if receipt.consumer_node_id != task.consumer_node_id:
        problems.append("receipt names a different consumer")
    if receipt.task_id != task.task_id or receipt.job_id != task.job_id:
        problems.append("receipt is for a different task")
    if receipt.offer_id != offer.offer_id:
        problems.append("receipt cites a different offer")
    if receipt.provider_node_id != offer.provider_node_id:
        problems.append("receipt is from a different provider than the offer")

    if receipt.work_units != task.work_units:
        problems.append(
            f"receipt bills {receipt.work_units} work units, the task specifies "
            f"{task.work_units}"
        )
    expected_request = task_fingerprint(task.task_id, task.payload_hash, task.work_units)
    if receipt.request_hash != expected_request:
        problems.append("receipt does not describe the task we signed")
    if receipt.response_hash != hash_text(result.output_hash):
        problems.append("receipt does not describe the result we received")

    expected = offer.task_price(task.work_units)
    if abs(receipt.amount_credits - expected) > 1e-9:
        problems.append(
            f"amount {receipt.amount_credits} does not match {expected} at the offered price"
        )
    if task.max_price_credits and receipt.amount_credits > task.max_price_credits + 1e-9:
        problems.append(
            f"amount {receipt.amount_credits} exceeds the {task.max_price_credits} "
            f"this task committed to"
        )
    return problems


def accept_and_settle(
    store: Store,
    ledger: Ledger,
    identity: Identity,
    *,
    receipt: Receipt,
    task: Task,
    result: TaskResult,
    offer: Offer,
) -> Receipt:
    """Check the bill, countersign it, and move the credits.

    The countersignature is what turns "the provider says it is owed this" into
    an agreed debt, and it goes on before the transfer rather than after: an
    unacknowledged receipt is recorded but not owed.
    """
    problems = verify_receipt(receipt, task=task, result=result, offer=offer)
    if problems:
        disputed = receipt.disputed("; ".join(problems))
        store.upsert_receipt(disputed.to_row())
        raise BillingError(f"refusing to pay: {'; '.join(problems)}")

    acknowledged = receipt.acknowledged_by(identity)
    store.upsert_receipt(acknowledged.to_row())
    if acknowledged.amount_credits > 0:
        ledger.settle(
            acknowledged.consumer_node_id,
            acknowledged.provider_node_id,
            acknowledged.amount_credits,
            receipt_id=acknowledged.receipt_id,
            job_id=acknowledged.job_id,
        )
    return acknowledged


def hold_for_job(
    ledger: Ledger, *, consumer_node_id: str, job_id: str, credits: float
) -> str:
    """Commit a budget before any work is queued.

    A provider runs a task because somebody has money set aside for it. Holding
    first is what makes that true rather than hopeful.
    """
    if credits <= 0:
        return ""
    try:
        return ledger.hold(consumer_node_id, credits, job_id)
    except InsufficientFunds as exc:
        raise BillingError(f"cannot commit a budget for {job_id}: {exc}") from exc


def release_unspent(ledger: Ledger, *, consumer_node_id: str, job_id: str) -> float:
    """Give back whatever the job did not spend."""
    return ledger.release_remaining(consumer_node_id, job_id)


# --------------------------------------------------------------------------
# Divergence


def open_divergence_dispute(
    store: Store,
    *,
    divergence: verify.Divergence,
    opened_by: str,
) -> list[dict[str, Any]]:
    """Open a dispute against each provider in a conclusive minority.

    Only conclusive ones. A two-way disagreement says somebody is wrong without
    saying who, and opening a dispute on that would put an honest provider's
    stake at risk on a coin flip.
    """
    from relay.disputes import open_dispute

    if not divergence.is_conclusive:
        return []

    evidence = verify.evidence_for(store, divergence)
    opened: list[dict[str, Any]] = []
    for receipt_row in store.list_receipts(limit=100000):
        receipt = Receipt.from_row(receipt_row)
        if receipt.work_kind != WORK_TASK:
            continue
        if receipt.task_id != divergence.task_id:
            continue
        if receipt.provider_node_id not in divergence.minority:
            continue
        dispute = open_dispute(
            store,
            receipt_id=receipt.receipt_id,
            opened_by=opened_by,
            reason=REASON_OUTPUT_DIVERGENCE,
            evidence=evidence,
        )
        opened.append(dispute.to_row())
    return opened
