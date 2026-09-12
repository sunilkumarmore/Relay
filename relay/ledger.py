"""Credits, in double entry.

Every movement of value is a transaction made of entries that sum to zero. That
is not bookkeeping ceremony: it is the property that makes the whole ledger
checkable. If credits can be created by a bug in one code path, nobody can trust
a balance, and a marketplace without trustworthy balances is a toy.

Accounts are per node and split in two:

``<node>:available``
    Spendable.
``<node>:held``
    Committed to a running job, and not spendable until that job settles or
    releases it. Holding up front is what stops a consumer starting work it
    cannot pay for.

``world`` is the outside: deposits come from it, slashes go to it. Its balance
is the negative of everything inside the system, which is how the invariant
stays checkable without a special case.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from relay.store import Store

WORLD = "world"

DEPOSIT = "deposit"
HOLD = "hold"
RELEASE = "release"
SETTLE = "settle"
REFUND = "refund"
SLASH = "slash"
KINDS = (DEPOSIT, HOLD, RELEASE, SETTLE, REFUND, SLASH)


class LedgerError(RuntimeError):
    pass


class InsufficientFunds(LedgerError):
    pass


def available(node_id: str) -> str:
    return f"{node_id}:available"


def held(node_id: str, job_id: str = "") -> str:
    """Holds are per job. A node running two jobs must not have one job's
    completion release the other's committed budget."""
    return f"{node_id}:held:{job_id}" if job_id else f"{node_id}:held"


def node_of(account: str) -> str:
    # Node ids are hex, so the first colon always separates node from purpose.
    return account.split(":", 1)[0]


@dataclass
class Entry:
    account: str
    debit: float = 0.0
    credit: float = 0.0

    def to_row(self, tx_id: str, kind: str, ref_receipt_id: str, ref_job_id: str) -> dict[str, Any]:
        return {
            "tx_id": tx_id,
            "kind": kind,
            "account": self.account,
            "debit": round(self.debit, 6),
            "credit": round(self.credit, 6),
            "ref_receipt_id": ref_receipt_id,
            "ref_job_id": ref_job_id,
        }


class Ledger:
    def __init__(self, store: Store) -> None:
        self.store = store

    # -- reading ----------------------------------------------------------
    def account_balance(self, account: str) -> float:
        rows = self.store.list_ledger_entries(account=account)
        return round(sum(float(r.get("debit") or 0) - float(r.get("credit") or 0) for r in rows), 6)

    def balance(self, node_id: str) -> float:
        """What this node can spend right now."""
        return self.account_balance(available(node_id))

    def held_balance(self, node_id: str, job_id: str = "") -> float:
        if job_id:
            return self.account_balance(held(node_id, job_id))
        prefix = f"{node_id}:held"
        rows = [
            r for r in self.store.list_ledger_entries() if str(r.get("account", "")).startswith(prefix)
        ]
        return round(sum(float(r.get("debit") or 0) - float(r.get("credit") or 0) for r in rows), 6)

    def total(self) -> float:
        """Every account together. Must always be zero."""
        rows = self.store.list_ledger_entries()
        return round(sum(float(r.get("debit") or 0) - float(r.get("credit") or 0) for r in rows), 6)

    def check_invariant(self) -> None:
        total = self.total()
        if abs(total) > 1e-6:
            raise LedgerError(f"Ledger does not balance: {total}")

    def reconcile(self, node_id: str) -> float:
        """Recompute the cached balance from the entries, which are the truth."""
        balance = self.balance(node_id)
        self.store.upsert_account(node_id, balance)
        return balance

    def history(self, node_id: str, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.store.list_ledger_entries(account=available(node_id), limit=limit)
        rows += self.store.list_ledger_entries(account=held(node_id), limit=limit)
        return sorted(rows, key=lambda r: str(r.get("created_at", "")))[-limit:]

    # -- writing ----------------------------------------------------------
    def _post(
        self,
        kind: str,
        entries: list[Entry],
        *,
        ref_receipt_id: str = "",
        ref_job_id: str = "",
        idempotency_key: str = "",
    ) -> str:
        debits = round(sum(e.debit for e in entries), 6)
        credits = round(sum(e.credit for e in entries), 6)
        if abs(debits - credits) > 1e-9:
            # A bug here would mint or destroy credits. Refuse rather than record.
            raise LedgerError(f"Unbalanced transaction: debits {debits} != credits {credits}")
        if debits <= 0:
            raise LedgerError("A transaction must move a positive amount")

        tx_id = str(uuid.uuid4())
        rows = [e.to_row(tx_id, kind, ref_receipt_id, ref_job_id) for e in entries]
        written = self.store.insert_ledger_tx(rows, idempotency_key or tx_id)
        if not written:
            # Someone already posted this exact transaction. Not an error — it is
            # what stops a retried settlement paying twice.
            return ""

        for node in {node_of(e.account) for e in entries if e.account != WORLD}:
            self.reconcile(node)
        return tx_id

    def deposit(self, node_id: str, amount: float, *, idempotency_key: str = "") -> str:
        """Credits enter the system. In v1 this is a dev faucet; a payment rail
        would attach exactly here and change nothing downstream."""
        return self._post(
            DEPOSIT,
            [Entry(available(node_id), debit=amount), Entry(WORLD, credit=amount)],
            idempotency_key=idempotency_key,
        )

    def hold(self, node_id: str, amount: float, job_id: str) -> str:
        """Commit budget to a job before starting it."""
        if self.balance(node_id) + 1e-9 < amount:
            raise InsufficientFunds(
                f"{node_id[:12]} has {self.balance(node_id)} credits, needs {amount}"
            )
        return self._post(
            HOLD,
            [Entry(held(node_id, job_id), debit=amount), Entry(available(node_id), credit=amount)],
            ref_job_id=job_id,
            idempotency_key=f"hold:{node_id}:{job_id}",
        )

    def settle(
        self, consumer_node_id: str, provider_node_id: str, amount: float, *, receipt_id: str, job_id: str
    ) -> str:
        """Pay one acknowledged receipt out of the job's hold.

        A settlement cannot exceed what the job committed. Without this a job
        could spend past its budget, which would make the hold decorative.
        """
        outstanding = self.held_balance(consumer_node_id, job_id)
        if outstanding + 1e-9 < amount:
            raise InsufficientFunds(
                f"job {job_id} holds {outstanding} credits, receipt needs {amount}"
            )
        return self._post(
            SETTLE,
            [
                Entry(available(provider_node_id), debit=amount),
                Entry(held(consumer_node_id, job_id), credit=amount),
            ],
            ref_receipt_id=receipt_id,
            ref_job_id=job_id,
            # Keyed on the receipt: replaying a settlement cannot pay twice.
            idempotency_key=f"settle:{receipt_id}",
        )

    def release(self, node_id: str, amount: float, job_id: str) -> str:
        """Return an unspent hold to the consumer."""
        if amount <= 0:
            return ""
        return self._post(
            RELEASE,
            [Entry(available(node_id), debit=amount), Entry(held(node_id, job_id), credit=amount)],
            ref_job_id=job_id,
            idempotency_key=f"release:{node_id}:{job_id}:{round(amount, 6)}",
        )

    def release_remaining(self, node_id: str, job_id: str) -> float:
        """Release whatever is still held for this job — on completion, or when
        an evicted job was never resumed."""
        outstanding = self.held_balance(node_id, job_id)
        if outstanding <= 0:
            return 0.0
        self.release(node_id, outstanding, job_id)
        return outstanding

    def recoverable(self, node_id: str, amount: float) -> float:
        """How much of `amount` this node can actually cover."""
        return round(min(amount, max(self.balance(node_id), 0.0)), 6)

    def refund(
        self,
        consumer_node_id: str,
        provider_node_id: str,
        amount: float,
        *,
        receipt_id: str,
        job_id: str = "",
    ) -> str:
        """Undo a settlement an upheld dispute reversed."""
        if amount <= 0:
            return ""
        return self._post(
            REFUND,
            [
                Entry(available(consumer_node_id), debit=amount),
                Entry(available(provider_node_id), credit=amount),
            ],
            ref_receipt_id=receipt_id,
            ref_job_id=job_id,
            idempotency_key=f"refund:{receipt_id}",
        )

    def slash(self, node_id: str, amount: float, *, reason: str = "", ref_receipt_id: str = "") -> str:
        """Take credits out of a node's balance as a penalty.

        Capped at what the node actually holds. A negative balance would be a
        debt nothing in the system can collect, and would mean credits were paid
        out that never existed — which is precisely what stake is for.
        """
        amount = min(amount, max(self.balance(node_id), 0.0))
        if amount <= 0:
            return ""
        return self._post(
            SLASH,
            [Entry(WORLD, debit=amount), Entry(available(node_id), credit=amount)],
            ref_receipt_id=ref_receipt_id,
            idempotency_key=f"slash:{ref_receipt_id or uuid.uuid4()}:{reason}",
        )


# How long a hold survives with no progress before it is swept back. An evicted
# job that nobody resumes must not tie up a consumer's credits forever.
DEFAULT_HOLD_TTL_SECONDS = 3600


def sweep_expired_holds(
    store: Store, ledger: Ledger, *, ttl_seconds: int = DEFAULT_HOLD_TTL_SECONDS
) -> list[tuple[str, str, float]]:
    """Release holds for jobs that have gone quiet.

    A job is quiet when its session has not been touched for longer than the
    TTL — which is exactly the eviction-with-no-resume case. Returns what was
    released, so an operator can see it rather than infer it.
    """
    from datetime import timedelta

    from relay.provider.offers import now_utc, parse_time

    cutoff = now_utc() - timedelta(seconds=ttl_seconds)
    released: list[tuple[str, str, float]] = []

    entries = [r for r in store.list_ledger_entries(kind=HOLD) if r.get("ref_job_id")]
    for job_id in sorted({str(r["ref_job_id"]) for r in entries}):
        session = store.get_session(job_id)
        if session is not None:
            touched = parse_time(session.get("updated_at"))
            if session.get("status") != "completed" and (touched is None or touched >= cutoff):
                continue

        for row in entries:
            if row["ref_job_id"] != job_id:
                continue
            node = node_of(str(row["account"]))
            amount = ledger.release_remaining(node, job_id)
            if amount > 0:
                released.append((node, job_id, amount))
    return released
