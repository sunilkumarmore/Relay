"""Persistence layer for Relay.

Every read and write of Relay state goes through the ``Store`` protocol. Only
this module may import ``supabase`` — the rest of the codebase depends on the
protocol, which is what makes the system testable without a network.

Three implementations ship today:

``SupabaseStore``
    Production. Backed by the ``relay_*`` tables in ``setup/supabase_setup.sql``.
``MemoryStore``
    In-process dictionaries. Used by unit tests.
``FileStore``
    ``MemoryStore`` persisted to a JSON file under an advisory lock, so several
    processes can share one store. Used by the subprocess eviction tests.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from relay import config

try:  # POSIX only; FileStore is test infrastructure and Windows CI is not a target.
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]


class RelayStoreError(RuntimeError):
    pass


# Kept under the old name so existing error handling keeps working.
RelayCheckpointError = RelayStoreError

CONNECT_HELP = (
    "ERROR: Cannot connect to Supabase\n"
    "Check SUPABASE_URL and SUPABASE_KEY in .env\n"
    "Visit supabase.com - project may be paused"
)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


@runtime_checkable
class Store(Protocol):
    """Everything Relay persists. Implementations must be safe to call from
    multiple threads within one process."""

    def verify_connection(self) -> None: ...

    # -- sessions ---------------------------------------------------------
    def get_session(self, session_id: str) -> dict[str, Any] | None: ...

    def upsert_session(
        self,
        session_id: str,
        worker_id: str,
        task_goal: str,
        steps_total: int,
        steps_completed: int,
        status: str,
        current_machine: str,
        inference_node: str,
        final_report: str | None = None,
    ) -> None: ...

    def update_session(self, session_id: str, **updates: Any) -> None: ...

    def list_sessions(self) -> list[dict[str, Any]]: ...

    # -- worker state -----------------------------------------------------
    def get_state(self, session_id: str) -> dict[str, Any] | None: ...

    def upsert_state(
        self,
        session_id: str,
        worker_id: str,
        next_step_number: int,
        next_problem: str,
        machine_id: str,
        inference_node: str,
        status: str,
        provider_node_id: str = "",
    ) -> None: ...

    def list_worker_state(self) -> list[dict[str, Any]]: ...

    # -- checkpoints ------------------------------------------------------
    def get_checkpoints(self, session_id: str) -> list[dict[str, Any]]: ...

    def insert_checkpoint(
        self,
        session_id: str,
        worker_id: str,
        step_number: int,
        problem: str,
        solution: str,
        reasoning: str,
        machine_id: str,
        inference_node: str,
        inference_latency_ms: int,
        tokens_used: int,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> bool:
        """Insert one step. Returns False when the step already exists.

        The uniqueness of ``(session_id, step_number)`` is what makes a step
        replay safe: a worker that dies after inference but before recording can
        redo the step without producing a second row.
        """
        ...

    # -- logs -------------------------------------------------------------
    def insert_inference_log(
        self,
        worker_id: str,
        session_id: str,
        inference_node: str,
        latency_ms: int,
        tokens_used: int,
        success: bool,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> None: ...

    def insert_migration_event(
        self,
        session_id: str,
        worker_id: str,
        event: str,
        from_machine: str | None,
        to_machine: str | None,
        step_at_event: int,
    ) -> None: ...

    def list_migration_events(self, limit: int = 20) -> list[dict[str, Any]]: ...

    # -- registry (durable node registrations and request log) ------------
    def upsert_node(
        self,
        worker_id: str,
        node_id: str,
        session_id: str,
        machine_id: str,
        inference_node: str,
        steps_completed: int,
        last_heartbeat: str,
        registered_at: str | None = None,
    ) -> None: ...

    def get_node(self, worker_id: str) -> dict[str, Any] | None: ...

    def list_nodes(self, inference_node: str | None = None) -> list[dict[str, Any]]: ...

    def delete_node(self, worker_id: str) -> None: ...

    def insert_registry_request(
        self,
        worker_id: str,
        node_id: str,
        session_id: str,
        inference_node: str,
        latency_ms: int,
        success: bool,
    ) -> None: ...

    def list_registry_requests(
        self, inference_node: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]: ...

    # -- directory (offers and provider events) ---------------------------
    def upsert_offer(self, offer: dict[str, Any]) -> None: ...

    def list_offers(self, model: str | None = None) -> list[dict[str, Any]]: ...

    def delete_offer(self, offer_id: str) -> None: ...

    def delete_offers_for(self, provider_node_id: str) -> None: ...

    def insert_provider_event(
        self,
        provider_node_id: str,
        kind: str,
        model: str = "",
        detail: dict[str, Any] | None = None,
    ) -> None: ...

    def list_provider_events(
        self, provider_node_id: str | None = None, kind: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]: ...

    # -- provider health (what consumers actually observed) ---------------
    def insert_provider_health(
        self,
        observer_node_id: str,
        provider_node_id: str,
        ok: bool,
        latency_ms: int | None = None,
        error: str = "",
    ) -> None: ...

    def list_provider_health(
        self, provider_node_id: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]: ...

    # -- receipts and ledger ----------------------------------------------
    def upsert_receipt(self, receipt: dict[str, Any]) -> None: ...

    def get_receipt(self, receipt_id: str) -> dict[str, Any] | None: ...

    def list_receipts(
        self,
        job_id: str | None = None,
        provider_node_id: str | None = None,
        consumer_node_id: str | None = None,
        status: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]: ...

    def upsert_account(self, node_id: str, balance_cached: float, stake: float = 0.0) -> None: ...

    def get_account(self, node_id: str) -> dict[str, Any] | None: ...

    def list_accounts(self) -> list[dict[str, Any]]: ...

    def insert_ledger_tx(self, entries: list[dict[str, Any]], idempotency_key: str) -> bool:
        """Append one transaction atomically. Returns False when this exact
        transaction was already posted — which is what stops a retried
        settlement paying twice."""
        ...

    def list_ledger_entries(
        self,
        account: str | None = None,
        tx_id: str | None = None,
        kind: str | None = None,
        ref_receipt_id: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]: ...

    # -- reputation and disputes -------------------------------------------
    def upsert_reputation(
        self, node_id: str, role: str, score: float, components: dict[str, Any], computed_at: str
    ) -> None: ...

    def get_reputation(self, node_id: str, role: str = "provider") -> dict[str, Any] | None: ...

    def list_reputation(self, role: str | None = None) -> list[dict[str, Any]]: ...

    def upsert_dispute(self, dispute: dict[str, Any]) -> None: ...

    def get_dispute(self, dispute_id: str) -> dict[str, Any] | None: ...

    def list_disputes(
        self, receipt_id: str | None = None, status: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]: ...

    def reset_session(self, session_id: str) -> None: ...


class SupabaseStore:
    """Store backed by the ``relay_*`` tables in a Supabase project."""

    def __init__(self, supabase_url: str, supabase_key: str) -> None:
        if not supabase_url or not supabase_key:
            raise RelayStoreError(CONNECT_HELP)
        from supabase import create_client

        self.client = create_client(supabase_url, supabase_key)

    @classmethod
    def from_env(cls, env_path: str | None = None) -> SupabaseStore:
        config.load_env(env_path)
        return cls(config.get("SUPABASE_URL"), config.get("SUPABASE_KEY"))

    def verify_connection(self) -> None:
        try:
            self.client.table("relay_sessions").select("id").limit(1).execute()
        except Exception as exc:
            raise RelayStoreError(CONNECT_HELP) from exc

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        rows = (
            self.client.table("relay_sessions")
            .select("*")
            .eq("session_id", session_id)
            .limit(1)
            .execute()
            .data
        )
        return rows[0] if rows else None

    def upsert_session(
        self,
        session_id: str,
        worker_id: str,
        task_goal: str,
        steps_total: int,
        steps_completed: int,
        status: str,
        current_machine: str,
        inference_node: str,
        final_report: str | None = None,
    ) -> None:
        self.client.table("relay_sessions").upsert(
            {
                "session_id": session_id,
                "worker_id": worker_id,
                "task_goal": task_goal,
                "steps_total": steps_total,
                "steps_completed": steps_completed,
                "status": status,
                "final_report": final_report,
                "current_machine": current_machine,
                "inference_node": inference_node,
                "updated_at": now_iso(),
            },
            on_conflict="session_id",
        ).execute()

    def update_session(self, session_id: str, **updates: Any) -> None:
        updates["updated_at"] = now_iso()
        self.client.table("relay_sessions").update(updates).eq("session_id", session_id).execute()

    def list_sessions(self) -> list[dict[str, Any]]:
        data = (
            self.client.table("relay_sessions").select("*").order("updated_at", desc=True).execute().data
        )
        return list(data or [])

    def get_state(self, session_id: str) -> dict[str, Any] | None:
        rows = (
            self.client.table("relay_worker_state")
            .select("*")
            .eq("session_id", session_id)
            .limit(1)
            .execute()
            .data
        )
        return rows[0] if rows else None

    def upsert_state(
        self,
        session_id: str,
        worker_id: str,
        next_step_number: int,
        next_problem: str,
        machine_id: str,
        inference_node: str,
        status: str,
        provider_node_id: str = "",
    ) -> None:
        self.client.table("relay_worker_state").upsert(
            {
                "session_id": session_id,
                "worker_id": worker_id,
                "next_step_number": next_step_number,
                "next_problem": next_problem,
                "machine_id": machine_id,
                "inference_node": inference_node,
                "provider_node_id": provider_node_id,
                "status": status,
                "updated_at": now_iso(),
            },
            on_conflict="session_id",
        ).execute()

    def list_worker_state(self) -> list[dict[str, Any]]:
        data = (
            self.client.table("relay_worker_state")
            .select("*")
            .order("updated_at", desc=True)
            .execute()
            .data
        )
        return list(data or [])

    def get_checkpoints(self, session_id: str) -> list[dict[str, Any]]:
        data = (
            self.client.table("relay_checkpoints")
            .select("*")
            .eq("session_id", session_id)
            .order("step_number")
            .execute()
            .data
        )
        return list(data or [])

    def insert_checkpoint(
        self,
        session_id: str,
        worker_id: str,
        step_number: int,
        problem: str,
        solution: str,
        reasoning: str,
        machine_id: str,
        inference_node: str,
        inference_latency_ms: int,
        tokens_used: int,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> bool:
        # Lean on the DB unique constraint on (session_id, step_number) so this is
        # atomic — two workers racing on the same step cannot both succeed.
        result = (
            self.client.table("relay_checkpoints")
            .upsert(
                {
                    "session_id": session_id,
                    "worker_id": worker_id,
                    "step_number": step_number,
                    "problem": problem,
                    "solution": solution,
                    "reasoning": reasoning,
                    "machine_id": machine_id,
                    "inference_node": inference_node,
                    "inference_latency_ms": inference_latency_ms,
                    "tokens_used": tokens_used,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                },
                on_conflict="session_id,step_number",
                ignore_duplicates=True,
            )
            .execute()
        )
        return bool(result.data)

    def insert_inference_log(
        self,
        worker_id: str,
        session_id: str,
        inference_node: str,
        latency_ms: int,
        tokens_used: int,
        success: bool,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> None:
        self.client.table("relay_inference_log").insert(
            {
                "worker_id": worker_id,
                "session_id": session_id,
                "inference_node": inference_node,
                "latency_ms": latency_ms,
                "tokens_used": tokens_used,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "success": success,
            }
        ).execute()

    def insert_migration_event(
        self,
        session_id: str,
        worker_id: str,
        event: str,
        from_machine: str | None,
        to_machine: str | None,
        step_at_event: int,
    ) -> None:
        self.client.table("relay_migration_log").insert(
            {
                "session_id": session_id,
                "worker_id": worker_id,
                "event": event,
                "from_machine": from_machine,
                "to_machine": to_machine,
                "step_at_event": step_at_event,
            }
        ).execute()

    def list_migration_events(self, limit: int = 20) -> list[dict[str, Any]]:
        data = (
            self.client.table("relay_migration_log")
            .select("*")
            .order("occurred_at", desc=True)
            .limit(limit)
            .execute()
            .data
        )
        return list(data or [])

    def upsert_node(
        self,
        worker_id: str,
        node_id: str,
        session_id: str,
        machine_id: str,
        inference_node: str,
        steps_completed: int,
        last_heartbeat: str,
        registered_at: str | None = None,
    ) -> None:
        record = {
            "worker_id": worker_id,
            "node_id": node_id,
            "session_id": session_id,
            "machine_id": machine_id,
            "inference_node": inference_node,
            "steps_completed": steps_completed,
            "last_heartbeat": last_heartbeat,
        }
        if registered_at is not None:
            record["registered_at"] = registered_at
        self.client.table("relay_nodes").upsert(record, on_conflict="worker_id").execute()

    def get_node(self, worker_id: str) -> dict[str, Any] | None:
        rows = (
            self.client.table("relay_nodes")
            .select("*")
            .eq("worker_id", worker_id)
            .limit(1)
            .execute()
            .data
        )
        return rows[0] if rows else None

    def list_nodes(self, inference_node: str | None = None) -> list[dict[str, Any]]:
        query = self.client.table("relay_nodes").select("*")
        if inference_node is not None:
            query = query.eq("inference_node", inference_node)
        return list(query.execute().data or [])

    def delete_node(self, worker_id: str) -> None:
        self.client.table("relay_nodes").delete().eq("worker_id", worker_id).execute()

    def insert_registry_request(
        self,
        worker_id: str,
        node_id: str,
        session_id: str,
        inference_node: str,
        latency_ms: int,
        success: bool,
    ) -> None:
        self.client.table("relay_registry_requests").insert(
            {
                "worker_id": worker_id,
                "node_id": node_id,
                "session_id": session_id,
                "inference_node": inference_node,
                "latency_ms": latency_ms,
                "success": success,
            }
        ).execute()

    def list_registry_requests(
        self, inference_node: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        query = self.client.table("relay_registry_requests").select("*")
        if inference_node is not None:
            query = query.eq("inference_node", inference_node)
        data = query.order("requested_at", desc=True).limit(limit).execute().data
        return list(reversed(list(data or [])))

    def upsert_offer(self, offer: dict[str, Any]) -> None:
        self.client.table("relay_offers").upsert(offer, on_conflict="offer_id").execute()

    def list_offers(self, model: str | None = None) -> list[dict[str, Any]]:
        query = self.client.table("relay_offers").select("*")
        if model is not None:
            query = query.eq("model", model)
        return list(query.execute().data or [])

    def delete_offer(self, offer_id: str) -> None:
        self.client.table("relay_offers").delete().eq("offer_id", offer_id).execute()

    def delete_offers_for(self, provider_node_id: str) -> None:
        self.client.table("relay_offers").delete().eq(
            "provider_node_id", provider_node_id
        ).execute()

    def insert_provider_event(
        self,
        provider_node_id: str,
        kind: str,
        model: str = "",
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.client.table("relay_provider_events").insert(
            {
                "provider_node_id": provider_node_id,
                "kind": kind,
                "model": model,
                "detail": detail or {},
            }
        ).execute()

    def list_provider_events(
        self, provider_node_id: str | None = None, kind: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        query = self.client.table("relay_provider_events").select("*")
        if provider_node_id is not None:
            query = query.eq("provider_node_id", provider_node_id)
        if kind is not None:
            query = query.eq("kind", kind)
        data = query.order("occurred_at", desc=True).limit(limit).execute().data
        return list(reversed(list(data or [])))

    def insert_provider_health(
        self,
        observer_node_id: str,
        provider_node_id: str,
        ok: bool,
        latency_ms: int | None = None,
        error: str = "",
    ) -> None:
        self.client.table("relay_provider_health").insert(
            {
                "observer_node_id": observer_node_id,
                "provider_node_id": provider_node_id,
                "ok": ok,
                "latency_ms": latency_ms,
                "error": error,
            }
        ).execute()

    def list_provider_health(
        self, provider_node_id: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        query = self.client.table("relay_provider_health").select("*")
        if provider_node_id is not None:
            query = query.eq("provider_node_id", provider_node_id)
        data = query.order("observed_at", desc=True).limit(limit).execute().data
        return list(data or [])

    def upsert_receipt(self, receipt: dict[str, Any]) -> None:
        self.client.table("relay_receipts").upsert(receipt, on_conflict="receipt_id").execute()

    def get_receipt(self, receipt_id: str) -> dict[str, Any] | None:
        rows = (
            self.client.table("relay_receipts")
            .select("*")
            .eq("receipt_id", receipt_id)
            .limit(1)
            .execute()
            .data
        )
        return rows[0] if rows else None

    def list_receipts(
        self,
        job_id: str | None = None,
        provider_node_id: str | None = None,
        consumer_node_id: str | None = None,
        status: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        query = self.client.table("relay_receipts").select("*")
        for column, value in (
            ("job_id", job_id),
            ("provider_node_id", provider_node_id),
            ("consumer_node_id", consumer_node_id),
            ("status", status),
        ):
            if value is not None:
                query = query.eq(column, value)
        return list(query.limit(limit).execute().data or [])

    def upsert_account(self, node_id: str, balance_cached: float, stake: float = 0.0) -> None:
        self.client.table("relay_accounts").upsert(
            {"node_id": node_id, "balance_cached": balance_cached, "stake": stake},
            on_conflict="node_id",
        ).execute()

    def get_account(self, node_id: str) -> dict[str, Any] | None:
        rows = (
            self.client.table("relay_accounts")
            .select("*")
            .eq("node_id", node_id)
            .limit(1)
            .execute()
            .data
        )
        return rows[0] if rows else None

    def list_accounts(self) -> list[dict[str, Any]]:
        return list(self.client.table("relay_accounts").select("*").execute().data or [])

    def insert_ledger_tx(self, entries: list[dict[str, Any]], idempotency_key: str) -> bool:
        if not entries:
            return False
        tx_id = entries[0]["tx_id"]
        # The unique index on idempotency_key is the guard: if this transaction
        # was already posted, the claim fails and no entries are written.
        claimed = (
            self.client.table("relay_ledger_tx")
            .upsert(
                {"tx_id": tx_id, "idempotency_key": idempotency_key, "kind": entries[0]["kind"]},
                on_conflict="idempotency_key",
                ignore_duplicates=True,
            )
            .execute()
        )
        if not claimed.data:
            return False
        self.client.table("relay_ledger_entries").insert(entries).execute()
        return True

    def list_ledger_entries(
        self,
        account: str | None = None,
        tx_id: str | None = None,
        kind: str | None = None,
        ref_receipt_id: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        query = self.client.table("relay_ledger_entries").select("*")
        for column, value in (
            ("account", account),
            ("tx_id", tx_id),
            ("kind", kind),
            ("ref_receipt_id", ref_receipt_id),
        ):
            if value is not None:
                query = query.eq(column, value)
        return list(query.limit(limit).execute().data or [])

    def upsert_reputation(
        self, node_id: str, role: str, score: float, components: dict[str, Any], computed_at: str
    ) -> None:
        self.client.table("relay_reputation").upsert(
            {
                "node_id": node_id,
                "role": role,
                "score": score,
                "components": components,
                "computed_at": computed_at,
            },
            on_conflict="node_id,role",
        ).execute()

    def get_reputation(self, node_id: str, role: str = "provider") -> dict[str, Any] | None:
        rows = (
            self.client.table("relay_reputation")
            .select("*")
            .eq("node_id", node_id)
            .eq("role", role)
            .limit(1)
            .execute()
            .data
        )
        return rows[0] if rows else None

    def list_reputation(self, role: str | None = None) -> list[dict[str, Any]]:
        query = self.client.table("relay_reputation").select("*")
        if role is not None:
            query = query.eq("role", role)
        return list(query.execute().data or [])

    def upsert_dispute(self, dispute: dict[str, Any]) -> None:
        self.client.table("relay_disputes").upsert(dispute, on_conflict="dispute_id").execute()

    def get_dispute(self, dispute_id: str) -> dict[str, Any] | None:
        rows = (
            self.client.table("relay_disputes")
            .select("*")
            .eq("dispute_id", dispute_id)
            .limit(1)
            .execute()
            .data
        )
        return rows[0] if rows else None

    def list_disputes(
        self, receipt_id: str | None = None, status: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        query = self.client.table("relay_disputes").select("*")
        if receipt_id is not None:
            query = query.eq("receipt_id", receipt_id)
        if status is not None:
            query = query.eq("status", status)
        return list(query.limit(limit).execute().data or [])

    def reset_session(self, session_id: str) -> None:
        for table in (
            "relay_worker_state",
            "relay_checkpoints",
            "relay_migration_log",
            "relay_inference_log",
            "relay_nodes",
            "relay_registry_requests",
            "relay_sessions",
        ):
            self.client.table(table).delete().eq("session_id", session_id).execute()


TABLES = (
    "sessions",
    "worker_state",
    "checkpoints",
    "inference_log",
    "migration_log",
    "nodes",
    "registry_requests",
    "offers",
    "provider_events",
    "provider_health",
    "receipts",
    "accounts",
    "ledger_tx",
    "ledger_entries",
    "reputation",
    "disputes",
)


def _empty_tables() -> dict[str, list[dict[str, Any]]]:
    return {name: [] for name in TABLES}


class MemoryStore:
    """In-process store. Mirrors the Supabase constraints that Relay relies on."""

    def __init__(self) -> None:
        self.tables = _empty_tables()
        self._lock = threading.RLock()

    # Hooks that FileStore overrides to persist between mutations.
    def _view(self) -> dict[str, list[dict[str, Any]]]:
        """Tables to read from. FileStore returns a fresh private copy, so a
        reader never disturbs the dict a concurrent writer is mutating."""
        return self.tables

    def _begin(self) -> None:
        return None

    def _commit(self) -> None:
        return None

    @contextlib.contextmanager
    def _txn(self):
        # Serializes writers inside this process. FileStore adds a file lock on
        # top for writers in other processes.
        with self._lock:
            self._begin()
            try:
                yield
            finally:
                self._commit()

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return json.loads(json.dumps(self._view()))

    def verify_connection(self) -> None:
        return None

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        tables = self._view()
        for row in tables["sessions"]:
            if row["session_id"] == session_id:
                return dict(row)
        return None

    def upsert_session(
        self,
        session_id: str,
        worker_id: str,
        task_goal: str,
        steps_total: int,
        steps_completed: int,
        status: str,
        current_machine: str,
        inference_node: str,
        final_report: str | None = None,
    ) -> None:
        record = {
            "session_id": session_id,
            "worker_id": worker_id,
            "task_goal": task_goal,
            "steps_total": steps_total,
            "steps_completed": steps_completed,
            "status": status,
            "final_report": final_report,
            "current_machine": current_machine,
            "inference_node": inference_node,
            "updated_at": now_iso(),
        }
        with self._txn():
            for row in self.tables["sessions"]:
                if row["session_id"] == session_id:
                    row.update(record)
                    return
            record.setdefault("created_at", now_iso())
            self.tables["sessions"].append(record)

    def update_session(self, session_id: str, **updates: Any) -> None:
        updates["updated_at"] = now_iso()
        with self._txn():
            for row in self.tables["sessions"]:
                if row["session_id"] == session_id:
                    row.update(updates)
                    return

    def list_sessions(self) -> list[dict[str, Any]]:
        tables = self._view()
        return sorted(
            (dict(r) for r in tables["sessions"]),
            key=lambda r: str(r.get("updated_at", "")),
            reverse=True,
        )

    def get_state(self, session_id: str) -> dict[str, Any] | None:
        tables = self._view()
        for row in tables["worker_state"]:
            if row["session_id"] == session_id:
                return dict(row)
        return None

    def upsert_state(
        self,
        session_id: str,
        worker_id: str,
        next_step_number: int,
        next_problem: str,
        machine_id: str,
        inference_node: str,
        status: str,
        provider_node_id: str = "",
    ) -> None:
        record = {
            "session_id": session_id,
            "worker_id": worker_id,
            "next_step_number": next_step_number,
            "next_problem": next_problem,
            "machine_id": machine_id,
            "inference_node": inference_node,
            "provider_node_id": provider_node_id,
            "status": status,
            "updated_at": now_iso(),
        }
        with self._txn():
            for row in self.tables["worker_state"]:
                if row["session_id"] == session_id:
                    row.update(record)
                    return
            self.tables["worker_state"].append(record)

    def list_worker_state(self) -> list[dict[str, Any]]:
        tables = self._view()
        return sorted(
            (dict(r) for r in tables["worker_state"]),
            key=lambda r: str(r.get("updated_at", "")),
            reverse=True,
        )

    def get_checkpoints(self, session_id: str) -> list[dict[str, Any]]:
        tables = self._view()
        rows = [dict(r) for r in tables["checkpoints"] if r["session_id"] == session_id]
        return sorted(rows, key=lambda r: int(r["step_number"]))

    def insert_checkpoint(
        self,
        session_id: str,
        worker_id: str,
        step_number: int,
        problem: str,
        solution: str,
        reasoning: str,
        machine_id: str,
        inference_node: str,
        inference_latency_ms: int,
        tokens_used: int,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> bool:
        with self._txn():
            # Stands in for UNIQUE(session_id, step_number).
            for row in self.tables["checkpoints"]:
                if row["session_id"] == session_id and int(row["step_number"]) == step_number:
                    return False
            self.tables["checkpoints"].append(
                {
                    "session_id": session_id,
                    "worker_id": worker_id,
                    "step_number": step_number,
                    "problem": problem,
                    "solution": solution,
                    "reasoning": reasoning,
                    "machine_id": machine_id,
                    "inference_node": inference_node,
                    "inference_latency_ms": inference_latency_ms,
                    "tokens_used": tokens_used,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "completed_at": now_iso(),
                }
            )
            return True

    def insert_inference_log(
        self,
        worker_id: str,
        session_id: str,
        inference_node: str,
        latency_ms: int,
        tokens_used: int,
        success: bool,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> None:
        with self._txn():
            self.tables["inference_log"].append(
                {
                    "worker_id": worker_id,
                    "session_id": session_id,
                    "inference_node": inference_node,
                    "latency_ms": latency_ms,
                    "tokens_used": tokens_used,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "success": success,
                    "requested_at": now_iso(),
                }
            )

    def insert_migration_event(
        self,
        session_id: str,
        worker_id: str,
        event: str,
        from_machine: str | None,
        to_machine: str | None,
        step_at_event: int,
    ) -> None:
        with self._txn():
            self.tables["migration_log"].append(
                {
                    "session_id": session_id,
                    "worker_id": worker_id,
                    "event": event,
                    "from_machine": from_machine,
                    "to_machine": to_machine,
                    "step_at_event": step_at_event,
                    "occurred_at": now_iso(),
                }
            )

    def list_migration_events(self, limit: int = 20) -> list[dict[str, Any]]:
        tables = self._view()
        rows = sorted(
            (dict(r) for r in tables["migration_log"]),
            key=lambda r: str(r.get("occurred_at", "")),
            reverse=True,
        )
        return rows[:limit]

    def upsert_node(
        self,
        worker_id: str,
        node_id: str,
        session_id: str,
        machine_id: str,
        inference_node: str,
        steps_completed: int,
        last_heartbeat: str,
        registered_at: str | None = None,
    ) -> None:
        record = {
            "worker_id": worker_id,
            "node_id": node_id,
            "session_id": session_id,
            "machine_id": machine_id,
            "inference_node": inference_node,
            "steps_completed": steps_completed,
            "last_heartbeat": last_heartbeat,
        }
        with self._txn():
            for row in self.tables["nodes"]:
                if row["worker_id"] == worker_id:
                    row.update(record)
                    if registered_at is not None:
                        row.setdefault("registered_at", registered_at)
                    return
            record["registered_at"] = registered_at or last_heartbeat
            self.tables["nodes"].append(record)

    def get_node(self, worker_id: str) -> dict[str, Any] | None:
        tables = self._view()
        for row in tables["nodes"]:
            if row["worker_id"] == worker_id:
                return dict(row)
        return None

    def list_nodes(self, inference_node: str | None = None) -> list[dict[str, Any]]:
        tables = self._view()
        return [
            dict(r)
            for r in tables["nodes"]
            if inference_node is None or r.get("inference_node") == inference_node
        ]

    def delete_node(self, worker_id: str) -> None:
        with self._txn():
            self.tables["nodes"] = [r for r in self.tables["nodes"] if r["worker_id"] != worker_id]

    def insert_registry_request(
        self,
        worker_id: str,
        node_id: str,
        session_id: str,
        inference_node: str,
        latency_ms: int,
        success: bool,
    ) -> None:
        with self._txn():
            self.tables["registry_requests"].append(
                {
                    "worker_id": worker_id,
                    "node_id": node_id,
                    "session_id": session_id,
                    "inference_node": inference_node,
                    "latency_ms": latency_ms,
                    "success": success,
                    "requested_at": now_iso(),
                }
            )

    def list_registry_requests(
        self, inference_node: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        tables = self._view()
        rows = [
            dict(r)
            for r in tables["registry_requests"]
            if inference_node is None or r.get("inference_node") == inference_node
        ]
        rows.sort(key=lambda r: str(r.get("requested_at", "")))
        return rows[-limit:]

    def upsert_offer(self, offer: dict[str, Any]) -> None:
        with self._txn():
            for row in self.tables["offers"]:
                if row["offer_id"] == offer["offer_id"]:
                    row.update(offer)
                    return
            self.tables["offers"].append(dict(offer))

    def list_offers(self, model: str | None = None) -> list[dict[str, Any]]:
        tables = self._view()
        return [dict(r) for r in tables["offers"] if model is None or r.get("model") == model]

    def delete_offer(self, offer_id: str) -> None:
        with self._txn():
            self.tables["offers"] = [r for r in self.tables["offers"] if r["offer_id"] != offer_id]

    def delete_offers_for(self, provider_node_id: str) -> None:
        with self._txn():
            self.tables["offers"] = [
                r for r in self.tables["offers"] if r.get("provider_node_id") != provider_node_id
            ]

    def insert_provider_event(
        self,
        provider_node_id: str,
        kind: str,
        model: str = "",
        detail: dict[str, Any] | None = None,
    ) -> None:
        with self._txn():
            self.tables["provider_events"].append(
                {
                    "provider_node_id": provider_node_id,
                    "kind": kind,
                    "model": model,
                    "detail": detail or {},
                    "occurred_at": now_iso(),
                }
            )

    def list_provider_events(
        self, provider_node_id: str | None = None, kind: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        tables = self._view()
        rows = [
            dict(r)
            for r in tables["provider_events"]
            if (provider_node_id is None or r.get("provider_node_id") == provider_node_id)
            and (kind is None or r.get("kind") == kind)
        ]
        rows.sort(key=lambda r: str(r.get("occurred_at", "")))
        return rows[-limit:]

    def insert_provider_health(
        self,
        observer_node_id: str,
        provider_node_id: str,
        ok: bool,
        latency_ms: int | None = None,
        error: str = "",
    ) -> None:
        with self._txn():
            self.tables["provider_health"].append(
                {
                    "observer_node_id": observer_node_id,
                    "provider_node_id": provider_node_id,
                    "ok": ok,
                    "latency_ms": latency_ms,
                    "error": error,
                    "observed_at": now_iso(),
                }
            )

    def list_provider_health(
        self, provider_node_id: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        tables = self._view()
        rows = [
            dict(r)
            for r in tables["provider_health"]
            if provider_node_id is None or r.get("provider_node_id") == provider_node_id
        ]
        rows.sort(key=lambda r: str(r.get("observed_at", "")), reverse=True)
        return rows[:limit]

    def upsert_receipt(self, receipt: dict[str, Any]) -> None:
        with self._txn():
            for row in self.tables["receipts"]:
                if row["receipt_id"] == receipt["receipt_id"]:
                    row.update(receipt)
                    return
            self.tables["receipts"].append(dict(receipt))

    def get_receipt(self, receipt_id: str) -> dict[str, Any] | None:
        tables = self._view()
        for row in tables["receipts"]:
            if row["receipt_id"] == receipt_id:
                return dict(row)
        return None

    def list_receipts(
        self,
        job_id: str | None = None,
        provider_node_id: str | None = None,
        consumer_node_id: str | None = None,
        status: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        tables = self._view()
        wanted = {
            "job_id": job_id,
            "provider_node_id": provider_node_id,
            "consumer_node_id": consumer_node_id,
            "status": status,
        }
        rows = [
            dict(r)
            for r in tables["receipts"]
            if all(v is None or r.get(k) == v for k, v in wanted.items())
        ]
        rows.sort(key=lambda r: (str(r.get("job_id", "")), int(r.get("step_number") or 0)))
        return rows[:limit]

    def upsert_account(self, node_id: str, balance_cached: float, stake: float = 0.0) -> None:
        with self._txn():
            for row in self.tables["accounts"]:
                if row["node_id"] == node_id:
                    row["balance_cached"] = balance_cached
                    row["stake"] = stake
                    return
            self.tables["accounts"].append(
                {"node_id": node_id, "balance_cached": balance_cached, "stake": stake}
            )

    def get_account(self, node_id: str) -> dict[str, Any] | None:
        tables = self._view()
        for row in tables["accounts"]:
            if row["node_id"] == node_id:
                return dict(row)
        return None

    def list_accounts(self) -> list[dict[str, Any]]:
        tables = self._view()
        return [dict(r) for r in tables["accounts"]]

    def insert_ledger_tx(self, entries: list[dict[str, Any]], idempotency_key: str) -> bool:
        if not entries:
            return False
        with self._txn():
            for row in self.tables["ledger_tx"]:
                if row["idempotency_key"] == idempotency_key:
                    return False
            self.tables["ledger_tx"].append(
                {
                    "tx_id": entries[0]["tx_id"],
                    "idempotency_key": idempotency_key,
                    "kind": entries[0]["kind"],
                    "created_at": now_iso(),
                }
            )
            for entry in entries:
                self.tables["ledger_entries"].append({**entry, "created_at": now_iso()})
            return True

    def list_ledger_entries(
        self,
        account: str | None = None,
        tx_id: str | None = None,
        kind: str | None = None,
        ref_receipt_id: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        tables = self._view()
        wanted = {
            "account": account,
            "tx_id": tx_id,
            "kind": kind,
            "ref_receipt_id": ref_receipt_id,
        }
        rows = [
            dict(r)
            for r in tables["ledger_entries"]
            if all(v is None or r.get(k) == v for k, v in wanted.items())
        ]
        return rows[:limit]

    def upsert_reputation(
        self, node_id: str, role: str, score: float, components: dict[str, Any], computed_at: str
    ) -> None:
        record = {
            "node_id": node_id,
            "role": role,
            "score": score,
            "components": components,
            "computed_at": computed_at,
        }
        with self._txn():
            for row in self.tables["reputation"]:
                if row["node_id"] == node_id and row.get("role") == role:
                    row.update(record)
                    return
            self.tables["reputation"].append(record)

    def get_reputation(self, node_id: str, role: str = "provider") -> dict[str, Any] | None:
        tables = self._view()
        for row in tables["reputation"]:
            if row["node_id"] == node_id and row.get("role") == role:
                return dict(row)
        return None

    def list_reputation(self, role: str | None = None) -> list[dict[str, Any]]:
        tables = self._view()
        return [dict(r) for r in tables["reputation"] if role is None or r.get("role") == role]

    def upsert_dispute(self, dispute: dict[str, Any]) -> None:
        with self._txn():
            for row in self.tables["disputes"]:
                if row["dispute_id"] == dispute["dispute_id"]:
                    row.update(dispute)
                    return
            self.tables["disputes"].append(dict(dispute))

    def get_dispute(self, dispute_id: str) -> dict[str, Any] | None:
        tables = self._view()
        for row in tables["disputes"]:
            if row["dispute_id"] == dispute_id:
                return dict(row)
        return None

    def list_disputes(
        self, receipt_id: str | None = None, status: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        tables = self._view()
        rows = [
            dict(r)
            for r in tables["disputes"]
            if (receipt_id is None or r.get("receipt_id") == receipt_id)
            and (status is None or r.get("status") == status)
        ]
        rows.sort(key=lambda r: str(r.get("opened_at", "")))
        return rows[:limit]

    def reset_session(self, session_id: str) -> None:
        with self._txn():
            for name in self.tables:
                self.tables[name] = [
                    r
                    for r in self.tables[name]
                    if r.get("session_id") != session_id and r.get("job_id") != session_id
                ]


class FileStore(MemoryStore):
    """MemoryStore persisted to JSON so separate processes share one store.

    Each mutation re-reads the file under an exclusive advisory lock, applies the
    change, and writes it back atomically. Slow, but it lets a worker running as a
    subprocess be SIGKILLed while the parent inspects what it had committed.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._depth = 0
        if not self.path.exists():
            self._write(_empty_tables())

    def _read(self) -> dict[str, list[dict[str, Any]]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return _empty_tables()
        tables = _empty_tables()
        tables.update({k: v for k, v in data.items() if k in tables})
        return tables

    def _write(self, tables: dict[str, list[dict[str, Any]]]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(tables, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def _view(self) -> dict[str, list[dict[str, Any]]]:
        # Writes land via os.replace, so an unlocked read always sees a whole file.
        return self._read()

    def _begin(self) -> None:
        # Only the outermost transaction takes the lock and loads the file. A
        # nested one — an eviction handler firing mid-write, say — must join the
        # transaction already in progress, not re-read over its uncommitted work.
        if self._depth == 0:
            if fcntl is not None:
                self._fh = open(self._lock_path, "a+")  # noqa: SIM115 - released in _commit
                fcntl.flock(self._fh, fcntl.LOCK_EX)
            self.tables = self._read()
        self._depth += 1

    def _commit(self) -> None:
        self._depth -= 1
        if self._depth <= 0:
            self._depth = 0
            self._write(self.tables)
            if fcntl is not None:
                fcntl.flock(self._fh, fcntl.LOCK_UN)
                self._fh.close()

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return self._read()

    def close(self) -> None:
        """Release a lock left held by an interrupted transaction."""
        if self._depth and fcntl is not None:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            self._fh.close()
            self._depth = 0


def store_from_env(env_path: str | None = None, *, optional: bool = False) -> Store | None:
    """Build the store named by ``RELAY_STORE`` (default ``supabase``).

    ``optional=True`` returns ``None`` instead of raising when the store cannot be
    reached — the registry uses that so telemetry logging degrades rather than
    taking the node down.
    """
    config.load_env(env_path)
    kind = config.get("RELAY_STORE", "supabase").lower()
    try:
        if kind == "memory":
            return MemoryStore()
        if kind == "file":
            path = config.get("RELAY_STORE_PATH")
            if not path:
                raise RelayStoreError("RELAY_STORE=file requires RELAY_STORE_PATH")
            return FileStore(path)
        if kind != "supabase":
            raise RelayStoreError(f"Unknown RELAY_STORE: {kind}")
        store = SupabaseStore.from_env(env_path)
        store.verify_connection()
        return store
    except Exception:
        if optional:
            return None
        raise
