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
    ) -> None:
        self.client.table("relay_worker_state").upsert(
            {
                "session_id": session_id,
                "worker_id": worker_id,
                "next_step_number": next_step_number,
                "next_problem": next_problem,
                "machine_id": machine_id,
                "inference_node": inference_node,
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
    ) -> None:
        self.client.table("relay_inference_log").insert(
            {
                "worker_id": worker_id,
                "session_id": session_id,
                "inference_node": inference_node,
                "latency_ms": latency_ms,
                "tokens_used": tokens_used,
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
)


def _empty_tables() -> dict[str, list[dict[str, Any]]]:
    return {name: [] for name in TABLES}


class MemoryStore:
    """In-process store. Mirrors the Supabase constraints that Relay relies on."""

    def __init__(self) -> None:
        self.tables = _empty_tables()

    # Hooks that FileStore overrides to persist between mutations.
    def _refresh(self) -> None:
        """Pull in writes made by other processes. No-op in memory."""
        return None

    def _begin(self) -> None:
        return None

    def _commit(self) -> None:
        return None

    @contextlib.contextmanager
    def _txn(self):
        self._begin()
        try:
            yield
        finally:
            self._commit()

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        self._refresh()
        return json.loads(json.dumps(self.tables))

    def verify_connection(self) -> None:
        return None

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        self._refresh()
        for row in self.tables["sessions"]:
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
        self._refresh()
        return sorted(
            (dict(r) for r in self.tables["sessions"]),
            key=lambda r: str(r.get("updated_at", "")),
            reverse=True,
        )

    def get_state(self, session_id: str) -> dict[str, Any] | None:
        self._refresh()
        for row in self.tables["worker_state"]:
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
    ) -> None:
        record = {
            "session_id": session_id,
            "worker_id": worker_id,
            "next_step_number": next_step_number,
            "next_problem": next_problem,
            "machine_id": machine_id,
            "inference_node": inference_node,
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
        self._refresh()
        return sorted(
            (dict(r) for r in self.tables["worker_state"]),
            key=lambda r: str(r.get("updated_at", "")),
            reverse=True,
        )

    def get_checkpoints(self, session_id: str) -> list[dict[str, Any]]:
        self._refresh()
        rows = [dict(r) for r in self.tables["checkpoints"] if r["session_id"] == session_id]
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
    ) -> None:
        with self._txn():
            self.tables["inference_log"].append(
                {
                    "worker_id": worker_id,
                    "session_id": session_id,
                    "inference_node": inference_node,
                    "latency_ms": latency_ms,
                    "tokens_used": tokens_used,
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
        self._refresh()
        rows = sorted(
            (dict(r) for r in self.tables["migration_log"]),
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
        self._refresh()
        for row in self.tables["nodes"]:
            if row["worker_id"] == worker_id:
                return dict(row)
        return None

    def list_nodes(self, inference_node: str | None = None) -> list[dict[str, Any]]:
        self._refresh()
        return [
            dict(r)
            for r in self.tables["nodes"]
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
        self._refresh()
        rows = [
            dict(r)
            for r in self.tables["registry_requests"]
            if inference_node is None or r.get("inference_node") == inference_node
        ]
        rows.sort(key=lambda r: str(r.get("requested_at", "")))
        return rows[-limit:]

    def reset_session(self, session_id: str) -> None:
        with self._txn():
            for name in self.tables:
                self.tables[name] = [r for r in self.tables[name] if r.get("session_id") != session_id]


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

    def _refresh(self) -> None:
        self.tables = self._read()

    def _begin(self) -> None:
        if self._depth == 0 and fcntl is not None:
            self._fh = open(self._lock_path, "a+")  # noqa: SIM115 - released in _commit
            fcntl.flock(self._fh, fcntl.LOCK_EX)
        self._depth += 1
        self.tables = self._read()

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
