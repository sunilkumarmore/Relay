from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from typing import Any

from dotenv import load_dotenv
from supabase import Client, create_client


class RelayCheckpointError(RuntimeError):
    pass


@dataclass
class SessionSnapshot:
    session_id: str
    worker_id: str
    steps_completed: int
    steps_total: int
    status: str
    current_machine: str | None
    inference_node: str | None


class RelayCheckpointClient:
    def __init__(self, supabase_url: str, supabase_key: str) -> None:
        if not supabase_url or not supabase_key:
            raise RelayCheckpointError(
                "ERROR: Cannot connect to Supabase\n"
                "Check SUPABASE_URL and SUPABASE_KEY in .env\n"
                "Visit supabase.com - project may be paused"
            )
        self.client: Client = create_client(supabase_url, supabase_key)

    @classmethod
    def from_env(cls, env_file: str = ".env") -> "RelayCheckpointClient":
        load_dotenv(env_file)
        return cls(
            os.getenv("SUPABASE_URL", "").strip(),
            os.getenv("SUPABASE_KEY", "").strip(),
        )

    def verify_connection(self) -> None:
        try:
            self.client.table("relay_sessions").select("id").limit(1).execute()
        except Exception as exc:
            raise RelayCheckpointError(
                "ERROR: Cannot connect to Supabase\n"
                "Check SUPABASE_URL and SUPABASE_KEY in .env\n"
                "Visit supabase.com - project may be paused"
            ) from exc

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
                "updated_at": self._now(),
            },
            on_conflict="session_id",
        ).execute()

    def update_session(self, session_id: str, **updates: Any) -> None:
        updates["updated_at"] = self._now()
        self.client.table("relay_sessions").update(updates).eq("session_id", session_id).execute()

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
                "updated_at": self._now(),
            },
            on_conflict="session_id",
        ).execute()

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
        # Use the DB unique constraint on (session_id, step_number) to make this
        # atomic — no separate SELECT needed, so two workers racing on the same
        # step can't both succeed.
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

    def list_sessions(self) -> list[dict[str, Any]]:
        data = (
            self.client.table("relay_sessions")
            .select("*")
            .order("updated_at", desc=True)
            .execute()
            .data
        )
        return list(data or [])

    def list_worker_state(self) -> list[dict[str, Any]]:
        data = (
            self.client.table("relay_worker_state")
            .select("*")
            .order("updated_at", desc=True)
            .execute()
            .data
        )
        return list(data or [])

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

    def reset_session(self, session_id: str) -> None:
        self.client.table("relay_worker_state").delete().eq("session_id", session_id).execute()
        self.client.table("relay_checkpoints").delete().eq("session_id", session_id).execute()
        self.client.table("relay_migration_log").delete().eq("session_id", session_id).execute()
        self.client.table("relay_inference_log").delete().eq("session_id", session_id).execute()
        self.client.table("relay_sessions").delete().eq("session_id", session_id).execute()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
