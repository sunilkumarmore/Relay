"""Supabase checkpoint store utilities for the distributed agent demo."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from supabase import Client, create_client


class CheckpointClientError(RuntimeError):
    """Base error for checkpoint client failures."""


class ConfigurationError(CheckpointClientError):
    """Raised when required configuration is missing."""


class ConnectionError(CheckpointClientError):
    """Raised when Supabase is unreachable or misconfigured."""


@dataclass(frozen=True)
class CheckpointWriteResult:
    inserted: bool
    record: Dict[str, Any]


class CheckpointClient:
    """Thin wrapper around Supabase tables used by the demo."""

    def __init__(self, supabase_url: str, supabase_key: str) -> None:
        if not supabase_url:
            raise ConfigurationError("SUPABASE_URL is required")
        if not supabase_key:
            raise ConfigurationError("SUPABASE_KEY is required")

        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.client: Client = create_client(supabase_url, supabase_key)

    @classmethod
    def from_env(cls, env_file: str = ".env") -> "CheckpointClient":
        load_dotenv(env_file)
        supabase_url = os.getenv("SUPABASE_URL", "").strip()
        supabase_key = os.getenv("SUPABASE_KEY", "").strip()
        return cls(supabase_url, supabase_key)

    def verify_connection(self) -> None:
        """Fail fast if the Supabase project is not reachable."""
        try:
            self.client.table("agent_sessions").select("id").limit(1).execute()
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            raise ConnectionError(
                f"ERROR: Cannot connect to Supabase at {self.supabase_url}\n"
                "Check your SUPABASE_URL and SUPABASE_KEY in .env file\n"
                "Make sure your Supabase project is active at supabase.com"
            ) from exc

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        data = (
            self.client.table("agent_sessions")
            .select("*")
            .eq("session_id", session_id)
            .limit(1)
            .execute()
        ).data
        return data[0] if data else None

    def create_session(
        self,
        *,
        session_id: str,
        task_goal: str,
        steps_total: int,
        machine_id: str,
        steps_completed: int = 0,
        status: str = "in_progress",
        final_report: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "session_id": session_id,
            "task_goal": task_goal,
            "steps_total": steps_total,
            "steps_completed": steps_completed,
            "status": status,
            "final_report": final_report,
            "machine_id": machine_id,
            "updated_at": self._utcnow(),
        }
        return self._execute_single(
            self.client.table("agent_sessions").insert(payload).execute(),
            "create session",
        )

    def upsert_session(
        self,
        *,
        session_id: str,
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = dict(updates)
        payload["session_id"] = session_id
        payload.setdefault("updated_at", self._utcnow())
        return self._execute_single(
            self.client.table("agent_sessions").upsert(payload, on_conflict="session_id").execute(),
            "upsert session",
        )

    def update_session(
        self,
        *,
        session_id: str,
        **updates: Any,
    ) -> Dict[str, Any]:
        updates = dict(updates)
        updates["updated_at"] = self._utcnow()
        return self._execute_single(
            self.client.table("agent_sessions")
            .update(updates)
            .eq("session_id", session_id)
            .execute(),
            "update session",
        )

    def list_sessions(self) -> List[Dict[str, Any]]:
        response = (
            self.client.table("agent_sessions")
            .select("*")
            .order("updated_at", desc=True)
            .execute()
        )
        return list(response.data or [])

    def delete_session(self, session_id: str) -> None:
        """Delete the state, checkpoints, and session row for a demo reset."""
        self.client.table("agent_state").delete().eq("session_id", session_id).execute()
        self.client.table("agent_checkpoints").delete().eq("session_id", session_id).execute()
        self.client.table("agent_sessions").delete().eq("session_id", session_id).execute()

    def get_checkpoints(self, session_id: str) -> List[Dict[str, Any]]:
        response = (
            self.client.table("agent_checkpoints")
            .select("*")
            .eq("session_id", session_id)
            .order("step_number")
            .execute()
        )
        return list(response.data or [])

    def get_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        response = (
            self.client.table("agent_state")
            .select("*")
            .eq("session_id", session_id)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    def upsert_state(
        self,
        *,
        session_id: str,
        next_step_number: int,
        next_problem: str,
        machine_id: str,
        agent_scratchpad: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "session_id": session_id,
            "next_step_number": next_step_number,
            "next_problem": next_problem,
            "agent_scratchpad": agent_scratchpad,
            "machine_id": machine_id,
            "updated_at": self._utcnow(),
        }
        return self._execute_single(
            self.client.table("agent_state").upsert(payload, on_conflict="session_id").execute(),
            "upsert state",
        )

    def insert_checkpoint(
        self,
        *,
        session_id: str,
        checkpoint_number: int,
        machine_id: str,
        step_number: int,
        problem: str,
        solution: str,
        reasoning: str,
        completed_at: Optional[str] = None,
    ) -> CheckpointWriteResult:
        existing = (
            self.client.table("agent_checkpoints")
            .select("*")
            .eq("session_id", session_id)
            .eq("step_number", step_number)
            .limit(1)
            .execute()
        ).data
        if existing:
            return CheckpointWriteResult(inserted=False, record=existing[0])

        payload = {
            "session_id": session_id,
            "checkpoint_number": checkpoint_number,
            "machine_id": machine_id,
            "step_number": step_number,
            "problem": problem,
            "solution": solution,
            "reasoning": reasoning,
            "completed_at": completed_at or self._utcnow(),
        }

        try:
            record = self._execute_single(
                self.client.table("agent_checkpoints").insert(payload).execute(),
                "insert checkpoint",
            )
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            if self._looks_like_duplicate(exc):
                existing = (
                    self.client.table("agent_checkpoints")
                    .select("*")
                    .eq("session_id", session_id)
                    .eq("step_number", step_number)
                    .limit(1)
                    .execute()
                ).data
                if existing:
                    return CheckpointWriteResult(inserted=False, record=existing[0])
            raise

        return CheckpointWriteResult(inserted=True, record=record)

    def get_status_details(self, session_id: str) -> Dict[str, Any]:
        session = self.get_session(session_id)
        state = self.get_state(session_id)
        checkpoints = self.get_checkpoints(session_id)
        return {
            "session": session,
            "state": state,
            "checkpoints": checkpoints,
        }

    def get_session_progress(self, session_id: str) -> Dict[str, Any]:
        session = self.get_session(session_id)
        if not session:
            return {
                "session": None,
                "completed_steps": 0,
                "remaining_steps": 0,
                "next_step_number": 1,
            }

        completed_steps = int(session.get("steps_completed") or 0)
        steps_total = int(session.get("steps_total") or 0)
        next_step_number = completed_steps + 1
        return {
            "session": session,
            "completed_steps": completed_steps,
            "remaining_steps": max(steps_total - completed_steps, 0),
            "next_step_number": next_step_number,
        }

    def _execute_single(self, response: Any, action: str) -> Dict[str, Any]:
        data = getattr(response, "data", None)
        if not data:
            raise CheckpointClientError(f"Failed to {action}")
        return data[0]

    def _looks_like_duplicate(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "duplicate" in message or "unique" in message or "conflict" in message

    def _utcnow(self) -> str:
        return datetime.now(timezone.utc).isoformat()


def load_checkpoint_client(env_file: str = ".env") -> CheckpointClient:
    """Convenience helper for callers that only need a ready-to-use client."""
    client = CheckpointClient.from_env(env_file=env_file)
    client.verify_connection()
    return client
