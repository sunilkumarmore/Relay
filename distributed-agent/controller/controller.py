from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from supabase import create_client


console = Console()

PROBLEMS = {
    1: "Number Theory",
    2: "Logic Puzzle",
    3: "Probability",
    4: "Algorithms",
    5: "Proof",
}


def _load_env() -> None:
    load_dotenv()


def _get_session_id(explicit: str | None = None) -> str:
    session_id = explicit or os.getenv("SESSION_ID", "").strip()
    if not session_id:
        console.print(
            'ERROR: SESSION_ID not set in .env file\n'
            'Generate one: python -c "import uuid; print(uuid.uuid4())"\n'
            "Use the SAME session ID on both machines"
        )
        raise SystemExit(1)
    return session_id


def _build_client():
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    if not url or not key:
        console.print(
            f"ERROR: Cannot connect to Supabase at {url or '<missing SUPABASE_URL>'}\n"
            "Check your SUPABASE_URL and SUPABASE_KEY in .env file\n"
            "Make sure your Supabase project is active at supabase.com"
        )
        raise SystemExit(1)
    try:
        client = create_client(url, key)
        client.table("agent_sessions").select("id").limit(1).execute()
        return client
    except Exception as exc:
        console.print(
            f"ERROR: Cannot connect to Supabase at {url}\n"
            "Check your SUPABASE_URL and SUPABASE_KEY in .env file\n"
            "Make sure your Supabase project is active at supabase.com"
        )
        raise SystemExit(1) from exc


def _format_timestamp(value: Any) -> str:
    if not value:
        return "-"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def _get_session(client, session_id: str) -> dict[str, Any]:
    result = (
        client.table("agent_sessions")
        .select("*")
        .eq("session_id", session_id)
        .limit(1)
        .execute()
    )
    data = list(result.data or [])
    if not data:
        console.print(f"ERROR: Session not found: {session_id}")
        raise SystemExit(1)
    return data[0]


def _get_checkpoints(client, session_id: str) -> list[dict[str, Any]]:
    result = (
        client.table("agent_checkpoints")
        .select("*")
        .eq("session_id", session_id)
        .order("step_number", desc=False)
        .execute()
    )
    return list(result.data or [])


def _latest_checkpoint(checkpoints: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not checkpoints:
        return None
    return max(
        checkpoints,
        key=lambda row: (
            str(row.get("completed_at", "")),
            int(row.get("checkpoint_number", 0) or 0),
            int(row.get("step_number", 0) or 0),
        ),
    )


def _print_status(session_id: str, session: dict[str, Any], checkpoints: list[dict[str, Any]]) -> None:
    by_step = {int(row["step_number"]): row for row in checkpoints if row.get("step_number") is not None}
    latest = _latest_checkpoint(checkpoints)

    console.print(f"Session: {session_id}")
    console.print(f"Status: {session.get('status', 'unknown')}")
    console.print(
        f"Steps: {int(session.get('steps_completed') or 0)} of {int(session.get('steps_total') or 5)}"
    )
    console.print(
        f"Last checkpoint: {_format_timestamp(latest.get('completed_at') if latest else None)}"
    )

    table = Table()
    table.add_column("Problem", justify="right")
    table.add_column("Topic")
    table.add_column("Status")
    table.add_column("Machine")

    for step in range(1, 6):
        row = by_step.get(step)
        table.add_row(
            str(step),
            PROBLEMS[step],
            "✓ Done" if row else "pending",
            str(row.get("machine_id")) if row else "-",
        )
    console.print(table)


def _build_report(session: dict[str, Any], checkpoints: list[dict[str, Any]]) -> str:
    existing = session.get("final_report")
    if existing:
        return str(existing)
    lines = [
        f"Session: {session.get('session_id')}",
        f"Goal: {session.get('task_goal')}",
        "",
    ]
    for row in sorted(checkpoints, key=lambda item: int(item.get("step_number", 0) or 0)):
        lines.extend(
            [
                f"Problem {row.get('step_number')}:",
                str(row.get("solution", "")),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _delete_session(client, session_id: str) -> None:
    client.table("agent_state").delete().eq("session_id", session_id).execute()
    client.table("agent_checkpoints").delete().eq("session_id", session_id).execute()
    client.table("agent_sessions").delete().eq("session_id", session_id).execute()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """CLI controller for distributed agent sessions."""


@cli.command()
@click.option("--session-id", default=None, help="Override SESSION_ID from .env")
def status(session_id: str | None) -> None:
    """Show current session state from Supabase."""
    _load_env()
    sid = _get_session_id(session_id)
    client = _build_client()
    session = _get_session(client, sid)
    checkpoints = _get_checkpoints(client, sid)
    _print_status(sid, session, checkpoints)


@cli.command()
@click.option("--session-id", default=None, help="Override SESSION_ID from .env")
def report(session_id: str | None) -> None:
    """Print final report if session is complete."""
    _load_env()
    sid = _get_session_id(session_id)
    client = _build_client()
    session = _get_session(client, sid)
    steps_total = int(session.get("steps_total") or 5)
    steps_completed = int(session.get("steps_completed") or 0)
    status_value = str(session.get("status") or "").lower()

    if status_value != "completed" or steps_completed < steps_total:
        console.print(
            f"ERROR: Session {sid} is not complete yet\n"
            f"Current progress: {steps_completed} of {steps_total}"
        )
        raise SystemExit(1)

    checkpoints = _get_checkpoints(client, sid)
    console.print(_build_report(session, checkpoints))


@cli.command()
@click.option("--session-id", default=None, help="Override SESSION_ID from .env")
def reset(session_id: str | None) -> None:
    """Delete all records for current SESSION_ID."""
    _load_env()
    sid = _get_session_id(session_id)
    sys.stdout.write("Are you sure? (y/n) ")
    sys.stdout.flush()
    answer = input().strip().lower()
    if answer not in {"y", "yes"}:
        console.print("Reset cancelled.")
        raise SystemExit(0)

    client = _build_client()
    _get_session(client, sid)
    _delete_session(client, sid)
    console.print(f"Deleted all records for session {sid}")


@cli.command()
def sessions() -> None:
    """List all sessions with status and progress."""
    _load_env()
    client = _build_client()
    result = client.table("agent_sessions").select("*").order("updated_at", desc=True).execute()
    sessions_rows = list(result.data or [])

    if not sessions_rows:
        console.print("No sessions found.")
        return

    table = Table()
    table.add_column("Session")
    table.add_column("Status")
    table.add_column("Progress")
    table.add_column("Machine")
    table.add_column("Updated")

    for row in sessions_rows:
        steps_completed = int(row.get("steps_completed") or 0)
        steps_total = int(row.get("steps_total") or 5)
        table.add_row(
            str(row.get("session_id", "-")),
            str(row.get("status", "-")),
            f"{steps_completed}/{steps_total}",
            str(row.get("machine_id", "-")),
            _format_timestamp(row.get("updated_at")),
        )
    console.print(table)


if __name__ == "__main__":
    try:
        cli()
    except Exception as exc:  # pragma: no cover
        console.print(f"ERROR: {exc}")
        raise SystemExit(1)
