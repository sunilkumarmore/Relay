from __future__ import annotations

import os

import click
import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "worker") not in sys.path:
    sys.path.append(str(ROOT / "worker"))
from checkpoint_client import RelayCheckpointClient, RelayCheckpointError


console = Console()


def load_config() -> tuple[RelayCheckpointClient, str]:
    env_file = os.getenv("ENV_FILE", ".env")
    load_dotenv(env_file)
    client = RelayCheckpointClient.from_env(env_file)
    client.verify_connection()
    registry = os.getenv("INFERENCE_REGISTRY", "http://localhost:8765")
    return client, registry


@click.group()
def cli() -> None:
    pass


@cli.command("status")
def status_cmd() -> None:
    """Show all session status from Supabase."""
    try:
        client, _ = load_config()
    except RelayCheckpointError as exc:
        console.print(str(exc))
        raise SystemExit(1)

    rows = client.list_sessions()
    table = Table(title="Relay Sessions")
    table.add_column("Session")
    table.add_column("Worker")
    table.add_column("Status")
    table.add_column("Progress")
    table.add_column("Machine")
    table.add_column("Inference")

    for row in rows:
        table.add_row(
            str(row.get("session_id", "-")),
            str(row.get("worker_id", "-")),
            str(row.get("status", "-")),
            f"{row.get('steps_completed', 0)}/{row.get('steps_total', 5)}",
            str(row.get("current_machine", "-")),
            str(row.get("inference_node", "-")),
        )

    console.print(table)


@cli.command("inference-status")
def inference_status_cmd() -> None:
    """Show inference registry runtime metrics."""
    _, registry = load_config()
    response = requests.get(f"{registry.rstrip('/')}/inference/status", timeout=10)
    response.raise_for_status()
    body = response.json()

    console.print(f"Inference Node: {body.get('inference_node_id', '-')}")
    console.print(f"Active Workers: {len(body.get('active_workers', []))}")
    console.print(f"Requests/min: {body.get('requests_last_minute', 0)}")
    console.print(f"Avg Latency (ms): {body.get('avg_latency_ms', 0)}")
    console.print(f"Total Requests: {body.get('total_requests', 0)}")


@cli.command("workers")
def workers_cmd() -> None:
    """Show active worker state and recent migration events."""
    client, _ = load_config()

    states = client.list_worker_state()
    events = client.list_migration_events(limit=20)

    state_table = Table(title="Worker State")
    state_table.add_column("Worker")
    state_table.add_column("Session")
    state_table.add_column("Status")
    state_table.add_column("Next Step")
    state_table.add_column("Machine")
    state_table.add_column("Inference")

    for row in states:
        state_table.add_row(
            str(row.get("worker_id", "-")),
            str(row.get("session_id", "-")),
            str(row.get("status", "-")),
            str(row.get("next_step_number", "-")),
            str(row.get("machine_id", "-")),
            str(row.get("inference_node", "-")),
        )

    event_table = Table(title="Migration Events")
    event_table.add_column("Time")
    event_table.add_column("Worker")
    event_table.add_column("Event")
    event_table.add_column("From")
    event_table.add_column("To")
    event_table.add_column("Step")

    for row in events:
        event_table.add_row(
            str(row.get("occurred_at", ""))[11:19],
            str(row.get("worker_id", "-")),
            str(row.get("event", "-")),
            str(row.get("from_machine", "-")),
            str(row.get("to_machine", "-")),
            str(row.get("step_at_event", "-")),
        )

    console.print(state_table)
    console.print(event_table)


@cli.command("reset")
@click.option("--session", "session_id", required=True, help="Session ID to delete")
def reset_cmd(session_id: str) -> None:
    """Delete all records for one session."""
    client, _ = load_config()
    answer = input("Are you sure? (y/n) ").strip().lower()
    if answer not in {"y", "yes"}:
        console.print("Reset cancelled.")
        return
    client.reset_session(session_id)
    console.print(f"Deleted session {session_id}")


@cli.command("report")
@click.option("--session", "session_id", required=True, help="Session ID")
def report_cmd(session_id: str) -> None:
    """Print final report for one session."""
    client, _ = load_config()
    session = client.get_session(session_id)
    if not session:
        console.print(f"Session not found: {session_id}")
        raise SystemExit(1)

    if str(session.get("status", "")).lower() != "completed":
        console.print(f"Session {session_id} is not complete.")
        raise SystemExit(1)

    report = session.get("final_report")
    if report:
        console.print(str(report))
        return

    checkpoints = client.get_checkpoints(session_id)
    for row in checkpoints:
        console.print(f"Problem {row.get('step_number')}:\n{row.get('solution')}\n")


if __name__ == "__main__":
    cli()
