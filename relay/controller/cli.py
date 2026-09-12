from __future__ import annotations

import click
import requests
from rich.console import Console
from rich.table import Table

from relay import config
from relay.store import RelayStoreError, Store, store_from_env

console = Console()


def load_config() -> tuple[Store, str]:
    config.load_env()
    store = store_from_env()
    assert store is not None
    store.verify_connection()
    registry = config.get("INFERENCE_REGISTRY", "http://localhost:8765") or "http://localhost:8765"
    return store, registry


@click.group()
def cli() -> None:
    pass


@cli.command("status")
def status_cmd() -> None:
    """Show all session status from Supabase."""
    try:
        store, _ = load_config()
    except RelayStoreError as exc:
        console.print(str(exc))
        raise SystemExit(1) from exc

    rows = store.list_sessions()
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
            f"{row.get('steps_completed', 0)}/{row.get('steps_total', '?')}",
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
    store, _ = load_config()

    states = store.list_worker_state()
    events = store.list_migration_events(limit=20)

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
    store, _ = load_config()
    if not click.confirm(f"Delete all data for session {session_id}?", default=False):
        console.print("Reset cancelled.")
        return
    store.reset_session(session_id)
    console.print(f"Deleted session {session_id}")


@cli.command("report")
@click.option("--session", "session_id", required=True, help="Session ID")
def report_cmd(session_id: str) -> None:
    """Print final report for one session."""
    store, _ = load_config()
    session = store.get_session(session_id)
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

    checkpoints = store.get_checkpoints(session_id)
    for row in checkpoints:
        console.print(f"Step {row.get('step_number')}:\n{row.get('solution')}\n")


def run() -> None:
    cli()


if __name__ == "__main__":
    cli()
