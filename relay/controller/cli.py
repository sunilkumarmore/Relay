from __future__ import annotations

import click
import requests
from rich.console import Console
from rich.table import Table

from relay import config
from relay.identity import identity_from_env
from relay.ledger import Ledger, LedgerError, sweep_expired_holds
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


@cli.group("wallet")
def wallet() -> None:
    """Credits: what you have, what is committed, and where it went."""


@wallet.command("balance")
def wallet_balance_cmd() -> None:
    """Show available and held credits for this node."""
    store, _ = load_config()
    ledger = Ledger(store)
    node_id = identity_from_env().node_id

    table = Table(title=f"Wallet {node_id[:16]}")
    table.add_column("Available")
    table.add_column("Held")
    table.add_column("Total")
    held = ledger.held_balance(node_id)
    balance = ledger.balance(node_id)
    table.add_row(f"{balance:.6f}", f"{held:.6f}", f"{balance + held:.6f}")
    console.print(table)


@wallet.command("deposit")
@click.option("--amount", type=float, required=True, help="Credits to add")
@click.option("--dev", is_flag=True, help="Use the development faucet")
def wallet_deposit_cmd(amount: float, dev: bool) -> None:
    """Add credits. In v1 the only source is the development faucet."""
    if not dev:
        console.print(
            "Relay has no payment rail yet, so --dev is the only way to add credits.\n"
            "A real deposit would post the same ledger entry from a payment webhook."
        )
        raise SystemExit(1)
    if config.get("RELAY_DEV_MODE") != "1":
        console.print("Refusing: set RELAY_DEV_MODE=1 to use the faucet.")
        raise SystemExit(1)
    if amount <= 0:
        console.print("Deposit must be positive.")
        raise SystemExit(1)

    store, _ = load_config()
    node_id = identity_from_env().node_id
    ledger = Ledger(store)
    ledger.deposit(node_id, amount)
    console.print(f"Deposited {amount} credits. Balance: {ledger.balance(node_id):.6f}")


@wallet.command("history")
@click.option("--limit", type=int, default=20, show_default=True)
def wallet_history_cmd(limit: int) -> None:
    """Show recent ledger entries for this node."""
    store, _ = load_config()
    node_id = identity_from_env().node_id

    table = Table(title="Ledger")
    for column in ("When", "Kind", "Account", "Debit", "Credit", "Job"):
        table.add_column(column)
    for row in Ledger(store).history(node_id, limit):
        account = str(row.get("account", ""))
        table.add_row(
            str(row.get("created_at", ""))[11:19],
            str(row.get("kind", "")),
            account.split(":", 1)[-1],
            f"{float(row.get('debit') or 0):.6f}",
            f"{float(row.get('credit') or 0):.6f}",
            str(row.get("ref_job_id", ""))[:12],
        )
    console.print(table)


@wallet.command("sweep")
@click.option("--ttl", type=int, default=3600, show_default=True, help="Seconds of silence")
def wallet_sweep_cmd(ttl: int) -> None:
    """Release holds for jobs that were abandoned rather than finished."""
    store, _ = load_config()
    released = sweep_expired_holds(store, Ledger(store), ttl_seconds=ttl)
    if not released:
        console.print("Nothing to release.")
        return
    for node_id, job_id, amount in released:
        console.print(f"Released {amount:.6f} held by {node_id[:12]} for job {job_id}")


@wallet.command("verify")
def wallet_verify_cmd() -> None:
    """Check that the ledger balances. Every account together must be zero."""
    store, _ = load_config()
    ledger = Ledger(store)
    try:
        ledger.check_invariant()
    except LedgerError as exc:
        console.print(f"LEDGER IS BROKEN: {exc}")
        raise SystemExit(1) from exc
    console.print(f"Ledger balances. Total across all accounts: {ledger.total():.6f}")


@cli.command("receipts")
@click.option("--session", "session_id", default=None, help="Only this job")
def receipts_cmd(session_id: str | None) -> None:
    """Show receipts and whether they were acknowledged."""
    store, _ = load_config()
    rows = store.list_receipts(job_id=session_id)

    table = Table(title="Receipts")
    for column in ("Job", "Step", "Provider", "Tokens in/out", "Credits", "Status"):
        table.add_column(column)
    for row in rows:
        table.add_row(
            str(row.get("job_id", ""))[:12],
            str(row.get("step_number", "")),
            str(row.get("provider_node_id", ""))[:12],
            f"{row.get('tokens_in', 0)}/{row.get('tokens_out', 0)}",
            f"{float(row.get('amount_credits') or 0):.6f}",
            str(row.get("status", "")),
        )
    console.print(table)

    total = sum(float(r.get("amount_credits") or 0) for r in rows if r.get("status") == "acknowledged")
    console.print(f"Acknowledged total: {total:.6f} credits")


@cli.command("earnings")
def earnings_cmd() -> None:
    """Show what this node has earned as a provider."""
    store, _ = load_config()
    node_id = identity_from_env().node_id
    rows = store.list_receipts(provider_node_id=node_id)

    buckets: dict[str, float] = {}
    for row in rows:
        status = str(row.get("status", "unacknowledged"))
        buckets[status] = buckets.get(status, 0.0) + float(row.get("amount_credits") or 0)

    table = Table(title=f"Earnings {node_id[:16]}")
    table.add_column("Status")
    table.add_column("Credits")
    for status in sorted(buckets):
        table.add_row(status, f"{buckets[status]:.6f}")
    console.print(table)
    console.print(f"Settled balance: {Ledger(store).balance(node_id):.6f}")


def run() -> None:
    cli()


if __name__ == "__main__":
    cli()
