from __future__ import annotations

from datetime import datetime
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "worker") not in sys.path:
    sys.path.append(str(ROOT / "worker"))
from checkpoint_client import RelayCheckpointClient


def get_client() -> RelayCheckpointClient:
    load_dotenv(os.getenv("ENV_FILE", ".env"))
    client = RelayCheckpointClient.from_env(os.getenv("ENV_FILE", ".env"))
    client.verify_connection()
    return client


def fetch_inference_status(url: str) -> dict:
    try:
        response = requests.get(f"{url.rstrip('/')}/inference/status", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {
            "active_workers": [],
            "requests_last_minute": 0,
            "avg_latency_ms": 0,
            "total_requests": 0,
            "inference_node_id": "offline",
            "recent_calls": [],
        }


def left_panel(status: dict, client: RelayCheckpointClient) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_row(f"Inference Node: {status.get('inference_node_id', 'unknown')}")
    table.add_row(f"Active Workers: {len(status.get('active_workers', []))}")
    table.add_row(f"Requests/min: {status.get('requests_last_minute', 0)}")
    table.add_row(f"Avg Latency: {status.get('avg_latency_ms', 0)} ms")
    table.add_row(f"Total Requests: {status.get('total_requests', 0)}")

    worker_table = Table(title="Active Workers")
    worker_table.add_column("Worker")
    worker_table.add_column("Session")
    worker_table.add_column("Progress")
    sessions_by_id = {s.get("session_id"): s for s in client.list_sessions()}
    for worker in status.get("active_workers", []):
        done = int(worker.get("steps_completed", 0))
        session = sessions_by_id.get(worker.get("session_id", ""))
        total = int(session.get("steps_total", done) if session else done) or done
        bar = "#" * done + "-" * max(0, total - done)
        worker_table.add_row(worker.get("worker_id", "?"), worker.get("session_id", "?"), f"{bar} {done}/{total}")

    recent = Table(title="Recent Calls")
    recent.add_column("Time")
    recent.add_column("Worker")
    recent.add_column("Latency")
    recent.add_column("OK")
    for call in status.get("recent_calls", [])[-5:]:
        ts = str(call.get("ts", ""))[11:19]
        recent.add_row(ts, str(call.get("worker_id", "")), f"{call.get('latency_ms', 0)} ms", "yes" if call.get("success") else "no")

    return Panel(Group(table, worker_table, recent), title="INFERENCE NODE STATUS")


def event_color(event: str) -> str:
    mapping = {
        "started": "green",
        "evicted": "yellow",
        "migrated": "magenta",
        "resumed": "cyan",
        "completed": "bright_green",
    }
    return mapping.get(event, "white")


def right_panel(client: RelayCheckpointClient) -> Panel:
    sessions = client.list_sessions()
    events = client.list_migration_events(limit=20)

    workers = Table(title="Worker Status")
    workers.add_column("Worker")
    workers.add_column("Session")
    workers.add_column("Status")
    workers.add_column("Steps")
    workers.add_column("Machine")

    for session in sessions[:8]:
        workers.add_row(
            str(session.get("worker_id", "-")),
            str(session.get("session_id", "-")),
            str(session.get("status", "-")),
            f"{session.get('steps_completed', 0)}/{session.get('steps_total', '?')}",
            str(session.get("current_machine", "-")),
        )

    log_table = Table(title="Migration Log")
    log_table.add_column("Time")
    log_table.add_column("Worker")
    log_table.add_column("Event")
    for event in events:
        time_str = str(event.get("occurred_at", ""))[11:19]
        e = str(event.get("event", "")).lower()
        text = Text(e.upper(), style=event_color(e))
        log_table.add_row(time_str, str(event.get("worker_id", "-")), text)

    return Panel(Group(workers, log_table), title="WORKER MIGRATION")


def build_layout(status: dict, client: RelayCheckpointClient) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=1),
    )
    layout["body"].split_row(Layout(name="left"), Layout(name="right"))

    layout["header"].update(Panel("RELAY - Distributed Agent Compute Network", style="bold white on blue"))
    layout["left"].update(left_panel(status, client))
    layout["right"].update(right_panel(client))
    layout["footer"].update(Text(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}", style="dim"))
    return layout


def main() -> int:
    load_dotenv(os.getenv("ENV_FILE", ".env"))
    inference_registry = os.getenv("INFERENCE_REGISTRY", "http://localhost:8765")
    client = get_client()

    with Live(refresh_per_second=4, screen=False) as live:
        while True:
            status = fetch_inference_status(inference_registry)
            live.update(build_layout(status, client))
            time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())
