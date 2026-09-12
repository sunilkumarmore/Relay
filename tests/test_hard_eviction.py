"""SIGKILL — the eviction that gets no warning.

Graceful eviction is the easy case: the worker catches the signal and writes its
state. Real preemption does not ask. A spot instance reclaimed, a power cut, an
OOM kill — the process simply stops, mid-inference, with no chance to save
anything.

Relay's claim is that this costs at most the step in flight, because progress
lives in the checkpoint table and `UNIQUE(session_id, step_number)` makes redoing
a step safe. These tests hold it to that.
"""

from __future__ import annotations

import signal
from collections import Counter

TOTAL_STEPS = 5
SESSION = "sess-sigkill"


def events(store, session_id: str) -> list[str]:
    return [e["event"] for e in store.snapshot()["migration_log"] if e["session_id"] == session_id]


def test_kill_9_loses_at_most_the_step_in_flight(store, registry_server, spawn_worker, task_file):
    server = registry_server(store, latency_ms=500)
    path = task_file(TOTAL_STEPS)

    worker = spawn_worker(
        store=store,
        registry_url=server.url,
        session_id=SESSION,
        task_path=path,
        worker_id="worker-alpha",
        machine_id="machine-a",
    )
    assert worker.wait_for_checkpoints(2), "worker never got far enough to be worth killing"

    worker.proc.send_signal(signal.SIGKILL)
    assert worker.proc.wait(timeout=30) == -signal.SIGKILL

    committed = [c["step_number"] for c in store.get_checkpoints(SESSION)]
    assert 0 < len(committed) < TOTAL_STEPS
    # Nothing was saved on the way out — there was no way out.
    assert "evicted" not in events(store, SESSION)
    calls_before = server.backend.calls

    resumed = spawn_worker(
        store=store,
        registry_url=server.url,
        session_id=SESSION,
        task_path=path,
        worker_id="worker-beta",
        machine_id="machine-b",
    )
    assert resumed.proc.wait(timeout=90) == 0, resumed.proc.stdout.read()

    final = [c["step_number"] for c in store.get_checkpoints(SESSION)]
    assert final == list(range(1, TOTAL_STEPS + 1))

    # The constraint held: one row per step, no matter how the first run died.
    duplicates = [step for step, n in Counter(final).items() if n > 1]
    assert duplicates == [], f"duplicate (session_id, step_number) rows: {duplicates}"

    # At most one step was recomputed — the one being generated when the kill landed.
    redone = (server.backend.calls - calls_before) - (TOTAL_STEPS - len(committed))
    assert 0 <= redone <= 1, f"recomputed {redone} steps, expected at most the in-flight one"

    session = store.get_session(SESSION)
    assert session["status"] == "completed"
    assert session["current_machine"] == "machine-b"
    assert events(store, SESSION)[-1] == "completed"


def test_kill_9_survives_even_before_the_first_checkpoint(
    store, registry_server, spawn_worker, task_file
):
    server = registry_server(store, latency_ms=500)
    path = task_file(3)

    worker = spawn_worker(
        store=store, registry_url=server.url, session_id="sess-early-kill", task_path=path
    )
    # Wait only until the session row exists, then kill during step 1.
    deadline_hit = False
    for _ in range(600):
        if store.get_session("sess-early-kill") is not None:
            break
        if worker.proc.poll() is not None:
            deadline_hit = True
            break
    assert not deadline_hit

    worker.proc.send_signal(signal.SIGKILL)
    worker.proc.wait(timeout=30)

    resumed = spawn_worker(
        store=store,
        registry_url=server.url,
        session_id="sess-early-kill",
        task_path=path,
        worker_id="worker-beta",
    )
    assert resumed.proc.wait(timeout=90) == 0
    assert [c["step_number"] for c in store.get_checkpoints("sess-early-kill")] == [1, 2, 3]


def test_store_file_stays_readable_after_a_kill(store, registry_server, spawn_worker, task_file):
    """A process killed mid-write must not corrupt the store for everyone else."""
    server = registry_server(store, latency_ms=200)
    worker = spawn_worker(
        store=store, registry_url=server.url, session_id="sess-torn", task_path=task_file(5)
    )
    assert worker.wait_for_checkpoints(1)
    worker.proc.send_signal(signal.SIGKILL)
    worker.proc.wait(timeout=30)

    snapshot = store.snapshot()
    assert set(snapshot) == {
        "sessions",
        "worker_state",
        "checkpoints",
        "inference_log",
        "migration_log",
    }
    assert store.get_session("sess-torn") is not None
