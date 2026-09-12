"""End-to-end worker runs, including eviction and resume.

Workers run as real subprocesses against a real registry on loopback. That is the
only way to test the thing Relay actually claims: that killing the process does
not lose the work.
"""

from __future__ import annotations

import signal

import pytest

TOTAL_STEPS = 4


def events(store, session_id: str) -> list[str]:
    return [e["event"] for e in store.snapshot()["migration_log"] if e["session_id"] == session_id]


def step_numbers(store, session_id: str) -> list[int]:
    return [c["step_number"] for c in store.get_checkpoints(session_id)]


def test_worker_completes_every_step(store, registry_server, spawn_worker, task_file, tmp_path):
    server = registry_server(store)
    path = task_file(TOTAL_STEPS)

    worker = spawn_worker(
        store=store, registry_url=server.url, session_id="sess-complete", task_path=path
    )
    assert worker.proc.wait(timeout=60) == 0, worker.proc.stdout.read()

    assert step_numbers(store, "sess-complete") == [1, 2, 3, 4]
    session = store.get_session("sess-complete")
    assert session["status"] == "completed"
    assert session["steps_completed"] == TOTAL_STEPS
    assert session["final_report"].startswith("Session: sess-complete")

    assert events(store, "sess-complete") == ["started", "completed"]
    # One row per call, written by the registry only — the worker used to write a
    # second copy of each, which doubled every figure on the dashboard.
    assert len(store.snapshot()["inference_log"]) == TOTAL_STEPS

    reports = list((tmp_path / "output").glob("final_report_*.txt"))
    assert len(reports) == 1
    assert "Step 4" in reports[0].read_text()


def test_worker_deregisters_on_exit(store, registry_server, spawn_worker, task_file):
    server = registry_server(store)
    worker = spawn_worker(
        store=store, registry_url=server.url, session_id="sess-dereg", task_path=task_file(2)
    )
    worker.proc.wait(timeout=60)
    assert server.registry.active_workers == {}


def test_graceful_eviction_saves_state_then_resumes_on_another_machine(
    store, registry_server, spawn_worker, task_file
):
    server = registry_server(store, latency_ms=400)
    path = task_file(TOTAL_STEPS)

    worker = spawn_worker(
        store=store,
        registry_url=server.url,
        session_id="sess-evict",
        task_path=path,
        worker_id="worker-alpha",
        machine_id="machine-a",
    )
    assert worker.wait_for_checkpoints(1), "worker never committed a step"

    worker.proc.send_signal(signal.SIGTERM)
    assert worker.proc.wait(timeout=30) == 0

    done_before = len(step_numbers(store, "sess-evict"))
    assert 0 < done_before < TOTAL_STEPS

    assert "evicted" in events(store, "sess-evict")
    state = store.get_state("sess-evict")
    assert state["status"] == "evicted"
    assert state["machine_id"] == "machine-a"
    # The runtime's view of "next step" is updated after the store write, so an
    # eviction landing in that window records the step it was mid-way through.
    # Either is fine: resume recomputes from the checkpoints, not from this field.
    assert done_before <= state["next_step_number"] <= done_before + 1

    # Resume as a different worker on a different machine.
    resumed = spawn_worker(
        store=store,
        registry_url=server.url,
        session_id="sess-evict",
        task_path=path,
        worker_id="worker-beta",
        machine_id="machine-b",
    )
    assert resumed.proc.wait(timeout=60) == 0, resumed.proc.stdout.read()

    assert step_numbers(store, "sess-evict") == [1, 2, 3, 4]
    seen = events(store, "sess-evict")
    assert "migrated" in seen, "a different worker picked the session up"
    assert "resumed" in seen
    assert seen[-1] == "completed"
    assert store.get_session("sess-evict")["current_machine"] == "machine-b"


def test_resume_does_not_redo_completed_steps(store, registry_server, spawn_worker, task_file):
    server = registry_server(store, latency_ms=400)
    path = task_file(TOTAL_STEPS)

    worker = spawn_worker(
        store=store, registry_url=server.url, session_id="sess-skip", task_path=path
    )
    assert worker.wait_for_checkpoints(2)
    worker.proc.send_signal(signal.SIGTERM)
    worker.proc.wait(timeout=30)

    done = len(step_numbers(store, "sess-skip"))
    calls_before = server.backend.calls

    resumed = spawn_worker(
        store=store,
        registry_url=server.url,
        session_id="sess-skip",
        task_path=path,
        worker_id="worker-beta",
    )
    assert resumed.proc.wait(timeout=60) == 0

    # The resumed run only pays for the steps that were still outstanding.
    assert server.backend.calls - calls_before == TOTAL_STEPS - done
    assert step_numbers(store, "sess-skip") == [1, 2, 3, 4]


def test_resumed_run_produces_the_same_answers_as_an_uninterrupted_one(
    store, registry_server, spawn_worker, task_file
):
    server = registry_server(store, latency_ms=300)
    path = task_file(TOTAL_STEPS)

    clean = spawn_worker(store=store, registry_url=server.url, session_id="sess-clean", task_path=path)
    assert clean.proc.wait(timeout=60) == 0
    expected = [c["solution"] for c in store.get_checkpoints("sess-clean")]

    interrupted = spawn_worker(
        store=store, registry_url=server.url, session_id="sess-broken", task_path=path
    )
    assert interrupted.wait_for_checkpoints(1)
    interrupted.proc.send_signal(signal.SIGTERM)
    interrupted.proc.wait(timeout=30)

    resumed = spawn_worker(
        store=store,
        registry_url=server.url,
        session_id="sess-broken",
        task_path=path,
        worker_id="worker-beta",
        machine_id="machine-b",
    )
    assert resumed.proc.wait(timeout=60) == 0

    assert [c["solution"] for c in store.get_checkpoints("sess-broken")] == expected


def test_a_second_worker_on_a_finished_session_adds_nothing(
    store, registry_server, spawn_worker, task_file
):
    server = registry_server(store)
    path = task_file(2)

    first = spawn_worker(store=store, registry_url=server.url, session_id="sess-dup", task_path=path)
    assert first.proc.wait(timeout=60) == 0
    calls_after_first = server.backend.calls

    second = spawn_worker(
        store=store,
        registry_url=server.url,
        session_id="sess-dup",
        task_path=path,
        worker_id="worker-beta",
    )
    assert second.proc.wait(timeout=60) == 0

    assert step_numbers(store, "sess-dup") == [1, 2]
    assert server.backend.calls == calls_after_first, "no step was recomputed"


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_both_eviction_signals_are_handled(store, registry_server, spawn_worker, task_file, sig):
    server = registry_server(store, latency_ms=400)
    worker = spawn_worker(
        store=store, registry_url=server.url, session_id=f"sess-{sig.name}", task_path=task_file(4)
    )
    assert worker.wait_for_checkpoints(1)

    worker.proc.send_signal(sig)
    assert worker.proc.wait(timeout=30) == 0
    assert "evicted" in events(store, f"sess-{sig.name}")
