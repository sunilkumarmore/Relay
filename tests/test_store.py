from __future__ import annotations

import pytest

from relay.store import FileStore, MemoryStore, Store

CHECKPOINT = {
    "session_id": "s1",
    "worker_id": "w1",
    "step_number": 1,
    "problem": "p",
    "solution": "sol",
    "reasoning": "sol",
    "machine_id": "m1",
    "inference_node": "n1",
    "inference_latency_ms": 5,
    "tokens_used": 10,
}


@pytest.fixture(params=["memory", "file"])
def any_store(request, tmp_path) -> Store:
    if request.param == "memory":
        return MemoryStore()
    return FileStore(tmp_path / "s.json")


def test_implements_protocol(any_store):
    assert isinstance(any_store, Store)


def test_insert_checkpoint_returns_false_on_duplicate_step(any_store):
    assert any_store.insert_checkpoint(**CHECKPOINT) is True
    # Same (session_id, step_number) — the constraint the whole resume story rests on.
    assert any_store.insert_checkpoint(**{**CHECKPOINT, "worker_id": "w2"}) is False
    rows = any_store.get_checkpoints("s1")
    assert len(rows) == 1
    assert rows[0]["worker_id"] == "w1"


def test_same_step_number_in_another_session_is_allowed(any_store):
    assert any_store.insert_checkpoint(**CHECKPOINT) is True
    assert any_store.insert_checkpoint(**{**CHECKPOINT, "session_id": "s2"}) is True
    assert len(any_store.get_checkpoints("s1")) == 1
    assert len(any_store.get_checkpoints("s2")) == 1


def test_checkpoints_come_back_in_step_order(any_store):
    for step in (3, 1, 2):
        any_store.insert_checkpoint(**{**CHECKPOINT, "step_number": step})
    assert [r["step_number"] for r in any_store.get_checkpoints("s1")] == [1, 2, 3]


def test_session_upsert_then_update(any_store):
    any_store.upsert_session(
        session_id="s1",
        worker_id="w1",
        task_goal="goal",
        steps_total=3,
        steps_completed=0,
        status="in_progress",
        current_machine="m1",
        inference_node="n1",
    )
    any_store.upsert_session(
        session_id="s1",
        worker_id="w1",
        task_goal="goal",
        steps_total=3,
        steps_completed=1,
        status="in_progress",
        current_machine="m1",
        inference_node="n1",
    )
    assert len(any_store.list_sessions()) == 1

    any_store.update_session("s1", steps_completed=2, status="completed")
    session = any_store.get_session("s1")
    assert session["steps_completed"] == 2
    assert session["status"] == "completed"


def test_state_is_unique_per_session(any_store):
    for worker in ("w1", "w2"):
        any_store.upsert_state(
            session_id="s1",
            worker_id=worker,
            next_step_number=2,
            next_problem="p2",
            machine_id="m1",
            inference_node="n1",
            status="active",
        )
    assert len(any_store.list_worker_state()) == 1
    # The session is the unit of identity; the worker holding it is replaceable.
    assert any_store.get_state("s1")["worker_id"] == "w2"


def test_migration_events_newest_first_and_limited(any_store):
    for event in ("started", "evicted", "resumed", "completed"):
        any_store.insert_migration_event(
            session_id="s1",
            worker_id="w1",
            event=event,
            from_machine=None,
            to_machine="m1",
            step_at_event=0,
        )
    assert len(any_store.list_migration_events(limit=2)) == 2
    assert {e["event"] for e in any_store.list_migration_events()} == {
        "started",
        "evicted",
        "resumed",
        "completed",
    }


def test_reset_session_clears_every_table(any_store):
    any_store.insert_checkpoint(**CHECKPOINT)
    any_store.upsert_session(
        session_id="s1",
        worker_id="w1",
        task_goal="g",
        steps_total=1,
        steps_completed=1,
        status="completed",
        current_machine="m1",
        inference_node="n1",
    )
    any_store.insert_inference_log(
        worker_id="w1", session_id="s1", inference_node="n1", latency_ms=1, tokens_used=1, success=True
    )
    any_store.insert_migration_event(
        session_id="s1", worker_id="w1", event="started", from_machine=None, to_machine="m1", step_at_event=0
    )

    any_store.reset_session("s1")

    assert any_store.get_session("s1") is None
    assert any_store.get_checkpoints("s1") == []
    assert any_store.list_migration_events() == []


def test_file_store_is_visible_to_a_second_handle(tmp_path):
    """Two processes share a FileStore; two handles is the same mechanism."""
    path = tmp_path / "shared.json"
    first = FileStore(path)
    first.insert_checkpoint(**CHECKPOINT)

    second = FileStore(path)
    assert len(second.get_checkpoints("s1")) == 1
    # And the constraint holds across handles, not just within one.
    assert second.insert_checkpoint(**CHECKPOINT) is False


def test_concurrent_writers_in_one_process_do_not_lose_rows(tmp_path):
    """Reads used to overwrite the table dict a writer was mid-transaction on,
    silently dropping appends."""
    import threading

    store = FileStore(tmp_path / "concurrent.json")
    errors: list[Exception] = []

    def write(n: int) -> None:
        try:
            for i in range(20):
                store.insert_migration_event(
                    session_id=f"s{n}",
                    worker_id=f"w{n}",
                    event="started",
                    from_machine=None,
                    to_machine="m",
                    step_at_event=i,
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def read() -> None:
        try:
            for _ in range(200):
                store.snapshot()
                store.get_checkpoints("s0")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(4)]
    threads += [threading.Thread(target=read) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    assert len(store.snapshot()["migration_log"]) == 80


def test_concurrent_writers_across_processes_do_not_lose_rows(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "multiproc.json"
    FileStore(path)
    script = (
        "import sys;from relay.store import FileStore;"
        "s=FileStore(sys.argv[1]);n=sys.argv[2];"
        "[s.insert_migration_event(session_id=n,worker_id=n,event='started',"
        "from_machine=None,to_machine='m',step_at_event=i) for i in range(15)]"
    )
    procs = [
        subprocess.Popen([sys.executable, "-c", script, str(path), f"p{n}"]) for n in range(4)
    ]
    for proc in procs:
        assert proc.wait(timeout=60) == 0

    assert len(FileStore(path).snapshot()["migration_log"]) == 60


def test_a_nested_write_does_not_discard_the_outer_one(tmp_path):
    """A signal handler firing mid-write is a nested transaction. It must join
    the one in progress, not re-read the file over its uncommitted work."""
    store = FileStore(tmp_path / "nested.json")

    with store._txn():
        store.tables["checkpoints"].append({**CHECKPOINT, "step_number": 1})
        # Re-entrant call, exactly as an eviction handler would make.
        store.insert_migration_event(
            session_id="s1",
            worker_id="w1",
            event="evicted",
            from_machine="m1",
            to_machine=None,
            step_at_event=1,
        )

    snapshot = store.snapshot()
    assert len(snapshot["checkpoints"]) == 1, "the outer write was lost"
    assert [e["event"] for e in snapshot["migration_log"]] == ["evicted"]
