"""The claim, tested against the operating system rather than a simulation.

A provider process is sent SIGKILL while it holds a lease — the closest thing to
a phone going into a tunnel: no shutdown, no message, no chance to hand the work
back. Other devices must finish the job, and the answer must be bit-identical to
the same multiplication done on one machine.

The sequence is deliberately ordered rather than timed. The victim runs alone
until the store shows it actually holding a lease, and only then is it killed;
the survivors start afterwards. A test that killed a process at a wall-clock
moment and hoped it was busy would pass whether or not anything was recovered,
which is the failure mode the first version of this file had.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from relay.identity import Identity
from relay.store import FileStore
from relay.tasks import matmul
from relay.tasks.cli import build_operands
from relay.tasks.queue import TaskQueue

# Short enough that a dead node's lease lapses inside the test, long enough that
# a live node renews before it does. Provider and store share this machine's
# clock, which is the only reason the slack can be this small.
LEASE_SECONDS = 6
MAX_TASK_SECONDS = 4
LEASE_SLACK = 2

# Sized so a block takes long enough to still be in flight when the kill lands,
# and the whole job is a couple of seconds of real arithmetic.
ROWS, INNER, COLS = 120, 300, 300
BLOCK_ROWS = 10
REPO_ROOT = Path(__file__).resolve().parent.parent


def node_env(store_path: Path, key_file: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "RELAY_STORE": "file",
            "RELAY_STORE_PATH": str(store_path),
            "RELAY_KEY_FILE": str(key_file),
            "RELAY_LEASE_SECONDS": str(LEASE_SECONDS),
            "RELAY_LEASE_SLACK_SECONDS": str(LEASE_SLACK),
            "RELAY_MAX_TASK_SECONDS": str(MAX_TASK_SECONDS),
            "RELAY_POLL_INTERVAL": "1",
            "RELAY_TASK_TYPES": "matmul_block",
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.pop("SUPABASE_URL", None)
    env.pop("SUPABASE_KEY", None)
    return env


def start_node(store_path: Path, key_file: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "relay.tasks", "node"],
        env=node_env(store_path, key_file),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(REPO_ROOT),
    )


def wait_for(predicate, *, timeout: float, interval: float = 0.1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = predicate()
        if found:
            return found
        time.sleep(interval)
    return None


@pytest.mark.slow
def test_a_killed_device_does_not_cost_the_job(tmp_path: Path) -> None:
    store_path = tmp_path / "relay.json"
    store = FileStore(str(store_path))
    queue = TaskQueue(
        store,
        lease_seconds=LEASE_SECONDS,
        max_task_seconds=MAX_TASK_SECONDS,
        lease_slack_seconds=LEASE_SLACK,
    )

    a, b = build_operands(ROWS, INNER, COLS, seed=99)
    job = matmul.submit_matmul(
        queue,
        Identity.generate(),
        a=a,
        b=b,
        block_rows=BLOCK_ROWS,
        max_seconds=MAX_TASK_SECONDS,
        ttl_seconds=600,
    )
    assert len(job.task_ids) == ROWS // BLOCK_ROWS

    # Knowing the victim's node id in advance is what makes the assertions
    # specific: we can name the task it was holding when it died.
    victim_key = tmp_path / "victim.key"
    victim_id = Identity.load_or_create(str(victim_key)).node_id

    processes: list[subprocess.Popen] = []
    try:
        victim = start_node(store_path, victim_key)
        processes.append(victim)

        def held_by_victim():
            return [
                row
                for row in store.list_tasks(job_id=job.job_id, limit=1000)
                if row.get("status") == "leased" and row.get("lease_holder") == victim_id
            ]

        in_flight = wait_for(held_by_victim, timeout=30)
        assert in_flight, "the victim never took a lease, so nothing would be recovered"
        abandoned = {row["task_id"] for row in in_flight}

        os.kill(victim.pid, signal.SIGKILL)
        victim.wait(timeout=10)

        # It really is gone, and the work really is stranded.
        stranded = {
            row["task_id"]
            for row in store.list_tasks(job_id=job.job_id, limit=1000)
            if row.get("status") == "leased" and row.get("lease_holder") == victim_id
        }
        assert stranded & abandoned

        for index in range(2):
            processes.append(start_node(store_path, tmp_path / f"survivor{index}.key"))

        done = wait_for(lambda: queue.is_job_done(job.job_id), timeout=120, interval=0.5)
        assert done, {
            row["task_id"][:8]: row["status"]
            for row in store.list_tasks(job_id=job.job_id, limit=1000)
        }
    finally:
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGKILL)
                process.wait(timeout=10)

    rows = {row["task_id"]: row for row in store.list_tasks(job_id=job.job_id, limit=1000)}
    assert all(row["status"] == "completed" for row in rows.values())

    # The specific claim: the dead node's work was reissued and finished by
    # someone else, having been attempted more than once.
    for task_id in abandoned:
        recovered = rows[task_id]
        assert recovered["completed_by"] != victim_id
        assert recovered["attempts"] >= 2, recovered

    # And nothing quietly settled for a partial answer.
    assert matmul.assemble(store, job).raw() == matmul.reference(a, b).raw()


@pytest.mark.slow
def test_the_demo_command_works(tmp_path: Path) -> None:
    """`python -m relay.tasks demo` is what a person runs first."""
    completed = subprocess.run(
        [
            sys.executable, "-m", "relay.tasks", "demo",
            "--rows", "40", "--inner", "20", "--cols", "20",
            "--block-rows", "10", "--devices", "3",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "bit-identical to the single-machine answer: True" in completed.stdout
    assert "assembly refuses, correctly" in completed.stdout
