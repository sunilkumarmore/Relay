"""Command line for the task marketplace.

    python -m relay.tasks node                 run a provider on this machine
    python -m relay.tasks submit  --rows 600   order a multiplication
    python -m relay.tasks status  JOB_ID       how far along it is
    python -m relay.tasks collect JOB_ID       assemble and check the answer
    python -m relay.tasks demo                 the whole thing, in one process

`submit` and `collect` take a `--seed`, and that is not a toy affordance: it is
how the demo proves the distributed answer is *right*. Both sides regenerate the
same operands from the seed, so `collect` can multiply them on one machine and
compare against what the devices returned, bit for bit.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from typing import Any

from relay import config
from relay.identity import Identity
from relay.store import Store, store_from_env
from relay.tasks import matmul
from relay.tasks.operands import Matrix
from relay.tasks.queue import DEFAULT_LEASE_SECONDS, TaskQueue


def build_operands(rows: int, inner: int, cols: int, seed: int) -> tuple[Matrix, Matrix]:
    """Two matrices from one seed, reproducible on any machine.

    `random.Random(seed)` is specified by CPython and stable across versions and
    platforms, so the consumer and the checker agree without shipping the data.
    """
    rng = random.Random(seed)
    a = Matrix.from_rows([[rng.uniform(-1.0, 1.0) for _ in range(inner)] for _ in range(rows)])
    b = Matrix.from_rows([[rng.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(inner)])
    return a, b


def _store(args: argparse.Namespace) -> Store:
    config.load_env(getattr(args, "env", None))
    store = store_from_env(getattr(args, "env", None))
    if store is None:
        raise SystemExit(
            "No store configured. Set SUPABASE_URL and SUPABASE_KEY in .env, or "
            "RELAY_STORE=file with RELAY_STORE_PATH for a local run."
        )
    return store


def _identity(args: argparse.Namespace) -> Identity:
    return Identity.load_or_create(getattr(args, "key_file", None) or config.get("RELAY_KEY_FILE") or None)


# -- commands ---------------------------------------------------------------


def cmd_submit(args: argparse.Namespace) -> int:
    store = _store(args)
    identity = _identity(args)
    queue = TaskQueue(
        store, lease_seconds=args.lease_seconds, max_task_seconds=args.max_seconds
    )
    a, b = build_operands(args.rows, args.inner, args.cols, args.seed)
    job = matmul.submit_matmul(
        queue,
        identity,
        a=a,
        b=b,
        job_id=args.job_id,
        block_rows=args.block_rows,
        block_cols=args.block_cols,
        max_seconds=args.max_seconds,
    )
    print(f"job {job.job_id}")
    print(f"  {job.describe()}")
    print(f"  seed {args.seed} — keep it, `collect` needs it to check the answer")
    print("\nStart providers elsewhere:  python -m relay.tasks node")
    print(f"Then:                       python -m relay.tasks collect {job.job_id} --seed {args.seed}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    store = _store(args)
    job = matmul.load_job(store, args.job_id)
    state = matmul.progress(store, job)
    print(f"job {job.job_id}: {job.describe()}")
    print(f"  {state['completed']}/{state['blocks']} tiles ({state['percent']}%)")
    for status, count in sorted(state["by_status"].items()):
        print(f"    {status:<10} {count}")
    workers = {
        row.get("completed_by")
        for row in store.list_tasks(job_id=job.job_id, limit=100000)
        if row.get("completed_by")
    }
    if workers:
        print(f"  {len(workers)} device(s) contributed")
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    store = _store(args)
    job = matmul.load_job(store, args.job_id)
    deadline = time.monotonic() + args.wait
    while True:
        state = matmul.progress(store, job)
        if state["completed"] >= state["blocks"]:
            break
        if time.monotonic() >= deadline:
            print(
                f"Still {state['blocks'] - state['completed']} tile(s) outstanding after "
                f"{args.wait}s. Not assembling a partial answer.",
                file=sys.stderr,
            )
            return 1
        print(f"  {state['completed']}/{state['blocks']} tiles…", flush=True)
        time.sleep(args.poll)

    result = matmul.assemble(store, job)
    print(f"assembled {result.rows}x{result.cols}")

    if args.seed is not None:
        a, b = build_operands(job.rows, job.inner, job.cols, args.seed)
        expected = matmul.reference(a, b)
        if result.raw() == expected.raw():
            print("verified: bit-identical to the same multiplication on one machine")
        else:
            print("MISMATCH: the distributed answer differs from the local one", file=sys.stderr)
            return 1
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            import json

            json.dump(result.to_payload(), handle)
        print(f"written to {args.out}")
    return 0


def cmd_node(args: argparse.Namespace) -> int:
    from relay.tasks.node import run

    return run()


def cmd_demo(args: argparse.Namespace) -> int:
    """The whole thing in one process, including a device that dies.

    Not a mock: real signed tasks, real leases, real reaping. The only thing
    standing in for the network is that the devices share this process.
    """
    from datetime import timedelta

    from relay.store import MemoryStore
    from relay.tasks.executor import TaskExecutor
    from relay.tasks.queue import now_utc

    store = MemoryStore()
    queue = TaskQueue(store, lease_seconds=args.lease_seconds, max_task_seconds=args.max_seconds)
    consumer = Identity.generate()
    a, b = build_operands(args.rows, args.inner, args.cols, args.seed)

    print(f"\nOrdering a {args.rows}x{args.inner} @ {args.inner}x{args.cols} multiplication")
    job = matmul.submit_matmul(
        queue,
        consumer,
        a=a,
        b=b,
        block_rows=args.block_rows,
        block_cols=args.block_cols,
        max_seconds=args.max_seconds,
    )
    print(f"  {job.describe()}\n")

    devices = [TaskExecutor(Identity.generate(), store) for _ in range(args.devices)]
    names = [f"device-{i + 1}" for i in range(args.devices)]
    for name, device in zip(names, devices, strict=True):
        print(f"  {name}: {device.identity.node_id[:16]}…")

    doomed = devices[-1] if args.kill_one and len(devices) > 1 else None
    held: list[Any] = []

    if doomed is not None:
        held = [
            queue.claim(
                provider_node_id=doomed.identity.node_id, task_types=["matmul_block"]
            )
            for _ in range(min(args.kill_after, len(job.task_ids)))
        ]
        held = [t for t in held if t is not None]
        print(f"\n  {names[-1]} takes {len(held)} tile(s) and is then unplugged mid-job.")

    started = time.monotonic()
    completed_by: dict[str, int] = {}

    def drain(at=None) -> None:
        while True:
            idle = True
            for name, device in zip(names, devices, strict=True):
                if device is doomed:
                    continue
                task = queue.claim(
                    provider_node_id=device.identity.node_id,
                    task_types=["matmul_block"],
                    at=at,
                )
                if task is None:
                    continue
                idle = False
                result = device.execute(task, at=at)
                if queue.complete(result, provider_node_id=device.identity.node_id, at=at):
                    completed_by[name] = completed_by.get(name, 0) + 1
            if idle:
                return

    drain()
    state = matmul.progress(store, job)
    print(f"\n  {state['completed']}/{state['blocks']} tiles done by the surviving devices")

    if held:
        try:
            matmul.assemble(store, job)
            print("  ! assembled a partial job — this should not happen")
            return 1
        except matmul.AssemblyError as exc:
            print(f"  assembly refuses, correctly: {exc}")

        later = now_utc() + timedelta(seconds=args.lease_seconds + 60)
        reaped = queue.reap(at=later)
        print(f"\n  {names[-1]} never renewed. {len(reaped)} lease(s) expired and requeued.")
        drain(at=later)

    elapsed = time.monotonic() - started
    result = matmul.assemble(store, job)
    expected = matmul.reference(a, b)
    exact = result.raw() == expected.raw()

    print(f"\n  assembled {result.rows}x{result.cols} in {elapsed:.1f}s")
    for name, count in sorted(completed_by.items()):
        print(f"    {name}: {count} tile(s)")
    print(f"\n  bit-identical to the single-machine answer: {exact}")
    print(f"  queue counters: {queue.stats()}")
    return 0 if exact else 1


# -- wiring -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="relay.tasks", description=__doc__)
    parser.add_argument("--env", help="path to a .env file")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_shape(target: argparse.ArgumentParser) -> None:
        target.add_argument("--rows", type=int, default=240, help="rows of A")
        target.add_argument("--inner", type=int, default=240, help="columns of A / rows of B")
        target.add_argument("--cols", type=int, default=240, help="columns of B")
        target.add_argument("--seed", type=int, default=1)
        target.add_argument("--block-rows", type=int, default=None, help="rows of A per tile")
        target.add_argument("--block-cols", type=int, default=None, help="columns of B per tile")
        target.add_argument("--max-seconds", type=int, default=300)
        target.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)

    submit = sub.add_parser("submit", help="queue a distributed multiplication")
    add_shape(submit)
    submit.add_argument("--job-id", default=None)
    submit.add_argument("--key-file", default=None)
    submit.set_defaults(func=cmd_submit)

    status = sub.add_parser("status", help="how far along a job is")
    status.add_argument("job_id")
    status.set_defaults(func=cmd_status)

    collect = sub.add_parser("collect", help="assemble the answer and check it")
    collect.add_argument("job_id")
    collect.add_argument("--seed", type=int, default=None, help="verify against a local run")
    collect.add_argument("--wait", type=int, default=600, help="seconds to wait for stragglers")
    collect.add_argument("--poll", type=int, default=5)
    collect.add_argument("--out", default=None, help="write the result matrix here")
    collect.set_defaults(func=cmd_collect)

    node = sub.add_parser("node", help="run a provider on this machine")
    node.set_defaults(func=cmd_node)

    demo = sub.add_parser("demo", help="the whole thing in one process")
    add_shape(demo)
    demo.set_defaults(rows=120, inner=60, cols=60, block_rows=20, block_cols=30)
    demo.add_argument("--devices", type=int, default=3)
    demo.add_argument("--kill-one", action="store_true", default=True)
    demo.add_argument("--no-kill", dest="kill_one", action="store_false")
    demo.add_argument("--kill-after", type=int, default=2, help="tiles the doomed device takes")
    demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
