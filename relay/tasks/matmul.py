"""Splitting a matrix multiplication across machines, and putting it back.

C = A @ B divides along the rows of A. Row *i* of C depends on row *i* of A and
on all of B, and on nothing else — no block needs any other block's answer. That
independence is what makes the work distributable at all, and it is why this is
the honest demonstration of the idea rather than a contrived one: there is no
coordination between providers, no partial state to migrate, and a block that
comes back late or twice costs nothing but the duplicate effort.

B is published once as a content-addressed operand and referenced by hash. The
rows of A ride inside their own task, because they differ per task and there
would be nothing to share.

## Choosing a block size

Larger blocks amortise the per-task overhead; smaller blocks recover faster from
a dead provider, because a lost lease loses one block's work. Since a phone may
vanish mid-job while a desktop will not, this leans small. `suggest_block_rows`
picks a size that keeps each block inside its deadline given a measured rate.
"""

from __future__ import annotations

import uuid
from array import array
from dataclasses import dataclass, field
from typing import Any

from relay.identity import Identity
from relay.store import Store
from relay.tasks import kernels
from relay.tasks import model as task_model
from relay.tasks.model import STATUS_COMPLETED, Task, TaskResult
from relay.tasks.operands import Matrix, OperandError
from relay.tasks.queue import TaskQueue

# A conservative floor for pure-Python multiply-accumulates per second. Used
# only to pick a block size; being wrong makes blocks the wrong size, not wrong.
ASSUMED_UNITS_PER_SECOND = 2_000_000


class AssemblyError(RuntimeError):
    """The blocks that came back do not tile the result."""


@dataclass
class MatmulJob:
    job_id: str
    rows: int
    inner: int
    cols: int
    b_hash: str
    block_rows: int
    task_ids: list[str] = field(default_factory=list)

    @property
    def total_work_units(self) -> int:
        return self.rows * self.inner * self.cols

    def describe(self) -> str:
        return (
            f"{self.rows}x{self.inner} @ {self.inner}x{self.cols} "
            f"= {self.total_work_units:,} multiply-accumulates "
            f"in {len(self.task_ids)} blocks of {self.block_rows} rows"
        )


def suggest_block_rows(
    *, inner: int, cols: int, max_seconds: int, units_per_second: int = ASSUMED_UNITS_PER_SECOND
) -> int:
    """How many rows of A fit comfortably inside one task's deadline.

    Aims at a quarter of the deadline rather than all of it. A block that only
    just fits on the machine it was sized for will overrun on a slower one, and
    an overrun means a lapsed lease and the work done twice.
    """
    per_row = max(1, inner * cols)
    budget_units = max(1, (units_per_second * max_seconds) // 4)
    return max(1, min(4096, budget_units // per_row))


def submit_matmul(
    queue: TaskQueue,
    identity: Identity,
    *,
    a: Matrix,
    b: Matrix,
    job_id: str | None = None,
    block_rows: int | None = None,
    max_seconds: int = task_model.DEFAULT_MAX_SECONDS,
    max_price_credits: float = 0.0,
    max_attempts: int = task_model.DEFAULT_MAX_ATTEMPTS,
    ttl_seconds: int = task_model.DEFAULT_TASK_TTL_SECONDS,
) -> MatmulJob:
    """Publish B, cut A into row blocks, sign and queue each one."""
    if a.cols != b.rows:
        raise ValueError(f"cannot multiply {a.rows}x{a.cols} by {b.rows}x{b.cols}")

    job = MatmulJob(
        job_id=job_id or str(uuid.uuid4()),
        rows=a.rows,
        inner=a.cols,
        cols=b.cols,
        b_hash=b.digest(),
        block_rows=block_rows
        or suggest_block_rows(inner=a.cols, cols=b.cols, max_seconds=max_seconds),
    )

    queue.store.put_operand(job.b_hash, b.to_payload())

    for offset in range(0, a.rows, job.block_rows):
        height = min(job.block_rows, a.rows - offset)
        start = offset * a.cols
        block = Matrix(
            rows=height,
            cols=a.cols,
            data=array("d", a.data[start : start + height * a.cols]),
        )
        task = task_model.build_task(
            identity,
            job_id=job.job_id,
            task_type=kernels.MATMUL_BLOCK,
            payload={
                "a_block": block.to_payload(),
                "b_hash": job.b_hash,
                "b_cols": b.cols,
                "row_offset": offset,
            },
            max_seconds=max_seconds,
            max_price_credits=max_price_credits,
            max_attempts=max_attempts,
            ttl_seconds=ttl_seconds,
        )
        queue.submit(task)
        job.task_ids.append(task.task_id)

    return job


def _accepted_results(store: Store, job: MatmulJob) -> dict[str, TaskResult]:
    """One trusted result per task.

    Where a task has several results — a reissued block that both providers
    finished — the one the queue recorded as closing the task wins, and any
    other is a candidate for the divergence check rather than for assembly.
    """
    tasks = {row["task_id"]: Task.from_row(row) for row in store.list_tasks(job_id=job.job_id)}
    chosen: dict[str, TaskResult] = {}
    for row in store.list_task_results(job_id=job.job_id):
        result = TaskResult.from_row(row)
        task = tasks.get(result.task_id)
        if task is None or task.status != STATUS_COMPLETED:
            continue
        if task.completed_by and result.provider_node_id != task.completed_by:
            continue
        if task_model.verify_result(result, task=task):
            continue
        chosen[result.task_id] = result
    return chosen


def assemble(store: Store, job: MatmulJob) -> Matrix:
    """Stitch the completed blocks into C, refusing anything that does not tile.

    The checks here are not defensive clutter. A block that lands at the wrong
    offset, or claims a height it does not have, would otherwise produce a
    plausible-looking matrix that is quietly wrong — the worst possible outcome
    for a computation someone is paying for.
    """
    results = _accepted_results(store, job)
    data = array("d", bytes(job.rows * job.cols * 8))
    covered = bytearray(job.rows)

    for result in results.values():
        payload = result.output.get("c_block")
        if payload is None:
            raise AssemblyError(f"result for {result.task_id} carries no block")
        try:
            block = Matrix.from_payload(payload)
        except OperandError as exc:
            raise AssemblyError(f"result for {result.task_id} is malformed: {exc}") from exc
        offset = int(result.output.get("row_offset", -1))
        if offset < 0 or offset + block.rows > job.rows:
            raise AssemblyError(
                f"block at row {offset} of height {block.rows} does not fit a "
                f"{job.rows}-row result"
            )
        if block.cols != job.cols:
            raise AssemblyError(
                f"block has {block.cols} columns, result needs {job.cols}"
            )
        start = offset * job.cols
        data[start : start + block.rows * job.cols] = block.data
        for row in range(offset, offset + block.rows):
            covered[row] = 1

    missing = [index for index, seen in enumerate(covered) if not seen]
    if missing:
        raise AssemblyError(
            f"{len(missing)} of {job.rows} rows never came back "
            f"(first missing row {missing[0]})"
        )
    return Matrix(rows=job.rows, cols=job.cols, data=data)


def reference(a: Matrix, b: Matrix) -> Matrix:
    """The same multiplication on one machine, for checking the distributed one.

    Uses the same kernel the providers use, so agreement proves the distribution
    and assembly were right — not that two different algorithms happen to agree.
    """
    kernel = kernels.get(kernels.MATMUL_BLOCK)
    payload = {
        "a_block": a.to_payload(),
        "b_hash": b.digest(),
        "b_cols": b.cols,
        "row_offset": 0,
    }
    output = kernel.run(payload, lambda _hash: b)
    return Matrix.from_payload(output["c_block"])


def progress(store: Store, job: MatmulJob) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in store.list_tasks(job_id=job.job_id):
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    done = counts.get(STATUS_COMPLETED, 0)
    return {
        "blocks": len(job.task_ids),
        "completed": done,
        "by_status": counts,
        "percent": round(100.0 * done / max(1, len(job.task_ids)), 1),
    }
