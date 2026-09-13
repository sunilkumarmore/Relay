"""Splitting a matrix multiplication across machines, and putting it back.

C = A @ B divides into a grid of tiles. Tile *(i, j)* needs row strip *i* of A
and column strip *j* of B, and nothing else — no tile needs any other tile's
answer. That independence is what makes the work distributable at all, and it is
why this is an honest demonstration rather than a contrived one: there is no
coordination between providers, no partial state to migrate, and a tile that
comes back late or twice costs nothing but the duplicate effort.

## Why both dimensions

Splitting only by rows is simpler, and it does not scale. Every task would need
the whole of B, so the smallest amount of data a device must hold and fetch is
the entire right operand — 141MB of float64 for the 4200x4200 job that takes an
hour on one core. Splitting both ways bounds it: a device holds one strip of
each, and the strips are sized to fit comfortably through a database row.

Every strip is a content-addressed operand, published once and referenced by
hash. A strip of A is reused by every tile in its row, a strip of B by every
tile in its column, and a device that has fetched one keeps it. Carrying them
inline instead would re-send the same numbers once per tile — hundreds of
gigabytes for a job that is only 282MB of actual operand.

## Choosing tile sizes

Larger tiles amortise per-task overhead; smaller tiles recover faster from a
dead provider, since a lost lease loses one tile's work, and keep each operand
strip small enough to move. A phone may vanish mid-job where a desktop will not,
so this leans small.
"""

from __future__ import annotations

import math
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

# Multiply-accumulates per second, measured at ~20M on a modern laptop core for
# strips a few hundred wide. Deliberately set well below that: underestimating
# makes tiles smaller, which costs a little overhead and buys faster recovery
# and better balance across a fleet of unequal devices. Overestimating makes
# tiles that overrun their deadline on the slowest device, which costs the work
# twice. Being wrong here makes tiles the wrong size, never the answer wrong.
ASSUMED_UNITS_PER_SECOND = 8_000_000

# Ceiling on one operand strip, in bytes before base64. Strips travel as a
# database row and have to sit in a phone's memory alongside the result.
MAX_STRIP_BYTES = 4 * 1024 * 1024

# What one tile should be worth, at the assumed rate. Sizing to a target
# duration rather than to a fraction of the deadline is what keeps a fleet
# balanced: a job cut into a handful of large tiles leaves most devices idle
# and, worse, finishes no faster than its slowest member. Small enough that
# losing one to a dead device is a rounding error; large enough that the
# per-task round trip is a small share of it.
TARGET_TILE_SECONDS = 5


class AssemblyError(RuntimeError):
    """The blocks that came back do not tile the result."""


@dataclass
class MatmulJob:
    job_id: str
    rows: int
    inner: int
    cols: int
    block_rows: int
    block_cols: int
    task_ids: list[str] = field(default_factory=list)

    @property
    def total_work_units(self) -> int:
        return self.rows * self.inner * self.cols

    def describe(self) -> str:
        down = math.ceil(self.rows / max(1, self.block_rows))
        across = math.ceil(self.cols / max(1, self.block_cols))
        return (
            f"{self.rows}x{self.inner} @ {self.inner}x{self.cols} "
            f"= {self.total_work_units:,} multiply-accumulates "
            f"in {down}x{across} tiles of {self.block_rows}x{self.block_cols}"
        )


def suggest_block_rows(
    *,
    inner: int,
    cols: int,
    max_seconds: int,
    units_per_second: int = ASSUMED_UNITS_PER_SECOND,
    max_strip_bytes: int = MAX_STRIP_BYTES,
) -> int:
    """How many rows of A fit comfortably inside one task's deadline *and* one
    operand strip.

    Three ceilings, and the lowest wins.

    The target one sizes a tile at `TARGET_TILE_SECONDS` of assumed work, which
    is what spreads a job across a fleet instead of handing it to one machine.

    The deadline one is a backstop: never more than a quarter of the time the
    task allows, because a tile that only just fits on the machine it was sized
    for will overrun on a slower one, and an overrun means a lapsed lease and
    the work done twice.

    The byte one exists because time says nothing about size. At inner=4200 a
    tile sized purely by the clock would want a 38MB strip of A, which is not
    something to push through a database row or hold on a phone.
    """
    per_row_units = max(1, inner * cols)
    by_target = max(1, units_per_second * TARGET_TILE_SECONDS) // per_row_units
    by_deadline = max(1, (units_per_second * max_seconds) // 4) // per_row_units

    per_row_bytes = max(1, inner * 8)
    by_size = max_strip_bytes // per_row_bytes

    return max(1, min(4096, by_target, by_deadline, by_size))


def suggest_block_cols(*, inner: int, max_strip_bytes: int = MAX_STRIP_BYTES) -> int:
    """How many columns of B fit in one operand strip.

    Bounded by bytes rather than by arithmetic: a strip is `inner x block_cols`
    float64, and it has to travel through a database row and sit in a phone's
    memory.
    """
    per_col = max(1, inner * 8)
    return max(1, max_strip_bytes // per_col)


def _row_strip(matrix: Matrix, offset: int, height: int) -> Matrix:
    start = offset * matrix.cols
    return Matrix(
        rows=height,
        cols=matrix.cols,
        data=array("d", matrix.data[start : start + height * matrix.cols]),
    )


def _col_strip(matrix: Matrix, offset: int, width: int) -> Matrix:
    data = array("d")
    for row in range(matrix.rows):
        start = row * matrix.cols + offset
        data.extend(matrix.data[start : start + width])
    return Matrix(rows=matrix.rows, cols=width, data=data)


def submit_matmul(
    queue: TaskQueue,
    identity: Identity,
    *,
    a: Matrix,
    b: Matrix,
    job_id: str | None = None,
    block_rows: int | None = None,
    block_cols: int | None = None,
    max_seconds: int = task_model.DEFAULT_MAX_SECONDS,
    max_price_credits: float = 0.0,
    max_attempts: int = task_model.DEFAULT_MAX_ATTEMPTS,
    ttl_seconds: int = task_model.DEFAULT_TASK_TTL_SECONDS,
) -> MatmulJob:
    """Publish the operand strips, then sign and queue one task per tile."""
    if a.cols != b.rows:
        raise ValueError(f"cannot multiply {a.rows}x{a.cols} by {b.rows}x{b.cols}")

    width = block_cols or suggest_block_cols(inner=a.cols)
    height = block_rows or suggest_block_rows(
        inner=a.cols, cols=min(width, b.cols), max_seconds=max_seconds
    )
    job = MatmulJob(
        job_id=job_id or str(uuid.uuid4()),
        rows=a.rows,
        inner=a.cols,
        cols=b.cols,
        block_rows=height,
        block_cols=width,
    )

    # Publish each strip once. A row strip of A is reused by every tile across
    # its row, a column strip of B by every tile down its column.
    row_strips: list[tuple[int, str, int]] = []
    for offset in range(0, a.rows, height):
        rows = min(height, a.rows - offset)
        strip = _row_strip(a, offset, rows)
        digest = strip.digest()
        queue.store.put_operand(digest, strip.to_payload())
        row_strips.append((offset, digest, rows))

    col_strips: list[tuple[int, str, int]] = []
    for offset in range(0, b.cols, width):
        cols = min(width, b.cols - offset)
        strip = _col_strip(b, offset, cols)
        digest = strip.digest()
        queue.store.put_operand(digest, strip.to_payload())
        col_strips.append((offset, digest, cols))

    for row_offset, a_hash, rows in row_strips:
        for col_offset, b_hash, cols in col_strips:
            task = task_model.build_task(
                identity,
                job_id=job.job_id,
                task_type=kernels.MATMUL_BLOCK,
                payload={
                    "a_hash": a_hash,
                    "a_rows": rows,
                    "a_cols": a.cols,
                    "b_hash": b_hash,
                    "b_rows": b.rows,
                    "b_cols": cols,
                    "row_offset": row_offset,
                    "col_offset": col_offset,
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
    covered = bytearray(job.rows * job.cols)

    for result in results.values():
        payload = result.output.get("c_block")
        if payload is None:
            raise AssemblyError(f"result for {result.task_id} carries no tile")
        try:
            tile = Matrix.from_payload(payload)
        except OperandError as exc:
            raise AssemblyError(f"result for {result.task_id} is malformed: {exc}") from exc
        row_offset = int(result.output.get("row_offset", -1))
        col_offset = int(result.output.get("col_offset", -1))
        if row_offset < 0 or row_offset + tile.rows > job.rows:
            raise AssemblyError(
                f"tile at row {row_offset} of height {tile.rows} does not fit a "
                f"{job.rows}-row result"
            )
        if col_offset < 0 or col_offset + tile.cols > job.cols:
            raise AssemblyError(
                f"tile at column {col_offset} of width {tile.cols} does not fit a "
                f"{job.cols}-column result"
            )
        for index in range(tile.rows):
            start = (row_offset + index) * job.cols + col_offset
            data[start : start + tile.cols] = tile.data[
                index * tile.cols : (index + 1) * tile.cols
            ]
            for column in range(col_offset, col_offset + tile.cols):
                covered[(row_offset + index) * job.cols + column] = 1

    missing = covered.count(0)
    if missing:
        first = covered.index(0)
        raise AssemblyError(
            f"{missing} of {job.rows * job.cols} cells never came back "
            f"(first missing at row {first // job.cols}, column {first % job.cols})"
        )
    return Matrix(rows=job.rows, cols=job.cols, data=data)


def reference(a: Matrix, b: Matrix) -> Matrix:
    """The same multiplication on one machine, for checking the distributed one.

    Uses the same kernel the providers use, so agreement proves the distribution
    and assembly were right — not that two different algorithms happen to agree.
    """
    kernel = kernels.get(kernels.MATMUL_BLOCK)
    operands = {a.digest(): a, b.digest(): b}
    payload = {
        "a_hash": a.digest(),
        "a_rows": a.rows,
        "a_cols": a.cols,
        "b_hash": b.digest(),
        "b_rows": b.rows,
        "b_cols": b.cols,
        "row_offset": 0,
        "col_offset": 0,
    }
    output = kernel.run(payload, lambda digest: operands[digest])
    return Matrix.from_payload(output["c_block"])


def progress(store: Store, job: MatmulJob) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in store.list_tasks(job_id=job.job_id):
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    done = counts.get(STATUS_COMPLETED, 0)
    return {
        "blocks": len(job.task_ids),
        "tiles": len(job.task_ids),
        "completed": done,
        "by_status": counts,
        "percent": round(100.0 * done / max(1, len(job.task_ids)), 1),
    }


def load_job(store: Store, job_id: str) -> MatmulJob:
    """Rebuild a job's shape from the tasks themselves.

    There is no jobs table, deliberately. Every fact about a matmul job — how
    tall the result is, how wide, which operand it multiplies by — is already
    stated in the tasks and signed by the consumer. A separate row recording the
    same facts could disagree with them, and then which one is the job?
    """
    rows = store.list_tasks(job_id=job_id, limit=100000)
    if not rows:
        raise AssemblyError(f"no tasks found for job {job_id}")

    tasks = [Task.from_row(row) for row in rows]
    blocks = [t for t in tasks if t.task_type == kernels.MATMUL_BLOCK]
    if not blocks:
        raise AssemblyError(f"job {job_id} holds no matmul blocks")

    height = 0
    width = 0
    inner = 0
    block_rows = 0
    block_cols = 0
    for task in blocks:
        payload = task.payload
        rows = int(payload["a_rows"])
        cols = int(payload["b_cols"])
        height = max(height, int(payload["row_offset"]) + rows)
        width = max(width, int(payload["col_offset"]) + cols)
        block_rows = max(block_rows, rows)
        block_cols = max(block_cols, cols)
        inner = int(payload["a_cols"])

    return MatmulJob(
        job_id=job_id,
        rows=height,
        inner=inner,
        cols=width,
        block_rows=block_rows,
        block_cols=block_cols,
        task_ids=[t.task_id for t in blocks],
    )
