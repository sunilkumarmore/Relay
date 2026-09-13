"""Running one task on one machine.

The executor is the provider's side of the bargain: it checks that the work was
ordered by someone who signed for it, that its own operator agreed to run that
kind of work, fetches whatever operands the task refers to, runs the kernel, and
signs what came out.

Order matters here. Verification comes before consent, and consent comes before
a single operand is fetched — a node should never spend bandwidth on work it was
always going to refuse. And both come before execution, because the point of
this module is that a provider is not an open compute proxy: work runs because a
consumer with a committed budget asked for it in writing.

## The deadline

`max_seconds` is enforced hard for `python_exec`, which runs in a subprocess that
can be killed. For the arithmetic kernels it is a *soft* deadline: they run in
this process and Python cannot safely interrupt a running C-level loop. That is
deliberate rather than an oversight. An overrunning task is already handled by
the lease: the provider stops renewing, the lease lapses, and the queue reissues
the work elsewhere. Adding a hard kill would mean running every matmul in a
subprocess and copying its operands into it, which costs more than the problem.
An overrun is recorded on the result so it is visible rather than silent.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from datetime import UTC, datetime
from typing import Any

from relay.identity import Identity
from relay.store import Store
from relay.tasks import kernels
from relay.tasks import model as task_model
from relay.tasks.consent import ConsentError, OperatorConsent
from relay.tasks.model import RESULT_ERROR, RESULT_OK, Task, TaskResult
from relay.tasks.operands import Matrix, OperandError

# Operands are addressed by hash, so a cached entry can never be stale — the
# only question is whether it is worth the memory. Bounded because a provider
# serving several jobs would otherwise hold every operand it had ever seen.
DEFAULT_OPERAND_CACHE_ENTRIES = 8


class ExecutionError(RuntimeError):
    """The task could not be run. Distinct from the task running and failing."""


class OperandCache:
    """Least-recently-used cache of decoded operands.

    The reason a matmul job is affordable at all: every block of a job wants the
    same right operand, so fetching it once per provider instead of once per
    task is the difference between sending it a handful of times and sending it
    hundreds of times.
    """

    def __init__(self, max_entries: int = DEFAULT_OPERAND_CACHE_ENTRIES) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[str, Matrix] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, operand_hash: str) -> Matrix | None:
        found = self._entries.get(operand_hash)
        if found is None:
            self.misses += 1
            return None
        self._entries.move_to_end(operand_hash)
        self.hits += 1
        return found

    def put(self, operand_hash: str, matrix: Matrix) -> None:
        self._entries[operand_hash] = matrix
        self._entries.move_to_end(operand_hash)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)


class TaskExecutor:
    def __init__(
        self,
        identity: Identity,
        store: Store,
        *,
        consent: OperatorConsent | None = None,
        cache: OperandCache | None = None,
    ) -> None:
        self.identity = identity
        self.store = store
        self.consent = consent or OperatorConsent()
        self.cache = cache or OperandCache()

    # -- admission --------------------------------------------------------

    def payload_bytes(self, task: Task) -> int:
        return len(task_model.canonical_json(task.payload).encode("utf-8"))

    def admit(self, task: Task, *, at: datetime | None = None) -> None:
        """Raise unless this node should run this task. Cheap checks first."""
        problems = task_model.verify_task(task, at=at)
        if problems:
            raise ExecutionError(f"refusing task: {'; '.join(problems)}")
        self.consent.check(
            task_type=task.task_type,
            consumer_node_id=task.consumer_node_id,
            payload_bytes=self.payload_bytes(task),
            max_seconds=task.max_seconds,
        )

    def will_accept(self, task: Task, *, at: datetime | None = None) -> bool:
        try:
            self.admit(task, at=at)
        except (ExecutionError, ConsentError):
            return False
        return True

    # -- operands ---------------------------------------------------------

    def resolve(self, operand_hash: str) -> Matrix:
        """Fetch an operand and prove it is the one that was asked for.

        Checking the hash after fetching is not paranoia about the database. It
        is what stops a task's meaning depending on anything but its own bytes:
        whatever served this operand, it served these numbers or the call fails.
        """
        cached = self.cache.get(operand_hash)
        if cached is not None:
            return cached
        row = self.store.get_operand(operand_hash)
        if row is None:
            raise ExecutionError(f"operand {operand_hash[:12]}… is not available")
        try:
            matrix = Matrix.from_payload(dict(row["payload"]))
        except (KeyError, OperandError) as exc:
            raise ExecutionError(f"operand {operand_hash[:12]}… is malformed: {exc}") from exc
        if matrix.digest() != operand_hash:
            raise ExecutionError(
                f"operand served for {operand_hash[:12]}… hashes to "
                f"{matrix.digest()[:12]}… — refusing to compute on it"
            )
        self.cache.put(operand_hash, matrix)
        return matrix

    # -- running ----------------------------------------------------------

    def execute(self, task: Task, *, at: datetime | None = None) -> TaskResult:
        """Run a task and sign the outcome.

        A task that raises is still a signed result, with `status=error`. That
        matters: a provider that stays silent on a task it cannot run makes the
        consumer wait out the whole lease, while one that says so returns the
        work to the queue immediately.
        """
        self.admit(task, at=at)
        kernel = kernels.get(task.task_type)
        started = time.monotonic()
        try:
            output = kernel.run(task.payload, self.resolve)
            status, error = RESULT_OK, ""
        except (kernels.TaskError, OperandError, ExecutionError) as exc:
            output, status, error = {}, RESULT_ERROR, str(exc)
        except MemoryError:
            output, status, error = {}, RESULT_ERROR, "ran out of memory"
        duration_ms = int((time.monotonic() - started) * 1000)

        if status == RESULT_OK and duration_ms > task.max_seconds * 1000:
            # Visible rather than silent. The lease has very likely lapsed and
            # the queue will have reissued this work; `complete` reports that
            # back as `completed_too_late` rather than pretending it landed.
            error = (
                f"completed in {duration_ms}ms, over the {task.max_seconds}s this task "
                f"allowed — the lease has probably expired"
            )

        return task_model.build_result(
            self.identity,
            task=task,
            output=output,
            duration_ms=duration_ms,
            status=status,
            error=error,
            at=at or datetime.now(UTC),
        )

    def stats(self) -> dict[str, Any]:
        return {
            "operand_cache_entries": len(self.cache),
            "operand_cache_hits": self.cache.hits,
            "operand_cache_misses": self.cache.misses,
        }
