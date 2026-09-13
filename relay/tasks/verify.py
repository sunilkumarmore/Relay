"""Checking a provider by having someone else do the same work.

There is no way to look at a number and tell whether it is the number the
arithmetic produces. The only check available is to run the work again somewhere
else and compare — which is affordable precisely because it does not have to
happen often. A provider that cheats on one task in ten will be caught by an
audit rate of one in twenty soon enough, and the cost is a few percent of the
fleet's capacity rather than doubling everything.

## Only deterministic task types

An audit compares output hashes. That is only meaningful where two honest
machines are *required* to produce identical bytes, which is true of
`matmul_block`, `text_transform` and `data_reduce` and false of `python_exec`.
Auditing `python_exec` this way would report divergence between two honest
providers running different Python patch releases, and slashing on that would
punish people for keeping their machines up to date. So `sample` skips
non-deterministic types entirely rather than "auditing them leniently" — a check
that cannot distinguish cheating from patch drift is not a lenient check, it is
a broken one.

## What a divergence does and does not prove

A mismatch proves the two providers disagree. It does not, by itself, say which
one is wrong. With two results and no third opinion the honest reading is "one
of these is wrong and we do not know which", so `adjudicate` asks for a third
run before it blames anybody. Only a provider in the minority of three is
slashed. Getting this wrong in the other direction — blaming whoever answered
first — would make an honest provider's reputation a coin flip.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from relay.identity import Identity
from relay.store import Store
from relay.tasks import kernels
from relay.tasks import model as task_model
from relay.tasks.model import STATUS_COMPLETED, Task, TaskResult
from relay.tasks.queue import TaskQueue

# One task in twenty. Low enough that verification costs a few percent of
# capacity, high enough that sustained cheating is found quickly.
DEFAULT_AUDIT_RATE = 0.05

REASON_OUTPUT_DIVERGENCE = "output_divergence"

VERDICT_AGREED = "agreed"
VERDICT_DIVERGED = "diverged"
VERDICT_INCONCLUSIVE = "inconclusive"
VERDICT_NOT_AUDITABLE = "not_auditable"


@dataclass
class Divergence:
    """Two or more providers, the same signed task, different answers."""

    task_id: str
    verdict: str
    by_hash: dict[str, list[str]]
    minority: list[str]
    majority_hash: str = ""

    @property
    def is_conclusive(self) -> bool:
        return self.verdict == VERDICT_DIVERGED and bool(self.minority)

    def describe(self) -> str:
        if self.verdict == VERDICT_AGREED:
            return f"{len(next(iter(self.by_hash.values()), []))} providers agree"
        if self.verdict == VERDICT_NOT_AUDITABLE:
            return "task type is not deterministic, so disagreement proves nothing"
        if self.verdict == VERDICT_INCONCLUSIVE:
            return (
                "providers disagree and there is no majority — one of them is wrong "
                "and the evidence does not say which"
            )
        return (
            f"{len(self.minority)} provider(s) disagree with the majority "
            f"{self.majority_hash[:12]}…"
        )


def auditable(task: Task) -> bool:
    """Whether re-running this task somewhere else would prove anything."""
    if not task.deterministic:
        return False
    try:
        return kernels.get(task.task_type).deterministic
    except kernels.TaskError:
        return False


def sample(
    store: Store,
    *,
    job_id: str | None = None,
    rate: float = DEFAULT_AUDIT_RATE,
    rng: random.Random | None = None,
    limit: int = 100000,
) -> list[Task]:
    """Pick completed, auditable tasks to re-run.

    Sampling is random rather than round-robin on purpose: a predictable audit
    is one a dishonest provider can answer honestly exactly when it is being
    watched.
    """
    chooser = rng or random.SystemRandom()
    picked: list[Task] = []
    for row in store.list_tasks(job_id=job_id, status=STATUS_COMPLETED, limit=limit):
        task = Task.from_row(row)
        if task.audit_of or not auditable(task):
            continue
        if chooser.random() < rate:
            picked.append(task)
    return picked


def queue_audit(
    queue: TaskQueue,
    identity: Identity,
    task: Task,
    *,
    ttl_seconds: int = task_model.DEFAULT_TASK_TTL_SECONDS,
) -> Task:
    """Queue the same work again, barred to the provider that already did it.

    The audit is a task in its own right: signed by the consumer, paid for like
    any other, and claimed by whoever is free. It has to be — a provider that
    could tell an audit from ordinary work would only cheat on the latter.
    """
    if not auditable(task):
        raise ValueError(f"{task.task_type} is not deterministic; auditing it proves nothing")
    if identity.node_id != task.consumer_node_id:
        raise ValueError("an audit must be ordered by the consumer that ordered the work")

    audit = task_model.build_task(
        identity,
        job_id=task.job_id,
        task_type=task.task_type,
        payload=task.payload,
        max_seconds=task.max_seconds,
        max_price_credits=task.max_price_credits,
        max_attempts=task.max_attempts,
        ttl_seconds=ttl_seconds,
    )
    queued = queue.submit(audit)
    # Written after submission because these are queue state, not signed terms.
    queue.store.compare_and_set_task(
        queued.task_id,
        expect_status=task_model.STATUS_QUEUED,
        updates={
            "audit_of": task.task_id,
            "excluded_provider": task.completed_by or task.lease_holder,
        },
    )
    return queued.model_copy(
        update={"audit_of": task.task_id, "excluded_provider": task.completed_by}
    )


def _trusted_results(store: Store, task_ids: list[str]) -> list[TaskResult]:
    """Results that are what they claim to be, one per provider.

    Only the first result from any given provider counts. Otherwise a provider
    that posted the same answer twice would look like two providers agreeing,
    and could out-vote an honest minority on its own.
    """
    tasks = {}
    for task_id in task_ids:
        row = store.get_task(task_id)
        if row is not None:
            tasks[task_id] = Task.from_row(row)

    seen: set[str] = set()
    trusted: list[TaskResult] = []
    for task_id in task_ids:
        task = tasks.get(task_id)
        if task is None:
            continue
        for row in store.list_task_results(task_id=task_id):
            result = TaskResult.from_row(row)
            if result.status != task_model.RESULT_OK:
                continue
            if result.provider_node_id in seen:
                continue
            if task_model.verify_result(result, task=task):
                continue
            seen.add(result.provider_node_id)
            trusted.append(result)
    return trusted


def compare(store: Store, *, task: Task, audit_task_ids: list[str] | None = None) -> Divergence:
    """Read every trustworthy answer to this task and see whether they agree."""
    ids = [task.task_id, *(audit_task_ids or [])]
    if not auditable(task):
        return Divergence(
            task_id=task.task_id, verdict=VERDICT_NOT_AUDITABLE, by_hash={}, minority=[]
        )

    by_hash: dict[str, list[str]] = {}
    for result in _trusted_results(store, ids):
        by_hash.setdefault(result.output_hash, []).append(result.provider_node_id)

    if len(by_hash) <= 1:
        return Divergence(
            task_id=task.task_id,
            verdict=VERDICT_AGREED,
            by_hash=by_hash,
            minority=[],
            majority_hash=next(iter(by_hash), ""),
        )

    ranked = sorted(by_hash.items(), key=lambda item: len(item[1]), reverse=True)
    top_hash, top_providers = ranked[0]
    runner_up = ranked[1]
    if len(top_providers) == len(runner_up[1]):
        # Two against two, or one against one. Somebody is wrong and the
        # evidence does not say who. Blaming either would be a coin flip.
        return Divergence(
            task_id=task.task_id, verdict=VERDICT_INCONCLUSIVE, by_hash=by_hash, minority=[]
        )

    minority = [
        provider
        for digest, providers in ranked[1:]
        for provider in providers
        if digest != top_hash
    ]
    return Divergence(
        task_id=task.task_id,
        verdict=VERDICT_DIVERGED,
        by_hash=by_hash,
        minority=minority,
        majority_hash=top_hash,
    )


def evidence_for(store: Store, divergence: Divergence) -> dict[str, Any]:
    """What a human or an adjudicator needs to see, without the payloads.

    Output bytes are deliberately left out. Both parties signed their hashes, so
    the hashes are the evidence; copying megabytes of disputed matrix into a
    dispute row would make the record expensive and no more convincing.
    """
    return {
        "task_id": divergence.task_id,
        "verdict": divergence.verdict,
        "answers": {digest: sorted(providers) for digest, providers in divergence.by_hash.items()},
        "majority_hash": divergence.majority_hash,
        "minority": sorted(divergence.minority),
        "basis": (
            "deterministic task type: two honest providers must produce identical bytes"
        ),
    }
