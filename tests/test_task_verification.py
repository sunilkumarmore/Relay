"""Catching a provider that returns the wrong answer.

The only check available on arithmetic is to have somebody else do it and
compare, so these tests are mostly about being careful with what a disagreement
actually proves — and about not slashing an honest provider for a divergence it
did not cause.
"""

from __future__ import annotations

import random

import pytest

from relay.identity import Identity
from relay.store import MemoryStore
from relay.tasks import kernels, verify
from relay.tasks import model as task_model
from relay.tasks.executor import TaskExecutor
from relay.tasks.model import Task
from relay.tasks.queue import TaskQueue


class Liar(TaskExecutor):
    """A provider that signs a plausible but wrong answer.

    It signs honestly — the signature is valid, the output hash matches the
    output. Everything about it is well-formed except the number. That is the
    only kind of cheating the marketplace cannot detect from one result alone,
    and the only kind worth testing.
    """

    def execute(self, task: Task, *, at=None):
        honest = super().execute(task, at=at)
        if honest.status != task_model.RESULT_OK:
            return honest
        wrong = dict(honest.output)
        if "value" in wrong:
            wrong["value"] = float(wrong["value"]) + 1.0
        else:
            wrong["tampered"] = True
        return task_model.build_result(
            self.identity, task=task, output=wrong, duration_ms=honest.duration_ms
        )


def sum_task(consumer: Identity, job_id: str = "job") -> Task:
    return task_model.build_task(
        consumer,
        job_id=job_id,
        task_type=kernels.DATA_REDUCE,
        payload={"values": [1.0, 2.0, 3.0], "op": "sum"},
    )


def run_one(queue: TaskQueue, executor: TaskExecutor) -> bool:
    task = queue.claim(
        provider_node_id=executor.identity.node_id, task_types=[kernels.DATA_REDUCE]
    )
    if task is None:
        return False
    queue.complete(executor.execute(task), provider_node_id=executor.identity.node_id)
    return True


def completed(store: MemoryStore, task_id: str) -> Task:
    return Task.from_row(store.get_task(task_id))


# -- what may be audited ----------------------------------------------------


def test_only_deterministic_work_can_be_audited_this_way():
    """python_exec diverges between honest providers on patch level alone.
    Auditing it by hash comparison would punish people for updating."""
    consumer = Identity.generate()
    deterministic = sum_task(consumer)
    assert verify.auditable(deterministic)

    from relay.tasks.pyexec import register_default

    register_default()
    code_task = task_model.build_task(
        consumer, job_id="j", task_type=kernels.PYTHON_EXEC, payload={"code": "print(1)"}
    )
    assert not verify.auditable(code_task)


def test_comparing_a_non_deterministic_task_reports_that_rather_than_a_verdict():
    from relay.tasks.pyexec import register_default

    register_default()
    consumer = Identity.generate()
    task = task_model.build_task(
        consumer, job_id="j", task_type=kernels.PYTHON_EXEC, payload={"code": "print(1)"}
    )
    outcome = verify.compare(MemoryStore(), task=task)
    assert outcome.verdict == verify.VERDICT_NOT_AUDITABLE
    assert not outcome.is_conclusive


def test_an_audit_cannot_be_ordered_for_work_that_cannot_be_checked():
    from relay.tasks.pyexec import register_default

    register_default()
    consumer = Identity.generate()
    store = MemoryStore()
    queue = TaskQueue(store)
    task = task_model.build_task(
        consumer, job_id="j", task_type=kernels.PYTHON_EXEC, payload={"code": "print(1)"}
    )
    with pytest.raises(ValueError, match="not deterministic"):
        verify.queue_audit(queue, consumer, task)


def test_only_the_consumer_may_order_an_audit_of_its_own_work():
    store = MemoryStore()
    queue = TaskQueue(store)
    consumer = Identity.generate()
    task = queue.submit(sum_task(consumer))
    with pytest.raises(ValueError, match="ordered by the consumer"):
        verify.queue_audit(queue, Identity.generate(), task)


# -- the exclusion ----------------------------------------------------------


def test_a_provider_cannot_audit_its_own_answer():
    """Otherwise the check is the accused marking their own paper."""
    store = MemoryStore()
    queue = TaskQueue(store)
    consumer = Identity.generate()
    task = queue.submit(sum_task(consumer))
    worker = TaskExecutor(Identity.generate(), store)
    assert run_one(queue, worker)

    audit = verify.queue_audit(queue, consumer, completed(store, task.task_id))
    assert audit.excluded_provider == worker.identity.node_id
    assert queue.claim(
        provider_node_id=worker.identity.node_id, task_types=[kernels.DATA_REDUCE]
    ) is None

    other = TaskExecutor(Identity.generate(), store)
    assert run_one(queue, other)


def test_the_exclusion_is_enforced_by_the_store_not_only_by_the_filter():
    """A dishonest client can skip the filter. It cannot skip the write."""
    store = MemoryStore()
    queue = TaskQueue(store)
    consumer = Identity.generate()
    task = queue.submit(sum_task(consumer))
    worker = TaskExecutor(Identity.generate(), store)
    run_one(queue, worker)
    audit = verify.queue_audit(queue, consumer, completed(store, task.task_id))

    stolen = store.compare_and_set_task(
        audit.task_id,
        expect_status=task_model.STATUS_QUEUED,
        updates={"status": "leased", "lease_holder": worker.identity.node_id},
    )
    assert not stolen


# -- verdicts ---------------------------------------------------------------


def test_two_honest_providers_agree():
    store = MemoryStore()
    queue = TaskQueue(store)
    consumer = Identity.generate()
    task = queue.submit(sum_task(consumer))
    run_one(queue, TaskExecutor(Identity.generate(), store))
    audit = verify.queue_audit(queue, consumer, completed(store, task.task_id))
    run_one(queue, TaskExecutor(Identity.generate(), store))

    outcome = verify.compare(store, task=completed(store, task.task_id), audit_task_ids=[audit.task_id])
    assert outcome.verdict == verify.VERDICT_AGREED
    assert outcome.minority == []


def test_one_liar_against_one_honest_provider_is_not_blamed_on_either():
    """Two answers, no majority. Somebody is wrong and the evidence does not say
    who — and picking whoever answered first would make reputation a coin flip."""
    store = MemoryStore()
    queue = TaskQueue(store)
    consumer = Identity.generate()
    task = queue.submit(sum_task(consumer))
    run_one(queue, Liar(Identity.generate(), store))
    audit = verify.queue_audit(queue, consumer, completed(store, task.task_id))
    run_one(queue, TaskExecutor(Identity.generate(), store))

    outcome = verify.compare(store, task=completed(store, task.task_id), audit_task_ids=[audit.task_id])
    assert outcome.verdict == verify.VERDICT_INCONCLUSIVE
    assert outcome.minority == []
    assert not outcome.is_conclusive


def test_a_liar_outvoted_two_to_one_is_named():
    store = MemoryStore()
    queue = TaskQueue(store)
    consumer = Identity.generate()
    task = queue.submit(sum_task(consumer))
    liar = Liar(Identity.generate(), store)
    run_one(queue, liar)

    first_audit = verify.queue_audit(queue, consumer, completed(store, task.task_id))
    run_one(queue, TaskExecutor(Identity.generate(), store))
    second_audit = verify.queue_audit(queue, consumer, completed(store, task.task_id))
    run_one(queue, TaskExecutor(Identity.generate(), store))

    outcome = verify.compare(
        store,
        task=completed(store, task.task_id),
        audit_task_ids=[first_audit.task_id, second_audit.task_id],
    )
    assert outcome.verdict == verify.VERDICT_DIVERGED
    assert outcome.minority == [liar.identity.node_id]
    assert outcome.is_conclusive


def test_a_provider_cannot_outvote_others_by_answering_twice():
    """Otherwise a liar manufactures its own majority."""
    store = MemoryStore()
    queue = TaskQueue(store)
    consumer = Identity.generate()
    task = queue.submit(sum_task(consumer))
    liar = Liar(Identity.generate(), store)
    run_one(queue, liar)

    # The same liar files a second, identical result against the same task.
    original = completed(store, task.task_id)
    duplicate = task_model.build_result(
        liar.identity, task=original, output={"value": 7.0}, duration_ms=1
    )
    store.insert_task_result(duplicate.to_row())

    audit = verify.queue_audit(queue, consumer, original)
    run_one(queue, TaskExecutor(Identity.generate(), store))

    outcome = verify.compare(store, task=completed(store, task.task_id), audit_task_ids=[audit.task_id])
    assert outcome.verdict == verify.VERDICT_INCONCLUSIVE


def test_a_result_that_does_not_verify_is_not_counted_as_an_opinion():
    store = MemoryStore()
    queue = TaskQueue(store)
    consumer = Identity.generate()
    task = queue.submit(sum_task(consumer))
    run_one(queue, TaskExecutor(Identity.generate(), store))

    original = completed(store, task.task_id)
    forged = task_model.build_result(
        Identity.generate(), task=original, output={"value": 99.0}, duration_ms=1
    ).model_copy(update={"signature": "00" * 64})
    store.insert_task_result(forged.to_row())

    outcome = verify.compare(store, task=original)
    assert outcome.verdict == verify.VERDICT_AGREED


# -- sampling ---------------------------------------------------------------


def test_sampling_picks_roughly_the_configured_share():
    store = MemoryStore()
    queue = TaskQueue(store)
    consumer = Identity.generate()
    worker = TaskExecutor(Identity.generate(), store)
    for _ in range(200):
        queue.submit(sum_task(consumer))
        run_one(queue, worker)

    picked = verify.sample(store, rate=0.25, rng=random.Random(5))
    assert 25 <= len(picked) <= 75


def test_sampling_never_audits_an_audit():
    store = MemoryStore()
    queue = TaskQueue(store)
    consumer = Identity.generate()
    task = queue.submit(sum_task(consumer))
    run_one(queue, TaskExecutor(Identity.generate(), store))
    verify.queue_audit(queue, consumer, completed(store, task.task_id))
    run_one(queue, TaskExecutor(Identity.generate(), store))

    assert [t.task_id for t in verify.sample(store, rate=1.0)] == [task.task_id]


def test_evidence_carries_hashes_rather_than_the_disputed_bytes():
    store = MemoryStore()
    queue = TaskQueue(store)
    consumer = Identity.generate()
    task = queue.submit(sum_task(consumer))
    run_one(queue, TaskExecutor(Identity.generate(), store))
    outcome = verify.compare(store, task=completed(store, task.task_id))
    evidence = verify.evidence_for(store, outcome)
    assert "answers" in evidence
    assert "output" not in str(evidence)
