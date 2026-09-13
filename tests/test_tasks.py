"""Tasks, leases, and what happens when a device stops answering."""

from __future__ import annotations

import random
from datetime import timedelta

import pytest

from relay.identity import Identity
from relay.store import MemoryStore
from relay.tasks import kernels, matmul
from relay.tasks import model as task_model
from relay.tasks.consent import ConsentError, OperatorConsent
from relay.tasks.executor import ExecutionError, OperandCache, TaskExecutor
from relay.tasks.model import STATUS_FAILED, STATUS_QUEUED, Task
from relay.tasks.operands import Matrix, OperandError
from relay.tasks.queue import LeaseConfigError, TaskQueue, assert_lease_sane, now_utc


def random_matrix(rows: int, cols: int, seed: int) -> Matrix:
    rng = random.Random(seed)
    return Matrix.from_rows([[rng.uniform(-4, 4) for _ in range(cols)] for _ in range(rows)])


def reduce_task(identity: Identity, job_id: str = "job", **kwargs) -> Task:
    return task_model.build_task(
        identity,
        job_id=job_id,
        task_type=kernels.DATA_REDUCE,
        payload={"values": [1.0, 2.0, 3.0], "op": "sum"},
        **kwargs,
    )


# -- operands ---------------------------------------------------------------


def test_operand_survives_a_round_trip_through_json():
    matrix = random_matrix(5, 3, seed=1)
    restored = Matrix.from_payload(matrix.to_payload())
    assert restored.to_rows() == matrix.to_rows()
    assert restored.digest() == matrix.digest()


def test_operand_hash_covers_shape_not_only_numbers():
    """Without the shape in the hash, a 2x3 and a 3x2 holding the same values
    would share an address and a task could be served the wrong one."""
    flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    wide = Matrix.from_rows([flat[:3], flat[3:]])
    tall = Matrix.from_rows([flat[:2], flat[2:4], flat[4:]])
    assert wide.raw() == tall.raw()
    assert wide.digest() != tall.digest()


def test_operand_that_lies_about_its_size_is_refused():
    payload = random_matrix(3, 3, seed=2).to_payload()
    payload["rows"] = 4
    with pytest.raises(OperandError):
        Matrix.from_payload(payload)


# -- kernels ----------------------------------------------------------------


def tile_payload(a: Matrix, b: Matrix, *, row_offset: int = 0, col_offset: int = 0) -> dict:
    return {
        "a_hash": a.digest(),
        "a_rows": a.rows,
        "a_cols": a.cols,
        "b_hash": b.digest(),
        "b_rows": b.rows,
        "b_cols": b.cols,
        "row_offset": row_offset,
        "col_offset": col_offset,
    }


def resolver(*matrices: Matrix):
    table = {matrix.digest(): matrix for matrix in matrices}
    return lambda digest: table[digest]


def test_matmul_block_computes_the_right_answer():
    a = Matrix.from_rows([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    b = Matrix.from_rows([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    output = kernels.get(kernels.MATMUL_BLOCK).run(
        tile_payload(a, b), resolver(a, b)
    )
    assert Matrix.from_payload(output["c_block"]).to_rows() == [
        [27.0, 30.0, 33.0],
        [61.0, 68.0, 75.0],
        [95.0, 106.0, 117.0],
    ]


def test_matmul_is_reproducible_bit_for_bit():
    """The whole basis for verifying a provider by re-running its work. If this
    fails, redundant execution would slash honest providers."""
    a, b = random_matrix(9, 7, seed=3), random_matrix(7, 6, seed=4)
    payload = tile_payload(a, b)
    kernel = kernels.get(kernels.MATMUL_BLOCK)
    first = kernel.run(payload, resolver(a, b))
    second = kernel.run(payload, resolver(a, b))
    assert kernels.output_hash(first) == kernels.output_hash(second)
    assert Matrix.from_payload(first["c_block"]).raw() == Matrix.from_payload(second["c_block"]).raw()


def test_matmul_refuses_operands_that_do_not_conform():
    a = random_matrix(3, 4, seed=5)
    b = random_matrix(5, 2, seed=6)  # 5 rows against a's 4 columns
    with pytest.raises(kernels.TaskError):
        kernels.get(kernels.MATMUL_BLOCK).run(tile_payload(a, b), resolver(a, b))


def test_a_tile_whose_operand_is_not_the_shape_it_declares_is_refused():
    a, b = random_matrix(3, 4, seed=20), random_matrix(4, 5, seed=21)
    payload = tile_payload(a, b) | {"a_rows": 99}
    with pytest.raises(kernels.TaskError, match="resolves to"):
        kernels.get(kernels.MATMUL_BLOCK).run(payload, resolver(a, b))


def test_work_units_are_derived_from_the_payload():
    a, b = random_matrix(4, 5, seed=7), random_matrix(5, 6, seed=22)
    assert kernels.get(kernels.MATMUL_BLOCK).work_units(tile_payload(a, b)) == 4 * 5 * 6


def test_text_and_reduce_kernels():
    text = kernels.get(kernels.TEXT_TRANSFORM)
    assert text.run({"text": " Hi There ", "ops": ["strip", "upper"]}, lambda _h: None) == {
        "text": "HI THERE"
    }
    reduce_kernel = kernels.get(kernels.DATA_REDUCE)
    assert reduce_kernel.run({"values": [1.0, 2.0, 4.0], "op": "sum"}, lambda _h: None) == {
        "value": 7.0
    }
    with pytest.raises(kernels.TaskError):
        reduce_kernel.validate({"values": [1.0], "op": "median"})


def test_python_exec_is_not_claimed_to_be_deterministic():
    """Hash-comparing two honest runs of it diverges on Python patch level, so
    slashing on divergence would punish honest providers."""
    from relay.tasks.pyexec import PythonExec

    assert PythonExec().deterministic is False
    assert kernels.MATMUL_BLOCK in kernels.deterministic_types()


# -- signing ----------------------------------------------------------------


def test_a_well_formed_task_verifies():
    assert task_model.verify_task(reduce_task(Identity.generate())) == []


def test_tampering_with_the_billed_quantity_is_caught_twice():
    task = reduce_task(Identity.generate())
    inflated = task.model_copy(update={"work_units": 10_000})
    problems = task_model.verify_task(inflated)
    assert "task signature is not valid" in problems
    assert any("work units" in p for p in problems)


def test_a_task_whose_payload_was_swapped_is_caught():
    task = reduce_task(Identity.generate())
    swapped = task.model_copy(update={"payload": {"values": [9.0], "op": "sum"}})
    assert "payload does not match the hash the consumer signed" in task_model.verify_task(swapped)


def test_only_the_consumer_can_sign_its_own_task():
    with pytest.raises(task_model.TaskSignatureError):
        reduce_task(Identity.generate()).signed_by(Identity.generate())


def test_a_result_must_match_the_task_it_claims():
    consumer, provider = Identity.generate(), Identity.generate()
    task = reduce_task(consumer)
    result = task_model.build_result(
        provider, task=task, output={"value": 6.0}, duration_ms=3
    )
    assert task_model.verify_result(result, task=task) == []
    lying = result.model_copy(update={"output": {"value": 999.0}})
    assert "output does not match the hash the provider signed" in task_model.verify_result(
        lying, task=task
    )


# -- the queue --------------------------------------------------------------


def test_a_lease_shorter_than_its_task_refuses_to_start():
    """Darkbloom's root cause A: a 60s expiry guarding a 90s call. The work gets
    issued twice and billed twice, silently."""
    with pytest.raises(LeaseConfigError):
        assert_lease_sane(100, 300)
    assert assert_lease_sane(600, 300) is None


def test_two_providers_racing_for_one_task_produce_one_winner():
    store = MemoryStore()
    queue = TaskQueue(store)
    queue.submit(reduce_task(Identity.generate()))
    first = queue.claim(provider_node_id="a", task_types=[kernels.DATA_REDUCE])
    second = queue.claim(provider_node_id="b", task_types=[kernels.DATA_REDUCE])
    assert first is not None
    assert second is None


def test_a_device_that_stops_answering_has_its_work_reissued():
    store = MemoryStore()
    queue = TaskQueue(store)
    task = queue.submit(reduce_task(Identity.generate()))
    queue.claim(provider_node_id="doomed", task_types=[kernels.DATA_REDUCE])

    later = now_utc() + timedelta(seconds=2000)
    assert queue.reap(at=later) == [(task.task_id, "requeued")]

    recovered = queue.claim(
        provider_node_id="survivor", task_types=[kernels.DATA_REDUCE], at=later
    )
    assert recovered is not None
    assert recovered.attempts == 2


def test_a_task_that_kills_every_provider_eventually_fails():
    """Otherwise a poisonous task cycles through the fleet forever."""
    store = MemoryStore()
    queue = TaskQueue(store)
    task = queue.submit(reduce_task(Identity.generate(), max_attempts=2))
    moment = now_utc()
    for _ in range(2):
        queue.claim(provider_node_id="p", task_types=[kernels.DATA_REDUCE], at=moment)
        moment += timedelta(seconds=2000)
        queue.reap(at=moment)
    assert store.get_task(task.task_id)["status"] == STATUS_FAILED


def test_a_renewed_lease_is_not_reaped():
    store = MemoryStore()
    queue = TaskQueue(store, lease_seconds=900)
    queue.submit(reduce_task(Identity.generate()))
    claimed = queue.claim(provider_node_id="slow", task_types=[kernels.DATA_REDUCE])
    midway = now_utc() + timedelta(seconds=800)
    assert queue.renew(claimed.task_id, provider_node_id="slow", at=midway)
    assert queue.reap(at=midway + timedelta(seconds=100)) == []


def test_only_the_lease_holder_can_renew():
    store = MemoryStore()
    queue = TaskQueue(store)
    queue.submit(reduce_task(Identity.generate()))
    claimed = queue.claim(provider_node_id="mine", task_types=[kernels.DATA_REDUCE])
    assert not queue.renew(claimed.task_id, provider_node_id="someone-else")


def test_reporting_a_failure_requeues_faster_than_waiting_for_the_lease():
    store = MemoryStore()
    queue = TaskQueue(store)
    queue.submit(reduce_task(Identity.generate()))
    claimed = queue.claim(provider_node_id="p", task_types=[kernels.DATA_REDUCE])
    assert queue.report_failure(claimed.task_id, provider_node_id="p", error="out of memory") == (
        "requeued"
    )
    assert store.get_task(claimed.task_id)["status"] == STATUS_QUEUED


def test_a_result_arriving_after_the_lease_lapsed_is_recorded_but_not_counted():
    """The work is not wasted — a redundant result is what verification wants —
    but it is no longer this provider's task to close."""
    store = MemoryStore()
    queue = TaskQueue(store)
    consumer = Identity.generate()
    provider = Identity.generate()
    task = queue.submit(reduce_task(consumer))
    claimed = queue.claim(provider_node_id=provider.node_id, task_types=[kernels.DATA_REDUCE])
    queue.reap(at=now_utc() + timedelta(seconds=2000))

    late = task_model.build_result(
        provider, task=claimed, output={"value": 6.0}, duration_ms=5
    )
    assert not queue.complete(late, provider_node_id=provider.node_id)
    assert queue.stats()["completed_too_late"] == 1
    assert len(store.list_task_results(task_id=task.task_id)) == 1


def test_the_queue_refuses_to_accept_an_invalid_task():
    store = MemoryStore()
    queue = TaskQueue(store)
    tampered = reduce_task(Identity.generate()).model_copy(update={"work_units": 5})
    with pytest.raises(ValueError, match="invalid task"):
        queue.submit(tampered)


# -- consent ----------------------------------------------------------------


def test_python_exec_is_off_unless_the_operator_turns_it_on():
    consent = OperatorConsent()
    assert kernels.PYTHON_EXEC not in consent.allowed_task_types
    assert consent.accepts(
        task_type=kernels.MATMUL_BLOCK, consumer_node_id="x", payload_bytes=1, max_seconds=10
    )
    assert not consent.accepts(
        task_type=kernels.PYTHON_EXEC, consumer_node_id="x", payload_bytes=1, max_seconds=10
    )


def test_enabling_code_execution_warns_in_plain_words():
    warnings = OperatorConsent(allowed_task_types=(kernels.PYTHON_EXEC,)).warnings()
    assert any("NOT stop it reading files" in w for w in warnings)


def test_a_paused_node_finishes_nothing_new():
    consent = OperatorConsent(paused=True)
    with pytest.raises(ConsentError, match="paused"):
        consent.check(
            task_type=kernels.MATMUL_BLOCK, consumer_node_id="x", payload_bytes=1, max_seconds=1
        )


def test_an_operator_can_name_the_consumers_it_works_for():
    consent = OperatorConsent(allowed_consumers=frozenset({"friend"}))
    assert consent.accepts(
        task_type=kernels.MATMUL_BLOCK, consumer_node_id="friend", payload_bytes=1, max_seconds=1
    )
    assert not consent.accepts(
        task_type=kernels.MATMUL_BLOCK, consumer_node_id="stranger", payload_bytes=1, max_seconds=1
    )


# -- executor ---------------------------------------------------------------


def test_an_operand_that_does_not_hash_to_its_address_is_refused():
    """Whatever served this operand, it served these numbers, or we do not
    compute on it."""
    store = MemoryStore()
    honest = random_matrix(3, 3, seed=8)
    substitute = random_matrix(3, 3, seed=9)
    store.put_operand(honest.digest(), substitute.to_payload())
    executor = TaskExecutor(Identity.generate(), store)
    with pytest.raises(ExecutionError, match="refusing to compute"):
        executor.resolve(honest.digest())


def test_a_missing_operand_is_an_error_not_a_wrong_answer():
    executor = TaskExecutor(Identity.generate(), MemoryStore())
    with pytest.raises(ExecutionError, match="not available"):
        executor.resolve("0" * 64)


def test_the_operand_cache_serves_the_second_task_without_a_fetch():
    store = MemoryStore()
    b = random_matrix(4, 4, seed=10)
    store.put_operand(b.digest(), b.to_payload())
    executor = TaskExecutor(Identity.generate(), store)
    executor.resolve(b.digest())
    executor.resolve(b.digest())
    assert executor.cache.hits == 1
    assert executor.cache.misses == 1


def test_the_cache_does_not_grow_without_bound():
    cache = OperandCache(max_entries=2)
    for index in range(5):
        cache.put(f"h{index}", random_matrix(1, 1, seed=index))
    assert len(cache) == 2


def test_a_failing_task_still_produces_a_signed_result():
    store = MemoryStore()
    consumer, provider = Identity.generate(), Identity.generate()
    task = task_model.build_task(
        consumer,
        job_id="j",
        task_type=kernels.MATMUL_BLOCK,
        payload={
            "a_hash": "0" * 64,
            "a_rows": 2,
            "a_cols": 2,
            "b_hash": "1" * 64,
            "b_rows": 2,
            "b_cols": 2,
            "row_offset": 0,
            "col_offset": 0,
        },
    )
    result = TaskExecutor(provider, store).execute(task)
    assert result.status == task_model.RESULT_ERROR
    assert result.signature_is_valid()
    assert "not available" in result.error


# -- the whole thing --------------------------------------------------------


def drain(queue: TaskQueue, executors: list[TaskExecutor], *, at=None) -> None:
    while True:
        idle = True
        for executor in executors:
            task = queue.claim(
                provider_node_id=executor.identity.node_id,
                task_types=[kernels.MATMUL_BLOCK],
                at=at,
            )
            if task is None:
                continue
            idle = False
            queue.complete(
                executor.execute(task, at=at),
                provider_node_id=executor.identity.node_id,
                at=at,
            )
        if idle:
            return


def test_a_matmul_split_across_machines_equals_the_same_one_on_a_single_machine():
    store = MemoryStore()
    queue = TaskQueue(store)
    a, b = random_matrix(24, 8, seed=12), random_matrix(8, 6, seed=13)
    job = matmul.submit_matmul(queue, Identity.generate(), a=a, b=b, block_rows=5)
    assert len(job.task_ids) == 5

    drain(queue, [TaskExecutor(Identity.generate(), store) for _ in range(3)])

    assert matmul.assemble(store, job).raw() == matmul.reference(a, b).raw()


def test_a_device_dying_mid_job_costs_a_block_not_the_job():
    """The headline claim: start a long multiplication, lose a device part-way,
    and still get the right answer."""
    store = MemoryStore()
    queue = TaskQueue(store)
    a, b = random_matrix(20, 6, seed=14), random_matrix(6, 5, seed=15)
    job = matmul.submit_matmul(queue, Identity.generate(), a=a, b=b, block_rows=4)

    survivor = TaskExecutor(Identity.generate(), store)
    doomed = TaskExecutor(Identity.generate(), store)

    held = [
        queue.claim(provider_node_id=doomed.identity.node_id, task_types=[kernels.MATMUL_BLOCK])
        for _ in range(2)
    ]
    assert all(task is not None for task in held)

    drain(queue, [survivor])

    # Half the answer is missing, and assembly says so rather than inventing it.
    with pytest.raises(matmul.AssemblyError, match="never came back"):
        matmul.assemble(store, job)

    later = now_utc() + timedelta(seconds=2000)
    assert len(queue.reap(at=later)) == 2
    drain(queue, [survivor], at=later)

    assert matmul.assemble(store, job).raw() == matmul.reference(a, b).raw()
    assert queue.stats()["expired_requeued"] == 2


def test_a_job_missing_blocks_is_never_assembled_into_a_plausible_wrong_answer():
    store = MemoryStore()
    queue = TaskQueue(store)
    a, b = random_matrix(12, 4, seed=16), random_matrix(4, 3, seed=17)
    job = matmul.submit_matmul(queue, Identity.generate(), a=a, b=b, block_rows=4)
    executor = TaskExecutor(Identity.generate(), store)
    task = queue.claim(
        provider_node_id=executor.identity.node_id, task_types=[kernels.MATMUL_BLOCK]
    )
    queue.complete(executor.execute(task), provider_node_id=executor.identity.node_id)
    with pytest.raises(matmul.AssemblyError):
        matmul.assemble(store, job)


def test_tile_size_leaves_headroom_for_a_slower_machine():
    rows = matmul.suggest_block_rows(inner=100, cols=100, max_seconds=60)
    assert rows * 100 * 100 <= matmul.ASSUMED_UNITS_PER_SECOND * 60


def test_operand_strips_stay_small_enough_to_move():
    """At inner=4200 a tile sized purely by the clock wants a 38MB strip of A,
    which is not something to push through a database row or hold on a phone."""
    for inner in (240, 1200, 4200):
        cols = matmul.suggest_block_cols(inner=inner)
        rows = matmul.suggest_block_rows(inner=inner, cols=cols, max_seconds=300)
        assert rows * inner * 8 <= matmul.MAX_STRIP_BYTES
        assert inner * cols * 8 <= matmul.MAX_STRIP_BYTES


def test_a_big_job_is_cut_into_enough_tiles_to_spread_over_a_fleet():
    """Sized by a fraction of the deadline instead of a target duration, a
    2400-cube job came out as a handful of tiles and most devices sat idle."""
    inner = 2400
    cols = matmul.suggest_block_cols(inner=inner)
    rows = matmul.suggest_block_rows(inner=inner, cols=cols, max_seconds=300)
    tiles = -(-2400 // rows) * -(-2400 // cols)
    assert tiles >= 100


def test_a_job_tiles_in_both_directions():
    store = MemoryStore()
    queue = TaskQueue(store)
    a, b = random_matrix(20, 8, seed=30), random_matrix(8, 20, seed=31)
    job = matmul.submit_matmul(
        queue, Identity.generate(), a=a, b=b, block_rows=7, block_cols=6
    )
    assert len(job.task_ids) == 3 * 4

    drain(queue, [TaskExecutor(Identity.generate(), store) for _ in range(3)])
    assert matmul.assemble(store, job).raw() == matmul.reference(a, b).raw()


def test_assembly_reports_a_hole_in_the_middle_of_the_grid():
    """A tile missing from the interior must not leave zeros that look like an
    answer."""
    store = MemoryStore()
    queue = TaskQueue(store)
    a, b = random_matrix(12, 4, seed=32), random_matrix(4, 12, seed=33)
    job = matmul.submit_matmul(
        queue, Identity.generate(), a=a, b=b, block_rows=6, block_cols=6
    )
    executor = TaskExecutor(Identity.generate(), store)
    for _ in range(3):
        task = queue.claim(
            provider_node_id=executor.identity.node_id, task_types=[kernels.MATMUL_BLOCK]
        )
        queue.complete(executor.execute(task), provider_node_id=executor.identity.node_id)
    with pytest.raises(matmul.AssemblyError, match="never came back"):
        matmul.assemble(store, job)
