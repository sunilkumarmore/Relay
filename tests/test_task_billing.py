"""Paying for tasks, and slashing a provider that returned the wrong answer."""

from __future__ import annotations

import pytest

from relay import disputes
from relay.identity import Identity
from relay.ledger import Ledger
from relay.provider.offers import build_offer
from relay.receipts import Receipt
from relay.store import MemoryStore
from relay.tasks import billing, kernels, verify
from relay.tasks import model as task_model
from relay.tasks.executor import TaskExecutor
from relay.tasks.model import Task
from relay.tasks.queue import TaskQueue
from tests.test_task_verification import Liar, sum_task


def priced_offer(provider: Identity, **kwargs) -> object:
    params = {
        "endpoint_url": "",
        "task_types": (kernels.DATA_REDUCE, kernels.MATMUL_BLOCK),
        "price_per_task": 0.0,
        "price_per_mega_unit": 1000.0,  # the toy task is 3 units, so ~0.003
    }
    params.update(kwargs)
    return build_offer(provider, **params)


class Market:
    """A consumer with credits, a queue, and providers that bill for their work."""

    def __init__(self, funding: float = 100.0) -> None:
        self.store = MemoryStore()
        self.ledger = Ledger(self.store)
        self.queue = TaskQueue(self.store)
        self.consumer = Identity.generate()
        self.ledger.deposit(self.consumer.node_id, funding)

    def run(self, executor: TaskExecutor, offer) -> Receipt:
        task = self.queue.claim(
            provider_node_id=executor.identity.node_id, task_types=[kernels.DATA_REDUCE]
        )
        assert task is not None
        result = executor.execute(task)
        self.queue.complete(result, provider_node_id=executor.identity.node_id)
        receipt = billing.issue_receipt(
            executor.identity, task=task, result=result, offer=offer
        )
        return billing.accept_and_settle(
            self.store,
            self.ledger,
            self.consumer,
            receipt=receipt,
            task=task,
            result=result,
            offer=offer,
        )


# -- the happy path ---------------------------------------------------------


def test_a_completed_task_is_paid_out_of_the_job_hold():
    market = Market()
    provider = TaskExecutor(Identity.generate(), market.store)
    offer = priced_offer(provider.identity)

    market.queue.submit(sum_task(market.consumer, job_id="job"))
    billing.hold_for_job(
        market.ledger, consumer_node_id=market.consumer.node_id, job_id="job", credits=10.0
    )
    before = market.ledger.balance(provider.identity.node_id)

    receipt = market.run(provider, offer)

    assert receipt.is_acknowledged
    assert market.ledger.balance(provider.identity.node_id) > before
    assert receipt.amount_credits == pytest.approx(offer.task_price(3))
    market.ledger.check_invariant()


def test_unspent_budget_comes_back():
    market = Market()
    billing.hold_for_job(
        market.ledger, consumer_node_id=market.consumer.node_id, job_id="job", credits=10.0
    )
    assert market.ledger.balance(market.consumer.node_id) == pytest.approx(90.0)
    billing.release_unspent(
        market.ledger, consumer_node_id=market.consumer.node_id, job_id="job"
    )
    assert market.ledger.balance(market.consumer.node_id) == pytest.approx(100.0)


def test_a_job_cannot_commit_a_budget_it_does_not_have():
    market = Market(funding=1.0)
    with pytest.raises(billing.BillingError, match="cannot commit"):
        billing.hold_for_job(
            market.ledger,
            consumer_node_id=market.consumer.node_id,
            job_id="job",
            credits=500.0,
        )


def test_a_free_market_still_runs_the_whole_verification_path():
    """Credits switched off is a price of zero, not a different code path."""
    market = Market()
    provider = TaskExecutor(Identity.generate(), market.store)
    offer = billing.free_offer(provider.identity)
    market.queue.submit(sum_task(market.consumer, job_id="job"))
    receipt = market.run(provider, offer)
    assert receipt.is_acknowledged
    assert receipt.amount_credits == 0.0


# -- refusing to pay --------------------------------------------------------


def test_a_receipt_billing_more_units_than_the_task_holds_is_refused():
    market = Market()
    provider = TaskExecutor(Identity.generate(), market.store)
    offer = priced_offer(provider.identity)
    task = market.queue.submit(sum_task(market.consumer, job_id="job"))
    billing.hold_for_job(
        market.ledger, consumer_node_id=market.consumer.node_id, job_id="job", credits=10.0
    )
    claimed = market.queue.claim(
        provider_node_id=provider.identity.node_id, task_types=[kernels.DATA_REDUCE]
    )
    result = provider.execute(claimed)

    honest = billing.issue_receipt(
        provider.identity, task=claimed, result=result, offer=offer
    )
    inflated = honest.model_copy(
        update={"work_units": 9_000_000, "amount_credits": 9000.0}
    ).issued_by(provider.identity)

    with pytest.raises(billing.BillingError, match="work units"):
        billing.accept_and_settle(
            market.store,
            market.ledger,
            market.consumer,
            receipt=inflated,
            task=Task.from_row(market.store.get_task(task.task_id)),
            result=result,
            offer=offer,
        )
    market.ledger.check_invariant()


def test_a_receipt_charging_more_than_the_offer_is_refused():
    market = Market()
    provider = TaskExecutor(Identity.generate(), market.store)
    offer = priced_offer(provider.identity)
    task = market.queue.submit(sum_task(market.consumer, job_id="job"))
    claimed = market.queue.claim(
        provider_node_id=provider.identity.node_id, task_types=[kernels.DATA_REDUCE]
    )
    result = provider.execute(claimed)
    overcharged = billing.issue_receipt(
        provider.identity, task=claimed, result=result, offer=offer
    ).model_copy(update={"amount_credits": 50.0}).issued_by(provider.identity)

    with pytest.raises(billing.BillingError, match="does not match"):
        billing.accept_and_settle(
            market.store,
            market.ledger,
            market.consumer,
            receipt=overcharged,
            task=Task.from_row(market.store.get_task(task.task_id)),
            result=result,
            offer=offer,
        )


def test_a_failed_task_is_not_billable():
    market = Market()
    provider = Identity.generate()
    task = sum_task(market.consumer)
    failed = task_model.build_result(
        provider, task=task, output={}, duration_ms=1, status=task_model.RESULT_ERROR,
        error="boom",
    )
    with pytest.raises(billing.BillingError, match="not billable"):
        billing.issue_receipt(
            provider, task=task, result=failed, offer=priced_offer(provider)
        )


# -- divergence -------------------------------------------------------------


def build_divergent_market(honest_count: int) -> tuple[Market, Liar, list[Receipt], str]:
    market = Market()
    billing.hold_for_job(
        market.ledger, consumer_node_id=market.consumer.node_id, job_id="job", credits=50.0
    )
    liar = Liar(Identity.generate(), market.store)
    market.ledger.deposit(liar.identity.node_id, 20.0)

    task = market.queue.submit(sum_task(market.consumer, job_id="job"))
    receipts = [market.run(liar, priced_offer(liar.identity))]

    for _ in range(honest_count):
        original = Task.from_row(market.store.get_task(task.task_id))
        verify.queue_audit(market.queue, market.consumer, original)
        honest = TaskExecutor(Identity.generate(), market.store)
        receipts.append(market.run(honest, priced_offer(honest.identity)))

    return market, liar, receipts, task.task_id


def test_a_liar_outvoted_two_to_one_is_disputed_and_slashed():
    market, liar, _receipts, task_id = build_divergent_market(honest_count=2)
    before = market.ledger.balance(liar.identity.node_id)

    outcome = verify.compare(
        market.store,
        task=Task.from_row(market.store.get_task(task_id)),
        audit_task_ids=[
            row["task_id"]
            for row in market.store.list_tasks(job_id="job")
            if row.get("audit_of") == task_id
        ],
    )
    assert outcome.minority == [liar.identity.node_id]

    opened = billing.open_divergence_dispute(
        market.store, divergence=outcome, opened_by=market.consumer.node_id
    )
    assert len(opened) == 1

    resolved = disputes.adjudicate_open(market.store, market.ledger)
    assert [d.status for d in resolved] == [disputes.UPHELD]
    assert market.ledger.balance(liar.identity.node_id) < before
    market.ledger.check_invariant()


def test_a_two_way_disagreement_slashes_nobody():
    """Somebody is wrong and the evidence does not say who."""
    market, liar, _receipts, task_id = build_divergent_market(honest_count=1)
    before = market.ledger.balance(liar.identity.node_id)

    outcome = verify.compare(
        market.store,
        task=Task.from_row(market.store.get_task(task_id)),
        audit_task_ids=[
            row["task_id"]
            for row in market.store.list_tasks(job_id="job")
            if row.get("audit_of") == task_id
        ],
    )
    assert outcome.verdict == verify.VERDICT_INCONCLUSIVE
    assert billing.open_divergence_dispute(
        market.store, divergence=outcome, opened_by=market.consumer.node_id
    ) == []
    assert market.ledger.balance(liar.identity.node_id) == pytest.approx(before)


def test_the_adjudicator_recomputes_rather_than_trusting_the_complainant():
    """The evidence was filed by one of the parties. Whoever opened the dispute
    does not get to say who lost it."""
    market, liar, receipts, task_id = build_divergent_market(honest_count=2)
    honest_receipt = next(
        r for r in receipts if r.provider_node_id != liar.identity.node_id
    )
    before = market.ledger.balance(honest_receipt.provider_node_id)

    # A malicious consumer accuses an honest provider, with fabricated evidence.
    disputes.open_dispute(
        market.store,
        receipt_id=honest_receipt.receipt_id,
        opened_by=market.consumer.node_id,
        reason=disputes.REASON_OUTPUT_DIVERGENCE,
        evidence={"minority": [honest_receipt.provider_node_id], "verdict": "diverged"},
    )
    resolved = disputes.adjudicate_open(market.store, market.ledger)
    assert [d.status for d in resolved] == [disputes.REJECTED]
    assert "majority" in str(resolved[0].evidence)
    assert market.ledger.balance(honest_receipt.provider_node_id) == pytest.approx(before)


def test_divergence_on_a_non_deterministic_type_is_never_adjudicated():
    from relay.tasks.pyexec import register_default

    register_default()
    market = Market()
    provider = Identity.generate()
    task = task_model.build_task(
        market.consumer,
        job_id="job",
        task_type=kernels.PYTHON_EXEC,
        payload={"code": "print(1)"},
    )
    market.store.enqueue_task(task.to_row())
    result = task_model.build_result(
        provider, task=task, output={"stdout": "1\n"}, duration_ms=1
    )
    receipt = billing.issue_receipt(
        provider, task=task, result=result, offer=priced_offer(provider)
    )
    market.store.upsert_receipt(receipt.to_row())

    disputes.open_dispute(
        market.store,
        receipt_id=receipt.receipt_id,
        opened_by=market.consumer.node_id,
        reason=disputes.REASON_OUTPUT_DIVERGENCE,
    )
    resolved = disputes.adjudicate_open(market.store, market.ledger)
    assert [d.status for d in resolved] == [disputes.UNADJUDICATED]
    assert "not deterministic" in str(resolved[0].evidence)
