"""The ledger. Every transaction balances, or it is not recorded."""

from __future__ import annotations

import pytest

from relay.ledger import (
    DEFAULT_HOLD_TTL_SECONDS,
    InsufficientFunds,
    Ledger,
    LedgerError,
    available,
    held,
    sweep_expired_holds,
)
from relay.store import MemoryStore


@pytest.fixture
def ledger():
    return Ledger(MemoryStore())


CONSUMER = "c" * 64
PROVIDER = "p" * 64


def test_new_node_has_nothing(ledger):
    assert ledger.balance(CONSUMER) == 0.0
    assert ledger.held_balance(CONSUMER) == 0.0


def test_deposit_increases_balance_and_still_sums_to_zero(ledger):
    ledger.deposit(CONSUMER, 10.0)
    assert ledger.balance(CONSUMER) == 10.0
    ledger.check_invariant()


def test_hold_moves_credits_out_of_reach_without_destroying_them(ledger):
    ledger.deposit(CONSUMER, 10.0)
    ledger.hold(CONSUMER, 4.0, "job-1")

    assert ledger.balance(CONSUMER) == 6.0, "held credits are not spendable"
    assert ledger.held_balance(CONSUMER, "job-1") == 4.0
    assert ledger.balance(CONSUMER) + ledger.held_balance(CONSUMER) == 10.0
    ledger.check_invariant()


def test_cannot_hold_more_than_you_have(ledger):
    ledger.deposit(CONSUMER, 1.0)
    with pytest.raises(InsufficientFunds):
        ledger.hold(CONSUMER, 5.0, "job-1")
    assert ledger.balance(CONSUMER) == 1.0
    ledger.check_invariant()


def test_settlement_moves_credits_from_hold_to_provider(ledger):
    ledger.deposit(CONSUMER, 10.0)
    ledger.hold(CONSUMER, 4.0, "job-1")
    ledger.settle(CONSUMER, PROVIDER, 1.5, receipt_id="r1", job_id="job-1")

    assert ledger.balance(PROVIDER) == 1.5
    assert ledger.held_balance(CONSUMER, "job-1") == 2.5
    assert ledger.balance(CONSUMER) == 6.0, "settling spends the hold, not the balance"
    ledger.check_invariant()


def test_releasing_returns_the_unspent_hold(ledger):
    ledger.deposit(CONSUMER, 10.0)
    ledger.hold(CONSUMER, 4.0, "job-1")
    ledger.settle(CONSUMER, PROVIDER, 1.5, receipt_id="r1", job_id="job-1")

    returned = ledger.release_remaining(CONSUMER, "job-1")
    assert returned == 2.5
    assert ledger.balance(CONSUMER) == 8.5, "10 deposited, 1.5 spent"
    assert ledger.held_balance(CONSUMER) == 0.0
    ledger.check_invariant()


def test_a_settlement_cannot_be_replayed(ledger):
    """The idempotency key is the receipt, so a retry cannot pay twice."""
    ledger.deposit(CONSUMER, 10.0)
    ledger.hold(CONSUMER, 5.0, "job-1")

    first = ledger.settle(CONSUMER, PROVIDER, 2.0, receipt_id="r1", job_id="job-1")
    second = ledger.settle(CONSUMER, PROVIDER, 2.0, receipt_id="r1", job_id="job-1")

    assert first != ""
    assert second == "", "the replay was recognised"
    assert ledger.balance(PROVIDER) == 2.0
    ledger.check_invariant()


def test_holding_twice_for_one_job_is_idempotent(ledger):
    ledger.deposit(CONSUMER, 10.0)
    ledger.hold(CONSUMER, 3.0, "job-1")
    ledger.hold(CONSUMER, 3.0, "job-1")
    assert ledger.held_balance(CONSUMER, "job-1") == 3.0


def test_holds_are_per_job(ledger):
    """One job finishing must not release another job's committed budget."""
    ledger.deposit(CONSUMER, 10.0)
    ledger.hold(CONSUMER, 3.0, "job-a")
    ledger.hold(CONSUMER, 4.0, "job-b")

    ledger.release_remaining(CONSUMER, "job-a")

    assert ledger.held_balance(CONSUMER, "job-b") == 4.0
    assert ledger.held_balance(CONSUMER, "job-a") == 0.0
    assert ledger.balance(CONSUMER) == 6.0
    ledger.check_invariant()


def test_an_unbalanced_transaction_is_refused(ledger):
    from relay.ledger import Entry

    with pytest.raises(LedgerError, match="does not balance|Unbalanced"):
        ledger._post("deposit", [Entry(available(CONSUMER), debit=5.0), Entry("world", credit=3.0)])
    assert ledger.balance(CONSUMER) == 0.0, "nothing was written"


def test_a_zero_transaction_is_refused(ledger):
    from relay.ledger import Entry

    with pytest.raises(LedgerError, match="positive amount"):
        ledger._post("deposit", [Entry(available(CONSUMER), debit=0), Entry("world", credit=0)])


def test_refund_reverses_a_settlement(ledger):
    ledger.deposit(CONSUMER, 10.0)
    ledger.hold(CONSUMER, 5.0, "job-1")
    ledger.settle(CONSUMER, PROVIDER, 2.0, receipt_id="r1", job_id="job-1")

    ledger.refund(CONSUMER, PROVIDER, 2.0, receipt_id="r1", job_id="job-1")

    assert ledger.balance(PROVIDER) == 0.0
    assert ledger.balance(CONSUMER) == 7.0
    ledger.check_invariant()


def test_slash_removes_credits_from_a_node(ledger):
    ledger.deposit(PROVIDER, 10.0)
    ledger.slash(PROVIDER, 3.0, reason="upheld dispute", ref_receipt_id="r1")

    assert ledger.balance(PROVIDER) == 7.0
    ledger.check_invariant()


def test_balance_is_derived_and_cache_is_only_a_cache(ledger):
    ledger.deposit(CONSUMER, 7.0)
    ledger.store.upsert_account(CONSUMER, 999.0)

    assert ledger.balance(CONSUMER) == 7.0, "entries are the truth"
    assert ledger.reconcile(CONSUMER) == 7.0
    assert ledger.store.get_account(CONSUMER)["balance_cached"] == 7.0


def test_a_full_job_leaves_the_ledger_balanced(ledger):
    ledger.deposit(CONSUMER, 20.0)
    ledger.hold(CONSUMER, 5.0, "job-1")
    for i in range(5):
        ledger.settle(CONSUMER, PROVIDER, 0.4, receipt_id=f"r{i}", job_id="job-1")
    ledger.release_remaining(CONSUMER, "job-1")

    assert ledger.balance(PROVIDER) == pytest.approx(2.0)
    assert ledger.balance(CONSUMER) == pytest.approx(18.0)
    assert ledger.held_balance(CONSUMER) == 0.0
    ledger.check_invariant()


def test_concurrent_settlements_cannot_double_spend():
    """Two threads settling the same receipt must produce one payment."""
    import threading

    ledger = Ledger(MemoryStore())
    ledger.deposit(CONSUMER, 100.0)
    ledger.hold(CONSUMER, 50.0, "job-1")

    def settle():
        ledger.settle(CONSUMER, PROVIDER, 3.0, receipt_id="same-receipt", job_id="job-1")

    threads = [threading.Thread(target=settle) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    assert ledger.balance(PROVIDER) == 3.0, "paid once, not eight times"
    ledger.check_invariant()


# -- sweeping abandoned holds ---------------------------------------------


def test_sweep_leaves_a_live_job_alone():
    store = MemoryStore()
    ledger = Ledger(store)
    ledger.deposit(CONSUMER, 10.0)
    ledger.hold(CONSUMER, 4.0, "job-live")
    store.upsert_session(
        session_id="job-live",
        worker_id="w",
        task_goal="g",
        steps_total=3,
        steps_completed=1,
        status="in_progress",
        current_machine="m",
        inference_node="n",
    )
    assert sweep_expired_holds(store, ledger) == []
    assert ledger.held_balance(CONSUMER, "job-live") == 4.0


def test_sweep_releases_a_hold_for_a_job_nobody_resumed():
    store = MemoryStore()
    ledger = Ledger(store)
    ledger.deposit(CONSUMER, 10.0)
    ledger.hold(CONSUMER, 4.0, "job-abandoned")
    # No session row at all: evicted before it ever recorded one.
    released = sweep_expired_holds(store, ledger, ttl_seconds=DEFAULT_HOLD_TTL_SECONDS)

    assert released == [(CONSUMER, "job-abandoned", 4.0)]
    assert ledger.balance(CONSUMER) == 10.0
    ledger.check_invariant()


def test_sweep_releases_a_hold_for_a_completed_job():
    store = MemoryStore()
    ledger = Ledger(store)
    ledger.deposit(CONSUMER, 10.0)
    ledger.hold(CONSUMER, 4.0, "job-done")
    store.upsert_session(
        session_id="job-done",
        worker_id="w",
        task_goal="g",
        steps_total=1,
        steps_completed=1,
        status="completed",
        current_machine="m",
        inference_node="n",
    )
    assert sweep_expired_holds(store, ledger)[0][2] == 4.0
    assert ledger.held_balance(CONSUMER) == 0.0


def test_account_names_separate_available_from_held():
    assert available("abc") == "abc:available"
    assert held("abc", "job-1") == "abc:held:job-1"


def test_a_settlement_cannot_exceed_the_job_hold(ledger):
    """Otherwise the hold is decorative and a job can spend past its budget."""
    ledger.deposit(CONSUMER, 100.0)
    ledger.hold(CONSUMER, 2.0, "job-1")

    with pytest.raises(InsufficientFunds, match="holds 2.0 credits"):
        ledger.settle(CONSUMER, PROVIDER, 5.0, receipt_id="r1", job_id="job-1")

    assert ledger.balance(PROVIDER) == 0.0
    ledger.check_invariant()


def test_a_slash_cannot_drive_a_balance_negative(ledger):
    """A negative balance is a debt nothing can collect — it means credits were
    paid out that never existed."""
    ledger.deposit(PROVIDER, 3.0)
    ledger.slash(PROVIDER, 100.0, reason="huge penalty")

    assert ledger.balance(PROVIDER) == 0.0
    ledger.check_invariant()


def test_recoverable_reports_what_a_node_can_actually_cover(ledger):
    ledger.deposit(PROVIDER, 4.0)
    assert ledger.recoverable(PROVIDER, 10.0) == 4.0
    assert ledger.recoverable(PROVIDER, 1.0) == 1.0
