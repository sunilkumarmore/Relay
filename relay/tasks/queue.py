"""The pull queue, and the leases that make a dead device survivable.

Providers are not called; they call. A node polls for work of the types it can
run, is granted a *lease* on one task, and renews that lease while it works. No
inbound connection ever reaches a provider, which is what lets a phone behind a
carrier NAT be a provider at all.

A lease is a row in the database, never a connection or an entry in some
scheduler's memory. That is deliberate, and it is the difference between "a
provider that reconnects loses its work" and "a provider that reconnects carries
on". Darkbloom lost 62% of its fleet to the opposite choice, and four of their
six root causes are impossible to reproduce here for exactly this reason.

## What happens when a device dies

Nothing special, which is the point. A provider that stops renewing stops
holding its lease; once it expires the task goes back to `queued` and the next
poller picks it up. There is no liveness probe, no heartbeat service, no
consumer-side death detection — a machine that has gone away simply stops
renewing, and the queue notices by not hearing from it. `attempts` climbs each
time, so a task that kills every provider it touches eventually fails rather
than cycling forever.

The one invariant that makes this safe is checked at startup and refuses to run
if violated: **a lease must outlive the work it covers**. If a lease could
expire while its task is still legitimately running, the queue would hand the
same work to a second provider and pay twice for it — Darkbloom's root cause A,
a 60-second expiry guarding a 90-second call.
"""

from __future__ import annotations

import threading
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

from relay.store import Store
from relay.tasks import model as task_model
from relay.tasks.model import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_LEASED,
    STATUS_QUEUED,
    Task,
    TaskResult,
)

# How long a lease is granted for. Must exceed the longest task it may cover by
# at least LEASE_SLACK_SECONDS; `assert_lease_sane` enforces it.
DEFAULT_LEASE_SECONDS = 900

# The margin between a task's own deadline and its lease expiring. Covers clock
# skew between provider and database, and the round trip carrying the result.
LEASE_SLACK_SECONDS = 120

# Three separate constants, deliberately not shared. Darkbloom's root cause C
# was one 20-minute device cooldown reused as a retry interval, which stranded
# providers for up to an hour. Reusing a constant because two numbers happen to
# be equal today is how that happens.
DEFAULT_POLL_INTERVAL_SECONDS = 5
DEFAULT_CLAIM_RETRY_SECONDS = 1
DEFAULT_PROVIDER_COOLDOWN_SECONDS = 60

# How many candidates a poller will fight over before giving up this round.
MAX_CLAIM_ATTEMPTS = 8


class LeaseConfigError(RuntimeError):
    """The lease would be shorter than the work it covers."""


def now_utc() -> datetime:
    return datetime.now(UTC)


def assert_lease_sane(lease_seconds: int, max_task_seconds: int) -> None:
    """Refuse to start rather than double-paying later.

    This is a startup check and not a comment because a comment does not stop
    a deployment. The failure it prevents is silent: the queue re-issues work
    that is still running, both providers finish, and both invoice.
    """
    if lease_seconds < max_task_seconds + LEASE_SLACK_SECONDS:
        raise LeaseConfigError(
            f"lease of {lease_seconds}s cannot cover a task allowed {max_task_seconds}s "
            f"plus {LEASE_SLACK_SECONDS}s of slack. A lease that expires while its task "
            f"is still running gets the work issued twice and billed twice. "
            f"Raise the lease to at least {max_task_seconds + LEASE_SLACK_SECONDS}s, "
            f"or lower the task's max_seconds."
        )


class TaskQueue:
    """Pull-based dispatch with database-held leases."""

    def __init__(
        self,
        store: Store,
        *,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        max_task_seconds: int = task_model.DEFAULT_MAX_SECONDS,
    ) -> None:
        assert_lease_sane(lease_seconds, max_task_seconds)
        self.store = store
        self.lease_seconds = lease_seconds
        self.max_task_seconds = max_task_seconds
        # Observability from the first commit, not once something goes wrong.
        # Outcomes mirror the ones Darkbloom wished they had had.
        self.counters: Counter[str] = Counter()
        self._lock = threading.Lock()

    def _count(self, outcome: str) -> None:
        with self._lock:
            self.counters[outcome] += 1

    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self.counters)

    # -- submitting -------------------------------------------------------

    def submit(self, task: Task, *, at: datetime | None = None) -> Task:
        """Queue a task, refusing anything a provider would be right to reject.

        Validating here as well as at the provider is not redundant: it keeps a
        malformed task from occupying a lease and burning an attempt before
        anyone discovers it cannot run.
        """
        problems = task_model.verify_task(task, at=at)
        if problems:
            self._count("rejected")
            raise ValueError(f"will not queue an invalid task: {'; '.join(problems)}")
        if task.max_seconds > self.max_task_seconds:
            self._count("rejected")
            raise LeaseConfigError(
                f"task allows {task.max_seconds}s but this queue is configured for "
                f"a maximum of {self.max_task_seconds}s"
            )
        queued = task.model_copy(
            update={
                "status": STATUS_QUEUED,
                "lease_holder": "",
                "lease_expires_at": "",
                "updated_at": (at or now_utc()).isoformat(),
            }
        )
        self.store.enqueue_task(queued.to_row())
        self._count("queued")
        return queued

    # -- claiming ---------------------------------------------------------

    def claim(
        self,
        *,
        provider_node_id: str,
        task_types: list[str],
        at: datetime | None = None,
    ) -> Task | None:
        """Take a lease on one task, or return None if there is nothing to do.

        Capability travels with every poll rather than being registered once.
        A provider that gains or loses a task type between polls is re-evaluated
        on the next one, with no re-registration and nothing to go stale —
        Darkbloom's root cause E was a token that only ever arrived at
        registration, leaving late nodes permanently unchallenged.
        """
        now = at or now_utc()
        now_iso = now.isoformat()
        candidates = self.store.claimable_tasks(
            task_types=task_types, now_iso_ts=now_iso, limit=MAX_CLAIM_ATTEMPTS
        )
        for row in candidates:
            task = Task.from_row(row)
            lease_until = now + timedelta(seconds=self.lease_seconds)
            won = self.store.compare_and_set_task(
                task.task_id,
                expect_status=STATUS_QUEUED,
                updates={
                    "status": STATUS_LEASED,
                    "lease_holder": provider_node_id,
                    "lease_expires_at": lease_until.isoformat(),
                    "attempts": task.attempts + 1,
                    "updated_at": now_iso,
                },
            )
            if not won:
                # Another provider took it between the read and the write.
                # Expected under load, not an error.
                self._count("contended")
                continue
            self._count("claimed")
            return task.model_copy(
                update={
                    "status": STATUS_LEASED,
                    "lease_holder": provider_node_id,
                    "lease_expires_at": lease_until.isoformat(),
                    "attempts": task.attempts + 1,
                }
            )
        return None

    def renew(
        self, task_id: str, *, provider_node_id: str, at: datetime | None = None
    ) -> bool:
        """Extend a lease the caller still holds.

        Accepted on any authenticated request from the lease holder, because the
        lease lives in the database and not in whatever connection happened to
        create it. A provider that dropped its socket and reconnected still owns
        its work.
        """
        now = at or now_utc()
        lease_until = now + timedelta(seconds=self.lease_seconds)
        ok = self.store.compare_and_set_task(
            task_id,
            expect_status=STATUS_LEASED,
            expect_lease_holder=provider_node_id,
            updates={
                "lease_expires_at": lease_until.isoformat(),
                "updated_at": now.isoformat(),
            },
        )
        self._count("renewed" if ok else "renew_refused")
        return ok

    # -- finishing --------------------------------------------------------

    def complete(
        self,
        result: TaskResult,
        *,
        provider_node_id: str,
        at: datetime | None = None,
    ) -> bool:
        """Record a result and close the lease.

        A result whose hash does not match its own output is refused outright:
        the provider signed the hash, so a mismatch is either corruption or a
        provider hoping nobody checks.
        """
        now = at or now_utc()
        row = self.store.get_task(result.task_id)
        if row is None:
            self._count("orphan_result")
            return False
        task = Task.from_row(row)
        problems = task_model.verify_result(result, task=task)
        if problems:
            self._count("result_rejected")
            return False
        if result.provider_node_id != provider_node_id:
            self._count("result_rejected")
            return False

        ok = self.store.compare_and_set_task(
            result.task_id,
            expect_status=STATUS_LEASED,
            expect_lease_holder=provider_node_id,
            updates={
                "status": STATUS_COMPLETED,
                "completed_by": provider_node_id,
                "output_hash": result.output_hash,
                "lease_holder": "",
                "lease_expires_at": "",
                "updated_at": now.isoformat(),
            },
        )
        if not ok:
            # The lease was reaped while this provider was working, and someone
            # else may already hold it. The work is not wasted — it is recorded
            # as a result either way, and a redundant result is what verification
            # wants. It is just no longer this provider's task to close.
            self._count("completed_too_late")
            self.store.insert_task_result(result.to_row())
            return False
        self.store.insert_task_result(result.to_row())
        self._count("completed")
        return True

    def report_failure(
        self,
        task_id: str,
        *,
        provider_node_id: str,
        error: str,
        at: datetime | None = None,
    ) -> str:
        """A provider saying up front that it cannot finish.

        Better than letting the lease lapse: the task returns to the queue
        immediately instead of after the full lease, and the reason is recorded
        against the attempt rather than lost.
        """
        now = at or now_utc()
        row = self.store.get_task(task_id)
        if row is None:
            return "unknown"
        task = Task.from_row(row)
        terminal = task.attempts >= task.max_attempts
        updates: dict[str, Any] = {
            "lease_holder": "",
            "lease_expires_at": "",
            "last_error": error[:500],
            "updated_at": now.isoformat(),
            "status": STATUS_FAILED if terminal else STATUS_QUEUED,
        }
        ok = self.store.compare_and_set_task(
            task_id,
            expect_status=STATUS_LEASED,
            expect_lease_holder=provider_node_id,
            updates=updates,
        )
        if not ok:
            self._count("failure_refused")
            return "refused"
        outcome = "failed" if terminal else "requeued"
        self._count(outcome)
        return outcome

    # -- reaping ----------------------------------------------------------

    def reap(self, *, at: datetime | None = None) -> list[tuple[str, str]]:
        """Return work whose holder has gone quiet.

        This is the whole of Relay's failure handling: a device that is
        switched off, loses signal, or is killed mid-task stops renewing, and
        its lease lapses. Nothing detects the death; the absence is the signal.

        Returns `(task_id, outcome)` where outcome is `requeued` or `failed`.
        """
        now = at or now_utc()
        now_iso = now.isoformat()
        reaped: list[tuple[str, str]] = []
        for row in self.store.expired_leases(now_iso):
            task = Task.from_row(row)
            terminal = task.attempts >= task.max_attempts
            outcome = STATUS_FAILED if terminal else STATUS_QUEUED
            updates = {
                "status": outcome,
                "lease_holder": "",
                "lease_expires_at": "",
                "updated_at": now_iso,
                "last_error": (
                    f"lease held by {task.lease_holder or 'unknown'} expired at "
                    f"{task.lease_expires_at} without a result"
                ),
            }
            ok = self.store.compare_and_set_task(
                task.task_id,
                expect_status=STATUS_LEASED,
                expect_lease_holder=task.lease_holder,
                updates=updates,
            )
            if not ok:
                # It completed or was reaped by another poller in the meantime.
                continue
            label = "failed" if terminal else "requeued"
            self._count(f"expired_{label}")
            reaped.append((task.task_id, label))
        return reaped

    # -- inspection -------------------------------------------------------

    def job_status(self, job_id: str) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for row in self.store.list_tasks(job_id=job_id, limit=100000):
            counts[str(row.get("status", "unknown"))] += 1
        return dict(counts)

    def is_job_done(self, job_id: str) -> bool:
        rows = self.store.list_tasks(job_id=job_id, limit=100000)
        if not rows:
            return False
        return all(row.get("status") in (STATUS_COMPLETED, STATUS_FAILED) for row in rows)
