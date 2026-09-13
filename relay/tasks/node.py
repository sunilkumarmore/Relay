"""The provider node: a loop that asks for work and does it.

This is the whole of the supply side. There is no server here and no port to
open, because a provider is never called — it calls. A node polls the queue for
task types its operator has enabled, takes a lease on one task, renews that
lease while it works, and posts a signed result. Everything travels outbound
over HTTPS, which is the only reason a phone behind carrier NAT or a laptop
behind a home router can be a provider at all.

There is also no scheduler process. Every node reaps expired leases before it
claims, so recovering a dead device's work is something the surviving devices do
for each other as a side effect of asking for their own. A deployment where the
coordinator is the thing that notices failures has a coordinator whose own
failure nobody notices.

## Stopping

Ctrl-C finishes the task in hand and then exits. That is worth the wait: a node
that abandons a lease costs its consumer a full lease period before the work is
reissued, and costs itself the fee for work it had nearly finished. A second
Ctrl-C gives up immediately and lets the lease lapse.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from relay import config
from relay.identity import Identity
from relay.provider.offers import DEFAULT_OFFER_TTL_SECONDS, Offer, build_offer
from relay.store import Store, store_from_env
from relay.tasks import billing, kernels
from relay.tasks.consent import OperatorConsent
from relay.tasks.executor import ExecutionError, TaskExecutor
from relay.tasks.model import Task
from relay.tasks.queue import (
    DEFAULT_LEASE_SECONDS,
    DEFAULT_POLL_INTERVAL_SECONDS,
    LEASE_SLACK_SECONDS,
    TaskQueue,
)

# How much of a lease to let elapse before renewing. A third leaves two chances
# to renew before expiry, so one dropped request does not lose the work —
# Darkbloom widened exactly this margin to stop single-missed-tick flapping.
RENEW_AT_FRACTION = 3

# Republish the offer with this much of its life left. An offer that lapses
# takes the node out of the directory until the next poll, and a node that is
# working hard is exactly the one whose republish is most likely to be late.
OFFER_REFRESH_FRACTION = 2


@dataclass
class NodeStats:
    claimed: int = 0
    completed: int = 0
    failed: int = 0
    reaped: int = 0
    work_units: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def as_dict(self) -> dict[str, Any]:
        return {
            "claimed": self.claimed,
            "completed": self.completed,
            "failed": self.failed,
            "reaped_for_others": self.reaped,
            "work_units": self.work_units,
            "uptime_seconds": int(time.monotonic() - self.started_at),
        }


class ProviderNode:
    def __init__(
        self,
        identity: Identity,
        store: Store,
        *,
        consent: OperatorConsent | None = None,
        queue: TaskQueue | None = None,
        poll_interval: int = DEFAULT_POLL_INTERVAL_SECONDS,
        max_tasks: int | None = None,
        price_per_mega_unit: float = 0.0,
        price_per_task: float = 0.0,
        publish_offers: bool = True,
    ) -> None:
        self.identity = identity
        self.store = store
        self.consent = consent or OperatorConsent()
        self.queue = queue or TaskQueue(store, lease_seconds=DEFAULT_LEASE_SECONDS)
        self.executor = TaskExecutor(identity, store, consent=self.consent)
        self.poll_interval = poll_interval
        self.max_tasks = max_tasks
        self.price_per_mega_unit = price_per_mega_unit
        self.price_per_task = price_per_task
        self.publish_offers = publish_offers
        self.offer: Offer | None = None
        self._offer_published_at = 0.0
        self.stats = NodeStats()
        self._stop = threading.Event()
        self._hard_stop = threading.Event()

    # -- lifecycle --------------------------------------------------------

    def request_stop(self) -> None:
        if self._stop.is_set():
            self._hard_stop.set()
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    # -- advertising ------------------------------------------------------

    def refresh_offer(self, *, force: bool = False) -> Offer | None:
        """Publish what this node will do and what it charges.

        Capability also travels in every poll, so the offer is not what gets a
        task claimed — it is what makes the node visible in the directory and,
        more importantly, what fixes the price a receipt is later checked
        against. A bill nobody quoted is not one a consumer should pay.
        """
        if not self.publish_offers:
            return None
        age = time.monotonic() - self._offer_published_at
        due = age >= DEFAULT_OFFER_TTL_SECONDS / OFFER_REFRESH_FRACTION
        if not force and self.offer is not None and not due:
            return self.offer
        self.offer = build_offer(
            self.identity,
            endpoint_url="",
            task_types=tuple(self.consent.allowed_task_types),
            price_per_mega_unit=self.price_per_mega_unit,
            price_per_task=self.price_per_task,
            max_payload_bytes=self.consent.max_payload_bytes,
            max_concurrency=self.consent.max_concurrent,
        )
        self.store.upsert_offer(self.offer.to_row())
        self._offer_published_at = time.monotonic()
        return self.offer

    def bill(self, task: Task, result) -> None:
        """Record what this node is owed for a completed task.

        Unacknowledged: the consumer countersigns and settles when it collects.
        Recording it either way means a provider that did the work has signed
        evidence of it even if the consumer never comes back.
        """
        offer = self.offer or self.refresh_offer(force=True)
        if offer is None:
            return
        try:
            receipt = billing.issue_receipt(
                self.identity, task=task, result=result, offer=offer
            )
            self.store.upsert_receipt(receipt.to_row())
        except Exception as exc:  # noqa: BLE001 - never lose finished work over a bill
            print(f"  ! could not record a receipt for {task.task_id[:8]}: {exc}", flush=True)

    # -- one task ---------------------------------------------------------

    def _renewer(self, task: Task, done: threading.Event) -> None:
        """Keep the lease alive while the work runs.

        Renewal is time-based rather than progress-based because the kernels do
        not report progress. If renewal starts failing the task is finished
        anyway — the result is still recorded, and `complete` reports it as
        having landed too late rather than pretending it counted.
        """
        interval = max(1, self.queue.lease_seconds // RENEW_AT_FRACTION)
        while not done.wait(interval):
            try:
                self.queue.renew(task.task_id, provider_node_id=self.identity.node_id)
            except Exception as exc:  # noqa: BLE001 - a failed renewal must not kill the work
                print(f"  ! could not renew lease on {task.task_id[:8]}: {exc}", flush=True)

    def run_one(self) -> bool:
        """Claim and run a single task. Returns whether there was work."""
        self.refresh_offer()
        self.stats.reaped += len(self.queue.reap())

        task = self.queue.claim(
            provider_node_id=self.identity.node_id,
            task_types=list(self.consent.allowed_task_types),
        )
        if task is None:
            return False
        self.stats.claimed += 1
        print(
            f"  → {task.task_type} {task.task_id[:8]} "
            f"({task.work_units:,} units, job {task.job_id[:8]})",
            flush=True,
        )

        done = threading.Event()
        renewer = threading.Thread(
            target=self._renewer, args=(task, done), daemon=True, name="relay-lease"
        )
        renewer.start()
        started = time.monotonic()
        try:
            result = self.executor.execute(task)
        except ExecutionError as exc:
            # Refused rather than attempted: hand it straight back so the next
            # node gets it now instead of after the lease.
            self.queue.report_failure(
                task.task_id, provider_node_id=self.identity.node_id, error=str(exc)
            )
            self.stats.failed += 1
            print(f"  ✗ refused {task.task_id[:8]}: {exc}", flush=True)
            return True
        finally:
            done.set()

        elapsed = time.monotonic() - started
        if result.status == "ok" and self.queue.complete(
            result, provider_node_id=self.identity.node_id
        ):
            self.stats.completed += 1
            self.stats.work_units += task.work_units
            self.bill(task, result)
            rate = task.work_units / elapsed if elapsed > 0 else 0
            print(
                f"  ✓ {task.task_id[:8]} in {elapsed:.1f}s ({rate:,.0f} units/s)",
                flush=True,
            )
        elif result.status != "ok":
            self.queue.report_failure(
                task.task_id, provider_node_id=self.identity.node_id, error=result.error
            )
            self.stats.failed += 1
            print(f"  ✗ {task.task_id[:8]}: {result.error}", flush=True)
        else:
            print(f"  ~ {task.task_id[:8]} finished after its lease lapsed", flush=True)
        return True

    # -- the loop ---------------------------------------------------------

    def run_forever(self) -> int:
        self._banner()
        idle_since: float | None = None
        while not self.stopping:
            if self.max_tasks is not None and self.stats.claimed >= self.max_tasks:
                print(f"\nReached the {self.max_tasks}-task limit.", flush=True)
                break
            try:
                worked = self.run_one()
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - a node must outlive one bad poll
                print(f"  ! poll failed, retrying: {exc}", flush=True)
                worked = False
            if worked:
                idle_since = None
                continue
            if idle_since is None:
                idle_since = time.monotonic()
                print("  … waiting for work", flush=True)
            if self._stop.wait(self.poll_interval):
                break
        self._summary()
        return 0

    def _banner(self) -> None:
        print(f"\nRelay provider node  {self.identity.node_id[:16]}…", flush=True)
        print(f"  task types : {', '.join(self.consent.allowed_task_types) or 'none'}", flush=True)
        if self.price_per_mega_unit or self.price_per_task:
            print(
                f"  price      : {self.price_per_mega_unit} credits per million units"
                f" + {self.price_per_task} per task",
                flush=True,
            )
        else:
            print("  price      : free (set RELAY_PRICE_PER_MEGA_UNIT to charge)", flush=True)
        print(f"  lease      : {self.queue.lease_seconds}s", flush=True)
        print(f"  poll every : {self.poll_interval}s", flush=True)
        for warning in self.consent.warnings():
            print(f"\n  WARNING: {warning}\n", flush=True)
        print("  Ctrl-C finishes the task in hand and exits.\n", flush=True)

    def _summary(self) -> None:
        stats = self.stats.as_dict()
        print(
            f"\nDone. {stats['completed']} completed, {stats['failed']} failed, "
            f"{stats['work_units']:,} work units, "
            f"{stats['reaped_for_others']} stale leases returned to the queue for others.",
            flush=True,
        )


def node_from_env(env_path: str | None = None) -> ProviderNode:
    config.load_env(env_path)
    consent = OperatorConsent.from_env()
    if kernels.PYTHON_EXEC in consent.allowed_task_types:
        from relay.tasks.pyexec import register_default

        register_default()

    store = store_from_env(env_path)
    if store is None:
        raise RuntimeError(
            "No store configured. Set SUPABASE_URL and SUPABASE_KEY in .env, "
            "or RELAY_STORE=file with RELAY_STORE_PATH to run against a shared file."
        )
    identity = Identity.load_or_create(config.get("RELAY_KEY_FILE") or None)
    lease_seconds = int(os.environ.get("RELAY_LEASE_SECONDS") or DEFAULT_LEASE_SECONDS)
    max_seconds = consent.max_task_seconds
    # Lowered only for local multi-process runs, where provider and store share
    # a clock and a loopback interface. Real deployments leave it alone.
    slack = int(os.environ.get("RELAY_LEASE_SLACK_SECONDS") or LEASE_SLACK_SECONDS)
    return ProviderNode(
        identity,
        store,
        consent=consent,
        price_per_mega_unit=float(os.environ.get("RELAY_PRICE_PER_MEGA_UNIT") or 0.0),
        price_per_task=float(os.environ.get("RELAY_PRICE_PER_TASK") or 0.0),
        queue=TaskQueue(
            store,
            lease_seconds=lease_seconds,
            max_task_seconds=max_seconds,
            lease_slack_seconds=slack,
        ),
        poll_interval=int(
            os.environ.get("RELAY_POLL_INTERVAL") or DEFAULT_POLL_INTERVAL_SECONDS
        ),
    )


def run() -> int:
    node = node_from_env()

    def handle(_signum, _frame) -> None:
        if node.stopping:
            print("\nSecond interrupt — dropping the lease and exiting now.", flush=True)
            sys.exit(130)
        print("\nFinishing the current task, then stopping. Ctrl-C again to give up.", flush=True)
        node.request_stop()

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)
    return node.run_forever()


if __name__ == "__main__":
    raise SystemExit(run())
