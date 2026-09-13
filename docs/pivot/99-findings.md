# Findings: what the task-dispatch pivot actually cost, and what it proved

Written after the work, against the questions the pivot prompt asked.

---

## The demo, stated plainly

A matrix multiplication that takes about an hour on one machine is cut into
tiles, handed to several devices that asked for work, and reassembled. Kill a
device mid-job and its tiles come back to the queue and are done by somebody
else. The assembled answer is bit-identical to the same multiplication computed
on one machine.

Proved three ways, all in CI:

1. **In-process**, `tests/test_tasks.py` — devices claim, one holds two tiles and
   vanishes, assembly refuses, leases lapse, tiles are reissued, result matches.
2. **Across real OS processes**, `tests/test_tasks_multiprocess.py` — a provider
   is `SIGKILL`ed while the store shows it holding a lease. Survivors are started
   *afterwards*, so nothing depends on timing. Verified by sabotage: with reaping
   disabled the test fails.
3. **At scale, by hand** — a 600×600×600 job (216M multiply-accumulates) across
   three separate node processes, ~21M units/s each, `collect` verified
   bit-identical.

---

## What the prompt asked, and what happened

### "Additive, not a rewrite. The inference path stays and stays green."

Held. No existing module was rewritten. The inference request path
(`MarketSession._attempt` → `/inference/complete`) is untouched; tasks got a
second, pull-based path beside it. All pre-existing tests pass unchanged.

### "All 372 existing tests stay green."

Held, and the suite is now **434**.

### "No broker. Providers pull."

Held, and it turned out to remove more machinery than expected. Because
providers poll, there is no dispatcher; and because every node reaps expired
leases before claiming, there is no reaper service either. Recovery from a dead
device is something the surviving devices do for each other as a side effect of
asking for their own work. A deployment whose coordinator notices failures has a
coordinator whose own failure nobody notices.

### "No new external dependencies without asking."

Held. Nothing was added. The temptation was numpy, and refusing it turned out to
be a feature rather than a sacrifice — see determinism below.

### "Don't touch payments, KYC, or real-money code."

Held. The ledger gained no new entry kinds. Offers and receipts gained fields for
pricing work that has no tokens, which is quoting, not settlement.

### "Ask before any destructive change or schema migration."

Asked, and the answer was to proceed. Migration 009 is additive: new tables, four
`NOT NULL` constraints relaxed with defaults. No column dropped, no row rewritten.

### "The node may read the task payload."

Taken at face value. Operands and payloads are in the clear. No attestation, no
confidential execution.

### "The provider must never become an open compute proxy."

Held, and it is the first check the executor makes. Work runs because a consumer
with a committed budget signed for it in writing. An unsigned task is refused
before it is parsed, and consent is checked before a single operand is fetched —
a node should not spend bandwidth on work it was always going to refuse.

---

## The decision that shaped everything: `python_exec` determinism

Deliverable 0 flagged the conflict: the plan wanted `python_exec` in Phase 0 *and*
wanted to slash providers whose redundant runs diverged. Both cannot hold —
`PYTHONHASHSEED`, float repr, locale and library drift all diverge across
machines, so we would have slashed honest providers for their Python patch
release. The ruling was to ship it as `deterministic: false`.

**That decision paid for itself in an unexpected place.** Having accepted that
determinism is a property worth designing *for* rather than hoping for, the
matmul kernel was written to have it: pure Python, fixed k-ascending
accumulation. Python floats are IEEE-754 doubles with correctly-rounded `*` and
`+`, so every honest machine produces identical bytes.

So the verification story does not run on toy task types after all. It runs on
the demo workload itself. That is a materially better position than the one I
described when I proposed option (a), and the reason is that BLAS — which we
would have reached for automatically with numpy in the dependency list —
reorders arithmetic and would have destroyed the property silently.

**Refusing the dependency bought the verifiability.** I did not anticipate that
and would not have found it without the constraint.

---

## Where the audit was right, and where it was wrong

Right: the ledger and reputation modules contain no reference to tokens, models
or inference, and generalised for free. They were built against receipt *status*
rather than receipt *contents*, and that accident of Phase 4 is what made this
pivot additive.

Right: there was no queue of any kind, and building one was new construction.

**Wrong about sizing, in a way worth recording.** The audit and the first
implementation both split the job by rows only. Writing the documentation
required benchmarking the kernel, and the benchmark contradicted the number I
had been using: ~20M multiply-accumulates per second per core, not the ~4.5M I
had inferred from tiny blocks where per-task overhead dominated. At the true
rate, the hour-long job is 4200³ — and at 4200², the right operand is 141MB that
a row-split design sends to every task.

So the design was wrong for the job it was built for, and only measuring found
it. Tiles now cut both ways and both operands are content-addressed, capped at
4MB a strip.

A second sizing error surfaced at the same time: tiles were sized at a fraction
of the task *deadline*, which cut a 2400³ job into a handful of large tiles and
left most of a fleet idle. A job in a few large tiles finishes no sooner than its
slowest member. Sizing now targets a tile *duration*.

Both errors were invisible until something real was measured. Neither would have
been caught by a test.

---

## What prior art was worth

**Darkbloom's `routing-v2-attestation-churn.md` was worth the whole study.** 62%
of their fleet unroutable, none of it hardware, every root cause a timing or
state bug in the mechanism our leases depend on. Four of the six are now
structurally impossible here:

| Their failure | What makes it impossible |
|---|---|
| Expiry shorter than the call it guarded | `assert_lease_sane` refuses to start |
| Round trip scoped to a connection | Leases are rows; completion accepted on any request from the holder |
| One constant serving two purposes | Separate poll, retry and cooldown constants |
| In-memory state wiped on deploy | Lease state is only ever in the database |
| Token only at registration | Capability travels in every poll |
| No observability | Queue counters from the first commit |

**PAIR** confirmed the boundary — one whole request to one node, never split —
in NVIDIA's own words, and its separation of capability gate from scheduler is
mirrored in `Requirements.accepts()` versus the selection policies.

**Golem** supplied the WASM-and-fuel argument that made the `python_exec` ruling
easy to state.

What could not be checked stayed unchecked: Akash Homenode, Golem's utilisation
post-mortems, and the sandbox vendors' threat models are all still unknown, and
the last of those remains the most valuable missing artifact.

---

## What is not built

- **No iOS app.** The node is Python: a laptop, a Pi, or an Android phone under
  Termux works today; an iPhone does not. The protocol is HTTPS against
  PostgREST plus Ed25519 signing, so an iOS client is a small app rather than a
  port — but writing it was not this work.
- **No real sandbox.** `python_exec` has resource limits, not a security
  boundary, and is off by default. WASM is the answer and remains unbuilt.
- **No payment rail.** Credits are internal. The economic loop is otherwise
  closed: a node publishes a signed offer, bills for each tile it finishes, and
  the consumer countersigns and settles out of the job's hold when it collects.
  Verified end to end — two providers each earned 2.16 credits on a 12-tile job,
  every receipt acknowledged, ledger invariant intact. What is missing is the
  rail between a credit and a pound, not the accounting.
- **Divergence is wired to slashing, with a deliberate brake.** A conclusive
  minority is disputed and slashed through the existing mechanism. A two-way
  disagreement is `UNADJUDICATED` and costs nobody anything. The adjudicator
  recomputes the verdict from the stored results rather than reading it out of
  the dispute, because the evidence was filed by one of the parties — tested by
  having a malicious consumer accuse an honest provider with fabricated
  evidence, which is rejected.
- **Still no Supabase code path has ever executed.** Every test runs on
  `MemoryStore` or `FileStore`. `SupabaseStore`, migration 009's policies, and
  the `relay_tasks` RLS rules are unverified against a real Postgres. This was
  the largest gap before the pivot and it is the largest gap after it.

---

## Honest limits of the demo

**Only independent work fans out.** Matrix rows and columns are independent,
which is why this is a fair demonstration rather than a contrived one. A
computation that is one long dependency chain uses one device however many are
connected.

**A disagreement between two providers proves only that one is wrong.**
`compare` returns `INCONCLUSIVE` rather than guessing; naming whoever answered
first would make an honest provider's reputation a coin flip. Conviction needs a
majority of three.

**The arithmetic is about a hundred times slower than numpy.** Deliberately, and
the trade is stated above — but it means Relay is not competitive on raw
throughput with a single machine that has BLAS. It is competitive on the axis it
was built for: work nobody is paying a data centre for, on devices already
bought and mostly idle.
