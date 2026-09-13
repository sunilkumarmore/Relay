# Audit: what survives the task-dispatch pivot

Deliverable 1. Written after `000-prior-art.md` and before any code, as the
prompt requires. Every module in `relay/` is sorted into one of four buckets,
and the token-centric couplings are listed individually, because they are the
ones that decide how much of deliverable 2 is a schema change and how much is a
rewrite.

**Decision carried in from Deliverable 0:** `python_exec` ships with
`deterministic: false`. Redundant-execution comparison and `output_divergence`
slashing apply to the deterministic task types only. WASM is the eventual
answer, not a Phase 0 detour.

---

## The headline

Relay's marketplace machinery is in better shape for this pivot than I expected,
and its dispatch machinery is in worse shape.

**Better:** the ledger, reputation, identity, auth, and dispute *framework* have
no idea what a token is. `relay/ledger.py` and `relay/reputation.py` contain
zero references to tokens, models, or inference — checked by grep, not by
impression. Reputation scores providers on the *status* of receipts
(acknowledged vs. disputed) and on health observations, never on what the
receipt was for. That machinery generalises for free.

**Worse:** there is no queue. The current model is the consumer **pushing** to a
chosen provider — `MarketSession.complete()` POSTs to the provider's
`/inference/complete` (`relay/consumer/session.py:232`). The pivot requires
providers to **pull**. That is not a modification of the existing path; it is a
second dispatch path alongside it. The prompt already anticipated this
(deliverable 4), and the audit confirms there is nothing to reuse — no lease
table, no claim protocol, no work queue of any kind.

**And the gap nobody has named yet:** the agent does not make tool calls. Steps
are LLM prompts with dependencies, executed by `Agent.execute_step()`. The thing
this pivot proposes to distribute — work between thinking steps — does not
currently exist in the codebase at all. Deliverable 2 is not generalising an
abstraction, it is introducing one.

The one encouraging detail: `AgentState.tool_state` already exists
(`relay/agent/state.py:95`) as a reserved, always-empty dict that is serialised
into the state hash and round-trips through `to_dict`/`from_dict`. Nothing
writes it. It was carved out in Phase 6 for exactly this, and it means task
results can join the hashed agent state without a state-format migration.

---

## A. Survives unchanged

Nothing in this bucket needs a line changed for tasks. It is roughly 1,400 lines
of the most security-sensitive code in the repo, and the pivot does not disturb
it.

| Module | Why it is indifferent to what the work was |
|---|---|
| `relay/identity.py` (109) | Ed25519 keypair, node id = public key. Knows nothing about workloads. |
| `relay/auth.py` (116) | Signs method + path + timestamp + sha256(body). A task request signs exactly as an inference request does; `canonical_message` never inspects the body. |
| `relay/authority/*` (446) | Challenge/response → JWT carrying `relay_node_id`. Wholly orthogonal. |
| `relay/ledger.py` (303) | Double-entry hold → settle → release. **Zero** token or model references. A hold is a hold. |
| `relay/reputation.py` (273) | Scores on receipt *status* and health observations. Integrity multiplies service. Never reads `tokens_in`. |
| `relay/config.py` (38) | Environment plumbing. |
| `relay/store.py` — the ledger/account/reputation/dispute/node halves | The `Store` Protocol's shape is per-table; the tables that survive, survive. |

**This is the load-bearing finding of the audit.** The reason Phase 4's ledger
and Phase 5's reputation generalise for free is that they were built against
*receipt status*, not against *receipt contents*. That was not foresight on my
part — it fell out of "only adjudicate what both parties signed" — but it is the
single thing that makes this pivot additive rather than a rewrite.

---

## B. Survives with generalisation

These need a new shape, not new logic. In every case the existing logic is
correct and the inference-specific *fields* are the problem.

### B1. `relay/provider/offers.py` (153)

`SIGNED_FIELDS` (line 29) commits an offer to `model`, `context_window`,
`price_in_per_1k`, `price_out_per_1k`. A node offering `python_exec` has no
model and no context window, and cannot price per 1k tokens.

`capabilities: dict` already exists in the signed set and is the natural home
for task-type advertisement. **But** the offer is signed over these fields, so
adding one changes `canonical_bytes()` and invalidates every offer signed by an
older node. Offers carry a 300-second TTL, so the churn window is five minutes,
not a migration — a genuine piece of luck.

Proposal for deliverable 3: keep the inference fields, make them optional
(defaulted), and add a parallel task-pricing block. Do **not** repurpose the
token price fields to mean CPU-seconds; a field whose meaning depends on
another field is how the Darkbloom timeout bug happened.

### B2. `relay/receipts.py` (226)

`SIGNED_FIELDS` (line 32) includes `model`, `tokens_in`, `tokens_out`,
`price_in_per_1k`, `price_out_per_1k`. `verify_receipt()` recomputes the amount
from token counts and rejects a mismatch.

The *verification* is general and good: signature authentic, request hash
matches what we sent, response hash matches what we got, price matches the
advertised offer. Only the billing *quantity* is inference-shaped. For a task,
`tokens_in`/`tokens_out` have no meaning and `request_fingerprint(prompt,
max_tokens, model)` (line 54) hashes fields a task does not have.

Proposal: a receipt gains a `work_kind` and a unit-agnostic quantity, with the
token fields retained for `work_kind="inference"`. The hash checks stay exactly
as they are.

### B3. `relay/consumer/market.py` (278)

`Requirements` (line 46) filters on `model`, `min_context_window`,
`max_price_in_per_1k`, `max_price_out_per_1k`. Selection policy (cheapest,
fastest, round-robin, pinned), failover cooldown, health windows, and the
`UNSCORED = 0.5` prior are all workload-agnostic and should be left alone.

Prior-art note: PAIR separates the **capability gate** from the **scheduler**
(§1.3 of `000-prior-art.md`). `Requirements.accepts()` is already the gate and
the policy functions are already the scheduler. Generalise the gate; do not
touch the scheduler.

### B4. `relay/disputes.py` (258)

`REASON_HASH_MISMATCH` is fully general — recompute hashes, compare. It carries
straight over and becomes the backbone of `output_divergence`.

`REASON_TOKEN_OVERCLAIM` is inference-only; it re-counts tokens with
`relay/tokens.py`. There is no task equivalent, because per §1.11 a node's
self-report of CPU-seconds cannot be checked the way a token count can be
re-counted against recorded text. **This is open question 1 from Deliverable 0
and it is still open.** The honest Phase 0 answer is per-task pricing, where
there is no quantity to over-claim.

`SLASH_MULTIPLE = 2.0` and the stake mechanism are general.

### B5. `relay/agent/*` (661)

`plan.py` (dependencies, topological waves) is entirely workload-agnostic and is
what makes the fan-out demo possible — it already computes which steps are
independent. `state.py` needs `tool_state` populated, nothing more.

`agent.py` needs the real work: `execute_step()` calls a `CompleteFn` and that
is the only thing a step can do. A step must be able to be *a task* instead of
*a prompt*. `Agent.budget` / `compact()` are inference-only concerns
(`agent.py:84`) and must not be applied to task steps.

### B6. `relay/store.py` (1665) — the directory half

Needs new tables (tasks, leases) and generalised offer/receipt columns. The
`Store` Protocol seam is exactly what it was built for. All three backends
(Supabase / Memory / File) must gain the same methods, and per §2.2.D of the
prior art, **lease state lives in Postgres, never in process memory.**

---

## C. Inference-only — stays, stays green, does not extend

These serve the inference path and should be left alone rather than
generalised. The prompt's first constraint is that the inference path stays and
stays green; the cheapest way to honour that is not to touch these.

- `relay/inference/backends.py` + `registry.py` (316) — Ollama/OpenAI adapters.
- `relay/tokens.py` (83) — the independent token estimator. Only meaningful for
  inference; it is the tool for `REASON_TOKEN_OVERCLAIM`.
- `relay/agent/prompts.py` (167) — reasoning/solution split prompt templates.
- `Provider.complete()` + `/inference/complete` (`provider/server.py:364, 592`)
  and `MarketSession._attempt()` (`consumer/session.py:219`) — the **push**
  path. Tasks get a pull path beside it, and these are untouched.
- `insert_inference_log` and `relay_inference_log` — telemetry for inference.
  Tasks need their own counters (§2.2.F: ship `task_claim_total{outcome=...}`
  from the first commit).

---

## D. Obsolete

**Almost nothing, which is the right answer for an additive pivot.**

- The dial-out broker: cancelled by the prompt, and never built. Nothing to
  remove. §4.3 of the prior art supports the cancellation — d-inference's
  outbound-WebSocket model exists to avoid inbound connections, and pull-based
  polling achieves the same with less machinery.
- The top-level `worker/`, `inference/`, `controller/`, `dashboard/` forwarding
  shims are already deprecated and unrelated to this pivot. Leave them.

I did not find a module the pivot makes dead. If you expected one, say which and
I will re-examine it — an audit that finds nothing to delete deserves suspicion.

---

## Token-centric leakage — the complete list

Every place where "tokens" is baked into something a task cannot fill. This is
the list deliverable 2 has to work through.

| # | Location | Leak | Severity |
|---|---|---|---|
| 1 | `receipts.py:32` `SIGNED_FIELDS` | `tokens_in`, `tokens_out`, `price_*_per_1k`, `model` are **signed** | **High** — changing the tuple invalidates historical receipt signatures. Receipts are permanent; offers expire in 5 min. Receipts must stay verifiable forever. |
| 2 | `offers.py:29` `SIGNED_FIELDS` | `model`, `context_window`, `price_*_per_1k` signed | Medium — 300s TTL bounds the blast radius. |
| 3 | `receipts.py:54` `request_fingerprint(prompt, max_tokens, model)` | Hashes three fields a task does not have | Medium — needs a task-shaped fingerprint. |
| 4 | `offers.py:104` `Offer.price(tokens_in, tokens_out)` | The only pricing function there is | Medium |
| 5 | `market.py:46` `Requirements` | Five of seven filters are inference-shaped | Medium |
| 6 | `server.py:74` `InferenceRequest{prompt, max_tokens, model}` | The only request shape a provider accepts | Low — tasks get their own endpoint. |
| 7 | `setup/005:19-23` | `tokens_in`/`tokens_out` are `NOT NULL` on `relay_receipts` | **High** — a task receipt cannot be inserted without a migration. Needs `DEFAULT 0` or nullable. **Schema change: proposing, not applying, per the prompt.** |
| 8 | `setup/003:12-15` | `model`, `context_window`, `price_*` `NOT NULL` on `relay_offers` | **High** — same, for offers. |
| 9 | `setup/003:45-46` | `tokens_in`/`tokens_out` on `relay_checkpoints` | Low — already `DEFAULT 0`. |
| 10 | `agent/agent.py:84` `budget()` from `context_window` | Context math applied per step | Medium — must not fire on task steps. |
| 11 | `daemon.py:414` | Context window read off the bound offer | Medium — a task provider has no context window. |
| 12 | `disputes.py` `REASON_TOKEN_OVERCLAIM` | Only quantity dispute that exists | Medium — no task analogue until pricing is decided. |

Items **7 and 8 are `NOT NULL` constraints and require a migration.** Per the
prompt I am proposing and waiting, not applying. The proposal is additive:
`ALTER COLUMN ... DROP NOT NULL` plus defaults, no data rewritten, no column
dropped, and existing rows untouched.

---

## What I need before writing code

1. **Ruling on items 7 and 8** — the two `NOT NULL` migrations. Nothing in
   deliverable 2 can be inserted into a real database without them.
2. **Confirmation on receipt signature versioning (leak 1).** My proposal: add a
   `schema_version` to `SIGNED_FIELDS` and have `canonical_bytes()` branch on
   it, so v1 receipts verify against the v1 field tuple forever. The alternative
   — a second receipt type — duplicates the verification logic, which is the
   code I least want two copies of.
3. **Open question 1 from Deliverable 0 is still unanswered**: what is a
   truthful CPU-second? My recommendation is now firmer having done the audit:
   **price per task in Phase 0.** It sidesteps leak 12 entirely, and it means a
   provider has no quantity to inflate.

---

## What the audit says about the demo

The fan-out demo — an agent's work split across several phones — depends on:
`agent/plan.py` waves (**exists, unchanged**), a pull queue with leases (**does
not exist, deliverable 4**), a task executor (**does not exist, deliverable
6**), and payment per completed task (**exists, needs leak 1/2/7/8 resolved**).

Two of four exist. The two that don't are new construction rather than
refactoring, which is the better kind of missing.

One honest constraint, unchanged from Phase 6 and not caused by this pivot: only
*independent* steps fan out. A plan that is one long dependency chain uses one
device no matter how many are connected.

---

## Cross-reference

| Prior-art finding | Where it lands in this audit |
|---|---|
| §1.3 capability gate ≠ scheduler | B3 — generalise `accepts()`, leave policies alone |
| §1.11 self-reports prove nothing | B4, need #3 — why per-task pricing wins in Phase 0 |
| §2.2.A lease TTL vs. task timeout | B6 — startup assertion, not a comment |
| §2.2.D in-memory state wiped on deploy | B6 — leases in Postgres |
| §2.2.E capability only at registration | B1 — capability travels in every poll |
| §2.2.F no observability | C — task counters from the first commit |
| §4.3 broker cancellation supported | D |
