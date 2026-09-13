# Relay → Marketplace: Build Plan (prompt format)

Relay today proves one primitive: an agent's progress lives outside the process running it, so the
process is disposable. This plan turns that primitive into a working peer-to-peer AI compute
marketplace. Each phase below is a **prompt to paste into a fresh Claude Code session on this repo**.

## Status

Phases 0-6.5 are **built**, and so is the task-dispatch pivot that came after
them. Phase 7 (CLI and web surfaces) and Phase 8 (hardening) are still prompts.

| Phase | State | What landed |
|---|---|---|
| 0 Foundation | done | `relay/` package, Store + InferenceBackend seams, CI, kill -9 eviction test |
| 1 Identity | done | Ed25519 nodes, signed requests, durable registry, RLS |
| 2 Offers | done | Signed expiring offers, multi-model providers, 429 back-pressure |
| 3 Discovery | done | Directory, selection policies, mid-job failover |
| 4 Ledger | done | Signed receipts, token bound, double-entry credits |
| 5 Reputation | done | Deterministic scores, adjudicated disputes, stake and slashing |
| 6 Stateful agent | done | Context between steps, hashed agent state, compaction, dependencies |
| 6.5 Token authority | done | The JWT exchange migration 002 assumed but never had, plus write policies scoped to the session owner |
| Pivot: task dispatch | done | Pull queue with leases, deterministic kernels, 2D-tiled matmul, operator consent, verification, settlement |
| 7 Surfaces | **next** | `relay` CLI, web market dashboard, docker compose |
| 8 Hardening | pending | TLS, rate limits, metrics, chaos runs |

With Phase 6 in, Relay is a resumable *agent* rather than a resumable queue:
steps share one conversation, quote each other's answers, and a run killed
without warning resumes into the same conversation it would have had.

**The pivot was not in this plan and it changes what the marketplace sells.**
Selling inference needs a GPU and a model; selling *tasks* — the arithmetic
between an agent's thinking steps — needs neither, which puts an ordinary
laptop or phone on the supply side. Providers pull work rather than being
called, so no provider opens a port. The planned dial-out broker was cancelled
before it was built: polling achieves the same with less machinery.

It also cancelled a piece of this plan's economics. Task pricing needs no
analogue of the token over-claim dispute, because `work_units` is fixed by the
task's own payload and re-derived by the queue — there is no quantity for a
provider to inflate, only an answer that can be wrong. See `docs/pivot/` for the
prior-art study, the audit, and the findings.

Phase 6.5 was not in the original plan. Migration 002 wrote RLS policies against
a JWT claim that nothing in the codebase produced, so applying it would have
locked the application out of its own database — and its write policies were
loose enough that any node could rewrite any other session. Both are fixed in
`relay/authority/` and migration 008.

The largest remaining gap is not a phase: **no Supabase code path has ever
executed.** Every test runs against MemoryStore or FileStore. `SupabaseStore`,
the `on_conflict` clauses, the ledger trigger and the RLS policies are all
unverified — and migration 009's task tables and policies are now unverified in
the same way. Integration tests against a local `supabase start` stack should
come before Phase 7.

## How to use this document

1. Start a new Claude Code session on the Relay repo.
2. Paste the **Preamble** block, then the **Phase N** block, as one message.
3. Let the phase land as a PR; merge it with CI green before starting the next phase.
4. Phases 0–5 and 7–8 are sequential. Phase 6 depends only on Phase 0 and can run in parallel.

```
0 → 1 → 2 → 3 → 4 → 5 → 7 → 8
0 → 6 (parallel with 2–5)
```

## Vocabulary — the market model

| Term | Meaning | Today |
|---|---|---|
| **Provider** | Node that sells inference | `inference/registry.py` + Ollama |
| **Consumer** | Node that buys inference to run an agent | `worker/daemon.py` |
| **Offer** | A provider's signed advertisement: model, context, price, capacity | — |
| **Job** | A consumer's session: task + budget + requirements | `relay_sessions` |
| **Receipt** | Signed proof of one inference call; the unit of settlement | `relay_inference_log` (unsigned) |
| **Directory** | Where offers, health and reputation live | — |
| **Ledger** | Double-entry credits: deposits, holds, settlements, slashes | — |

## One architectural decision, stated up front

**v1 is peer-to-peer compute with centralized coordination.** Supabase remains the directory and
ledger. Every call to it goes through one interface (`relay/store.py`) so the coordination layer
can be swapped later. This plan does not include a blockchain, a token, or a payment rail; credits
are an internal unit and the ledger is designed so a deposit can be backed by real money later.

---

## Preamble (paste before every phase)

```
You are working in the Relay repo (github.com/sunilkumarmore/Relay), a two-tier distributed AI
agent system. Inference nodes (inference/registry.py — FastAPI in front of Ollama) serve worker
nodes (worker/daemon.py) that run multi-step agent tasks and checkpoint every step to Supabase
(tables relay_sessions, relay_checkpoints, relay_worker_state, relay_inference_log,
relay_migration_log; schema in setup/supabase_setup.sql; client in worker/checkpoint_client.py).
Workers can be evicted (SIGINT/SIGTERM, worker/eviction_handler.py) and resumed on any machine
from the last checkpoint. controller/controller.py is a click CLI; dashboard/dashboard.py is a
rich TUI. Tasks are YAML (tasks/example.yaml, loaded by worker/problems.py).

We are evolving Relay into a peer-to-peer AI compute marketplace: Providers sell inference,
Consumers buy it to run agents, and settlement is based on signed receipts. Read
docs/MARKETPLACE_PLAN.md first for the vocabulary and the full phase plan, then implement ONLY the
phase below.

Rules for this session:
- Work on a new branch named for the phase; open a PR when done.
- Keep the existing two-machine demo in README working; update README where commands change.
- Everything you build gets pytest coverage that runs with no network (use the in-memory Store
  and the fake inference backend from Phase 0).
- Run ruff and the full test suite before every commit.
- Do not refactor beyond what the phase requires. Do not start the next phase.
```

---

## Phase 0 — Foundation: package, tests, CI, prove hard eviction

```
PHASE 0 — Foundation.

Goal: make Relay verifiable so every later phase can be trusted. Today there are zero tests and
no CI, and the only way to exercise the system is two machines plus a live Supabase project.

Build:
1. Restructure into an installable package `relay/` (relay/worker, relay/inference,
   relay/controller, relay/dashboard) with pyproject.toml. Remove the try/except relative-import
   and sys.path hacks. `python -m relay.worker`, `python -m relay.inference`,
   `python -m relay.controller`, `python -m relay.dashboard` must work; update README. Leave thin
   shims at the old paths (worker/daemon.py etc.) for one release.
2. `relay/store.py`: a `Store` Protocol covering every Supabase call in checkpoint_client.py
   (sessions, checkpoints, worker_state, inference_log, migration_log). `SupabaseStore`
   implements it. `MemoryStore` is an in-memory implementation for tests. Nothing outside
   store.py imports supabase.
3. `relay/inference/backends.py`: an `InferenceBackend` Protocol
   (`complete(prompt, max_tokens) -> (text, tokens_in, tokens_out)`, `health() -> bool`),
   `OllamaBackend` (wraps the existing client; read prompt_eval_count / eval_count for the
   token split) and `FakeBackend` (deterministic output, configurable latency and failure
   injection).
4. pytest suite under tests/:
   - worker runs N steps to completion against MemoryStore + FakeBackend and writes the report
   - graceful eviction: SIGTERM mid-step writes worker_state and an 'evicted' event; a resumed
     run completes the remaining steps and emits 'resumed' (and 'migrated' when machine_id
     differs)
   - HARD eviction: run the worker as a subprocess, kill -9 it after step k is checkpointed but
     before step k+1 completes; the resumed run must skip steps <= k and finish; assert no
     duplicate (session_id, step_number) rows
   - insert_checkpoint returns False on a duplicate step
   - registry: register / heartbeat / prune / deregister; 401 for an unregistered worker on
     /inference/complete; 502 when the backend raises
   - call_inference retry policy with the HTTP layer mocked: 4xx does not retry, 5xx and
     timeouts retry with 2s/4s/8s backoff
5. GitHub Actions: ruff + pytest on push and PR, Python 3.11 and 3.12.
6. Add a ruff config and fix what it flags.

Acceptance: `pytest` is green locally with no network; CI is green on the PR; the README
two-machine demo still works with the new commands; the hard-eviction test exists and passes.

Don't: change the Supabase schema, add auth, or touch pricing.
```

---

## Phase 1 — Identity, signed requests, durable registry, RLS

```
PHASE 1 — Identity and the trust boundary.

Goal: replace "worker_id says hello" with cryptographic identity, make the inference tier as
crash-safe as the worker tier, and close the hole where the Supabase anon key gives anyone full
read/write on every relay_* table.

Build:
1. `relay/identity.py`: Ed25519 keypair per node (cryptography or pynacl), stored at
   RELAY_KEY_PATH (default ~/.relay/node.key, mode 0600), generated on first run. node_id is the
   hex public key. WORKER_ID and MACHINE_ID remain human labels; identity is the key.
2. Signed HTTP in `relay/auth.py`: every request to a provider carries X-Relay-Node,
   X-Relay-Timestamp and X-Relay-Signature over (method, path, timestamp, sha256(body)).
   Provide a `requests` auth adapter for clients and a FastAPI dependency for servers. Reject
   bad signatures with 401 and timestamps older than 60s (replay protection) with 401.
3. Registry state moves into the Store: active workers, heartbeats and the request log become
   Store methods backed by new tables relay_nodes and relay_registry_requests, so a registry
   restart loses nothing. Keep a small in-memory cache for /inference/status.
4. Supabase security: a new migration in setup/ enabling RLS on every relay_* table. Nodes
   obtain a scoped JWT by exchanging a signed challenge (small FastAPI endpoint or Supabase edge
   function); the anon key can only read public directory tables. Document the upgrade path for
   existing deployments.
5. Tests: signature verify pass/fail, replay rejection, registry restart preserves registrations
   (MemoryStore survives a server re-instantiation), and a SQL RLS test file runnable against
   the supabase CLI local stack (marked integration, skipped unless SUPABASE_TEST_URL is set).

Acceptance: an unsigned or mis-signed /inference/complete returns 401; a registry restarted
mid-job keeps serving its registered workers without re-registration; the anon key cannot write
any relay_* table; all Phase 0 tests pass.

Don't: add pricing, offers, or multiple providers.
```

---

## Phase 2 — Provider abstraction and offers

```
PHASE 2 — Providers publish offers.

Goal: a provider is any node that can serve a model and states its terms. Many providers can
exist at once.

Build:
1. Move inference/registry.py to relay/provider/server.py, keeping the /worker/* and
   /inference/* routes and adding /offer. A provider has: node_id, one or more backends (Ollama,
   plus an OpenAI-compatible backend for vLLM / llama.cpp / LM Studio), and for each model
   served: context_window, price_in_per_1k, price_out_per_1k, max_concurrency, region.
2. `Offer` (pydantic) and table relay_offers: offer_id, provider_node_id, endpoint_url, model,
   context_window, price_in_per_1k, price_out_per_1k, max_concurrency, region, capabilities
   (json), signature (provider signs the canonical offer), published_at, expires_at. The
   provider republishes every OFFER_TTL/2; expired offers drop out of the directory.
3. Provider config lives in relay-provider.yaml (backends, models, prices) — not env vars.
   Validate on start and refuse to start if a backend does not actually have a configured
   model.
4. Concurrency control: a per-provider semaphore of max_concurrency; return 429 with
   Retry-After when saturated and record saturation events in the Store.
5. Token accounting: tokens_in and tokens_out are recorded separately on relay_inference_log
   and relay_checkpoints (migration for the new columns).
6. Tests: offer signing and verification, expiry, 429 under saturation, price fields
   round-trip through the Store, OpenAI-compatible backend against a fake HTTP server.

Acceptance: two providers run on two ports with different prices and both appear in
relay_offers with valid signatures; a consumer can complete a job against either by pointing
INFERENCE_REGISTRY at it (discovery is Phase 3).

Don't: build matching or the ledger.
```

---

## Phase 3 — Discovery, matching, failover

```
PHASE 3 — Consumers shop.

Goal: remove the hardcoded INFERENCE_REGISTRY. Consumers discover providers from the directory,
choose by policy, and fail over when one disappears.

Build:
1. Job requirements in the task YAML: model (exact or family), max_price_in_per_1k,
   max_price_out_per_1k, min_context_window, region preference, min_reputation (everyone is
   0.0 until Phase 5 — treat as pass), budget_credits.
2. `relay/consumer/market.py`: `Directory.find_offers(requirements)` returns unexpired,
   validly-signed offers matching the requirements; `select(offers, policy)` with policies
   cheapest | fastest (recent latency from relay_inference_log) | round_robin |
   pinned(node_id). Default cheapest.
3. Failover in the worker loop: after the existing retries fail with 5xx / timeout / 429, mark
   the provider unhealthy for FAILOVER_COOLDOWN seconds, select the next offer, register with
   it, and retry the same step. Persist the chosen provider on relay_worker_state so a resumed
   worker prefers it but re-selects if it is gone.
4. relay_migration_log gains a 'provider_switched' event with from_node, to_node and reason.
5. Consumers record health observations in relay_provider_health (observer_node_id,
   provider_node_id, ok, latency_ms, error, observed_at). This is reputation input for Phase 5.
6. INFERENCE_REGISTRY stays as an optional override (pinned policy) so the two-machine demo
   still works unchanged.
7. Tests: each selection policy, expired and mis-signed offers excluded, failover switches
   provider mid-job without skipping or duplicating a step, a resumed worker re-selects when
   its provider's offer has expired.

Acceptance: with three FakeBackend providers at different prices, a job picks the cheapest,
keeps going when that provider is killed, completes, and the dashboard shows the switch.

Don't: touch money — no holds or settlements yet.
```

---

## Phase 4 — Metering, signed receipts, ledger

```
PHASE 4 — Money: receipts and a ledger.

Goal: every inference call produces a signed receipt both sides can verify, and credits move
only against receipts.

Build:
1. Receipt: {receipt_id, job_id (session_id), step_number, consumer_node_id, provider_node_id,
   offer_id, request_hash (sha256 of prompt + params), response_hash, tokens_in, tokens_out,
   latency_ms, price_in_per_1k, price_out_per_1k, amount_credits, issued_at}. The provider
   signs it and returns it with the response. The consumer verifies: signature, request_hash
   matches what it sent, response_hash matches what it received, prices match the selected
   offer, amount_credits is correctly computed. The consumer then countersigns. Both
   signatures are stored in relay_receipts. A receipt lacking the consumer signature is
   'unacknowledged'.
2. Consumer-side token bound: re-tokenize prompt and response locally (tiktoken or HF
   tokenizers for the model; fall back to a chars/4 heuristic with a wide tolerance) and refuse
   to countersign when provider-claimed tokens exceed the local count by more than
   TOKEN_TOLERANCE (default 10%). Record the refusal as status 'disputed' with a reason; Phase 5
   adjudicates.
3. `relay/ledger.py`, double-entry: relay_accounts (node_id, balance_cached) and
   relay_ledger_entries (entry_id, tx_id, account, debit, credit, kind:
   deposit|hold|release|settle|refund|slash, ref_receipt_id, ref_job_id, created_at).
   Invariant: per tx_id, sum(debit) == sum(credit), enforced by a DB trigger. Balance is derived
   from entries; balance_cached is a materialization with a reconcile command.
4. Job lifecycle: on start, hold budget_credits from the consumer (refuse to start on
   insufficient balance); on each acknowledged receipt, settle amount_credits from the hold to
   the provider; on completion — or eviction with no resume within HOLD_TTL — release the
   remainder.
5. Dev faucet: `relay wallet deposit --dev` credits test accounts when RELAY_DEV_MODE=1.
6. Provider side: providers store their receipts and can list what they are owed.
7. Tests: receipt signing and verification in both directions; hash mismatch, price mismatch
   and token-bound violations rejected; ledger invariant; hold/settle/release math including
   eviction mid-job; concurrent settlements cannot double-spend (row locks plus an idempotency
   key on receipt_id).

Acceptance: a full job yields exactly one acknowledged receipt per step, the ledger sums to zero
across all accounts, the provider balance increases by exactly the sum of its receipts, and a
tampered receipt is rejected.

Don't: integrate any payment rail. Credits are internal.
```

---

## Phase 5 — Reputation and disputes

```
PHASE 5 — Reputation and disputes.

Goal: make lying expensive and reliability visible.

Build:
1. `relay/reputation.py`: a per-provider score in [0, 1] computed over a rolling window from
   acknowledged-receipt rate, dispute rate, consumer health observations (Phase 3), latency
   versus advertised, and offer uptime. Written to relay_reputation (node_id, score, components
   json, computed_at) by `relay reputation recompute`, which anyone can run — the score is a
   deterministic function of the stored data so results are checkable. Consumer selection
   honors min_reputation; the default policy becomes cheapest among providers at or above the
   threshold.
2. Disputes: relay_disputes (dispute_id, receipt_id, opened_by, reason, status
   open|upheld|rejected, evidence json). Automated adjudication for the two objective cases:
   (a) hash mismatch — decidable from the signed data alone; (b) token over-claim — re-tokenize
   with the canonical tokenizer for the model. Upheld: refund the consumer and slash a penalty
   from the provider's stake. Rejected: the consumer pays. Subjective quality disputes are out
   of scope — record and surface them only.
3. Provider stake: providers must hold a minimum balance to publish offers; slashes draw from
   it; under-staked providers are excluded from discovery.
4. Sampled verification: a consumer may set VERIFY_SAMPLE_RATE; sampled steps are also sent to
   a second provider and compared — exact match when temperature is 0 on the same model,
   otherwise token-count and latency plausibility only. Results feed reputation.
5. Consumer reputation as well (pays, does not abuse disputes) so providers can refuse low-rep
   consumers.
6. Tests: score determinism, adjudication in both outcomes with the right ledger entries,
   under-staked exclusion, sampled comparison.

Acceptance: a FakeBackend that inflates token counts is disputed, slashed and drops below the
selection threshold within one recompute; an honest FakeBackend does not.

Don't: build a human arbitration UI.
```

---

## Phase 6 — Stateful agent checkpointing (parallel with Phases 2–5)

```
PHASE 6 — Make the worker a real agent.

Goal: today each step is an independent prompt — no context flows between steps, and the
checkpoint stores reasoning == solution. Make the worker a stateful agent whose entire state is
checkpointed, so a resume rehydrates the agent rather than just the step counter.

Build:
1. `relay/agent/state.py`: AgentState {messages: list[{role, content}], scratchpad: dict,
   artifacts: dict[str, str], step_outputs: dict[int, str], tool_state: dict, version}.
   JSON-serializable. New table relay_agent_state (session_id, step_number, state_blob or a
   storage reference, state_hash, saved_at), one append-only row per completed step so any
   earlier step is restorable.
2. Step execution builds the prompt from the task goal, prior step outputs and the message
   history; calls inference; appends the assistant message; extracts distinct `solution` and
   `reasoning` by asking the model for a delimited format and parsing it. Never store the same
   string in both columns.
3. Context management: before each call, if estimated tokens exceed context_window -
   max_tokens for the selected provider, summarize older messages into one compaction message
   via inference and record the compaction in state. Checkpoint the post-compaction state.
4. Resume: load the latest relay_agent_state for the session, verify state_hash, rehydrate,
   continue. The eviction handler saves in-flight partial state flagged 'partial'; resume
   discards partial state and redoes that step (safe under UNIQUE(session_id, step_number)).
5. Task YAML: steps may reference prior outputs ({{ steps.1.solution }}) and declare
   depends_on; execute in dependency order; independent steps may run concurrently up to a
   configurable limit.
6. Tests: state round-trip; a resumed run yields message history byte-identical to an
   uninterrupted run (FakeBackend is deterministic); compaction triggers exactly at the
   boundary; template references resolve; partial state is discarded on resume.

Acceptance: kill -9 a 5-step dependent task after step 3; the resumed run's final report is
identical to an uninterrupted run's.

Don't: adopt a tool-calling framework; the state format is ours.
```

---

## Phase 7 — Consumer and provider surfaces

```
PHASE 7 — Product surface.

Goal: someone can join the market without reading Python.

Build:
1. A `relay` CLI (click) that replaces controller.py:
     relay init                              # identity + config
     relay provider start | offers | receipts | earnings
     relay run task.yaml [--policy cheapest|fastest|pinned NODE] [--max-price N] [--budget N]
     relay jobs ls | show ID | resume ID | cancel ID
     relay market offers [--model M] [--max-price N]
     relay wallet balance | deposit --dev | history
     relay disputes ls | open RECEIPT_ID --reason ...
     relay reputation show NODE
   Every command supports --json.
2. A web dashboard (FastAPI + HTMX, or a small static React app) for the market view: live
   offers and prices, providers with reputation, active jobs with migration timelines, the
   receipts stream, ledger totals. Keep the rich TUI for the two-machine demo.
3. docs/: provider quickstart, consumer quickstart, and a `docker compose up` that runs a local
   market — 3 providers on FakeBackend, 2 consumers, and the supabase CLI stack.
4. Tests: CLI commands against MemoryStore with --json snapshot tests; dashboard API endpoints.

Acceptance: on a clean machine, `pip install relay && relay init && relay provider start` puts
an offer in the directory; from another machine `relay run task.yaml` buys from it and both
wallets reflect the settlement.
```

---

## Phase 8 — Hardening and operations

```
PHASE 8 — Hardening.

Goal: survive strangers.

Build:
1. TLS everywhere: providers serve HTTPS; offers carry endpoint_url plus a certificate
   fingerprint; consumers pin it.
2. Rate limiting and abuse controls on provider endpoints (per node_id and per IP), request
   size limits, and a prompt-size-vs-context-window check before any billing.
3. Secrets: key files 0600, no keys or service tokens in .env.example, and a documented path
   for a secret manager.
4. Observability: structured JSON logs; Prometheus /metrics on providers (requests, tokens,
   latency, saturation, disputes); OpenTelemetry traces across consumer → provider → backend.
5. Load and chaos: a locust or k6 scenario with 20 consumers and 5 providers, and a chaos
   script that SIGKILLs random providers and consumers. After every run assert the ledger
   invariant and that no step was lost or duplicated.
6. Supabase: connection pooling, indexes on all hot queries (offers by model and price,
   receipts by job, ledger by account), and a retention policy for log tables.
7. Release engineering: versioned releases, CHANGELOG, and `relay db migrate` as the schema
   migration runner.

Acceptance: a 30-minute chaos run ends with the ledger summing to zero, every job either
completed or resumable, and no receipt without matching ledger entries.
```

---

## Deliberately out of scope for this plan

- **Payment rails** (Stripe, crypto). The ledger's `deposit` kind is where an external payment
  would attach.
- **Decentralized directory or gossip.** `relay/store.py` is the seam where that would go.
- **Human arbitration** of subjective quality disputes.
- **Model hosting or scheduling on the provider.** Relay sells what a backend already serves.
