# Prior art for the task-dispatch pivot

Deliverable 0. Written before any code or audit work, from primary sources where
they were reachable.

---

## Verification status — read this first

Network access from this environment is partial. GitHub is reachable; almost
everything else is refused by the egress proxy with a 403 at the CONNECT stage.
What follows separates what was read from what was not. **Nothing in the
"unverified" list has been filled in from memory**, because a wrong recollection
of a competitor's design is worse than a gap — we would build on it.

### Read directly (primary source, cloned and read)

| Source | What was read |
|---|---|
| `NVIDIA/Personal-AI-Router` | Full `docs/architecture.mdx` (737 lines), `docs/inference-dispatcher.mdx`, `README.md`, and Go source in `services/` |
| `darkbloomdev/darkbloom` | `README.md`, repository layout |
| `Layr-Labs/d-inference` | `README.md`, `docs/design/*`, `docs/consumer/privacy-expectations.md` |
| `golemfactory/yapapi` | Command vocabulary in `yapapi/script/command.py`, engine layout |
| `golemfactory/ya-runtime-wasi` | `README.md` — Wasmtime sandbox model |
| `golemfactory/ya-runtime-vm` | `README.md` — VM/gvmkit model |
| `bigscience-workshop/petals` | `README.md` — throughput figures |
| `cloudflare/sandbox-sdk` | `README.md` |
| `e2b-dev/E2B` | Repository layout (README is thin; the substance is on the blocked docs site) |

### Could NOT be checked — treat every claim about these as unknown

| Source | Why | What we therefore do not know |
|---|---|---|
| **Akash Homenode** | `akash-network/homenode` returns 404; `homenode.akash.network` and `akash.network/blog/` blocked by egress policy | The entire supply-side onboarding section. I cannot confirm they ship an ISO at all, why, how hardware eligibility is gated, or what operator controls exist. **Deliverable 3's Akash questions are unanswered.** |
| **Golem utilisation post-mortems** | `docs.golem.network` blocked; nothing in the cloned repos discusses utilisation | Why utilisation stayed low — the single most cautionary question asked. Unanswered. |
| **Sandbox vendor engineering** | `e2b.dev/docs`, `modal.com/docs`, `developers.cloudflare.com/sandbox`, `vercel.com/docs/vercel-sandbox` all blocked | Cold-start techniques, checkpoint/restore of in-memory state, and — most costly — **their published threat models for running untrusted LLM-generated code**. The prompt correctly identified that as the single most useful artifact to borrow. We do not have it. |
| **Darkbloom's own blog** | `blog.eigencloud.xyz`, `darkbloom.dev` blocked | Mitigated: both source repositories were readable, which is better evidence than the blog would have been. |
| `golemfactory/yagna` | Repository 404s at that path | Provider-side agent internals; the runtimes were readable separately. |

**Recommendation:** the sandbox threat models and the Golem utilisation question
are worth a human spending an hour on, and should not be re-attempted from this
environment.

---

## 1. Adopt

### 1.1 Confirmed: one whole request to one node (PAIR)

The strongest available evidence for our architectural choice is stated in
PAIR's own `README.md`, as a blockquote immediately under the overview:

> PAIR routes each independent request to one node. It does **not** pool GPU
> memory, combine GPUs into a larger logical GPU, shard one model across
> machines, or split an in-flight inference request between nodes.

NVIDIA, with unlimited resources and the same heterogeneous-consumer-hardware
target, drew exactly the boundary the pivot draws. `docs/architecture.mdx`
repeats it: *"One request goes to one node and the proxy never splits it."*

**Adopt:** state the same boundary in our own README in the same place, in the
same plain language. It is a load-bearing constraint, not a limitation to
apologise for.

### 1.2 Failover as an ordered list, not a chosen winner (PAIR)

`services/ollama-proxy/proxy.go` builds an ordered failover list per request and
walks it. Two details are worth copying exactly:

`proxy.go:1197-1214` — what is retryable:

```go
// shouldRetry reports whether an upstream status warrants failing over to
// the next candidate: busy/unavailable/gateway statuses, plus a 404 on an
// inference call (an advertised owner's inventory may have become stale).
// Genuine client errors (400/401/422…) are not retried — they'd fail
// identically on every node.
```

This is the same rule `relay/consumer/session.py` already implements for
inference (4xx raises, 5xx/timeout/429 fails over). Independent convergence on
the same rule is good evidence it is right. **Adopt unchanged for tasks.**

`proxy.go:1312-1317` — the limit we do not inherit:

> We can only retry before the first byte reaches the client; once a response
> starts streaming we're committed.

**This is PAIR's hard ceiling on durability and we are structurally free of it.**
A task returns an artifact and an `output_hash`, not a stream, so there is no
commit point before completion. A task can be re-dispatched at any moment up to
the instant its result is accepted. When deliverable 2 claims we are "strictly
better on durability", this is the concrete reason, and it should be cited.

### 1.3 Capability gate and scheduler are separate mechanisms (PAIR)

From `docs/architecture.mdx`:

- **The scheduler is model-blind.** It ranks every node by pending work and GPU
  pressure and never mentions a model.
- **The proxy enforces model capability.** It removes nodes that cannot serve the
  request, then applies the scheduler's ordering to the survivors.

So load balancing happens *among the nodes that can serve the request*, not
across the cluster.

**Adopt directly into `min_capability` (deliverable 2).** Capability is a hard
filter producing a candidate set; our existing discovery policies
(cheapest/fastest/round-robin/pinned) then order that set unchanged. This means
**deliverable 3's claim that "existing discovery policies should apply
unchanged" is sound**, and PAIR is the evidence.

### 1.4 Hysteresis and freshness in load signals (PAIR)

Verified in `services/nvpair-job-scheduler/telemetry.go:111-137`:

```go
func pressureBand(utilization float64) int { /* 40 / 70 / 85 */ }

func pressureWithHysteresis(utilization float64, previous int) int {
	up := [...]float64{40, 70, 85}
	down := [...]float64{0, 35, 65, 80}
	...
}
```

Separate up and down thresholds, explicitly "to avoid rank thrash". Stale
telemetry (>10s) contributes a **neutral** value, not an idle one —
`effectiveGPUPressure` in the same file. Ranking sorts by
`pending + pressure`, then `pressure`, then stable node ID so cold start is
deterministic (`schedule.go:74-127`).

**Adopt:** any provider-load signal we add for task routing gets hysteresis and
a neutral-on-stale rule from day one. Both are cheap and both prevent a class of
oscillation we would otherwise discover in production.

### 1.5 Liveness must not be tested on the serving path (PAIR)

Node eviction, from `docs/architecture.mdx`: the scanner tolerates three
consecutive missed announcements, then still tries four escape hatches before
dropping a node — recent inference it served, recent telemetry, a node-info
probe, an engine-manager probe. Two justifications are worth quoting:

> The activity check comes first because a node under load is the one most
> likely to miss an announcement, and evicting it would pull a working machine
> out of the routing pool at its busiest.

> The remaining checks deliberately avoid the inference proxy ports. Those ports
> exist to serve requests, and using them as a liveness test would put
> connection attempts on the serving path every time multicast dropped a packet.

**Adopt both.** For us: a provider actively completing tasks must never be
considered dead, and lease expiry must never be the *first* signal we act on if
cheaper evidence of life exists. Deliverable 4 should record "last completed
task" and consult it before reaping a lease.

### 1.6 Pairing is a standard, not an invention (PAIR)

`services/eap-noob/` is a full Go implementation of **RFC 9140 (EAP-NOOB)** —
ephemeral ECDH authenticated by a short user-assisted out-of-band message,
yielding a long-lived association key `Kz` from which arbitrary secrets are
derived. Cryptosuite 1 is Curve25519/SHA-256. The six-digit PIN is the OOB
message.

This answers the prompt's question *"what survives the jump off the LAN?"*
precisely: **mDNS and the IP fallback do not survive; the trust bootstrap does.**
EAP-NOOB never depended on the LAN for its security, only for transport. The
shape — install agent, agent displays a short code, owner types it into a web
console, node is thereafter bound by a long-lived key — works identically over
the internet.

**Adopt the shape** for binding a node to an owner's account. We already have
Ed25519 identity and a challenge-response token exchange in `relay/authority/`;
the missing piece is the human-assisted step that says *this key belongs to this
person*. Do not invent a pairing protocol — this is the standard.

### 1.7 Transport: ours is sufficient, theirs is different (PAIR)

PAIR uses mTLS with self-signed leaf certificates pinned against a node UUID at
pairing time, owned by `nvpair-cluster-manager`. Cleverly, proxies serve two
personalities on one port, discriminated by the connection's first byte (`0x16`
is a TLS handshake record, and no HTTP method begins with it).

**Do not adopt mTLS.** Our Ed25519 signed-request scheme (method, path,
timestamp, body hash) is sufficient and better suited to a pull model: it
authenticates each *message* rather than each *connection*, which is what a
provider polling from behind NAT through arbitrary intermediaries needs. mTLS
would add certificate issuance, rotation and pinning for no gain here.

**Do adopt one thing:** their loopback enforcement rule. The plaintext
personality is refused from any non-loopback address with a `403`, and the
reasoning is stated: *"Without that check the port would be an open relay for
anything on the network."* Deliverable 6 requires that a provider "must never
become an open compute proxy" — this is the same hazard and should be enforced
the same way, as a hard check rather than a convention.

### 1.8 The "no new API" surface (PAIR) — and our proposal

PAIR's compatibility trick is **port takeover**: the proxy claims `11434`
(Ollama) and `1234` (LM Studio), and moves the real engine to `11435`/`1235`.
Rationale from `docs/architecture.mdx`:

> An application that already works with Ollama is configured for `11434`. If
> PAIR listened somewhere else, every tool would need reconfiguring to gain
> anything, so PAIR inverts it.

It yields rather than fights: an unknown process holding the port is never moved
or killed, and the takeover is reported as blocked.

**Our equivalent, proposed (this is a design proposal, not prior art):** the
agent-side surface for task dispatch should be **an MCP server**. An agent
harness that already speaks MCP gains Relay by adding a server entry — no code
change, no new SDK, no framework patch. Relay's tools appear as ordinary tools;
the fan-out happens behind them. A thin `@relay.task`-style decorator for
frameworks that do not speak MCP is the fallback, not the primary.

This preserves PAIR's actual lesson — *the integration cost must be
configuration, not code* — without copying a mechanism (port seizure) that makes
no sense for us.

### 1.9 Content never reaches logs (PAIR)

From `docs/inference-dispatcher.mdx`:

> Neither log, nor stdout, ever contains prompt text or response bodies. Prompts
> are identified by a `sha256:<prefix> len=<n>` digest and responses by their
> byte count and their own digest... This is a hard rule for every Personal AI
> Router component, not a setting. There is no flag that turns it off.

**Adopt verbatim as a rule.** It costs nothing now and is very expensive to
retrofit. It matters more for us than for PAIR: our stated trust boundary is that
*the node may read the payload*, so everything the coordinator and the operator
log is the part we can actually control. Deliverable 6's operator-readable task
log should record task id, type, digest, duration and CPU-seconds — never
payload or output bodies.

### 1.10 State the trust boundary as two lists (Darkbloom)

`docs/consumer/privacy-expectations.md` in `Layr-Labs/d-inference` is the best
example of this practice I have seen. It has a **"What you can rely on"** section
and a **"What you cannot rely on"** section, and the second opens:

> 1. "The coordinator never sees plaintext" is false. The coordinator opens your
>    request in memory to route, bill and enforce the request contract, then
>    re-encrypts it for exactly one provider.

> 2. The provider sees your prompt and the completion in plaintext: it is the
>    decryption endpoint, because in-process inference at native speed requires
>    it. The guarantee is about *which* process holds the key.

**Adopt the format exactly** for `docs/pivot/02-operator-consent.md` and for a
consumer-facing counterpart. Our Phase 0 boundary is blunter than theirs and
should be written just as plainly: *the node operator can read your task payload
and its output; do not send anything you would not hand to a stranger.*

### 1.11 A self-reported measurement proves nothing (Darkbloom)

From `docs/design/apns-code-attestation.md`:

> A self-reported `binaryHash` cannot prove code identity, because the measurer
> is the potentially-malicious provider.

**Adopt as a design axiom.** It rules out a whole category of cheap-looking
Phase 0 shortcuts: a node reporting its own CPU-seconds, its own runtime version,
its own sandbox state, or its own resource limits is reporting an *assertion*,
not a *fact*. Our answer in Phase 0 is not attestation (explicitly out of scope)
but **redundant execution and hash comparison**, which is measurement by a
second party. That is the right instinct and this is the reason it is necessary.

### 1.12 Fuel, not wall-clock, is the deterministic meter (Golem)

`golemfactory/ya-runtime-wasi` README, on the Wasmtime sandbox:

> Fuel limits guest computation, not wall-clock

and memory is handled as a *"non-moving 1 GiB virtual reservation and a matching
hard limit"* which reserves address space without committing physical RAM. Mounts
are declared in a manifest with `ro` / `rw` / `wo` permissions **enforced by
Wasmtime** through the WASI preopen model.

Two consequences for us:

1. **Fuel is a deterministic unit.** Our deliverable 3 prices tasks "per task and
   per CPU-second". CPU-seconds are machine-dependent and self-reported, which
   collides directly with 1.11. Fuel is neither. If we ever want a metered unit a
   consumer can verify, fuel is it — and that is an argument for the WASM path
   being about *billing integrity*, not just phone support.
2. **The preopen/manifest model is the filesystem design** for deliverable 6's
   "no filesystem access outside a scratch dir". Golem has a decade of iteration
   here; we should express our sandbox's filesystem policy in the same shape
   (declared mounts with explicit modes) even while the executor is a subprocess,
   so the WASM swap is a substitution rather than a redesign.

### 1.13 Command vocabulary (Golem)

`yapapi/yapapi/script/command.py` defines: `Deploy`, `Start`, `Terminate`,
`Run`, `SendBytes` / `SendFile` (via `_SendContent`), `DownloadFile`. Lifecycle
and data movement are separate verbs from execution.

**Adopt the separation.** Our `Task` should not conflate "get the input there"
with "run the thing". The deliverable-2 design already gestures at this with
inputs "by value when small, by reference when not" — Golem's split makes that a
first-class distinction rather than a size heuristic.

---

## 2. Avoid

### 2.1 Do not split a model across machines (Petals — verified)

`petals/README.md`:

> Single-batch inference runs at up to **6 tokens/sec** for **Llama 2** (70B) and
> up to **4 tokens/sec** for **Falcon** (180B)

That is the throughput ceiling of per-token round trips over a WAN, from the
project that did it best. It is the boundary we are staying behind and the
numbers are worth keeping in the repo so nobody re-proposes it.

### 2.2 Do not let two timeouts be ordered wrongly (Darkbloom — the most important finding)

`docs/design/routing-v2-attestation-churn.md` opens with a number that should
change how we build deliverable 4:

> Goal: grow the routable pool (**≈67/176 today**) by making attestation
> resilient

**62% of their fleet was unroutable**, and none of it was hardware. Every root
cause is a timing or state-management bug in exactly the mechanism our lease
design depends on:

| Their root cause | Our equivalent risk | Mitigation for deliverable 4 |
|---|---|---|
| **A.** Push expiry `60s` < response timeout `90s` — the coordinator waited for a reply that could no longer arrive | Lease TTL shorter than the task's own timeout, so a task is re-queued while still running, then completed twice | Assert `lease_ttl > task_timeout + slack` **in code**, not in a comment. Make it a startup check that refuses to run. |
| **B.** Round-trip was connection-scoped; any reconnect stranded the attempt | A provider that reconnects mid-task cannot complete a lease it still legitimately holds | Leases are rows, not connections. Completion must be accepted on **any** authenticated connection from the lease holder. Our pull model makes this natural — do not undo it with per-connection state. |
| **C.** Retry spacing reused a 20-minute device cooldown constant, stranding providers 20–60 min | One constant serving two purposes — e.g. poll interval doubling as backoff | Separate constants for poll interval, claim retry, and provider cooldown. Never reuse one for another. |
| **D.** In-memory throttle wiped on every deploy → post-deploy push storm | Lease or claim state held in process memory, lost on restart, causing a thundering herd | Lease state lives in Postgres. Keep it there. No in-process claim cache. |
| **E.** Token arrived only in the registration message; late/headless nodes were never challenged and never re-armed | A provider whose capability changes after registration is never re-evaluated | Carry capability in the **poll** request, not only at registration. Every poll is a re-registration. |
| **F.** No observability in the code-attest path | Silent lease churn | Ship `task_claim_total{outcome=...}` from the first commit, with outcomes mirroring theirs: `claimed`, `expired`, `completed`, `rejected`, `timeout`, `divergent`. |

Their Fix 4 is the one to internalise: widening challenge freshness from 6 to 16
minutes to stop *"single-missed-tick routable flapping"*. **Lease TTL must be
comfortably larger than the poll interval**, for the same reason PAIR tolerates
three missed announcements before eviction.

### 2.3 Do not describe a coordinator as trustless when it is not (Darkbloom)

Darkbloom's coordinator runs in a GCP AMD SEV Confidential VM and *still* says
plainly that it decrypts request bodies in memory to route and bill. We have no
CVM and no attestation in Phase 0. Any wording that implies Relay's coordinator
or its nodes cannot see task payloads would be false. Deliverable 5's honesty
requirement about determinism limits should extend to this.

### 2.4 Do not assume a scheduler's view is shared (PAIR)

From the scheduler limitations section — unusually candid and worth reading in
full as a model for our own docs:

> **Every node ranks from its own view, and views lag.** There is no shared
> schedule. Two nodes dispatching at the same moment can briefly steer work to
> the same idle peer... The system self-corrects through workload relay and
> periodic reconciliation rather than preventing the collision.

They patch the local case with per-proxy reservations, which are explicitly local
only.

**Our pull-based queue makes this class of bug impossible**, and that is a
genuine architectural win worth stating in deliverable 4's design doc: the queue
is the single source of truth, a claim is a row-level transaction, and no two
providers can take the same task. PAIR had to build reservations *because* it
pushes. We do not, *because* we pull.

---

## 3. Open questions

1. **What is a truthful CPU-second?** (blocks deliverable 3.) Pricing per
   CPU-second requires a measurement the consumer can trust, and 1.11 says a
   node's self-report is not one. Options: price per task only in Phase 0;
   derive a normalised unit from redundant execution; or accept self-reporting
   with the dispute mechanism as the backstop. **Needs a decision before the
   offer schema is extended.**
2. **Can `python_exec` honestly be `deterministic: true`?** See §4.1 — I believe
   not, on a subprocess executor.
3. **What do the sandbox vendors' threat models actually say?** Unverifiable from
   here, and the most valuable single artifact identified in the prompt. Worth a
   human hour.
4. **Why did Golem's utilisation stay low?** Unverifiable from here. This is the
   cautionary question and it remains open. If the answer is "demand never
   materialised for generic compute", it bears directly on whether task dispatch
   for *agents specifically* is a different market or the same one.
5. **Does Akash's ISO decision apply to a CPU-only sandbox?** Entirely
   unverifiable — I could not confirm they ship an ISO at all. The underlying
   question stands on its own merits though: is a normal installed application
   sufficient isolation for running strangers' code on a machine its owner is
   also using, or does safe sharing require owning the whole OS?
6. **How does a provider prove its runtime version?** Deliverable 5 requires a
   "pinned-runtime requirement" for determinism, but per 1.11 the node's claim
   about its own Python version is an assertion. Redundant execution detects
   divergence but cannot attribute it. Is version pinning enforceable in Phase 0,
   or is it documentation?

---

## 4. Direct conflicts with the plan — stopping for a decision

### 4.1 CONFLICT: `python_exec` as a deterministic task type

**Deliverable 2** lists `python_exec` among the Phase 0 task types. **Deliverable
5** builds the entire defensibility argument on running a fraction of tasks twice
and hash-comparing, with `output_divergence` as an *objectively decidable*
dispute that slashes a provider.

Prior art says these two cannot both be true for `python_exec` on a subprocess
executor:

- Golem reached for **WASM with fuel metering** for exactly this workload, and
  their README is explicit that fuel bounds computation *because* wall-clock does
  not.
- CPython on two different machines differs in ways the task author does not
  control: `PYTHONHASHSEED` randomisation affects set and dict iteration order in
  output; float repr and libm differences across platforms; locale-dependent
  formatting; library version drift; and anything touching `time`, `random`, or
  the filesystem.

The risk is not that verification is imperfect. It is that **we would slash an
honest provider** for a divergence caused by their Python patch release. That
turns our headline differentiator into a mechanism that punishes good actors,
and a reputation system that does that is worse than none.

**Options:**

- **(a)** Ship `python_exec` as `deterministic: false` in Phase 0 — recorded,
  billed, never adjudicated — and make `text_transform` and `data_reduce` the
  deterministic ones, since both can be specified over bytes with no float or
  hash-order surface.
- **(b)** Keep `python_exec` deterministic but constrain it hard: no float
  formatting in output, `PYTHONHASHSEED=0`, sorted collections enforced by the
  canonicaliser, pinned interpreter, stdlib-only. Narrow, but honest.
- **(c)** Bring WASM into Phase 0 rather than deferring it, accepting the scope
  increase, because fuel metering solves determinism *and* the CPU-second
  question in 3.1 at once.

**My recommendation is (a) for the spike, with (b) as the Phase 1 target and (c)
named as the eventual answer.** This keeps the verification story true on day
one for the two task types where it genuinely holds, and it means the first
`output_divergence` we adjudicate is a real one. **I have not implemented
anything yet and am waiting on this decision.**

### 4.2 CONFLICT (soft): "no consumer-side dead-provider detection needed"

**Deliverable 4** states lease expiry *is* the failover mechanism, so no
consumer-side detection is needed. Structurally correct. But §2.2 shows lease
timing is precisely where a comparable system lost 62% of its fleet, and §1.5
shows PAIR deliberately checks cheaper evidence of life *before* acting on a
missed signal.

**Proposed amendment, not a contradiction:** keep lease expiry as the only
*mechanism*, but record `last_completed_at` per provider and never reap a lease
from a provider that completed another task within the current lease window.
Cheap, and it prevents the "evict the busiest node" failure PAIR calls out
explicitly.

### 4.3 Note, not a conflict: the broker cancellation is well supported

The prompt cancels the dial-out broker in favour of pull. Darkbloom independently
arrived at outbound-only for the same reason — from `Layr-Labs/d-inference`
`README.md`:

> Providers connect *outbound* over WebSocket — **no port forwarding or inbound
> firewall changes are needed**

They use a persistent WebSocket; we use polling over the existing signed-request
path. Ours is weaker on latency and stronger on simplicity and on surviving
flaky links, which suits latency-tolerant task work. **The cancellation is
correct and now has independent support from two directions.**

### 4.4 Opportunity the plan does not claim: we already beat Darkbloom on settlement

`docs/consumer/verification.md` and the privacy doc list, under *what you cannot
rely on*:

> There is no per-response signature or receipt: the `X-Provider-*` headers are
> the coordinator's assertion over TLS, not a provider-signed proof.

Relay already has provider-signed, consumer-countersigned receipts with request
and output hashes, plus a double-entry ledger. Extending them to tasks
(deliverable 5) is a small generalisation of something that exists, and it is a
capability a well-funded adjacent team explicitly does not have. Worth saying out
loud in the pivot's positioning.

---

## Cross-references for later deliverables

| Deliverable | Sections to revisit |
|---|---|
| 2 — task model | 1.2, 1.3, 1.13, 4.1 |
| 3 — offers and pricing | 1.3, 1.12, 3.1 |
| 4 — pull queue and leases | 1.5, 2.2, 2.4, 4.2 |
| 5 — verification and disputes | 1.11, 1.12, 4.1, 4.4 |
| 6 — executor and operator consent | 1.7, 1.9, 1.10, 1.12, 3.5 |
