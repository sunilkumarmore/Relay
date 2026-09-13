# Relay

Relay is a peer-to-peer marketplace for AI compute.

**Providers** sell inference and *tasks*. **Consumers** buy them to run agents.
Settlement runs on signed receipts, and a job survives losing either the machine
running it or the provider serving it.

The load-bearing idea is that an agent's progress lives outside the process
running it. That is what makes the process disposable — and a disposable process
is a rentable one. If you cannot evict a tenant safely, you cannot rent to them.

**Selling inference needs a GPU and a model. Selling tasks needs neither.** A
task is the work between an agent's thinking steps — multiplying a block of a
matrix, transforming text, reducing numbers — and any machine that runs Python
and can reach the internet can do it. There is no server on a provider and no
port to open: a node polls for work, leases it, does it, and posts a signed
result, all outbound. That is what puts a laptop, a Raspberry Pi or a phone on
the supply side.

```bash
python -m relay.tasks demo     # a matmul split across devices, one killed mid-job
```

See **[docs/DEMO.md](docs/DEMO.md)** to run it across your own machines.

## How it fits together

| Tier | What it does | Where |
|---|---|---|
| Provider | Serves inference, publishes signed offers, issues receipts | `relay/provider/` |
| Consumer | Picks a provider, runs the agent, checks and pays the bill | `relay/consumer/`, `relay/worker/` |
| Directory & ledger | Offers, receipts, credits, reputation | Supabase, behind `relay/store.py` |
| Observability | Live terminal dashboard | `relay/dashboard/tui.py` |

What holds it together:

- **Identity** (`relay/identity.py`) — every node is an Ed25519 keypair; the
  public key *is* the node id. `WORKER_ID` is a label and authorizes nothing.
- **Signed requests** (`relay/auth.py`) — every call carries a signature over
  the method, path, timestamp and body hash.
- **Offers** (`relay/provider/offers.py`) — signed, expiring statements of
  model, context window, price and capacity.
- **Receipts** (`relay/receipts.py`) — the provider signs what it did; the
  consumer checks it against the offer and its own token count, then
  countersigns. An unchecked receipt is never paid.
- **Ledger** (`relay/ledger.py`) — double entry. Every transaction sums to
  zero, in the application and in a Postgres trigger.
- **Reputation** (`relay/reputation.py`) — a deterministic function of public
  rows. Recompute it yourself and compare.
- **Agent state** (`relay/agent/`) — the conversation, intermediate results and
  artifacts, hashed and checkpointed after every step, so a resumed run picks up
  the agent rather than just the step counter.
- **Token authority** (`relay/authority/`) — turns a node's Ed25519 identity
  into a short-lived database token, so row-level security can be written
  against *which node* is asking.

Two ways to run it: **pinned**, where a worker is pointed at one provider (the
two-machine demo below), or **market**, where it discovers providers, picks by
policy, and fails over.

## Prerequisites

Both machines need:
- Python 3.11+
- `pip install -e .`
- Supabase account and project (free tier is fine)
- Both machines reachable on the same network

Machine 1 (inference node) also needs:
- [Ollama](https://ollama.com/download) installed
- `ollama pull llama3`

## Supabase Setup

1. Create a project at [supabase.com](https://supabase.com)
2. Open the SQL Editor
3. Run `setup/supabase_setup.sql`, then migrations `002` through `008` **in order**
4. Copy your project URL and anon key

### Row-level security needs the token authority

Migrations 002 and 008 turn on RLS. From that point the anon key reaches almost
nothing on its own: every policy is written against a `relay_node_id` claim, and
only the token authority can mint one. **Apply those migrations without running
an authority and your nodes will be locked out of the database.**

The authority is the one component that holds the project's JWT secret. Run it
somewhere your nodes can reach, and nowhere else:

```bash
# On the authority host only
export RELAY_JWT_SECRET=<Supabase → Settings → API → JWT Settings>
python -m relay.authority           # listens on :8790
```

Then point every node at it:

```bash
# On each worker, provider, controller and dashboard
export RELAY_AUTHORITY_URL=http://<authority-host>:8790
```

A node signs a server-issued challenge with its key and gets back a token
carrying its node id, refreshed automatically before it expires. Nothing but the
authority ever sees the JWT secret, and a node only ever receives a credential
for itself.

Check it end to end with:

```bash
python -m relay.controller wallet balance     # any command that touches the store
```

**Running without it.** Leave `RELAY_AUTHORITY_URL` unset and nodes connect with
`SUPABASE_KEY` as before. That is fine for a single-operator demo, and it is the
only thing that works if you have not applied 002 — but it means any holder of
the key can read and write everything.

## Machine 1 Setup (Inference Node)

1. Clone repo and `pip install -e .`
2. Create `.env` from `.env.example`:

```env
NODE_ROLE=inference
MACHINE_ID=machine-1
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3
INFERENCE_REGISTRY=http://localhost:8765
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
```

3. Start Ollama bound to all interfaces so workers on other machines can reach it:

```bash
# Linux / Mac
OLLAMA_HOST=0.0.0.0 ollama serve

# Windows
set OLLAMA_HOST=0.0.0.0
ollama serve
```

4. Start the inference registry:

```bash
python -m relay.inference
```

5. Verify it's running:

```bash
curl http://localhost:8765/health
```

If Machine 2 needs to reach Machine 1 over the network, open ports 8765 and 11434 in your firewall:

```bash
# Linux (ufw)
sudo ufw allow 8765/tcp
sudo ufw allow 11434/tcp

# Mac — no action needed by default
# Windows
netsh advfirewall firewall add rule name="Relay Registry" dir=in action=allow protocol=TCP localport=8765
netsh advfirewall firewall add rule name="Relay Ollama"   dir=in action=allow protocol=TCP localport=11434
```

## Machine 2 Setup (Worker Node)

1. Clone repo and `pip install -e .`
2. Find Machine 1's IP address:

```bash
# Linux / Mac
ip route get 1 | awk '{print $7}'

# Windows
ipconfig
```

3. Create `.env.alpha` and `.env.beta` — `SESSION_ID` is optional (auto-generated if omitted):

`.env.alpha`:
```env
NODE_ROLE=worker
MACHINE_ID=machine-2
WORKER_ID=worker-alpha
INFERENCE_REGISTRY=http://<machine1-ip>:8765
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
```

`.env.beta`:
```env
NODE_ROLE=worker
MACHINE_ID=machine-2
WORKER_ID=worker-beta
INFERENCE_REGISTRY=http://<machine1-ip>:8765
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
```

## Running the Demo

Open 4 terminals on Machine 2:

```bash
# Terminal 1 — worker-alpha
ENV_FILE=.env.alpha python -m relay.worker

# Terminal 2 — worker-beta
ENV_FILE=.env.beta python -m relay.worker

# Terminal 3 — live dashboard
python -m relay.dashboard

# Terminal 4 — controller
python -m relay.controller workers
```

**Windows:**
```bat
set ENV_FILE=.env.alpha && python -m relay.worker
```

Kill `worker-alpha` during a sleep window (`Ctrl+C`), then restart it with the same `.env.alpha`.  
Observe `EVICTED -> MIGRATED -> RESUMED` and continuation from the last checkpoint.

## Using Custom Tasks

By default workers run the built-in math demo problems. To use your own tasks, create a YAML file:

```yaml
# tasks/my-task.yaml
goal: "Review our Q3 roadmap"

steps:
  - topic: "Risk analysis"
    prompt: "List the top 3 risks for shipping Feature A before the auth rewrite. Be specific."

  - topic: "Dependencies"
    prompt: "What external dependencies does Feature B introduce? Summarize each."

  - topic: "Recommendation"
    prompt: "Given the above, recommend a shipping order for Features A and B with justification."
```

Point a worker at it with `TASK_FILE`:

```bash
TASK_FILE=tasks/my-task.yaml ENV_FILE=.env.alpha python -m relay.worker
```

Or add it to your `.env.alpha`:

```env
TASK_FILE=tasks/my-task.yaml
```

See `tasks/example.yaml` for a ready-to-run example.

## Resuming a Session

Each worker prints its `SESSION_ID` on startup. To resume after an eviction, add that ID to the env file:

```env
SESSION_ID=the-uuid-printed-on-first-run
```

The worker will pick up from the last saved checkpoint automatically.

## Controller Commands

```bash
python -m relay.controller status                    # all sessions
python -m relay.controller inference-status          # inference node metrics
python -m relay.controller workers                   # worker state + migration log
python -m relay.controller reset --session <id>      # wipe a session
python -m relay.controller report --session <id>     # print final report
```

## What the Demo Proves

1. A single inference node can serve multiple workers simultaneously.
2. Worker migration survives eviction — state lives in Supabase, not the worker process.
3. Any number of steps are supported via the task YAML format.
4. Dashboard shows real-time request and migration telemetry.
5. Machine + inference node audit trail is persisted per step.

## Writing a task

Steps share one conversation. Each sees what came before, and can quote an
earlier answer directly — which also declares the dependency, so execution
order follows from the prompts:

```yaml
goal: "Review our Q3 roadmap"

steps:
  - topic: "Risk analysis"
    prompt: "List the top 3 risks for shipping Feature A before the auth rewrite."

  - topic: "Dependencies"
    prompt: "What external dependencies does Feature B introduce?"

  - topic: "Recommendation"
    prompt: >
      Given {{ steps.1.solution }} and {{ steps.2.solution }}, recommend a
      shipping order with justification.
```

The model is asked for its working and its answer separately, and they are
stored separately — `{{ steps.1.solution }}` gets the answer alone, not the
answer buried in its reasoning.

When the next prompt would not fit the provider's advertised context window,
older turns are folded into a summary and that compaction becomes part of the
checkpointed state, so a resumed run inherits the same history rather than
rebuilding a different one.

Steps that need nothing from each other can run concurrently with
`RELAY_MAX_PARALLEL`. It defaults to 1, and worth knowing before raising it: the
agent keeps one linear conversation, so a parallel wave sees the history as of
the *start* of the wave rather than as of each other. Concurrency changes what
each step reads, not just how fast it runs.

## Running as a market

The demo above pins a worker to one provider. To run the actual marketplace,
drop `INFERENCE_REGISTRY` and give each side its own configuration.

**As a provider.** Copy `relay-provider.example.yaml` to `relay-provider.yaml`,
set your endpoint and your prices, then:

```bash
python -m relay.controller wallet deposit --amount 100 --dev   # needs RELAY_DEV_MODE=1
python -m relay.controller wallet stake --amount 10            # credits at risk
python -m relay.provider                                       # publishes your offers
python -m relay.controller earnings
```

A provider refuses to start if it advertises a model its backend does not
actually serve, and stops advertising if its stake falls below the floor.

**As a consumer.** State what the job needs in the task file:

```yaml
goal: "Review our Q3 roadmap"

requirements:
  model: llama3
  max_price_out_per_1k: 0.20
  min_context_window: 8192
  budget_credits: 5.0

steps:
  - topic: "Risk analysis"
    prompt: "List the top 3 risks for shipping Feature A."
```

Then run it with no endpoint configured — the worker shops the directory:

```bash
python -m relay.controller wallet deposit --amount 100 --dev
TASK_FILE=tasks/my-task.yaml python -m relay.worker
python -m relay.controller receipts --session <id>
```

The worker holds its budget before starting, settles each step against a
countersigned receipt, and releases whatever it did not spend. If the provider
it chose disappears, it picks another and retries the step — no step is
duplicated or skipped.

**Keeping the market honest:**

```bash
python -m relay.controller reputation recompute      # anyone can run this
python -m relay.controller reputation show <node>    # and check the working
python -m relay.controller disputes ls
python -m relay.controller disputes adjudicate
python -m relay.controller wallet verify             # the ledger sums to zero
```

Selection policies are `cheapest` (default), `fastest`, `round_robin` and
`pinned` — set with `RELAY_POLICY`. See `.env.example` for the rest, and
`docs/MARKETPLACE_PLAN.md` for what is built and what is not.

### What is deliberately not built

- **No payment rail.** Credits are internal; the ledger's `deposit` entry is
  where real money would attach.
- **No decentralized directory.** Supabase is the coordination layer, behind
  one interface so it can be swapped.
- **No human arbitration.** Only hash mismatches and token over-claims are
  adjudicated; subjective quality disputes are recorded and surfaced.
- **Tool use.** The agent carries `tool_state` through checkpoints, but nothing
  calls tools yet.

## Selling tasks from a device with no GPU

```bash
# .env needs only SUPABASE_URL and SUPABASE_KEY
python -m relay.tasks node
```

The node advertises what its operator has allowed — by default `matmul_block`,
`text_transform` and `data_reduce`, all pure arithmetic on data carried inside
the task, touching no filesystem and no network. `python_exec` runs code written
by whoever signed the task; it is **off unless you turn it on**, and turning it
on prints a warning explaining that its resource limits stop a task exhausting
your machine but do not stop it reading files your user account can read. It is
not a security sandbox.

Failure handling is a single mechanism. A node holds a *lease* on the work it is
doing and renews it while it works. A device that is switched off, loses signal
or is killed stops renewing; the lease lapses, the work returns to the queue, and
another device takes it. Nothing detects the death — the absence is the signal.
There is no scheduler process either: every node returns other nodes' expired
leases as a side effect of asking for its own work.

Ordering work:

```bash
python -m relay.tasks submit  --rows 2400 --inner 2400 --cols 2400 --seed 7
python -m relay.tasks status  <job-id>
python -m relay.tasks collect <job-id> --seed 7
```

`collect` recomputes the whole multiplication locally from the seed and compares
it to what the devices returned, byte for byte. It refuses to assemble a job with
a tile missing rather than returning a plausible wrong matrix.

### Why the arithmetic is deliberately slow

The kernel is pure Python and would be far faster with numpy. It does not use
numpy because BLAS reorders and blocks its arithmetic for speed, so two builds
can disagree in the last bit of a float. Pure Python floats are IEEE-754 doubles
whose `*` and `+` are correctly rounded, so a fixed accumulation order gives
**identical bytes on every machine**.

That is what makes a provider checkable at all: an audit re-runs a tile
elsewhere and compares hashes, and it can only mean something if honest machines
are required to agree. Speed is traded for the ability to verify the work.

The same reasoning is why `python_exec` is marked non-deterministic and is never
audited by hash comparison — two honest providers on different Python patch
releases diverge routinely, and slashing on that would punish people for keeping
their machines updated.

## Development

Install with the dev extras and run the checks:

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

Run it as `pytest`, the way CI does — not `python -m pytest`, which silently
adds the working directory to `sys.path` and can hide an import that only
resolves locally.

The test suite needs no Supabase project, no Ollama, and no network. Two
substitutions make that possible:

- `RELAY_STORE=memory|file` swaps Supabase for an in-process or JSON-file store
  (`relay/store.py`). `file` is shared across processes, which is what lets a
  worker be killed in one process and inspected from another.
- `RELAY_BACKEND=fake` swaps Ollama for a deterministic backend
  (`relay/inference/backends.py`) with configurable latency and failure
  injection.

The eviction tests are the ones worth reading first: they run the worker as a
real subprocess, send it `SIGTERM` (graceful) or `SIGKILL` (preemption with no
warning), and assert that a resumed run finishes the task with no duplicated or
skipped steps — see `tests/test_hard_eviction.py`.

## Troubleshooting

**Machine 2 cannot reach Machine 1:**
```bash
curl http://<machine1-ip>:8765/health
```
Check firewall rules and that both machines are on the same network.

**Import errors after upgrading:** the package is now installable — run
`pip install -e .` from the repo root.

**Worker registers but inference fails:**
```bash
curl http://<machine1-ip>:11434/api/tags
```
Make sure Ollama is running with `OLLAMA_HOST=0.0.0.0`.

**Dashboard shows no data:**
- Check `INFERENCE_REGISTRY` value in `.env`
- Check Supabase URL and key

**worker-beta not visible:**
- Verify `.env.beta` uses a unique `WORKER_ID`
- Verify the terminal is using `ENV_FILE=.env.beta`

## File Structure

```
relay/
├── relay/
│   ├── config.py           # env loading
│   ├── identity.py         # Ed25519 node identity
│   ├── auth.py             # signed requests
│   ├── store.py            # Store protocol + Supabase / Memory / File
│   ├── receipts.py         # signed, checkable bills
│   ├── ledger.py           # double-entry credits
│   ├── reputation.py       # deterministic scores
│   ├── tokens.py           # independent token counting
│   ├── disputes.py         # adjudication and stake
│   ├── inference/
│   │   ├── backends.py     # InferenceBackend protocol + Ollama / OpenAI / Fake
│   │   └── registry.py     # compatibility shim over relay/provider
│   ├── provider/
│   │   ├── server.py       # the provider: serves, advertises, bills
│   │   ├── offers.py       # signed, expiring terms
│   │   └── config.py       # relay-provider.yaml
│   ├── consumer/
│   │   ├── market.py       # discovery and selection policies
│   │   └── session.py      # binding, failover, paying
│   ├── agent/
│   │   ├── state.py        # the conversation, hashed and checkpointed
│   │   ├── agent.py        # the step loop and context compaction
│   │   ├── prompts.py      # prompt building, reasoning/solution split
│   │   └── plan.py         # dependencies and execution order
│   ├── worker/
│   │   ├── daemon.py       # the agent loop
│   │   ├── tasks.py        # task YAML loading
│   │   └── eviction.py     # signal handling
│   ├── controller/cli.py
│   └── dashboard/tui.py
├── tests/                  # runs with no network
├── tasks/example.yaml
├── setup/                  # schema + migrations 002-006
├── docs/MARKETPLACE_PLAN.md
├── output/
├── .env.example
└── pyproject.toml
```

The old top-level `worker/`, `inference/`, `controller/` and `dashboard/`
directories still exist as thin shims that forward to the package and emit a
`DeprecationWarning`. They will be removed in the next release.
