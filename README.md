# Relay

Relay is a two-tier distributed AI agent system.

- **Inference nodes** run the LLM and serve many workers.
- **Worker nodes** run agent logic and call inference nodes.
- **Supabase** stores checkpoints and migration state outside both machines.
- Workers can be evicted and resumed without losing progress.

## What is Relay

Relay demonstrates an architecture for a peer-to-peer AI compute marketplace:

- Inference tier: shared LLM service (`relay/inference/registry.py` + Ollama)
- Worker tier: distributed agent workers (`relay/worker/daemon.py`)
- External state tier: Supabase (`relay_*` tables)
- Observability tier: terminal dashboard (`relay/dashboard/tui.py`)

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
3. Run `setup/supabase_setup.sql`
4. Copy your project URL and anon key

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

## Development

Install with the dev extras and run the checks:

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

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
│   ├── store.py            # Store protocol + Supabase / Memory / File
│   ├── inference/
│   │   ├── backends.py     # InferenceBackend protocol + Ollama / Fake
│   │   └── registry.py     # the HTTP front door
│   ├── worker/
│   │   ├── daemon.py       # the agent loop
│   │   ├── tasks.py        # task YAML loading
│   │   └── eviction.py     # signal handling
│   ├── controller/cli.py
│   └── dashboard/tui.py
├── tests/                  # runs with no network
├── tasks/example.yaml
├── setup/supabase_setup.sql
├── docs/MARKETPLACE_PLAN.md
├── output/
├── .env.example
└── pyproject.toml
```

The old top-level `worker/`, `inference/`, `controller/` and `dashboard/`
directories still exist as thin shims that forward to the package and emit a
`DeprecationWarning`. They will be removed in the next release.
