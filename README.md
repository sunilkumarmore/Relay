# Relay

Relay is a two-tier distributed AI agent system.

- **Inference nodes** run the LLM and serve many workers.
- **Worker nodes** run agent logic and call inference nodes.
- **Supabase** stores checkpoints and migration state outside both machines.
- Workers can be evicted and resumed without losing progress.

## What is Relay

Relay demonstrates an architecture for a peer-to-peer AI compute marketplace:

- Inference tier: shared LLM service (`inference/registry.py` + Ollama)
- Worker tier: distributed agent workers (`worker/daemon.py`)
- External state tier: Supabase (`relay_*` tables)
- Observability tier: terminal dashboard (`dashboard/dashboard.py`)

## Prerequisites

Both machines need:
- Python 3.11+
- `pip install -r requirements.txt`
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

1. Clone repo and `pip install -r requirements.txt`
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
python inference/registry.py
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

1. Clone repo and `pip install -r requirements.txt`
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
ENV_FILE=.env.alpha python worker/daemon.py

# Terminal 2 — worker-beta
ENV_FILE=.env.beta python worker/daemon.py

# Terminal 3 — live dashboard
python dashboard/dashboard.py

# Terminal 4 — controller
python controller/controller.py workers
```

**Windows:**
```bat
set ENV_FILE=.env.alpha && python worker/daemon.py
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
TASK_FILE=tasks/my-task.yaml ENV_FILE=.env.alpha python worker/daemon.py
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
python controller/controller.py status                    # all sessions
python controller/controller.py inference-status          # inference node metrics
python controller/controller.py workers                   # worker state + migration log
python controller/controller.py reset --session <id>      # wipe a session
python controller/controller.py report --session <id>     # print final report
```

## What the Demo Proves

1. A single inference node can serve multiple workers simultaneously.
2. Worker migration survives eviction — state lives in Supabase, not the worker process.
3. Any number of steps are supported via the task YAML format.
4. Dashboard shows real-time request and migration telemetry.
5. Machine + inference node audit trail is persisted per step.

## Troubleshooting

**Machine 2 cannot reach Machine 1:**
```bash
curl http://<machine1-ip>:8765/health
```
Check firewall rules and that both machines are on the same network.

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
├── inference/
│   ├── registry.py
│   └── ollama_client.py
├── worker/
│   ├── daemon.py
│   ├── problems.py
│   ├── checkpoint_client.py
│   └── eviction_handler.py
├── dashboard/
│   └── dashboard.py
├── controller/
│   └── controller.py
├── tasks/
│   └── example.yaml
├── setup/
│   └── supabase_setup.sql
├── output/
├── .env.example
└── requirements.txt
```
