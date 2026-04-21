# Relay

Relay is a two-tier distributed AI agent proof of concept.

- Inference nodes run the LLM and serve many workers.
- Worker nodes run agent logic and call inference nodes.
- Supabase stores checkpoints and migration state outside both machines.
- Workers can be evicted and resumed without losing progress.

## Section 1 - What is Relay

Relay demonstrates an architecture for a peer-to-peer AI compute marketplace:

- Inference tier: shared LLM service (`inference/registry.py` + Ollama)
- Worker tier: distributed agent workers (`worker/daemon.py`)
- External state tier: Supabase (`relay_*` tables)
- Observability tier: terminal dashboard (`dashboard/dashboard.py`)

## Section 2 - Prerequisites

- Python 3.11+ on both machines
- Ollama installed on Machine 1 only
  - `winget install Ollama.Ollama`
- Pull model on Machine 1:
  - `ollama pull llama3`
- Supabase account and project
- Both machines on same WiFi

## Section 3 - Supabase Setup

1. Create project at `https://supabase.com`
2. Open SQL Editor
3. Run `setup/supabase_setup.sql`
4. Copy project URL and anon key

## Section 4 - Machine 1 Setup (Inference Node)

1. Clone repo
2. `pip install -r requirements.txt`
3. Configure `.env`:
   - `NODE_ROLE=inference`
   - `MACHINE_ID=WINDOWS-INF`
   - `OLLAMA_HOST=http://localhost:11434`
   - `INFERENCE_REGISTRY=http://localhost:8765`
4. Allow registry port:
   - `netsh advfirewall firewall add rule name="Relay Registry" dir=in action=allow protocol=TCP localport=8765`
5. Allow Ollama port:
   - `netsh advfirewall firewall add rule name="Relay Ollama" dir=in action=allow protocol=TCP localport=11434`
6. Start Ollama:
   - `set OLLAMA_HOST=0.0.0.0`
   - `ollama serve`
7. Start inference registry:
   - `python inference/registry.py`
8. Verify:
   - `http://localhost:8765/health`

## Section 5 - Machine 2 Setup (Worker Node)

1. Clone repo
2. `pip install -r requirements.txt`
3. Find Machine 1 IP with `ipconfig`
4. Generate two worker session IDs:
   - `python -c "import uuid; print(uuid.uuid4())"` (run twice)
5. Create `.env.alpha` and `.env.beta`

`.env.alpha`:

```env
NODE_ROLE=worker
MACHINE_ID=WINDOWS-WORKER
WORKER_ID=worker-alpha
SESSION_ID=<first uuid>
INFERENCE_REGISTRY=http://<machine1-ip>:8765
SUPABASE_URL=<same as machine 1>
SUPABASE_KEY=<same as machine 1>
```

`.env.beta`:

```env
NODE_ROLE=worker
MACHINE_ID=WINDOWS-WORKER
WORKER_ID=worker-beta
SESSION_ID=<second uuid>
INFERENCE_REGISTRY=http://<machine1-ip>:8765
SUPABASE_URL=<same as machine 1>
SUPABASE_KEY=<same as machine 1>
```

## Section 6 - Running the Demo

Open 4 terminals on Machine 2:

Terminal 1 (worker-alpha):

```powershell
set ENV_FILE=.env.alpha
python worker/daemon.py
```

Terminal 2 (worker-beta):

```powershell
set ENV_FILE=.env.beta
python worker/daemon.py
```

Terminal 3 (dashboard):

```powershell
python dashboard/dashboard.py
```

Terminal 4 (controller):

```powershell
python controller/controller.py workers
```

Kill `worker-alpha` during a sleep window (`Ctrl+C`), then restart it with same `.env.alpha`.
Observe `EVICTED -> MIGRATED -> RESUMED` and continuation from checkpoint.

## Section 7 - What the Demo Proves

1. Single inference node can serve multiple workers simultaneously.
2. Worker migration survives eviction.
3. State is externalized in Supabase.
4. Dashboard shows real request and migration telemetry.
5. Machine + inference node audit trail is persisted per step.

## Section 8 - Troubleshooting

Machine 2 cannot reach Machine 1:
- `curl http://<machine1-ip>:8765/health`
- Check firewall rules and WiFi network

Worker registers but inference fails:
- Check Ollama on Machine 1
- `curl http://<machine1-ip>:11434/api/tags`

Dashboard shows no data:
- Check `INFERENCE_REGISTRY` value
- Check Supabase URL/key

worker-beta not visible:
- Verify `.env.beta` uses unique `WORKER_ID` and `SESSION_ID`
- Verify terminal uses `set ENV_FILE=.env.beta`

## Commands

```powershell
python controller/controller.py status
python controller/controller.py inference-status
python controller/controller.py workers
python controller/controller.py reset --session <id>
python controller/controller.py report --session <id>
```

## File Structure

```text
relay/
|-- inference/
|   |-- registry.py
|   `-- ollama_client.py
|-- worker/
|   |-- daemon.py
|   |-- problems.py
|   |-- checkpoint_client.py
|   `-- eviction_handler.py
|-- dashboard/
|   `-- dashboard.py
|-- controller/
|   `-- controller.py
|-- setup/
|   `-- supabase_setup.sql
|-- output/
|   `-- .gitkeep
|-- .env.example
|-- requirements.txt
`-- README.md
```
