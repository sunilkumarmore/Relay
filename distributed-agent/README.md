# Distributed Agent

Proof-of-concept distributed AI agent system showing stateful agent migration across two machines on the same WiFi network.

Core demonstration:
- Machine 1 (Windows) starts solving 5 math and logic problems.
- The daemon is stopped mid-run.
- Machine 2 (Mac) resumes from Supabase checkpoint state with the same `SESSION_ID`.
- Work is not lost and the machine audit trail shows who solved each problem.

## Section 1 - Prerequisites

- Python 3.11+ installed on Windows and Mac
- Ollama installed on Windows machine only
- Supabase account and project
- Both machines on the same WiFi network

Install model on Windows:

```powershell
ollama pull llama3
```

Make Ollama accept network connections on Windows:

```powershell
$env:OLLAMA_HOST="0.0.0.0"; ollama serve
```

Find Windows machine IP:
1. Open `cmd`
2. Run `ipconfig`
3. Read the `IPv4 Address`

## Section 2 - Supabase Setup

1. Create an account at `https://supabase.com`
2. Create a new project (any name)
3. Wait for provisioning to complete (about 2 minutes)
4. Open `SQL Editor`
5. Paste and run `setup/supabase_setup.sql`
6. Open `Settings > API`
7. Copy Project URL and anon/public key

## Section 3 - Installation (both machines)

```powershell
git clone <your-repo-url>
cd distributed-agent
pip install -r requirements.txt
copy .env.example .env
```

Fill `.env` on both machines:

```env
# Supabase - get these from Supabase dashboard > Settings > API
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key

# Ollama - Machine 2 (Mac) points to Machine 1 (Windows) IP
# Machine 1 uses: http://localhost:11434
# Machine 2 uses: http://<windows-machine-ip>:11434
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3

# Session - use same SESSION_ID on both machines
# Generate once: python -c "import uuid; print(uuid.uuid4())"
# Use the SAME value on both machines for the demo
SESSION_ID=your-session-uuid-here

# Identity - change this per machine
MACHINE_ID=WINDOWS-PC
# On Mac set to: MACHINE_ID=MACBOOK
```

## Section 4 - The Demo (step by step)

Step 1: Generate a session ID (do this once)

```powershell
python -c "import uuid; print(uuid.uuid4())"
```

Copy this value and use it on both machines in `SESSION_ID`.

Step 2: Configure Machine 1 (Windows)

```env
SUPABASE_URL=<your url>
SUPABASE_KEY=<your key>
OLLAMA_HOST=http://localhost:11434
SESSION_ID=<generated uuid>
MACHINE_ID=WINDOWS-PC
```

Step 3: Start Ollama on Machine 1

```powershell
$env:OLLAMA_HOST="0.0.0.0"; ollama serve
```

Step 4: Start agent on Machine 1

```powershell
python agent/daemon.py
```

Watch Problem 1 and Problem 2 solve. When `Sleeping 5 seconds before next problem...` appears, press `Ctrl+C`.

Step 5: Verify checkpoint in Supabase

```powershell
python controller/controller.py status
```

Expected: Problems 1 and 2 complete by `WINDOWS-PC`.

Step 6: Configure Machine 2 (Mac)

```env
SUPABASE_URL=<same url>
SUPABASE_KEY=<same key>
OLLAMA_HOST=http://<windows-ip>:11434
SESSION_ID=<same uuid as Machine 1>
MACHINE_ID=MACBOOK
```

Step 7: Start agent on Machine 2

```bash
python agent/daemon.py
```

Expected: `RESUMING FROM CHECKPOINT` banner, then continuation from Problem 3.

Step 8: View final report

```powershell
python controller/controller.py report
```

Expected: Problems 1-2 from `WINDOWS-PC`, Problems 3-5 from `MACBOOK`.

## Section 5 - What This Proves

- Agent state is not tied to one machine
- Checkpoints survive machine failure
- Any machine with valid credentials and `SESSION_ID` can resume
- `machine_id` proves node-level work ownership
- This is a base pattern for distributed agent compute migration

## Section 6 - Troubleshooting

Mac cannot reach Ollama on Windows:
- Allow port `11434` in Windows Firewall
- Verify from Mac:

```bash
curl http://<windows-ip>:11434/api/tags
```

Supabase connection fails:
- Confirm project is active (free tier can pause)
- Open Supabase dashboard and wake project
- Recheck `SUPABASE_URL` and `SUPABASE_KEY`

Agent gives wrong answers:
- `llama3` quality can vary
- Demo goal is checkpoint continuity, not answer correctness

Session not found on Machine 2:
- Verify `SESSION_ID` is identical on both machines
- Inspect `agent_checkpoints` table in Supabase dashboard

## Commands

```powershell
python controller/controller.py status
python controller/controller.py report
python controller/controller.py reset
python controller/controller.py sessions
```

## File Structure

```text
distributed-agent/
|-- agent/
|   |-- daemon.py
|   |-- agent_config.py
|   |-- checkpoint_client.py
|   `-- eviction_handler.py
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
