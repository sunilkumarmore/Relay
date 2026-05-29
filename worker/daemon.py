from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time
from typing import Any
import uuid

import requests
from dotenv import load_dotenv

try:
    from .checkpoint_client import RelayCheckpointClient, RelayCheckpointError
    from .eviction_handler import EvictionManager, EvictionState
    from .problems import get_default_problems, get_problem, load_problems_from_file
except ImportError:
    from checkpoint_client import RelayCheckpointClient, RelayCheckpointError
    from eviction_handler import EvictionManager, EvictionState
    from problems import get_default_problems, get_problem, load_problems_from_file


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output"


@dataclass
class WorkerConfig:
    supabase_url: str
    supabase_key: str
    inference_registry: str
    worker_id: str
    session_id: str
    machine_id: str


@dataclass
class Runtime:
    worker_id: str
    session_id: str
    machine_id: str
    inference_node: str
    steps_completed: int
    next_step_number: int
    next_problem: str


def load_config() -> WorkerConfig:
    env_file = os.getenv("ENV_FILE", ".env")
    load_dotenv(env_file)

    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    supabase_key = os.getenv("SUPABASE_KEY", "").strip()
    inference_registry = os.getenv("INFERENCE_REGISTRY", "").strip().rstrip("/")
    worker_id = os.getenv("WORKER_ID", "").strip()
    machine_id = os.getenv("MACHINE_ID", "").strip()

    # SESSION_ID is optional — auto-generate one if not provided.
    session_id = os.getenv("SESSION_ID", "").strip()
    if not session_id:
        session_id = str(uuid.uuid4())
        print(f"No SESSION_ID set — generated: {session_id}")

    missing = [
        name
        for name, value in {
            "SUPABASE_URL": supabase_url,
            "SUPABASE_KEY": supabase_key,
            "INFERENCE_REGISTRY": inference_registry,
            "WORKER_ID": worker_id,
            "MACHINE_ID": machine_id,
        }.items()
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")

    return WorkerConfig(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        inference_registry=inference_registry,
        worker_id=worker_id,
        session_id=session_id,
        machine_id=machine_id,
    )


def verify_registry(url: str) -> dict[str, Any]:
    try:
        response = requests.get(f"{url}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        print(
            f"ERROR: Cannot reach Inference Registry at {url}\n"
            " On Machine 1: Make sure registry is running\n"
            "   python inference/registry.py\n"
            " On Machine 2: Check INFERENCE_REGISTRY points to Machine 1 IP\n"
            " Check firewall allows port 8765"
        )
        raise SystemExit(1) from exc


def register_worker(config: WorkerConfig) -> str:
    payload = {
        "worker_id": config.worker_id,
        "session_id": config.session_id,
        "machine_id": config.machine_id,
    }
    response = requests.post(f"{config.inference_registry}/worker/register", json=payload, timeout=10)
    response.raise_for_status()
    body = response.json()
    return str(body.get("inference_node_id", "unknown"))


def send_heartbeat(stop_event: threading.Event, config: WorkerConfig, runtime: Runtime) -> None:
    while not stop_event.is_set():
        payload = {
            "worker_id": config.worker_id,
            "last_checkpoint": runtime.steps_completed,
            "steps_completed": runtime.steps_completed,
        }
        try:
            requests.post(f"{config.inference_registry}/worker/heartbeat", json=payload, timeout=5)
        except Exception:
            pass
        stop_event.wait(30)


def deregister_worker(config: WorkerConfig) -> None:
    payload = {"worker_id": config.worker_id}
    try:
        requests.post(f"{config.inference_registry}/worker/deregister", json=payload, timeout=5)
    except Exception:
        pass


def call_inference(
    config: WorkerConfig,
    runtime: Runtime,
    prompt: str,
    max_tokens: int,
    *,
    max_retries: int = 3,
) -> dict[str, Any]:
    payload = {
        "worker_id": config.worker_id,
        "session_id": config.session_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
    }
    url = f"{config.inference_registry}/inference/complete"
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=240)
            if response.status_code == 401:
                register_worker(config)
                response = requests.post(url, json=payload, timeout=240)
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code < 500:
                raise
            last_exc = exc
        if attempt < max_retries:
            wait = 2 ** attempt
            print(f"Inference attempt {attempt + 1}/{max_retries + 1} failed, retrying in {wait}s: {last_exc}")
            time.sleep(wait)
    raise RuntimeError(f"Inference failed after {max_retries + 1} attempts") from last_exc


def render_banner(title: str, lines: list[str]) -> None:
    width = max(40, len(title), *(len(item) for item in lines))
    print("+" + "-" * (width + 2) + "+")
    print(f"| {title.ljust(width)} |")
    for line in lines:
        print(f"| {line.ljust(width)} |")
    print("+" + "-" * (width + 2) + "+")


def build_report(client: RelayCheckpointClient, session_id: str) -> str:
    rows = client.get_checkpoints(session_id)
    lines = [f"Session: {session_id}", "", "=== Worker Report ===", ""]
    for row in rows:
        lines.extend(
            [
                f"Step {row['step_number']} ({row['worker_id']}):",
                str(row.get("solution", "")),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tasks — from TASK_FILE if set, otherwise use built-in demo problems.
    task_file = os.getenv("TASK_FILE", "").strip()
    if task_file:
        try:
            task_goal, problems = load_problems_from_file(task_file)
            print(f"Loaded {len(problems)} steps from {task_file}")
        except Exception as exc:
            raise SystemExit(f"Failed to load TASK_FILE '{task_file}': {exc}") from exc
    else:
        task_goal, problems = get_default_problems()

    steps_total = len(problems)
    max_tokens = int(os.getenv("MAX_TOKENS", "1200"))
    step_sleep = int(os.getenv("STEP_SLEEP_SECONDS", "5"))

    client = RelayCheckpointClient(config.supabase_url, config.supabase_key)
    try:
        client.verify_connection()
    except RelayCheckpointError as exc:
        print(str(exc))
        return 1

    health = verify_registry(config.inference_registry)
    if not health.get("ollama_connected", False):
        print(
            f"ERROR: Cannot connect to Ollama at {os.getenv('OLLAMA_HOST', 'http://localhost:11434')}\n"
            " Run: ollama serve\n"
            " Make sure Ollama binds to 0.0.0.0:\n"
            "   OLLAMA_HOST=0.0.0.0 ollama serve"
        )
        return 1

    inference_node = register_worker(config)

    existing_session = client.get_session(config.session_id)
    checkpoints = client.get_checkpoints(config.session_id)
    existing_state = client.get_state(config.session_id)

    solved_steps = {int(row["step_number"]) for row in checkpoints}
    steps_completed = max(solved_steps) if solved_steps else 0
    if existing_session:
        steps_completed = max(steps_completed, int(existing_session.get("steps_completed") or 0))

    client.upsert_session(
        session_id=config.session_id,
        worker_id=config.worker_id,
        task_goal=task_goal,
        steps_total=steps_total,
        steps_completed=steps_completed,
        status="in_progress",
        current_machine=config.machine_id,
        inference_node=inference_node,
    )

    client.insert_migration_event(
        session_id=config.session_id,
        worker_id=config.worker_id,
        event="started",
        from_machine=None,
        to_machine=config.machine_id,
        step_at_event=steps_completed,
    )

    if steps_completed > 0:
        if existing_state and existing_state.get("worker_id") != config.worker_id:
            client.insert_migration_event(
                session_id=config.session_id,
                worker_id=config.worker_id,
                event="migrated",
                from_machine=existing_state.get("machine_id"),
                to_machine=config.machine_id,
                step_at_event=steps_completed,
            )

        client.insert_migration_event(
            session_id=config.session_id,
            worker_id=config.worker_id,
            event="resumed",
            from_machine=existing_state.get("machine_id") if existing_state else None,
            to_machine=config.machine_id,
            step_at_event=steps_completed,
        )

        render_banner(
            "RESUMING FROM CHECKPOINT",
            [
                f"Worker:   {config.worker_id}",
                f"Session:  {config.session_id}",
                f"Task:     {task_goal}",
                f"Progress: {steps_completed} of {steps_total}",
                f"Previous: {existing_state.get('machine_id') if existing_state else 'unknown'}",
                f"Current:  {config.machine_id}",
            ],
        )
    else:
        render_banner(
            "STARTING FRESH",
            [
                f"Worker:  {config.worker_id}",
                f"Session: {config.session_id}",
                f"Task:    {task_goal}",
                f"Steps:   {steps_total}",
                f"Machine: {config.machine_id}",
                f"Inference node: {inference_node}",
            ],
        )

    next_step = steps_completed + 1
    next_problem_prompt = ""
    if next_step <= steps_total:
        next_problem_prompt = get_problem(problems, next_step).prompt

    runtime = Runtime(
        worker_id=config.worker_id,
        session_id=config.session_id,
        machine_id=config.machine_id,
        inference_node=inference_node,
        steps_completed=steps_completed,
        next_step_number=next_step,
        next_problem=next_problem_prompt,
    )

    heartbeat_stop = threading.Event()
    heartbeat_thread = threading.Thread(
        target=send_heartbeat,
        args=(heartbeat_stop, config, runtime),
        daemon=True,
    )
    heartbeat_thread.start()

    def on_evict(state: EvictionState) -> None:
        print("Saving worker state...")
        client.insert_migration_event(
            session_id=state.session_id,
            worker_id=state.worker_id,
            event="evicted",
            from_machine=state.machine_id,
            to_machine=None,
            step_at_event=state.steps_completed,
        )
        client.upsert_state(
            session_id=state.session_id,
            worker_id=state.worker_id,
            next_step_number=state.next_step_number,
            next_problem=state.next_problem,
            machine_id=state.machine_id,
            inference_node=runtime.inference_node,
            status="evicted",
        )
        deregister_worker(config)
        heartbeat_stop.set()

    eviction = EvictionManager(
        state_provider=lambda: EvictionState(
            session_id=runtime.session_id,
            worker_id=runtime.worker_id,
            machine_id=runtime.machine_id,
            steps_completed=runtime.steps_completed,
            next_step_number=runtime.next_step_number,
            next_problem=runtime.next_problem,
        ),
        on_evict=on_evict,
    )
    eviction.install()

    try:
        for problem in problems:
            if problem.step_number in solved_steps:
                continue

            runtime.next_step_number = problem.step_number
            runtime.next_problem = problem.prompt

            print(f"--- [{config.worker_id}] Step {problem.step_number}/{steps_total}: {problem.topic} ---")
            result = call_inference(config, runtime, problem.prompt, max_tokens)
            latency_ms = int(result.get("latency_ms") or 0)
            tokens_used = int(result.get("tokens_used") or 0)
            solution = str(result.get("response", "")).strip()
            print(f"Inference latency: {latency_ms} ms")

            inserted = client.insert_checkpoint(
                session_id=config.session_id,
                worker_id=config.worker_id,
                step_number=problem.step_number,
                problem=problem.prompt,
                solution=solution,
                reasoning=solution,
                machine_id=config.machine_id,
                inference_node=runtime.inference_node,
                inference_latency_ms=latency_ms,
                tokens_used=tokens_used,
            )

            if not inserted:
                solved_steps.add(problem.step_number)
                continue

            next_step_num = problem.step_number + 1
            next_problem_text = ""
            if next_step_num <= steps_total:
                next_problem_text = get_problem(problems, next_step_num).prompt

            client.update_session(
                config.session_id,
                steps_completed=problem.step_number,
                status="in_progress",
                current_machine=config.machine_id,
                inference_node=runtime.inference_node,
            )
            client.upsert_state(
                session_id=config.session_id,
                worker_id=config.worker_id,
                next_step_number=next_step_num,
                next_problem=next_problem_text,
                machine_id=config.machine_id,
                inference_node=runtime.inference_node,
                status="active",
            )
            client.insert_inference_log(
                worker_id=config.worker_id,
                session_id=config.session_id,
                inference_node=runtime.inference_node,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                success=True,
            )

            runtime.steps_completed = problem.step_number
            runtime.next_step_number = next_step_num
            runtime.next_problem = next_problem_text
            solved_steps.add(problem.step_number)

            print("Checkpoint saved")
            if problem.step_number < steps_total and step_sleep > 0:
                print(f"Sleeping {step_sleep} seconds...")
                time.sleep(step_sleep)

        report = build_report(client, config.session_id)
        report_path = OUTPUT_DIR / f"final_report_{config.worker_id}_{config.session_id}.txt"
        report_path.write_text(report, encoding="utf-8")

        client.update_session(
            config.session_id,
            steps_completed=steps_total,
            status="completed",
            final_report=report,
            current_machine=config.machine_id,
            inference_node=runtime.inference_node,
        )
        client.upsert_state(
            session_id=config.session_id,
            worker_id=config.worker_id,
            next_step_number=steps_total + 1,
            next_problem="",
            machine_id=config.machine_id,
            inference_node=runtime.inference_node,
            status="completed",
        )
        client.insert_migration_event(
            session_id=config.session_id,
            worker_id=config.worker_id,
            event="completed",
            from_machine=config.machine_id,
            to_machine=config.machine_id,
            step_at_event=steps_total,
        )

        print(f"Completed all {steps_total} steps. Report: {report_path}")
        heartbeat_stop.set()
        deregister_worker(config)
        eviction.mark_done()
        return 0
    finally:
        heartbeat_stop.set()
        deregister_worker(config)  # no-op if already called above; safe since the endpoint is idempotent


if __name__ == "__main__":
    raise SystemExit(main())
