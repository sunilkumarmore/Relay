from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import threading
import time
from typing import Any

import requests
from dotenv import load_dotenv

try:
    from .checkpoint_client import RelayCheckpointClient, RelayCheckpointError
    from .eviction_handler import EvictionManager, EvictionState
    from .problems import PROBLEMS, get_problem
except ImportError:
    from checkpoint_client import RelayCheckpointClient, RelayCheckpointError
    from eviction_handler import EvictionManager, EvictionState
    from problems import PROBLEMS, get_problem


TASK_GOAL = "Solve 5 complex math and logic problems"
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
    session_id = os.getenv("SESSION_ID", "").strip()
    machine_id = os.getenv("MACHINE_ID", "").strip()

    missing = [
        name
        for name, value in {
            "SUPABASE_URL": supabase_url,
            "SUPABASE_KEY": supabase_key,
            "INFERENCE_REGISTRY": inference_registry,
            "WORKER_ID": worker_id,
            "SESSION_ID": session_id,
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
            " Check Windows Firewall allows port 8765"
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


def call_inference(config: WorkerConfig, runtime: Runtime, prompt: str, max_tokens: int = 1200) -> dict[str, Any]:
    payload = {
        "worker_id": config.worker_id,
        "session_id": config.session_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
    }
    response = requests.post(f"{config.inference_registry}/inference/complete", json=payload, timeout=240)
    if response.status_code == 401:
        register_worker(config)
        response = requests.post(f"{config.inference_registry}/inference/complete", json=payload, timeout=240)
    response.raise_for_status()
    return response.json()


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
                f"Problem {row['step_number']} ({row['worker_id']}):",
                str(row.get("solution", "")),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
            "   set OLLAMA_HOST=0.0.0.0\n"
            "   ollama serve"
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
        task_goal=TASK_GOAL,
        steps_total=5,
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
                f"Worker: {config.worker_id}",
                f"Session: {config.session_id}",
                f"Completed: {steps_completed} of 5",
                f"Previous machine: {existing_state.get('machine_id') if existing_state else 'unknown'}",
                f"Current machine: {config.machine_id}",
            ],
        )
    else:
        render_banner(
            "STARTING FRESH",
            [
                f"Worker: {config.worker_id}",
                f"Session: {config.session_id}",
                f"Machine: {config.machine_id}",
                f"Inference node: {inference_node}",
            ],
        )

    next_step = steps_completed + 1
    runtime = Runtime(
        worker_id=config.worker_id,
        session_id=config.session_id,
        machine_id=config.machine_id,
        inference_node=inference_node,
        steps_completed=steps_completed,
        next_step_number=next_step,
        next_problem=get_problem(next_step).prompt if next_step <= 5 else "",
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
        for problem in PROBLEMS:
            if problem.step_number in solved_steps:
                continue

            runtime.next_step_number = problem.step_number
            runtime.next_problem = problem.prompt

            print(f"--- [{config.worker_id}] Problem {problem.step_number} of 5 ---")
            result = call_inference(config, runtime, problem.prompt)
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
                next_step_number=problem.step_number + 1,
                next_problem=get_problem(problem.step_number + 1).prompt if problem.step_number < 5 else "",
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
            runtime.next_step_number = problem.step_number + 1
            runtime.next_problem = get_problem(runtime.next_step_number).prompt if runtime.next_step_number <= 5 else ""
            solved_steps.add(problem.step_number)

            print("Checkpoint saved")
            if problem.step_number < 5:
                print("Sleeping 5 seconds...")
                time.sleep(5)

        report = build_report(client, config.session_id)
        report_path = OUTPUT_DIR / f"final_report_{config.worker_id}_{config.session_id}.txt"
        report_path.write_text(report, encoding="utf-8")

        client.update_session(
            config.session_id,
            steps_completed=5,
            status="completed",
            final_report=report,
            current_machine=config.machine_id,
            inference_node=runtime.inference_node,
        )
        client.upsert_state(
            session_id=config.session_id,
            worker_id=config.worker_id,
            next_step_number=6,
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
            step_at_event=5,
        )

        print(f"Completed all problems. Report: {report_path}")
        heartbeat_stop.set()
        deregister_worker(config)
        eviction.mark_done()
        return 0
    finally:
        heartbeat_stop.set()


if __name__ == "__main__":
    raise SystemExit(main())

