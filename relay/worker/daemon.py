"""The worker daemon: runs a multi-step task, checkpointing every step.

The worker holds no durable state of its own. Progress lives in the store, which
is what lets the process be killed at any point and the work be picked up again —
here or on another machine.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from relay import config
from relay.auth import RelayAuth
from relay.identity import Identity, identity_from_env
from relay.store import RelayStoreError, Store, store_from_env
from relay.worker.eviction import EvictionManager, EvictionState
from relay.worker.tasks import Problem, get_default_problems, get_problem, load_problems_from_file

HEARTBEAT_INTERVAL_SECONDS = 30
INFERENCE_TIMEOUT_SECONDS = 240


@dataclass
class WorkerConfig:
    inference_registry: str
    worker_id: str
    session_id: str
    machine_id: str
    identity: Identity

    @property
    def auth(self) -> RelayAuth:
        """Signs every call to a provider. The node id is the real identity;
        worker_id and machine_id are just labels."""
        return RelayAuth(self.identity)


@dataclass
class Runtime:
    worker_id: str
    session_id: str
    machine_id: str
    inference_node: str
    steps_completed: int
    next_step_number: int
    next_problem: str


def output_dir() -> Path:
    return Path(config.get("RELAY_OUTPUT_DIR", "output") or "output")


def load_config() -> WorkerConfig:
    config.load_env()

    inference_registry = config.get("INFERENCE_REGISTRY").rstrip("/")
    worker_id = config.get("WORKER_ID")
    machine_id = config.get("MACHINE_ID")

    # SESSION_ID is optional — auto-generate one if not provided.
    session_id = config.get("SESSION_ID")
    if not session_id:
        session_id = str(uuid.uuid4())
        print(f"No SESSION_ID set — generated: {session_id}")

    required = {
        "INFERENCE_REGISTRY": inference_registry,
        "WORKER_ID": worker_id,
        "MACHINE_ID": machine_id,
    }
    if config.get("RELAY_STORE", "supabase").lower() == "supabase":
        required["SUPABASE_URL"] = config.get("SUPABASE_URL")
        required["SUPABASE_KEY"] = config.get("SUPABASE_KEY")

    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")

    return WorkerConfig(
        inference_registry=inference_registry,
        worker_id=worker_id,
        session_id=session_id,
        machine_id=machine_id,
        identity=identity_from_env(),
    )


def load_task() -> tuple[str, list[Problem]]:
    task_file = config.get("TASK_FILE")
    if not task_file:
        return get_default_problems()
    try:
        task_goal, problems = load_problems_from_file(task_file)
    except Exception as exc:
        raise SystemExit(f"Failed to load TASK_FILE '{task_file}': {exc}") from exc
    print(f"Loaded {len(problems)} steps from {task_file}")
    return task_goal, problems


def verify_registry(url: str) -> dict[str, Any]:
    try:
        response = requests.get(f"{url}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        print(
            f"ERROR: Cannot reach Inference Registry at {url}\n"
            " On Machine 1: Make sure registry is running\n"
            "   python -m relay.inference\n"
            " On Machine 2: Check INFERENCE_REGISTRY points to Machine 1 IP\n"
            " Check firewall allows port 8765"
        )
        raise SystemExit(1) from exc


def register_worker(cfg: WorkerConfig) -> str:
    payload = {
        "worker_id": cfg.worker_id,
        "session_id": cfg.session_id,
        "machine_id": cfg.machine_id,
    }
    response = requests.post(
        f"{cfg.inference_registry}/worker/register", json=payload, timeout=10, auth=cfg.auth
    )
    response.raise_for_status()
    return str(response.json().get("inference_node_id", "unknown"))


def send_heartbeat(stop_event: threading.Event, cfg: WorkerConfig, runtime: Runtime) -> None:
    while not stop_event.is_set():
        payload = {
            "worker_id": cfg.worker_id,
            "last_checkpoint": runtime.steps_completed,
            "steps_completed": runtime.steps_completed,
        }
        try:
            requests.post(
                f"{cfg.inference_registry}/worker/heartbeat", json=payload, timeout=5, auth=cfg.auth
            )
        except Exception:
            pass
        stop_event.wait(HEARTBEAT_INTERVAL_SECONDS)


def deregister_worker(cfg: WorkerConfig) -> None:
    try:
        requests.post(
            f"{cfg.inference_registry}/worker/deregister",
            json={"worker_id": cfg.worker_id},
            timeout=5,
            auth=cfg.auth,
        )
    except Exception:
        pass


def call_inference(
    cfg: WorkerConfig,
    runtime: Runtime,
    prompt: str,
    max_tokens: int,
    *,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Post one completion request, retrying transient failures.

    Connection errors, timeouts and 5xx responses are retried with exponential
    backoff. A 4xx is the caller's fault and fails immediately — except 401,
    which means the registry forgot us (it restarted, or pruned us as stale), so
    we re-register once and retry that attempt.
    """
    payload = {
        "worker_id": cfg.worker_id,
        "session_id": cfg.session_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
    }
    url = f"{cfg.inference_registry}/inference/complete"
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                url, json=payload, timeout=INFERENCE_TIMEOUT_SECONDS, auth=cfg.auth
            )
            if response.status_code == 401:
                register_worker(cfg)
                response = requests.post(
                    url, json=payload, timeout=INFERENCE_TIMEOUT_SECONDS, auth=cfg.auth
                )
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code < 500:
                raise
            last_exc = exc
        if attempt < max_retries:
            wait = 2**attempt
            print(
                f"Inference attempt {attempt + 1}/{max_retries + 1} failed, "
                f"retrying in {wait}s: {last_exc}"
            )
            time.sleep(wait)

    raise RuntimeError(f"Inference failed after {max_retries + 1} attempts") from last_exc


def render_banner(title: str, lines: list[str]) -> None:
    width = max(40, len(title), *(len(item) for item in lines))
    print("+" + "-" * (width + 2) + "+")
    print(f"| {title.ljust(width)} |")
    for line in lines:
        print(f"| {line.ljust(width)} |")
    print("+" + "-" * (width + 2) + "+")


def build_report(store: Store, session_id: str) -> str:
    rows = store.get_checkpoints(session_id)
    lines = [f"Session: {session_id}", "", "=== Worker Report ===", ""]
    for row in rows:
        lines.extend([f"Step {row['step_number']} ({row['worker_id']}):", str(row.get("solution", "")), ""])
    return "\n".join(lines).rstrip() + "\n"


def run_worker(
    cfg: WorkerConfig,
    store: Store,
    task_goal: str,
    problems: list[Problem],
    *,
    max_tokens: int = 1200,
    step_sleep: int = 5,
) -> int:
    steps_total = len(problems)
    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    health = verify_registry(cfg.inference_registry)
    if not health.get("ollama_connected", health.get("backend_connected", False)):
        print(
            f"ERROR: Cannot connect to the inference backend at "
            f"{config.get('OLLAMA_HOST', 'http://localhost:11434')}\n"
            " Run: ollama serve\n"
            " Make sure Ollama binds to 0.0.0.0:\n"
            "   OLLAMA_HOST=0.0.0.0 ollama serve"
        )
        return 1

    inference_node = register_worker(cfg)

    existing_session = store.get_session(cfg.session_id)
    checkpoints = store.get_checkpoints(cfg.session_id)
    existing_state = store.get_state(cfg.session_id)

    solved_steps = {int(row["step_number"]) for row in checkpoints}
    steps_completed = max(solved_steps) if solved_steps else 0
    if existing_session:
        steps_completed = max(steps_completed, int(existing_session.get("steps_completed") or 0))

    store.upsert_session(
        session_id=cfg.session_id,
        worker_id=cfg.worker_id,
        task_goal=task_goal,
        steps_total=steps_total,
        steps_completed=steps_completed,
        status="in_progress",
        current_machine=cfg.machine_id,
        inference_node=inference_node,
    )
    store.insert_migration_event(
        session_id=cfg.session_id,
        worker_id=cfg.worker_id,
        event="started",
        from_machine=None,
        to_machine=cfg.machine_id,
        step_at_event=steps_completed,
    )

    if steps_completed > 0:
        if existing_state and existing_state.get("worker_id") != cfg.worker_id:
            store.insert_migration_event(
                session_id=cfg.session_id,
                worker_id=cfg.worker_id,
                event="migrated",
                from_machine=existing_state.get("machine_id"),
                to_machine=cfg.machine_id,
                step_at_event=steps_completed,
            )
        store.insert_migration_event(
            session_id=cfg.session_id,
            worker_id=cfg.worker_id,
            event="resumed",
            from_machine=existing_state.get("machine_id") if existing_state else None,
            to_machine=cfg.machine_id,
            step_at_event=steps_completed,
        )
        render_banner(
            "RESUMING FROM CHECKPOINT",
            [
                f"Worker:   {cfg.worker_id}",
                f"Session:  {cfg.session_id}",
                f"Task:     {task_goal}",
                f"Progress: {steps_completed} of {steps_total}",
                f"Previous: {existing_state.get('machine_id') if existing_state else 'unknown'}",
                f"Current:  {cfg.machine_id}",
            ],
        )
    else:
        render_banner(
            "STARTING FRESH",
            [
                f"Worker:  {cfg.worker_id}",
                f"Session: {cfg.session_id}",
                f"Task:    {task_goal}",
                f"Steps:   {steps_total}",
                f"Machine: {cfg.machine_id}",
                f"Inference node: {inference_node}",
            ],
        )

    next_step = steps_completed + 1
    next_problem_prompt = get_problem(problems, next_step).prompt if next_step <= steps_total else ""

    runtime = Runtime(
        worker_id=cfg.worker_id,
        session_id=cfg.session_id,
        machine_id=cfg.machine_id,
        inference_node=inference_node,
        steps_completed=steps_completed,
        next_step_number=next_step,
        next_problem=next_problem_prompt,
    )

    heartbeat_stop = threading.Event()
    heartbeat_thread = threading.Thread(
        target=send_heartbeat, args=(heartbeat_stop, cfg, runtime), daemon=True
    )
    heartbeat_thread.start()

    def on_evict(state: EvictionState) -> None:
        print("Saving worker state...")
        store.insert_migration_event(
            session_id=state.session_id,
            worker_id=state.worker_id,
            event="evicted",
            from_machine=state.machine_id,
            to_machine=None,
            step_at_event=state.steps_completed,
        )
        store.upsert_state(
            session_id=state.session_id,
            worker_id=state.worker_id,
            next_step_number=state.next_step_number,
            next_problem=state.next_problem,
            machine_id=state.machine_id,
            inference_node=runtime.inference_node,
            status="evicted",
        )
        deregister_worker(cfg)
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

            print(f"--- [{cfg.worker_id}] Step {problem.step_number}/{steps_total}: {problem.topic} ---")
            result = call_inference(cfg, runtime, problem.prompt, max_tokens)
            latency_ms = int(result.get("latency_ms") or 0)
            tokens_used = int(result.get("tokens_used") or 0)
            tokens_in = int(result.get("tokens_in") or 0)
            tokens_out = int(result.get("tokens_out") or 0)
            solution = str(result.get("response", "")).strip()
            print(f"Inference latency: {latency_ms} ms")

            inserted = store.insert_checkpoint(
                session_id=cfg.session_id,
                worker_id=cfg.worker_id,
                step_number=problem.step_number,
                problem=problem.prompt,
                solution=solution,
                reasoning=solution,
                machine_id=cfg.machine_id,
                inference_node=runtime.inference_node,
                inference_latency_ms=latency_ms,
                tokens_used=tokens_used,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )
            if not inserted:
                # Another worker already committed this step; adopt it and move on.
                solved_steps.add(problem.step_number)
                continue

            next_step_num = problem.step_number + 1
            next_problem_text = ""
            if next_step_num <= steps_total:
                next_problem_text = get_problem(problems, next_step_num).prompt

            store.update_session(
                cfg.session_id,
                steps_completed=problem.step_number,
                status="in_progress",
                current_machine=cfg.machine_id,
                inference_node=runtime.inference_node,
            )
            store.upsert_state(
                session_id=cfg.session_id,
                worker_id=cfg.worker_id,
                next_step_number=next_step_num,
                next_problem=next_problem_text,
                machine_id=cfg.machine_id,
                inference_node=runtime.inference_node,
                status="active",
            )
            # No insert_inference_log here: the registry already logged this call
            # when it served it, and it measures latency and tokens at the source.
            # Writing it from both sides double-counted every request.
            runtime.steps_completed = problem.step_number
            runtime.next_step_number = next_step_num
            runtime.next_problem = next_problem_text
            solved_steps.add(problem.step_number)

            print("Checkpoint saved")
            if problem.step_number < steps_total and step_sleep > 0:
                print(f"Sleeping {step_sleep} seconds...")
                time.sleep(step_sleep)

        report = build_report(store, cfg.session_id)
        report_path = out_dir / f"final_report_{cfg.worker_id}_{cfg.session_id}.txt"
        report_path.write_text(report, encoding="utf-8")

        store.update_session(
            cfg.session_id,
            steps_completed=steps_total,
            status="completed",
            final_report=report,
            current_machine=cfg.machine_id,
            inference_node=runtime.inference_node,
        )
        store.upsert_state(
            session_id=cfg.session_id,
            worker_id=cfg.worker_id,
            next_step_number=steps_total + 1,
            next_problem="",
            machine_id=cfg.machine_id,
            inference_node=runtime.inference_node,
            status="completed",
        )
        store.insert_migration_event(
            session_id=cfg.session_id,
            worker_id=cfg.worker_id,
            event="completed",
            from_machine=cfg.machine_id,
            to_machine=cfg.machine_id,
            step_at_event=steps_total,
        )

        print(f"Completed all {steps_total} steps. Report: {report_path}")
        heartbeat_stop.set()
        deregister_worker(cfg)
        eviction.mark_done()
        return 0
    finally:
        heartbeat_stop.set()
        # Idempotent on the registry side, so calling it twice is safe.
        deregister_worker(cfg)


def main() -> int:
    cfg = load_config()
    task_goal, problems = load_task()

    try:
        store = store_from_env()
    except RelayStoreError as exc:
        print(str(exc))
        return 1
    assert store is not None

    return run_worker(
        cfg,
        store,
        task_goal,
        problems,
        max_tokens=config.get_int("MAX_TOKENS", 1200),
        step_sleep=config.get_int("STEP_SLEEP_SECONDS", 5),
    )


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
