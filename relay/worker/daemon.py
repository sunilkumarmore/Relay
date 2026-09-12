"""The worker daemon: runs a multi-step task, checkpointing every step.

The worker holds no durable state of its own. Progress lives in the store, which
is what lets the process be killed at any point and the work be picked up again —
here, or on another machine.

Since Phase 3 it also holds no fixed provider. It states what the job needs,
picks a provider from the directory, and replaces it mid-job if that provider
stops working. INFERENCE_REGISTRY still pins it to one endpoint when set, which
is how the two-machine demo keeps working.
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
from relay.consumer.market import NoProviderAvailable, Requirements, Selector
from relay.consumer.session import MarketSession
from relay.identity import Identity, identity_from_env
from relay.ledger import InsufficientFunds, Ledger
from relay.store import RelayStoreError, Store, store_from_env
from relay.worker.eviction import EvictionManager, EvictionState
from relay.worker.tasks import Problem, Task, default_task, get_problem, load_task

HEARTBEAT_INTERVAL_SECONDS = 30


@dataclass
class WorkerConfig:
    worker_id: str
    session_id: str
    machine_id: str
    identity: Identity
    inference_registry: str = ""
    policy: str = "cheapest"
    pinned_node_id: str | None = None
    failover_cooldown_seconds: int = 120


@dataclass
class Runtime:
    worker_id: str
    session_id: str
    machine_id: str
    inference_node: str
    steps_completed: int
    next_step_number: int
    next_problem: str
    provider_node_id: str = ""


def output_dir() -> Path:
    return Path(config.get("RELAY_OUTPUT_DIR", "output") or "output")


def load_config() -> WorkerConfig:
    config.load_env()

    worker_id = config.get("WORKER_ID")
    machine_id = config.get("MACHINE_ID")

    # SESSION_ID is optional — auto-generate one if not provided.
    session_id = config.get("SESSION_ID")
    if not session_id:
        session_id = str(uuid.uuid4())
        print(f"No SESSION_ID set — generated: {session_id}")

    required = {"WORKER_ID": worker_id, "MACHINE_ID": machine_id}
    if config.get("RELAY_STORE", "supabase").lower() == "supabase":
        required["SUPABASE_URL"] = config.get("SUPABASE_URL")
        required["SUPABASE_KEY"] = config.get("SUPABASE_KEY")

    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")

    pinned_url = config.get("INFERENCE_REGISTRY").rstrip("/")
    pinned_node = config.get("RELAY_PINNED_NODE") or None
    # Naming a node means you want that node, so the policy follows.
    policy = config.get("RELAY_POLICY", "pinned" if pinned_node else "cheapest") or "cheapest"

    return WorkerConfig(
        worker_id=worker_id,
        session_id=session_id,
        machine_id=machine_id,
        identity=identity_from_env(),
        inference_registry=pinned_url,
        policy=policy,
        pinned_node_id=pinned_node,
        failover_cooldown_seconds=config.get_int("RELAY_FAILOVER_COOLDOWN", 120),
    )


def load_job() -> Task:
    task_file = config.get("TASK_FILE")
    if not task_file:
        return default_task()
    try:
        task = load_task(task_file)
    except Exception as exc:
        raise SystemExit(f"Failed to load TASK_FILE '{task_file}': {exc}") from exc
    print(f"Loaded {len(task.steps)} steps from {task_file}")
    return task


def verify_registry(url: str) -> dict[str, Any]:
    try:
        response = requests.get(f"{url}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        print(
            f"ERROR: Cannot reach the provider at {url}\n"
            " On Machine 1: Make sure the provider is running\n"
            "   python -m relay.inference\n"
            " On Machine 2: Check INFERENCE_REGISTRY points to Machine 1 IP\n"
            " Check firewall allows port 8765"
        )
        raise SystemExit(1) from exc


def send_heartbeat(stop_event: threading.Event, session: MarketSession, runtime: Runtime) -> None:
    while not stop_event.is_set():
        session.heartbeat(runtime.steps_completed)
        stop_event.wait(HEARTBEAT_INTERVAL_SECONDS)


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
    task: Task,
    *,
    max_tokens: int = 1200,
    step_sleep: int = 5,
    max_retries: int = 3,
) -> int:
    problems: list[Problem] = task.steps
    steps_total = len(problems)
    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    requirements = Requirements.from_dict(task.requirements)
    ledger = Ledger(store)

    if cfg.inference_registry:
        # Pinned to one endpoint: check it is there before doing anything else,
        # so an unreachable provider is a clear message rather than a stack trace.
        health = verify_registry(cfg.inference_registry)
        if not health.get("ollama_connected", health.get("backend_connected", False)):
            print(
                "ERROR: The provider is up but its inference backend is not.\n"
                " Run: ollama serve\n"
                " Make sure Ollama binds to 0.0.0.0:\n"
                "   OLLAMA_HOST=0.0.0.0 ollama serve"
            )
            return 1

    existing_session = store.get_session(cfg.session_id)
    checkpoints = store.get_checkpoints(cfg.session_id)
    existing_state = store.get_state(cfg.session_id)

    solved_steps = {int(row["step_number"]) for row in checkpoints}
    steps_completed = max(solved_steps) if solved_steps else 0
    if existing_session:
        steps_completed = max(steps_completed, int(existing_session.get("steps_completed") or 0))

    runtime = Runtime(
        worker_id=cfg.worker_id,
        session_id=cfg.session_id,
        machine_id=cfg.machine_id,
        inference_node="unknown",
        steps_completed=steps_completed,
        next_step_number=steps_completed + 1,
        next_problem="",
    )

    def record_switch(from_node: str, to_node: str, reason: str) -> None:
        print(f"Switched provider: {from_node[:12]} -> {to_node[:12]} ({reason})")
        store.insert_migration_event(
            session_id=cfg.session_id,
            worker_id=cfg.worker_id,
            event="provider_switched",
            from_machine=from_node,
            to_machine=to_node,
            step_at_event=runtime.steps_completed,
        )

    session = MarketSession(
        cfg.identity,
        store,
        worker_id=cfg.worker_id,
        session_id=cfg.session_id,
        machine_id=cfg.machine_id,
        requirements=requirements,
        selector=Selector(policy=cfg.policy, pinned_node_id=cfg.pinned_node_id),
        pinned_url=cfg.inference_registry or None,
        cooldown_seconds=cfg.failover_cooldown_seconds,
        max_retries=max_retries,
        on_switch=record_switch,
        # A resumed worker goes back to the provider it was using, if it still qualifies.
        prefer_provider=(existing_state or {}).get("provider_node_id"),
        ledger=ledger,
    )

    if requirements.budget_credits:
        # Commit the budget before starting. A job that cannot pay should not
        # consume a provider's capacity discovering that.
        try:
            ledger.hold(cfg.identity.node_id, requirements.budget_credits, cfg.session_id)
        except InsufficientFunds as exc:
            print(
                f"ERROR: {exc}\n"
                " Top up with: python -m relay.controller wallet deposit --amount N --dev"
            )
            return 1

    try:
        binding = session.bind()
    except (NoProviderAvailable, requests.RequestException) as exc:
        print(
            f"ERROR: No provider could take this job.\n {exc}\n"
            " Either start a provider that meets the job's requirements, or set"
            " INFERENCE_REGISTRY to point at one directly."
        )
        return 1

    runtime.inference_node = binding.inference_node_id
    runtime.provider_node_id = binding.provider_node_id

    store.upsert_session(
        session_id=cfg.session_id,
        worker_id=cfg.worker_id,
        task_goal=task.goal,
        steps_total=steps_total,
        steps_completed=steps_completed,
        status="in_progress",
        current_machine=cfg.machine_id,
        inference_node=runtime.inference_node,
    )
    store.insert_migration_event(
        session_id=cfg.session_id,
        worker_id=cfg.worker_id,
        event="started",
        from_machine=None,
        to_machine=cfg.machine_id,
        step_at_event=steps_completed,
    )

    price_line = (
        f"Price:    {binding.offer.price_in_per_1k}/{binding.offer.price_out_per_1k} per 1k in/out"
        if binding.offer
        else "Price:    direct (pinned endpoint, no offer)"
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
                f"Task:     {task.goal}",
                f"Progress: {steps_completed} of {steps_total}",
                f"Previous: {existing_state.get('machine_id') if existing_state else 'unknown'}",
                f"Current:  {cfg.machine_id}",
                f"Provider: {binding.provider_node_id[:16] or binding.endpoint_url}",
                price_line,
            ],
        )
    else:
        render_banner(
            "STARTING FRESH",
            [
                f"Worker:  {cfg.worker_id}",
                f"Session: {cfg.session_id}",
                f"Task:    {task.goal}",
                f"Steps:   {steps_total}",
                f"Machine: {cfg.machine_id}",
                f"Provider: {binding.provider_node_id[:16] or binding.endpoint_url}",
                price_line,
            ],
        )

    next_step = steps_completed + 1
    runtime.next_problem = get_problem(problems, next_step).prompt if next_step <= steps_total else ""

    heartbeat_stop = threading.Event()
    heartbeat_thread = threading.Thread(
        target=send_heartbeat, args=(heartbeat_stop, session, runtime), daemon=True
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
            provider_node_id=runtime.provider_node_id,
        )
        session.release()
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
            result = session.complete(problem.prompt, max_tokens, problem.step_number)

            # The provider may have changed while this step ran.
            if session.binding is not None:
                runtime.inference_node = session.binding.inference_node_id
                runtime.provider_node_id = session.binding.provider_node_id

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
                provider_node_id=runtime.provider_node_id,
            )
            # No insert_inference_log here: the provider already logged this call
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
            provider_node_id=runtime.provider_node_id,
        )
        store.insert_migration_event(
            session_id=cfg.session_id,
            worker_id=cfg.worker_id,
            event="completed",
            from_machine=cfg.machine_id,
            to_machine=cfg.machine_id,
            step_at_event=steps_total,
        )

        if requirements.budget_credits:
            returned = ledger.release_remaining(cfg.identity.node_id, cfg.session_id)
            print(f"Spent {session.spent} credits; released {round(returned, 6)} back to balance")
        if session.disputed:
            print(f"WARNING: {len(session.disputed)} receipt(s) were not acknowledged")

        print(f"Completed all {steps_total} steps. Report: {report_path}")
        heartbeat_stop.set()
        session.release()
        eviction.mark_done()
        return 0
    finally:
        heartbeat_stop.set()
        # Idempotent on the provider side, so calling it twice is safe.
        session.release()


def main() -> int:
    cfg = load_config()
    task = load_job()

    try:
        store = store_from_env()
    except RelayStoreError as exc:
        print(str(exc))
        return 1
    assert store is not None

    return run_worker(
        cfg,
        store,
        task,
        max_tokens=config.get_int("MAX_TOKENS", 1200),
        step_sleep=config.get_int("STEP_SLEEP_SECONDS", 5),
        max_retries=config.get_int("RELAY_MAX_RETRIES", 3),
    )


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
