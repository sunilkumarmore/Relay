from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

try:
    from crewai import Crew, Process
except ImportError:  # pragma: no cover
    Crew = None  # type: ignore[assignment]
    Process = None  # type: ignore[assignment]

try:
    from langchain_ollama import ChatOllama
except ImportError:  # pragma: no cover
    ChatOllama = None  # type: ignore[assignment]

from agent_config import PROBLEMS, build_math_agent, build_problem_task, get_problem
from checkpoint_client import CheckpointClient, CheckpointWriteResult
from eviction_handler import EvictionSnapshot, install_eviction_handlers


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output"
FINAL_REPORT_PATH = OUTPUT_DIR / "final_report.txt"
TASK_GOAL = "Solve 5 math/logic problems"


@dataclass
class RuntimeConfig:
    supabase_url: str
    supabase_key: str
    ollama_host: str
    ollama_model: str
    session_id: str
    machine_id: str


@dataclass
class RuntimeState:
    session_id: str
    machine_id: str
    steps_completed: int
    next_step_number: int
    next_problem: str
    current_problem: str = ""
    last_output: str = ""


def _normalize_ollama_host(host: str) -> str:
    value = host.strip().rstrip("/")
    if not value.startswith(("http://", "https://")):
        value = f"http://{value}"
    return value


def load_config() -> RuntimeConfig:
    load_dotenv()
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    supabase_key = os.getenv("SUPABASE_KEY", "").strip()
    ollama_host = os.getenv("OLLAMA_HOST", "").strip()
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3").strip() or "llama3"
    session_id = os.getenv("SESSION_ID", "").strip()
    machine_id = os.getenv("MACHINE_ID", "").strip()

    if not session_id:
        print(
            'ERROR: SESSION_ID not set in .env file\n'
            'Generate one: python -c "import uuid; print(uuid.uuid4())"\n'
            "Use the SAME session ID on both machines"
        )
        raise SystemExit(1)

    missing = []
    if not supabase_url:
        missing.append("SUPABASE_URL")
    if not supabase_key:
        missing.append("SUPABASE_KEY")
    if not ollama_host:
        missing.append("OLLAMA_HOST")
    if not machine_id:
        missing.append("MACHINE_ID")

    if missing:
        raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")

    return RuntimeConfig(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        ollama_host=_normalize_ollama_host(ollama_host),
        ollama_model=ollama_model,
        session_id=session_id,
        machine_id=machine_id,
    )


def verify_ollama_connection(host: str) -> None:
    url = f"{host}/api/tags"
    try:
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
    except Exception as exc:
        print(
            f"ERROR: Cannot connect to Ollama at {host}\n"
            "On Machine 1 (Windows): Make sure Ollama is running\n"
            "On Machine 2 (Mac): Make sure OLLAMA_HOST points to\n"
            "Windows machine IP, not localhost\n"
            "Run: ollama serve\n"
            f"Check: curl {host}/api/tags"
        )
        raise SystemExit(1) from exc


def _render_banner(title: str, lines: list[str]) -> None:
    width = max(38, len(title), *(len(line) for line in lines))
    print("╔" + "═" * (width + 2) + "╗")
    print(f"║ {title.ljust(width)} ║")
    for line in lines:
        print(f"║ {line.ljust(width)} ║")
    print("╚" + "═" * (width + 2) + "╝")


def _extract_text(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()
    for attr in ("raw", "output", "content", "text"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(result).strip()


def _build_llm(host: str, model: str) -> Any:
    if ChatOllama is None:
        raise SystemExit("langchain-ollama is not installed")
    try:
        return ChatOllama(model=model, base_url=host)
    except TypeError:
        return ChatOllama(model=model)


def _solve_problem(agent: Any, problem) -> str:
    if Crew is None or Process is None:
        raise SystemExit("crewai is not installed")
    task = build_problem_task(problem, agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    return _extract_text(result)


def _snapshot_from_runtime(runtime: RuntimeState) -> EvictionSnapshot:
    return EvictionSnapshot(
        session_id=runtime.session_id,
        machine_id=runtime.machine_id,
        completed_steps=runtime.steps_completed,
        next_step_number=runtime.next_step_number,
        next_problem=runtime.next_problem,
        current_problem=runtime.current_problem,
        agent_scratchpad=runtime.last_output,
    )


def _persist_snapshot(client: CheckpointClient, snapshot: EvictionSnapshot) -> None:
    client.upsert_state(
        session_id=snapshot.session_id,
        next_step_number=snapshot.next_step_number,
        next_problem=snapshot.next_problem,
        agent_scratchpad=snapshot.agent_scratchpad,
        machine_id=snapshot.machine_id,
    )


def _build_final_report(session_id: str, machine_id: str, checkpoints: list[dict[str, Any]]) -> str:
    lines = [
        f"Session ID: {session_id}",
        f"Status: completed",
        f"Machine completing run: {machine_id}",
        "",
        "=== Final Report ===",
        "",
    ]
    for checkpoint in sorted(checkpoints, key=lambda row: int(row["step_number"])):
        step = int(checkpoint["step_number"])
        topic = get_problem(step).topic
        lines.extend(
            [
                f"--- Problem {step}: {topic} ---",
                f"Machine: {checkpoint.get('machine_id', 'unknown')}",
                "",
                "Problem:",
                str(checkpoint.get("problem", "")),
                "",
                "Solution:",
                str(checkpoint.get("solution", "")),
                "",
                "Reasoning:",
                str(checkpoint.get("reasoning", "")),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = CheckpointClient(config.supabase_url, config.supabase_key)
    try:
        client.verify_connection()
    except Exception as exc:
        print(
            f"ERROR: Cannot connect to Supabase at {config.supabase_url}\n"
            "Check your SUPABASE_URL and SUPABASE_KEY in .env file\n"
            "Make sure your Supabase project is active at supabase.com"
        )
        raise SystemExit(1) from exc

    verify_ollama_connection(config.ollama_host)

    existing_session = client.get_session(config.session_id)
    checkpoints = client.get_checkpoints(config.session_id)
    state = client.get_state(config.session_id)
    solved_steps = {int(row["step_number"]) for row in checkpoints}
    max_solved_step = max(solved_steps) if solved_steps else 0

    if existing_session:
        session_steps_completed = int(existing_session.get("steps_completed", 0) or 0)
        session_steps_completed = max(session_steps_completed, max_solved_step)
        previous_machine = (
            existing_session.get("machine_id")
            or (state.get("machine_id") if state else None)
            or (checkpoints[-1].get("machine_id") if checkpoints else None)
            or "unknown"
        )
        if session_steps_completed > 0:
            next_step = session_steps_completed + 1
            _render_banner(
                "RESUMING FROM CHECKPOINT",
                [
                    f"Session: {config.session_id}",
                    f"Completed: {session_steps_completed} of 5 problems",
                    f"Resuming at: Problem {next_step}",
                    f"Previous machine: {previous_machine}",
                    f"This machine: {config.machine_id}",
                ],
            )
        else:
            _render_banner(
                "STARTING FRESH",
                [
                    f"Session: {config.session_id}",
                    "Task: Solve 5 math/logic problems",
                    f"Machine: {config.machine_id}",
                ],
            )
    else:
        _render_banner(
            "STARTING FRESH",
            [
                f"Session: {config.session_id}",
                "Task: Solve 5 math/logic problems",
                f"Machine: {config.machine_id}",
            ],
        )
        client.create_session(
            session_id=config.session_id,
            task_goal=TASK_GOAL,
            steps_total=5,
            steps_completed=0,
            status="in_progress",
            machine_id=config.machine_id,
        )
        session_steps_completed = 0

    if not existing_session:
        existing_session = client.get_session(config.session_id) or {}

    next_step_from_state = int(state["next_step_number"]) if state and state.get("next_step_number") else 1
    next_step_number = max(next_step_from_state, session_steps_completed + 1)
    next_problem = (
        get_problem(next_step_number).prompt if 1 <= next_step_number <= len(PROBLEMS) else ""
    )

    runtime = RuntimeState(
        session_id=config.session_id,
        machine_id=config.machine_id,
        steps_completed=session_steps_completed,
        next_step_number=next_step_number,
        next_problem=next_problem,
    )

    eviction = install_eviction_handlers(
        snapshot_provider=lambda: _snapshot_from_runtime(runtime),
        persist_snapshot=lambda snapshot: _persist_snapshot(client, snapshot),
    )

    llm = _build_llm(config.ollama_host, config.ollama_model)
    agent = build_math_agent(llm)

    for problem in PROBLEMS:
        if problem.step_number in solved_steps:
            continue

        runtime.next_step_number = problem.step_number
        runtime.next_problem = problem.prompt
        runtime.current_problem = problem.prompt
        print(f"━━━ Problem {problem.step_number} of 5 ━━━")
        print(problem.prompt)

        solution_text = _solve_problem(agent, problem)
        runtime.last_output = solution_text

        print(f"✓ Problem {problem.step_number} solved — saving checkpoint...")
        write_result = client.insert_checkpoint(
            session_id=config.session_id,
            checkpoint_number=problem.step_number,
            machine_id=config.machine_id,
            step_number=problem.step_number,
            problem=problem.prompt,
            solution=solution_text,
            reasoning=solution_text,
        )

        if isinstance(write_result, CheckpointWriteResult) and not write_result.inserted:
            # Duplicate step edge case: silently skip to next unsolved step.
            solved_steps.add(problem.step_number)
            continue

        client.update_session(
            session_id=config.session_id,
            steps_completed=problem.step_number,
            status="in_progress",
            machine_id=config.machine_id,
        )

        next_step = problem.step_number + 1
        runtime.steps_completed = problem.step_number
        runtime.next_step_number = next_step
        runtime.next_problem = get_problem(next_step).prompt if next_step <= len(PROBLEMS) else ""
        runtime.current_problem = ""

        client.upsert_state(
            session_id=config.session_id,
            next_step_number=runtime.next_step_number,
            next_problem=runtime.next_problem,
            agent_scratchpad=solution_text,
            machine_id=config.machine_id,
        )
        solved_steps.add(problem.step_number)
        print("✓ Checkpoint saved to Supabase")

        if problem.step_number < len(PROBLEMS):
            print("   Sleeping 5 seconds before next problem...")
            time.sleep(5)

    final_checkpoints = client.get_checkpoints(config.session_id)
    final_report = _build_final_report(config.session_id, config.machine_id, final_checkpoints)

    client.update_session(
        session_id=config.session_id,
        steps_completed=len(final_checkpoints),
        status="completed",
        final_report=final_report,
        machine_id=config.machine_id,
    )
    client.upsert_state(
        session_id=config.session_id,
        next_step_number=len(PROBLEMS) + 1,
        next_problem="",
        agent_scratchpad="",
        machine_id=config.machine_id,
    )

    FINAL_REPORT_PATH.write_text(final_report, encoding="utf-8")

    summary_lines = [
        "All 5 problems solved.",
    ]
    for row in sorted(final_checkpoints, key=lambda item: int(item["step_number"])):
        solution = str(row.get("solution", "")).replace("\n", " ").strip()
        summary_lines.append(
            f"Problem {row['step_number']}: {row.get('machine_id', 'unknown')} - {solution[:90]}"
        )
    _render_banner("COMPLETED", summary_lines)

    print(f"Final report written to {FINAL_REPORT_PATH}")
    eviction.mark_completed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
