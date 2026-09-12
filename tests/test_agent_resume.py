"""Resuming a stateful agent.

Phase 0 proved a job survives losing its worker. Phase 3 proved it survives
losing its provider. Neither was hard while every step was an independent
prompt. This is the one that was: an agent with a conversation, intermediate
results, and steps that depend on each other, resumed into the same state it
would have been in had nothing happened.
"""

from __future__ import annotations

import signal

import yaml

from relay.agent.state import AgentState
from relay.ledger import Ledger


def dependent_task(tmp_path, name="task.yaml", steps=5) -> str:
    """A chain: every step needs the one before it, so nothing can be reordered
    or skipped without the result changing."""
    body = {
        "goal": "Work through a dependent chain",
        "steps": [{"topic": "Step 1", "prompt": "State an opening position."}],
    }
    for i in range(2, steps + 1):
        body["steps"].append(
            {
                "topic": f"Step {i}",
                "prompt": f"Given {{{{ steps.{i - 1}.solution }}}}, take the next step.",
            }
        )
    path = tmp_path / name
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return str(path)


def report_text(tmp_path, session_id):
    matches = list((tmp_path / "output").glob(f"*_{session_id}.txt"))
    assert len(matches) == 1, f"expected one report for {session_id}, found {matches}"
    return matches[0].read_text()


def state_for(store, session_id) -> AgentState:
    row = store.get_agent_state(session_id)
    assert row is not None, "no agent state was saved"
    return AgentState.from_json(row["state_blob"])


def test_kill_9_mid_chain_resumes_to_an_identical_result(
    store, market, spawn_worker, tmp_path
):
    """The acceptance test for stateful checkpointing."""
    # Seven steps at half a second each, killed after two, leaves several
    # seconds of work outstanding — enough margin that a loaded machine cannot
    # finish the run before the kill lands and make the test vacuous.
    steps = 7
    market(store, name="p1", price_out=0.01, latency_ms=500, context_window=100_000)
    task = dependent_task(tmp_path, steps=steps)

    # A clean run, for comparison.
    clean = spawn_worker(
        store=store, registry_url="", session_id="clean", task_path=task, worker_id="w"
    )
    assert clean.proc.wait(timeout=180) == 0, clean.proc.stdout.read()
    expected = report_text(tmp_path, "clean")

    # The same task, killed without warning partway through.
    broken = spawn_worker(
        store=store, registry_url="", session_id="broken", task_path=task, worker_id="w"
    )
    assert broken.wait_for_checkpoints(2), "never got far enough to be worth killing"
    broken.proc.send_signal(signal.SIGKILL)
    assert broken.proc.wait(timeout=30) == -signal.SIGKILL

    committed = len(store.get_checkpoints("broken"))
    assert 0 < committed < steps, "the kill has to land mid-run for this to test anything"

    resumed = spawn_worker(
        store=store, registry_url="", session_id="broken", task_path=task, worker_id="w"
    )
    assert resumed.proc.wait(timeout=180) == 0, resumed.proc.stdout.read()

    actual = report_text(tmp_path, "broken")
    assert actual.replace("broken", "SESSION") == expected.replace("clean", "SESSION")


def test_the_resumed_conversation_is_the_one_it_would_have_had(
    store, market, spawn_worker, tmp_path
):
    """Not just the same answers — the same history behind them."""
    market(store, name="p1", price_out=0.01, latency_ms=500, context_window=100_000)
    task = dependent_task(tmp_path, steps=7)

    clean = spawn_worker(
        store=store, registry_url="", session_id="clean2", task_path=task, worker_id="w"
    )
    assert clean.proc.wait(timeout=120) == 0

    broken = spawn_worker(
        store=store, registry_url="", session_id="broken2", task_path=task, worker_id="w"
    )
    assert broken.wait_for_checkpoints(2)
    broken.proc.send_signal(signal.SIGKILL)
    broken.proc.wait(timeout=30)

    resumed = spawn_worker(
        store=store, registry_url="", session_id="broken2", task_path=task, worker_id="w"
    )
    assert resumed.proc.wait(timeout=180) == 0

    clean_state = state_for(store, "clean2")
    resumed_state = state_for(store, "broken2")
    assert [m.to_dict() for m in resumed_state.messages] == [
        m.to_dict() for m in clean_state.messages
    ]
    assert resumed_state.state_hash() == clean_state.state_hash()


def test_a_later_step_actually_used_the_earlier_answer(store, market, spawn_worker, tmp_path):
    """If the chain were not really connected, the test above would pass anyway."""
    market(store, name="p1", price_out=0.01, context_window=100_000)
    worker = spawn_worker(
        store=store,
        registry_url="",
        session_id="chained",
        task_path=dependent_task(tmp_path, steps=3),
    )
    assert worker.proc.wait(timeout=120) == 0, worker.proc.stdout.read()

    state = state_for(store, "chained")
    first_answer = state.step_outputs[1].solution
    checkpoints = {int(c["step_number"]): c for c in store.get_checkpoints("chained")}

    # The prompt we actually sent for step 2 contains step 1's answer, resolved.
    assert first_answer in checkpoints[2]["problem"]
    assert "{{ steps.1.solution }}" not in checkpoints[2]["problem"]


def test_working_and_answer_are_stored_separately(store, market, spawn_worker, tmp_path):
    """relay_checkpoints has always had both columns and always written the same
    string into both."""
    market(store, name="p1", price_out=0.01, context_window=100_000)
    worker = spawn_worker(
        store=store,
        registry_url="",
        session_id="split",
        task_path=dependent_task(tmp_path, steps=2),
    )
    assert worker.proc.wait(timeout=120) == 0

    for checkpoint in store.get_checkpoints("split"):
        assert checkpoint["solution"], "no answer recorded"
        assert checkpoint["reasoning"], "no working recorded"
        assert checkpoint["solution"] != checkpoint["reasoning"]
        assert checkpoint["topic"]


def test_a_partial_state_is_recorded_on_eviction_and_then_discarded(
    store, market, spawn_worker, tmp_path
):
    market(store, name="p1", price_out=0.01, latency_ms=400, context_window=100_000)
    task = dependent_task(tmp_path)

    worker = spawn_worker(
        store=store, registry_url="", session_id="partial", task_path=task, worker_id="w"
    )
    assert worker.wait_for_checkpoints(1)
    worker.proc.send_signal(signal.SIGTERM)
    assert worker.proc.wait(timeout=30) == 0

    rows = store.list_agent_states("partial")
    assert any(r["status"] == "partial" for r in rows), "nothing recorded about what was in flight"

    resumed = spawn_worker(
        store=store, registry_url="", session_id="partial", task_path=task, worker_id="w"
    )
    assert resumed.proc.wait(timeout=120) == 0

    after = store.list_agent_states("partial")
    assert all(r["status"] == "complete" for r in after), "partial state was never cleared"
    assert [c["step_number"] for c in store.get_checkpoints("partial")] == [1, 2, 3, 4, 5]


def test_independent_steps_run_together_and_dependencies_still_hold(
    store, market, spawn_worker, tmp_path
):
    """Concurrency must respect the graph and stay reproducible.

    It does NOT produce the same answers as running one step at a time: the
    agent keeps one linear conversation, so a parallel wave sees the history as
    of the start of the wave rather than as of each other. That is a real
    trade-off of asking for concurrency, not a defect — see relay/agent/plan.py.
    """
    market(store, name="p1", price_out=0.01, context_window=100_000)
    body = {
        "goal": "Four independent questions, then one that needs them all",
        "steps": [{"topic": f"Q{i}", "prompt": f"Answer question {i}."} for i in range(1, 5)],
    }
    body["steps"].append(
        {
            "topic": "Synthesis",
            "depends_on": [1, 2, 3, 4],
            "prompt": "Reconcile {{ steps.1.solution }} and {{ steps.4.solution }}.",
        }
    )
    path = tmp_path / "parallel.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")

    first = spawn_worker(
        store=store,
        registry_url="",
        session_id="par1",
        task_path=str(path),
        worker_id="w",
        max_parallel=4,
    )
    assert first.proc.wait(timeout=120) == 0, first.proc.stdout.read()

    second = spawn_worker(
        store=store,
        registry_url="",
        session_id="par2",
        task_path=str(path),
        worker_id="w",
        max_parallel=4,
    )
    assert second.proc.wait(timeout=120) == 0, second.proc.stdout.read()

    # Reproducible: the same task at the same concurrency gives the same run,
    # regardless of which step happened to finish first.
    assert state_for(store, "par1").state_hash() == state_for(store, "par2").state_hash()
    assert report_text(tmp_path, "par1").replace("par1", "S") == report_text(
        tmp_path, "par2"
    ).replace("par2", "S")

    # And the dependency held: the synthesis really saw the earlier answers.
    state = state_for(store, "par1")
    checkpoints = {int(c["step_number"]): c for c in store.get_checkpoints("par1")}
    assert state.step_outputs[1].solution in checkpoints[5]["problem"]
    assert state.step_outputs[4].solution in checkpoints[5]["problem"]


def test_parallel_waves_commit_in_step_order(store, market, spawn_worker, tmp_path):
    """Whichever finishes first, the transcript reads 1, 2, 3, 4."""
    market(store, name="p1", price_out=0.01, latency_ms=100, context_window=100_000)
    body = {
        "goal": "Independent questions",
        "steps": [{"topic": f"Q{i}", "prompt": f"Answer question {i}."} for i in range(1, 5)],
    }
    path = tmp_path / "order.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")

    worker = spawn_worker(
        store=store,
        registry_url="",
        session_id="order",
        task_path=str(path),
        worker_id="w",
        max_parallel=4,
    )
    assert worker.proc.wait(timeout=120) == 0, worker.proc.stdout.read()

    state = state_for(store, "order")
    assert [m.step_number for m in state.messages] == [1, 1, 2, 2, 3, 3, 4, 4]
    assert [c["step_number"] for c in store.get_checkpoints("order")] == [1, 2, 3, 4]


def test_a_session_with_no_agent_state_rebuilds_its_conversation(
    store, market, spawn_worker, tmp_path
):
    """Checkpoints outlive the state row — a session that predates agent state,
    or one killed in the window between committing a checkpoint and saving the
    state that records it.

    Recovering the answers alone is not enough: later steps would run against an
    empty history and quietly produce different work than the run would have
    produced uninterrupted. Each checkpoint stores the instruction it was given
    alongside the answer, which is exactly the pair the conversation appends, so
    the transcript can be rebuilt.
    """
    market(store, name="p1", price_out=0.01, context_window=100_000)
    task = dependent_task(tmp_path, steps=5)

    clean = spawn_worker(
        store=store, registry_url="", session_id="whole2", task_path=task, worker_id="w"
    )
    assert clean.proc.wait(timeout=180) == 0, clean.proc.stdout.read()
    expected_answers = [c["solution"] for c in store.get_checkpoints("whole2")]
    expected_state = state_for(store, "whole2")

    partial_run = spawn_worker(
        store=store, registry_url="", session_id="lost", task_path=task, worker_id="w"
    )
    assert partial_run.wait_for_checkpoints(2)
    partial_run.proc.send_signal(signal.SIGKILL)
    partial_run.proc.wait(timeout=30)

    # Reproduce the window deterministically: every agent-state row gone, the
    # checkpoints that were committed still there.
    snapshot = store.snapshot()
    snapshot["agent_state"] = [r for r in snapshot["agent_state"] if r["session_id"] != "lost"]
    store._begin()
    store.tables = snapshot
    store._commit()
    assert store.get_agent_state("lost") is None
    committed = len(store.get_checkpoints("lost"))
    assert 0 < committed < 5

    resumed = spawn_worker(
        store=store, registry_url="", session_id="lost", task_path=task, worker_id="w"
    )
    assert resumed.proc.wait(timeout=180) == 0, resumed.proc.stdout.read()

    # The steps that ran after the loss saw the same history they would have.
    assert [c["solution"] for c in store.get_checkpoints("lost")] == expected_answers
    recovered = state_for(store, "lost")
    assert [m.to_dict() for m in recovered.messages] == [
        m.to_dict() for m in expected_state.messages
    ]


def test_receipts_still_line_up_with_the_steps(store, market, spawn_worker, tmp_path):
    """Phase 4's billing has to survive prompts that now include history."""
    provider = market(store, name="p1", price_out=0.05, context_window=100_000)
    ledger = Ledger(store)
    from relay.identity import Identity

    consumer = Identity.load_or_create(tmp_path / "w.key")
    ledger.deposit(consumer.node_id, 100.0)

    body = yaml.safe_load(open(dependent_task(tmp_path, steps=3)))
    body["requirements"] = {"budget_credits": 50.0}
    path = tmp_path / "billed.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")

    worker = spawn_worker(
        store=store, registry_url="", session_id="billed", task_path=str(path), worker_id="w"
    )
    assert worker.proc.wait(timeout=120) == 0, worker.proc.stdout.read()

    receipts = store.list_receipts(job_id="billed")
    assert {r["step_number"] for r in receipts} == {1, 2, 3}
    assert all(r["status"] == "acknowledged" for r in receipts)
    billed = sum(float(r["amount_credits"]) for r in receipts)
    assert ledger.balance(provider.provider_node_id) == round(billed, 6)
    ledger.check_invariant()


def test_a_checkpoint_with_no_conversation_behind_it_is_redone(
    store, market, spawn_worker, tmp_path
):
    """The crash window between writing a checkpoint and saving the state.

    Skipping that step on the strength of the checkpoint alone would leave its
    turns missing from the transcript for the rest of the run — a hole nothing
    later could fill. Redoing it is free, because committing a step is
    idempotent.
    """
    market(store, name="p1", price_out=0.01, context_window=100_000)
    task = dependent_task(tmp_path, steps=4)

    clean = spawn_worker(
        store=store, registry_url="", session_id="whole", task_path=task, worker_id="w"
    )
    assert clean.proc.wait(timeout=180) == 0, clean.proc.stdout.read()
    expected = state_for(store, "whole")

    torn = spawn_worker(
        store=store, registry_url="", session_id="torn", task_path=task, worker_id="w"
    )
    assert torn.proc.wait(timeout=180) == 0

    # Reproduce the window exactly: the last step's checkpoint survives, the
    # state row recording it does not.
    snapshot = store.snapshot()
    snapshot["agent_state"] = [
        r
        for r in snapshot["agent_state"]
        if not (r["session_id"] == "torn" and int(r["step_number"]) == 4)
    ]
    snapshot["sessions"] = [
        {**r, "status": "in_progress"} if r["session_id"] == "torn" else r
        for r in snapshot["sessions"]
    ]
    store._begin()
    store.tables = snapshot
    store._commit()

    assert store.get_agent_state("torn")["step_number"] == 3
    assert len(store.get_checkpoints("torn")) == 4, "the billing row is still there"

    resumed = spawn_worker(
        store=store, registry_url="", session_id="torn", task_path=task, worker_id="w"
    )
    assert resumed.proc.wait(timeout=180) == 0, resumed.proc.stdout.read()

    recovered = state_for(store, "torn")
    assert [m.to_dict() for m in recovered.messages] == [m.to_dict() for m in expected.messages]
    assert recovered.state_hash() == expected.state_hash()
    # And the step was not billed a second time.
    assert len(store.get_checkpoints("torn")) == 4
