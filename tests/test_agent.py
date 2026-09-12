"""The agent loop: context that flows, and working kept apart from answers."""

from __future__ import annotations

import pytest

from relay.agent.agent import Agent
from relay.agent.plan import PlanError, waves
from relay.agent.prompts import TemplateError, parse_response, references, render
from relay.agent.state import ROLE_SUMMARY, AgentState, StepOutput
from relay.worker.tasks import Problem


def tagged(reasoning: str, solution: str) -> str:
    return f"<reasoning>\n{reasoning}\n</reasoning>\n<solution>\n{solution}\n</solution>"


class Recorder:
    """Stands in for a provider, and remembers what it was asked."""

    def __init__(self, responses=None, *, default="fine"):
        self.prompts: list[str] = []
        self.responses = list(responses or [])
        self.default = default

    def __call__(self, prompt: str, max_tokens: int, step_number: int = 0) -> dict:
        self.prompts.append(prompt)
        body = self.responses.pop(0) if self.responses else tagged("working", self.default)
        return {"response": body, "tokens_in": 10, "tokens_out": 5, "tokens_used": 15, "latency_ms": 1}


# -- parsing ---------------------------------------------------------------


def test_working_and_answer_are_kept_apart():
    parsed = parse_response(tagged("I considered X then Y", "The answer is 23"))
    assert parsed.solution == "The answer is 23"
    assert parsed.reasoning == "I considered X then Y"
    assert parsed.well_formed


def test_an_untagged_response_becomes_the_answer_with_no_working():
    """The old code wrote the same string into both columns, which made the
    distinction a lie. An unparsed response has no working, not duplicate working."""
    parsed = parse_response("just an answer, no tags")
    assert parsed.solution == "just an answer, no tags"
    assert parsed.reasoning == ""
    assert parsed.is_duplicate is False
    assert not parsed.well_formed


def test_a_model_that_echoes_itself_does_not_produce_duplicates():
    parsed = parse_response(tagged("same text", "same text"))
    assert parsed.solution == "same text"
    assert parsed.reasoning == ""
    assert parsed.is_duplicate is False


def test_only_the_working_tagged_still_yields_an_answer():
    parsed = parse_response("<reasoning>\nthinking\n</reasoning>\nThe answer is 7")
    assert parsed.reasoning == "thinking"
    assert parsed.solution == "The answer is 7"
    assert not parsed.well_formed


# -- templates -------------------------------------------------------------


def test_references_resolve_to_the_answer_not_the_working():
    state = AgentState(goal="G")
    state.record_step(StepOutput(1, "T", solution="ANSWER", reasoning="WORKING"))
    assert render("Build on {{ steps.1.solution }}", state) == "Build on ANSWER"
    assert render("Critique {{ steps.1.reasoning }}", state) == "Critique WORKING"
    assert render("Goal was {{ goal }}", state) == "Goal was G"


def test_an_unresolvable_reference_fails_loudly():
    """A typo must not be sent to the model as literal braces."""
    state = AgentState(goal="G")
    with pytest.raises(TemplateError, match="step 9, which has no output"):
        render("{{ steps.9.solution }}", state)


def test_an_unknown_field_is_rejected():
    state = AgentState()
    state.record_step(StepOutput(1, "T", "a"))
    with pytest.raises(TemplateError, match="unknown field"):
        render("{{ steps.1.mood }}", state)


def test_references_are_discovered_from_text():
    assert references("use {{ steps.2.solution }} and {{ steps.5.reasoning }}") == {2, 5}
    assert references("no refs here") == set()


# -- planning --------------------------------------------------------------


def test_independent_steps_share_a_wave():
    steps = [Problem(1, "a", "x"), Problem(2, "b", "y"), Problem(3, "c", "{{ steps.1.solution }}")]
    assert [[s.step_number for s in w] for w in waves(steps)] == [[1, 2], [3]]


def test_a_reference_declares_a_dependency_on_its_own():
    """Forgetting depends_on must not produce an unresolvable template."""
    steps = [Problem(1, "a", "x"), Problem(2, "b", "needs {{ steps.1.solution }}")]
    plan = waves(steps)
    assert [[s.step_number for s in w] for w in plan] == [[1], [2]]


def test_explicit_dependencies_are_honoured():
    steps = [Problem(1, "a", "x"), Problem(2, "b", "y", depends_on=(1,))]
    assert [[s.step_number for s in w] for w in waves(steps)] == [[1], [2]]


def test_completed_steps_count_as_satisfied():
    steps = [Problem(1, "a", "x"), Problem(2, "b", "{{ steps.1.solution }}")]
    assert [[s.step_number for s in w] for w in waves(steps, done={1})] == [[2]]


def test_a_cycle_is_rejected():
    steps = [
        Problem(1, "a", "{{ steps.2.solution }}"),
        Problem(2, "b", "{{ steps.1.solution }}"),
    ]
    with pytest.raises(PlanError, match="cycle"):
        waves(steps)


def test_depending_on_a_step_that_does_not_exist_is_rejected():
    with pytest.raises(PlanError, match="does not exist"):
        waves([Problem(1, "a", "x", depends_on=(4,))])


def test_a_step_cannot_depend_on_itself():
    with pytest.raises(PlanError, match="depends on itself"):
        waves([Problem(1, "a", "{{ steps.1.solution }}")])


# -- the loop --------------------------------------------------------------


def test_a_step_sees_what_came_before():
    """The difference between an agent and a queue."""
    provider = Recorder()
    agent = Agent(AgentState(goal="Ship it"), provider)

    agent.run_step(1, "First", "do the first thing")
    agent.run_step(2, "Second", "do the second thing")

    assert "Ship it" in provider.prompts[0]
    assert "do the first thing" not in provider.prompts[0].split("Now do this step")[0]
    # The second prompt carries the first step's exchange.
    assert "do the first thing" in provider.prompts[1]
    assert "fine" in provider.prompts[1]


def test_the_transcript_carries_answers_not_working():
    """Replaying every chain of thought is how a context window gets spent on
    nothing."""
    provider = Recorder([tagged("long rambling working", "the answer")])
    agent = Agent(AgentState(goal="G"), provider)
    agent.run_step(1, "First", "go")
    agent.run_step(2, "Second", "again")

    assert "the answer" in provider.prompts[1]
    assert "long rambling working" not in provider.prompts[1]
    # But the working is not lost — it is on the step output.
    assert agent.state.step_outputs[1].reasoning == "long rambling working"


def test_a_reference_is_resolved_before_sending():
    provider = Recorder([tagged("w", "FORTY TWO")])
    agent = Agent(AgentState(goal="G"), provider)
    agent.run_step(1, "First", "compute it")
    agent.run_step(2, "Second", "double {{ steps.1.solution }}")

    assert "double FORTY TWO" in provider.prompts[1]
    assert "{{ steps.1.solution }}" not in provider.prompts[1]


def test_the_resolved_instruction_goes_into_the_transcript():
    provider = Recorder([tagged("w", "RESULT")])
    agent = Agent(AgentState(goal="G"), provider)
    agent.run_step(1, "First", "go")
    agent.run_step(2, "Second", "use {{ steps.1.solution }}")

    user_turns = [m.content for m in agent.state.messages if m.role == "user"]
    assert "use RESULT" in user_turns[1]


def test_history_can_be_turned_off():
    provider = Recorder()
    agent = Agent(AgentState(goal="G"), provider, include_history=False)
    agent.run_step(1, "First", "one")
    agent.run_step(2, "Second", "two")
    assert "Work so far" not in provider.prompts[1]


# -- compaction ------------------------------------------------------------


def test_compaction_triggers_only_when_the_prompt_would_not_fit():
    provider = Recorder(default="x" * 40)
    roomy = Agent(AgentState(goal="G"), provider, context_window=100_000, max_tokens=100)
    for i in range(1, 6):
        roomy.run_step(i, f"S{i}", "go")
    assert roomy.state.compactions == 0, "nothing was close to the limit"


def test_compaction_folds_older_turns_into_a_summary():
    provider = Recorder(default="y" * 400)
    agent = Agent(AgentState(goal="G"), provider, context_window=700, max_tokens=100, keep_recent=2)

    for i in range(1, 6):
        agent.run_step(i, f"S{i}", "go")

    assert agent.state.compactions >= 1, "the history should have outgrown the window"
    assert agent.state.messages[0].role == ROLE_SUMMARY
    assert len(agent.state.messages) < 10, "older turns were folded away"
    # Answers are never lost, only the verbatim conversation.
    assert set(agent.state.step_outputs) == {1, 2, 3, 4, 5}


def test_compaction_is_recorded_in_the_state():
    provider = Recorder(default="z" * 400)
    seen: list[int] = []
    agent = Agent(
        AgentState(goal="G"),
        provider,
        context_window=700,
        max_tokens=100,
        keep_recent=2,
        on_compact=lambda s: seen.append(s.compactions),
    )
    for i in range(1, 6):
        agent.run_step(i, f"S{i}", "go")

    assert seen, "the caller was never told a compaction happened"
    assert agent.state.compactions == len(seen)


def test_nothing_to_compact_is_not_an_error():
    agent = Agent(AgentState(goal="G"), Recorder(), context_window=10, max_tokens=1, keep_recent=10)
    assert agent.compact(1) is False


def test_compaction_triggers_at_the_budget_boundary_and_not_before():
    """The boundary is the whole mechanism: compacting early wastes a call,
    compacting late overflows the provider's window."""
    agent = Agent(AgentState(goal="G"), Recorder(), context_window=4096, max_tokens=1000)
    budget = agent.budget()

    # A prompt whose estimate sits just under the budget must be left alone.
    under = "x" * ((budget - 2) * 4)
    assert agent.would_overflow(under) is False

    over = "x" * ((budget + 2) * 4)
    assert agent.would_overflow(over) is True


def test_a_wave_compacts_once_not_once_per_step():
    """Compaction is per batch so a parallel wave shares one history and never
    compacts concurrently with itself."""
    provider = Recorder(default="y" * 400)
    agent = Agent(AgentState(goal="G"), provider, context_window=700, max_tokens=100, keep_recent=2)
    for i in range(1, 5):
        agent.run_step(i, f"S{i}", "go")

    before = agent.state.compactions
    agent.prepare_wave([(5, "S5", "go"), (6, "S6", "go"), (7, "S7", "go")])
    assert agent.state.compactions - before <= 1


def test_the_budget_leaves_room_for_the_response():
    agent = Agent(AgentState(), Recorder(), context_window=8192, max_tokens=1200)
    assert agent.budget() < 8192 - 1200
    assert agent.budget() >= 256
