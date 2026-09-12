"""Agent state: what survives the process."""

from __future__ import annotations

import pytest

from relay.agent.state import (
    ROLE_ASSISTANT,
    ROLE_USER,
    STATE_VERSION,
    AgentState,
    StateError,
    StepOutput,
    verify,
)
from relay.store import MemoryStore


def populated() -> AgentState:
    state = AgentState(goal="Ship the thing")
    state.add_message(ROLE_USER, "do step one", 1)
    state.add_message(ROLE_ASSISTANT, "one is done", 1)
    state.record_step(StepOutput(1, "First", "one is done", "here is why"))
    state.scratchpad["note"] = "remember this"
    state.artifacts["plan.md"] = "# Plan"
    state.tool_state["cursor"] = 7
    return state


def test_round_trips_through_json():
    original = populated()
    restored = AgentState.from_json(original.to_json())

    assert restored.goal == original.goal
    assert [m.to_dict() for m in restored.messages] == [m.to_dict() for m in original.messages]
    assert restored.step_outputs[1].solution == "one is done"
    assert restored.step_outputs[1].reasoning == "here is why"
    assert restored.scratchpad == {"note": "remember this"}
    assert restored.artifacts == {"plan.md": "# Plan"}
    assert restored.tool_state == {"cursor": 7}


def test_the_hash_is_a_function_of_content_only():
    a, b = populated(), populated()
    assert a.state_hash() == b.state_hash()

    b.scratchpad["note"] = "something else"
    assert a.state_hash() != b.state_hash()


def test_verify_accepts_the_state_that_was_saved():
    state = populated()
    restored = verify(state.to_json(), state.state_hash())
    assert restored.state_hash() == state.state_hash()


def test_verify_refuses_state_that_was_altered():
    """Resuming into a conversation the agent never had is worse than redoing
    the step."""
    state = populated()
    blob = state.to_json().replace("one is done", "something else entirely")
    with pytest.raises(StateError, match="does not match its recorded hash"):
        verify(blob, state.state_hash())


def test_verify_refuses_unreadable_state():
    with pytest.raises(StateError, match="not readable JSON"):
        verify("{ not json", "")


def test_state_from_a_newer_relay_is_refused():
    state = populated()
    blob = state.to_json().replace(f'"version":{STATE_VERSION}', f'"version":{STATE_VERSION + 5}')
    with pytest.raises(StateError, match="newer Relay"):
        AgentState.from_json(blob)


def test_step_numbers_survive_json_string_keys():
    state = populated()
    restored = AgentState.from_json(state.to_json())
    assert 1 in restored.step_outputs, "keys must come back as ints, not strings"
    assert restored.solved() == {1}


def test_copy_is_independent():
    original = populated()
    clone = original.copy()
    clone.scratchpad["note"] = "changed"
    clone.add_message(ROLE_USER, "another", 2)

    assert original.scratchpad["note"] == "remember this"
    assert len(original.messages) == 2


def test_transcript_renders_roles():
    state = populated()
    text = state.transcript()
    assert "User:" in text and "Assistant:" in text
    assert "do step one" in text and "one is done" in text


def test_state_is_stored_append_only_per_step():
    store = MemoryStore()
    state = populated()
    store.insert_agent_state("s1", 1, state.to_json(), state.state_hash())

    state.record_step(StepOutput(2, "Second", "two is done"))
    store.insert_agent_state("s1", 2, state.to_json(), state.state_hash())

    rows = store.list_agent_states("s1")
    assert [r["step_number"] for r in rows] == [1, 2], "earlier points stay restorable"
    assert store.get_agent_state("s1")["step_number"] == 2, "latest by default"

    earlier = AgentState.from_json(store.get_agent_state("s1", step_number=1)["state_blob"])
    assert earlier.solved() == {1}


def test_partial_state_is_not_returned_by_default():
    store = MemoryStore()
    state = populated()
    store.insert_agent_state("s1", 1, state.to_json(), state.state_hash(), "complete")
    store.insert_agent_state("s1", 2, state.to_json(), state.state_hash(), "partial")

    assert store.get_agent_state("s1")["step_number"] == 1
    assert store.get_agent_state("s1", include_partial=True)["step_number"] == 2

    store.delete_partial_agent_state("s1")
    assert store.get_agent_state("s1", include_partial=True)["step_number"] == 1
