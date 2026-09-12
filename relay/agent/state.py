"""Agent state — everything the worker knows, in a form that survives it.

Until now a "resumed" worker only recovered its place in a list: step 4 of 5.
Each step was an independent prompt, so there was nothing else to recover. That
made Relay a resumable queue, not a resumable agent.

An agent accumulates. It has a conversation, intermediate results it refers back
to, and artifacts it built along the way. Checkpointing *that* is the harder
problem, and it is the one this module exists for: state is a plain
JSON-serializable value, hashed so a rehydrated copy can be proven to be the one
that was saved.

State rows are append-only, one per completed step, so any earlier point in the
run is restorable — not just the latest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

# Bump when the shape changes in a way older rows cannot be read as.
STATE_VERSION = 1

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SUMMARY = "summary"

STATUS_COMPLETE = "complete"
# Written on eviction, mid-step. Never rehydrated: resume discards it and redoes
# the step, which is safe because UNIQUE(session_id, step_number) makes the
# checkpoint idempotent.
STATUS_PARTIAL = "partial"


class StateError(RuntimeError):
    pass


@dataclass
class Message:
    role: str
    content: str
    step_number: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content, "step_number": self.step_number}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(
            role=str(data.get("role", ROLE_USER)),
            content=str(data.get("content", "")),
            step_number=int(data.get("step_number") or 0),
        )


@dataclass
class StepOutput:
    """One completed step. `reasoning` and `solution` are distinct: the working
    and the answer are different things, and later steps usually want the
    answer alone."""

    step_number: int
    topic: str
    solution: str
    reasoning: str = ""
    tokens_in: int = 0
    tokens_out: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepOutput:
        return cls(
            step_number=int(data.get("step_number") or 0),
            topic=str(data.get("topic", "")),
            solution=str(data.get("solution", "")),
            reasoning=str(data.get("reasoning", "")),
            tokens_in=int(data.get("tokens_in") or 0),
            tokens_out=int(data.get("tokens_out") or 0),
        )


@dataclass
class AgentState:
    goal: str = ""
    messages: list[Message] = field(default_factory=list)
    step_outputs: dict[int, StepOutput] = field(default_factory=dict)
    scratchpad: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    tool_state: dict[str, Any] = field(default_factory=dict)
    compactions: int = 0
    version: int = STATE_VERSION

    # -- conversation -----------------------------------------------------
    def add_message(self, role: str, content: str, step_number: int = 0) -> None:
        self.messages.append(Message(role=role, content=content, step_number=step_number))

    def record_step(self, output: StepOutput) -> None:
        self.step_outputs[output.step_number] = output

    def solved(self) -> set[int]:
        return set(self.step_outputs)

    def transcript(self) -> str:
        """The conversation as the backend sees it.

        Flattened rather than sent as a message array because the inference
        protocol is a completion API — the provider may be serving a raw model.
        """
        parts = []
        for message in self.messages:
            if message.role == ROLE_SUMMARY:
                parts.append(f"[Summary of earlier work]\n{message.content}")
            elif message.role == ROLE_ASSISTANT:
                parts.append(f"Assistant:\n{message.content}")
            else:
                parts.append(f"User:\n{message.content}")
        return "\n\n".join(parts)

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "goal": self.goal,
            "messages": [m.to_dict() for m in self.messages],
            # JSON object keys must be strings; step numbers are restored on load.
            "step_outputs": {str(k): v.to_dict() for k, v in sorted(self.step_outputs.items())},
            "scratchpad": self.scratchpad,
            "artifacts": self.artifacts,
            "tool_state": self.tool_state,
            "compactions": self.compactions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        version = int(data.get("version") or STATE_VERSION)
        if version > STATE_VERSION:
            raise StateError(
                f"State was written by a newer Relay (version {version}; this build reads "
                f"{STATE_VERSION}). Refusing to guess at its shape."
            )
        return cls(
            goal=str(data.get("goal", "")),
            messages=[Message.from_dict(m) for m in data.get("messages") or []],
            step_outputs={
                int(k): StepOutput.from_dict(v) for k, v in (data.get("step_outputs") or {}).items()
            },
            scratchpad=dict(data.get("scratchpad") or {}),
            artifacts=dict(data.get("artifacts") or {}),
            tool_state=dict(data.get("tool_state") or {}),
            compactions=int(data.get("compactions") or 0),
            version=version,
        )

    def to_json(self) -> str:
        # Sorted and separator-stable so the hash is a function of content only.
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, blob: str) -> AgentState:
        try:
            return cls.from_dict(json.loads(blob))
        except json.JSONDecodeError as exc:
            raise StateError(f"Agent state is not readable JSON: {exc}") from exc

    def state_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def copy(self) -> AgentState:
        return AgentState.from_dict(json.loads(self.to_json()))


def verify(blob: str, expected_hash: str) -> AgentState:
    """Rehydrate, refusing state that is not the state that was saved.

    A corrupted or half-written blob would otherwise resume the agent into a
    conversation it never had, which is worse than starting the step again.
    """
    state = AgentState.from_json(blob)
    actual = state.state_hash()
    if expected_hash and actual != expected_hash:
        raise StateError(
            f"Agent state does not match its recorded hash (saved {expected_hash[:12]}, "
            f"got {actual[:12]}). Refusing to resume into state we cannot vouch for."
        )
    return state
