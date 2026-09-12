"""Running an agent: one step at a time, with a conversation that persists.

The loop is small. What makes it an agent rather than a queue is that each step
sees what came before, and that everything it accumulates is written down in a
form another process can pick up.

Context is finite, so the conversation cannot simply grow. When the next prompt
would not fit in the provider's advertised window, older turns are folded into a
summary — and that compaction is itself part of the state, so a resumed run
inherits the compacted history rather than rebuilding a different one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from relay import tokens as tokenizer
from relay.agent import prompts
from relay.agent.state import (
    ROLE_ASSISTANT,
    ROLE_SUMMARY,
    ROLE_USER,
    AgentState,
    Message,
    StepOutput,
)

# Turns always kept verbatim at the end of the history: the most recent work is
# what the next step is most likely to build on.
DEFAULT_KEEP_RECENT = 4

# Headroom left under the context window for the response and for the fact that
# our token estimate is an estimate.
SAFETY_MARGIN_TOKENS = 256


class CompleteFn(Protocol):
    def __call__(self, prompt: str, max_tokens: int, step_number: int = 0) -> dict[str, Any]: ...


@dataclass
class StepRun:
    """What one step produced, including how it was billed."""

    output: StepOutput
    prompt: str
    raw_response: str
    # The step instruction with its {{ steps.N.* }} references resolved — what
    # goes into the transcript, rather than the template.
    instruction: str
    latency_ms: int
    tokens_in: int
    tokens_out: int
    tokens_used: int
    compacted: bool = False
    well_formed: bool = True


class Agent:
    def __init__(
        self,
        state: AgentState,
        complete: CompleteFn,
        *,
        context_window: int = 8192,
        max_tokens: int = 1200,
        model: str = "",
        keep_recent: int = DEFAULT_KEEP_RECENT,
        include_history: bool = True,
        on_compact: Callable[[AgentState], None] | None = None,
    ) -> None:
        self.state = state
        self.complete = complete
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.model = model
        self.keep_recent = keep_recent
        self.include_history = include_history
        self.on_compact = on_compact

    # -- context ----------------------------------------------------------
    def budget(self) -> int:
        """Prompt tokens available once the response and margin are reserved."""
        return max(256, self.context_window - self.max_tokens - SAFETY_MARGIN_TOKENS)

    def estimate(self, text: str) -> int:
        return tokenizer.estimate(text, self.model).tokens

    def would_overflow(self, prompt: str) -> bool:
        return self.estimate(prompt) > self.budget()

    def compact(self, step_number: int = 0) -> bool:
        """Fold older turns into one summary. Returns whether anything changed.

        The summary is produced by the model, which means it costs a call — and
        it goes through the same metered, receipted path as any other, because
        it is the agent's own work, not free bookkeeping.
        """
        foldable = self.state.messages[: -self.keep_recent] if self.keep_recent else self.state.messages
        if len(foldable) < 2:
            # Nothing to gain: one message cannot be summarized into fewer.
            return False

        prompt = prompts.build_compaction_prompt(self.state, foldable)
        result = self.complete(prompt=prompt, max_tokens=self.max_tokens, step_number=step_number)
        # The summary comes back under the same response contract as any other
        # call, so take the answer rather than the tags around it.
        summary = prompts.parse_response(str(result.get("response", "")).strip()).solution
        if not summary:
            return False

        kept = self.state.messages[len(foldable) :]
        folded_at = foldable[-1].step_number if foldable else 0
        self.state.messages = [Message(ROLE_SUMMARY, summary, folded_at), *kept]
        self.state.compactions += 1
        if self.on_compact:
            self.on_compact(self.state)
        return True

    # -- steps ------------------------------------------------------------
    def build_step_prompt(self, topic: str, instruction: str) -> str:
        resolved = prompts.render(instruction, self.state)
        return prompts.build_prompt(
            self.state,
            topic=topic,
            instruction=resolved,
            include_history=self.include_history,
        )

    def execute_step(self, step_number: int, topic: str, instruction: str) -> StepRun:
        """Run a step without touching the state.

        Kept separate from committing so independent steps can run at the same
        time and still be recorded in step order — a run has to be reproducible,
        and "whichever finished first" is not.
        """
        prompt = self.build_step_prompt(topic, instruction)
        result = self.complete(prompt=prompt, max_tokens=self.max_tokens, step_number=step_number)
        raw = str(result.get("response", "")).strip()
        parsed = prompts.parse_response(raw)

        output = StepOutput(
            step_number=step_number,
            topic=topic,
            solution=parsed.solution,
            reasoning=parsed.reasoning,
            tokens_in=int(result.get("tokens_in") or 0),
            tokens_out=int(result.get("tokens_out") or 0),
        )
        return StepRun(
            output=output,
            prompt=prompt,
            raw_response=raw,
            instruction=prompts.render(instruction, self.state),
            latency_ms=int(result.get("latency_ms") or 0),
            tokens_in=output.tokens_in,
            tokens_out=output.tokens_out,
            tokens_used=int(result.get("tokens_used") or 0),
            well_formed=parsed.well_formed,
        )

    def commit_step(self, run: StepRun) -> None:
        """Fold a finished step into the conversation.

        The transcript records the instruction and the answer. The working stays
        in the step output: replaying every chain of thought into the next
        prompt is how a context window gets spent on nothing.
        """
        self.state.add_message(ROLE_USER, run.instruction, run.output.step_number)
        self.state.add_message(ROLE_ASSISTANT, run.output.solution, run.output.step_number)
        self.state.record_step(run.output)

    def prepare_wave(self, steps: list[tuple[int, str, str]]) -> bool:
        """Compact if any step in the coming wave would not fit.

        Done once per wave rather than per step so that a parallel wave shares
        one history, and so compaction never happens concurrently with itself.
        """
        if not self.include_history:
            return False
        for step_number, topic, instruction in steps:
            if self.would_overflow(self.build_step_prompt(topic, instruction)):
                return self.compact(step_number)
        return False

    def run_step(self, step_number: int, topic: str, instruction: str) -> StepRun:
        compacted = self.prepare_wave([(step_number, topic, instruction)])
        run = self.execute_step(step_number, topic, instruction)
        run.compacted = compacted
        self.commit_step(run)
        return run
