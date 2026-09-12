"""Building what we send, and reading what comes back.

Two jobs here. First, a step's prompt is assembled from the goal, the
conversation so far, and any earlier step outputs it explicitly references — so
step 3 can build on step 1 instead of starting from nothing.

Second, the response is split into working and answer. Relay's schema has always
had separate `reasoning` and `solution` columns and has always written the same
string into both, which made the distinction a lie. Asking for a delimited
format and parsing it makes it real: later steps referencing
``{{ steps.1.solution }}`` get the answer, not the answer buried in its working.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from relay.agent.state import AgentState

REASONING_TAG = "reasoning"
SOLUTION_TAG = "solution"

RESPONSE_CONTRACT = f"""Respond in exactly this format, including the tags:

<{REASONING_TAG}>
Your working: how you got there, what you considered, what you ruled out.
</{REASONING_TAG}>
<{SOLUTION_TAG}>
Your final answer on its own. Later steps will be given this part alone, so it
must stand without the working above.
</{SOLUTION_TAG}>"""

_TAG = re.compile(
    r"<(?P<tag>reasoning|solution)\s*>(?P<body>.*?)</(?P=tag)\s*>", re.DOTALL | re.IGNORECASE
)

# {{ steps.2.solution }}, {{ steps.2.reasoning }}, {{ goal }}
_REF = re.compile(r"\{\{\s*(?P<ref>[a-zA-Z0-9_.]+)\s*\}\}")


class TemplateError(RuntimeError):
    pass


@dataclass
class ParsedResponse:
    solution: str
    reasoning: str
    well_formed: bool

    @property
    def is_duplicate(self) -> bool:
        return bool(self.solution) and self.solution == self.reasoning


def parse_response(text: str) -> ParsedResponse:
    """Split a response into answer and working.

    When the model ignores the format the whole text becomes the solution and
    the reasoning stays empty. It must never become both: two identical columns
    carry no more information than one, and pretending otherwise is what the old
    ``reasoning=solution`` did.
    """
    found = {m.group("tag").lower(): m.group("body").strip() for m in _TAG.finditer(text)}
    solution = found.get(SOLUTION_TAG, "")
    reasoning = found.get(REASONING_TAG, "")

    if not solution:
        stripped = text.strip()
        if reasoning:
            # Only the working was tagged. Whatever sits outside the tags is the
            # closest thing to an answer we were given.
            solution = _TAG.sub("", text).strip() or stripped
        else:
            solution = stripped
        if solution == reasoning:
            reasoning = ""
        return ParsedResponse(solution=solution, reasoning=reasoning, well_formed=False)

    if reasoning == solution:
        # A model that echoed itself gave us one piece of information, not two.
        reasoning = ""

    return ParsedResponse(solution=solution, reasoning=reasoning, well_formed=True)


def render(template: str, state: AgentState) -> str:
    """Resolve ``{{ steps.N.solution }}`` and ``{{ goal }}`` against the state.

    An unresolvable reference raises rather than being sent through: a typo
    should fail the run loudly, not quietly ask the model about
    ``{{ steps.9.solution }}``.
    """

    def replace(match: re.Match[str]) -> str:
        ref = match.group("ref")
        if ref == "goal":
            return state.goal

        parts = ref.split(".")
        if len(parts) == 3 and parts[0] == "steps":
            try:
                number = int(parts[1])
            except ValueError as exc:
                raise TemplateError(f"{{{{ {ref} }}}}: step must be a number") from exc

            output = state.step_outputs.get(number)
            if output is None:
                known = ", ".join(str(n) for n in sorted(state.step_outputs)) or "none yet"
                raise TemplateError(
                    f"{{{{ {ref} }}}} refers to step {number}, which has no output. "
                    f"Completed: {known}. Did you forget depends_on?"
                )
            field = parts[2]
            if field not in {"solution", "reasoning", "topic"}:
                raise TemplateError(f"{{{{ {ref} }}}}: unknown field {field!r}")
            return str(getattr(output, field))

        raise TemplateError(f"Unknown reference {{{{ {ref} }}}}")

    return _REF.sub(replace, template)


def references(template: str) -> set[int]:
    """Step numbers a template refers to. Used to check declared dependencies."""
    found: set[int] = set()
    for match in _REF.finditer(template):
        parts = match.group("ref").split(".")
        if len(parts) == 3 and parts[0] == "steps":
            try:
                found.add(int(parts[1]))
            except ValueError:
                continue
    return found


def build_prompt(state: AgentState, *, topic: str, instruction: str, include_history: bool) -> str:
    """Assemble one step's prompt: goal, what happened so far, what to do now."""
    sections = []
    if state.goal:
        sections.append(f"You are working towards this goal:\n{state.goal}")

    if include_history and state.messages:
        sections.append(f"Work so far:\n\n{state.transcript()}")

    sections.append(f"Now do this step.\nTopic: {topic}\n\n{instruction}")
    sections.append(RESPONSE_CONTRACT)
    return "\n\n---\n\n".join(sections)


def build_compaction_prompt(state: AgentState, messages_to_fold: list) -> str:
    """Ask the model to fold older turns into one summary.

    What matters is what later steps will need — conclusions and decisions, not
    the prose that produced them.
    """
    transcript = "\n\n".join(
        f"{m.role.title()} (step {m.step_number}):\n{m.content}" for m in messages_to_fold
    )
    return (
        "Summarize the work below so it can replace the original in a longer "
        "conversation. Keep every conclusion, decision and figure a later step "
        "might need; drop the prose that produced them. Write it as notes, not "
        f"as a narrative.\n\nGoal of the overall task: {state.goal}\n\n"
        f"---\n\n{transcript}\n\n---\n\nSummary:"
    )
