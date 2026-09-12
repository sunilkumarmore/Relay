"""Execution order.

Steps used to be a list, run front to back, because nothing connected them. Now
a step can name what it needs — explicitly with ``depends_on``, or implicitly by
referring to ``{{ steps.2.solution }}`` — and the order follows from that rather
than from how they happened to be written down.

Steps that need nothing from each other form a wave and may run together. Within
a wave results are committed in step order regardless of which finished first,
so a run is reproducible: the conversation a resumed worker inherits is the same
one an uninterrupted run would have built.

One consequence is worth stating plainly, because it is a real trade-off rather
than an oversight. The agent keeps a single linear conversation, and a step sees
everything committed before it. Running a wave concurrently therefore means its
steps all see the history as of the *start* of the wave, not as of each other.
Independent steps are independent by the dependency graph, but they are not
independent of the transcript — so `max_parallel > 1` can produce different
answers than the same task run one step at a time. That is why the default is 1:
the sequential path is the plain linear agent, and concurrency is something you
ask for knowing it changes what each step reads.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from relay.agent import prompts


class PlanError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlannedStep:
    step_number: int
    topic: str
    prompt: str
    depends_on: frozenset[int]


def dependencies_for(step, all_numbers: set[int]) -> frozenset[int]:
    """What this step needs: what it declares, plus what it actually refers to.

    A reference is a dependency whether or not it was declared — forgetting
    `depends_on` should not silently produce an unresolvable template.
    """
    declared = {int(n) for n in getattr(step, "depends_on", ()) or ()}
    referenced = prompts.references(step.prompt)

    for number in declared | referenced:
        if number not in all_numbers:
            raise PlanError(f"Step {step.step_number} depends on step {number}, which does not exist")
        if number == step.step_number:
            raise PlanError(f"Step {step.step_number} depends on itself")

    return frozenset(declared | referenced)


def plan(steps: Iterable) -> list[PlannedStep]:
    ordered = list(steps)
    numbers = {s.step_number for s in ordered}
    if len(numbers) != len(ordered):
        raise PlanError("Step numbers must be unique")
    return [
        PlannedStep(
            step_number=s.step_number,
            topic=s.topic,
            prompt=s.prompt,
            depends_on=dependencies_for(s, numbers),
        )
        for s in ordered
    ]


def waves(steps: Iterable, *, done: set[int] | None = None) -> list[list[PlannedStep]]:
    """Group steps into batches that can run together.

    Each wave contains only steps whose dependencies are already satisfied.
    Steps already completed (a resumed run) count as satisfied without being
    rerun.
    """
    planned = plan(steps)
    done = set(done or ())
    remaining = [s for s in planned if s.step_number not in done]
    satisfied = set(done)

    result: list[list[PlannedStep]] = []
    while remaining:
        ready = [s for s in remaining if s.depends_on <= satisfied]
        if not ready:
            stuck = ", ".join(
                f"{s.step_number} (needs {sorted(s.depends_on - satisfied)})" for s in remaining
            )
            raise PlanError(f"Steps form a cycle or depend on something unreachable: {stuck}")

        # Deterministic order within a wave, so results commit the same way every run.
        ready.sort(key=lambda s: s.step_number)
        result.append(ready)
        satisfied |= {s.step_number for s in ready}
        remaining = [s for s in remaining if s.step_number not in satisfied]

    return result


def sequence(steps: Iterable, *, done: set[int] | None = None) -> list[PlannedStep]:
    """Flatten the waves into the order steps will actually be committed in."""
    return [step for wave in waves(steps, done=done) for step in wave]
