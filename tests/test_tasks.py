from __future__ import annotations

import pytest

from relay.worker.tasks import get_default_problems, get_problem, load_problems_from_file


def test_default_problems_are_numbered_from_one():
    goal, problems = get_default_problems()
    assert goal
    assert [p.step_number for p in problems] == list(range(1, len(problems) + 1))


def test_defaults_are_a_copy_not_the_module_list():
    _, first = get_default_problems()
    first.clear()
    _, second = get_default_problems()
    assert second, "mutating one caller's task list must not empty the defaults"


def test_load_from_yaml(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text(
        "goal: Ship it\nsteps:\n"
        "  - topic: First\n    prompt: Do the first thing\n"
        "  - topic: Second\n    prompt: Do the second thing\n",
        encoding="utf-8",
    )
    goal, problems = load_problems_from_file(str(path))
    assert goal == "Ship it"
    assert [(p.step_number, p.topic) for p in problems] == [(1, "First"), (2, "Second")]


def test_topic_defaults_when_omitted(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("goal: G\nsteps:\n  - prompt: Only a prompt\n", encoding="utf-8")
    _, problems = load_problems_from_file(str(path))
    assert problems[0].topic == "Step 1"


def test_empty_steps_is_an_error(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("goal: G\nsteps: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no steps"):
        load_problems_from_file(str(path))


def test_get_problem_by_step_number():
    _, problems = get_default_problems()
    assert get_problem(problems, 2).step_number == 2
    with pytest.raises(ValueError, match="Unknown step"):
        get_problem(problems, 999)
