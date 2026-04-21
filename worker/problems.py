from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Problem:
    step_number: int
    topic: str
    prompt: str


PROBLEMS: list[Problem] = [
    Problem(
        step_number=1,
        topic="Number Theory",
        prompt=(
            "A number when divided by 3 leaves remainder 2, when divided "
            "by 5 leaves remainder 3, and when divided by 7 leaves remainder "
            "2. Find the smallest such positive integer and prove your answer "
            "using the Chinese Remainder Theorem. Show all working."
        ),
    ),
    Problem(
        step_number=2,
        topic="Logic Puzzle",
        prompt=(
            "Five houses in a row are each painted a different color. In each "
            "house lives a person of a different nationality. Each person drinks "
            "a different beverage, smokes a different brand, and owns a different "
            "pet. The Brit lives in the red house. The Swede keeps dogs. The Dane "
            "drinks tea. The green house is on the left of the white house. The "
            "green house owner drinks coffee. The person who smokes Pall Mall "
            "keeps birds. The owner of the yellow house smokes Dunhill. The man "
            "living in the center house drinks milk. The Norwegian lives in the "
            "first house. The man who smokes Blends lives next to the one who "
            "keeps cats. The man who keeps horses lives next to the man who "
            "smokes Dunhill. The owner who smokes Blue Master drinks beer. The "
            "German smokes Prince. The Norwegian lives next to the blue house. "
            "The man who smokes Blends has a neighbor who drinks water. Who owns "
            "the fish? Show complete logical deduction."
        ),
    ),
    Problem(
        step_number=3,
        topic="Probability",
        prompt=(
            "A factory produces widgets. Machine A produces 60% of all widgets "
            "with a 2% defect rate. Machine B produces 30% with a 5% defect rate. "
            "Machine C produces 10% with a 10% defect rate. If a randomly selected "
            "widget is found to be defective, what is the probability it came from "
            "each machine? Use Bayes theorem, show complete working."
        ),
    ),
    Problem(
        step_number=4,
        topic="Algorithm Analysis",
        prompt=(
            "Analyze the time and space complexity of finding all pairs that "
            "sum to target k in an unsorted array of n integers. Describe and "
            "compare three approaches: brute force, sorting-based, hash-based. "
            "For each: write pseudocode, derive Big O time and space complexity, "
            "identify crossover points, recommend optimal approach for "
            "n=100, n=10000, and n=1000000."
        ),
    ),
    Problem(
        step_number=5,
        topic="Mathematical Proof",
        prompt=(
            "Prove that the square root of 2 is irrational using proof by "
            "contradiction. Extend to show square root of any non-perfect-square "
            "integer is irrational. Prove there are infinitely many irrational "
            "numbers between any two rational numbers. Show rigorous working."
        ),
    ),
]


def get_problem(step_number: int) -> Problem:
    for problem in PROBLEMS:
        if problem.step_number == step_number:
            return problem
    raise ValueError(f"Unknown step: {step_number}")
