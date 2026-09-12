"""The token bound: catch inflation without accusing honest providers."""

from __future__ import annotations

from relay.tokens import EXACT_TOLERANCE, Estimate, estimate


def test_estimates_something_for_any_text():
    result = estimate("hello world, this is a prompt of some length")
    assert result.tokens > 0


def test_empty_text_still_estimates_at_least_one():
    assert estimate("").tokens >= 1


def test_under_claiming_is_never_an_objection():
    """A provider charging less than it could is not a problem we have."""
    measured = Estimate(tokens=1000, exact=True)
    assert measured.permits(500) is True
    assert measured.permits(0) is True


def test_a_claim_inside_the_tolerance_is_permitted():
    measured = Estimate(tokens=1000, exact=True)
    assert measured.permits(1050) is True
    assert measured.permits(int(1000 * (1 + EXACT_TOLERANCE))) is True


def test_a_claim_far_outside_is_refused():
    measured = Estimate(tokens=1000, exact=True)
    assert measured.permits(5000) is False
    assert measured.permits(1200) is False


def test_the_heuristic_tolerance_is_wider_than_the_exact_one():
    """chars/4 is a rule of thumb, so the bound built on it must be generous —
    the failure to avoid is accusing an honest provider."""
    rough = Estimate(tokens=1000, exact=False)
    tight = Estimate(tokens=1000, exact=True)
    assert rough.tolerance > tight.tolerance
    assert rough.permits(1800) is True, "well within a rough bound"
    assert tight.permits(1800) is False


def test_even_a_rough_bound_catches_gross_inflation():
    rough = Estimate(tokens=1000, exact=False)
    assert rough.permits(50_000) is False


def test_small_texts_get_absolute_slack():
    """A proportional bound is meaningless on a handful of tokens."""
    tiny = Estimate(tokens=2, exact=True)
    assert tiny.permits(8) is True, "a few tokens either way proves nothing"
    assert tiny.permits(500) is False


def test_longer_text_estimates_higher():
    short = estimate("hi")
    long = estimate("word " * 500)
    assert long.tokens > short.tokens
