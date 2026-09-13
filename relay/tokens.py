"""Independent token counting.

A provider reports how many tokens it processed, and gets paid on that number.
Nothing stops it reporting a larger one. The consumer therefore needs its own
estimate — not to bill from, but to know when a claim is not credible.

Precision here is not the point; a bound is. If a real tokenizer for the model
is available the bound is tight. If not, the fallback is characters/4, and the
tolerance widens to match how rough that is. A wide honest bound beats a narrow
false one: the failure mode to avoid is accusing an honest provider.
"""

from __future__ import annotations

from dataclasses import dataclass

# A tight bound when we can actually tokenize the way the model does.
EXACT_TOLERANCE = 0.10
# chars/4 is a rule of thumb, not a measurement. Only a claim far outside it
# means anything, and this is deliberately generous to the provider.
HEURISTIC_TOLERANCE = 1.00

CHARS_PER_TOKEN = 4


@dataclass
class Estimate:
    tokens: int
    exact: bool

    @property
    def tolerance(self) -> float:
        return EXACT_TOLERANCE if self.exact else HEURISTIC_TOLERANCE

    def permits(self, claimed: int) -> bool:
        """Is a claim of this many tokens credible against what we measured?

        Only over-claiming matters. A provider reporting fewer tokens than we
        estimated is charging us less than it could, which is not a dispute.
        """
        if claimed <= self.tokens:
            return True
        ceiling = self.tokens * (1.0 + self.tolerance)
        # Small texts are where a proportional bound is least meaningful, so
        # allow a few tokens of slack regardless.
        return claimed <= max(ceiling, self.tokens + 8)


def _tiktoken_count(text: str, model: str) -> int | None:
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None
    try:
        return len(encoding.encode(text))
    except Exception:
        return None


def _hf_count(text: str, model: str) -> int | None:
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None
    try:
        return len(Tokenizer.from_pretrained(model).encode(text).ids)
    except Exception:
        return None


def estimate(text: str, model: str = "") -> Estimate:
    for counter in (_tiktoken_count, _hf_count):
        count = counter(text, model)
        if count is not None:
            return Estimate(tokens=count, exact=True)
    return Estimate(tokens=max(1, len(text) // CHARS_PER_TOKEN), exact=False)
