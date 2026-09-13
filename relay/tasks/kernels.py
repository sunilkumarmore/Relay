"""What a task actually does, and whether two honest machines agree on it.

Each kernel declares three things: how much work its payload represents, how to
validate that payload before anyone is paid to run it, and whether two honest
providers running it on different hardware will produce *identical bytes*.

That last flag decides whether a task can be verified by redundant execution.
It is not a quality rating. `python_exec` is not deterministic — `PYTHONHASHSEED`
randomisation, float formatting, locale and library drift all diverge between
machines — so hash-comparing two runs of it would slash honest providers for
their Python patch release. It runs; it just cannot be checked that way.

`matmul_block` *is* deterministic, which is worth being precise about. Python's
float is an IEEE-754 double, and `*` and `+` on two of them are correctly
rounded: the result is a function of the inputs alone, not of the machine. Fix
the order of accumulation and every honest provider produces the same bits.
This is exactly the property BLAS gives up — it reorders and blocks for speed,
so two numpy builds can disagree in the last bit. We are slower on purpose and
verifiable as a result.

## work_units

Every kernel derives its work quantity *from the payload*, and the consumer
computes the same number before it signs. So the provider is never asked how
much work it did, and cannot inflate the bill: for a matmul block the answer is
`rows x K x cols`, which is fixed the moment the task is written.
"""

from __future__ import annotations

import hashlib
import json
from array import array
from collections.abc import Callable
from typing import Any, Protocol

from relay.tasks.operands import Matrix

MATMUL_BLOCK = "matmul_block"
TEXT_TRANSFORM = "text_transform"
DATA_REDUCE = "data_reduce"
PYTHON_EXEC = "python_exec"


class TaskError(ValueError):
    """A payload this kernel cannot run. Raised before any work is charged for."""


# Given an operand hash, return its Matrix. Supplied by the executor, which owns
# fetching and caching; kernels never reach the network themselves.
Resolver = Callable[[str], Matrix]


class Kernel(Protocol):
    name: str
    deterministic: bool

    def work_units(self, payload: dict[str, Any]) -> int: ...
    def validate(self, payload: dict[str, Any]) -> None: ...
    def run(self, payload: dict[str, Any], resolve: Resolver) -> dict[str, Any]: ...


def _require(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise TaskError(f"payload is missing {key!r}")
    return payload[key]


def _positive_int(value: Any, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise TaskError(f"{label} must be an integer, got {value!r}") from exc
    if number < 0:
        raise TaskError(f"{label} cannot be negative")
    return number


# --------------------------------------------------------------------------
# matmul_block


class MatmulBlock:
    """One tile of C = A @ B.

    Both operands are referenced by hash rather than carried in the payload, and
    that is what makes a large job possible at all. A 4200x4200 matrix is 141MB
    of float64; carried inline it would be re-sent with every task that touches
    it, which for a job tiled 160 x 30 means hundreds of gigabytes of the same
    numbers. Referenced by hash, each row strip of A and each column strip of B
    travels once per device and is cached there.

    Tiling in two dimensions rather than one is for the same reason. Splitting
    only by rows means every task needs the whole of B, so the smallest unit of
    data a device must hold is the entire right operand. Splitting both ways
    bounds it: a device holds one strip of each.
    """

    name = MATMUL_BLOCK
    deterministic = True

    def work_units(self, payload: dict[str, Any]) -> int:
        rows = _positive_int(_require(payload, "a_rows"), "a_rows")
        inner = _positive_int(_require(payload, "a_cols"), "a_cols")
        cols = _positive_int(_require(payload, "b_cols"), "b_cols")
        return rows * inner * cols

    def validate(self, payload: dict[str, Any]) -> None:
        for key in ("a_hash", "b_hash"):
            digest = str(_require(payload, key))
            if len(digest) != 64:
                raise TaskError(f"{key} is not a sha256 digest")
        rows = _positive_int(_require(payload, "a_rows"), "a_rows")
        inner = _positive_int(_require(payload, "a_cols"), "a_cols")
        b_rows = _positive_int(_require(payload, "b_rows"), "b_rows")
        cols = _positive_int(_require(payload, "b_cols"), "b_cols")
        _positive_int(_require(payload, "row_offset"), "row_offset")
        _positive_int(_require(payload, "col_offset"), "col_offset")
        if inner != b_rows:
            raise TaskError(f"cannot multiply {rows}x{inner} by {b_rows}x{cols}")

    def run(self, payload: dict[str, Any], resolve: Resolver) -> dict[str, Any]:
        self.validate(payload)
        a = resolve(str(payload["a_hash"]))
        b = resolve(str(payload["b_hash"]))

        # The operands were fetched by hash, so these can only disagree if the
        # task itself is inconsistent with what it asked for.
        if (a.rows, a.cols) != (int(payload["a_rows"]), int(payload["a_cols"])):
            raise TaskError(
                f"a_hash resolves to {a.rows}x{a.cols}, task declares "
                f"{payload['a_rows']}x{payload['a_cols']}"
            )
        if (b.rows, b.cols) != (int(payload["b_rows"]), int(payload["b_cols"])):
            raise TaskError(
                f"b_hash resolves to {b.rows}x{b.cols}, task declares "
                f"{payload['b_rows']}x{payload['b_cols']}"
            )
        if a.cols != b.rows:
            raise TaskError(f"cannot multiply {a.rows}x{a.cols} by {b.rows}x{b.cols}")

        # Accumulation is k-ascending for every element of the result, on every
        # machine. That fixed order is the whole basis of the determinism claim
        # above — do not reorder these loops for speed without also setting
        # deterministic = False.
        width = b.cols
        b_rows = [list(b.row(k)) for k in range(b.rows)]
        out: list[float] = []
        for i in range(a.rows):
            a_row = a.row(i)
            acc = [0.0] * width
            for k in range(a.cols):
                scale = a_row[k]
                b_row = b_rows[k]
                acc = [c + scale * v for c, v in zip(acc, b_row, strict=True)]
            out.extend(acc)

        block = Matrix(rows=a.rows, cols=width, data=array("d", out))
        return {
            "c_block": block.to_payload(),
            "row_offset": int(payload["row_offset"]),
            "col_offset": int(payload["col_offset"]),
            "rows": a.rows,
            "cols": width,
        }


# --------------------------------------------------------------------------
# text_transform

_TEXT_OPS: dict[str, Callable[[str], str]] = {
    "upper": str.upper,
    "lower": str.lower,
    "strip": str.strip,
    "reverse": lambda s: s[::-1],
    "collapse_space": lambda s: " ".join(s.split()),
    "sha256": lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest(),
}


class TextTransform:
    """Ordered string operations. Deterministic because every operation is
    defined on code points, not on locale — which is why `casefold`-style and
    locale-sensitive operations are deliberately absent from the table."""

    name = TEXT_TRANSFORM
    deterministic = True

    def work_units(self, payload: dict[str, Any]) -> int:
        return len(str(_require(payload, "text")))

    def validate(self, payload: dict[str, Any]) -> None:
        _require(payload, "text")
        ops = _require(payload, "ops")
        if not isinstance(ops, list) or not ops:
            raise TaskError("ops must be a non-empty list")
        unknown = [op for op in ops if op not in _TEXT_OPS]
        if unknown:
            raise TaskError(f"unknown text operations: {unknown}")

    def run(self, payload: dict[str, Any], resolve: Resolver) -> dict[str, Any]:
        self.validate(payload)
        text = str(payload["text"])
        for op in payload["ops"]:
            text = _TEXT_OPS[op](text)
        return {"text": text}


# --------------------------------------------------------------------------
# data_reduce

_REDUCE_OPS = ("sum", "min", "max", "count", "mean", "sorted")


class DataReduce:
    """Reduce a list of numbers. `sum` and `mean` add in the order given rather
    than sorting or pairwise-summing, so the result is reproducible; a more
    accurate summation that reordered would not be."""

    name = DATA_REDUCE
    deterministic = True

    def work_units(self, payload: dict[str, Any]) -> int:
        values = _require(payload, "values")
        if not isinstance(values, list):
            raise TaskError("values must be a list")
        return len(values)

    def validate(self, payload: dict[str, Any]) -> None:
        values = _require(payload, "values")
        if not isinstance(values, list):
            raise TaskError("values must be a list")
        op = str(_require(payload, "op"))
        if op not in _REDUCE_OPS:
            raise TaskError(f"unknown reduce operation {op!r}")
        for value in values:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TaskError(f"values must be numbers, found {type(value).__name__}")

    def run(self, payload: dict[str, Any], resolve: Resolver) -> dict[str, Any]:
        self.validate(payload)
        values = [float(v) for v in payload["values"]]
        op = str(payload["op"])
        if op == "count":
            return {"value": len(values)}
        if not values:
            raise TaskError(f"{op} is undefined on an empty list")
        if op == "sum":
            total = 0.0
            for value in values:
                total += value
            return {"value": total}
        if op == "mean":
            total = 0.0
            for value in values:
                total += value
            return {"value": total / len(values)}
        if op == "min":
            return {"value": min(values)}
        if op == "max":
            return {"value": max(values)}
        return {"values": sorted(values)}


# --------------------------------------------------------------------------
# registry

_KERNELS: dict[str, Kernel] = {
    MATMUL_BLOCK: MatmulBlock(),
    TEXT_TRANSFORM: TextTransform(),
    DATA_REDUCE: DataReduce(),
}


def register(kernel: Kernel) -> None:
    _KERNELS[kernel.name] = kernel


def get(task_type: str) -> Kernel:
    kernel = _KERNELS.get(task_type)
    if kernel is None:
        raise TaskError(f"unknown task type {task_type!r}")
    return kernel


def known_types() -> tuple[str, ...]:
    return tuple(sorted(_KERNELS))


def deterministic_types() -> tuple[str, ...]:
    return tuple(sorted(name for name, k in _KERNELS.items() if k.deterministic))


def output_hash(output: dict[str, Any]) -> str:
    """The bytes two honest providers are expected to agree on.

    Canonical JSON — sorted keys, no incidental whitespace — so that agreement
    is about the values and never about how they were serialised.
    """
    blob = json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
