"""Content-addressed operands.

A matrix multiplication splits into many tasks that all need the *same* right
operand. Shipping it inside every task payload would send the same megabytes
once per block. Instead an operand is stored once under the hash of its bytes,
and a task payload refers to it by that hash.

Addressing by hash rather than by name is what makes caching safe: a provider
that already holds `sha256:abc…` knows it holds exactly the bytes the task
means, with no version to get wrong and no cache to invalidate. It also means a
provider cannot substitute different operands without the output diverging from
every honest provider's.

Matrices are raw little-endian float64, row-major — the layout `array('d')`
produces on every platform Relay runs on. Base64 is applied only at the JSON
boundary, so the hash always covers the raw bytes, never their encoding.
"""

from __future__ import annotations

import base64
import hashlib
import sys
from array import array
from dataclasses import dataclass
from typing import Any

DTYPE = "f8"
_ITEMSIZE = 8


class OperandError(ValueError):
    """An operand is malformed, or is not the one its hash claims."""


def _to_le(values: array) -> bytes:
    """Little-endian bytes regardless of the host. Big-endian machines are rare
    and this is cheap; a silently byte-swapped operand would not be."""
    if sys.byteorder == "big":
        swapped = array("d", values)
        swapped.byteswap()
        return swapped.tobytes()
    return values.tobytes()


def _from_le(raw: bytes) -> array:
    values = array("d")
    values.frombytes(raw)
    if sys.byteorder == "big":
        values.byteswap()
    return values


@dataclass(frozen=True)
class Matrix:
    """A dense row-major matrix of float64."""

    rows: int
    cols: int
    data: array

    def __post_init__(self) -> None:
        if self.rows < 0 or self.cols < 0:
            raise OperandError("matrix dimensions cannot be negative")
        if len(self.data) != self.rows * self.cols:
            raise OperandError(
                f"matrix claims {self.rows}x{self.cols} but holds {len(self.data)} values"
            )

    @classmethod
    def from_rows(cls, rows: list[list[float]]) -> Matrix:
        height = len(rows)
        width = len(rows[0]) if height else 0
        flat = array("d")
        for row in rows:
            if len(row) != width:
                raise OperandError("matrix rows are not all the same length")
            flat.extend(float(v) for v in row)
        return cls(rows=height, cols=width, data=flat)

    def row(self, index: int) -> memoryview:
        start = index * self.cols
        return memoryview(self.data)[start : start + self.cols]

    def to_rows(self) -> list[list[float]]:
        return [list(self.data[i * self.cols : (i + 1) * self.cols]) for i in range(self.rows)]

    def raw(self) -> bytes:
        return _to_le(self.data)

    def digest(self) -> str:
        return operand_hash(self.rows, self.cols, self.raw())

    def to_payload(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "dtype": DTYPE,
            "b64": base64.b64encode(self.raw()).decode("ascii"),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Matrix:
        if payload.get("dtype", DTYPE) != DTYPE:
            raise OperandError(f"unsupported dtype {payload.get('dtype')!r}")
        try:
            raw = base64.b64decode(payload["b64"], validate=True)
        except (KeyError, ValueError) as exc:
            raise OperandError(f"operand is not valid base64: {exc}") from exc
        rows = int(payload["rows"])
        cols = int(payload["cols"])
        if len(raw) != rows * cols * _ITEMSIZE:
            raise OperandError(
                f"operand claims {rows}x{cols} ({rows * cols * _ITEMSIZE} bytes) "
                f"but carries {len(raw)}"
            )
        return cls(rows=rows, cols=cols, data=_from_le(raw))


def operand_hash(rows: int, cols: int, raw: bytes) -> str:
    """Covers the shape as well as the bytes.

    Without the shape, a 2x3 and a 3x2 operand holding the same numbers would
    share an address, and a task meaning one could be served the other.
    """
    digest = hashlib.sha256()
    digest.update(f"{DTYPE}:{rows}:{cols}:".encode("ascii"))
    digest.update(raw)
    return digest.hexdigest()


def verify(payload: dict[str, Any], expected_hash: str) -> Matrix:
    """Decode an operand and refuse it if it is not the one asked for."""
    matrix = Matrix.from_payload(payload)
    actual = matrix.digest()
    if actual != expected_hash:
        raise OperandError(f"operand hash {actual} does not match requested {expected_hash}")
    return matrix
