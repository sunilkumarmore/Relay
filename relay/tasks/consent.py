"""What the machine's owner has agreed to.

A provider is somebody's laptop or phone. Everything here exists so that the
owner, not the market, decides what runs on it — and so that the defaults are
the cautious ones. An operator who never opens the config file should end up
with a node that runs arithmetic for strangers and nothing else.

`python_exec` is **off by default and must be switched on deliberately.** The
executor runs it in a subprocess under resource limits, which stops a task from
exhausting the machine. It does *not* stop the code reading files the operator's
own user account can read. It is not a security sandbox and this module does not
pretend otherwise: enabling it is a statement that you trust whoever is signing
your tasks. WASM is the answer to that and it is not built yet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from relay.tasks import kernels

# Task types a node will run unless its operator says otherwise. Every one of
# these is pure arithmetic on data carried in the task: no filesystem, no
# network, no code supplied by the consumer.
SAFE_DEFAULT_TASK_TYPES = (
    kernels.MATMUL_BLOCK,
    kernels.TEXT_TRANSFORM,
    kernels.DATA_REDUCE,
)

# Task types that execute consumer-supplied code and are therefore opt-in.
UNSAFE_TASK_TYPES = (kernels.PYTHON_EXEC,)

DEFAULT_MAX_CONCURRENT = 1
DEFAULT_MAX_TASK_SECONDS = 300
DEFAULT_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_MEMORY_MB = 512


class ConsentError(RuntimeError):
    """The operator has not agreed to run this."""


@dataclass
class OperatorConsent:
    """The owner's standing answer to "may this run here?"."""

    allowed_task_types: tuple[str, ...] = SAFE_DEFAULT_TASK_TYPES
    max_concurrent: int = DEFAULT_MAX_CONCURRENT
    max_task_seconds: int = DEFAULT_MAX_TASK_SECONDS
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES
    max_memory_mb: int = DEFAULT_MAX_MEMORY_MB
    # A node can be told to finish what it holds and stop taking more. Distinct
    # from stopping the process: an operator who wants their laptop back should
    # not have to abandon the task they are part-way through and lose the fee.
    paused: bool = False
    # Consumers this node will work for. Empty means anyone — which is the
    # marketplace default, since a market where you must know your customers in
    # advance is not one.
    allowed_consumers: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_env(cls) -> OperatorConsent:
        raw_types = os.environ.get("RELAY_TASK_TYPES", "").strip()
        if raw_types:
            allowed = tuple(t.strip() for t in raw_types.split(",") if t.strip())
        else:
            allowed = SAFE_DEFAULT_TASK_TYPES
        consumers = os.environ.get("RELAY_ALLOWED_CONSUMERS", "").strip()
        return cls(
            allowed_task_types=allowed,
            max_concurrent=_int_env("RELAY_MAX_CONCURRENT_TASKS", DEFAULT_MAX_CONCURRENT),
            max_task_seconds=_int_env("RELAY_MAX_TASK_SECONDS", DEFAULT_MAX_TASK_SECONDS),
            max_payload_bytes=_int_env("RELAY_MAX_PAYLOAD_BYTES", DEFAULT_MAX_PAYLOAD_BYTES),
            max_memory_mb=_int_env("RELAY_MAX_TASK_MEMORY_MB", DEFAULT_MAX_MEMORY_MB),
            paused=os.environ.get("RELAY_PAUSED", "").lower() in ("1", "true", "yes"),
            allowed_consumers=frozenset(
                c.strip() for c in consumers.split(",") if c.strip()
            ),
        )

    def unsafe_types_enabled(self) -> tuple[str, ...]:
        return tuple(t for t in self.allowed_task_types if t in UNSAFE_TASK_TYPES)

    def warnings(self) -> list[str]:
        """Things the operator should be told on startup, in plain words."""
        notes: list[str] = []
        for task_type in self.unsafe_types_enabled():
            notes.append(
                f"{task_type} is enabled. It runs code written by whoever signed the "
                f"task, in a subprocess under resource limits. Those limits stop a task "
                f"exhausting this machine; they do NOT stop it reading files your user "
                f"account can read. Enable it only for consumers you trust."
            )
        if self.unsafe_types_enabled() and not self.allowed_consumers:
            notes.append(
                "No RELAY_ALLOWED_CONSUMERS set, so any consumer with a keypair and a "
                "budget can send this node code to run. Consider naming the consumers "
                "you accept."
            )
        return notes

    def check(self, *, task_type: str, consumer_node_id: str, payload_bytes: int,
              max_seconds: int) -> None:
        """Raise unless the operator has agreed to this specific piece of work."""
        if self.paused:
            raise ConsentError("this node is paused and is not accepting new tasks")
        if task_type not in self.allowed_task_types:
            raise ConsentError(
                f"operator has not enabled {task_type!r} on this node "
                f"(enabled: {', '.join(self.allowed_task_types) or 'none'})"
            )
        if self.allowed_consumers and consumer_node_id not in self.allowed_consumers:
            raise ConsentError("operator does not accept work from this consumer")
        if payload_bytes > self.max_payload_bytes:
            raise ConsentError(
                f"payload of {payload_bytes} bytes exceeds the operator's limit of "
                f"{self.max_payload_bytes}"
            )
        if max_seconds > self.max_task_seconds:
            raise ConsentError(
                f"task asks for up to {max_seconds}s, operator allows {self.max_task_seconds}s"
            )

    def accepts(self, *, task_type: str, consumer_node_id: str, payload_bytes: int,
                max_seconds: int) -> bool:
        try:
            self.check(
                task_type=task_type,
                consumer_node_id=consumer_node_id,
                payload_bytes=payload_bytes,
                max_seconds=max_seconds,
            )
        except ConsentError:
            return False
        return True


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
