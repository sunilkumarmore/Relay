"""Tasks and results, and the signatures that make them binding.

A task is a consumer's signed statement: *I want this work done, I have budgeted
this much for it, and here is exactly what "this work" means.* A result is a
provider's signed statement about what came back.

The signature on a task is not ceremony. It is the only thing standing between
a provider and becoming an open compute proxy: a node runs work because a
consumer with a committed budget asked for it in writing, never because
something arrived on a socket. An unsigned task is refused before it is parsed.

`work_units` is carried on the task rather than reported by the provider, and
the queue re-derives it from the payload before accepting it. So the quantity
being billed is fixed by the *task*, and a provider has no number to inflate —
which is why task pricing needs no equivalent of the token over-claim dispute.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from relay import identity as ident
from relay.identity import Identity
from relay.tasks import kernels

TASK_SCHEMA_VERSION = 1

DEFAULT_TASK_TTL_SECONDS = 3600
DEFAULT_MAX_SECONDS = 300
DEFAULT_MAX_ATTEMPTS = 3

STATUS_QUEUED = "queued"
STATUS_LEASED = "leased"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

TERMINAL_STATUSES = (STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED)

RESULT_OK = "ok"
RESULT_ERROR = "error"

TASK_SIGNED_FIELDS = (
    "schema_version",
    "task_id",
    "job_id",
    "task_type",
    "payload_hash",
    "deterministic",
    "work_units",
    "consumer_node_id",
    "max_seconds",
    "max_price_credits",
    "created_at",
    "expires_at",
)

RESULT_SIGNED_FIELDS = (
    "schema_version",
    "task_id",
    "job_id",
    "provider_node_id",
    "consumer_node_id",
    "payload_hash",
    "output_hash",
    "work_units",
    "status",
    "duration_ms",
    "finished_at",
)


class TaskSignatureError(ValueError):
    """A task or result is not signed by the node it claims to come from."""


def now_utc() -> datetime:
    return datetime.now(UTC)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def payload_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _sign_over(fields: tuple[str, ...], obj: Any) -> bytes:
    return canonical_json({name: getattr(obj, name) for name in fields}).encode("utf-8")


class Task(BaseModel):
    schema_version: int = TASK_SCHEMA_VERSION
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str
    task_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    payload_hash: str = ""
    deterministic: bool = False
    work_units: int = 0
    consumer_node_id: str
    max_seconds: int = DEFAULT_MAX_SECONDS
    max_price_credits: float = 0.0
    created_at: str = ""
    expires_at: str = ""
    signature: str = ""

    # Queue-owned state. Deliberately outside the signature: the consumer signs
    # what it wants done, not what has happened to the request since.
    status: str = STATUS_QUEUED
    attempts: int = 0
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    lease_holder: str = ""
    lease_expires_at: str = ""
    completed_by: str = ""
    output_hash: str = ""
    # An audit copy must not be answered by the node whose work it is checking.
    # Advisory in `claimable_tasks` (so pollers skip it cheaply) and enforced in
    # the store's compare-and-set, which is what a dishonest client cannot skip.
    excluded_provider: str = ""
    audit_of: str = ""
    last_error: str = ""
    updated_at: str = ""

    def canonical_bytes(self) -> bytes:
        return _sign_over(TASK_SIGNED_FIELDS, self)

    def signed_by(self, identity: Identity) -> Task:
        if identity.node_id != self.consumer_node_id:
            raise TaskSignatureError("a task must be signed by the consumer that ordered it")
        return self.model_copy(
            update={"signature": identity.sign(self.canonical_bytes()).hex()}
        )

    def signature_is_valid(self) -> bool:
        if not self.signature:
            return False
        try:
            raw = bytes.fromhex(self.signature)
        except ValueError:
            return False
        return ident.verify(self.consumer_node_id, self.canonical_bytes(), raw)

    def payload_matches(self) -> bool:
        return self.payload_hash == payload_digest(self.payload)

    def is_expired(self, at: datetime | None = None) -> bool:
        if not self.expires_at:
            return False
        try:
            deadline = datetime.fromisoformat(self.expires_at)
        except ValueError:
            return True
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=UTC)
        return deadline <= (at or now_utc())

    def to_row(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Task:
        return cls(**{k: v for k, v in row.items() if k in cls.model_fields})


class TaskResult(BaseModel):
    schema_version: int = TASK_SCHEMA_VERSION
    task_id: str
    job_id: str
    provider_node_id: str
    consumer_node_id: str
    payload_hash: str
    output: dict[str, Any] = Field(default_factory=dict)
    output_hash: str = ""
    work_units: int = 0
    status: str = RESULT_OK
    error: str = ""
    duration_ms: int = 0
    finished_at: str = ""
    signature: str = ""

    def canonical_bytes(self) -> bytes:
        return _sign_over(RESULT_SIGNED_FIELDS, self)

    def signed_by(self, identity: Identity) -> TaskResult:
        if identity.node_id != self.provider_node_id:
            raise TaskSignatureError("a result must be signed by the provider that produced it")
        return self.model_copy(
            update={"signature": identity.sign(self.canonical_bytes()).hex()}
        )

    def signature_is_valid(self) -> bool:
        if not self.signature:
            return False
        try:
            raw = bytes.fromhex(self.signature)
        except ValueError:
            return False
        return ident.verify(self.provider_node_id, self.canonical_bytes(), raw)

    def output_matches(self) -> bool:
        """The signature covers `output_hash`, not `output` — a large result
        should not have to be re-serialised to be checked. This is what ties the
        two together, and a caller that trusts the hash without calling it has
        verified nothing about the bytes it is about to use."""
        if self.status != RESULT_OK:
            return True
        return self.output_hash == kernels.output_hash(self.output)

    def to_row(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> TaskResult:
        return cls(**{k: v for k, v in row.items() if k in cls.model_fields})


def build_task(
    identity: Identity,
    *,
    job_id: str,
    task_type: str,
    payload: dict[str, Any],
    max_seconds: int = DEFAULT_MAX_SECONDS,
    max_price_credits: float = 0.0,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ttl_seconds: int = DEFAULT_TASK_TTL_SECONDS,
    task_id: str | None = None,
    at: datetime | None = None,
) -> Task:
    """Validate, price, and sign one unit of work.

    The kernel validates the payload here — at authoring time, on the consumer's
    machine — so a task that could never run is never signed, never queued, and
    never occupies a provider's lease.
    """
    kernel = kernels.get(task_type)
    kernel.validate(payload)
    created = at or now_utc()
    task = Task(
        task_id=task_id or str(uuid.uuid4()),
        job_id=job_id,
        task_type=task_type,
        payload=payload,
        payload_hash=payload_digest(payload),
        deterministic=kernel.deterministic,
        work_units=kernel.work_units(payload),
        consumer_node_id=identity.node_id,
        max_seconds=max_seconds,
        max_price_credits=max_price_credits,
        max_attempts=max_attempts,
        created_at=created.isoformat(),
        expires_at=(created + timedelta(seconds=ttl_seconds)).isoformat(),
        updated_at=created.isoformat(),
    )
    return task.signed_by(identity)


def build_result(
    identity: Identity,
    *,
    task: Task,
    output: dict[str, Any],
    duration_ms: int,
    status: str = RESULT_OK,
    error: str = "",
    at: datetime | None = None,
) -> TaskResult:
    result = TaskResult(
        task_id=task.task_id,
        job_id=task.job_id,
        provider_node_id=identity.node_id,
        consumer_node_id=task.consumer_node_id,
        payload_hash=task.payload_hash,
        output=output,
        output_hash=kernels.output_hash(output) if status == RESULT_OK else "",
        work_units=task.work_units,
        status=status,
        error=error[:500],
        duration_ms=duration_ms,
        finished_at=(at or now_utc()).isoformat(),
    )
    return result.signed_by(identity)


def verify_task(task: Task, *, at: datetime | None = None) -> list[str]:
    """Everything a provider checks before spending a second on a task.

    A provider that skips this is an open compute proxy: it would run whatever
    arrived, for whoever sent it, with no one committed to paying.
    """
    problems: list[str] = []
    if task.schema_version != TASK_SCHEMA_VERSION:
        problems.append(f"unsupported task schema version {task.schema_version}")
        return problems
    if not task.signature_is_valid():
        problems.append("task signature is not valid")
    if not task.payload_matches():
        problems.append("payload does not match the hash the consumer signed")
    if task.is_expired(at):
        problems.append("task has expired")
    try:
        kernel = kernels.get(task.task_type)
    except kernels.TaskError as exc:
        problems.append(str(exc))
        return problems
    if task.deterministic != kernel.deterministic:
        problems.append(
            f"task claims deterministic={task.deterministic} but {task.task_type} "
            f"is {kernel.deterministic}"
        )
    try:
        kernel.validate(task.payload)
        declared = kernel.work_units(task.payload)
    except kernels.TaskError as exc:
        problems.append(str(exc))
        return problems
    if declared != task.work_units:
        problems.append(
            f"task claims {task.work_units} work units, payload is worth {declared}"
        )
    if task.max_seconds <= 0:
        problems.append("max_seconds must be positive")
    return problems


def verify_result(result: TaskResult, *, task: Task) -> list[str]:
    """What a consumer checks before paying for a result."""
    problems: list[str] = []
    if result.schema_version != TASK_SCHEMA_VERSION:
        problems.append(f"unsupported result schema version {result.schema_version}")
        return problems
    if not result.signature_is_valid():
        problems.append("result signature is not valid")
    if result.task_id != task.task_id:
        problems.append("result is for a different task")
    if result.job_id != task.job_id:
        problems.append("result cites a different job")
    if result.consumer_node_id != task.consumer_node_id:
        problems.append("result names a different consumer")
    if result.payload_hash != task.payload_hash:
        problems.append("result describes a different payload than the task we signed")
    if result.work_units != task.work_units:
        problems.append("result claims different work units than the task specifies")
    if not result.output_matches():
        problems.append("output does not match the hash the provider signed")
    return problems
