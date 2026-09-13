"""`python_exec` — running code the consumer wrote.

Registered separately from the arithmetic kernels because it is a different kind
of thing, and the separation should be visible: everything in `kernels.py`
operates on numbers carried in the task, while this executes a program.

**Not deterministic, and the flag says so.** `PYTHONHASHSEED` randomises set and
dict iteration between runs, `repr` of a float differs across libm versions,
locale changes formatting, and any library the code imports drifts with the
provider's patch level. Hash-comparing two honest runs would therefore diverge
routinely, and a marketplace that slashed on divergence would be punishing
providers for their Python version. So `deterministic = False`, redundant
execution is not attempted for this type, and `output_divergence` disputes never
apply to it.

**Not a security sandbox.** The subprocess runs under CPU, memory and output
limits, which stop a task from exhausting the machine. It runs as the operator's
own user, so it can read what that user can read. `consent.py` keeps it off by
default for that reason. WASM is the real answer and is not built.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Any

from relay.tasks.kernels import PYTHON_EXEC, Resolver, TaskError

MAX_OUTPUT_BYTES = 1 << 20
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MEMORY_MB = 512

# Prelude for the child. Pins what can be pinned, so that two runs on the *same*
# machine agree even though two runs on different machines may not.
_PREAMBLE = "import sys\nsys.setrecursionlimit(10000)\n"


class PythonExec:
    name = PYTHON_EXEC
    deterministic = False

    def __init__(
        self,
        *,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        memory_mb: int = DEFAULT_MEMORY_MB,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.memory_mb = memory_mb

    def work_units(self, payload: dict[str, Any]) -> int:
        """Bytes of source. Not a measure of runtime — nothing is, before the
        code runs — but it is derivable from the payload, so it cannot be
        inflated by the provider. Pricing `python_exec` by the second would
        need a trustworthy clock on the provider, which per the prior art we
        do not have."""
        return len(str(payload.get("code", "")).encode("utf-8"))

    def validate(self, payload: dict[str, Any]) -> None:
        if "code" not in payload:
            raise TaskError("payload is missing 'code'")
        code = payload["code"]
        if not isinstance(code, str):
            raise TaskError("code must be a string")
        if not code.strip():
            raise TaskError("code is empty")
        stdin = payload.get("stdin", "")
        if not isinstance(stdin, str):
            raise TaskError("stdin must be a string")

    def run(self, payload: dict[str, Any], resolve: Resolver) -> dict[str, Any]:
        self.validate(payload)
        code = _PREAMBLE + str(payload["code"])
        stdin = str(payload.get("stdin", ""))
        timeout = int(payload.get("timeout_seconds") or self.timeout_seconds)

        with tempfile.TemporaryDirectory(prefix="relay-task-") as workdir:
            script = os.path.join(workdir, "task.py")
            with open(script, "w", encoding="utf-8") as handle:
                handle.write(code)

            env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                # Fixed so that at least repeat runs on one machine agree; it
                # does not make the type deterministic across machines.
                "PYTHONHASHSEED": "0",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONDONTWRITEBYTECODE": "1",
                "HOME": workdir,
                "TMPDIR": workdir,
                "LC_ALL": "C",
                "LANG": "C",
            }
            try:
                completed = subprocess.run(  # noqa: S603 - argv is fixed, never a shell
                    [sys.executable, "-I", "-S", script],
                    input=stdin,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=workdir,
                    env=env,
                    preexec_fn=self._limits() if os.name == "posix" else None,
                )
            except subprocess.TimeoutExpired as exc:
                raise TaskError(f"code exceeded its {timeout}s limit") from exc
            except OSError as exc:
                raise TaskError(f"could not start the interpreter: {exc}") from exc

        return {
            "stdout": completed.stdout[:MAX_OUTPUT_BYTES],
            "stderr": completed.stderr[:MAX_OUTPUT_BYTES],
            "exit_code": completed.returncode,
        }

    def _limits(self):
        memory_bytes = self.memory_mb * 1024 * 1024
        cpu_seconds = self.timeout_seconds

        def apply() -> None:  # pragma: no cover - runs in the forked child
            import resource

            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        return apply


def register_default() -> None:
    """Make `python_exec` a known task type.

    Knowing the type is not permission to run it — `consent.py` keeps it out of
    the default allow-list, so a node that has not opted in will refuse the task
    at claim time regardless.
    """
    from relay.tasks import kernels

    kernels.register(PythonExec())
