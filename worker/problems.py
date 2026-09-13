"""Deprecated path — kept for one release. Use `relay.worker.tasks`."""

import warnings

warnings.warn(
    "worker/problems.py has moved to relay.worker.tasks; this shim "
    "will be removed in the next release.",
    DeprecationWarning,
    stacklevel=2,
)

from relay.worker.tasks import *  # noqa: F403,E402
