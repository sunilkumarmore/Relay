"""Deprecated path — kept for one release. Use `relay.worker.eviction`."""

import warnings

warnings.warn(
    "worker/eviction_handler.py has moved to relay.worker.eviction; this shim "
    "will be removed in the next release.",
    DeprecationWarning,
    stacklevel=2,
)

from relay.worker.eviction import *  # noqa: F403,E402
