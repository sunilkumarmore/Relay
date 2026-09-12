"""Deprecated path — kept for one release. Use `relay.inference.backends`."""

import warnings

warnings.warn(
    "inference/ollama_client.py has moved to relay.inference.backends; this shim "
    "will be removed in the next release.",
    DeprecationWarning,
    stacklevel=2,
)

from relay.inference.backends import *  # noqa: F403,E402
