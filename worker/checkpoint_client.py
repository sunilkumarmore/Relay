"""Deprecated path — kept for one release. Use `relay.store`.

`RelayCheckpointClient` is now `relay.store.SupabaseStore`, one implementation of
the `Store` protocol rather than the only way to reach persistence.
"""

import warnings

warnings.warn(
    "worker/checkpoint_client.py has moved to relay.store; this shim will be "
    "removed in the next release.",
    DeprecationWarning,
    stacklevel=2,
)

from relay.store import (  # noqa: E402,F401
    RelayCheckpointError,
    RelayStoreError,
    Store,
    SupabaseStore,
)

RelayCheckpointClient = SupabaseStore
