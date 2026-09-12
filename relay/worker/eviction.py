from __future__ import annotations

import atexit
import signal
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class EvictionState:
    session_id: str
    worker_id: str
    machine_id: str
    steps_completed: int
    next_step_number: int
    next_problem: str


class EvictionManager:
    def __init__(
        self,
        state_provider: Callable[[], EvictionState],
        on_evict: Callable[[EvictionState], None],
        printer: Callable[[str], None] = print,
    ) -> None:
        self._state_provider = state_provider
        self._on_evict = on_evict
        self._printer = printer
        self._handled = False

    def install(self) -> None:
        signal.signal(signal.SIGINT, self._handle_signal)
        if hasattr(signal, "SIGTERM"):
            try:
                signal.signal(signal.SIGTERM, self._handle_signal)
            except (ValueError, OSError):
                pass
        atexit.register(self._handle_exit)

    def mark_done(self) -> None:
        self._handled = True

    def _handle_signal(self, signum, frame) -> None:  # noqa: ANN001
        if self._handled:
            raise SystemExit(0)
        self._handled = True
        self._printer("[EVICTION] signal received")
        state = self._state_provider()
        self._on_evict(state)
        self._printer("Safe to resume on any machine")
        raise SystemExit(0)

    def _handle_exit(self) -> None:
        if self._handled:
            return
        self._handled = True
        try:
            state = self._state_provider()
            self._on_evict(state)
        except Exception:
            pass
