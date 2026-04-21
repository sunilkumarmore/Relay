from __future__ import annotations

import atexit
import signal
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class EvictionSnapshot:
    session_id: str
    machine_id: str
    completed_steps: int
    next_step_number: int
    next_problem: str
    current_problem: str = ""
    agent_scratchpad: str = ""
    checkpoint_payload: dict = field(default_factory=dict)


class EvictionController:
    def __init__(
        self,
        snapshot_provider: Callable[[], EvictionSnapshot],
        persist_snapshot: Callable[[EvictionSnapshot], None],
        printer: Callable[[str], None] = print,
    ) -> None:
        self._snapshot_provider = snapshot_provider
        self._persist_snapshot = persist_snapshot
        self._print = printer
        self._handled = False
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        self._installed = True
        signal.signal(signal.SIGINT, self._handle_signal)
        if hasattr(signal, "SIGTERM"):
            try:
                signal.signal(signal.SIGTERM, self._handle_signal)
            except (ValueError, OSError):
                pass
        atexit.register(self._handle_atexit)

    def mark_completed(self) -> None:
        self._handled = True

    def _print_eviction_summary(self, snapshot: EvictionSnapshot) -> None:
        self._print("✓ Checkpoint saved successfully")
        self._print("✓ Safe to start agent on another machine")
        self._print(f"Session ID: {snapshot.session_id}")
        self._print(f"Problems completed: {snapshot.completed_steps} of 5")

    def _handle_signal(self, signum, frame) -> None:  # noqa: ANN001
        if self._handled:
            raise SystemExit(0)
        self._handled = True
        self._print("⚡ EVICTION SIGNAL RECEIVED")
        self._print("Saving current state to Supabase...")
        try:
            snapshot = self._snapshot_provider()
            self._persist_snapshot(snapshot)
            self._print_eviction_summary(snapshot)
        finally:
            raise SystemExit(0)

    def _handle_atexit(self) -> None:
        if self._handled:
            return
        self._handled = True
        try:
            self._print("⚡ EVICTION SIGNAL RECEIVED")
            self._print("Saving current state to Supabase...")
            snapshot = self._snapshot_provider()
            self._persist_snapshot(snapshot)
            self._print_eviction_summary(snapshot)
        except Exception:
            # Best-effort fallback for Windows shutdown semantics.
            pass


def install_eviction_handlers(
    snapshot_provider: Callable[[], EvictionSnapshot],
    persist_snapshot: Callable[[EvictionSnapshot], None],
    printer: Callable[[str], None] = print,
) -> EvictionController:
    controller = EvictionController(snapshot_provider, persist_snapshot, printer=printer)
    controller.install()
    return controller
