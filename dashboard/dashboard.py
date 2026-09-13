"""Deprecated entrypoint — kept for one release. Use `python -m relay.dashboard`."""

from relay.dashboard.tui import main, run  # noqa: F401

if __name__ == "__main__":
    print("NOTE: dashboard/dashboard.py is deprecated — use: python -m relay.dashboard")
    run()
