"""Deprecated entrypoint — kept for one release. Use `python -m relay.worker`."""

from relay.worker.daemon import main, run  # noqa: F401

if __name__ == "__main__":
    print("NOTE: worker/daemon.py is deprecated — use: python -m relay.worker")
    run()
