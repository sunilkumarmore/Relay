"""Deprecated entrypoint — kept for one release. Use `python -m relay.inference`."""

from relay.inference.registry import app_from_env, create_app, run  # noqa: F401

if __name__ == "__main__":
    print("NOTE: inference/registry.py is deprecated — use: python -m relay.inference")
    run()
