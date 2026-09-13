"""Deprecated entrypoint — kept for one release. Use `python -m relay.controller`."""

from relay.controller.cli import cli  # noqa: F401

if __name__ == "__main__":
    print("NOTE: controller/controller.py is deprecated — use: python -m relay.controller")
    cli()
