from __future__ import annotations

import os

from dotenv import load_dotenv

_loaded: set[str] = set()


def env_file() -> str:
    return os.getenv("ENV_FILE", ".env")


def load_env(path: str | None = None) -> None:
    """Load the process env file once per path.

    Loading is idempotent so that importing several relay modules in one process
    does not repeatedly re-read the same file.
    """
    target = path or env_file()
    if target in _loaded:
        return
    load_dotenv(target)
    _loaded.add(target)


def get(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def get_int(name: str, default: int) -> int:
    raw = get(name)
    return int(raw) if raw else default


def get_float(name: str, default: float) -> float:
    raw = get(name)
    return float(raw) if raw else default
