"""Shared fixtures.

Nothing here touches the network beyond loopback: the registry runs in a thread
in the test process, and workers run as subprocesses that talk to it over
127.0.0.1. Inference is always FakeBackend.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import requests
import yaml

from relay.inference.backends import FakeBackend
from relay.inference.registry import Registry, create_app
from relay.store import FileStore


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class RegistryServer:
    url: str
    registry: Registry
    backend: FakeBackend


@pytest.fixture
def registry_server(tmp_path: Path):
    """A real registry on loopback, backed by FakeBackend and a FileStore."""
    import uvicorn

    def _start(
        store: FileStore,
        *,
        latency_ms: int = 0,
        node_id: str = "test-node",
        require_signatures: bool = True,
    ) -> RegistryServer:
        backend = FakeBackend(latency_ms=latency_ms)
        registry = Registry(backend, store, node_id=node_id)
        app = create_app(
            registry, prune_in_background=False, require_signatures=require_signatures
        )
        port = free_port()
        cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(cfg)
        server.install_signal_handlers = lambda: None  # not the main thread
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        url = f"http://127.0.0.1:{port}"
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                if requests.get(f"{url}/health", timeout=1).ok:
                    break
            except requests.RequestException:
                time.sleep(0.02)
        else:
            raise RuntimeError("registry did not come up")

        started.append((server, thread))
        return RegistryServer(url=url, registry=registry, backend=backend)

    started: list = []
    yield _start
    for server, thread in started:
        server.should_exit = True
        thread.join(timeout=5)


@pytest.fixture
def task_file(tmp_path: Path):
    def _write(steps: int, goal: str = "Test task") -> str:
        path = tmp_path / "task.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "goal": goal,
                    "steps": [
                        {"topic": f"Topic {i}", "prompt": f"Prompt for step {i}"}
                        for i in range(1, steps + 1)
                    ],
                }
            ),
            encoding="utf-8",
        )
        return str(path)

    return _write


@pytest.fixture
def store(tmp_path: Path) -> FileStore:
    return FileStore(tmp_path / "store.json")


@dataclass
class WorkerProcess:
    proc: subprocess.Popen
    store: FileStore
    session_id: str

    def checkpoint_count(self) -> int:
        return len(self.store.snapshot()["checkpoints"])

    def wait_for_checkpoints(self, count: int, timeout: float = 30.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.checkpoint_count() >= count:
                return True
            if self.proc.poll() is not None:
                return self.checkpoint_count() >= count
            time.sleep(0.01)
        return False


@pytest.fixture
def spawn_worker(tmp_path: Path):
    """Run `python -m relay.worker` as a subprocess, configured purely by env."""
    procs: list[subprocess.Popen] = []

    def _spawn(
        *,
        store: FileStore,
        registry_url: str,
        session_id: str,
        task_path: str,
        worker_id: str = "worker-alpha",
        machine_id: str = "machine-test",
        step_sleep: int = 0,
    ) -> WorkerProcess:
        env = dict(os.environ)
        env.update(
            {
                "ENV_FILE": str(tmp_path / "nonexistent.env"),
                "RELAY_KEY_PATH": str(tmp_path / f"{worker_id}.key"),
                "RELAY_STORE": "file",
                "RELAY_STORE_PATH": str(store.path),
                "RELAY_OUTPUT_DIR": str(tmp_path / "output"),
                "INFERENCE_REGISTRY": registry_url,
                "WORKER_ID": worker_id,
                "MACHINE_ID": machine_id,
                "SESSION_ID": session_id,
                "TASK_FILE": task_path,
                "STEP_SLEEP_SECONDS": str(step_sleep),
                "PYTHONUNBUFFERED": "1",
            }
        )
        proc = subprocess.Popen(
            [sys.executable, "-m", "relay.worker"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        procs.append(proc)
        return WorkerProcess(proc=proc, store=store, session_id=session_id)

    yield _spawn
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
