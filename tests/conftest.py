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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
import requests
import yaml

from relay.identity import Identity
from relay.inference.backends import FakeBackend
from relay.inference.registry import Registry, create_app
from relay.provider.config import ModelOffering, ProviderConfig
from relay.provider.server import Provider
from relay.store import FileStore


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _noop() -> None:
    return None


@dataclass
class RegistryServer:
    url: str
    registry: Registry
    backend: FakeBackend
    stop: Callable[[], None] = _noop
    provider_node_id: str = ""


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

        def stop() -> None:
            server.should_exit = True
            thread.join(timeout=10)

        started.append((server, thread))
        return RegistryServer(
            url=url,
            registry=registry,
            backend=backend,
            stop=stop,
            provider_node_id=registry.identity.node_id,
        )

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
        max_retries: int = 3,
        policy: str = "cheapest",
    ) -> WorkerProcess:
        env = dict(os.environ)
        env.update(
            {
                "ENV_FILE": str(tmp_path / "nonexistent.env"),
                "RELAY_KEY_PATH": str(tmp_path / f"{worker_id}.key"),
                "RELAY_STORE": "file",
                "RELAY_STORE_PATH": str(store.path),
                "RELAY_OUTPUT_DIR": str(tmp_path / "output"),
                # Empty means "shop the directory" rather than "use this node".
                "INFERENCE_REGISTRY": registry_url or "",
                "RELAY_MAX_RETRIES": str(max_retries),
                "RELAY_POLICY": policy,
                "RELAY_FAILOVER_COOLDOWN": "300",
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


@pytest.fixture
def market(tmp_path):
    """Run one or more real providers advertising into a shared directory."""
    import uvicorn

    from relay.provider.server import create_app as create_provider_app

    running: list = []

    def _start(
        store: FileStore,
        *,
        name: str,
        model: str = "llama3",
        price_out: float = 0.10,
        price_in: float = 0.02,
        latency_ms: int = 0,
        max_concurrency: int = 8,
        region: str = "lab",
        context_window: int = 8192,
    ) -> RegistryServer:
        port = free_port()
        endpoint = f"http://127.0.0.1:{port}"
        backend = FakeBackend(latency_ms=latency_ms, available_models=[model])
        provider = Provider(
            ProviderConfig(
                endpoint_url=endpoint,
                region=region,
                offer_ttl_seconds=300,
                models=[
                    ModelOffering(model, "fake", context_window, price_in, price_out, max_concurrency)
                ],
                backends={"fake": backend},
            ),
            store,
            identity=Identity.load_or_create(tmp_path / f"{name}.key"),
            node_id=name,
        )
        app = create_provider_app(provider, prune_in_background=False)
        cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(cfg)
        server.install_signal_handlers = lambda: None
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                if requests.get(f"{endpoint}/health", timeout=1).ok:
                    break
            except requests.RequestException:
                time.sleep(0.02)
        else:
            raise RuntimeError(f"provider {name} did not come up")

        def stop() -> None:
            server.should_exit = True
            thread.join(timeout=10)

        running.append((server, thread))
        return RegistryServer(
            url=endpoint,
            registry=provider,
            backend=backend,
            stop=stop,
            provider_node_id=provider.identity.node_id,
        )

    yield _start
    for server, thread in running:
        server.should_exit = True
        thread.join(timeout=5)
