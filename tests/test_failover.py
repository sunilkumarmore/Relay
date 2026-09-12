"""Failover: the job survives a provider, not just a machine.

Phase 0 proved a job survives losing its worker. This proves it survives losing
the node doing the inference — with real providers on loopback, one of which
stops mid-job.
"""

from __future__ import annotations

import yaml


def write_task(tmp_path, steps: int, requirements: dict | None = None, name="task.yaml") -> str:
    path = tmp_path / name
    body: dict = {
        "goal": "Failover test",
        "steps": [{"topic": f"T{i}", "prompt": f"Prompt {i}"} for i in range(1, steps + 1)],
    }
    if requirements:
        body["requirements"] = requirements
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return str(path)


def events(store, session_id):
    return [e["event"] for e in store.snapshot()["migration_log"] if e["session_id"] == session_id]


def test_worker_finds_a_provider_with_no_endpoint_configured(store, market, spawn_worker, tmp_path):
    """No INFERENCE_REGISTRY at all — the worker shops the directory."""
    provider = market(store, name="solo")
    worker = spawn_worker(
        store=store,
        registry_url="",
        session_id="sess-discover",
        task_path=write_task(tmp_path, 3),
    )
    assert worker.proc.wait(timeout=90) == 0, worker.proc.stdout.read()

    assert [c["step_number"] for c in store.get_checkpoints("sess-discover")] == [1, 2, 3]
    assert provider.backend.calls == 3
    assert store.get_state("sess-discover")["provider_node_id"] == provider.provider_node_id


def test_cheapest_provider_wins(store, market, spawn_worker, tmp_path):
    dear = market(store, name="dear", price_out=0.90)
    cheap = market(store, name="cheap", price_out=0.01)

    worker = spawn_worker(
        store=store, registry_url="", session_id="sess-cheap", task_path=write_task(tmp_path, 2)
    )
    assert worker.proc.wait(timeout=90) == 0, worker.proc.stdout.read()

    assert cheap.backend.calls == 2
    assert dear.backend.calls == 0


def test_price_ceiling_excludes_a_provider(store, market, spawn_worker, tmp_path):
    dear = market(store, name="dear", price_out=0.90)
    cheap = market(store, name="cheap", price_out=0.05)

    task = write_task(tmp_path, 2, {"max_price_out_per_1k": 0.10})
    worker = spawn_worker(
        store=store, registry_url="", session_id="sess-ceiling", task_path=task
    )
    assert worker.proc.wait(timeout=90) == 0

    assert cheap.backend.calls == 2
    assert dear.backend.calls == 0


def test_a_job_nobody_can_serve_fails_clearly(store, market, spawn_worker, tmp_path):
    market(store, name="dear", price_out=0.90)
    task = write_task(tmp_path, 2, {"max_price_out_per_1k": 0.001})

    worker = spawn_worker(store=store, registry_url="", session_id="sess-nobody", task_path=task)
    assert worker.proc.wait(timeout=90) == 1
    output = worker.proc.stdout.read()
    assert "No provider could take this job" in output
    assert store.get_checkpoints("sess-nobody") == []


def test_requiring_a_model_nobody_serves_fails_clearly(store, market, spawn_worker, tmp_path):
    market(store, name="a", model="llama3")
    task = write_task(tmp_path, 1, {"model": "gpt-4"})

    worker = spawn_worker(store=store, registry_url="", session_id="sess-model", task_path=task)
    assert worker.proc.wait(timeout=90) == 1


def test_job_survives_its_provider_dying_mid_run(store, market, spawn_worker, tmp_path):
    """The whole point of a market: one seller leaving is not the end of the job."""
    cheap = market(store, name="cheap", price_out=0.01, latency_ms=400)
    backup = market(store, name="backup", price_out=0.50)

    worker = spawn_worker(
        store=store,
        registry_url="",
        session_id="sess-failover",
        task_path=write_task(tmp_path, 5),
        max_retries=0,
    )
    assert worker.wait_for_checkpoints(2), "worker never reached the cheap provider"

    cheap.stop()  # the provider we chose goes away

    assert worker.proc.wait(timeout=120) == 0, worker.proc.stdout.read()

    assert [c["step_number"] for c in store.get_checkpoints("sess-failover")] == [1, 2, 3, 4, 5]
    assert "provider_switched" in events(store, "sess-failover")
    # The worker may have committed another step on `cheap` between the poll and
    # the shutdown, so assert the property rather than an exact split: both nodes
    # did real work, and between them they covered the job.
    assert cheap.backend.calls >= 1
    assert backup.backend.calls >= 1, "never actually failed over"
    assert cheap.backend.calls + backup.backend.calls >= 5
    assert store.get_state("sess-failover")["provider_node_id"] == backup.provider_node_id


def test_switching_providers_does_not_duplicate_or_skip_a_step(store, market, spawn_worker, tmp_path):
    from collections import Counter

    cheap = market(store, name="cheap", price_out=0.01, latency_ms=400)
    market(store, name="backup", price_out=0.50)

    worker = spawn_worker(
        store=store,
        registry_url="",
        session_id="sess-nodup",
        task_path=write_task(tmp_path, 5),
        max_retries=0,
    )
    assert worker.wait_for_checkpoints(1)
    cheap.stop()
    assert worker.proc.wait(timeout=120) == 0

    steps = [c["step_number"] for c in store.get_checkpoints("sess-nodup")]
    assert steps == [1, 2, 3, 4, 5]
    assert [s for s, n in Counter(steps).items() if n > 1] == []


def test_failure_is_recorded_so_other_consumers_can_learn(store, market, spawn_worker, tmp_path):
    cheap = market(store, name="cheap", price_out=0.01, latency_ms=400)
    market(store, name="backup", price_out=0.50)

    worker = spawn_worker(
        store=store,
        registry_url="",
        session_id="sess-health",
        task_path=write_task(tmp_path, 4),
        max_retries=0,
    )
    assert worker.wait_for_checkpoints(1)
    cheap.stop()
    assert worker.proc.wait(timeout=120) == 0

    observations = store.list_provider_health()
    assert any(not o["ok"] for o in observations), "the failure went unrecorded"
    assert any(o["ok"] and o["latency_ms"] is not None for o in observations)


def test_saturated_provider_is_passed_over(store, market, spawn_worker, tmp_path):
    """429 means try someone else now, not wait in line."""
    busy = market(store, name="busy", price_out=0.01, max_concurrency=1, latency_ms=600)
    spare = market(store, name="spare", price_out=0.50)

    first = spawn_worker(
        store=store,
        registry_url="",
        session_id="sess-busy-a",
        task_path=write_task(tmp_path, 4, name="a.yaml"),
        worker_id="worker-a",
        max_retries=0,
    )
    second = spawn_worker(
        store=store,
        registry_url="",
        session_id="sess-busy-b",
        task_path=write_task(tmp_path, 4, name="b.yaml"),
        worker_id="worker-b",
        max_retries=0,
    )
    assert first.proc.wait(timeout=120) == 0, first.proc.stdout.read()
    assert second.proc.wait(timeout=120) == 0, second.proc.stdout.read()

    assert len(store.get_checkpoints("sess-busy-a")) == 4
    assert len(store.get_checkpoints("sess-busy-b")) == 4
    # With one slot on the cheap node, some work had to go to the spare one.
    assert spare.backend.calls > 0
    assert busy.backend.calls > 0


def test_resumed_worker_returns_to_its_previous_provider(store, market, spawn_worker, tmp_path):
    import signal

    market(store, name="cheap", price_out=0.01, latency_ms=400)
    task = write_task(tmp_path, 5)

    worker = spawn_worker(
        store=store, registry_url="", session_id="sess-sticky", task_path=task
    )
    assert worker.wait_for_checkpoints(1)
    worker.proc.send_signal(signal.SIGTERM)
    worker.proc.wait(timeout=30)

    chosen = store.get_state("sess-sticky")["provider_node_id"]
    assert chosen

    resumed = spawn_worker(
        store=store,
        registry_url="",
        session_id="sess-sticky",
        task_path=task,
        worker_id="worker-beta",
    )
    assert resumed.proc.wait(timeout=90) == 0
    assert store.get_state("sess-sticky")["provider_node_id"] == chosen


def test_pinned_endpoint_still_bypasses_the_market(store, registry_server, spawn_worker, task_file):
    """The two-machine demo predates the directory and must keep working."""
    server = registry_server(store)
    worker = spawn_worker(
        store=store, registry_url=server.url, session_id="sess-pinned", task_path=task_file(3)
    )
    assert worker.proc.wait(timeout=90) == 0, worker.proc.stdout.read()
    assert len(store.get_checkpoints("sess-pinned")) == 3
    assert store.list_offers() == [], "the legacy registry advertises nothing"
