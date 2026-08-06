"""Tests for job control: pause / resume / stop / restart.

Covers the JobManager control plane (checkpoint semantics, task registry,
pause-all persistence, restart sweep) and the /ingest/jobs control
endpoints. Endpoint tests use httpx.ASGITransport so the app, the manager,
and the test all share one event loop.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.ingestion.job_manager import JobCancelled, JobManager
from backend.routers import ingestion


@pytest.fixture()
def manager(tmp_path):
    return JobManager(tmp_path / "jobs.sqlite")


async def _make_job(manager: JobManager, **kwargs) -> str:
    defaults = dict(
        source_path="/tmp/x.pdf", filename="x.pdf", categories=[], tags=[]
    )
    defaults.update(kwargs)
    job = await manager.create(**defaults)
    return job.job_id


async def _wait_for(cond, timeout: float = 5.0, interval: float = 0.05):
    """Poll a (sync or async) condition until truthy or the timeout hits."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        result = cond()
        if asyncio.iscoroutine(result):
            result = await result
        if result:
            return result
        if loop.time() > deadline:
            raise AssertionError("condition not met within timeout")
        await asyncio.sleep(interval)


async def _counting_worker(manager: JobManager, job_id: str, state: dict):
    """Simulates a pipeline loop: one unit of work per checkpoint."""
    await manager.update(job_id, status="processing")
    while state["count"] < state["target"]:
        await manager.checkpoint(job_id)
        state["count"] += 1
        await asyncio.sleep(0.01)
    await manager.complete(job_id)


# --------------------------------------------------------- checkpoint core


@pytest.mark.asyncio
async def test_checkpoint_noop_when_not_paused(manager):
    await manager.init()
    job_id = await _make_job(manager)
    await manager.update(job_id, status="processing")
    await manager.checkpoint(job_id)  # must not raise or block


@pytest.mark.asyncio
async def test_pause_resume_roundtrip(manager):
    await manager.init()
    job_id = await _make_job(manager)
    state = {"count": 0, "target": 10_000}
    manager.spawn(job_id, _counting_worker(manager, job_id, state))

    await _wait_for(lambda: state["count"] > 2)
    assert await manager.request_pause(job_id)

    # Worker must land in 'paused' and stop making progress.
    await _wait_for(_status_is(manager, job_id, "paused"))
    frozen = state["count"]
    await asyncio.sleep(0.4)
    assert state["count"] == frozen, "paused job kept doing work"

    # Resume: status returns to processing and the counter moves again.
    assert await manager.request_resume(job_id)
    await _wait_for(_status_is(manager, job_id, "processing"))
    await _wait_for(lambda: state["count"] > frozen)

    # Let it finish cleanly.
    state["target"] = state["count"] + 3
    await _wait_for(_status_is(manager, job_id, "completed"))


@pytest.mark.asyncio
async def test_cancel_processing_job_finalizes_row(manager):
    await manager.init()
    job_id = await _make_job(manager)
    await manager.set_steps(job_id, ["extracting_entities"])
    await manager.update_step(job_id, "extracting_entities", "running")
    state = {"count": 0, "target": 10_000}
    task = manager.spawn(job_id, _counting_worker(manager, job_id, state))

    await _wait_for(lambda: state["count"] > 2)
    assert await manager.request_cancel(job_id)

    await _wait_for(_status_is(manager, job_id, "cancelled"))
    await _wait_for(lambda: task.done())

    job = await manager.get(job_id)
    assert job.status == "cancelled"
    # The running step was flagged, not left dangling.
    step = job.steps[0]
    assert step.status == "warning"
    assert "stopped by user" in (step.detail or "")
    # Registry cleaned up.
    assert job_id not in manager._tasks


@pytest.mark.asyncio
async def test_cancel_paused_job(manager):
    await manager.init()
    job_id = await _make_job(manager)
    state = {"count": 0, "target": 10_000}
    manager.spawn(job_id, _counting_worker(manager, job_id, state))
    await _wait_for(lambda: state["count"] > 1)
    await manager.request_pause(job_id)
    await _wait_for(_status_is(manager, job_id, "paused"))

    assert await manager.request_cancel(job_id)
    await _wait_for(_status_is(manager, job_id, "cancelled"))


@pytest.mark.asyncio
async def test_cancel_queued_job_is_immediate(manager):
    """A job still waiting for its slot (status 'queued') is hard-cancelled
    without waiting for a checkpoint — it hasn't touched anything yet."""
    await manager.init()
    job_id = await _make_job(manager)
    blocker = asyncio.Event()  # never set — simulates the ingest semaphore

    async def queued_worker():
        await blocker.wait()

    task = manager.spawn(job_id, queued_worker())
    await asyncio.sleep(0.05)

    assert await manager.request_cancel(job_id)
    await _wait_for(_status_is(manager, job_id, "cancelled"))
    await _wait_for(lambda: task.done())


@pytest.mark.asyncio
async def test_cancel_inactive_job_refused(manager):
    await manager.init()
    job_id = await _make_job(manager)
    await manager.complete(job_id)
    assert not await manager.request_cancel(job_id)
    assert not await manager.request_pause(job_id)
    assert not await manager.request_resume(job_id)


@pytest.mark.asyncio
async def test_checkpoint_raises_jobcancelled_outside_spawn(manager):
    """Direct checkpoint contract: cancel flag -> finalize + raise."""
    await manager.init()
    job_id = await _make_job(manager)
    await manager.update(job_id, status="processing")
    dummy = asyncio.create_task(asyncio.sleep(30))
    manager._tasks[job_id] = dummy
    try:
        await manager.request_cancel(job_id)
        with pytest.raises(JobCancelled):
            await manager.checkpoint(job_id)
        job = await manager.get(job_id)
        assert job.status == "cancelled"
    finally:
        manager._tasks.pop(job_id, None)
        dummy.cancel()
        with pytest.raises(asyncio.CancelledError):
            await dummy


# ------------------------------------------------------------- pause-all


@pytest.mark.asyncio
async def test_pause_all_holds_every_job_then_resumes(manager):
    await manager.init()
    ids, states = [], []
    for _ in range(3):
        job_id = await _make_job(manager)
        state = {"count": 0, "target": 10_000}
        manager.spawn(job_id, _counting_worker(manager, job_id, state))
        ids.append(job_id)
        states.append(state)

    await _wait_for(lambda: all(s["count"] > 1 for s in states))
    await manager.set_pause_all(True)
    for job_id in ids:
        await _wait_for(_status_is(manager, job_id, "paused"))

    # resume-all lets everything finish.
    await manager.set_pause_all(False)
    for state in states:
        state["target"] = state["count"] + 3
    for job_id in ids:
        await _wait_for(_status_is(manager, job_id, "completed"))


@pytest.mark.asyncio
async def test_pause_all_persists_across_restart(manager, tmp_path):
    await manager.init()
    await manager.set_pause_all(True)

    other = JobManager(tmp_path / "jobs.sqlite")
    await other.init()
    assert other.pause_all_active

    await other.set_pause_all(False)
    third = JobManager(tmp_path / "jobs.sqlite")
    await third.init()
    assert not third.pause_all_active


@pytest.mark.asyncio
async def test_restart_sweep_marks_paused_jobs_failed(manager, tmp_path):
    await manager.init()
    job_id = await _make_job(manager)
    await manager.update(job_id, status="paused")

    fresh = JobManager(tmp_path / "jobs.sqlite")
    await fresh.init()
    job = await fresh.get(job_id)
    assert job.status == "failed"
    assert "restarted" in (job.error_message or "")


# ------------------------------------------------------- list filters


@pytest.mark.asyncio
async def test_list_recent_active_and_terminal_filters(manager):
    await manager.init()
    a = await _make_job(manager)  # stays queued
    b = await _make_job(manager)
    await manager.update(b, status="processing")
    c = await _make_job(manager)
    await manager.update(c, status="paused")
    d = await _make_job(manager)
    await manager.complete(d)
    e = await _make_job(manager)
    await manager.fail(e, "boom")

    active = await manager.list_recent(status="active", limit=50)
    assert {j.job_id for j in active} == {a, b, c}
    # processing first, then paused, then queued
    assert [j.job_id for j in active] == [b, c, a]

    terminal = await manager.list_recent(status="terminal", limit=50)
    assert {j.job_id for j in terminal} == {d, e}

    counts = await manager.status_counts()
    assert counts["queued"] == 1 and counts["processing"] == 1
    assert counts["paused"] == 1


# ----------------------------------------------------------- endpoints


class StubPipeline:
    """Records restart dispatches instead of doing pipeline work."""

    def __init__(self):
        self.calls: list[tuple] = []

    async def run_fill_missing(self, job_id, doc_id, *, do_text=True,
                               do_visual=True, do_entities=False,
                               do_recover_text=False):
        self.calls.append(
            ("fill-missing", job_id, doc_id, do_text, do_visual,
             do_entities, do_recover_text)
        )

    async def run_resummarize(self, job_id):
        self.calls.append(("resummarize", job_id))


@pytest.fixture()
async def api(tmp_path):
    app = FastAPI()
    app.include_router(ingestion.router)
    manager = JobManager(tmp_path / "jobs.sqlite")
    await manager.init()
    app.state.job_manager = manager
    app.state.pipeline = StubPipeline()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, manager, app.state.pipeline


@pytest.mark.asyncio
async def test_controls_endpoint(api):
    client, manager, _ = api
    await _make_job(manager)
    r = await client.get("/ingest/jobs/controls")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["pause_all"] is False
    assert data["active"] == 1
    assert data["counts"]["queued"] == 1


@pytest.mark.asyncio
async def test_pause_all_and_resume_all_endpoints(api):
    client, manager, _ = api
    r = await client.post("/ingest/jobs/pause-all")
    assert r.status_code == 200 and r.json()["data"]["pause_all"] is True
    assert manager.pause_all_active

    r = await client.get("/ingest/jobs/controls")
    assert r.json()["data"]["pause_all"] is True

    r = await client.post("/ingest/jobs/resume-all")
    assert r.status_code == 200 and r.json()["data"]["pause_all"] is False
    assert not manager.pause_all_active


@pytest.mark.asyncio
async def test_pause_resume_cancel_endpoints(api):
    client, manager, _ = api
    job_id = await _make_job(manager)
    state = {"count": 0, "target": 10_000}
    manager.spawn(job_id, _counting_worker(manager, job_id, state))
    await _wait_for(lambda: state["count"] > 1)

    r = await client.post(f"/ingest/jobs/{job_id}/pause")
    assert r.status_code == 200
    await _wait_for(_status_is(manager, job_id, "paused"))

    r = await client.post(f"/ingest/jobs/{job_id}/resume")
    assert r.status_code == 200
    await _wait_for(_status_is(manager, job_id, "processing"))

    r = await client.post(f"/ingest/jobs/{job_id}/cancel")
    assert r.status_code == 200
    await _wait_for(_status_is(manager, job_id, "cancelled"))

    # Controls on a finished job -> 409.
    for action in ("pause", "resume", "cancel"):
        r = await client.post(f"/ingest/jobs/{job_id}/{action}")
        assert r.status_code == 409


@pytest.mark.asyncio
async def test_restart_endpoint_relaunches_with_same_params(api):
    client, manager, stub = api
    job_id = await _make_job(
        manager,
        source_path="(fill-missing of doc-1)",
        job_type="fill-missing",
        doc_id="doc-1",
        params={"text": False, "visual": False, "entities": True},
    )
    await manager.fail(job_id, "LLM endpoint unreachable")

    r = await client.post(f"/ingest/jobs/{job_id}/restart")
    assert r.status_code == 200
    new_id = r.json()["data"]["job_id"]
    assert new_id != job_id

    await _wait_for(lambda: len(stub.calls) == 1)
    kind, called_job, doc, do_text, do_visual, do_entities, do_recover = stub.calls[0]
    assert kind == "fill-missing"
    assert called_job == new_id
    assert doc == "doc-1"
    assert (do_text, do_visual, do_entities, do_recover) == (
        False, False, True, False
    )

    new_job = await manager.get(new_id)
    assert new_job.job_type == "fill-missing"
    assert new_job.doc_id == "doc-1"


@pytest.mark.asyncio
async def test_restart_endpoint_guards(api):
    client, manager, _ = api

    # Active job -> 409
    active_id = await _make_job(manager, job_type="resummarize")
    r = await client.post(f"/ingest/jobs/{active_id}/restart")
    assert r.status_code == 409

    # Legacy job (no job_type) -> 400
    legacy_id = await _make_job(manager, job_type="")
    await manager.fail(legacy_id, "boom")
    r = await client.post(f"/ingest/jobs/{legacy_id}/restart")
    assert r.status_code == 400

    # Unknown job -> 404
    r = await client.post("/ingest/jobs/nope/restart")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_list_jobs_endpoint_accepts_pseudo_filters(api):
    client, manager, _ = api
    a = await _make_job(manager)
    b = await _make_job(manager)
    await manager.complete(b)

    r = await client.get("/ingest/jobs", params={"status": "active"})
    assert [j["job_id"] for j in r.json()["data"]] == [a]
    r = await client.get("/ingest/jobs", params={"status": "terminal"})
    assert [j["job_id"] for j in r.json()["data"]] == [b]
    # New fields serialize.
    row = r.json()["data"][0]
    assert "job_type" in row and "current_item" in row


# ------------------------------------------------------------- helpers


def _status_is(manager: JobManager, job_id: str, status: str):
    async def check():
        job = await manager.get(job_id)
        return job is not None and job.status == status

    return check
