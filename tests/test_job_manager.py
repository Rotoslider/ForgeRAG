"""Tests for the JobManager step ledger and per-job log capture."""

from __future__ import annotations

import asyncio
import logging

import pytest

from backend.ingestion.job_logs import (
    current_job_id,
    install_job_log_handler,
)
from backend.ingestion.job_manager import JobManager


@pytest.fixture()
def manager(tmp_path):
    return JobManager(tmp_path / "jobs.sqlite")


async def _make_job(manager: JobManager) -> str:
    job = await manager.create(
        source_path="/tmp/x.pdf", filename="x.pdf", categories=[], tags=[]
    )
    return job.job_id


@pytest.mark.asyncio
async def test_step_ledger_lifecycle(manager):
    await manager.init()
    job_id = await _make_job(manager)

    await manager.set_steps(job_id, ["registering", "rendering_pages", "embedding_text"])
    job = await manager.get(job_id)
    assert [s.name for s in job.steps] == ["registering", "rendering_pages", "embedding_text"]
    assert all(s.status == "pending" for s in job.steps)

    await manager.update_step(job_id, "registering", "running")
    await manager.update_step(job_id, "registering", "done", detail="42 pages")
    await manager.update_step(job_id, "rendering_pages", "error", detail="boom")
    await manager.update_step(job_id, "embedding_text", "skipped", detail="service off")

    job = await manager.get(job_id)
    by_name = {s.name: s for s in job.steps}
    assert by_name["registering"].status == "done"
    assert by_name["registering"].detail == "42 pages"
    assert by_name["registering"].started_at is not None
    assert by_name["registering"].finished_at is not None
    assert by_name["rendering_pages"].status == "error"
    assert by_name["embedding_text"].status == "skipped"


@pytest.mark.asyncio
async def test_step_not_in_plan_is_appended(manager):
    await manager.init()
    job_id = await _make_job(manager)
    await manager.set_steps(job_id, ["registering"])
    await manager.update_step(job_id, "surprise_step", "done")
    job = await manager.get(job_id)
    assert [s.name for s in job.steps] == ["registering", "surprise_step"]


@pytest.mark.asyncio
async def test_stale_running_steps_marked_on_restart(manager, tmp_path):
    await manager.init()
    job_id = await _make_job(manager)
    await manager.set_steps(job_id, ["registering", "embedding_text"])
    await manager.update(job_id, status="processing")
    await manager.update_step(job_id, "registering", "done")
    await manager.update_step(job_id, "embedding_text", "running")

    # Simulate a service restart: a fresh manager on the same DB file.
    manager2 = JobManager(tmp_path / "jobs.sqlite")
    await manager2.init()
    job = await manager2.get(job_id)
    assert job.status == "failed"
    by_name = {s.name: s for s in job.steps}
    assert by_name["registering"].status == "done"  # untouched
    assert by_name["embedding_text"].status == "error"
    assert "restarted" in (by_name["embedding_text"].detail or "")


@pytest.mark.asyncio
async def test_log_capture_and_retrieval(manager):
    await manager.init()
    job_id = await _make_job(manager)
    handler = install_job_log_handler(manager.log_buffer)
    logger = logging.getLogger("test.capture")
    logger.setLevel(logging.INFO)
    try:
        token = current_job_id.set(job_id)
        logger.info("hello from the pipeline")
        logger.warning("something partial failed: page %d", 7)
        current_job_id.reset(token)
        logger.info("this line has no job context and must NOT be captured")

        # Logs from a worker thread inherit the contextvar via to_thread.
        current_job_id.set(job_id)
        await asyncio.to_thread(logger.info, "from a worker thread")

        lines = await manager.get_logs(job_id)
        messages = [ln["message"] for ln in lines]
        assert "hello from the pipeline" in messages
        assert "something partial failed: page 7" in messages
        assert "from a worker thread" in messages
        assert not any("must NOT be captured" in m for m in messages)
        levels = {ln["message"]: ln["level"] for ln in lines}
        assert levels["something partial failed: page 7"] == "WARNING"

        # get_logs drains: buffer is now empty and a second read still works.
        assert len(manager.log_buffer) == 0
        again = await manager.get_logs(job_id)
        assert len(again) == len(lines)
    finally:
        logging.getLogger().removeHandler(handler)


@pytest.mark.asyncio
async def test_logs_endpoint_empty_for_unknown_job(manager):
    await manager.init()
    job_id = await _make_job(manager)
    assert await manager.get_logs(job_id) == []


@pytest.mark.asyncio
async def test_migration_adds_steps_column(tmp_path):
    """A pre-ledger jobs.sqlite (no steps column) is migrated on init."""
    import aiosqlite

    db_path = tmp_path / "jobs.sqlite"
    old_schema = """
    CREATE TABLE jobs (
        job_id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        current_step TEXT NOT NULL,
        progress_pct REAL NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        error_message TEXT,
        source_path TEXT NOT NULL,
        filename TEXT NOT NULL,
        requested_categories TEXT NOT NULL DEFAULT '[]',
        requested_tags TEXT NOT NULL DEFAULT '[]',
        doc_id TEXT,
        file_hash TEXT,
        pages_processed INTEGER NOT NULL DEFAULT 0,
        pages_total INTEGER NOT NULL DEFAULT 0
    );
    """
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(old_schema)
        await db.execute(
            """INSERT INTO jobs (job_id, status, current_step, created_at,
               updated_at, source_path, filename)
               VALUES ('old-job', 'completed', 'done',
                       '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00',
                       '/tmp/old.pdf', 'old.pdf')"""
        )
        await db.commit()

    mgr = JobManager(db_path)
    await mgr.init()
    job = await mgr.get("old-job")
    assert job is not None
    assert job.steps == []  # old job, empty ledger — no crash
