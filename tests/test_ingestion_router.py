"""API-level tests for the ingestion job endpoints (steps + logs)."""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.ingestion.job_logs import current_job_id, install_job_log_handler
from backend.ingestion.job_manager import JobManager
from backend.routers import ingestion


@pytest.fixture()
def client(tmp_path):
    app = FastAPI()
    app.include_router(ingestion.router)
    manager = JobManager(tmp_path / "jobs.sqlite")
    app.state.job_manager = manager
    with TestClient(app) as c:
        yield c, manager


def test_job_response_includes_steps(client):
    c, manager = client
    import asyncio as _asyncio

    async def seed():
        await manager.init()
        job = await manager.create(
            source_path="/tmp/a.pdf", filename="a.pdf", categories=[], tags=[]
        )
        await manager.set_steps(job.job_id, ["registering", "embedding_text"])
        await manager.update_step(job.job_id, "registering", "done", detail="10 pages")
        await manager.update_step(
            job.job_id, "embedding_text", "skipped", detail="service off"
        )
        return job.job_id

    # The manager is independent of the app's event loop, so seeding on a
    # private loop via asyncio.run is safe here.
    job_id = _asyncio.run(seed())

    r = c.get(f"/ingest/jobs/{job_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    steps = body["data"]["steps"]
    assert [s["name"] for s in steps] == ["registering", "embedding_text"]
    assert steps[0]["status"] == "done"
    assert steps[0]["detail"] == "10 pages"
    assert steps[1]["status"] == "skipped"

    r = c.get("/ingest/jobs")
    assert r.status_code == 200
    rows = r.json()["data"]
    assert rows and rows[0]["steps"][0]["name"] == "registering"


def test_logs_endpoint(client):
    c, manager = client
    import asyncio as _asyncio

    async def seed():
        await manager.init()
        job = await manager.create(
            source_path="/tmp/b.pdf", filename="b.pdf", categories=[], tags=[]
        )
        handler = install_job_log_handler(manager.log_buffer)
        try:
            token = current_job_id.set(job.job_id)
            logging.getLogger("seed").warning("chunker exploded on page 3")
            current_job_id.reset(token)
        finally:
            logging.getLogger().removeHandler(handler)
        return job.job_id

    job_id = _asyncio.run(seed())

    r = c.get(f"/ingest/jobs/{job_id}/logs")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["job_id"] == job_id
    assert any("chunker exploded" in ln["message"] for ln in data["lines"])
    assert data["lines"][0]["level"] == "WARNING"

    r = c.get("/ingest/jobs/nonexistent/logs")
    assert r.status_code == 404


def test_folder_upload_relative_path_is_flattened(client, tmp_path):
    """Folder uploads (webkitdirectory) send 'subdir/file.pdf' as the
    filename. Staging must flatten to the basename — embedding the path
    pointed into a never-created subdirectory and failed with ENOENT."""
    c, manager = client
    import asyncio as _asyncio

    _asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        manager.init()
    )

    class _Server:
        data_dir = str(tmp_path)

    class _Settings:
        server = _Server()

    class _PipelineStub:
        async def run_job(self, job_id, collection):
            pass

        async def run_job_now(self, job_id, collection):
            pass

    c.app.state.settings = _Settings()
    c.app.state.pipeline = _PipelineStub()

    r = c.post(
        "/ingest",
        files={"file": ("ai/paper one.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["data"]["filename"] == "paper one.pdf"
    staged = list((tmp_path / "uploads").glob("*_paper one.pdf"))
    assert len(staged) == 1
    # No stray subdirectory was created from the relative path.
    assert not list((tmp_path / "uploads").glob("*_ai"))
