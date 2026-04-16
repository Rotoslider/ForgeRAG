"""Ingestion endpoints: upload PDFs, start jobs, poll status."""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

from backend.models.common import ForgeResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])


def _uploads_dir(settings) -> Path:
    """Directory where uploaded PDFs are staged before processing."""
    d = Path(settings.server.data_dir) / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


@router.post("")
async def start_ingestion(
    request: Request,
    file: UploadFile = File(..., description="PDF file to ingest"),
    categories: str = Form("", description="Comma-separated category names"),
    tags: str = Form("", description="Comma-separated tag names"),
) -> ForgeResult:
    """Upload a PDF and enqueue an ingestion job.

    Returns a job_id that can be polled via GET /ingest/jobs/{job_id}.
    The actual processing happens asynchronously in a background task.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename required")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported")

    settings = request.app.state.settings
    jobs = request.app.state.job_manager
    pipeline = request.app.state.pipeline

    # Save upload to disk (uploads are large — don't keep in memory)
    staged_name = f"{uuid.uuid4().hex}_{file.filename}"
    staged_path = _uploads_dir(settings) / staged_name
    try:
        with staged_path.open("wb") as out:
            while chunk := await file.read(1 << 20):  # 1 MB chunks
                out.write(chunk)
    except Exception as exc:  # noqa: BLE001
        staged_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    # Parse categories/tags (comma-separated, trim whitespace, drop empty)
    cats = [c.strip() for c in categories.split(",") if c.strip()]
    tgs = [t.strip() for t in tags.split(",") if t.strip()]

    # Create job record
    job = await jobs.create(
        source_path=str(staged_path),
        filename=file.filename,
        categories=cats,
        tags=tgs,
    )

    # Kick off the pipeline in the background. Do not await — we want to
    # return the job_id to the caller immediately so they can poll progress.
    asyncio.create_task(pipeline.run_job(job.job_id))

    logger.info(
        "Enqueued ingestion job %s for %s (categories=%s, tags=%s)",
        job.job_id, file.filename, cats, tgs,
    )
    return ForgeResult(
        success=True,
        data={"job_id": job.job_id, "status": job.status, "filename": file.filename},
    )


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> ForgeResult:
    """Poll the status of an ingestion job."""
    jobs = request.app.state.job_manager
    job = await jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return ForgeResult(success=True, data=job.model_dump(mode="json"))


@router.get("/jobs")
async def list_jobs(
    request: Request,
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=500),
) -> ForgeResult:
    """List recent ingestion jobs, newest first."""
    jobs = request.app.state.job_manager
    rows = await jobs.list_recent(status=status, limit=limit)
    return ForgeResult(
        success=True,
        data=[j.model_dump(mode="json") for j in rows],
    )
