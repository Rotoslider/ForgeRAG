"""Ingestion endpoints: upload PDFs, start jobs, poll status."""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel

from backend.models.common import ForgeResult


class DuplicateCheckRequest(BaseModel):
    hashes: list[str]

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
    collection: str = Form("default", description="Collection name (e.g. asm_references, mechanical_design)"),
    categories: str = Form("", description="Comma-separated category names"),
    tags: str = Form("", description="Comma-separated tag names"),
    priority: bool = Form(False, description="Run now: skip the FIFO queue and the pause-all hold"),
) -> ForgeResult:
    """Upload a PDF and enqueue an ingestion job.

    Returns a job_id that can be polled via GET /ingest/jobs/{job_id}.
    The actual processing happens asynchronously in a background task.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename required")
    # Folder uploads (webkitdirectory) send the RELATIVE path as the
    # filename ("ai/paper.pdf") — embedding that in the staged name points
    # into a subdirectory that was never created and the write fails with
    # ENOENT. Keep the basename only, for the staged file AND the display
    # name (also neutralizes any path components an odd client sends).
    clean_name = Path(file.filename).name
    if not clean_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported")

    settings = request.app.state.settings
    jobs = request.app.state.job_manager
    pipeline = request.app.state.pipeline

    # Save upload to disk (uploads are large — don't keep in memory)
    staged_name = f"{uuid.uuid4().hex}_{clean_name}"
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
    col = collection.strip() or "default"

    # Create job record. Collection rides along in job_params so a Restart
    # can reconstruct the exact run_job call.
    job = await jobs.create(
        source_path=str(staged_path),
        filename=clean_name,
        categories=cats,
        tags=tgs,
        job_type="ingest",
        params={"collection": col},
    )

    # Kick off the pipeline in the background, tracked so the job can be
    # paused/stopped from the UI. priority = the "run now" lane: skips the
    # FIFO queue and the pause-all hold.
    if priority:
        jobs.exempt_from_pause(job.job_id)
        jobs.spawn(job.job_id, pipeline.run_job_now(job.job_id, collection=col))
    else:
        jobs.spawn(job.job_id, pipeline.run_job(job.job_id, collection=col))

    logger.info(
        "Enqueued ingestion job %s for %s (collection=%s, categories=%s, tags=%s)",
        job.job_id, clean_name, col, cats, tgs,
    )
    return ForgeResult(
        success=True,
        data={"job_id": job.job_id, "status": job.status, "filename": clean_name},
    )


@router.post("/check-duplicates")
async def check_duplicates(body: DuplicateCheckRequest, request: Request) -> ForgeResult:
    """Look up which of the given SHA-256 hashes already exist as :Document.

    Frontend hashes selected files in the browser and calls this before upload
    so the user can choose to skip or re-ingest. Hashes not present in the
    response are not duplicates.
    """
    if not body.hashes:
        return ForgeResult(success=True, data={"duplicates": {}})
    neo4j = request.app.state.neo4j
    rows = await neo4j.run_query(
        """
        MATCH (d:Document) WHERE d.file_hash IN $hashes
        RETURN d.file_hash AS file_hash,
               d.doc_id AS doc_id,
               d.title AS title,
               d.filename AS filename,
               coalesce(d.collection, 'default') AS collection,
               d.page_count AS page_count,
               toString(d.ingested_at) AS ingested_at
        """,
        {"hashes": body.hashes},
    )
    duplicates = {r["file_hash"]: r for r in rows}
    return ForgeResult(success=True, data={"duplicates": duplicates})


# --------------------------------------------------------------- job control
#
# NOTE: /jobs/controls is registered BEFORE /jobs/{job_id} — FastAPI matches
# routes in declaration order, and "controls" must not be parsed as a job id.


@router.get("/jobs/controls")
async def job_controls(request: Request) -> ForgeResult:
    """Global job-control state: the pause-all switch plus per-status job
    counts (the Active Jobs panel header)."""
    jobs = request.app.state.job_manager
    counts = await jobs.status_counts()
    return ForgeResult(success=True, data={
        "pause_all": jobs.pause_all_active,
        "counts": counts,
        "active": sum(
            counts.get(s, 0) for s in ("queued", "processing", "paused")
        ),
    })


@router.post("/jobs/pause-all")
async def pause_all_jobs(request: Request) -> ForgeResult:
    """Pause every running and queued job (each holds after its current
    page/batch). Frees the GPU/LLM for other work; persists across service
    restarts until resume-all."""
    jobs = request.app.state.job_manager
    await jobs.set_pause_all(True)
    counts = await jobs.status_counts()
    return ForgeResult(success=True, data={"pause_all": True, "counts": counts})


@router.post("/jobs/resume-all")
async def resume_all_jobs(request: Request) -> ForgeResult:
    """Clear the global pause AND any per-job pauses; paused jobs continue
    within a second."""
    jobs = request.app.state.job_manager
    await jobs.set_pause_all(False)
    counts = await jobs.status_counts()
    return ForgeResult(success=True, data={"pause_all": False, "counts": counts})


@router.post("/jobs/{job_id}/pause")
async def pause_job(job_id: str, request: Request) -> ForgeResult:
    """Pause one job at its next checkpoint (current page/batch finishes
    first, so this can take a few seconds to show)."""
    jobs = request.app.state.job_manager
    ok = await jobs.request_pause(job_id)
    if not ok:
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is not active (already finished or unknown)",
        )
    return ForgeResult(success=True, data={"job_id": job_id, "pausing": True})


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str, request: Request) -> ForgeResult:
    """Resume one paused job. If pause-all is on, the job stays held until
    resume-all — the response says so instead of pretending it resumed."""
    jobs = request.app.state.job_manager
    ok = await jobs.request_resume(job_id)
    if not ok:
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is not active (already finished or unknown)",
        )
    return ForgeResult(success=True, data={
        "job_id": job_id,
        "resuming": not jobs.pause_all_active,
        "held_by_pause_all": jobs.pause_all_active,
    })


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, request: Request) -> ForgeResult:
    """Stop a job. Queued jobs stop immediately; running ones stop after
    the current page/batch. All repair job types are safe to restart later —
    they re-check what's missing and never redo finished work."""
    jobs = request.app.state.job_manager
    ok = await jobs.request_cancel(job_id)
    if not ok:
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is not active (already finished or unknown)",
        )
    return ForgeResult(success=True, data={"job_id": job_id, "cancelling": True})


# Job types the restart endpoint knows how to re-launch. Kept in sync with
# _build_restart_coro below; the frontend shows Restart only for these.
RESTARTABLE_JOB_TYPES = {
    "ingest", "fill-missing", "extract-entities", "rebuild-chunks",
    "re-embed", "text-reembed", "resummarize", "autotag",
    "build-communities", "build-summaries", "build-intermediates",
}


def _build_restart_coro(pipeline, new_job_id: str, job):
    """Reconstruct the pipeline call for a finished job. Returns a coroutine
    or None if the job type can't be restarted. Callers must validate with
    RESTARTABLE_JOB_TYPES first so no orphan job rows get created."""
    p = job.job_params or {}
    jt = job.job_type
    doc_id = job.doc_id
    if jt == "ingest":
        return pipeline.run_job(
            new_job_id, collection=p.get("collection", "default")
        )
    if jt == "fill-missing" and doc_id:
        return pipeline.run_fill_missing(
            new_job_id, doc_id,
            do_text=bool(p.get("text", True)),
            do_visual=bool(p.get("visual", True)),
            do_entities=bool(p.get("entities", False)),
            do_recover_text=bool(p.get("recover_text", False)),
        )
    if jt == "extract-entities" and doc_id:
        return pipeline.run_extraction_only(new_job_id, doc_id)
    if jt == "rebuild-chunks" and doc_id:
        return pipeline.run_rebuild_chunks(
            new_job_id, doc_id,
            extract_only=bool(p.get("extract_only")),
            skip_extract=bool(p.get("skip_extract")),
        )
    if jt == "re-embed" and doc_id:
        return pipeline.run_embeddings_only(new_job_id, doc_id)
    if jt == "text-reembed" and doc_id:
        return pipeline.run_text_reembed_only(new_job_id, doc_id)
    if jt == "resummarize":
        return pipeline.run_resummarize(new_job_id)
    if jt == "autotag":
        return pipeline.run_autotag_missing(new_job_id)
    if jt == "build-communities":
        return pipeline.run_communities_only(new_job_id)
    if jt == "build-summaries" and doc_id:
        return pipeline.run_build_summaries(new_job_id, doc_id)
    if jt == "build-intermediates" and doc_id:
        return pipeline.run_build_intermediates(new_job_id, doc_id)
    return None


@router.post("/jobs/{job_id}/restart")
async def restart_job(job_id: str, request: Request) -> ForgeResult:
    """Re-launch a finished (failed/cancelled/completed) job as a NEW job.

    Safe for every repair type: the underlying steps re-query what's
    missing, so a restart continues where the stopped run left off instead
    of redoing finished work.
    """
    jobs = request.app.state.job_manager
    pipeline = request.app.state.pipeline
    job = await jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.status in ("queued", "processing", "paused"):
        raise HTTPException(
            status_code=409, detail="Job is still active — stop it first"
        )
    if job.job_type not in RESTARTABLE_JOB_TYPES:
        raise HTTPException(
            status_code=400,
            detail="This job predates restart support (or is a type that "
            "can't be restarted here) — use the repair buttons on the "
            "Manage tab instead",
        )
    if job.job_type == "ingest" and not Path(job.source_path).exists():
        raise HTTPException(
            status_code=409,
            detail="The staged upload for this job no longer exists "
            "(uploads were cleaned) — re-upload the PDF instead",
        )
    if job.job_type in ("fill-missing", "extract-entities", "rebuild-chunks",
                        "re-embed", "text-reembed", "build-summaries") and not job.doc_id:
        raise HTTPException(
            status_code=409,
            detail="Job has no doc_id recorded — it failed before touching "
            "a document; use the repair buttons on the Manage tab",
        )

    new = await jobs.create(
        source_path=job.source_path,
        filename=job.filename,
        categories=job.requested_categories,
        tags=job.requested_tags,
        job_type=job.job_type,
        doc_id=job.doc_id,
        params=job.job_params,
    )
    coro = _build_restart_coro(pipeline, new.job_id, job)
    assert coro is not None  # guarded by the checks above
    jobs.spawn(new.job_id, coro)
    logger.info(
        "Restarted job %s as %s (%s)", job_id, new.job_id, job.job_type
    )
    return ForgeResult(success=True, data={
        "job_id": new.job_id, "restarted_from": job_id,
        "job_type": job.job_type,
    })


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> ForgeResult:
    """Poll the status of an ingestion job."""
    jobs = request.app.state.job_manager
    job = await jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return ForgeResult(success=True, data=job.model_dump(mode="json"))


@router.get("/jobs/{job_id}/logs")
async def get_job_logs(
    job_id: str,
    request: Request,
    limit: int = Query(500, ge=1, le=5000),
) -> ForgeResult:
    """Return captured log lines for a job, oldest first.

    Lines are captured live while the job runs (INFO and above from every
    backend module active during the job), so this works both for watching
    a running job and for post-mortem on a finished one. Jobs that ran
    before log capture existed return an empty list.
    """
    jobs = request.app.state.job_manager
    job = await jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    lines = await jobs.get_logs(job_id, limit=limit)
    return ForgeResult(
        success=True,
        data={"job_id": job_id, "filename": job.filename, "lines": lines},
    )


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
