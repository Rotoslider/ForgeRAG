"""Schedule & automation endpoints: processing window + watch folder.

Backed by the JobScheduler service (app.state.scheduler). Kept under its
own /schedule prefix (proxied in vite.config.ts for dev).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from backend.models.common import ForgeResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/schedule", tags=["schedule"])


@router.get("")
async def get_schedule(request: Request) -> ForgeResult:
    """Current schedule + watch-folder config and live status (window
    open/closed, next boundary, inbox counts, recent scheduler events)."""
    sched = request.app.state.scheduler
    return ForgeResult(success=True, data={
        "schedule": sched.schedule,
        "watch": sched.watch,
        "status": sched.status(),
    })


@router.put("")
async def put_schedule(request: Request, payload: dict) -> ForgeResult:
    """Update the processing-window schedule.

    Body (all optional): {enabled, start "HH:MM", end "HH:MM", days [0-6,
    Mon=0]}. Takes effect within one tick — enabling mid-window resumes
    jobs immediately.
    """
    sched = request.app.state.scheduler
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="body must be an object")
    try:
        await sched.update_schedule(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ForgeResult(success=True, data={
        "schedule": sched.schedule, "status": sched.status(),
    })


@router.put("/watch")
async def put_watch(request: Request, payload: dict) -> ForgeResult:
    """Update the watch-folder (auto-ingest inbox) config.

    Body (all optional): {enabled, path, collection}. Empty path with
    enabled=true selects the default inbox under the data dir (created if
    missing).
    """
    sched = request.app.state.scheduler
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="body must be an object")
    try:
        await sched.update_watch(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ForgeResult(success=True, data={
        "watch": sched.watch, "status": sched.status(),
    })


@router.post("/watch/scan-now")
async def scan_now(request: Request) -> ForgeResult:
    """Scan the inbox immediately, skipping the two-pass stability wait.

    Ignores pause-all on purpose: it's an explicit user action — but the
    queued jobs still hold at their first checkpoint while paused.
    """
    sched = request.app.state.scheduler
    if not sched.watch["enabled"] or not sched.watch["path"]:
        raise HTTPException(
            status_code=409, detail="watch folder is not enabled"
        )
    result = await sched.scan_watch_folder(force=True)
    return ForgeResult(success=True, data={
        **result, "status": sched.status(),
    })
