"""Schedule & automation endpoints: processing window + watch folder.

Backed by the JobScheduler service (app.state.scheduler). Kept under its
own /schedule prefix (proxied in vite.config.ts for dev).
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

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
    # Same confinement as /browse — the scanner MOVES files out of the
    # watch path, so pointing it at an arbitrary directory must not be
    # possible on an unauthenticated API.
    if payload.get("path"):
        _confine(Path(str(payload["path"])).expanduser(), sched)
    try:
        await sched.update_watch(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ForgeResult(success=True, data={
        "watch": sched.watch, "status": sched.status(),
    })


# Directory-picker confinement: the API has no auth and (by deliberate
# choice) binds beyond localhost so local agents can use it, so the browse
# endpoint must not enumerate the whole filesystem. Books live in the home
# directory (Downloads subfolders) and on external drives — allow exactly
# those roots plus the data dir.
def _browse_roots(sched) -> list[Path]:
    return [
        Path.home().resolve(),
        Path("/media"),
        Path("/mnt"),
        sched.default_watch_path().parent.resolve(),
    ]


def _confine(base: Path, sched) -> Path:
    resolved = base.resolve()
    for root in _browse_roots(sched):
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise HTTPException(
        status_code=403,
        detail="browsing is limited to the home directory, external "
        "drives (/media, /mnt), and the data directory",
    )


@router.get("/browse")
async def browse_directories(
    request: Request,
    path: str | None = Query(None, description="Directory to list; omit for a sensible start"),
) -> ForgeResult:
    """List subdirectories of a server-side folder — the GUI's folder
    picker for choosing the watch inbox. Directories only, no files.
    Confined to the home directory, external drives, and the data dir."""
    sched = request.app.state.scheduler
    if path:
        base = Path(path).expanduser()
    elif sched.watch.get("path"):
        base = Path(sched.watch["path"])
    else:
        base = sched.default_watch_path().parent
    try:
        base = _confine(base, sched)
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not base.is_dir():
        raise HTTPException(
            status_code=400, detail=f"not a directory: {base}"
        )
    try:
        dirs = sorted(
            (d for d in base.iterdir()
             if d.is_dir() and not d.name.startswith(".")),
            key=lambda d: d.name.lower(),
        )
    except PermissionError:
        raise HTTPException(
            status_code=400, detail=f"permission denied: {base}"
        )
    return ForgeResult(success=True, data={
        "path": str(base),
        "parent": str(base.parent) if base.parent != base else None,
        "dirs": [{"name": d.name, "path": str(d)} for d in dirs],
        "home": str(Path.home()),
        "default": str(sched.default_watch_path()),
    })


@router.post("/watch/open-folder")
async def open_watch_folder(request: Request) -> ForgeResult:
    """Open the inbox folder in the desktop file manager (on the ForgeRAG
    machine itself — this is a single-box deployment)."""
    sched = request.app.state.scheduler
    try:
        sched.open_watch_folder()
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OSError as exc:  # xdg-open missing, no display, ...
        raise HTTPException(status_code=500, detail=f"could not open: {exc}") from exc
    return ForgeResult(success=True, data={"opened": sched.watch["path"]})


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
    if result.get("error"):
        # A missing/unmounted inbox used to come back success:true with an
        # "error" key the frontend never read — the GUI showed a green
        # "Scan done: 0 queued" over a hard failure.
        return ForgeResult(success=False, reason=str(result["error"]), data={
            **result, "status": sched.status(),
        })
    return ForgeResult(success=True, data={
        **result, "status": sched.status(),
    })
