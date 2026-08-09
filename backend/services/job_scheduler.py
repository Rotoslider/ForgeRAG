"""Scheduled processing window + watch-folder auto-ingest.

Two automations, both driven by one background loop:

1. **Processing window** — "jobs may run from 21:00 to 06:30 on these
   days". At a window start the scheduler fires resume-all, at a window
   end pause-all, through the exact same JobManager switch the GUI buttons
   use. It acts only at boundaries, so a manual pause/resume in between
   holds until the next boundary. Boundary firing is catch-up based: the
   most recent boundary is fired once even if the service was down when it
   passed (a reboot at 22:00 still opens the 21:00 window), and enabling
   or editing the schedule applies the current window state immediately.

2. **Watch folder** — an inbox directory scanned about once a minute
   *while processing is allowed* (pause-all off). PDFs are picked up only
   once their size/mtime is stable across two scans (never mid-copy),
   hash-checked against the library (duplicates are filed to
   ``duplicates/`` without a job), then staged into uploads/ and ingested
   through the normal pipeline — same semaphore, same LLM caps, same job
   cards — and the original is moved to ``ingested/``. With a schedule
   enabled, files dropped during the day simply wait for the window.

Config and state persist in the jobs.sqlite meta table.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")

_TICK_SECONDS = 15.0
_SCAN_INTERVAL_SECONDS = 60.0
_MAX_FILES_PER_SCAN = 25
_EVENT_LIMIT = 30
# How far back catch-up looks for a missed boundary. Past this, firing a
# stale boundary is more surprising than useful.
_CATCHUP_DAYS = 8

_META_CONFIG = "scheduler_config"
_META_STATE = "scheduler_state"

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

DEFAULT_SCHEDULE: dict[str, Any] = {
    "enabled": False,
    "start": "21:00",   # window start -> resume-all
    "end": "06:30",     # window end -> pause-all (may cross midnight)
    "days": [0, 1, 2, 3, 4, 5, 6],  # weekday of the window START; Mon=0
}

DEFAULT_WATCH: dict[str, Any] = {
    "enabled": False,
    "path": "",          # empty -> <data_dir>/ingest-inbox
    "collection": "default",
}


def _parse_hhmm(value: str):
    m = _TIME_RE.match(value or "")
    if not m:
        raise ValueError(f"invalid time {value!r} — use HH:MM (24h)")
    return int(m.group(1)), int(m.group(2))


def window_pairs(
    schedule: dict[str, Any], now: datetime, days_back: int, days_forward: int = 2
) -> list[tuple[datetime, datetime]]:
    """(start, end) datetimes for every enabled window near `now`.

    The day mask applies to the window's START day; the end is start +
    duration, so a 21:00->06:30 window started Monday ends Tuesday 06:30.
    """
    sh, sm = _parse_hhmm(schedule["start"])
    eh, em = _parse_hhmm(schedule["end"])
    duration = ((eh - sh) * 60 + (em - sm)) % (24 * 60)
    if duration == 0:
        return []
    pairs = []
    for offset in range(-days_back, days_forward + 1):
        day = (now + timedelta(days=offset)).date()
        if day.weekday() in schedule["days"]:
            start = datetime.combine(day, datetime.min.time()).replace(
                hour=sh, minute=sm
            )
            pairs.append((start, start + timedelta(minutes=duration)))
    return pairs


def compute_last_boundary(
    schedule: dict[str, Any], now: datetime
) -> tuple[datetime, str] | None:
    """Most recent boundary at or before `now`: (when, "resume"|"pause").

    "resume" for a window start, "pause" for a window end. Later boundary
    wins; on a tie (end of one window == start of the next) resume wins so
    back-to-back windows don't glitch closed.
    """
    boundaries: list[tuple[datetime, int, str]] = []
    for start, end in window_pairs(schedule, now, _CATCHUP_DAYS):
        if start <= now:
            boundaries.append((start, 1, "resume"))
        if end <= now:
            boundaries.append((end, 0, "pause"))
    if not boundaries:
        return None
    when, _prio, action = max(boundaries)
    return when, action


def compute_next_boundary(
    schedule: dict[str, Any], now: datetime
) -> tuple[datetime, str] | None:
    """Earliest boundary strictly after `now`."""
    boundaries: list[tuple[datetime, int, str]] = []
    for start, end in window_pairs(schedule, now, 1, days_forward=8):
        if start > now:
            boundaries.append((start, 0, "resume"))
        if end > now:
            boundaries.append((end, 1, "pause"))
    if not boundaries:
        return None
    when, _prio, action = min(boundaries)
    return when, action


class JobScheduler:
    """Background loop owning the processing-window schedule and the
    watch-folder scanner. One instance per app, started from the lifespan."""

    def __init__(self, *, job_manager, pipeline, settings):
        self.jobs = job_manager
        self.pipeline = pipeline
        self.settings = settings
        self.schedule: dict[str, Any] = dict(DEFAULT_SCHEDULE)
        self.watch: dict[str, Any] = dict(DEFAULT_WATCH)
        self._events: list[dict[str, str]] = []
        self._last_boundary: str | None = None
        self._task: asyncio.Task | None = None
        # path -> (size, mtime) from the previous scan; a file is ingested
        # only when two consecutive scans see it unchanged.
        self._file_snapshots: dict[str, tuple[int, float]] = {}
        self._last_scan_at: datetime | None = None
        self._last_scan_note: str = ""
        self._scan_lock = asyncio.Lock()
        # Injectable for tests; opens the file manager in production.
        self._opener = lambda cmd, env: subprocess.Popen(
            cmd, env=env, start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        await self._load()
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "JobScheduler started (schedule %s, watch %s)",
            "on" if self.schedule["enabled"] else "off",
            "on" if self.watch["enabled"] else "off",
        )

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _load(self) -> None:
        raw = await self.jobs.meta_get(_META_CONFIG)
        if raw:
            try:
                cfg = json.loads(raw)
                self.schedule = {**DEFAULT_SCHEDULE, **cfg.get("schedule", {})}
                self.watch = {**DEFAULT_WATCH, **cfg.get("watch", {})}
            except json.JSONDecodeError:
                logger.warning("Corrupt scheduler config — using defaults")
        raw = await self.jobs.meta_get(_META_STATE)
        if raw:
            try:
                state = json.loads(raw)
                self._last_boundary = state.get("last_boundary")
                self._events = list(state.get("events", []))[-_EVENT_LIMIT:]
            except json.JSONDecodeError:
                pass

    async def _save_config(self) -> None:
        await self.jobs.meta_set(
            _META_CONFIG,
            json.dumps({"schedule": self.schedule, "watch": self.watch}),
        )

    async def _save_state(self) -> None:
        await self.jobs.meta_set(
            _META_STATE,
            json.dumps(
                {"last_boundary": self._last_boundary, "events": self._events}
            ),
        )

    async def _event(self, message: str) -> None:
        logger.info("Scheduler: %s", message)
        self._events.append(
            {"ts": datetime.now().isoformat(timespec="seconds"), "message": message}
        )
        self._events = self._events[-_EVENT_LIMIT:]
        await self._save_state()

    # ------------------------------------------------------------ config API

    async def update_schedule(self, patch: dict[str, Any]) -> None:
        """Validate + apply schedule config. The current window state is
        enforced on the next tick (within seconds), so enabling a schedule
        mid-window resumes immediately — the GUI says so."""
        cfg = {**self.schedule, **patch}
        _parse_hhmm(cfg["start"])
        _parse_hhmm(cfg["end"])
        if cfg["start"] == cfg["end"]:
            raise ValueError("start and end must differ")
        days = cfg.get("days") or []
        if not isinstance(days, list) or not all(
            isinstance(d, int) and 0 <= d <= 6 for d in days
        ):
            raise ValueError("days must be a list of weekday numbers 0-6 (Mon=0)")
        if cfg["enabled"] and not days:
            raise ValueError("select at least one day")
        cfg["days"] = sorted(set(days))
        cfg["enabled"] = bool(cfg["enabled"])
        self.schedule = cfg
        # Forget the last fired boundary so the next tick re-applies the
        # current expected state under the NEW config.
        self._last_boundary = None
        await self._save_config()
        await self._event(
            "schedule updated: "
            + (
                f"run {cfg['start']}–{cfg['end']} on "
                + ",".join(DAY_NAMES[d] for d in cfg["days"])
                if cfg["enabled"]
                else "disabled"
            )
        )

    def default_watch_path(self) -> Path:
        return Path(self.settings.server.data_dir) / "ingest-inbox"

    async def update_watch(self, patch: dict[str, Any]) -> None:
        cfg = {**self.watch, **patch}
        cfg["enabled"] = bool(cfg["enabled"])
        cfg["collection"] = (cfg.get("collection") or "default").strip() or "default"
        path = (cfg.get("path") or "").strip()
        if cfg["enabled"]:
            if not path:
                inbox = self.default_watch_path()
                inbox.mkdir(parents=True, exist_ok=True)
                path = str(inbox)
            else:
                p = Path(path)
                if not p.is_absolute():
                    raise ValueError("watch path must be absolute")
                if not p.is_dir():
                    raise ValueError(f"watch path does not exist: {path}")
        cfg["path"] = path
        self.watch = cfg
        self._file_snapshots = {}
        await self._save_config()
        await self._event(
            f"watch folder {'enabled: ' + path if cfg['enabled'] else 'disabled'}"
        )

    def status(self) -> dict[str, Any]:
        now = datetime.now()
        nxt = (
            compute_next_boundary(self.schedule, now)
            if self.schedule["enabled"]
            else None
        )
        last = (
            compute_last_boundary(self.schedule, now)
            if self.schedule["enabled"]
            else None
        )
        pending = 0
        path_ok = False
        if self.watch["path"]:
            inbox = Path(self.watch["path"])
            path_ok = inbox.is_dir()
            if path_ok:
                pending = len(self._list_inbox(inbox))
        return {
            "now": now.isoformat(timespec="seconds"),
            "pause_all": self.jobs.pause_all_active,
            "window_open": (last[1] == "resume") if last else None,
            "next_boundary": (
                {"at": nxt[0].isoformat(timespec="seconds"), "action": nxt[1]}
                if nxt
                else None
            ),
            "watch": {
                "path_ok": path_ok,
                "pending_files": pending,
                "last_scan_at": (
                    self._last_scan_at.isoformat(timespec="seconds")
                    if self._last_scan_at
                    else None
                ),
                "last_scan_note": self._last_scan_note,
                "default_path": str(self.default_watch_path()),
            },
            "events": list(reversed(self._events)),
        }

    # ------------------------------------------------------------------ loop

    async def _loop(self) -> None:
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — the loop must survive
                logger.exception("Scheduler tick failed")
            await asyncio.sleep(_TICK_SECONDS)

    async def _tick(self) -> None:
        if self.schedule["enabled"]:
            boundary = compute_last_boundary(self.schedule, datetime.now())
            if boundary is not None:
                when, action = boundary
                key = f"{when.isoformat()}|{action}"
                if key != self._last_boundary:
                    pause = action == "pause"
                    if self.jobs.pause_all_active != pause:
                        await self.jobs.set_pause_all(pause)
                        await self._event(
                            f"window {'closed — paused all jobs' if pause else 'opened — resumed all jobs'}"
                            f" (boundary {when.strftime('%a %H:%M')})"
                        )
                    self._last_boundary = key
                    await self._save_state()

        if (
            self.watch["enabled"]
            and self.watch["path"]
            and not self.jobs.pause_all_active
        ):
            due = (
                self._last_scan_at is None
                or (datetime.now() - self._last_scan_at).total_seconds()
                >= _SCAN_INTERVAL_SECONDS
            )
            if due:
                await self.scan_watch_folder()

    # ----------------------------------------------------------- watch folder

    # Output subtrees the scanner must never re-scan.
    _SKIP_DIRS = frozenset({"ingested", "duplicates"})

    @classmethod
    def _list_inbox(cls, inbox: Path) -> list[Path]:
        """PDFs anywhere under the inbox (subfolders included), excluding
        the ingested/ and duplicates/ output trees.

        Prunes the skip dirs DURING traversal — the old rglob descended the
        entire ingested/ archive (which grows by one file per processed PDF
        forever) before filtering, a stat storm re-paid on every walk.
        """
        out = []
        for root, dirs, files in os.walk(inbox):
            if Path(root) == inbox:
                dirs[:] = [d for d in dirs if d not in cls._SKIP_DIRS]
            for f in files:
                if f.lower().endswith(".pdf"):
                    out.append(Path(root) / f)
        return sorted(out)

    def open_watch_folder(self) -> None:
        """Open the inbox in the desktop file manager. ForgeRAG is a
        single-box deployment, so the window opens on the server's own
        desktop session — the GUI env vars are filled in because the
        systemd service doesn't inherit them."""
        path = self.watch.get("path")
        if not path or not Path(path).is_dir():
            raise ValueError("watch folder is not configured or missing")
        uid = os.getuid()
        env = dict(os.environ)
        env.setdefault("DISPLAY", ":0")
        env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{uid}")
        env.setdefault(
            "DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{uid}/bus"
        )
        self._opener(["xdg-open", path], env)

    async def scan_watch_folder(self, force: bool = False) -> dict[str, Any]:
        """One scan pass. `force` skips the stability wait (used by the
        GUI's "scan now" so a just-dropped file doesn't need two passes)."""
        async with self._scan_lock:
            return await self._scan_locked(force)

    async def _scan_locked(self, force: bool) -> dict[str, Any]:
        self._last_scan_at = datetime.now()
        inbox = Path(self.watch["path"])
        if not inbox.is_dir():
            self._last_scan_note = f"inbox missing: {inbox}"
            return {"queued": 0, "skipped": 0, "error": self._last_scan_note}

        pdfs = self._list_inbox(inbox)
        ready: list[Path] = []
        snapshots: dict[str, tuple[int, float]] = {}
        for p in pdfs:
            try:
                st = p.stat()
            except OSError:
                continue
            snapshots[str(p)] = (st.st_size, st.st_mtime)
            if force or self._file_snapshots.get(str(p)) == (
                st.st_size,
                st.st_mtime,
            ):
                ready.append(p)
        self._file_snapshots = snapshots

        queued = 0
        duplicates = 0
        for p in ready[:_MAX_FILES_PER_SCAN]:
            try:
                if await self._ingest_inbox_file(inbox, p):
                    queued += 1
                else:
                    duplicates += 1
            except Exception as exc:  # noqa: BLE001 — one bad file must not stop the rest
                logger.exception("Watch-folder ingest failed for %s", p.name)
                await self._event(f"inbox error on {p.name}: {exc}")

        waiting = len(pdfs) - queued - duplicates
        self._last_scan_note = (
            f"queued {queued}, duplicates {duplicates}, waiting {max(0, waiting)}"
            if pdfs
            else "inbox empty"
        )
        if queued or duplicates:
            await self._event(
                f"inbox scan: queued {queued} file(s)"
                + (f", filed {duplicates} duplicate(s)" if duplicates else "")
            )
        return {"queued": queued, "duplicates": duplicates, "waiting": waiting}

    async def _ingest_inbox_file(self, inbox: Path, p: Path) -> bool:
        """Queue one inbox PDF. Returns False if it was a duplicate (filed
        to duplicates/ without creating a job). Files from subfolders keep
        their relative path under ingested/ and duplicates/."""
        rel = p.relative_to(inbox)
        file_hash = await _sha256_helper(p)
        existing = await self.pipeline.neo4j.run_query(
            "MATCH (d:Document {file_hash: $h}) "
            "RETURN d.title AS title LIMIT 1",
            {"h": file_hash},
        )
        if existing:
            _move_out(p, inbox / "duplicates" / rel.parent)
            await self._event(
                f"duplicate skipped: {rel} (already ingested as "
                f"“{existing[0]['title']}”) — moved to duplicates/"
            )
            return False

        uploads = Path(self.settings.server.data_dir) / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        staged = uploads / f"{uuid.uuid4().hex}_{p.name}"
        shutil.copy2(p, staged)

        collection = self.watch.get("collection") or "default"
        job = await self.jobs.create(
            source_path=str(staged),
            filename=p.name,
            categories=[],
            tags=[],
            job_type="ingest",
            params={"collection": collection, "source": "watch-folder"},
        )
        self.jobs.spawn(
            job.job_id, self.pipeline.run_job(job.job_id, collection=collection)
        )
        _move_out(p, inbox / "ingested" / rel.parent)
        await self._event(f"queued from inbox: {rel}")
        return True


async def _sha256_helper(path: Path) -> str:
    from backend.ingestion.pipeline import _sha256_file

    return await _sha256_file(path)


def _move_out(src: Path, dest_dir: Path) -> Path:
    """Move a processed inbox file into a subfolder, never overwriting."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = dest_dir / f"{src.stem}.{stamp}{src.suffix}"
    shutil.move(str(src), str(dest))
    return dest
