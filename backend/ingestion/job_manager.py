"""SQLite-backed ingestion job queue.

A lightweight alternative to Redis/Celery for a single-server batch workload.
Uses aiosqlite for async access, with WAL mode for concurrent read/write.

Jobs are long-lived records — not deleted after completion, so users can
inspect history via GET /ingest/jobs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from backend.ingestion.job_logs import LogRow, make_log_buffer
from backend.models.ingestion import Job, JobStatus, JobStep, StepRecord, StepStatus

logger = logging.getLogger(__name__)

# How long a connection waits for a competing writer before giving up.
# WAL serializes writers; without a busy timeout, concurrent ingestion jobs
# writing job progress collide and SQLite raises "database is locked"
# immediately. 30 s is far longer than any single job-row write needs.
_CONNECT_TIMEOUT = 30.0
_BUSY_TIMEOUT_MS = 30000


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    current_step TEXT NOT NULL,
    progress_pct REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    error_message TEXT,
    source_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    requested_categories TEXT NOT NULL DEFAULT '[]',  -- JSON array
    requested_tags TEXT NOT NULL DEFAULT '[]',        -- JSON array
    doc_id TEXT,
    file_hash TEXT,
    pages_processed INTEGER NOT NULL DEFAULT 0,
    pages_total INTEGER NOT NULL DEFAULT 0,
    steps TEXT NOT NULL DEFAULT '[]',                 -- JSON array of StepRecord
    job_type TEXT NOT NULL DEFAULT '',                -- ingest / fill-missing / ...
    job_params TEXT NOT NULL DEFAULT '{}',            -- JSON kwargs for restart
    current_item TEXT                                 -- live "now working on" label
);

CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status);
CREATE INDEX IF NOT EXISTS jobs_created_idx ON jobs(created_at);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS job_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    level TEXT NOT NULL,
    logger TEXT NOT NULL,
    message TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS job_logs_job_idx ON job_logs(job_id, id);
"""

# How many jobs keep their logs around. Older jobs' logs are pruned at
# startup so the DB doesn't grow forever.
_LOG_RETAIN_JOBS = 300
# Per-job line cap, enforced when a job finishes.
_LOG_RETAIN_LINES = 5000


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobCancelled(BaseException):
    """Raised inside a job task when the user stops the job.

    Deliberately a BaseException: every pipeline entry point wraps its work
    in `except Exception` to record failures, and a user-requested stop must
    not be converted into a job *failure* by those handlers. The checkpoint
    that raises this has already finalized the job row as 'cancelled'.
    """


# Statuses a live background task can be in. 'paused' is only ever set by
# checkpoint() while the task is blocked inside it.
ACTIVE_STATUSES = ("queued", "processing", "paused")

# How often a paused job re-checks whether it may continue.
_PAUSE_POLL_SECONDS = 0.5


class JobManager:
    """Async SQLite-backed job store. One instance per process."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._ready = False
        # SQLite allows only one writer at a time. Serializing writes from
        # this process through an asyncio lock removes intra-process
        # contention entirely; the busy_timeout below covers other processes
        # (e.g. the maintenance scripts) that open the same file.
        self._write_lock = asyncio.Lock()
        # Per-job log lines captured by JobLogHandler (see job_logs.py).
        # Drained into the job_logs table on every job write and log read.
        self.log_buffer: deque[LogRow] = make_log_buffer()
        # ---- job control (in-memory; tasks die with the process) ----
        # Live background tasks by job_id, registered via spawn(). This is
        # what makes pause/stop possible at all — before it existed, every
        # create_task() result was dropped and running jobs were unreachable.
        self._tasks: dict[str, asyncio.Task] = {}
        # Jobs whose next checkpoint() should hold (per-job pause).
        self._paused_jobs: set[str] = set()
        # Jobs whose next checkpoint() should stop the task.
        self._cancel_requested: set[str] = set()
        # Global pause switch ("free the GPU"). Persisted in the meta table
        # so a service restart doesn't silently resume queued work.
        self._pause_all = False

    @asynccontextmanager
    async def _connect(self):
        """Open a connection with a generous busy timeout applied.

        Every code path goes through here so the busy timeout is never
        forgotten — the original bug was connections defaulting to a tiny
        timeout and failing instantly under concurrent ingestion.
        """
        async with aiosqlite.connect(
            self.db_path, timeout=_CONNECT_TIMEOUT
        ) as db:
            await db.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            yield db

    async def init(self) -> None:
        """Create the database file and schema if they don't exist, and mark
        any 'processing' or 'queued' jobs from the previous process as failed
        (their background tasks died when the service restarted)."""
        if self._ready:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._write_lock, self._connect() as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.executescript(_SCHEMA)
            # Migration: the steps column was added after the first release;
            # CREATE TABLE IF NOT EXISTS doesn't alter existing tables.
            cur = await db.execute("PRAGMA table_info(jobs)")
            cols = {row[1] for row in await cur.fetchall()}
            if "steps" not in cols:
                await db.execute(
                    "ALTER TABLE jobs ADD COLUMN steps TEXT NOT NULL DEFAULT '[]'"
                )
                logger.info("Migrated jobs table: added steps column")
            for col, ddl in (
                ("job_type", "job_type TEXT NOT NULL DEFAULT ''"),
                ("job_params", "job_params TEXT NOT NULL DEFAULT '{}'"),
                ("current_item", "current_item TEXT"),
            ):
                if col not in cols:
                    await db.execute(f"ALTER TABLE jobs ADD COLUMN {ddl}")
                    logger.info("Migrated jobs table: added %s column", col)
            # Mark in-flight jobs from the previous run as failed. Any
            # background task they were running was killed at shutdown;
            # without this cleanup the Ingest UI would show them stuck.
            # Their step ledgers get the same treatment: whichever step was
            # 'running' when the process died is marked as an error.
            cur = await db.execute(
                """SELECT job_id, steps FROM jobs
                   WHERE status IN ('processing', 'queued', 'paused')"""
            )
            stale = await cur.fetchall()
            for job_id, steps_json in stale:
                try:
                    steps = json.loads(steps_json or "[]")
                except json.JSONDecodeError:
                    steps = []
                for s in steps:
                    if s.get("status") == "running":
                        s["status"] = "error"
                        s["detail"] = "Service restarted while step was running"
                        s["finished_at"] = _utcnow_iso()
                await db.execute(
                    """UPDATE jobs
                       SET status = 'failed',
                           current_step = 'error',
                           error_message = coalesce(error_message,
                               'Service restarted while job was running'),
                           steps = ?,
                           updated_at = ?
                       WHERE job_id = ?""",
                    (json.dumps(steps), _utcnow_iso(), job_id),
                )
            # Prune logs of jobs that have aged out of the recent list.
            await db.execute(
                """DELETE FROM job_logs WHERE job_id NOT IN (
                       SELECT job_id FROM jobs
                       ORDER BY created_at DESC LIMIT ?
                   )""",
                (_LOG_RETAIN_JOBS,),
            )
            cur = await db.execute(
                "SELECT value FROM meta WHERE key = 'pause_all'"
            )
            row = await cur.fetchone()
            self._pause_all = bool(row and row[0] == "1")
            await db.commit()
            if stale:
                logger.info(
                    "Marked %d stale in-flight job(s) as failed", len(stale)
                )
        self._ready = True
        if self._pause_all:
            logger.info(
                "Job processing is PAUSED (pause-all was set before the last "
                "restart) — new jobs will hold until resume-all"
            )
        logger.info("JobManager initialized at %s", self.db_path)

    async def create(
        self,
        *,
        source_path: str,
        filename: str,
        categories: list[str],
        tags: list[str],
        job_type: str = "ingest",
        doc_id: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> Job:
        """Enqueue a new job. Returns the created Job.

        job_type + doc_id + params together describe how to re-launch the
        same work later (the Restart button), so pass everything the
        pipeline call needs to be reconstructed.
        """
        job_id = str(uuid.uuid4())
        now = _utcnow_iso()
        async with self._write_lock, self._connect() as db:
            await db.execute(
                """
                INSERT INTO jobs (
                    job_id, status, current_step, progress_pct,
                    created_at, updated_at,
                    source_path, filename,
                    requested_categories, requested_tags,
                    job_type, doc_id, job_params
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id, "queued", "pending", 0.0,
                    now, now,
                    source_path, filename,
                    json.dumps(categories), json.dumps(tags),
                    job_type, doc_id, json.dumps(params or {}),
                ),
            )
            await db.commit()
        return await self.get(job_id)  # type: ignore[return-value]

    async def get(self, job_id: str) -> Job | None:
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            )
            row = await cur.fetchone()
            return _row_to_job(row) if row else None

    async def list_recent(
        self, status: str | None = None, limit: int = 50
    ) -> list[Job]:
        """List jobs. `status` accepts a concrete status, or the pseudo-
        filters "active" (queued/processing/paused, running ones first) and
        "terminal" (completed/failed/cancelled, most recently finished
        first)."""
        params: tuple[Any, ...] = ()
        if status == "active":
            # created_at (not updated_at) within each status group: active
            # jobs update every few seconds, and sorting on that made rows
            # swap places under the user's cursor on every poll.
            query = (
                "SELECT * FROM jobs WHERE status IN "
                "('queued', 'processing', 'paused') "
                "ORDER BY CASE status WHEN 'processing' THEN 0 "
                "WHEN 'paused' THEN 1 ELSE 2 END, "
                "created_at ASC"
            )
        elif status == "terminal":
            query = (
                "SELECT * FROM jobs WHERE status IN "
                "('completed', 'failed', 'cancelled') "
                "ORDER BY updated_at DESC"
            )
        elif status is not None:
            query = "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC"
            params = (status,)
        else:
            query = "SELECT * FROM jobs ORDER BY created_at DESC"
        query += " LIMIT ?"
        params = (*params, limit)

        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(query, params)
            rows = await cur.fetchall()
            return [_row_to_job(r) for r in rows]

    async def status_counts(self) -> dict[str, int]:
        async with self._connect() as db:
            cur = await db.execute(
                "SELECT status, COUNT(*) FROM jobs GROUP BY status"
            )
            return {r[0]: r[1] for r in await cur.fetchall()}

    async def update(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        current_step: JobStep | None = None,
        progress_pct: float | None = None,
        error_message: str | None = None,
        doc_id: str | None = None,
        file_hash: str | None = None,
        pages_processed: int | None = None,
        pages_total: int | None = None,
        current_item: str | None = None,
    ) -> None:
        """Update mutable fields on a job. Only non-None args are applied."""
        sets = []
        params: list[Any] = []

        def _add(col: str, val: Any) -> None:
            if val is not None:
                sets.append(f"{col} = ?")
                params.append(val)

        _add("status", status)
        _add("current_step", current_step)
        _add("progress_pct", progress_pct)
        _add("error_message", error_message)
        _add("doc_id", doc_id)
        _add("file_hash", file_hash)
        _add("pages_processed", pages_processed)
        _add("pages_total", pages_total)
        _add("current_item", current_item)

        if not sets:
            return

        sets.append("updated_at = ?")
        params.append(_utcnow_iso())
        params.append(job_id)

        async with self._write_lock, self._connect() as db:
            await db.execute(
                f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ?", params
            )
            await self._drain_logs_locked(db)
            await db.commit()

    # ------------------------------------------------------------ job control

    def spawn(self, job_id: str, coro) -> asyncio.Task:
        """Launch a job coroutine as a tracked background task.

        Every pipeline job MUST be launched through here (not bare
        asyncio.create_task) so pause/stop can reach it. Also swallows the
        JobCancelled a stop raises — by then the job row is already
        finalized as 'cancelled'.
        """

        async def _runner() -> None:
            try:
                await coro
            except JobCancelled:
                logger.info("Job %s stopped by user", job_id)
            except asyncio.CancelledError:
                # Hard-cancelled while still queued (waiting on the ingest
                # semaphore, before any real work started) — safe to just
                # mark the row. May also fire at service shutdown; the
                # startup sweep is the backstop if this write doesn't land.
                try:
                    await self._finalize_cancel(job_id)
                except Exception:  # noqa: BLE001 — teardown path
                    pass

        task = asyncio.create_task(_runner())
        self._tasks[job_id] = task

        def _cleanup(_t: asyncio.Task, jid: str = job_id) -> None:
            self._tasks.pop(jid, None)
            self._paused_jobs.discard(jid)
            self._cancel_requested.discard(jid)

        task.add_done_callback(_cleanup)
        return task

    @property
    def pause_all_active(self) -> bool:
        return self._pause_all

    def is_pause_requested(self, job_id: str) -> bool:
        return self._pause_all or job_id in self._paused_jobs

    async def checkpoint(self, job_id: str) -> None:
        """Cooperative pause/stop gate, called by pipelines between units of
        work (a page, a batch, a document). Returns immediately in the
        normal case. Blocks while the job is paused (status flipped to
        'paused' and restored on resume). Raises JobCancelled — after
        finalizing the job row — when a stop was requested."""
        if job_id in self._cancel_requested:
            await self._finalize_cancel(job_id)
            raise JobCancelled(job_id)
        if not self.is_pause_requested(job_id):
            return

        job = await self.get(job_id)
        if job is None or job.status not in ACTIVE_STATUSES:
            return
        prev_status = "queued" if job.status == "queued" else "processing"
        await self.update(job_id, status="paused")
        logger.info("Job %s paused (was %s)", job_id, prev_status)
        while self.is_pause_requested(job_id):
            if job_id in self._cancel_requested:
                await self._finalize_cancel(job_id)
                raise JobCancelled(job_id)
            await asyncio.sleep(_PAUSE_POLL_SECONDS)
        await self.update(job_id, status=prev_status)
        logger.info("Job %s resumed", job_id)

    async def request_pause(self, job_id: str) -> bool:
        """Ask a job to hold at its next checkpoint. Returns False if the
        job doesn't exist or is already finished."""
        job = await self.get(job_id)
        if job is None or job.status not in ACTIVE_STATUSES:
            return False
        self._paused_jobs.add(job_id)
        return True

    async def request_resume(self, job_id: str) -> bool:
        """Clear a job's pause request. The paused checkpoint notices within
        _PAUSE_POLL_SECONDS. Returns False for unknown/finished jobs. Note:
        while pause-all is on, the job stays held — resume-all clears that."""
        job = await self.get(job_id)
        if job is None or job.status not in ACTIVE_STATUSES:
            return False
        self._paused_jobs.discard(job_id)
        return True

    async def request_cancel(self, job_id: str) -> bool:
        """Stop a job. Processing/paused jobs stop cooperatively at their
        next checkpoint (the current page/batch finishes first). Jobs still
        queued behind the semaphore are cancelled immediately — they haven't
        touched anything yet. Returns False if the job isn't active."""
        job = await self.get(job_id)
        if job is None or job.status not in ACTIVE_STATUSES:
            return False
        task = self._tasks.get(job_id)
        if task is None:
            # No live task for an active row (e.g. leftover from a crash the
            # startup sweep somehow missed) — just finalize the record.
            await self._finalize_cancel(job_id)
            return True
        if job.status == "queued":
            task.cancel()  # safe: still waiting for a slot, no work started
        else:
            self._cancel_requested.add(job_id)
        return True

    async def set_pause_all(self, paused: bool) -> None:
        """Global pause switch. Persisted so it survives restarts — 'I
        paused everything to use the GPU' must not silently un-pause."""
        self._pause_all = paused
        if not paused:
            self._paused_jobs.clear()
        async with self._write_lock, self._connect() as db:
            await db.execute(
                "INSERT INTO meta (key, value) VALUES ('pause_all', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                ("1" if paused else "0",),
            )
            await db.commit()
        logger.info("Pause-all %s", "ENABLED" if paused else "cleared")

    # ------------------------------------------------------------------- meta

    async def meta_get(self, key: str) -> str | None:
        """Read a value from the small key/value meta table (also used for
        the pause-all flag and the scheduler's persisted config/state)."""
        async with self._connect() as db:
            cur = await db.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            )
            row = await cur.fetchone()
            return row[0] if row else None

    async def meta_set(self, key: str, value: str) -> None:
        async with self._write_lock, self._connect() as db:
            await db.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            await db.commit()

    async def _finalize_cancel(self, job_id: str) -> None:
        """Write the terminal 'cancelled' state for a stopped job: flag any
        running step, keep current_step for context, clear the live item."""
        self._cancel_requested.discard(job_id)
        self._paused_jobs.discard(job_id)
        now = _utcnow_iso()
        async with self._write_lock, self._connect() as db:
            cur = await db.execute(
                "SELECT steps, status FROM jobs WHERE job_id = ?", (job_id,)
            )
            row = await cur.fetchone()
            if row is None:
                return
            if row[1] in ("completed", "failed", "cancelled"):
                return  # finished before the stop landed — keep the truth
            try:
                steps = json.loads(row[0] or "[]")
            except json.JSONDecodeError:
                steps = []
            for s in steps:
                if s.get("status") == "running":
                    s["status"] = "warning"
                    s["detail"] = (
                        "stopped by user — restart the job to continue; "
                        "it re-checks what is missing"
                    )
                    s["finished_at"] = now
            await db.execute(
                """UPDATE jobs SET status = 'cancelled', steps = ?,
                       current_item = NULL, updated_at = ?
                   WHERE job_id = ?""",
                (json.dumps(steps), now, job_id),
            )
            await self._drain_logs_locked(db)
            await db.commit()
        await self._prune_job_logs(job_id)
        logger.info("Job %s cancelled", job_id)

    # ------------------------------------------------------------ step ledger

    async def set_steps(self, job_id: str, step_names: list[str]) -> None:
        """Initialize the per-step ledger with all steps pending.

        Called once at the start of each pipeline run with the planned steps
        for that job type, so the UI can show what will (and won't) happen.
        """
        steps = [StepRecord(name=n).model_dump() for n in step_names]
        async with self._write_lock, self._connect() as db:
            await db.execute(
                "UPDATE jobs SET steps = ?, updated_at = ? WHERE job_id = ?",
                (json.dumps(steps), _utcnow_iso(), job_id),
            )
            await self._drain_logs_locked(db)
            await db.commit()

    async def update_step(
        self,
        job_id: str,
        name: str,
        status: StepStatus,
        detail: str | None = None,
    ) -> None:
        """Set one step's status in the ledger (read-modify-write under the
        write lock). A step not present in the plan is appended — that keeps
        the ledger honest if a pipeline runs a step it didn't announce."""
        now = _utcnow_iso()
        async with self._write_lock, self._connect() as db:
            cur = await db.execute(
                "SELECT steps FROM jobs WHERE job_id = ?", (job_id,)
            )
            row = await cur.fetchone()
            if row is None:
                return
            try:
                steps = json.loads(row[0] or "[]")
            except json.JSONDecodeError:
                steps = []

            entry = next((s for s in steps if s.get("name") == name), None)
            if entry is None:
                entry = StepRecord(name=name).model_dump()
                steps.append(entry)
            entry["status"] = status
            if detail is not None:
                entry["detail"] = detail
            if status == "running" and not entry.get("started_at"):
                entry["started_at"] = now
            if status in ("done", "warning", "skipped", "error"):
                entry["finished_at"] = now

            await db.execute(
                "UPDATE jobs SET steps = ?, updated_at = ? WHERE job_id = ?",
                (json.dumps(steps), now, job_id),
            )
            await self._drain_logs_locked(db)
            await db.commit()

    # -------------------------------------------------------------- job logs

    async def _drain_logs_locked(self, db: aiosqlite.Connection) -> None:
        """Persist buffered log lines. Caller must hold the write lock and
        commit afterwards. deque.popleft is thread-safe against the handler
        appending from worker threads."""
        rows: list[LogRow] = []
        while True:
            try:
                rows.append(self.log_buffer.popleft())
            except IndexError:
                break
        if rows:
            await db.executemany(
                """INSERT INTO job_logs (job_id, ts, level, logger, message)
                   VALUES (?, ?, ?, ?, ?)""",
                rows,
            )

    async def get_logs(
        self, job_id: str, limit: int = 1000
    ) -> list[dict[str, str]]:
        """Return the last `limit` captured log lines for a job, oldest
        first. Flushes the in-memory buffer first so the result is live."""
        async with self._write_lock, self._connect() as db:
            await self._drain_logs_locked(db)
            await db.commit()
            cur = await db.execute(
                """SELECT ts, level, logger, message FROM job_logs
                   WHERE job_id = ? ORDER BY id DESC LIMIT ?""",
                (job_id, limit),
            )
            rows = await cur.fetchall()
        return [
            {"ts": r[0], "level": r[1], "logger": r[2], "message": r[3]}
            for r in reversed(rows)
        ]

    async def _prune_job_logs(self, job_id: str) -> None:
        """Cap a finished job's stored log lines at _LOG_RETAIN_LINES."""
        async with self._write_lock, self._connect() as db:
            await db.execute(
                """DELETE FROM job_logs WHERE job_id = ? AND id NOT IN (
                       SELECT id FROM job_logs WHERE job_id = ?
                       ORDER BY id DESC LIMIT ?
                   )""",
                (job_id, job_id, _LOG_RETAIN_LINES),
            )
            await db.commit()

    async def fail(self, job_id: str, error_message: str) -> None:
        await self.update(
            job_id,
            status="failed",
            current_step="error",
            error_message=error_message,
        )
        await self._prune_job_logs(job_id)

    async def complete(self, job_id: str) -> None:
        await self.update(
            job_id,
            status="completed",
            current_step="done",
            progress_pct=100.0,
            current_item="",
        )
        await self._prune_job_logs(job_id)


def _parse_steps(raw: str | None) -> list[StepRecord]:
    try:
        return [StepRecord(**s) for s in json.loads(raw or "[]")]
    except Exception:  # noqa: BLE001 — a corrupt ledger shouldn't hide the job
        return []


def _row_to_job(row: aiosqlite.Row) -> Job:
    return Job(
        job_id=row["job_id"],
        status=row["status"],
        current_step=row["current_step"],
        progress_pct=row["progress_pct"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        error_message=row["error_message"],
        source_path=row["source_path"],
        filename=row["filename"],
        requested_categories=json.loads(row["requested_categories"]),
        requested_tags=json.loads(row["requested_tags"]),
        doc_id=row["doc_id"],
        file_hash=row["file_hash"],
        pages_processed=row["pages_processed"],
        pages_total=row["pages_total"],
        steps=_parse_steps(row["steps"]),
        job_type=row["job_type"] or "",
        job_params=_parse_params(row["job_params"]),
        current_item=row["current_item"] or None,
    )


def _parse_params(raw: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}
