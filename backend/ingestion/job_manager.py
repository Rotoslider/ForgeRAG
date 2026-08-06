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
    steps TEXT NOT NULL DEFAULT '[]'                  -- JSON array of StepRecord
);

CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status);
CREATE INDEX IF NOT EXISTS jobs_created_idx ON jobs(created_at);

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
            # Mark in-flight jobs from the previous run as failed. Any
            # background task they were running was killed at shutdown;
            # without this cleanup the Ingest UI would show them stuck.
            # Their step ledgers get the same treatment: whichever step was
            # 'running' when the process died is marked as an error.
            cur = await db.execute(
                """SELECT job_id, steps FROM jobs
                   WHERE status IN ('processing', 'queued')"""
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
            await db.commit()
            if stale:
                logger.info(
                    "Marked %d stale in-flight job(s) as failed", len(stale)
                )
        self._ready = True
        logger.info("JobManager initialized at %s", self.db_path)

    async def create(
        self,
        *,
        source_path: str,
        filename: str,
        categories: list[str],
        tags: list[str],
    ) -> Job:
        """Enqueue a new job. Returns the created Job."""
        import json

        job_id = str(uuid.uuid4())
        now = _utcnow_iso()
        async with self._write_lock, self._connect() as db:
            await db.execute(
                """
                INSERT INTO jobs (
                    job_id, status, current_step, progress_pct,
                    created_at, updated_at,
                    source_path, filename,
                    requested_categories, requested_tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id, "queued", "pending", 0.0,
                    now, now,
                    source_path, filename,
                    json.dumps(categories), json.dumps(tags),
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
        self, status: JobStatus | None = None, limit: int = 50
    ) -> list[Job]:
        query = "SELECT * FROM jobs"
        params: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY created_at DESC LIMIT ?"
        params = (*params, limit)

        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(query, params)
            rows = await cur.fetchall()
            return [_row_to_job(r) for r in rows]

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
    )
