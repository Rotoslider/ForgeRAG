"""SQLite-backed ingestion job queue.

A lightweight alternative to Redis/Celery for a single-server batch workload.
Uses aiosqlite for async access, with WAL mode for concurrent read/write.

Jobs are long-lived records — not deleted after completion, so users can
inspect history via GET /ingest/jobs.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from backend.models.ingestion import Job, JobStatus, JobStep

logger = logging.getLogger(__name__)


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
    pages_total INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status);
CREATE INDEX IF NOT EXISTS jobs_created_idx ON jobs(created_at);
"""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    """Async SQLite-backed job store. One instance per process."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._ready = False

    async def init(self) -> None:
        """Create the database file and schema if they don't exist."""
        if self._ready:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.executescript(_SCHEMA)
            await db.commit()
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
        async with aiosqlite.connect(self.db_path) as db:
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
        async with aiosqlite.connect(self.db_path) as db:
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

        async with aiosqlite.connect(self.db_path) as db:
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

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ?", params
            )
            await db.commit()

    async def fail(self, job_id: str, error_message: str) -> None:
        await self.update(
            job_id,
            status="failed",
            current_step="error",
            error_message=error_message,
        )

    async def complete(self, job_id: str) -> None:
        await self.update(
            job_id,
            status="completed",
            current_step="done",
            progress_pct=100.0,
        )


def _row_to_job(row: aiosqlite.Row) -> Job:
    import json

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
    )
