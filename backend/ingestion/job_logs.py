"""Per-job log capture.

A logging.Handler attached to the root logger snapshots every INFO+ record
emitted while a job's contextvar is set. Records land in a thread-safe deque
and are persisted to SQLite (job_logs table) by JobManager, which drains the
buffer on every job update and on log reads — so GET /ingest/jobs/{id}/logs
always reflects the latest activity.

The contextvar is set at the top of each pipeline run method. It propagates
into asyncio.to_thread workers (contextvars are copied into the executor),
so logs from CPU-bound helper threads (Docling, rasterization, embedding)
are captured too.
"""

from __future__ import annotations

import logging
import traceback
from collections import deque
from contextvars import ContextVar
from datetime import datetime, timezone

current_job_id: ContextVar[str | None] = ContextVar(
    "forge_current_job_id", default=None
)

# Global cap on unflushed lines. If drains fall far behind (no job updates
# during a very long single step), the oldest lines drop first — better than
# unbounded memory growth.
MAX_BUFFERED_LINES = 10_000
_MAX_MSG_CHARS = 4000

# (job_id, ts_iso, level, logger_name, message)
LogRow = tuple[str, str, str, str, str]


def make_log_buffer() -> deque[LogRow]:
    return deque(maxlen=MAX_BUFFERED_LINES)


class JobLogHandler(logging.Handler):
    """Buffers log records for whichever job is active in the current context."""

    def __init__(self, buffer: deque[LogRow]):
        super().__init__(level=logging.INFO)
        self.buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        job_id = current_job_id.get()
        if job_id is None:
            return
        try:
            msg = record.getMessage()
            if record.exc_info and record.exc_info[0] is not None:
                msg += "\n" + "".join(
                    traceback.format_exception(*record.exc_info)
                )
            self.buffer.append((
                job_id,
                datetime.now(timezone.utc).isoformat(),
                record.levelname,
                record.name,
                msg[:_MAX_MSG_CHARS],
            ))
        except Exception:  # noqa: BLE001 — log capture must never break the pipeline
            pass


def install_job_log_handler(buffer: deque[LogRow]) -> JobLogHandler:
    """Attach the capture handler to the root logger. Idempotent."""
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, JobLogHandler):
            return h
    handler = JobLogHandler(buffer)
    root.addHandler(handler)
    return handler
