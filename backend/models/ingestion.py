"""Pydantic models for ingestion jobs."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


JobStatus = Literal["queued", "processing", "completed", "failed", "cancelled"]

JobStep = Literal[
    "pending",
    "registering",       # compute hash, check dedup, create :Document
    "rendering_pages",   # PDF -> PNGs
    "extracting_text",   # PyMuPDF text + scanned detection
    "auto_tagging",      # LLM collection/category/tag suggestion
    "embedding_text",    # Phase 3
    "embedding_visual",  # Phase 3 (ColPali)
    "extracting_entities",  # Phase 4
    "building_graph",    # Phase 4
    # Phase 9 chunk pipeline
    "building_chunks",   # umbrella step when fired during run_job
    "chunking",          # Docling structural parse
    "summarizing",       # per-chunk LLM summaries
    "embedding_chunks",  # BGE-M3 embeddings over (summary + text)
    "writing_chunks",    # MERGE Chunk nodes + HAS_CHUNK edges
    "dedup_entities",    # post-ingestion near-duplicate entity merge
    "done",
    "error",
]

# Per-step ledger statuses. "warning" means the step ran to completion but
# some units of work inside it failed (e.g. 12 of 900 page extractions);
# "skipped" means the pipeline deliberately did not run it (service not
# wired, manual tags provided, ...) — the detail field says why.
StepStatus = Literal["pending", "running", "done", "warning", "skipped", "error"]


class StepRecord(BaseModel):
    """One entry in a job's per-step status ledger."""

    name: str
    status: StepStatus = "pending"
    detail: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class Job(BaseModel):
    """Ingestion job record."""

    job_id: str
    status: JobStatus = "queued"
    current_step: JobStep = "pending"
    progress_pct: float = 0.0
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None

    # What we're ingesting
    source_path: str               # original uploaded PDF path (temp or workspace)
    filename: str
    requested_collection: str = "default"
    requested_categories: list[str] = Field(default_factory=list)
    requested_tags: list[str] = Field(default_factory=list)

    # What we produced
    doc_id: str | None = None       # set once Document node is created
    file_hash: str | None = None    # set once hash is computed
    pages_processed: int = 0
    pages_total: int = 0

    # Per-step ledger. Populated at job start with the planned steps, then
    # updated as each step runs/finishes/skips/fails. Old jobs (before the
    # ledger existed) have an empty list.
    steps: list[StepRecord] = Field(default_factory=list)


class IngestResponse(BaseModel):
    """Response body for POST /ingest."""

    job_id: str
    status: JobStatus
    message: str


class JobListFilter(BaseModel):
    status: JobStatus | None = None
    limit: int = Field(default=50, ge=1, le=500)
