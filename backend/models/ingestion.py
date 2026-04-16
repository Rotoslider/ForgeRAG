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
    "embedding_text",    # Phase 3
    "embedding_visual",  # Phase 3 (ColPali)
    "extracting_entities",  # Phase 4
    "building_graph",    # Phase 4
    "done",
    "error",
]


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
    requested_categories: list[str] = Field(default_factory=list)
    requested_tags: list[str] = Field(default_factory=list)

    # What we produced
    doc_id: str | None = None       # set once Document node is created
    file_hash: str | None = None    # set once hash is computed
    pages_processed: int = 0
    pages_total: int = 0


class IngestResponse(BaseModel):
    """Response body for POST /ingest."""

    job_id: str
    status: JobStatus
    message: str


class JobListFilter(BaseModel):
    status: JobStatus | None = None
    limit: int = Field(default=50, ge=1, le=500)
