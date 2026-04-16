"""Ingestion pipeline orchestrator.

Runs the Phase 2 steps for a single PDF:
  1. Register — compute SHA-256, check for dedup, create :Document node with metadata
  2. Rasterize — PDF -> PNGs + reduced JPGs on disk
  3. Extract text — PyMuPDF for digital-native; mark scanned pages for Phase 3 OCR

Creates :Document and :Page nodes in Neo4j with HAS_PAGE relationships, plus
IN_CATEGORY and TAGGED_WITH relationships for organization. Updates the job
record throughout so the user can poll /ingest/jobs/{id} for progress.

Runs as an asyncio background task — submit one job at a time to keep GPU
(later phases) and disk I/O predictable.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from backend.config import Settings
from backend.ingestion.job_manager import JobManager
from backend.ingestion.pdf_processor import PDFProcessor
from backend.ingestion.text_extractor import TextExtractor
from backend.services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)


async def _sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file, reading in 1 MB chunks. Off-main-thread via asyncio.to_thread."""
    def _hash() -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        return h.hexdigest()
    return await asyncio.to_thread(_hash)


class IngestionPipeline:
    """Orchestrates PDF ingestion. One instance per app, shared across jobs."""

    def __init__(
        self,
        *,
        settings: Settings,
        neo4j: Neo4jService,
        job_manager: JobManager,
    ):
        self.settings = settings
        self.neo4j = neo4j
        self.jobs = job_manager
        self.pdf_processor = PDFProcessor(
            data_dir=Path(settings.server.data_dir),
            dpi=settings.ingestion.pdf_dpi,
            reduction_pct=settings.ingestion.reduction_percentage,
            reduction_min_dimension=settings.ingestion.reduction_min_dimension,
        )
        self.text_extractor = TextExtractor(
            scanned_text_threshold_chars=settings.ingestion.scanned_text_threshold_chars,
        )

    async def run_job(self, job_id: str) -> None:
        """Run the full pipeline for a queued job. Catches and records errors."""
        try:
            job = await self.jobs.get(job_id)
            if job is None:
                logger.error("Job %s not found", job_id)
                return

            await self.jobs.update(job_id, status="processing", current_step="registering")
            doc_id, file_hash, page_count = await self._register(job)

            await self.jobs.update(
                job_id,
                current_step="rendering_pages",
                progress_pct=10.0,
                doc_id=doc_id,
                file_hash=file_hash,
                pages_total=page_count,
            )
            await self._rasterize(job_id, job.source_path, file_hash, page_count)

            await self.jobs.update(
                job_id, current_step="extracting_text", progress_pct=60.0
            )
            await self._extract_text(job_id, job.source_path, doc_id, file_hash)

            await self.jobs.complete(job_id)
            logger.info("Job %s completed successfully", job_id)

        except Exception as exc:  # noqa: BLE001
            logger.exception("Job %s failed", job_id)
            await self.jobs.fail(job_id, str(exc))

    # ------------------------------------------------------------------ step 1

    async def _register(self, job) -> tuple[str, str, int]:
        """Compute hash, dedup, create :Document node + category/tag relationships.

        Returns (doc_id, file_hash, page_count).
        """
        source_path = Path(job.source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source PDF not found: {source_path}")

        # Hash for dedup
        file_hash = await _sha256_file(source_path)

        # Quick page count from PyMuPDF (fast — doesn't rasterize)
        import fitz
        with fitz.open(str(source_path)) as doc:
            page_count = doc.page_count

        size_bytes = source_path.stat().st_size

        # Check for existing document with this hash. If present, reuse doc_id.
        existing = await self.neo4j.run_query(
            "MATCH (d:Document {file_hash: $h}) RETURN d.doc_id AS doc_id LIMIT 1",
            {"h": file_hash},
        )
        if existing:
            doc_id = existing[0]["doc_id"]
            logger.info("Document %s already registered (hash=%s...)", doc_id, file_hash[:12])
            # Still apply any new categories/tags below.
        else:
            doc_id = str(uuid.uuid4())
            # Derive title from the *original* filename (not the staged path,
            # which has a UUID prefix from the upload handler).
            title = Path(job.filename).stem
            now_iso = datetime.now(timezone.utc).isoformat()
            await self.neo4j.run_write(
                """
                CREATE (d:Document {
                    doc_id: $doc_id,
                    title: $title,
                    filename: $filename,
                    file_hash: $file_hash,
                    page_count: $page_count,
                    file_size_bytes: $file_size,
                    ingested_at: datetime($ingested_at),
                    source_type: 'unknown'
                })
                """,
                {
                    "doc_id": doc_id,
                    "title": title,
                    "filename": job.filename,
                    "file_hash": file_hash,
                    "page_count": page_count,
                    "file_size": size_bytes,
                    "ingested_at": now_iso,
                },
            )
            logger.info("Created :Document %s for %s (%d pages)", doc_id, job.filename, page_count)

        # Attach categories (MERGE to create category nodes if missing)
        for cat in job.requested_categories:
            await self.neo4j.run_write(
                """
                MERGE (c:Category {name: $name})
                WITH c
                MATCH (d:Document {doc_id: $doc_id})
                MERGE (d)-[:IN_CATEGORY]->(c)
                """,
                {"name": cat, "doc_id": doc_id},
            )

        # Attach tags
        for tag in job.requested_tags:
            await self.neo4j.run_write(
                """
                MERGE (t:Tag {name: $name})
                WITH t
                MATCH (d:Document {doc_id: $doc_id})
                MERGE (d)-[:TAGGED_WITH]->(t)
                """,
                {"name": tag, "doc_id": doc_id},
            )

        return doc_id, file_hash, page_count

    # ------------------------------------------------------------------ step 2

    async def _rasterize(
        self, job_id: str, source_path: str, file_hash: str, page_count: int
    ) -> None:
        """Convert PDF to per-page PNGs + reduced JPGs."""
        loop = asyncio.get_running_loop()

        def _progress(done: int, total: int) -> None:
            # schedule an async update on the main loop without blocking this thread
            try:
                pct = 10.0 + 50.0 * (done / max(total, 1))  # rasterize spans 10% -> 60%
                fut = asyncio.run_coroutine_threadsafe(
                    self.jobs.update(
                        job_id, progress_pct=pct, pages_processed=done
                    ),
                    loop,
                )
                # Wait briefly so updates don't pile up; don't block on errors
                fut.result(timeout=5)
            except Exception as exc:  # noqa: BLE001
                logger.debug("progress update failed: %s", exc)

        await asyncio.to_thread(
            self.pdf_processor.convert_pdf_sync,
            Path(source_path),
            file_hash,
            progress_cb=_progress,
        )
        logger.info("Rasterized %d pages for hash=%s", page_count, file_hash[:12])

    # ------------------------------------------------------------------ step 3

    async def _extract_text(
        self, job_id: str, source_path: str, doc_id: str, file_hash: str
    ) -> None:
        """Extract text per page and create :Page nodes linked to :Document."""
        extraction = await asyncio.to_thread(
            self.text_extractor.extract_sync, Path(source_path)
        )

        # Update Document.source_type from aggregate classification
        await self.neo4j.run_write(
            "MATCH (d:Document {doc_id: $doc_id}) SET d.source_type = $st",
            {"doc_id": doc_id, "st": extraction.document_source_type},
        )

        # Batch create :Page nodes in a single transaction per document.
        # Neo4j UNWIND makes this efficient for large documents.
        now_iso = datetime.now(timezone.utc).isoformat()
        pages_params = []
        for p in extraction.pages:
            page_id = str(uuid.uuid4())
            image_path = str(
                self.pdf_processor.page_image_path(file_hash, p.page_number)
            )
            reduced_path = str(
                self.pdf_processor.reduced_image_path(file_hash, p.page_number)
            )
            pages_params.append({
                "page_id": page_id,
                "page_number": p.page_number,
                "image_path": image_path,
                "reduced_image_path": reduced_path,
                "extracted_text": p.text,
                "text_char_count": p.char_count,
                "source_type": p.source_type,
            })

        if pages_params:
            await self.neo4j.run_write(
                """
                MATCH (d:Document {doc_id: $doc_id})
                UNWIND $pages AS page
                CREATE (p:Page {
                    page_id: page.page_id,
                    page_number: page.page_number,
                    image_path: page.image_path,
                    reduced_image_path: page.reduced_image_path,
                    extracted_text: page.extracted_text,
                    text_char_count: page.text_char_count,
                    source_type: page.source_type
                })
                CREATE (d)-[:HAS_PAGE {page_number: page.page_number}]->(p)
                """,
                {"doc_id": doc_id, "pages": pages_params, "now": now_iso},
            )

        await self.jobs.update(
            job_id,
            progress_pct=95.0,
            pages_processed=extraction.page_count,
        )
        logger.info(
            "Created %d :Page nodes for doc %s (source=%s)",
            extraction.page_count, doc_id, extraction.document_source_type,
        )
