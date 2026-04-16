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
from backend.ingestion.community_detector import CommunityDetector
from backend.ingestion.entity_extractor import EntityExtractor
from backend.ingestion.graph_builder import GraphBuilder
from backend.ingestion.job_manager import JobManager
from backend.ingestion.pdf_processor import PDFProcessor
from backend.ingestion.text_extractor import TextExtractor
from backend.services.colpali_service import ColPaliService, serialize_colpali
from backend.services.gpu_manager import GPUManager
from backend.services.llm_service import LLMService
from backend.services.neo4j_service import Neo4jService
from backend.services.text_embedding_service import TextEmbeddingService

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
        gpu: GPUManager | None = None,
        text_embedding: TextEmbeddingService | None = None,
        colpali: ColPaliService | None = None,
        llm: LLMService | None = None,
    ):
        self.settings = settings
        self.neo4j = neo4j
        self.jobs = job_manager
        self.gpu = gpu
        self.text_embedding = text_embedding
        self.colpali = colpali
        self.llm = llm
        self.entity_extractor = EntityExtractor(llm) if llm is not None else None
        self.graph_builder = GraphBuilder(neo4j)
        self.community_detector = (
            CommunityDetector(neo4j=neo4j, llm=llm, text_embedding=text_embedding)
            if (llm is not None and text_embedding is not None)
            else None
        )
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
                progress_pct=5.0,
                doc_id=doc_id,
                file_hash=file_hash,
                pages_total=page_count,
            )
            await self._rasterize(job_id, job.source_path, file_hash, page_count)

            await self.jobs.update(
                job_id, current_step="extracting_text", progress_pct=40.0
            )
            await self._extract_text(job_id, job.source_path, doc_id, file_hash)

            # Phase 3 steps — only run if services are wired up
            if self.text_embedding is not None:
                await self.jobs.update(
                    job_id, current_step="embedding_text", progress_pct=60.0
                )
                await self._embed_text(job_id, doc_id)

            if self.colpali is not None:
                await self.jobs.update(
                    job_id, current_step="embedding_visual", progress_pct=75.0
                )
                await self._embed_visual(job_id, doc_id, file_hash)

            if self.entity_extractor is not None:
                await self.jobs.update(
                    job_id, current_step="extracting_entities", progress_pct=88.0
                )
                await self._extract_entities(job_id, doc_id)

            await self.jobs.complete(job_id)
            logger.info("Job %s completed successfully", job_id)

        except Exception as exc:  # noqa: BLE001
            logger.exception("Job %s failed", job_id)
            await self.jobs.fail(job_id, str(exc))

    async def run_communities_only(self, job_id: str) -> None:
        """Rebuild all :Community nodes globally from the current graph.

        Not per-document — community detection spans all ingested pages
        because engineering topics connect across handbooks. The job is
        scoped to tracking the long operation, not to a specific doc.
        """
        try:
            if self.community_detector is None:
                raise ValueError("LLM or text embedding unavailable — cannot detect communities")
            await self.jobs.update(
                job_id, status="processing", current_step="building_graph", progress_pct=5.0
            )
            assert self.gpu is not None
            async with self.gpu.load_scope("text_embedding"):
                counts = await self.community_detector.build()
            logger.info("Community detection complete: %s", counts)
            await self.jobs.complete(job_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Community job %s failed", job_id)
            await self.jobs.fail(job_id, str(exc))

    async def run_extraction_only(self, job_id: str, doc_id: str) -> None:
        """Re-run only entity extraction for an already-ingested document.

        Used by POST /documents/{doc_id}/extract-entities.
        """
        try:
            await self.jobs.update(job_id, status="processing", doc_id=doc_id)
            if self.entity_extractor is None:
                raise ValueError("LLM service not configured — cannot extract entities")
            await self.jobs.update(
                job_id, current_step="extracting_entities", progress_pct=10.0
            )
            await self._extract_entities(job_id, doc_id)
            await self.jobs.complete(job_id)
            logger.info("Extraction-only job %s completed for doc %s", job_id, doc_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Extraction-only job %s failed", job_id)
            await self.jobs.fail(job_id, str(exc))

    async def run_embeddings_only(self, job_id: str, doc_id: str) -> None:
        """Re-run only the embedding steps for an already-ingested document.

        Used by POST /documents/{doc_id}/reembed to backfill Phase 3 embeddings
        on documents ingested during Phase 2.
        """
        try:
            # Look up file_hash from Neo4j
            rows = await self.neo4j.run_query(
                "MATCH (d:Document {doc_id: $id}) RETURN d.file_hash AS h",
                {"id": doc_id},
            )
            if not rows:
                raise ValueError(f"Document {doc_id} not found")
            file_hash = rows[0]["h"]

            await self.jobs.update(job_id, status="processing", doc_id=doc_id, file_hash=file_hash)

            if self.text_embedding is not None:
                await self.jobs.update(
                    job_id, current_step="embedding_text", progress_pct=10.0
                )
                await self._embed_text(job_id, doc_id)

            if self.colpali is not None:
                await self.jobs.update(
                    job_id, current_step="embedding_visual", progress_pct=50.0
                )
                await self._embed_visual(job_id, doc_id, file_hash)

            await self.jobs.complete(job_id)
            logger.info("Reembed job %s completed for doc %s", job_id, doc_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Reembed job %s failed", job_id)
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
        """Extract text per page and create :Page nodes linked to :Document.

        Skips extraction entirely if the document already has :Page nodes
        (the previous ingestion run already created them). Avoids creating
        duplicate Pages on resume after a failed ColPali / entity-extraction
        step.
        """
        existing = await self.neo4j.run_query(
            """
            MATCH (d:Document {doc_id: $doc_id})-[:HAS_PAGE]->(p:Page)
            RETURN count(p) AS n
            """,
            {"doc_id": doc_id},
        )
        existing_count = existing[0]["n"] if existing else 0
        if existing_count > 0:
            logger.info(
                "Document %s already has %d :Page nodes — skipping text extraction",
                doc_id, existing_count,
            )
            await self.jobs.update(
                job_id, progress_pct=55.0, pages_processed=existing_count
            )
            return

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
            progress_pct=55.0,
            pages_processed=extraction.page_count,
        )
        logger.info(
            "Created %d :Page nodes for doc %s (source=%s)",
            extraction.page_count, doc_id, extraction.document_source_type,
        )

    # ------------------------------------------------------------------ step 4

    async def _embed_text(self, job_id: str, doc_id: str) -> None:
        """Embed page texts and store on Page.text_embedding (Neo4j vector index)."""
        assert self.text_embedding is not None
        assert self.gpu is not None

        # Pull pages that have text and no embedding yet
        pages = await self.neo4j.run_query(
            """
            MATCH (d:Document {doc_id: $doc_id})-[:HAS_PAGE]->(p:Page)
            WHERE p.text_char_count > 0 AND p.text_embedding IS NULL
            RETURN p.page_id AS page_id, p.extracted_text AS text
            ORDER BY p.page_number
            """,
            {"doc_id": doc_id},
        )
        if not pages:
            logger.info("No pages need text embedding for doc %s", doc_id)
            return

        total = len(pages)
        batch_size = self.settings.ingestion.text_embedding_batch_size

        # Embed in batches under the GPU semaphore
        async with self.gpu.load_scope("text_embedding"):
            for start in range(0, total, batch_size):
                batch = pages[start:start + batch_size]
                texts = [row["text"] for row in batch]
                ids = [row["page_id"] for row in batch]

                # Embedding is CPU/GPU-bound — run in a worker thread
                embeddings = await asyncio.to_thread(
                    self.text_embedding.embed_documents, texts, batch_size=batch_size
                )

                # Write back in one UNWIND query
                payload = [
                    {"page_id": pid, "vec": emb.tolist()}
                    for pid, emb in zip(ids, embeddings, strict=True)
                ]
                await self.neo4j.run_write(
                    """
                    UNWIND $rows AS row
                    MATCH (p:Page {page_id: row.page_id})
                    SET p.text_embedding = row.vec
                    """,
                    {"rows": payload},
                )

                done = min(start + batch_size, total)
                # Text embedding spans 60% -> 75% in full runs, 10% -> 50% in reembed runs
                await self.jobs.update(
                    job_id, pages_processed=done
                )

        logger.info("Embedded text for %d pages of doc %s", total, doc_id)

    # ------------------------------------------------------------------ step 5

    async def _embed_visual(self, job_id: str, doc_id: str, file_hash: str) -> None:
        """Generate ColPali embeddings for every page and store as bytes on Page."""
        assert self.colpali is not None
        assert self.gpu is not None

        # Find all pages that don't yet have a ColPali embedding
        rows = await self.neo4j.run_query(
            """
            MATCH (d:Document {doc_id: $doc_id})-[:HAS_PAGE]->(p:Page)
            WHERE p.colpali_vector_count IS NULL OR p.colpali_vector_count = 0
            RETURN p.page_id AS page_id, p.page_number AS page_number
            ORDER BY p.page_number
            """,
            {"doc_id": doc_id},
        )
        if not rows:
            logger.info("No pages need ColPali embedding for doc %s", doc_id)
            return

        total = len(rows)
        batch_size = self.settings.ingestion.colpali_batch_size

        async with self.gpu.load_scope("colpali"):
            for start in range(0, total, batch_size):
                batch = rows[start:start + batch_size]
                image_paths = [
                    self.pdf_processor.page_image_path(file_hash, r["page_number"])
                    for r in batch
                ]
                page_ids = [r["page_id"] for r in batch]

                # ColPali returns a list of (K, D) float32 arrays
                embeddings = await asyncio.to_thread(
                    self.colpali.embed_images, image_paths
                )

                payload = []
                for pid, arr in zip(page_ids, embeddings, strict=True):
                    blob, k = serialize_colpali(arr)
                    payload.append(
                        {"page_id": pid, "blob": blob, "count": k, "dim": int(arr.shape[1]) if arr.size else 128}
                    )

                await self.neo4j.run_write(
                    """
                    UNWIND $rows AS row
                    MATCH (p:Page {page_id: row.page_id})
                    SET p.colpali_vectors = row.blob,
                        p.colpali_vector_count = row.count,
                        p.colpali_vector_dim = row.dim
                    """,
                    {"rows": payload},
                )

                done = min(start + batch_size, total)
                await self.jobs.update(job_id, pages_processed=done)

        logger.info("Embedded ColPali for %d pages of doc %s", total, doc_id)

    # ------------------------------------------------------------------ step 6

    async def _extract_entities(self, job_id: str, doc_id: str) -> None:
        """Run LLM entity extraction on each page and write results into the graph.

        I/O-bound on the LLM endpoint. Sequential per page (local LLM
        serves one request at a time). Skips pages that already have any
        MENTIONS_* outgoing relationship so re-runs after a partial failure
        don't double-count support_count on existing edges.
        """
        assert self.entity_extractor is not None

        # Pull title + pages that have text AND haven't had entities extracted yet.
        # We detect "already extracted" as having any of the page-level entity
        # relationships — if extraction ran on this page it wrote at least one,
        # unless the page was empty of entities. For safety we also accept pages
        # that still genuinely have nothing relevant; those pages just get
        # re-run (fast path since the LLM returns empty arrays).
        rows = await self.neo4j.run_query(
            """
            MATCH (d:Document {doc_id: $doc_id})
            OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Page)
            WHERE p.text_char_count > 0
              AND NOT EXISTS {
                (p)-[:MENTIONS_MATERIAL|DESCRIBES_PROCESS|REFERENCES_STANDARD|MENTIONS_EQUIPMENT]->()
              }
            RETURN d.title AS title,
                   collect({page_id: p.page_id, page_number: p.page_number, text: p.extracted_text}) AS pages
            """,
            {"doc_id": doc_id},
        )
        if not rows:
            logger.warning("No document %s found for entity extraction", doc_id)
            return
        title = rows[0]["title"] or "(untitled)"
        pages = [p for p in rows[0]["pages"] if p["page_id"] is not None]

        if not pages:
            logger.info("Document %s has no pages with text — skipping extraction", doc_id)
            return

        total = len(pages)
        logger.info(
            "Extracting entities for %d pages of %s via LLM %s "
            "(skipping any already extracted)",
            total, title, self.llm.settings.endpoint if self.llm else "?",
        )
        # Reflect the actual work queue in pages_total so the UI progress
        # counter is 'N/queue_size' not 'N/document_size'. Pages already
        # extracted in a prior run are outside this queue.
        await self.jobs.update(job_id, pages_total=total)

        done = 0
        aggregate = {"materials": 0, "processes": 0, "standards": 0,
                     "clauses": 0, "equipment": 0,
                     "page_rels": 0, "entity_rels": 0}
        for page in pages:
            try:
                extraction = await self.entity_extractor.extract_page(
                    document_title=title,
                    page_number=page["page_number"],
                    page_text=page["text"],
                )
                counts = await self.graph_builder.write_page(
                    page_id=page["page_id"], extraction=extraction
                )
                for k, v in counts.items():
                    aggregate[k] = aggregate.get(k, 0) + v
            except Exception as exc:  # noqa: BLE001
                logger.warning("Entity extraction failed for page %d: %s",
                               page["page_number"], exc)

            done += 1
            # Extraction spans 88% -> 99% in full runs, 10% -> 95% in extract-only
            progress = min(99.0, 88.0 + 11.0 * done / total)
            await self.jobs.update(
                job_id, progress_pct=progress, pages_processed=done
            )

        logger.info("Entity extraction complete for doc %s: %s", doc_id, aggregate)
