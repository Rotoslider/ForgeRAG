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
import os
import tempfile
import uuid
from collections import Counter as _Counter
from datetime import datetime, timezone
from pathlib import Path

from backend.config import Settings
from backend.ingestion.auto_tagger import AutoTagger
from backend.ingestion.chunk_summarizer import ChunkSummarizer
from backend.ingestion.chunker import StructuralChunker, StructuralChunk
from backend.ingestion.community_detector import CommunityDetector
from backend.ingestion.entity_extractor import EntityExtractor
from backend.ingestion.graph_builder import GraphBuilder
from backend.ingestion.job_logs import current_job_id
from backend.ingestion.job_manager import JobManager
from backend.ingestion.pdf_processor import PDFProcessor
from backend.ingestion.text_extractor import TextExtractor
from backend.services.colpali_service import ColPaliService, serialize_colpali
from backend.services.entity_merge import merge_entity
from backend.services.work_predicates import (
    ENTITY_NEEDS_EXTRACTION,
    SUSPICIOUS_EMPTY_MIN_CHARS,
    TEXT_EMBED_MISSING,
    VISUAL_EMBED_MISSING,
)
from backend.services.nemotron_service import NemotronService, serialize_nemotron
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


def _is_blank_page(image_path: Path, text_char_count: int) -> bool:
    """True if a page appears visually blank with no meaningful text.

    Pages with more than a trivial amount of extracted text are never blank.
    For short-text pages, we load the (small) reduced JPG and check grayscale
    std-dev — a uniformly white page has stddev near 0. Threshold 5.0 accepts
    page-number footers but rejects truly empty pages.

    Returns False on any load error — we'd rather embed a page than silently
    lose content to a misbehaving image file.
    """
    if text_char_count > 20:
        return False
    try:
        from PIL import Image, ImageStat
        with Image.open(image_path) as img:
            stat = ImageStat.Stat(img.convert("L"))
            stddev = stat.stddev[0]
        return stddev < 5.0
    except Exception as exc:  # noqa: BLE001
        logger.debug("blank-page check failed for %s: %s", image_path, exc)
        return False


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
        self.auto_tagger = AutoTagger(llm) if llm is not None else None
        self.chunk_summarizer = ChunkSummarizer(llm) if llm is not None else None
        # Docling initialization is expensive; keep it lazy via the chunker
        # class's own lazy-load.
        self.structural_chunker = StructuralChunker()
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
        # Cap how many full ingestion jobs run concurrently. Every upload
        # spawns its own background task; without this gate, adding dozens of
        # PDFs at once floods the job DB (the bug that caused "database is
        # locked") and thrashes the GPU. Jobs beyond the limit wait here.
        self._ingest_semaphore = asyncio.Semaphore(
            max(1, settings.ingestion.max_concurrent_ingestions)
        )
        # Separate, small lane for user-triggered "run now" repairs so they
        # never wait behind a hundreds-deep FIFO drain queue. The LLM
        # request cap and GPU manager still bound the actual load.
        self._priority_semaphore = asyncio.Semaphore(2)

    async def run_job(self, job_id: str, collection: str = "default") -> None:
        """Run the full pipeline for a queued job. Catches and records errors.

        Bounded by the ingestion semaphore so a large batch of uploads drains
        a few at a time instead of all at once.
        """
        # Gate before AND after the semaphore: pause-all must hold queued
        # jobs where they are, and must also catch a job whose slot came up
        # while everything was paused.
        await self.jobs.checkpoint(job_id)
        async with self._ingest_semaphore:
            await self.jobs.checkpoint(job_id)
            await self._run_job_inner(job_id, collection)

    async def run_job_now(self, job_id: str, collection: str = "default") -> None:
        """Priority ("run now") variant of run_job: the priority lane
        instead of the FIFO ingest queue, for an upload the user needs
        processed immediately (and, with JobManager.exempt_from_pause set
        by the caller, even while pause-all is on). Also what the E2E smoke
        suite uses so it can run without unleashing the paused backlog."""
        async with self._priority_semaphore:
            await self.jobs.checkpoint(job_id)
            await self._run_job_inner(job_id, collection)

    # The full-ingestion plan, in execution order. Written to the job's step
    # ledger up front so the UI can show every step — including ones that end
    # up skipped — with an explicit status instead of silently omitting them.
    FULL_PLAN = [
        "registering",
        "rendering_pages",
        "extracting_text",
        "auto_tagging",
        "embedding_text",
        "building_chunks",
        "embedding_visual",
        "extracting_entities",
        "dedup_entities",
    ]

    async def _run_job_inner(self, job_id: str, collection: str = "default") -> None:
        current_job_id.set(job_id)  # route log records into this job's log
        step = None  # ledger name of the in-flight fatal step, for the except
        try:
            job = await self.jobs.get(job_id)
            if job is None:
                logger.error("Job %s not found", job_id)
                return

            await self.jobs.set_steps(job_id, self.FULL_PLAN)

            step = "registering"
            await self.jobs.update(job_id, status="processing", current_step="registering")
            await self.jobs.update_step(job_id, "registering", "running")
            doc_id, file_hash, page_count = await self._register(job, collection=collection)
            await self.jobs.update_step(
                job_id, "registering", "done", detail=f"{page_count} pages"
            )

            step = "rendering_pages"
            await self.jobs.update(
                job_id,
                current_step="rendering_pages",
                progress_pct=5.0,
                doc_id=doc_id,
                file_hash=file_hash,
                pages_total=page_count,
            )
            await self.jobs.update_step(job_id, "rendering_pages", "running")
            await self._rasterize(job_id, job.source_path, file_hash, page_count)
            await self.jobs.update_step(
                job_id, "rendering_pages", "done", detail=f"{page_count} pages rendered"
            )

            step = "extracting_text"
            await self.jobs.update(
                job_id, current_step="extracting_text", progress_pct=40.0
            )
            await self.jobs.update_step(job_id, "extracting_text", "running")
            text_detail = await self._extract_text(job_id, job.source_path, doc_id, file_hash)
            await self.jobs.update_step(
                job_id, "extracting_text", "done", detail=text_detail
            )

            # Auto-tag if the user didn't manually specify tags/categories
            # and the LLM is available. Runs after text extraction so we have
            # page text to analyze.
            if self.auto_tagger is None:
                await self.jobs.update_step(
                    job_id, "auto_tagging", "skipped", detail="LLM service not available"
                )
            elif job.requested_categories or job.requested_tags:
                await self.jobs.update_step(
                    job_id, "auto_tagging", "skipped",
                    detail="manual categories/tags provided",
                )
            elif collection != "default":
                await self.jobs.update_step(
                    job_id, "auto_tagging", "skipped",
                    detail=f"explicit collection '{collection}' selected",
                )
            else:
                await self.jobs.update(job_id, current_step="auto_tagging")
                await self.jobs.update_step(job_id, "auto_tagging", "running")
                try:
                    tag_detail = await self._auto_tag(doc_id, collection)
                    await self.jobs.update_step(
                        job_id, "auto_tagging", "done", detail=tag_detail
                    )
                except Exception as exc:
                    logger.warning("Auto-tagging failed (continuing): %s", exc)
                    await self.jobs.update_step(
                        job_id, "auto_tagging", "error", detail=str(exc)
                    )

            # Phase 3 steps — only run if services are wired up
            if self.text_embedding is None:
                await self.jobs.update_step(
                    job_id, "embedding_text", "skipped",
                    detail="text embedding service not available",
                )
            else:
                step = "embedding_text"
                await self.jobs.update(
                    job_id, current_step="embedding_text", progress_pct=60.0
                )
                await self.jobs.update_step(job_id, "embedding_text", "running")
                n = await self._embed_text(job_id, doc_id)
                await self.jobs.update_step(
                    job_id, "embedding_text", "done",
                    detail=f"{n} pages embedded" if n else "all pages already embedded",
                )

            # Phase 5: structural chunking + per-chunk summarization + embedding.
            # Replaces whole-page text as the primary retrieval target. Runs
            # only if all pieces are wired (chunker, summarizer, embedder).
            if self.chunk_summarizer is None or self.text_embedding is None:
                await self.jobs.update_step(
                    job_id, "building_chunks", "skipped",
                    detail="LLM or text embedding service not available",
                )
            else:
                await self.jobs.update(
                    job_id, current_step="building_chunks", progress_pct=68.0
                )
                await self.jobs.update_step(job_id, "building_chunks", "running")
                try:
                    counts = await self._build_chunks(job_id, doc_id, file_hash, job.source_path)
                    n_chunks = counts.get("chunks", 0)
                    n_preview = counts.get("preview_summaries", 0)
                    if not n_chunks:
                        await self.jobs.update_step(
                            job_id, "building_chunks", "warning",
                            detail="chunker produced no chunks",
                        )
                    elif n_preview:
                        await self.jobs.update_step(
                            job_id, "building_chunks", "warning",
                            detail=f"{n_chunks} chunks written, but {n_preview} "
                            "summaries fell back to text previews (LLM "
                            "failures) — run 'Resummarize fallbacks' to repair",
                        )
                    else:
                        await self.jobs.update_step(
                            job_id, "building_chunks", "done",
                            detail=f"{n_chunks} chunks written",
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Chunking failed for doc %s (continuing): %s", doc_id, exc
                    )
                    await self.jobs.update_step(
                        job_id, "building_chunks", "error", detail=str(exc)
                    )

            if self.colpali is None:
                await self.jobs.update_step(
                    job_id, "embedding_visual", "skipped",
                    detail="visual embedding service not available",
                )
            else:
                step = "embedding_visual"
                await self.jobs.update(
                    job_id, current_step="embedding_visual", progress_pct=75.0
                )
                await self.jobs.update_step(job_id, "embedding_visual", "running")
                n = await self._embed_visual(job_id, doc_id, file_hash)
                await self.jobs.update_step(
                    job_id, "embedding_visual", "done",
                    detail=f"{n} pages embedded" if n else "all pages already embedded",
                )

            if self.entity_extractor is None:
                await self.jobs.update_step(
                    job_id, "extracting_entities", "skipped",
                    detail="LLM service not available",
                )
            else:
                step = "extracting_entities"
                await self.jobs.update(
                    job_id, current_step="extracting_entities", progress_pct=88.0
                )
                await self.jobs.update_step(job_id, "extracting_entities", "running")
                done, failed, last_err = await self._extract_entities(
                    job_id, doc_id
                )
                if done and failed == done:
                    # Every page failed — fail the job instead of completing
                    # with a warning nobody reads.
                    raise RuntimeError(
                        f"entity extraction failed for all {done} pages "
                        f"(last error: {last_err}) — pages remain unstamped "
                        "and will be retried on the next run"
                    )
                if failed:
                    await self.jobs.update_step(
                        job_id, "extracting_entities", "warning",
                        detail=f"{failed} of {done} pages failed — see logs",
                    )
                else:
                    await self.jobs.update_step(
                        job_id, "extracting_entities", "done",
                        detail=f"{done} pages extracted",
                    )

            # Post-extraction: merge near-duplicate entities created by this doc.
            # Runs a lightweight dedup pass on entities linked to this document
            # so "Stainless Steel" and "stainless steel" don't pile up as
            # separate nodes across ingestions.
            try:
                await self.jobs.update(
                    job_id, current_step="dedup_entities", progress_pct=95.0
                )
                await self.jobs.update_step(job_id, "dedup_entities", "running")
                merged = await self._dedup_doc_entities(doc_id)
                if merged:
                    logger.info("Post-ingestion dedup merged %d entities for doc %s", merged, doc_id)
                await self.jobs.update_step(
                    job_id, "dedup_entities", "done",
                    detail=f"{merged} duplicate entities merged",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Post-ingestion dedup failed (non-fatal): %s", exc)
                await self.jobs.update_step(
                    job_id, "dedup_entities", "error", detail=str(exc)
                )

            await self.jobs.complete(job_id)
            logger.info("Job %s completed successfully", job_id)

        except Exception as exc:  # noqa: BLE001
            logger.exception("Job %s failed", job_id)
            if step is not None:
                await self.jobs.update_step(job_id, step, "error", detail=str(exc))
            await self.jobs.fail(job_id, str(exc))

    async def run_build_summaries(self, job_id: str, doc_id: str) -> None:
        """Build the RAPTOR-by-TOC section-summary tree for one document.

        Bottom-up LLM summaries over the Docling section structure (leaf
        sections -> chapters -> whole document), embedded and stored as
        :SectionSummary nodes with PARENT_OF edges and page ranges.
        Idempotent per doc: existing summaries are replaced wholesale.
        Stamps d.summaries_built_at only when every summary landed
        (count-verified) so the audit never treats a partial tree as done.
        """
        current_job_id.set(job_id)
        try:
            # Queue behind the shared ingest semaphore — a bulk drain
            # spawns hundreds of these, and without the gate they would
            # ALL run concurrently (the LLM client's own semaphore
            # throttles model calls, but chunk loads and checkpoint
            # traffic would hammer Neo4j and make every doc crawl).
            async with self._ingest_semaphore:
                await self._run_build_summaries_inner(job_id, doc_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Build-summaries job %s failed", job_id)
            await self.jobs.fail(job_id, str(exc))

    async def _run_build_summaries_inner(self, job_id: str, doc_id: str) -> None:
        from backend.ingestion.toc_summarizer import (
            TocSummarizer,
            build_section_tree,
            iter_nodes_bottom_up,
            summary_id,
        )

        step = None
        try:
            await self.jobs.checkpoint(job_id)
            await self.jobs.set_steps(job_id, [
                "loading_chunks", "summarizing_sections",
                "embedding_summaries", "writing_summaries",
            ])
            if self.llm is None or self.text_embedding is None:
                raise ValueError(
                    "LLM and text-embedding services are required to build "
                    "summaries"
                )
            rows = await self.neo4j.run_query(
                "MATCH (d:Document {doc_id: $d}) "
                "RETURN d.title AS title, d.file_hash AS h",
                {"d": doc_id},
            )
            if not rows:
                raise ValueError(f"Document {doc_id} not found")
            title = rows[0]["title"] or doc_id

            step = "loading_chunks"
            await self.jobs.update(
                job_id, status="processing", doc_id=doc_id,
                file_hash=rows[0]["h"], current_step="loading_chunks",
                progress_pct=2.0,
            )
            await self.jobs.update_step(job_id, "loading_chunks", "running")
            chunks = await self.neo4j.run_query(
                """
                MATCH (d:Document {doc_id: $d})-[:HAS_PAGE]->(:Page)
                      -[:HAS_CHUNK]->(c:Chunk)
                RETURN c.section_path AS section_path,
                       c.page_number AS page_number,
                       c.summary AS summary, c.chunk_type AS chunk_type,
                       left(c.text, 600) AS text
                ORDER BY c.page_number, c.chunk_index
                """,
                {"d": doc_id}, timeout=300.0,
            )
            if not chunks:
                raise ValueError(
                    f"Document {doc_id} has no chunks — build chunks first "
                    "(summaries are rolled up from chunk summaries)"
                )
            root = build_section_tree(title, chunks)
            await self.jobs.update_step(
                job_id, "loading_chunks", "done",
                detail=f"{len(chunks)} chunks into section tree",
            )

            step = "summarizing_sections"
            nodes = list(iter_nodes_bottom_up(root))
            await self.jobs.update(
                job_id, current_step="summarizing_sections",
                progress_pct=8.0, pages_total=len(nodes), pages_processed=0,
            )
            await self.jobs.update_step(
                job_id, "summarizing_sections", "running",
                detail=f"{len(nodes)} sections via LLM",
            )
            summarizer = TocSummarizer(self.llm)
            done = 0

            # summarize_tree walks bottom-up itself; we wrap its checkpoint
            # to also report progress.
            async def _progress_checkpoint():
                nonlocal done
                await self.jobs.checkpoint(job_id)
                done += 1
                if done % 5 == 0 or done >= len(nodes):
                    await self.jobs.update(
                        job_id, pages_processed=min(done, len(nodes)),
                        progress_pct=min(80.0, 8.0 + 72.0 * done / max(len(nodes), 1)),
                    )

            n_summarized = await summarizer.summarize_tree(
                title, root, checkpoint=_progress_checkpoint,
            )
            filled = [n for n in nodes if n.summary]
            if not filled:
                raise RuntimeError(
                    "summarization produced no section summaries — see "
                    "per-call log warnings; document remains unstamped"
                )
            await self.jobs.update_step(
                job_id, "summarizing_sections", "done",
                detail=f"{n_summarized} section summaries",
            )

            step = "embedding_summaries"
            await self.jobs.update(
                job_id, current_step="embedding_summaries", progress_pct=82.0,
            )
            await self.jobs.update_step(job_id, "embedding_summaries", "running")
            assert self.gpu is not None
            async with self.gpu.load_scope("text_embedding"):
                vectors = await asyncio.to_thread(
                    self.text_embedding.embed_documents,
                    [n.summary for n in filled],
                    batch_size=self.settings.ingestion.text_embedding_batch_size,
                )
            await self.jobs.update_step(
                job_id, "embedding_summaries", "done",
                detail=f"{len(vectors)} vectors",
            )

            step = "writing_summaries"
            await self.jobs.update(
                job_id, current_step="writing_summaries", progress_pct=90.0,
            )
            await self.jobs.update_step(job_id, "writing_summaries", "running")
            # Self-ensure schema (seed_schema is a manual script; a fresh
            # install must not need it before this job works).
            await self.neo4j.run_write(
                "CREATE CONSTRAINT summary_id_unique IF NOT EXISTS "
                "FOR (s:SectionSummary) REQUIRE s.summary_id IS UNIQUE"
            )
            await self.neo4j.run_write(
                f"""CREATE VECTOR INDEX section_summary_embedding IF NOT EXISTS
                FOR (s:SectionSummary) ON (s.embedding)
                OPTIONS {{ indexConfig: {{
                    `vector.dimensions`: {self.settings.models.text_embedding_dim},
                    `vector.similarity_function`: 'cosine'
                }} }}"""
            )
            # Replace wholesale — the tree is derived data.
            await self.neo4j.run_write(
                "MATCH (d:Document {doc_id: $d})-[:HAS_SUMMARY]->(s:SectionSummary) "
                "DETACH DELETE s",
                {"d": doc_id},
            )
            srows = []
            for node, vec in zip(filled, vectors):
                srows.append({
                    "summary_id": summary_id(doc_id, node.path),
                    "parent_id": (
                        summary_id(doc_id, node.path[:-1]) if node.path else None
                    ),
                    "path": list(node.path),
                    "title": node.title,
                    "level": node.level,
                    "summary": node.summary,
                    "page_start": node.page_start,
                    "page_end": node.page_end,
                    "embedding": vec.tolist(),
                })
            await self.neo4j.run_write(
                """
                UNWIND $rows AS row
                MATCH (d:Document {doc_id: $d})
                MERGE (s:SectionSummary {summary_id: row.summary_id})
                SET s.doc_id = $d, s.path = row.path, s.title = row.title,
                    s.level = row.level, s.summary = row.summary,
                    s.page_start = row.page_start, s.page_end = row.page_end,
                    s.embedding = row.embedding
                MERGE (d)-[:HAS_SUMMARY]->(s)
                """,
                {"d": doc_id, "rows": srows},
            )
            await self.neo4j.run_write(
                """
                UNWIND $rows AS row
                WITH row WHERE row.parent_id IS NOT NULL
                MATCH (p:SectionSummary {summary_id: row.parent_id})
                MATCH (s:SectionSummary {summary_id: row.summary_id})
                MERGE (p)-[:PARENT_OF]->(s)
                """,
                {"rows": srows},
            )
            # Count-verify before stamping (house rule: never stamp
            # shrunken coverage as final).
            wrote = await self.neo4j.run_query(
                "MATCH (d:Document {doc_id: $d})-[:HAS_SUMMARY]->(s) "
                "RETURN count(s) AS n",
                {"d": doc_id},
            )
            actually = wrote[0]["n"] if wrote else 0
            if actually != len(srows):
                await self.jobs.update_step(
                    job_id, "writing_summaries", "warning",
                    detail=f"{len(srows) - actually} of {len(srows)} summary "
                    "rows dropped — summaries_built_at NOT stamped",
                )
            else:
                await self.neo4j.run_write(
                    "MATCH (d:Document {doc_id: $d}) "
                    "SET d.summaries_built_at = datetime()",
                    {"d": doc_id},
                )
                await self.jobs.update_step(
                    job_id, "writing_summaries", "done",
                    detail=f"{actually} summaries written",
                )
            await self.jobs.complete(job_id)
            logger.info(
                "Summary tree built for doc %s: %d sections", doc_id, actually,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Build-summaries job %s failed", job_id)
            if step is not None:
                await self.jobs.update_step(job_id, step, "error", detail=str(exc))
            await self.jobs.fail(job_id, str(exc))

    async def run_communities_only(self, job_id: str) -> None:
        """Rebuild all :Community nodes globally from the current graph.

        Not per-document — community detection spans all ingested pages
        because engineering topics connect across handbooks. The job is
        scoped to tracking the long operation, not to a specific doc.
        """
        current_job_id.set(job_id)
        try:
            # The build itself is one monolithic call — pause/stop can only
            # take effect before it starts.
            await self.jobs.checkpoint(job_id)
            await self.jobs.set_steps(job_id, ["building_graph"])
            if self.community_detector is None:
                raise ValueError("LLM or text embedding unavailable — cannot detect communities")
            await self.jobs.update(
                job_id, status="processing", current_step="building_graph", progress_pct=5.0
            )
            await self.jobs.update_step(job_id, "building_graph", "running")
            assert self.gpu is not None
            async with self.gpu.load_scope("text_embedding"):
                counts = await self.community_detector.build()
            logger.info("Community detection complete: %s", counts)
            await self.jobs.update_step(
                job_id, "building_graph", "done", detail=str(counts)
            )
            await self.jobs.complete(job_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Community job %s failed", job_id)
            msg = str(exc) or f"{type(exc).__name__} (no message)"
            await self.jobs.update_step(job_id, "building_graph", "error", detail=msg)
            await self.jobs.fail(job_id, msg)

    # Chunks whose summary is a failure fallback: marked explicitly by
    # current code, or detectable on legacy rows because the fallback was
    # exactly the first 240 chars of the (stripped) text. Long chunks only —
    # short chunks legitimately use their text as the summary.
    _FALLBACK_SUMMARY_PREDICATE = (
        "c.summary_source = 'preview' OR "
        "(size(c.text) >= 400 AND c.summary IN "
        "[left(c.text, 240), left(trim(c.text), 240)])"
    )

    async def run_resummarize(self, job_id: str) -> None:
        """Regenerate chunk summaries that fell back to text previews.

        Global job (all documents). For each fallback chunk: LLM summary,
        re-embed (the retrieval vector is summary + text, so a new summary
        needs a new vector), write back with summary_source='llm'. Chunks
        whose regeneration fails again keep their preview marking and are
        picked up by the next run — failures are never silently converted
        into 'done'.

        Deliberately NOT behind the ingestion job semaphore: semaphore
        waiters wake in FIFO order, so queued bulk-drain jobs would starve
        this repair for days. Its LLM calls are bounded by the LLMService
        request cap and its embed batches are serialized by the GPU
        manager — those are the actual shared resources.
        """
        current_job_id.set(job_id)
        await self._run_resummarize_inner(job_id)

    async def _run_resummarize_inner(self, job_id: str) -> None:
        try:
            await self.jobs.set_steps(job_id, ["resummarizing"])
            if self.chunk_summarizer is None or self.text_embedding is None:
                raise ValueError(
                    "LLM and text embedding services are required to "
                    "regenerate summaries"
                )
            assert self.gpu is not None

            rows = await self.neo4j.run_query(
                f"MATCH (c:Chunk) WHERE {self._FALLBACK_SUMMARY_PREDICATE} "
                "RETURN count(c) AS n",
                timeout=600.0,
            )
            total = rows[0]["n"] if rows else 0
            await self.jobs.update(
                job_id, status="processing", current_step="resummarizing",
                pages_total=total, progress_pct=1.0,
            )
            if not total:
                await self.jobs.update_step(
                    job_id, "resummarizing", "done",
                    detail="no fallback summaries found",
                )
                await self.jobs.complete(job_id)
                return
            await self.jobs.update_step(
                job_id, "resummarizing", "running",
                detail=f"{total} fallback summaries to regenerate",
            )

            done = 0
            failed = 0
            failed_ids: list[str] = []
            BATCH = 100
            while True:
                await self.jobs.checkpoint(job_id)
                batch = await self.neo4j.run_query(
                    f"""
                    MATCH (c:Chunk)
                    WHERE ({self._FALLBACK_SUMMARY_PREDICATE})
                      AND NOT c.chunk_id IN $skip
                    OPTIONAL MATCH (d:Document {{doc_id: c.doc_id}})
                    RETURN c.chunk_id AS chunk_id,
                           c.page_number AS page_number,
                           c.chunk_type AS chunk_type,
                           c.text AS text,
                           c.section_path AS section_path,
                           d.filename AS filename
                    LIMIT $batch
                    """,
                    {"skip": failed_ids, "batch": BATCH},
                )
                if not batch:
                    break
                names = sorted({r.get("filename") for r in batch if r.get("filename")})
                item = f"chunks {done + 1}–{done + len(batch)} of {total}"
                if names:
                    item += f" — {names[0]}"
                    if len(names) > 1:
                        item += f" +{len(names) - 1} more doc(s)"
                await self.jobs.update(job_id, current_item=item)
                chunks = [
                    StructuralChunk(
                        chunk_id=r["chunk_id"],
                        page_number=r["page_number"] or 0,
                        chunk_index=0,
                        chunk_type=r["chunk_type"] or "text",
                        text=r["text"] or "",
                        section_path=list(r["section_path"] or []),
                    )
                    for r in batch
                ]
                results = await self.chunk_summarizer.summarize_batch(
                    chunks, concurrency=2,
                )
                repaired: list[tuple[StructuralChunk, str, str]] = []
                for ch, (summary, source) in zip(chunks, results):
                    if source == "preview":
                        # LLM failed again — leave the chunk marked so the
                        # next run retries it; skip it in this run's queries.
                        failed += 1
                        failed_ids.append(ch.chunk_id)
                        continue
                    repaired.append((ch, summary, source))

                if repaired:
                    embed_inputs = [
                        f"{s}\n\n{ch.text[:2000]}" for ch, s, _src in repaired
                    ]
                    async with self.gpu.load_scope("text_embedding"):
                        vectors = await asyncio.to_thread(
                            self.text_embedding.embed_documents, embed_inputs,
                            batch_size=(
                                self.settings.ingestion.text_embedding_batch_size
                            ),
                        )
                    out = [
                        {"chunk_id": ch.chunk_id, "summary": s,
                         "source": src, "embedding": vec.tolist()}
                        for (ch, s, src), vec in zip(repaired, vectors)
                    ]
                    await self.neo4j.run_write(
                        """
                        UNWIND $rows AS row
                        MATCH (c:Chunk {chunk_id: row.chunk_id})
                        SET c.summary = row.summary,
                            c.embedding = row.embedding,
                            c.summary_source = row.source
                        """,
                        {"rows": out},
                    )

                done += len(batch)
                await self.jobs.update(
                    job_id, pages_processed=done,
                    progress_pct=min(99.0, 1.0 + 98.0 * done / max(total, 1)),
                )

            if done and failed == done:
                raise RuntimeError(
                    f"re-summarization failed for all {done} chunks — "
                    "LLM endpoint unreachable? Chunks remain marked and "
                    "will be retried on the next run"
                )
            if failed:
                await self.jobs.update_step(
                    job_id, "resummarizing", "warning",
                    detail=f"{failed} of {done} chunks failed — "
                    "run again to retry them",
                )
            else:
                await self.jobs.update_step(
                    job_id, "resummarizing", "done",
                    detail=f"{done - failed} summaries regenerated",
                )
            await self.jobs.complete(job_id)
            logger.info(
                "Resummarize job %s: %d regenerated, %d failed",
                job_id, done - failed, failed,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Resummarize job %s failed", job_id)
            await self.jobs.update_step(
                job_id, "resummarizing", "error", detail=str(exc)
            )
            await self.jobs.fail(job_id, str(exc))

    async def run_autotag_missing(self, job_id: str) -> None:
        """Auto-tag every unorganized document.

        Unorganized = default collection with zero categories and zero
        tags — the state a doc lands in when auto-tagging fails during
        ingest (the failure is marked on the job step, but the doc itself
        just looks untagged). Global job. Not behind the ingestion
        semaphore (the LLM request cap is the shared-resource guard, and
        queued bulk jobs would starve this repair). Documents that fail
        again stay unorganized and are picked up by the next run.
        """
        current_job_id.set(job_id)
        try:
            await self.jobs.set_steps(job_id, ["auto_tagging"])
            if self.auto_tagger is None:
                raise ValueError("LLM service not configured — cannot auto-tag")
            rows = await self.neo4j.run_query(
                """
                MATCH (d:Document)
                WHERE coalesce(d.collection, 'default') = 'default'
                  AND NOT (d)-[:IN_CATEGORY]->()
                  AND NOT (d)-[:TAGGED_WITH]->()
                RETURN d.doc_id AS doc_id, d.filename AS filename
                """,
            )
            total = len(rows)
            await self.jobs.update(
                job_id, status="processing", current_step="auto_tagging",
                pages_total=total, progress_pct=1.0,
            )
            if not total:
                await self.jobs.update_step(
                    job_id, "auto_tagging", "done",
                    detail="every document is organized",
                )
                await self.jobs.complete(job_id)
                return
            await self.jobs.update_step(
                job_id, "auto_tagging", "running",
                detail=f"{total} unorganized documents",
            )
            done = 0
            failed = 0
            no_text = 0
            for r in rows:
                await self.jobs.checkpoint(job_id)
                await self.jobs.update(
                    job_id,
                    current_item=f"{r.get('filename') or r['doc_id']} "
                    f"({done + 1}/{total})",
                )
                try:
                    detail = await self._auto_tag(r["doc_id"], "default")
                    # suggest_for_doc returns None for docs with no usable
                    # text — that is not a success, it's a doc whose text
                    # extraction needs repair first. Count it separately so
                    # "N documents tagged" is never claimed for no-ops.
                    if detail == "no suggestions from LLM":
                        no_text += 1
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    logger.warning(
                        "Auto-tag failed for doc %s: %s", r["doc_id"], exc
                    )
                done += 1
                await self.jobs.update(
                    job_id, pages_processed=done,
                    progress_pct=min(99.0, 1.0 + 98.0 * done / total),
                )
            tagged = done - failed - no_text
            if done and failed == done:
                raise RuntimeError(
                    f"auto-tagging failed for all {done} documents — "
                    "LLM endpoint unreachable? Documents remain unorganized "
                    "and will be retried on the next run"
                )
            parts = [f"{tagged} documents tagged"]
            if no_text:
                parts.append(
                    f"{no_text} skipped (no text to analyze — repair text "
                    "extraction first)"
                )
            if failed:
                parts.append(f"{failed} failed — run again to retry")
            await self.jobs.update_step(
                job_id, "auto_tagging",
                "warning" if (failed or no_text) else "done",
                detail=", ".join(parts),
            )
            await self.jobs.complete(job_id)
            logger.info(
                "Autotag job %s: %d tagged, %d failed",
                job_id, done - failed, failed,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Autotag job %s failed", job_id)
            await self.jobs.update_step(
                job_id, "auto_tagging", "error", detail=str(exc)
            )
            await self.jobs.fail(job_id, str(exc))

    async def run_extraction_only(self, job_id: str, doc_id: str) -> None:
        """Re-run only entity extraction for an already-ingested document.

        Used by POST /documents/{doc_id}/extract-entities.
        """
        current_job_id.set(job_id)
        try:
            await self.jobs.checkpoint(job_id)
            await self.jobs.set_steps(job_id, ["extracting_entities"])
            await self.jobs.update(job_id, status="processing", doc_id=doc_id)
            if self.entity_extractor is None:
                raise ValueError("LLM service not configured — cannot extract entities")
            if await self._page_count(doc_id) == 0:
                raise ValueError(
                    f"Document {doc_id} has 0 pages — it was only partially "
                    "ingested. Delete it and re-ingest the PDF; entity "
                    "extraction has no pages to work on."
                )
            await self.jobs.update(
                job_id, current_step="extracting_entities", progress_pct=10.0
            )
            await self.jobs.update_step(job_id, "extracting_entities", "running")
            done, failed, last_err = await self._extract_entities(job_id, doc_id)
            if done and failed == done:
                raise RuntimeError(
                    f"entity extraction failed for all {done} pages "
                    f"(last error: {last_err}) — pages remain unstamped "
                    "and will be retried on the next run"
                )
            if failed:
                await self.jobs.update_step(
                    job_id, "extracting_entities", "warning",
                    detail=f"{failed} of {done} pages failed — see logs",
                )
            else:
                await self.jobs.update_step(
                    job_id, "extracting_entities", "done",
                    detail=f"{done} pages extracted",
                )
            await self.jobs.complete(job_id)
            logger.info("Extraction-only job %s completed for doc %s", job_id, doc_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Extraction-only job %s failed", job_id)
            await self.jobs.update_step(
                job_id, "extracting_entities", "error", detail=str(exc)
            )
            await self.jobs.fail(job_id, str(exc))

    async def run_rebuild_chunks(
        self, job_id: str, doc_id: str, *,
        extract_only: bool = False,
        skip_extract: bool = False,
    ) -> None:
        """Phase 5 rebuild: structural chunks + summaries + embeddings +
        Phase 3 entity re-extraction on pages missing topic_tags.

        Modes:
        - default: chunks + summaries + embeddings + entity re-extraction
        - extract_only=True: only re-extract entities on pages that need it
          (cheap resume after an extractor bug fix)
        - skip_extract=True: chunks/summaries/embeddings only (no entity work)

        Runs on the long-lived in-process services so a GUI-triggered
        rebuild doesn't re-download models or re-apply schema on every run.
        """
        current_job_id.set(job_id)
        step = None
        REBUILD_PLAN = [
            "chunking", "summarizing", "embedding_chunks",
            "writing_chunks", "extracting_entities",
        ]
        try:
            await self.jobs.checkpoint(job_id)
            await self.jobs.set_steps(job_id, REBUILD_PLAN)
            await self.jobs.update(job_id, status="processing", doc_id=doc_id)

            if extract_only and skip_extract:
                raise ValueError(
                    "extract_only and skip_extract are mutually exclusive"
                )

            rows = await self.neo4j.run_query(
                "MATCH (d:Document {doc_id: $d}) "
                "RETURN d.title AS title, d.file_hash AS file_hash, "
                "       d.filename AS filename",
                {"d": doc_id},
            )
            if not rows:
                raise ValueError(f"Document {doc_id} not found")
            title = rows[0]["title"] or doc_id
            file_hash = rows[0]["file_hash"]
            filename = rows[0]["filename"] or ""

            # Guard: a rebuild attaches chunks/entities to existing :Page
            # nodes. A document with zero pages (e.g. one whose ingestion
            # died during registration) would otherwise rebuild to nothing
            # and report success. Fail loudly so the user knows it needs a
            # full re-ingestion, not a rebuild.
            if await self._page_count(doc_id) == 0:
                raise ValueError(
                    f"Document {doc_id} has 0 pages — it was only partially "
                    "ingested. Delete it and re-ingest the PDF; a rebuild "
                    "cannot recreate missing pages."
                )

            # Locate the source PDF and VERIFY it by content hash. Uploads
            # are staged as "{uuid4().hex}_{basename}", so the old
            # hash-prefix glob could never match, and its filename-suffix
            # fallback could silently pick a same-named DIFFERENT document —
            # rebuilding this doc's chunks from the wrong PDF's text.
            upload_dir = Path(self.settings.server.data_dir) / "uploads"
            candidates = sorted(upload_dir.glob(f"*_{filename}")) if filename else []
            pdf_path = None
            for cand in candidates:
                if await _sha256_file(cand) == file_hash:
                    pdf_path = cand
                    break
            if pdf_path is None:
                raise ValueError(
                    f"No staged upload matching this document's file hash "
                    f"found in {upload_dir} for doc {doc_id} "
                    f"({len(candidates)} same-named candidate(s) rejected by "
                    "hash check) — re-upload the PDF, then rebuild"
                )

            if extract_only:
                for name in ("chunking", "summarizing", "embedding_chunks", "writing_chunks"):
                    await self.jobs.update_step(
                        job_id, name, "skipped", detail="extract_only mode"
                    )
            else:
                if self.text_embedding is None or self.chunk_summarizer is None:
                    raise ValueError(
                        "text_embedding or chunk_summarizer not configured — "
                        "cannot build chunks"
                    )
                step = "chunking"
                await self.jobs.update(
                    job_id, current_step="chunking", progress_pct=5.0,
                )
                await self.jobs.update_step(job_id, "chunking", "running")
                chunks = await asyncio.to_thread(
                    self.structural_chunker.chunk_pdf, pdf_path, file_hash,
                )
                if not chunks:
                    # Same self-heal as the ingest lane: text-less PDFs
                    # (vector-outline exports) yield nothing until Docling
                    # is handed a rasterized rebuild it can OCR. This is
                    # the exact lane docs_have_chunks tells users to run —
                    # without the fallback here, the recommended repair
                    # dead-ended on precisely the PDFs it exists to fix.
                    rebuilt = await asyncio.to_thread(
                        self._rasterized_pdf_from_page_images, file_hash,
                    )
                    if rebuilt is not None:
                        await self.jobs.update_step(
                            job_id, "chunking", "running",
                            detail="no chunks from original PDF (text-less?)"
                            " — retrying OCR on rasterized rebuild",
                        )
                        try:
                            chunks = await asyncio.to_thread(
                                self.structural_chunker.chunk_pdf,
                                rebuilt, file_hash,
                            )
                        finally:
                            Path(rebuilt).unlink(missing_ok=True)
                if not chunks:
                    logger.warning(
                        "Chunker produced no chunks for doc %s", doc_id,
                    )
                    await self.jobs.update_step(
                        job_id, "chunking", "warning",
                        detail="chunker produced no chunks "
                        "(rasterized retry included)",
                    )
                    for name in ("summarizing", "embedding_chunks", "writing_chunks"):
                        await self.jobs.update_step(
                            job_id, name, "skipped", detail="no chunks to process"
                        )
                else:
                    await self.jobs.update_step(
                        job_id, "chunking", "done", detail=f"{len(chunks)} chunks"
                    )
                    step = "summarizing"
                    await self.jobs.update(
                        job_id, current_step="summarizing",
                        pages_total=len(chunks), progress_pct=20.0,
                    )
                    await self.jobs.update_step(job_id, "summarizing", "running")
                    summarized = await self.chunk_summarizer.summarize_batch(
                        chunks, concurrency=4,
                    )
                    summaries = [s for s, _src in summarized]
                    sources = [src for _s, src in summarized]
                    preview_count = sum(
                        1 for src in sources if src == "preview"
                    )
                    await self.jobs.update_step(
                        job_id, "summarizing",
                        "warning" if preview_count else "done",
                        detail=(
                            f"{preview_count} of {len(summaries)} summaries "
                            "fell back to text previews (LLM failures) — "
                            "run 'Resummarize fallbacks' to repair"
                        ) if preview_count else f"{len(summaries)} summaries",
                    )
                    step = "embedding_chunks"
                    await self.jobs.update(
                        job_id, current_step="embedding_chunks",
                        progress_pct=55.0,
                    )
                    await self.jobs.update_step(job_id, "embedding_chunks", "running")
                    embed_inputs = [
                        f"{s}\n\n{c.text[:2000]}"
                        for s, c in zip(summaries, chunks)
                    ]
                    assert self.gpu is not None
                    async with self.gpu.load_scope("text_embedding"):
                        vectors = await asyncio.to_thread(
                            self.text_embedding.embed_documents,
                            embed_inputs,
                            batch_size=self.settings.ingestion.text_embedding_batch_size,
                        )
                    await self.jobs.update_step(
                        job_id, "embedding_chunks", "done",
                        detail=f"{len(vectors)} vectors",
                    )
                    step = "writing_chunks"
                    await self.jobs.update(
                        job_id, current_step="writing_chunks",
                        progress_pct=65.0,
                    )
                    await self.jobs.update_step(job_id, "writing_chunks", "running")
                    BATCH = 200
                    for i in range(0, len(chunks), BATCH):
                        end = min(i + BATCH, len(chunks))
                        rows_to_write = []
                        for ch, summ, src, vec in zip(
                            chunks[i:end], summaries[i:end],
                            sources[i:end], vectors[i:end]
                        ):
                            rows_to_write.append({
                                "chunk_id": ch.chunk_id,
                                "page_number": ch.page_number,
                                "chunk_index": ch.chunk_index,
                                "chunk_type": ch.chunk_type,
                                "text": ch.text,
                                "summary": summ,
                                "summary_source": src,
                                "section_path": ch.section_path,
                                "embedding": vec.tolist(),
                                "bbox": list(ch.bbox) if ch.bbox is not None else None,
                            })
                        await self.neo4j.run_write(
                            """
                            UNWIND $rows AS row
                            MATCH (d:Document {doc_id: $doc_id})-[:HAS_PAGE]->(p:Page {page_number: row.page_number})
                            MERGE (c:Chunk {chunk_id: row.chunk_id})
                            ON CREATE SET c.page_number = row.page_number,
                                          c.chunk_index = row.chunk_index,
                                          c.chunk_type = row.chunk_type,
                                          c.text = row.text,
                                          c.summary = row.summary,
                                          c.summary_source = row.summary_source,
                                          c.section_path = row.section_path,
                                          c.embedding = row.embedding,
                                          c.bbox = row.bbox,
                                          c.doc_id = $doc_id
                            ON MATCH SET  c.text = row.text,
                                          c.summary = row.summary,
                                          c.summary_source = row.summary_source,
                                          c.section_path = row.section_path,
                                          c.chunk_type = row.chunk_type,
                                          c.embedding = row.embedding,
                                          c.bbox = row.bbox
                            MERGE (p)-[:HAS_CHUNK]->(c)
                            """,
                            {"doc_id": doc_id, "rows": rows_to_write},
                        )
                        # pages_total was set to the chunk count during
                        # summarizing; keep pages_processed in step so the
                        # job card shows 26/26 instead of a misleading 0/26.
                        await self.jobs.update(job_id, pages_processed=end)
                    # Verify what actually landed — the UNWIND's MATCH
                    # silently drops rows whose page_number has no :Page
                    # node, and the old code reported the CHUNKER's count
                    # as "written" and stamped regardless.
                    written_rows = await self.neo4j.run_query(
                        "MATCH (c:Chunk {doc_id: $id}) RETURN count(c) AS n",
                        {"id": doc_id},
                    )
                    actually_written = written_rows[0]["n"] if written_rows else 0
                    dropped = max(0, len(chunks) - actually_written)
                    if dropped > 0:
                        logger.warning(
                            "%d of %d chunk rows dropped for doc %s (no "
                            "matching page node) — chunks_built_at NOT "
                            "stamped", dropped, len(chunks), doc_id,
                        )
                        await self.jobs.update_step(
                            job_id, "writing_chunks", "warning",
                            detail=f"{dropped} of {len(chunks)} chunk rows "
                            "dropped — no matching page node",
                        )
                    else:
                        # Mark the doc as chunk-built so the completeness
                        # audit treats whatever coverage Docling achieved as
                        # final — some pages legitimately yield no chunks,
                        # and without the marker those docs read as
                        # forever-incomplete.
                        await self.neo4j.run_write(
                            "MATCH (d:Document {doc_id: $id}) "
                            "SET d.chunks_built_at = datetime()",
                            {"id": doc_id},
                        )
                        await self.jobs.update_step(
                            job_id, "writing_chunks", "done",
                            detail=f"{actually_written} chunks written",
                        )
                    step = None

            if skip_extract:
                await self.jobs.update_step(
                    job_id, "extracting_entities", "skipped",
                    detail="skip_extract mode",
                )
            else:
                if self.entity_extractor is None:
                    logger.warning(
                        "Entity extractor not configured — skipping re-extraction"
                    )
                    await self.jobs.update_step(
                        job_id, "extracting_entities", "skipped",
                        detail="LLM service not available",
                    )
                else:
                    step = "extracting_entities"
                    await self.jobs.update(
                        job_id, current_step="extracting_entities",
                        progress_pct=75.0,
                    )
                    await self.jobs.update_step(
                        job_id, "extracting_entities", "running"
                    )
                    # Re-extract only pages missing topic_tags — keeps retries cheap
                    todo = await self.neo4j.run_query(
                        """
                        MATCH (d:Document {doc_id: $d})-[:HAS_PAGE]->(p:Page)
                        WHERE p.extracted_text IS NOT NULL
                          AND (p.topic_tags IS NULL OR size(p.topic_tags) = 0)
                          AND coalesce(p.is_blank, false) = false
                        RETURN p.page_id AS page_id, p.page_number AS page_number,
                               p.extracted_text AS text,
                               p.text_char_count AS char_count
                        ORDER BY p.page_number
                        """,
                        {"d": doc_id},
                    )
                    total = len(todo)
                    await self.jobs.update(
                        job_id, pages_total=total, pages_processed=0,
                    )
                    done = 0
                    failed = 0
                    for p in todo:
                        try:
                            extraction = await self.entity_extractor.extract_page(
                                document_title=title,
                                page_number=p["page_number"],
                                page_text=p["text"],
                            )
                            page_counts = await self.graph_builder.write_page(
                                page_id=p["page_id"], extraction=extraction,
                            )
                            await self._stamp_page_extracted(
                                p["page_id"], page_counts, p.get("char_count"),
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "rebuild-chunks: page %d extraction failed: %s",
                                p["page_number"], exc,
                            )
                            failed += 1
                        done += 1
                        if done % 10 == 0 or done == total:
                            await self.jobs.update(
                                job_id, pages_processed=done,
                                progress_pct=min(99.0, 75.0 + 24.0 * done / max(total, 1)),
                            )
                    if total and failed == total:
                        raise RuntimeError(
                            f"entity extraction failed for all {total} pages "
                            "— see per-page log warnings for the actual "
                            "errors; pages remain unstamped and will be "
                            "retried on the next run"
                        )
                    if failed:
                        logger.warning(
                            "rebuild-chunks: %d/%d page extractions failed",
                            failed, total,
                        )
                        await self.jobs.update_step(
                            job_id, "extracting_entities", "warning",
                            detail=f"{failed} of {total} pages failed — see logs",
                        )
                    else:
                        await self.jobs.update_step(
                            job_id, "extracting_entities", "done",
                            detail=f"{total} pages extracted",
                        )

            await self.jobs.complete(job_id)
            logger.info("Rebuild-chunks job %s completed for doc %s", job_id, doc_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Rebuild-chunks job %s failed", job_id)
            if step is not None:
                await self.jobs.update_step(job_id, step, "error", detail=str(exc))
            await self.jobs.fail(job_id, str(exc))

    async def _page_count(self, doc_id: str) -> int:
        """Number of :Page nodes attached to a document. Zero means the doc
        was only partially ingested (registered but text-extraction never
        ran)."""
        rows = await self.neo4j.run_query(
            """
            MATCH (d:Document {doc_id: $doc_id})
            OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Page)
            RETURN count(p) AS n
            """,
            {"doc_id": doc_id},
        )
        return rows[0]["n"] if rows else 0

    async def _backfill_blank_flags(self, doc_id: str, file_hash: str) -> int:
        """Compute and store is_blank on any Page that doesn't have it yet.

        Used before re-embed so existing docs (ingested before the blank-page
        filter existed) can have their blank pages skipped on re-processing.
        Only touches pages where is_blank IS NULL — never overwrites an
        already-computed value.
        """
        rows = await self.neo4j.run_query(
            """
            MATCH (d:Document {doc_id: $doc_id})-[:HAS_PAGE]->(p:Page)
            WHERE p.is_blank IS NULL
            RETURN p.page_id AS page_id,
                   p.page_number AS page_number,
                   p.text_char_count AS text_char_count
            """,
            {"doc_id": doc_id},
        )
        if not rows:
            return 0

        def _compute_batch() -> list[dict]:
            out = []
            for r in rows:
                reduced = self.pdf_processor.reduced_image_path(
                    file_hash, r["page_number"]
                )
                blank = _is_blank_page(reduced, r["text_char_count"] or 0)
                out.append({"page_id": r["page_id"], "is_blank": blank})
            return out

        payload = await asyncio.to_thread(_compute_batch)
        await self.neo4j.run_write(
            """
            UNWIND $rows AS row
            MATCH (p:Page {page_id: row.page_id})
            SET p.is_blank = row.is_blank
            """,
            {"rows": payload},
        )
        flagged = sum(1 for r in payload if r["is_blank"])
        logger.info(
            "Backfilled is_blank on %d pages of doc %s (%d flagged blank)",
            len(payload), doc_id, flagged,
        )
        return flagged

    async def run_embeddings_only(self, job_id: str, doc_id: str) -> None:
        """Re-run the embedding steps for an already-ingested document.

        Clears existing visual embeddings first so re-embed with a different
        model (e.g., switching from ColPali to Nemotron) actually re-processes
        all pages instead of skipping them.
        """
        current_job_id.set(job_id)
        step = None
        try:
            # Gate BEFORE the destructive clear below — a queued re-embed
            # stopped or held here has not nulled anything yet.
            await self.jobs.checkpoint(job_id)
            await self.jobs.set_steps(job_id, ["embedding_text", "embedding_visual"])
            # Look up file_hash from Neo4j
            rows = await self.neo4j.run_query(
                "MATCH (d:Document {doc_id: $id}) RETURN d.file_hash AS h",
                {"id": doc_id},
            )
            if not rows:
                raise ValueError(f"Document {doc_id} not found")
            file_hash = rows[0]["h"]

            await self.jobs.update(job_id, status="processing", doc_id=doc_id, file_hash=file_hash)

            # Refuse to run at all when NEITHER embedding service is up —
            # and clear only the lanes whose service can actually re-fill
            # them. The old order (clear everything, then notice a service
            # is None and mark the step "skipped") destroyed the library's
            # embeddings under all-green completed jobs when a bulk
            # re-embed ran while ColPali/LM Studio was down.
            if self.text_embedding is None and self.colpali is None:
                raise RuntimeError(
                    "no embedding service is available — refusing to clear "
                    "existing embeddings (nothing could re-create them)"
                )
            clear_sets = []
            if self.colpali is not None:
                clear_sets += ["p.colpali_vectors = NULL",
                               "p.colpali_vector_count = NULL",
                               "p.colpali_vector_dim = NULL"]
            if self.text_embedding is not None:
                clear_sets += ["p.text_embedding = NULL"]

            # Clear existing embeddings so the new model re-processes them.
            # Without this, switching models (e.g. ColPali to Nemotron, or
            # nomic 768-d to bge-m3 1024-d) would skip all pages because
            # _embed_text filters on "text_embedding IS NULL" and
            # _embed_visual checks colpali_vector_count > 0.
            await self.neo4j.run_write(
                "MATCH (d:Document {doc_id: $doc_id})-[:HAS_PAGE]->(p:Page) "
                "SET " + ", ".join(clear_sets),
                {"doc_id": doc_id},
            )
            logger.info(
                "Cleared existing embeddings for doc %s (lanes: %s)",
                doc_id,
                ", ".join(
                    l for l, ok in
                    [("visual", self.colpali is not None),
                     ("text", self.text_embedding is not None)] if ok
                ),
            )

            # Ensure is_blank is populated before re-embedding so we don't
            # waste GPU cycles on pages that are visually empty.
            await self._backfill_blank_flags(doc_id, file_hash)

            if self.text_embedding is None:
                await self.jobs.update_step(
                    job_id, "embedding_text", "skipped",
                    detail="text embedding service not available",
                )
            else:
                step = "embedding_text"
                await self.jobs.update(
                    job_id, current_step="embedding_text", progress_pct=10.0
                )
                await self.jobs.update_step(job_id, "embedding_text", "running")
                n = await self._embed_text(job_id, doc_id)
                await self.jobs.update_step(
                    job_id, "embedding_text", "done", detail=f"{n} pages embedded"
                )

            if self.colpali is None:
                await self.jobs.update_step(
                    job_id, "embedding_visual", "skipped",
                    detail="visual embedding service not available",
                )
            else:
                step = "embedding_visual"
                await self.jobs.update(
                    job_id, current_step="embedding_visual", progress_pct=50.0
                )
                await self.jobs.update_step(job_id, "embedding_visual", "running")
                n = await self._embed_visual(job_id, doc_id, file_hash)
                await self.jobs.update_step(
                    job_id, "embedding_visual", "done", detail=f"{n} pages embedded"
                )

            await self.jobs.complete(job_id)
            logger.info("Reembed job %s completed for doc %s", job_id, doc_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Reembed job %s failed", job_id)
            if step is not None:
                await self.jobs.update_step(job_id, step, "error", detail=str(exc))
            await self.jobs.fail(job_id, str(exc))

    async def run_text_reembed_only(self, job_id: str, doc_id: str) -> None:
        """Re-run ONLY text embedding for an already-ingested document.

        Unlike run_embeddings_only() which clears both text and visual
        embeddings, this method:
          - Only clears p.text_embedding (leaves colpali_vectors untouched)
          - Only calls _embed_text() (no visual embedding, no entity extraction)

        Use this when only the text embedding model changed (e.g., switching
        from nomic 768-d to bge-m3 1024-d) to avoid hours of GPU time
        re-generating visual embeddings that haven't changed.
        """
        current_job_id.set(job_id)
        try:
            # Gate BEFORE the destructive clear below (see run_embeddings_only).
            await self.jobs.checkpoint(job_id)
            await self.jobs.set_steps(job_id, ["embedding_text"])
            # Look up file_hash from Neo4j
            rows = await self.neo4j.run_query(
                "MATCH (d:Document {doc_id: $id}) RETURN d.file_hash AS h",
                {"id": doc_id},
            )
            if not rows:
                raise ValueError(f"Document {doc_id} not found")
            file_hash = rows[0]["h"]

            await self.jobs.update(job_id, status="processing", doc_id=doc_id, file_hash=file_hash)

            # Check BEFORE the destructive clear — clearing first and then
            # failing would leave the doc with no text embeddings and no
            # service able to re-create them.
            if self.text_embedding is None:
                raise ValueError("Text embedding service not configured")

            # Clear ONLY text embeddings — leave visual embeddings intact
            await self.neo4j.run_write(
                """
                MATCH (d:Document {doc_id: $doc_id})-[:HAS_PAGE]->(p:Page)
                SET p.text_embedding = NULL
                """,
                {"doc_id": doc_id},
            )
            logger.info("Cleared text embeddings (visual untouched) for doc %s", doc_id)

            await self.jobs.update(
                job_id, current_step="embedding_text", progress_pct=10.0
            )
            await self.jobs.update_step(job_id, "embedding_text", "running")
            n = await self._embed_text(job_id, doc_id)
            await self.jobs.update_step(
                job_id, "embedding_text", "done", detail=f"{n} pages embedded"
            )

            await self.jobs.complete(job_id)
            logger.info("Text-only reembed job %s completed for doc %s", job_id, doc_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Text-only reembed job %s failed", job_id)
            await self.jobs.update_step(
                job_id, "embedding_text", "error", detail=str(exc)
            )
            await self.jobs.fail(job_id, str(exc))

    async def run_fill_missing(
        self, job_id: str, doc_id: str, *,
        do_text: bool = True,
        do_visual: bool = True,
        do_entities: bool = False,
        do_recover_text: bool = False,
    ) -> None:
        """Bounded entry point — see _run_fill_missing_inner for the work.

        The bulk drain endpoints queue one fill-missing job per doc (hundreds
        at once). Bounded by the same ingestion semaphore as uploads so they
        drain a few at a time instead of all at once — the unbounded version
        of this path is what overloaded the LLM server on 2026-08-06.
        """
        # Same double gate as run_job: hold queued jobs under pause-all, and
        # re-check when the semaphore slot finally frees.
        await self.jobs.checkpoint(job_id)
        async with self._ingest_semaphore:
            await self.jobs.checkpoint(job_id)
            await self._run_fill_missing_inner(
                job_id, doc_id,
                do_text=do_text, do_visual=do_visual,
                do_entities=do_entities, do_recover_text=do_recover_text,
            )

    async def run_fill_missing_now(
        self, job_id: str, doc_id: str, *,
        do_text: bool = True,
        do_visual: bool = True,
        do_entities: bool = False,
        do_recover_text: bool = False,
    ) -> None:
        """Priority ("run now") variant of run_fill_missing: skips the FIFO
        ingest queue so a user-triggered repair for a document they need
        RIGHT NOW doesn't wait behind a hundreds-deep drain backlog, and
        (via JobManager.exempt_from_pause, set by the caller) runs even
        while pause-all is on. Bounded by its own small semaphore; the LLM
        request cap and GPU manager bound the real resource load, same
        rationale as run_resummarize not taking the ingest semaphore.
        """
        async with self._priority_semaphore:
            await self.jobs.checkpoint(job_id)
            await self._run_fill_missing_inner(
                job_id, doc_id,
                do_text=do_text, do_visual=do_visual,
                do_entities=do_entities, do_recover_text=do_recover_text,
            )

    async def _run_fill_missing_inner(
        self, job_id: str, doc_id: str, *,
        do_text: bool = True,
        do_visual: bool = True,
        do_entities: bool = False,
        do_recover_text: bool = False,
    ) -> None:
        """Fill ONLY missing artifacts for an already-ingested document.

        Never clears existing work — the underlying steps are incremental
        (_embed_text filters text_embedding IS NULL, _embed_visual filters
        colpali_vector_count 0/NULL, _extract_entities skips pages with
        entity relationships), so pages that are already complete cost
        nothing. This is the repair path for completeness-audit gaps; use
        run_embeddings_only instead when the embedding MODEL changed and
        everything must be regenerated.

        do_recover_text copies Docling OCR text from a page's chunks onto
        Page.extracted_text for pages that have chunks but no text (scanned
        PDFs — PyMuPDF finds no text layer, but Docling OCRs the images
        during chunking). It runs first so the newly-texted pages are picked
        up by the embedding and extraction steps in the same job.
        """
        current_job_id.set(job_id)
        step = None
        plan = []
        if do_recover_text:
            plan.append("recovering_text")
        if do_text:
            plan.append("embedding_text")
        if do_visual:
            plan.append("embedding_visual")
        if do_entities:
            plan.append("extracting_entities")
        try:
            await self.jobs.set_steps(job_id, plan)
            rows = await self.neo4j.run_query(
                "MATCH (d:Document {doc_id: $id}) RETURN d.file_hash AS h",
                {"id": doc_id},
            )
            if not rows:
                raise ValueError(f"Document {doc_id} not found")
            file_hash = rows[0]["h"]
            await self.jobs.update(
                job_id, status="processing", doc_id=doc_id, file_hash=file_hash
            )
            if await self._page_count(doc_id) == 0:
                raise ValueError(
                    f"Document {doc_id} has 0 pages — it was only partially "
                    "ingested. Delete it and re-ingest the PDF; there is "
                    "nothing to fill."
                )

            if do_recover_text:
                step = "recovering_text"
                await self.jobs.update(
                    job_id, current_step="recovering_text", progress_pct=2.0
                )
                await self.jobs.update_step(job_id, "recovering_text", "running")
                n = await self._recover_page_text(doc_id)
                await self.jobs.update_step(
                    job_id, "recovering_text", "done",
                    detail=f"{n} pages recovered from chunk OCR text" if n
                    else "nothing to recover",
                )

            if do_text:
                if self.text_embedding is None:
                    await self.jobs.update_step(
                        job_id, "embedding_text", "skipped",
                        detail="text embedding service not available",
                    )
                else:
                    step = "embedding_text"
                    await self.jobs.update(
                        job_id, current_step="embedding_text", progress_pct=5.0
                    )
                    await self.jobs.update_step(job_id, "embedding_text", "running")
                    n = await self._embed_text(job_id, doc_id)
                    await self.jobs.update_step(
                        job_id, "embedding_text", "done",
                        detail=f"{n} missing pages embedded" if n
                        else "nothing missing",
                    )

            if do_visual:
                if self.colpali is None:
                    await self.jobs.update_step(
                        job_id, "embedding_visual", "skipped",
                        detail="visual embedding service not available",
                    )
                else:
                    step = "embedding_visual"
                    await self.jobs.update(
                        job_id, current_step="embedding_visual", progress_pct=40.0
                    )
                    await self.jobs.update_step(job_id, "embedding_visual", "running")
                    # Populate is_blank on old pages first so blank pages are
                    # excluded rather than embedded (or endlessly re-queued).
                    await self._backfill_blank_flags(doc_id, file_hash)
                    n = await self._embed_visual(job_id, doc_id, file_hash)
                    await self.jobs.update_step(
                        job_id, "embedding_visual", "done",
                        detail=f"{n} missing pages embedded" if n
                        else "nothing missing",
                    )

            if do_entities:
                if self.entity_extractor is None:
                    await self.jobs.update_step(
                        job_id, "extracting_entities", "skipped",
                        detail="LLM service not available",
                    )
                else:
                    step = "extracting_entities"
                    await self.jobs.update(
                        job_id, current_step="extracting_entities", progress_pct=70.0
                    )
                    await self.jobs.update_step(
                        job_id, "extracting_entities", "running"
                    )
                    done, failed, last_err = await self._extract_entities(
                        job_id, doc_id
                    )
                    if done and failed == done:
                        raise RuntimeError(
                            f"entity extraction failed for all {done} pages "
                            f"(last error: {last_err}) — pages remain unstamped "
                            "and will be retried on the next run"
                        )
                    if failed:
                        await self.jobs.update_step(
                            job_id, "extracting_entities", "warning",
                            detail=f"{failed} of {done} pages failed — see logs",
                        )
                    else:
                        await self.jobs.update_step(
                            job_id, "extracting_entities", "done",
                            detail=f"{done} pages extracted" if done
                            else "nothing missing",
                        )

            await self.jobs.complete(job_id)
            logger.info("Fill-missing job %s completed for doc %s", job_id, doc_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Fill-missing job %s failed", job_id)
            if step is not None:
                await self.jobs.update_step(job_id, step, "error", detail=str(exc))
            await self.jobs.fail(job_id, str(exc))

    async def _recover_page_text(self, doc_id: str) -> int:
        """Copy chunk text onto pages that have chunks but no extracted text.

        Scanned PDFs have no text layer, so PyMuPDF extraction leaves
        Page.extracted_text empty — but Docling OCRs the page images during
        chunking, so the real text exists on the :Chunk nodes. This copies
        it back (ordered by chunk_index) so keyword search, text embedding,
        and entity extraction work on scanned documents too.

        Only touches pages with text_char_count 0/NULL — never overwrites
        genuine PyMuPDF text. Returns the number of pages recovered.
        """
        rows = await self.neo4j.run_query(
            """
            MATCH (d:Document {doc_id: $id})-[:HAS_PAGE]->(p:Page)
            WHERE coalesce(p.text_char_count, 0) = 0
              AND EXISTS { (p)-[:HAS_CHUNK]->(:Chunk) }
            MATCH (p)-[:HAS_CHUNK]->(c:Chunk)
            WITH p, c ORDER BY c.chunk_index
            RETURN p.page_id AS page_id, collect(c.text) AS texts
            """,
            {"id": doc_id},
        )
        payload = []
        for r in rows:
            text = "\n\n".join(t for t in r["texts"] if t and t.strip())
            if text.strip():
                payload.append({"page_id": r["page_id"], "text": text})
        if not payload:
            return 0

        BATCH = 100
        for i in range(0, len(payload), BATCH):
            await self.neo4j.run_write(
                """
                UNWIND $rows AS row
                MATCH (p:Page {page_id: row.page_id})
                SET p.extracted_text = row.text,
                    p.text_char_count = size(row.text),
                    p.text_recovered_from_chunks = true
                """,
                {"rows": payload[i:i + BATCH]},
            )
        logger.info(
            "Recovered chunk OCR text onto %d pages of doc %s",
            len(payload), doc_id,
        )
        return len(payload)

    # ------------------------------------------------------------------ step 1

    async def _register(self, job, collection: str = "default") -> tuple[str, str, int]:
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
                    source_type: 'unknown',
                    collection: $collection
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
                    "collection": collection,
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

    async def _auto_tag(self, doc_id: str, current_collection: str) -> str:
        """Use the LLM to suggest collection, categories, and tags from page text.

        Returns a short human-readable summary for the step ledger."""
        assert self.auto_tagger is not None

        result = await self.auto_tagger.suggest_for_doc(self.neo4j, doc_id)
        if result is None:
            return "no suggestions from LLM"

        # Apply collection if suggested and currently default
        if result.collection and result.collection != "default" and current_collection == "default":
            await self.neo4j.run_write(
                "MATCH (d:Document {doc_id: $id}) SET d.collection = $col",
                {"id": doc_id, "col": result.collection},
            )

        # Apply categories
        for cat in result.categories:
            await self.neo4j.run_write(
                """
                MERGE (c:Category {name: $name})
                WITH c
                MATCH (d:Document {doc_id: $doc_id})
                MERGE (d)-[:IN_CATEGORY]->(c)
                """,
                {"name": cat, "doc_id": doc_id},
            )

        # Apply tags
        for tag in result.tags:
            await self.neo4j.run_write(
                """
                MERGE (t:Tag {name: $name})
                WITH t
                MATCH (d:Document {doc_id: $doc_id})
                MERGE (d)-[:TAGGED_WITH]->(t)
                """,
                {"name": tag, "doc_id": doc_id},
            )

        logger.info(
            "Auto-tagged doc %s: collection=%s, categories=%s, tags=%s",
            doc_id, result.collection, result.categories, result.tags,
        )
        return (
            f"collection={result.collection or current_collection}, "
            f"{len(result.categories)} categories, {len(result.tags)} tags"
        )

    async def _extract_text(
        self, job_id: str, source_path: str, doc_id: str, file_hash: str
    ) -> str:
        """Extract text per page and create :Page nodes linked to :Document.

        Skips extraction entirely if the document already has :Page nodes
        (the previous ingestion run already created them). Avoids creating
        duplicate Pages on resume after a failed ColPali / entity-extraction
        step.

        Returns a short summary string for the step ledger.
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
            return f"reused {existing_count} existing pages"

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
        # Flag visually-blank pages so downstream visual embedding can skip them.
        now_iso = datetime.now(timezone.utc).isoformat()
        pages_params = []
        blank_count = 0
        for p in extraction.pages:
            page_id = str(uuid.uuid4())
            reduced_path_obj = self.pdf_processor.reduced_image_path(
                file_hash, p.page_number
            )
            image_path = str(
                self.pdf_processor.page_image_path(file_hash, p.page_number)
            )
            reduced_path = str(reduced_path_obj)
            is_blank = await asyncio.to_thread(
                _is_blank_page, reduced_path_obj, p.char_count
            )
            if is_blank:
                blank_count += 1
            pages_params.append({
                "page_id": page_id,
                "page_number": p.page_number,
                "image_path": image_path,
                "reduced_image_path": reduced_path,
                "extracted_text": p.text,
                "text_char_count": p.char_count,
                "source_type": p.source_type,
                "is_blank": is_blank,
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
                    source_type: page.source_type,
                    is_blank: page.is_blank
                })
                CREATE (d)-[:HAS_PAGE {page_number: page.page_number}]->(p)
                """,
                {"doc_id": doc_id, "pages": pages_params, "now": now_iso},
            )
        if blank_count:
            logger.info(
                "Flagged %d blank page(s) in doc %s (will skip visual embedding)",
                blank_count, doc_id,
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
        detail = f"{extraction.page_count} pages ({extraction.document_source_type})"
        if blank_count:
            detail += f", {blank_count} blank"
        return detail

    # ------------------------------------------------------------------ step 4

    async def _embed_text(self, job_id: str, doc_id: str) -> int:
        """Embed page texts and store on Page.text_embedding (Neo4j vector index).

        Returns the number of pages embedded (0 if all were already done)."""
        assert self.text_embedding is not None
        assert self.gpu is not None

        # Pull pages that have text and no embedding yet (shared predicate —
        # deep verify's repair_coverage check compares the audit against it)
        pages = await self.neo4j.run_query(
            f"""
            MATCH (d:Document {{doc_id: $doc_id}})-[:HAS_PAGE]->(p:Page)
            WHERE {TEXT_EMBED_MISSING}
            RETURN p.page_id AS page_id, p.extracted_text AS text
            ORDER BY p.page_number
            """,
            {"doc_id": doc_id},
        )
        if not pages:
            logger.info("No pages need text embedding for doc %s", doc_id)
            return 0

        total = len(pages)
        batch_size = self.settings.ingestion.text_embedding_batch_size

        # Embed in batches. The GPU scope is taken per batch (not around the
        # whole loop) so a job paused at the checkpoint below never holds
        # the GPU semaphore — search-time embedding/reranking stays live and
        # the idle watcher can actually unload the model. The model itself
        # stays cached across batches by the GPU manager.
        for start in range(0, total, batch_size):
            await self.jobs.checkpoint(job_id)
            batch = pages[start:start + batch_size]
            texts = [row["text"] for row in batch]
            ids = [row["page_id"] for row in batch]
            await self.jobs.update(
                job_id,
                current_item=f"text-embedding pages "
                f"{start + 1}–{min(start + batch_size, total)} of {total}",
            )

            async with self.gpu.load_scope("text_embedding"):
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
        return total

    # ------------------------------------------------------------------ step 5

    async def _embed_visual(self, job_id: str, doc_id: str, file_hash: str) -> int:
        """Generate visual embeddings for every page and store as bytes on Page.

        Works with both Nemotron ColEmbed and ColPali — both implement
        embed_images() returning list[np.ndarray] of (K, D) float32.

        Returns the number of pages embedded (0 if all were already done).
        """
        assert self.colpali is not None
        assert self.gpu is not None

        # Find all pages that don't yet have visual embeddings, excluding
        # pages flagged as visually blank (no signal for ColPali/Nemotron).
        rows = await self.neo4j.run_query(
            f"""
            MATCH (d:Document {{doc_id: $doc_id}})-[:HAS_PAGE]->(p:Page)
            WHERE {VISUAL_EMBED_MISSING}
            RETURN p.page_id AS page_id, p.page_number AS page_number
            ORDER BY p.page_number
            """,
            {"doc_id": doc_id},
        )
        if not rows:
            logger.info("No pages need visual embedding for doc %s", doc_id)
            return 0

        total = len(rows)
        batch_size = self.settings.ingestion.colpali_batch_size

        # Use the appropriate GPU scope name based on model type
        scope_name = "visual_embed" if isinstance(self.colpali, NemotronService) else "colpali"

        # Per-batch GPU scope, same rationale as _embed_text: a paused job
        # must never hold the GPU semaphore.
        for start in range(0, total, batch_size):
            await self.jobs.checkpoint(job_id)
            batch = rows[start:start + batch_size]
            image_paths = [
                self.pdf_processor.page_image_path(file_hash, r["page_number"])
                for r in batch
            ]
            page_ids = [r["page_id"] for r in batch]
            await self.jobs.update(
                job_id,
                current_item=f"visual-embedding pages "
                f"{start + 1}–{min(start + batch_size, total)} of {total}",
            )

            async with self.gpu.load_scope(scope_name):
                # Both models return list of (K, D) float32 arrays
                embeddings = await asyncio.to_thread(
                    self.colpali.embed_images, image_paths
                )

            # Serialize — format is identical for both models
            _serialize = serialize_nemotron if isinstance(self.colpali, NemotronService) else serialize_colpali
            payload = []
            for pid, arr in zip(page_ids, embeddings, strict=True):
                blob, k = _serialize(arr)
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
        return total

    # ------------------------------------------------------------------ step 6

    async def _stamp_page_extracted(
        self, page_id: str, counts: dict[str, int], char_count: int | None,
    ) -> None:
        """Stamp a page entities_extracted_at with the confirmed-empty
        protocol — THE single stamp writer for every extraction lane.

        entities_confirmed_empty means: the post-fix extractor ran this
        dense page and wrote no entity mentions; the suspicious-empty check
        must not re-flag it and drains must not re-pay it. Gate on the
        per-type entity counts (counts["page_rels"] tallies the model's
        explicit relationship list — wrong signal), density on the stored
        text_char_count (the SAME property the predicate compares), and
        always write the marker so a successful extraction CLEARS a stale
        one. Every rule here is a scar from a live incident; do not fork
        this logic into per-lane copies again.
        """
        wrote_entities = any(
            counts.get(k, 0) > 0
            for k in ("materials", "processes", "standards",
                      "clauses", "equipment")
        )
        confirmed_empty = (
            not wrote_entities
            and (char_count or 0) >= SUSPICIOUS_EMPTY_MIN_CHARS
        )
        await self.neo4j.run_write(
            "MATCH (p:Page {page_id: $pid}) "
            "SET p.entities_extracted_at = datetime(), "
            "p.entities_confirmed_empty = "
            + ("true" if confirmed_empty else "null"),
            {"pid": page_id},
        )

    async def _extract_entities(
        self, job_id: str, doc_id: str
    ) -> tuple[int, int, str | None]:
        """Run LLM entity extraction on each page and write results into the graph.

        I/O-bound on the LLM endpoint. Sequential per page (local LLM
        serves one request at a time). Skips pages that already have any
        MENTIONS_* outgoing relationship so re-runs after a partial failure
        don't double-count support_count on existing edges.

        Returns (pages_processed, pages_failed, last_error) so callers can
        surface partial failures — with the actual reason, not a guess —
        instead of reporting a clean run.
        """
        assert self.entity_extractor is not None

        # Pull title + pages that have text AND haven't had entities extracted yet.
        # We detect "already extracted" as having any of the page-level entity
        # relationships — if extraction ran on this page it wrote at least one,
        # unless the page was empty of entities. For safety we also accept pages
        # that still genuinely have nothing relevant; those pages just get
        # re-run (fast path since the LLM returns empty arrays).
        rows = await self.neo4j.run_query(
            f"""
            MATCH (d:Document {{doc_id: $doc_id}})
            OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Page)
            WHERE {ENTITY_NEEDS_EXTRACTION}
            RETURN d.title AS title,
                   collect({{page_id: p.page_id, page_number: p.page_number,
                             text: p.extracted_text,
                             char_count: p.text_char_count}}) AS pages
            """,
            {"doc_id": doc_id},
        )
        if not rows:
            logger.warning("No document %s found for entity extraction", doc_id)
            return 0, 0, None
        title = rows[0]["title"] or "(untitled)"
        pages = [p for p in rows[0]["pages"] if p["page_id"] is not None]

        if not pages:
            logger.info("Document %s has no pages with text — skipping extraction", doc_id)
            return 0, 0, None

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
        failed = 0
        last_error: str | None = None
        aggregate = {"materials": 0, "processes": 0, "standards": 0,
                     "clauses": 0, "equipment": 0,
                     "page_rels": 0, "entity_rels": 0}
        for page in pages:
            await self.jobs.checkpoint(job_id)
            await self.jobs.update(
                job_id,
                current_item=f"page {page['page_number']} "
                f"({done + 1}/{total}) — {title[:80]}",
            )
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
                # Stamp the page as extracted even when zero entities were
                # found — otherwise "ran and found nothing" is
                # indistinguishable from "never ran", the completeness audit
                # reports it as a gap forever, and repair jobs re-pay the
                # LLM for the same empty pages on every run.
                await self._stamp_page_extracted(
                    page["page_id"], counts, page.get("char_count"),
                )
            except Exception as exc:  # noqa: BLE001
                failed += 1
                last_error = str(exc)
                logger.warning("Entity extraction failed for page %d: %s",
                               page["page_number"], exc)

            done += 1
            # Extraction spans 88% -> 99% in full runs, 10% -> 95% in extract-only
            progress = min(99.0, 88.0 + 11.0 * done / total)
            await self.jobs.update(
                job_id, progress_pct=progress, pages_processed=done
            )

        logger.info("Entity extraction complete for doc %s: %s (%d/%d pages failed)",
                    doc_id, aggregate, failed, total)
        return done, failed, last_error

    # --------------------------------------------------------- chunk building

    def _rasterized_pdf_from_page_images(self, file_hash: str) -> str | None:
        """Build a temp image-PDF from the doc's already-rendered page images.

        Used as the chunking fallback for text-less PDFs (vector-outline
        exports and anything else Docling parses to nothing). Pages are
        appended one at a time (PIL append=True) so memory stays at one
        decoded page regardless of book size; each page is downscaled to
        OCR-sufficient resolution first. Returns None when no rendered
        images exist (conversion failed or never ran) — caller falls back
        to the plain no-chunks path.
        """
        from PIL import Image

        img_dir = self.pdf_processor.doc_folder(file_hash)
        pages = sorted(img_dir.glob("page_*.png")) if img_dir.is_dir() else []
        if not pages:
            return None
        # Positions in the rebuilt PDF become the chunker's page numbers, so
        # the image set must be exactly pages 1..N with no gaps — a missing
        # render would shift every later page and silently attribute OCR
        # text to the wrong page_number. (Also guards the 4-digit padding:
        # lexical sort breaks past page_9999.)
        expected = [img_dir / f"page_{i:04d}.png" for i in range(1, len(pages) + 1)]
        if pages != expected:
            logger.warning(
                "Rasterized-rebuild fallback skipped for %s: page images "
                "are not a contiguous 1..%d sequence — a rebuilt PDF would "
                "misnumber pages", file_hash, len(pages),
            )
            return None

        fd, out_path = tempfile.mkstemp(suffix=".pdf", prefix="rechunk_")
        os.close(fd)
        try:
            first = True
            for p in pages:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    # 1800px on the long edge ≈ 200+ DPI for letter-size —
                    # comfortably above what Docling's OCR needs, well below
                    # the full-resolution ColPali renders.
                    im.thumbnail((1800, 1800))
                    im.save(
                        out_path, "PDF",
                        append=not first,
                        resolution=200,
                    )
                first = False
        except Exception:
            Path(out_path).unlink(missing_ok=True)
            raise
        return out_path

    async def _build_chunks(
        self, job_id: str, doc_id: str, file_hash: str, pdf_path: str,
    ) -> dict[str, int]:
        """Parse the PDF into structural chunks, summarize, embed, write.

        Idempotent per (doc_hash, chunk_id) — re-running on an already-chunked
        doc updates chunks in place instead of duplicating them. Safe to rerun
        after a partial failure.
        """
        # Sub-phases surface via current_step (and the running step's detail)
        # so the UI shows chunking/summarizing/embedding/writing live instead
        # of one opaque "building_chunks" for the whole stretch. The ledger
        # keeps them under the single "building_chunks" dot.

        # 1. Docling pass. Runs off the asyncio loop because it's CPU-bound.
        await self.jobs.checkpoint(job_id)
        await self.jobs.update(job_id, current_step="chunking")
        await self.jobs.update_step(
            job_id, "building_chunks", "running", detail="chunking (Docling parse)"
        )
        chunks: list[StructuralChunk] = await asyncio.to_thread(
            self.structural_chunker.chunk_pdf, pdf_path, file_hash,
        )
        if not chunks:
            # Text-less PDFs slip between both extraction strategies:
            # vector-outline exports (Illustrator with fonts converted to
            # curves — zero fonts in the file) LOOK born-digital so Docling
            # trusts the (empty) text layer instead of OCRing, yet render
            # perfectly. Rebuild a rasterized PDF from the page images we
            # already rendered and retry — Docling then sees an honest
            # scanned doc and OCRs it like any historical book. This is the
            # remedy that fixed the live Camplux manual, automated.
            rebuilt = await asyncio.to_thread(
                self._rasterized_pdf_from_page_images, file_hash,
            )
            if rebuilt is not None:
                logger.warning(
                    "Chunker returned no chunks for %s — retrying OCR on a "
                    "rasterized rebuild from the rendered page images",
                    pdf_path,
                )
                await self.jobs.update_step(
                    job_id, "building_chunks", "running",
                    detail="no chunks from original PDF (text-less?) — "
                    "retrying OCR on rasterized rebuild",
                )
                try:
                    chunks = await asyncio.to_thread(
                        self.structural_chunker.chunk_pdf, rebuilt, file_hash,
                    )
                finally:
                    Path(rebuilt).unlink(missing_ok=True)
        if not chunks:
            logger.warning("Chunker returned no chunks for %s", pdf_path)
            return {"chunks": 0}

        logger.info(
            "Chunker produced %d chunks for doc %s; summarizing...",
            len(chunks), doc_id,
        )

        # 2. Summaries — bounded-concurrency LLM calls. Short chunks skip
        # the LLM and reuse their text (see ChunkSummarizer).
        assert self.chunk_summarizer is not None
        await self.jobs.checkpoint(job_id)
        await self.jobs.update(job_id, current_step="summarizing", progress_pct=70.0)
        await self.jobs.update_step(
            job_id, "building_chunks", "running",
            detail=f"summarizing {len(chunks)} chunks via LLM",
        )
        summarized = await self.chunk_summarizer.summarize_batch(chunks, concurrency=4)
        summaries = [s for s, _src in summarized]
        sources = [src for _s, src in summarized]
        preview_count = sum(1 for src in sources if src == "preview")
        if preview_count:
            logger.warning(
                "%d of %d chunk summaries fell back to text previews for doc %s "
                "(LLM failures) — marked summary_source='preview' for the "
                "resummarize repair",
                preview_count, len(chunks), doc_id,
            )

        # 3. Embed (summary + text concatenated produces the retrieval vector).
        # We embed the pair so a match on either the raw text or the summary
        # contributes to the dense score. BM25 runs on both fields
        # independently via the chunk_text_fulltext index.
        assert self.text_embedding is not None
        assert self.gpu is not None
        await self.jobs.checkpoint(job_id)
        await self.jobs.update(job_id, current_step="embedding_chunks", progress_pct=72.0)
        await self.jobs.update_step(
            job_id, "building_chunks", "running",
            detail=f"embedding {len(chunks)} chunk summaries",
        )
        embed_inputs = [
            f"{s}\n\n{c.text[:2000]}" for s, c in zip(summaries, chunks)
        ]
        async with self.gpu.load_scope("text_embedding"):
            vectors = await asyncio.to_thread(
                self.text_embedding.embed_documents, embed_inputs,
                batch_size=self.settings.ingestion.text_embedding_batch_size,
            )

        # 4. Map each chunk to its Page node via (file_hash, page_number).
        # Write in batches to keep Neo4j write latency low.
        await self.jobs.update(job_id, current_step="writing_chunks", progress_pct=74.0)
        await self.jobs.update_step(
            job_id, "building_chunks", "running", detail="writing chunks to graph"
        )
        BATCH = 200
        total_written = 0
        for i in range(0, len(chunks), BATCH):
            await self.jobs.checkpoint(job_id)
            batch = chunks[i : i + BATCH]
            rows = []
            for ch, summ, src, vec in zip(
                batch, summaries[i : i + BATCH], sources[i : i + BATCH],
                vectors[i : i + BATCH]
            ):
                rows.append({
                    "chunk_id": ch.chunk_id,
                    "page_number": ch.page_number,
                    "chunk_index": ch.chunk_index,
                    "chunk_type": ch.chunk_type,
                    "text": ch.text,
                    "summary": summ,
                    "summary_source": src,
                    "section_path": ch.section_path,
                    "embedding": vec.tolist(),
                    "bbox": list(ch.bbox) if ch.bbox is not None else None,
                })
            await self.neo4j.run_write(
                """
                UNWIND $rows AS row
                MATCH (d:Document {doc_id: $doc_id})-[:HAS_PAGE]->(p:Page {page_number: row.page_number})
                MERGE (c:Chunk {chunk_id: row.chunk_id})
                ON CREATE SET c.page_number = row.page_number,
                              c.chunk_index = row.chunk_index,
                              c.chunk_type = row.chunk_type,
                              c.text = row.text,
                              c.summary = row.summary,
                              c.summary_source = row.summary_source,
                              c.section_path = row.section_path,
                              c.embedding = row.embedding,
                              c.bbox = row.bbox,
                              c.doc_id = $doc_id
                ON MATCH SET  c.text = row.text,
                              c.summary = row.summary,
                              c.summary_source = row.summary_source,
                              c.section_path = row.section_path,
                              c.chunk_type = row.chunk_type,
                              c.embedding = row.embedding,
                              c.bbox = row.bbox
                MERGE (p)-[:HAS_CHUNK]->(c)
                RETURN count(c) AS written
                """,
                {"doc_id": doc_id, "rows": rows},
            )
            # Count what the MATCH actually let through, not what we sent:
            # a row whose page_number has no :Page node is silently filtered
            # by Cypher — counting len(rows) reported dropped chunks as
            # written and stamped the shrunken coverage as final.
            written_rows = await self.neo4j.run_query(
                "MATCH (c:Chunk {doc_id: $doc_id}) "
                "WHERE c.chunk_id IN $ids RETURN count(c) AS n",
                {"doc_id": doc_id, "ids": [r["chunk_id"] for r in rows]},
            )
            total_written += written_rows[0]["n"] if written_rows else 0

        dropped = len(chunks) - total_written
        if dropped > 0:
            logger.warning(
                "%d of %d chunk rows were DROPPED for doc %s — their "
                "page_number matched no :Page node (page-count skew?). "
                "chunks_built_at NOT stamped so the audit keeps the doc "
                "visible.", dropped, len(chunks), doc_id,
            )
            await self.jobs.update_step(
                job_id, "building_chunks", "warning",
                detail=f"{dropped} of {len(chunks)} chunk rows dropped — "
                "no matching page node",
            )
        elif total_written:
            await self.neo4j.run_write(
                "MATCH (d:Document {doc_id: $id}) "
                "SET d.chunks_built_at = datetime()",
                {"id": doc_id},
            )
        logger.info(
            "Wrote %d chunks for doc %s (types: %s)",
            total_written, doc_id,
            dict(_Counter(c.chunk_type for c in chunks)),
        )
        return {"chunks": total_written, "preview_summaries": preview_count}

    # --------------------------------------------------------- post-ingestion dedup

    async def _dedup_doc_entities(self, doc_id: str) -> int:
        """Merge near-duplicate entities linked to a specific document.

        After entity extraction, the same real-world entity often appears
        as separate nodes due to casing, hyphens, or plural differences
        (e.g., "Stainless Steel" vs "stainless steel"). This method finds
        entities linked to the just-ingested document and merges them with
        existing nodes that normalize to the same form.

        Returns the number of entities merged.
        """
        import re as _re
        from difflib import SequenceMatcher as _SM

        _STRIP = _re.compile(r"[®©™°\-–—\s]+")

        def _norm(name: str) -> str:
            return _STRIP.sub("", name).lower()

        # Fetch entities linked to this document's pages
        rel_types = [
            ("MENTIONS_MATERIAL", "Material", "name"),
            ("DESCRIBES_PROCESS", "Process", "name"),
            ("REFERENCES_STANDARD", "Standard", "code"),
            ("MENTIONS_EQUIPMENT", "Equipment", "name"),
        ]

        total_merged = 0
        for rel, label, pk in rel_types:
            # Get entity names linked to this doc
            doc_entities = await self.neo4j.run_query(
                f"""
                MATCH (d:Document {{doc_id: $doc_id}})-[:HAS_PAGE]->(p:Page)-[:{rel}]->(e:{label})
                RETURN DISTINCT e.{pk} AS name
                """,
                {"doc_id": doc_id},
            )
            if not doc_entities:
                continue

            doc_names = {r["name"] for r in doc_entities if r["name"]}
            if not doc_names:
                continue

            # For each doc entity, check if there's an existing entity with
            # a different name that normalizes to the same form, or is very
            # similar (>= 0.92 threshold, high to avoid false merges).
            for name in doc_names:
                norm = _norm(name)
                if len(norm) < 3:
                    continue

                # Find candidates: same label, different name, similar normalized form
                candidates = await self.neo4j.run_query(
                    f"""
                    MATCH (e:{label})
                    WHERE e.{pk} <> $name
                      AND e.{pk} IS NOT NULL
                      AND toLower(e.{pk}) CONTAINS $prefix
                    OPTIONAL MATCH (p:Page)-[:{rel}]->(e)
                    RETURN e.{pk} AS name, count(DISTINCT p) AS mentions
                    LIMIT 20
                    """,
                    {"name": name, "prefix": norm[:4] if len(norm) >= 4 else norm[:3]},
                )
                if not candidates:
                    continue

                for cand in candidates:
                    cand_name = cand["name"]
                    cand_norm = _norm(cand_name)

                    # Check similarity
                    if cand_norm == norm:
                        sim = 1.0
                    elif _SM(None, norm, cand_norm).ratio() >= 0.92:
                        sim = _SM(None, norm, cand_norm).ratio()
                    else:
                        continue

                    # Safety: different numbers → skip
                    nums_a = set(_re.findall(r"\d{2,}", name))
                    nums_b = set(_re.findall(r"\d{2,}", cand_name))
                    if nums_a and nums_b and nums_a != nums_b:
                        continue

                    # Pick winner by mention count
                    doc_ent_mentions = await self.neo4j.run_query(
                        f"MATCH (p:Page)-[:{rel}]->(e:{label} {{{pk}: $name}}) RETURN count(p) AS c",
                        {"name": name},
                    )
                    my_mentions = doc_ent_mentions[0]["c"] if doc_ent_mentions else 0

                    if cand["mentions"] >= my_mentions:
                        winner_name, loser_name = cand_name, name
                    else:
                        winner_name, loser_name = name, cand_name

                    # Redirect relationships and delete the loser via the
                    # shared per-type merge (backend/services/entity_merge).
                    await merge_entity(
                        self.neo4j, label, pk, winner_name, loser_name
                    )
                    total_merged += 1
                    logger.debug(
                        "Dedup: merged %s %r into %r (sim=%.2f)",
                        label, loser_name, winner_name, sim,
                    )
                    break  # one merge per doc entity per pass

        return total_merged
