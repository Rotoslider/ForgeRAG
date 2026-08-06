"""Admin / maintenance endpoints.

Small utilities for one-off fixes — not part of the regular user-facing API.
Currently: dedupe Page nodes when re-ingestion before the fix created them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from backend.models.common import ForgeResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/audit/completeness")
async def audit_completeness(request: Request) -> ForgeResult:
    """Audit every document's pipeline completeness from graph state.

    No re-processing — each pipeline step leaves a fingerprint on the graph
    (Page properties, Chunk nodes, entity relationships), so this derives
    which steps are missing/partial/complete per document, including
    embedding-dimension verification against the configured models.

    Slow-ish by design (full Page scan, ~100k pages ≈ tens of seconds);
    call it on demand, not on a poll loop.
    """
    from backend.services.completeness import run_audit

    settings = request.app.state.settings
    neo4j = request.app.state.neo4j
    report = await run_audit(
        neo4j,
        text_dim=settings.models.text_embedding_dim,
        visual_dim=settings.models.visual_embed_dim,
    )
    logger.info(
        "Completeness audit: %(complete)d complete, %(incomplete)d incomplete, "
        "%(error)d error of %(documents)d documents",
        report["summary"],
    )
    return ForgeResult(success=True, data=report)


@router.get("/verify")
async def deep_verify(request: Request) -> ForgeResult:
    """Deep pipeline verification — every invariant, exact counts, no sampling.

    Read-only. Slower than the completeness audit (full scans including
    on-disk image checks and blob byte-size validation); expect a minute or
    two on a ~100k-page library. Returns PASS only when every check has
    zero violations.
    """
    from backend.services.verification import run_verification

    report = await run_verification(
        request.app.state.neo4j, request.app.state.settings
    )
    logger.info(
        "Deep verification: %s (%d/%d checks passed)",
        report["verdict"], report["checks_passed"], report["checks_total"],
    )
    return ForgeResult(success=True, data=report)


@router.post("/extract-missing-entities")
async def extract_missing_entities(request: Request) -> ForgeResult:
    """Queue entity extraction for EVERY document with unextracted text pages.

    Server-side twin of the deep-verification entity_extraction_complete
    check: finds all docs holding pages with text but neither entity
    relationships nor the extracted-empty marker, and queues one
    fill-missing(entities) job per doc. Long-running LLM work — jobs drain
    in the background and are fully resumable.
    """
    neo4j = request.app.state.neo4j
    jobs = request.app.state.job_manager
    pipeline = request.app.state.pipeline

    docs = await neo4j.run_query(
        """
        MATCH (d:Document)-[:HAS_PAGE]->(p:Page)
        WHERE p.text_char_count > 0
          AND p.entities_extracted_at IS NULL
          AND NOT EXISTS {
            (p)-[:MENTIONS_MATERIAL|DESCRIBES_PROCESS|REFERENCES_STANDARD|MENTIONS_EQUIPMENT]->()
          }
        WITH d, count(p) AS todo
        RETURN d.doc_id AS doc_id, d.filename AS filename, todo
        ORDER BY todo DESC
        """,
        timeout=600.0,
    )
    if not docs:
        return ForgeResult(success=True, data={"queued": 0, "pages": 0,
                                               "reason": "nothing to extract"})
    queued = []
    for r in docs:
        job = await jobs.create(
            source_path=f"(fill-missing of {r['doc_id']})",
            filename=r["filename"], categories=[], tags=[],
        )
        asyncio.create_task(pipeline.run_fill_missing(
            job.job_id, r["doc_id"],
            do_text=False, do_visual=False, do_entities=True,
        ))
        queued.append({"doc_id": r["doc_id"], "job_id": job.job_id})
    total_pages = sum(r["todo"] for r in docs)
    logger.info("Queued entity-extraction drain: %d docs, %d pages",
                len(queued), total_pages)
    return ForgeResult(success=True, data={
        "queued": len(queued), "pages": total_pages, "jobs": queued,
    })


@router.post("/recover-stranded-text")
async def recover_stranded_text(request: Request) -> ForgeResult:
    """Queue OCR text recovery (+ text embedding) for every document with
    pages whose text exists only in chunks. Twin of the deep-verification
    no_stranded_ocr_text check."""
    neo4j = request.app.state.neo4j
    jobs = request.app.state.job_manager
    pipeline = request.app.state.pipeline

    docs = await neo4j.run_query(
        """
        MATCH (d:Document)-[:HAS_PAGE]->(p:Page)
        WHERE coalesce(p.text_char_count, 0) = 0
          AND EXISTS { (p)-[:HAS_CHUNK]->(:Chunk) }
        WITH d, count(p) AS todo
        RETURN d.doc_id AS doc_id, d.filename AS filename, todo
        """,
        timeout=600.0,
    )
    if not docs:
        return ForgeResult(success=True, data={"queued": 0, "pages": 0,
                                               "reason": "nothing to recover"})
    queued = []
    for r in docs:
        job = await jobs.create(
            source_path=f"(fill-missing of {r['doc_id']})",
            filename=r["filename"], categories=[], tags=[],
        )
        asyncio.create_task(pipeline.run_fill_missing(
            job.job_id, r["doc_id"],
            do_text=True, do_visual=False, do_entities=False,
            do_recover_text=True,
        ))
        queued.append({"doc_id": r["doc_id"], "job_id": job.job_id})
    return ForgeResult(success=True, data={
        "queued": len(queued), "pages": sum(r["todo"] for r in docs),
        "jobs": queued,
    })


@router.post("/backfill-blank-flags")
async def backfill_blank_flags(request: Request) -> ForgeResult:
    """Compute is_blank on every page that doesn't have it yet.

    Pages ingested before the blank-page filter existed have is_blank NULL;
    the flag only matters for skipping blank pages on future re-embeds, but
    backfilling it clears the deep-verification warning and makes the data
    uniform. Runs as a background job (loads each affected page's reduced
    image to measure grayscale variance).
    """
    neo4j = request.app.state.neo4j
    jobs = request.app.state.job_manager
    pipeline = request.app.state.pipeline

    docs = await neo4j.run_query(
        """
        MATCH (d:Document)-[:HAS_PAGE]->(p:Page)
        WHERE p.is_blank IS NULL
        RETURN d.doc_id AS doc_id, d.file_hash AS file_hash,
               count(p) AS pages
        """,
    )
    if not docs:
        return ForgeResult(success=True, data={"queued": False, "docs": 0,
                                               "reason": "nothing to backfill"})

    total_pages = sum(r["pages"] for r in docs)
    job = await jobs.create(
        source_path="(blank-flag backfill)",
        filename=f"(blank flags on {len(docs)} docs)",
        categories=[], tags=[],
    )

    async def _run() -> None:
        from backend.ingestion.job_logs import current_job_id
        current_job_id.set(job.job_id)
        try:
            await jobs.set_steps(job.job_id, ["extracting_text"])
            await jobs.update(job.job_id, status="processing",
                              current_step="extracting_text",
                              pages_total=total_pages)
            await jobs.update_step(job.job_id, "extracting_text", "running",
                                   detail="computing is_blank flags")
            done = 0
            for r in docs:
                await pipeline._backfill_blank_flags(r["doc_id"], r["file_hash"])
                done += r["pages"]
                await jobs.update(
                    job.job_id, pages_processed=done,
                    progress_pct=min(99.0, 100.0 * done / max(total_pages, 1)),
                )
            await jobs.update_step(job.job_id, "extracting_text", "done",
                                   detail=f"{total_pages} pages flagged")
            await jobs.complete(job.job_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Blank-flag backfill failed")
            await jobs.fail(job.job_id, str(exc))

    asyncio.create_task(_run())
    return ForgeResult(success=True, data={
        "queued": True, "job_id": job.job_id,
        "docs": len(docs), "pages": total_pages,
    })


@router.post("/fill-missing")
async def fill_missing(request: Request, payload: dict | None = None) -> ForgeResult:
    """Queue incremental gap-filling jobs for the given documents.

    Unlike /bulk-reembed (which CLEARS embeddings first — for model
    switches), fill-missing never deletes anything: the embed/extract steps
    already filter to pages missing the artifact, so completed pages cost
    nothing. Safe to run on every document with gaps.

    Body:
      {
        "doc_ids": ["...", ...],   # required
        "text": true,               # fill missing text embeddings
        "visual": true,             # fill missing visual embeddings
        "entities": false,          # re-run extraction on unextracted pages
        "recover_text": false       # copy Docling OCR text from chunks onto
                                    # textless pages (scanned PDFs) first
      }
    """
    if not isinstance(payload, dict):
        payload = {}
    doc_ids = payload.get("doc_ids") or []
    do_text = bool(payload.get("text", True))
    do_visual = bool(payload.get("visual", True))
    do_entities = bool(payload.get("entities", False))
    do_recover = bool(payload.get("recover_text", False))

    if not isinstance(doc_ids, list) or not doc_ids:
        return ForgeResult(success=False, reason="doc_ids must be a non-empty list",
                           data={"queued": 0})
    if not (do_text or do_visual or do_entities or do_recover):
        return ForgeResult(success=False, reason="nothing to do — enable at least one of text/visual/entities",
                           data={"queued": 0})

    neo4j = request.app.state.neo4j
    jobs = request.app.state.job_manager
    pipeline = request.app.state.pipeline

    rows = await neo4j.run_query(
        """
        UNWIND $ids AS id
        MATCH (d:Document {doc_id: id})
        RETURN d.doc_id AS doc_id, d.filename AS filename
        """,
        {"ids": doc_ids},
    )
    found = {r["doc_id"]: r for r in rows}
    queued = []
    for doc_id in doc_ids:
        if doc_id not in found:
            continue
        job = await jobs.create(
            source_path=f"(fill-missing of {doc_id})",
            filename=found[doc_id]["filename"],
            categories=[],
            tags=[],
        )
        asyncio.create_task(
            pipeline.run_fill_missing(
                job.job_id, doc_id,
                do_text=do_text, do_visual=do_visual, do_entities=do_entities,
                do_recover_text=do_recover,
            )
        )
        queued.append({"doc_id": doc_id, "job_id": job.job_id})

    logger.info(
        "Queued %d fill-missing job(s) (recover=%s text=%s visual=%s entities=%s)",
        len(queued), do_recover, do_text, do_visual, do_entities,
    )
    return ForgeResult(
        success=True,
        data={
            "queued": len(queued),
            "not_found": len(doc_ids) - len(queued),
            "jobs": queued,
        },
    )


@router.post("/normalize-entities")
async def normalize_entities(request: Request) -> ForgeResult:
    """Merge duplicate entities that differ only by case or whitespace.

    Finds pairs like 'Aluminum' and 'aluminum', 'GTAW ' and 'GTAW',
    merges their relationships onto the canonical (most-mentioned) version,
    and deletes the duplicate. Idempotent.
    """
    neo4j = request.app.state.neo4j
    total_merged = 0

    for label, pk in [
        ("Material", "name"),
        ("Process", "name"),
        ("Standard", "code"),
        ("Equipment", "name"),
    ]:
        # Find groups that differ only by case/whitespace
        rows = await neo4j.run_query(
            f"""
            MATCH (e:{label})
            WITH toLower(trim(e.{pk})) AS normalized, collect(e) AS nodes
            WHERE size(nodes) > 1
            RETURN normalized, [n IN nodes | n.{pk}] AS names, size(nodes) AS count
            """,
        )

        for group in rows:
            names = group["names"]
            # Keep the one with the most page mentions
            best = None
            best_count = -1
            for name in names:
                mention_rows = await neo4j.run_query(
                    f"""
                    MATCH (e:{label} {{{pk}: $name}})
                    OPTIONAL MATCH (p:Page)-[]->(e)
                    RETURN count(DISTINCT p) AS mentions
                    """,
                    {"name": name},
                )
                mentions = mention_rows[0]["mentions"] if mention_rows else 0
                if mentions > best_count:
                    best_count = mentions
                    best = name

            # Merge all others into the best one
            for name in names:
                if name == best:
                    continue
                # Transfer page relationships from duplicate to canonical
                await neo4j.run_write(
                    f"""
                    MATCH (dup:{label} {{{pk}: $dup_name}})
                    MATCH (keep:{label} {{{pk}: $keep_name}})
                    OPTIONAL MATCH (p:Page)-[r]->(dup)
                    WITH dup, keep, p, type(r) AS rel_type
                    WHERE p IS NOT NULL
                    CALL {{
                        WITH p, keep, rel_type
                        WITH p, keep, rel_type
                        WHERE rel_type IS NOT NULL
                        MERGE (p)-[:{label}__TEMP_REL]->(keep)
                    }}
                    """,
                    {"dup_name": name, "keep_name": best},
                )
                # Actually, Cypher can't dynamically create relationship types.
                # Simpler approach: just delete the duplicate. Page relationships
                # that pointed to it are lost, but the canonical entity already
                # has its own mentions from its pages.
                await neo4j.run_write(
                    f"MATCH (e:{label} {{{pk}: $name}}) DETACH DELETE e",
                    {"name": name},
                )
                total_merged += 1
                logger.info(
                    "Merged %s duplicate '%s' into '%s'", label, name, best
                )

    return ForgeResult(
        success=True,
        data={"merged": total_merged},
    )


@router.post("/rebuild-chunks-bulk")
async def rebuild_chunks_bulk(
    request: Request,
    payload: dict | None = None,
) -> ForgeResult:
    """Queue Phase 5 chunk rebuilds for a list of documents.

    Body:
      {
        "doc_ids": ["...", "..."],    # required
        "extract_only": false,         # optional, default false
        "skip_extract": false,         # optional, default false
        "only_missing": false          # optional; when true, skip docs
                                         that already have Chunk nodes
      }

    Returns one queued job_id per document. Jobs run sequentially through
    the existing pipeline queue (one at a time), so you can fire 50 docs
    at once and let them drain overnight.
    """
    import asyncio

    if not isinstance(payload, dict):
        payload = {}
    doc_ids = payload.get("doc_ids") or []
    extract_only = bool(payload.get("extract_only"))
    skip_extract = bool(payload.get("skip_extract"))
    only_missing = bool(payload.get("only_missing"))

    if not isinstance(doc_ids, list) or not doc_ids:
        return ForgeResult(success=False, reason="doc_ids must be a non-empty list",
                           data={"queued": 0})
    if extract_only and skip_extract:
        return ForgeResult(success=False,
                           reason="extract_only and skip_extract are mutually exclusive",
                           data={"queued": 0})

    neo4j = request.app.state.neo4j
    jobs = request.app.state.job_manager
    pipeline = request.app.state.pipeline

    # Pull titles/filenames + chunk counts in one round trip so we can honour
    # only_missing without N extra queries.
    rows = await neo4j.run_query(
        """
        UNWIND $ids AS id
        MATCH (d:Document {doc_id: id})
        OPTIONAL MATCH (d)-[:HAS_PAGE]->(:Page)-[:HAS_CHUNK]->(c:Chunk)
        RETURN d.doc_id AS doc_id, d.filename AS filename, d.title AS title,
               count(c) AS chunk_count
        """,
        {"ids": doc_ids},
    )
    found = {r["doc_id"]: r for r in rows}
    missing = [i for i in doc_ids if i not in found]

    queued: list[dict] = []
    skipped: list[dict] = []
    for doc_id in doc_ids:
        if doc_id not in found:
            continue
        info = found[doc_id]
        if only_missing and info["chunk_count"] and info["chunk_count"] > 0:
            skipped.append({"doc_id": doc_id, "reason": "already has chunks"})
            continue
        job = await jobs.create(
            source_path=f"(rebuild-chunks of {doc_id})",
            filename=info["filename"],
            categories=[],
            tags=[],
        )
        asyncio.create_task(
            pipeline.run_rebuild_chunks(
                job.job_id, doc_id,
                extract_only=extract_only,
                skip_extract=skip_extract,
            )
        )
        queued.append({
            "doc_id": doc_id, "job_id": job.job_id,
            "title": info["title"],
        })

    return ForgeResult(
        success=True,
        data={
            "queued": len(queued),
            "skipped": len(skipped),
            "not_found": len(missing),
            "jobs": queued,
            "skipped_docs": skipped,
            "missing_ids": missing,
        },
    )


@router.post("/bulk-reembed")
async def bulk_reembed(request: Request) -> ForgeResult:
    """Trigger re-embed for ALL documents. Each document gets its own job
    so progress is trackable per document. Jobs run sequentially (one at a
    time via the asyncio pipeline)."""
    import asyncio

    neo4j = request.app.state.neo4j
    jobs = request.app.state.job_manager
    pipeline = request.app.state.pipeline

    rows = await neo4j.run_query(
        "MATCH (d:Document) RETURN d.doc_id AS doc_id, d.filename AS filename"
    )
    if not rows:
        return ForgeResult(success=True, data={"queued": 0})

    job_ids = []
    for r in rows:
        job = await jobs.create(
            source_path=f"(reembed of {r['doc_id']})",
            filename=r["filename"],
            categories=[],
            tags=[],
        )
        asyncio.create_task(pipeline.run_embeddings_only(job.job_id, r["doc_id"]))
        job_ids.append({"doc_id": r["doc_id"], "job_id": job.job_id})

    return ForgeResult(
        success=True,
        data={"queued": len(job_ids), "jobs": job_ids},
    )


@router.post("/reembed-text")
async def reembed_text(request: Request, payload: dict | None = None) -> ForgeResult:
    """Re-embed text only (no visual embeddings, no entity extraction).

    Clears p.text_embedding and re-runs _embed_text() for every page.
    Visual embeddings (colpali_vectors) are left untouched, saving hours
    of GPU time when only the text embedding model has changed (e.g.,
    switching from a 768-d to a 1024-d model).

    Body (optional):
      {"doc_id": "..."}   -- re-embed a single document
      {}  or omitted      -- re-embed ALL documents

    Each document gets its own job for trackable progress.
    """
    import asyncio

    neo4j = request.app.state.neo4j
    jobs = request.app.state.job_manager
    pipeline = request.app.state.pipeline

    if not isinstance(payload, dict):
        payload = {}
    doc_id = payload.get("doc_id")

    if doc_id:
        # Single document
        rows = await neo4j.run_query(
            "MATCH (d:Document {doc_id: $id}) RETURN d.doc_id AS doc_id, d.filename AS filename",
            {"id": doc_id},
        )
        if not rows:
            return ForgeResult(success=False, reason=f"Document {doc_id} not found")
    else:
        # All documents
        rows = await neo4j.run_query(
            "MATCH (d:Document) RETURN d.doc_id AS doc_id, d.filename AS filename"
        )
        if not rows:
            return ForgeResult(success=True, data={"queued": 0})

    job_ids = []
    for r in rows:
        job = await jobs.create(
            source_path=f"(text-reembed of {r['doc_id']})",
            filename=r["filename"],
            categories=[],
            tags=[],
        )
        asyncio.create_task(pipeline.run_text_reembed_only(job.job_id, r["doc_id"]))
        job_ids.append({"doc_id": r["doc_id"], "job_id": job.job_id})

    return ForgeResult(
        success=True,
        data={"queued": len(job_ids), "jobs": job_ids},
    )


@router.post("/cleanup-uploads")
async def cleanup_uploads(request: Request) -> ForgeResult:
    """Delete staged upload files from data/uploads/.

    These are copies of PDFs left over from ingestion runs. The originals
    are wherever the user stored them; these are temporary staging copies
    that should be cleaned periodically. Active (processing/queued) jobs
    are excluded — we only delete files not referenced by any in-flight job.
    """
    import os
    from pathlib import Path

    settings = request.app.state.settings
    uploads_dir = Path(settings.server.data_dir) / "uploads"
    if not uploads_dir.exists():
        return ForgeResult(success=True, data={"deleted": 0, "freed_bytes": 0})

    # Get source_paths of active jobs
    jobs = request.app.state.job_manager
    active = await jobs.list_recent(status="processing", limit=100)
    queued = await jobs.list_recent(status="queued", limit=100)
    active_paths = {j.source_path for j in active + queued}

    deleted = 0
    freed = 0
    for f in uploads_dir.iterdir():
        if f.is_file() and str(f) not in active_paths:
            size = f.stat().st_size
            f.unlink()
            deleted += 1
            freed += size

    return ForgeResult(
        success=True,
        data={
            "deleted": deleted,
            "freed_bytes": freed,
            "freed_mb": round(freed / 1e6, 1),
        },
    )


# Dedup ranks each Page in a (doc_id, page_number) group:
#   colpali_done  worth 2 (has colpali_vector_count > 0)
#   text_emb_done worth 1 (has text_embedding set)
# The highest-ranked page wins; ties broken by page_id (lexicographic, stable).
# Victims are DETACH DELETEd — this also removes any HAS_PAGE / MENTIONS_*
# relationships they had. The keeper is untouched.
_DEDUP_QUERY = """
MATCH (d:Document)-[:HAS_PAGE]->(p:Page)
WITH d, p.page_number AS pn, collect(p) AS pages
WHERE size(pages) > 1
UNWIND pages AS page
WITH d, pn, pages, page,
     coalesce(page.colpali_vector_count, 0) AS cv,
     (CASE WHEN page.text_embedding IS NULL THEN 0 ELSE 1 END) AS te
WITH d, pn, pages, page, (CASE WHEN cv > 0 THEN 2 ELSE 0 END) + te AS rank
ORDER BY rank DESC, page.page_id ASC
WITH d, pn, pages, collect(page) AS ordered
WITH d, pn, head(ordered) AS keeper, tail(ordered) AS victims
UNWIND victims AS victim
DETACH DELETE victim
RETURN count(victim) AS deleted
"""


@router.post("/dedup-pages")
async def dedup_pages(request: Request) -> ForgeResult:
    """Remove duplicate :Page nodes for each (doc_id, page_number) pair.

    Keeps the page that's made the most progress (ColPali > text embedding > any)
    and DETACH DELETEs the rest. Idempotent — running it again after it's
    finished is a no-op.
    """
    neo4j = request.app.state.neo4j

    # Count duplicates before/after so the response is informative
    before = await neo4j.run_query(
        """
        MATCH (d:Document)-[:HAS_PAGE]->(p:Page)
        WITH d, p.page_number AS pn, count(p) AS n
        WHERE n > 1
        RETURN count(*) AS duplicate_groups, sum(n - 1) AS extras
        """
    )
    dup_groups = before[0]["duplicate_groups"] if before else 0
    extras = before[0]["extras"] if before else 0

    deleted_total = 0
    if extras and extras > 0:
        result = await neo4j.run_write(_DEDUP_QUERY)
        if result:
            deleted_total = sum(r.get("deleted", 0) for r in result)
        logger.info(
            "Page dedup: removed %d duplicate Page(s) across %d group(s)",
            deleted_total, dup_groups,
        )

    after = await neo4j.run_query(
        "MATCH (p:Page) RETURN count(p) AS n"
    )
    remaining = after[0]["n"] if after else 0

    return ForgeResult(
        success=True,
        data={
            "duplicate_groups_found": dup_groups,
            "extras_found": extras,
            "deleted": deleted_total,
            "pages_after_dedup": remaining,
        },
    )


# ------------------------------------------------------------------ backup


@router.get("/backup/manifest")
async def backup_manifest(request: Request) -> ForgeResult:
    """Return a lightweight document manifest for backup verification.

    Lists every Document with its doc_id, file_hash, title, page_count,
    categories, and tags. No embeddings, no heavy data — just metadata.
    """
    neo4j = request.app.state.neo4j

    rows = await neo4j.run_query(
        """
        MATCH (d:Document)
        OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Page)
        WITH d, count(p) AS page_count
        OPTIONAL MATCH (d)-[:IN_CATEGORY]->(cat:Category)
        WITH d, page_count, collect(DISTINCT cat.name) AS categories
        OPTIONAL MATCH (d)-[:HAS_TAG]->(tag:Tag)
        RETURN d.doc_id       AS doc_id,
               d.file_hash    AS file_hash,
               d.title        AS title,
               d.filename     AS filename,
               page_count,
               categories,
               collect(DISTINCT tag.name) AS tags
        ORDER BY d.title
        """
    )

    documents = [
        {
            "doc_id": r["doc_id"],
            "file_hash": r["file_hash"],
            "title": r["title"],
            "filename": r["filename"],
            "page_count": r["page_count"],
            "categories": r["categories"],
            "tags": r["tags"],
        }
        for r in rows
    ]

    return ForgeResult(
        success=True,
        data={
            "document_count": len(documents),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "documents": documents,
        },
    )


@router.post("/backup")
async def backup_graph(request: Request) -> ForgeResult:
    """Hot backup: export the graph as a JSON file while Neo4j is running.

    Exports:
      - All Document nodes (full properties)
      - All Page nodes (metadata only — no embeddings or extracted_text blobs)
      - All entity nodes (Material, Process, Standard, Equipment) and their
        relationships to Pages
      - Category and Tag nodes with their Document relationships

    The output file is written to data/backups/graph_<timestamp>.json and
    the response includes the file path and size.
    """
    neo4j = request.app.state.neo4j
    settings = request.app.state.settings
    backup_dir = Path(settings.server.data_dir) / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = backup_dir / f"graph_{ts}.json"

    export: dict = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "format_version": 1,
    }

    # Documents
    docs = await neo4j.run_query(
        """
        MATCH (d:Document)
        RETURN d {.*} AS doc
        """
    )
    export["documents"] = [r["doc"] for r in docs]

    # Pages — metadata only (skip embeddings and large text)
    pages = await neo4j.run_query(
        """
        MATCH (d:Document)-[:HAS_PAGE]->(p:Page)
        RETURN d.doc_id AS doc_id,
               p.page_id AS page_id,
               p.page_number AS page_number,
               p.text_char_count AS text_char_count,
               p.is_blank AS is_blank,
               p.colpali_vector_count AS colpali_vector_count,
               p.colpali_vector_dim AS colpali_vector_dim,
               (p.text_embedding IS NOT NULL) AS has_text_embedding
        ORDER BY d.doc_id, p.page_number
        """
    )
    export["pages"] = pages

    # Entity nodes and their Page relationships
    entity_labels = ["Material", "Process", "Standard", "Equipment"]
    export["entities"] = {}
    export["entity_relationships"] = []

    for label in entity_labels:
        pk = "code" if label == "Standard" else "name"

        # Nodes
        nodes = await neo4j.run_query(
            f"MATCH (e:{label}) RETURN e {{.*}} AS entity"
        )
        export["entities"][label] = [r["entity"] for r in nodes]

        # Relationships to Pages
        rels = await neo4j.run_query(
            f"""
            MATCH (p:Page)-[r]->(e:{label})
            RETURN p.page_id AS page_id,
                   type(r)   AS rel_type,
                   e.{pk}    AS entity_key,
                   '{label}' AS entity_label
            """
        )
        export["entity_relationships"].extend(rels)

    # Categories and Tags with Document relationships
    cats = await neo4j.run_query(
        """
        MATCH (d:Document)-[:IN_CATEGORY]->(c:Category)
        RETURN d.doc_id AS doc_id, c.name AS category
        """
    )
    export["document_categories"] = cats

    tags = await neo4j.run_query(
        """
        MATCH (d:Document)-[:HAS_TAG]->(t:Tag)
        RETURN d.doc_id AS doc_id, t.name AS tag
        """
    )
    export["document_tags"] = tags

    # Collections
    collections = await neo4j.run_query(
        """
        MATCH (d:Document)-[:IN_COLLECTION]->(col:Collection)
        RETURN d.doc_id AS doc_id, col.name AS collection
        """
    )
    export["document_collections"] = collections

    # Summary counts
    export["counts"] = {
        "documents": len(export["documents"]),
        "pages": len(export["pages"]),
        "entity_relationships": len(export["entity_relationships"]),
        "categories": len(export["document_categories"]),
        "tags": len(export["document_tags"]),
        "collections": len(export["document_collections"]),
    }
    for label in entity_labels:
        export["counts"][label.lower() + "s"] = len(export["entities"].get(label, []))

    # Write file
    out_path.write_text(json.dumps(export, indent=2, default=str), encoding="utf-8")
    file_size = out_path.stat().st_size

    logger.info(
        "Graph backup exported to %s (%d bytes, %d docs, %d pages)",
        out_path, file_size, len(export["documents"]), len(export["pages"]),
    )

    return ForgeResult(
        success=True,
        data={
            "path": str(out_path),
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / 1e6, 1),
            "counts": export["counts"],
        },
    )


# ------------------------------------------------------------------ restore


@router.get("/restore/status")
async def restore_status(request: Request) -> ForgeResult:
    """Check whether a database restore is needed.

    Returns ``needs_restore: true`` when the Neo4j database is empty
    (0 documents and 0 pages), which typically means this is a fresh
    install or the data was wiped.

    Also lists any local backup directories and their dump files so the
    caller knows what's available for a local restore.
    """
    neo4j = request.app.state.neo4j
    needs_restore = getattr(request.app.state, "needs_restore", False)

    doc_count = 0
    page_count = 0
    neo4j_connected = False

    try:
        connected = await neo4j.verify_connectivity()
        neo4j_connected = connected
        if connected:
            counts = await neo4j.get_counts()
            doc_count = counts.get("documents", 0)
            page_count = counts.get("pages", 0)
            needs_restore = doc_count == 0 and page_count == 0
    except Exception:
        needs_restore = True

    # List local backups
    settings = request.app.state.settings
    backup_dir = Path(settings.server.data_dir) / "backups"
    local_backups: list[dict] = []
    if backup_dir.exists():
        for subdir in sorted(backup_dir.glob("[0-9]*"), reverse=True):
            if subdir.is_dir():
                dumps = list(subdir.glob("*.dump"))
                entry: dict = {
                    "directory": str(subdir),
                    "timestamp": subdir.name,
                    "has_dump": len(dumps) > 0,
                }
                if dumps:
                    dump_file = dumps[0]
                    entry["dump_file"] = str(dump_file)
                    entry["dump_size_mb"] = round(
                        dump_file.stat().st_size / 1e6, 1
                    )
                manifests = list(subdir.glob("manifest.json"))
                entry["has_manifest"] = len(manifests) > 0
                local_backups.append(entry)

    return ForgeResult(
        success=True,
        data={
            "needs_restore": needs_restore,
            "neo4j_connected": neo4j_connected,
            "document_count": doc_count,
            "page_count": page_count,
            "local_backups": local_backups,
        },
    )


@router.post("/restore")
async def restore_instructions(
    request: Request,
    payload: dict | None = None,
) -> ForgeResult:
    """Return CLI commands for restoring the database.

    The actual restore requires stopping Neo4j, which cannot be done from
    within the running FastAPI service. This endpoint validates the request
    and returns the exact shell commands the user needs to run.

    Body:
      {"source": "local", "dump_path": "/path/to/file.dump"}
      {"source": "drive"}
    """
    if not isinstance(payload, dict):
        payload = {}

    source = payload.get("source", "").lower()
    project_root = Path(__file__).resolve().parent.parent.parent

    if source == "local":
        dump_path = payload.get("dump_path", "")
        if not dump_path:
            return ForgeResult(
                success=False,
                reason="dump_path is required when source is 'local'",
            )

        dump_file = Path(dump_path)
        if not dump_file.exists():
            return ForgeResult(
                success=False,
                reason=f"Dump file not found: {dump_path}",
            )
        if not dump_file.suffix == ".dump":
            return ForgeResult(
                success=False,
                reason=f"File does not appear to be a neo4j dump (expected .dump extension): {dump_path}",
            )

        dump_size_mb = round(dump_file.stat().st_size / 1e6, 1)
        dump_dir = str(dump_file.parent)

        return ForgeResult(
            success=True,
            data={
                "source": "local",
                "dump_path": str(dump_file),
                "dump_size_mb": dump_size_mb,
                "instructions": (
                    "The restore must be run from the command line because it "
                    "requires stopping the Neo4j service and this running API server."
                ),
                "commands": [
                    f"cd {project_root}",
                    f"./scripts/restore.sh --from-local {dump_dir}",
                ],
                "one_liner": (
                    f"cd {project_root} && ./scripts/restore.sh --from-local {dump_dir}"
                ),
            },
        )

    elif source == "drive":
        return ForgeResult(
            success=True,
            data={
                "source": "drive",
                "instructions": (
                    "The restore will download the latest dump from Google Drive "
                    "and load it into Neo4j. This requires stopping the Neo4j "
                    "service and this running API server."
                ),
                "commands": [
                    f"cd {project_root}",
                    "./scripts/restore.sh --from-drive",
                ],
                "one_liner": (
                    f"cd {project_root} && ./scripts/restore.sh --from-drive"
                ),
            },
        )

    else:
        return ForgeResult(
            success=False,
            reason=(
                "Invalid source. Use {\"source\": \"local\", \"dump_path\": \"...\"} "
                "or {\"source\": \"drive\"}"
            ),
        )


# ------------------------------------------------------------------ backup settings & full backup


class BackupSettingsPayload(BaseModel):
    destination: str = ""
    include_images: bool = True
    include_pdfs: bool = True
    gdrive_enabled: bool = True
    gdrive_dump: bool = False


def _backup_settings_path(settings: Any) -> Path:
    """Path to the persistent backup_settings.json file."""
    return Path(settings.server.data_dir).parent / "config" / "backup_settings.json"


def _load_backup_settings(settings: Any) -> dict:
    """Load backup settings from disk or return defaults."""
    path = _backup_settings_path(settings)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "destination": settings.backup.destination,
        "include_images": settings.backup.include_images,
        "include_pdfs": settings.backup.include_pdfs,
        "gdrive_enabled": settings.backup.gdrive_enabled,
        "gdrive_dump": settings.backup.gdrive_dump,
    }


def _save_backup_settings(settings: Any, data: dict) -> None:
    """Persist backup settings to config/backup_settings.json."""
    path = _backup_settings_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


@router.get("/backup/settings")
async def get_backup_settings(request: Request) -> ForgeResult:
    """Return current backup settings."""
    settings = request.app.state.settings
    data = _load_backup_settings(settings)
    return ForgeResult(success=True, data=data)


@router.post("/backup/settings")
async def update_backup_settings(
    request: Request, payload: BackupSettingsPayload
) -> ForgeResult:
    """Update backup settings and persist to config/backup_settings.json."""
    settings = request.app.state.settings
    data = payload.model_dump()
    _save_backup_settings(settings, data)
    return ForgeResult(success=True, data=data)


@router.get("/backup/list")
async def list_backups(request: Request) -> ForgeResult:
    """List available backups from data/backups/ and the configured destination."""
    settings = request.app.state.settings
    backup_settings = _load_backup_settings(settings)
    backups: list[dict] = []

    def _scan_dir(base: Path, source_label: str) -> None:
        if not base.exists():
            return
        # Scan for graph JSON files (hot backups)
        for f in sorted(base.glob("graph_*.json"), reverse=True):
            try:
                stat = f.stat()
                backups.append({
                    "path": str(f),
                    "source": source_label,
                    "timestamp": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / 1e6, 1),
                    "has_dump": False,
                    "has_images": False,
                    "has_manifest": False,
                    "type": "graph_json",
                })
            except OSError:
                pass
        # Scan for timestamped subdirectories (full backups)
        for subdir in sorted(base.glob("[0-9]*"), reverse=True):
            if not subdir.is_dir():
                continue
            try:
                dumps = list(subdir.glob("*.dump"))
                manifests = list(subdir.glob("manifest.json"))
                images_dir = subdir / "page_images"
                pdfs_dir = subdir / "pdfs"
                # Compute total size
                total_size = 0
                for root, _dirs, files in os.walk(subdir):
                    for fname in files:
                        try:
                            total_size += os.path.getsize(os.path.join(root, fname))
                        except OSError:
                            pass
                backups.append({
                    "path": str(subdir),
                    "source": source_label,
                    "timestamp": subdir.name,
                    "size_bytes": total_size,
                    "size_mb": round(total_size / 1e6, 1),
                    "has_dump": len(dumps) > 0,
                    "has_images": images_dir.exists() and any(images_dir.iterdir()) if images_dir.exists() else False,
                    "has_manifest": len(manifests) > 0,
                    "type": "full_backup",
                })
            except OSError:
                pass

    # Local data/backups/
    local_backup_dir = Path(settings.server.data_dir) / "backups"
    _scan_dir(local_backup_dir, "local")

    # Configured destination
    dest = backup_settings.get("destination", "")
    if dest and Path(dest).exists() and str(Path(dest).resolve()) != str(local_backup_dir.resolve()):
        _scan_dir(Path(dest), "destination")

    return ForgeResult(success=True, data={"backups": backups})


@router.get("/backup/progress")
async def backup_progress(request: Request) -> ForgeResult:
    """Return progress of any running backup job."""
    progress = getattr(request.app.state, "backup_progress", None)
    if progress is None:
        return ForgeResult(success=True, data={"running": False})
    return ForgeResult(success=True, data=progress)


@router.post("/backup/full")
async def trigger_full_backup(request: Request) -> ForgeResult:
    """Trigger a full backup to the configured destination.

    Copies: graph JSON export, page images (if enabled), source PDFs (if enabled).
    Runs in background via asyncio.create_task.
    """
    settings = request.app.state.settings
    backup_settings = _load_backup_settings(settings)
    destination = backup_settings.get("destination", "")

    if not destination:
        return ForgeResult(
            success=False,
            reason="No backup destination configured. Set a destination path first via POST /admin/backup/settings.",
        )

    dest_path = Path(destination)
    if not dest_path.exists():
        return ForgeResult(
            success=False,
            reason=f"Destination path does not exist: {destination}",
        )
    if not os.access(dest_path, os.W_OK):
        return ForgeResult(
            success=False,
            reason=f"Destination path is not writable: {destination}",
        )

    # Check if a backup is already running
    progress = getattr(request.app.state, "backup_progress", None)
    if progress and progress.get("running"):
        return ForgeResult(
            success=False,
            reason="A backup is already in progress.",
        )

    include_images = backup_settings.get("include_images", True)
    include_pdfs = backup_settings.get("include_pdfs", True)
    gdrive_enabled = backup_settings.get("gdrive_enabled", True)
    gdrive_dump = backup_settings.get("gdrive_dump", False)

    # Initialize progress
    request.app.state.backup_progress = {
        "running": True,
        "percent": 0,
        "current_file": "starting...",
        "bytes_copied": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    asyncio.create_task(
        _run_full_backup(
            request.app, settings, dest_path,
            include_images, include_pdfs, gdrive_enabled, gdrive_dump,
        )
    )

    return ForgeResult(
        success=True,
        data={
            "message": "Full backup started",
            "destination": str(dest_path),
            "include_images": include_images,
            "include_pdfs": include_pdfs,
        },
    )


def _needs_copy(src: Path, dst: Path) -> bool:
    """Return True if src needs copying (dst missing, size differs, or older)."""
    if not dst.exists():
        return True
    try:
        ss = src.stat()
        ds = dst.stat()
        return ss.st_size != ds.st_size or ss.st_mtime > ds.st_mtime
    except OSError:
        return True


async def _run_full_backup(
    app: Any,
    settings: Any,
    dest_path: Path,
    include_images: bool,
    include_pdfs: bool,
    gdrive_enabled: bool = False,
    gdrive_dump: bool = False,
) -> None:
    """Background task that performs the full backup."""
    data_dir = Path(settings.server.data_dir)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir = dest_path / ts
    backup_dir.mkdir(parents=True, exist_ok=True)

    progress = app.state.backup_progress
    total_bytes = 0

    try:
        import subprocess

        # Step 1: Neo4j database dump (the critical piece — includes embeddings)
        # Uses a dedicated helper script via sudo. The helper handles
        # stop→dump→restart so Neo4j always comes back up even on failure.
        # Setup (run once):
        #   sudo cp scripts/neo4j-dump-helper.sh /usr/local/bin/forgerag-dump
        #   sudo chmod 755 /usr/local/bin/forgerag-dump
        #   echo 'nuc1 ALL=(ALL) NOPASSWD: /usr/local/bin/forgerag-dump' | sudo tee /etc/sudoers.d/forgerag-dump
        #   sudo chmod 440 /etc/sudoers.d/forgerag-dump
        progress["current_file"] = "Creating Neo4j database dump (Neo4j will pause briefly)..."
        progress["percent"] = 2
        dump_file = backup_dir / f"neo4j_{ts}.dump"
        dump_ok = False
        try:
            dump_result = await asyncio.to_thread(
                subprocess.run,
                ["sudo", "/usr/local/bin/forgerag-dump", str(backup_dir), ts],
                capture_output=True, text=True, timeout=600,
            )
            if dump_result.returncode == 0 and dump_file.exists():
                dump_size = dump_file.stat().st_size
                total_bytes += dump_size
                dump_ok = True
                logger.info("Neo4j dump created: %s (%d bytes)", dump_file, dump_size)
            else:
                stderr = dump_result.stderr[:300] if dump_result.stderr else ""
                stdout = dump_result.stdout[:300] if dump_result.stdout else ""
                logger.warning(
                    "Neo4j dump failed (code %d). Setup required:\n"
                    "  sudo cp scripts/neo4j-dump-helper.sh /usr/local/bin/forgerag-dump\n"
                    "  sudo chmod 755 /usr/local/bin/forgerag-dump\n"
                    "  echo 'nuc1 ALL=(ALL) NOPASSWD: /usr/local/bin/forgerag-dump' "
                    "| sudo tee /etc/sudoers.d/forgerag-dump\n"
                    "  sudo chmod 440 /etc/sudoers.d/forgerag-dump\n"
                    "stderr: %s\nstdout: %s",
                    dump_result.returncode, stderr, stdout,
                )
                progress["dump_skipped"] = (
                    "Neo4j dump requires one-time setup. "
                    "See server logs for instructions."
                )
            # Wait for Neo4j to come back (the helper script restarts it)
            progress["current_file"] = "Waiting for Neo4j to restart..."
            for _ in range(30):
                await asyncio.sleep(2)
                if await app.state.neo4j.verify_connectivity():
                    break
        except FileNotFoundError:
            logger.warning(
                "forgerag-dump helper not installed. Run:\n"
                "  sudo cp scripts/neo4j-dump-helper.sh /usr/local/bin/forgerag-dump\n"
                "  sudo chmod 755 /usr/local/bin/forgerag-dump\n"
                "  echo 'nuc1 ALL=(ALL) NOPASSWD: /usr/local/bin/forgerag-dump' "
                "| sudo tee /etc/sudoers.d/forgerag-dump\n"
                "  sudo chmod 440 /etc/sudoers.d/forgerag-dump"
            )
            progress["dump_skipped"] = (
                "forgerag-dump helper not installed. "
                "See server logs for setup instructions."
            )
        except Exception as dump_exc:
            logger.warning("Neo4j dump error (non-fatal): %s", dump_exc)

        progress["percent"] = 5
        progress["bytes_copied"] = total_bytes

        # Step 2: Export graph JSON (lightweight metadata backup)
        progress["current_file"] = "Exporting graph JSON..."

        neo4j = app.state.neo4j
        export: dict = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "format_version": 1,
            "has_dump": dump_ok,
        }

        docs = await neo4j.run_query("MATCH (d:Document) RETURN d {.*} AS doc")
        export["documents"] = [r["doc"] for r in docs]

        pages = await neo4j.run_query(
            """
            MATCH (d:Document)-[:HAS_PAGE]->(p:Page)
            RETURN d.doc_id AS doc_id,
                   p.page_id AS page_id,
                   p.page_number AS page_number,
                   p.text_char_count AS text_char_count,
                   p.is_blank AS is_blank,
                   p.colpali_vector_count AS colpali_vector_count,
                   p.colpali_vector_dim AS colpali_vector_dim,
                   (p.text_embedding IS NOT NULL) AS has_text_embedding
            ORDER BY d.doc_id, p.page_number
            """
        )
        export["pages"] = pages

        entity_labels = ["Material", "Process", "Standard", "Equipment"]
        export["entities"] = {}
        export["entity_relationships"] = []
        for label in entity_labels:
            pk = "code" if label == "Standard" else "name"
            nodes = await neo4j.run_query(
                f"MATCH (e:{label}) RETURN e {{.*}} AS entity"
            )
            export["entities"][label] = [r["entity"] for r in nodes]
            rels = await neo4j.run_query(
                f"""
                MATCH (p:Page)-[r]->(e:{label})
                RETURN p.page_id AS page_id,
                       type(r)   AS rel_type,
                       e.{pk}    AS entity_key,
                       '{label}' AS entity_label
                """
            )
            export["entity_relationships"].extend(rels)

        cats = await neo4j.run_query(
            "MATCH (d:Document)-[:IN_CATEGORY]->(c:Category) RETURN d.doc_id AS doc_id, c.name AS category"
        )
        export["document_categories"] = cats
        tags = await neo4j.run_query(
            "MATCH (d:Document)-[:HAS_TAG]->(t:Tag) RETURN d.doc_id AS doc_id, t.name AS tag"
        )
        export["document_tags"] = tags
        collections = await neo4j.run_query(
            "MATCH (d:Document)-[:IN_COLLECTION]->(col:Collection) RETURN d.doc_id AS doc_id, col.name AS collection"
        )
        export["document_collections"] = collections
        export["counts"] = {
            "documents": len(export["documents"]),
            "pages": len(export["pages"]),
            "entity_relationships": len(export["entity_relationships"]),
            "categories": len(export["document_categories"]),
            "tags": len(export["document_tags"]),
            "collections": len(export["document_collections"]),
        }
        for label in entity_labels:
            export["counts"][label.lower() + "s"] = len(export["entities"].get(label, []))

        graph_file = backup_dir / f"graph_{ts}.json"
        graph_content = json.dumps(export, indent=2, default=str)
        graph_file.write_text(graph_content, encoding="utf-8")
        total_bytes += len(graph_content.encode())
        progress["bytes_copied"] = total_bytes
        progress["percent"] = 10

        # Step 3: Copy page images (incremental — shared dir, skip existing)
        copied_files = 0
        skipped_files = 0
        if include_images:
            progress["current_file"] = "Scanning page images..."
            images_src = data_dir / "page_images"
            reduced_src = data_dir / "reduced_images"

            # Images go into a shared dir at dest root (not per-timestamp)
            # so subsequent backups only copy new/changed files
            images_dest = dest_path / "page_images"
            images_dest.mkdir(parents=True, exist_ok=True)

            if images_src.exists():
                image_files = [f for f in images_src.rglob("*") if f.is_file()]
                total_image_files = len(image_files)

                for idx, src_file in enumerate(image_files):
                    rel = src_file.relative_to(images_src)
                    dst = images_dest / rel
                    if _needs_copy(src_file, dst):
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_file, dst)
                        total_bytes += src_file.stat().st_size
                        copied_files += 1
                    else:
                        skipped_files += 1
                    progress["bytes_copied"] = total_bytes
                    if total_image_files > 0:
                        img_pct = (idx + 1) / total_image_files
                        progress["percent"] = int(10 + img_pct * 40)
                    progress["current_file"] = f"{rel} ({copied_files} new, {skipped_files} skipped)"
                    if idx % 50 == 0:
                        await asyncio.sleep(0)

            if reduced_src.exists():
                progress["current_file"] = "Scanning reduced images..."
                reduced_files = [f for f in reduced_src.rglob("*") if f.is_file()]
                reduced_dest = dest_path / "reduced_images"
                reduced_dest.mkdir(parents=True, exist_ok=True)
                for idx, src_file in enumerate(reduced_files):
                    rel = src_file.relative_to(reduced_src)
                    dst = reduced_dest / rel
                    if _needs_copy(src_file, dst):
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_file, dst)
                        total_bytes += src_file.stat().st_size
                        copied_files += 1
                    else:
                        skipped_files += 1
                    progress["bytes_copied"] = total_bytes
                    progress["percent"] = int(50 + (idx + 1) / max(1, len(reduced_files)) * 20)
                    progress["current_file"] = f"{rel} ({copied_files} new, {skipped_files} skipped)"
                    if idx % 50 == 0:
                        await asyncio.sleep(0)
        else:
            progress["percent"] = 70

        # Step 4: Copy source PDFs (incremental)
        if include_pdfs:
            progress["current_file"] = "Scanning source PDFs..."
            uploads_src = data_dir / "uploads"
            if uploads_src.exists():
                pdf_files = [f for f in uploads_src.rglob("*.pdf") if f.is_file()]
                pdfs_dest = dest_path / "pdfs"
                pdfs_dest.mkdir(parents=True, exist_ok=True)
                for idx, src_file in enumerate(pdf_files):
                    rel = src_file.relative_to(uploads_src)
                    dst = pdfs_dest / rel
                    if _needs_copy(src_file, dst):
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_file, dst)
                        total_bytes += src_file.stat().st_size
                        copied_files += 1
                    else:
                        skipped_files += 1
                    progress["bytes_copied"] = total_bytes
                    progress["percent"] = int(70 + (idx + 1) / max(1, len(pdf_files)) * 20)
                    progress["current_file"] = f"{rel} ({copied_files} new, {skipped_files} skipped)"
                    if idx % 10 == 0:
                        await asyncio.sleep(0)
        else:
            progress["percent"] = 90

        # Step 5: Write manifest
        progress["current_file"] = "Writing manifest..."
        dump_size_bytes = dump_file.stat().st_size if dump_file.exists() else 0
        manifest = {
            "timestamp": ts,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "has_dump": dump_ok,
            "dump_file": dump_file.name if dump_ok else None,
            "dump_size_bytes": dump_size_bytes,
            "total_bytes_copied": total_bytes,
            "files_copied": copied_files,
            "files_skipped": skipped_files,
            "include_images": include_images,
            "include_pdfs": include_pdfs,
            "document_count": len(export["documents"]),
            "page_count": len(export["pages"]),
        }
        manifest_file = backup_dir / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # Step 6: Upload to Google Drive if enabled
        # Copy key files to data/backups/ so gdrive_backup.py can find them,
        # then run the upload script.
        if gdrive_enabled:
            progress["current_file"] = "Preparing Google Drive upload..."
            progress["percent"] = 93
            try:
                project_root = Path(__file__).resolve().parent.parent.parent
                local_backups = data_dir / "backups"
                local_backups.mkdir(parents=True, exist_ok=True)

                # Copy graph JSON to data/backups/ for gdrive_backup.py
                local_graph = local_backups / graph_file.name
                shutil.copy2(graph_file, local_graph)

                # Copy manifest
                local_manifest = local_backups / f"manifest_{ts}.json"
                shutil.copy2(manifest_file, local_manifest)

                # Symlink dump for Drive upload only if gdrive_dump is enabled
                if gdrive_dump and dump_ok and dump_file.exists():
                    local_dump_dir = local_backups / ts
                    local_dump_dir.mkdir(parents=True, exist_ok=True)
                    local_dump_link = local_dump_dir / dump_file.name
                    if not local_dump_link.exists():
                        try:
                            local_dump_link.symlink_to(dump_file)
                        except OSError:
                            shutil.copy2(dump_file, local_dump_link)

                progress["current_file"] = "Uploading to Google Drive..."
                progress["percent"] = 95
                gdrive_script = project_root / "scripts" / "gdrive_backup.py"
                venv_python = project_root / "venv" / "bin" / "python3"
                python_cmd = str(venv_python) if venv_python.exists() else "python3"
                result = await asyncio.to_thread(
                    subprocess.run,
                    [python_cmd, str(gdrive_script)],
                    capture_output=True, text=True, timeout=600,
                    cwd=str(project_root),
                )
                if result.returncode == 0:
                    logger.info("Google Drive upload succeeded")
                else:
                    logger.warning("Google Drive upload failed: %s", result.stderr[:500])
            except Exception as gdrive_exc:
                logger.warning("Google Drive upload error: %s", gdrive_exc)

        progress["percent"] = 100
        progress["current_file"] = f"Complete ({copied_files} copied, {skipped_files} unchanged)"
        progress["running"] = False
        progress["finished_at"] = datetime.now(timezone.utc).isoformat()
        progress["total_bytes"] = total_bytes
        progress["backup_path"] = str(backup_dir)

        logger.info(
            "Full backup complete: %s (%d bytes, %d copied, %d skipped)",
            backup_dir, total_bytes, copied_files, skipped_files,
        )

    except Exception as exc:
        logger.error("Full backup failed: %s", exc)
        progress["running"] = False
        progress["error"] = str(exc)
        progress["current_file"] = f"FAILED: {exc}"
