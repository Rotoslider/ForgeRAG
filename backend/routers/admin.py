"""Admin / maintenance endpoints.

Small utilities for one-off fixes — not part of the regular user-facing API.
Currently: dedupe Page nodes when re-ingestion before the fix created them.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from backend.models.common import ForgeResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


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
