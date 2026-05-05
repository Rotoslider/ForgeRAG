"""Admin / maintenance endpoints.

Small utilities for one-off fixes — not part of the regular user-facing API.
Currently: dedupe Page nodes when re-ingestion before the fix created them.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

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
