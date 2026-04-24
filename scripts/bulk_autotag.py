#!/usr/bin/env python3
"""Bulk auto-tag documents that were ingested before auto-tagging existed.

For every Document missing categories AND tags, asks the LLM for suggested
collection/categories/tags and (optionally) writes them to the graph.

Default behavior is MERGE — existing data is preserved. Use --overwrite to
replace all Tag/Category edges with the suggestion (collection is only
changed when the suggestion isn't "default" or when --overwrite is set).

Usage:
    # Preview only — no writes, prints suggestions per doc.
    NEO4J_PASSWORD=... python scripts/bulk_autotag.py --dry-run

    # Apply to all untagged docs.
    NEO4J_PASSWORD=... python scripts/bulk_autotag.py

    # Limit to the first 5 docs.
    NEO4J_PASSWORD=... python scripts/bulk_autotag.py --limit 5

    # Specific docs only (comma-separated doc_ids).
    NEO4J_PASSWORD=... python scripts/bulk_autotag.py --doc-ids DOC_a,DOC_b

    # Replace existing categories/tags with the suggestion.
    NEO4J_PASSWORD=... python scripts/bulk_autotag.py --overwrite
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_settings  # noqa: E402
from backend.ingestion.auto_tagger import AutoTagger  # noqa: E402
from backend.services.llm_service import create_llm_service  # noqa: E402
from backend.services.neo4j_service import Neo4jService  # noqa: E402


log = logging.getLogger("bulk_autotag")


async def list_candidate_docs(
    svc: Neo4jService,
    *,
    doc_ids: list[str] | None,
    overwrite: bool,
    limit: int | None,
) -> list[dict]:
    """Pick which docs to process.

    Without --overwrite: only docs with no Tag AND no Category edges.
    With --overwrite: every doc (or just --doc-ids if given).
    """
    params: dict = {}
    where = []
    if doc_ids:
        where.append("d.doc_id IN $doc_ids")
        params["doc_ids"] = doc_ids
    if not overwrite:
        where.append(
            "NOT EXISTS { (d)-[:TAGGED_WITH]->(:Tag) } "
            "AND NOT EXISTS { (d)-[:IN_CATEGORY]->(:Category) }"
        )
    where_clause = (" WHERE " + " AND ".join(where)) if where else ""
    lim_clause = f" LIMIT {int(limit)}" if limit else ""
    rows = await svc.run_query(
        f"""
        MATCH (d:Document){where_clause}
        RETURN d.doc_id AS doc_id,
               d.title AS title,
               coalesce(d.collection, 'default') AS collection
        ORDER BY d.ingested_at DESC{lim_clause}
        """,
        params,
    )
    return list(rows)


async def apply_suggestion(
    svc: Neo4jService,
    *,
    doc_id: str,
    collection: str,
    categories: list[str],
    tags: list[str],
    current_collection: str,
    overwrite: bool,
) -> None:
    if overwrite:
        await svc.run_write(
            """
            MATCH (d:Document {doc_id: $id})
            OPTIONAL MATCH (d)-[r1:TAGGED_WITH]->(:Tag)
            OPTIONAL MATCH (d)-[r2:IN_CATEGORY]->(:Category)
            DELETE r1, r2
            """,
            {"id": doc_id},
        )

    # Only change collection when it's currently default (merge) or always
    # (overwrite). Never blow away a non-default collection in merge mode.
    should_set_collection = (
        collection and collection != "default"
        and (overwrite or current_collection == "default")
    )
    if should_set_collection:
        await svc.run_write(
            "MATCH (d:Document {doc_id: $id}) SET d.collection = $col",
            {"id": doc_id, "col": collection},
        )

    if categories:
        await svc.run_write(
            """
            UNWIND $cats AS cat
            MERGE (c:Category {name: cat})
            WITH c
            MATCH (d:Document {doc_id: $id})
            MERGE (d)-[:IN_CATEGORY]->(c)
            """,
            {"id": doc_id, "cats": categories},
        )

    if tags:
        await svc.run_write(
            """
            UNWIND $tags AS tag
            MERGE (t:Tag {name: tag})
            WITH t
            MATCH (d:Document {doc_id: $id})
            MERGE (d)-[:TAGGED_WITH]->(t)
            """,
            {"id": doc_id, "tags": tags},
        )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print suggestions without writing to Neo4j.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N documents.",
    )
    parser.add_argument(
        "--doc-ids", type=str, default=None,
        help="Comma-separated doc_ids. Restricts processing to just these.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace existing Tag/Category edges instead of merging.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    svc = Neo4jService(settings.neo4j)
    await svc.connect()

    llm = create_llm_service(settings)
    if llm is None:
        log.error(
            "LLM service not configured — set LLM_BASE_URL and LLM_MODEL "
            "in your environment."
        )
        await svc.close()
        return 2

    tagger = AutoTagger(llm)

    doc_id_filter = (
        [s.strip() for s in args.doc_ids.split(",") if s.strip()]
        if args.doc_ids
        else None
    )

    docs = await list_candidate_docs(
        svc,
        doc_ids=doc_id_filter,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    if not docs:
        log.info(
            "No candidate documents found "
            "(all docs already have tags/categories, or filter matched nothing)."
        )
        await svc.close()
        return 0

    log.info(
        "Processing %d document(s)%s%s",
        len(docs),
        " (dry-run)" if args.dry_run else "",
        " (overwrite)" if args.overwrite else "",
    )

    processed = 0
    skipped = 0
    failed = 0
    for d in docs:
        doc_id = d["doc_id"]
        title = d["title"] or doc_id
        current_collection = d["collection"] or "default"
        try:
            result = await tagger.suggest_for_doc(svc, doc_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("Suggestion failed for %s (%s): %s", doc_id, title, exc)
            failed += 1
            continue
        if result is None:
            log.info(
                "Skipping %s (%s) — no pages with usable text", doc_id, title
            )
            skipped += 1
            continue

        log.info(
            "%s  %s  → collection=%s  categories=%s  tags=%s",
            doc_id, title,
            result.collection, result.categories, result.tags,
        )

        if args.dry_run:
            continue

        try:
            await apply_suggestion(
                svc,
                doc_id=doc_id,
                collection=result.collection,
                categories=result.categories,
                tags=result.tags,
                current_collection=current_collection,
                overwrite=args.overwrite,
            )
            processed += 1
        except Exception as exc:  # noqa: BLE001
            log.exception("Apply failed for %s: %s", doc_id, exc)
            failed += 1

    log.info(
        "Done. %s processed, %d skipped (no text), %d failed.",
        "would-apply: " + str(len(docs) - skipped - failed) if args.dry_run else processed,
        skipped,
        failed,
    )
    await svc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
