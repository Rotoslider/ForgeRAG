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
