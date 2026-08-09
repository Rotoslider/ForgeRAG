"""Choom-friendly skill endpoints for LLM-driven discovery and search.

Provides a manifest describing ForgeRAG capabilities, a unified
auto-routing search endpoint, and a batch search endpoint for running
multiple queries in parallel.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.models.common import ForgeResult
from backend.models.search import HybridSearchRequest, SearchFilters

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/skills", tags=["skills"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Matches code/standard identifiers like "QW-451", "ASTM A36", "SA-516.70",
# "AWS D1.1", "ASME IX", etc.  Pattern: word-chars with at least one letter
# and at least one digit, optionally separated by dots/dashes/spaces.
_CODE_RE = re.compile(
    r"(?:[A-Za-z]+[\s\-\.]*\d[\w\-\.]*)|(?:\d[\w\-\.]*[\s\-\.]*[A-Za-z]+)",
)

_QUESTION_WORDS = {"what", "how", "why", "explain", "describe", "compare", "summarize"}


def _pick_strategy(query: str) -> str:
    """Choose the best search strategy based on query characteristics.

    - Looks like a code/standard reference -> keyword (exact match)
    - Broad question word present -> answer (LLM-synthesized)
    - Otherwise -> hybrid/rrf (best general-purpose retrieval)
    """
    stripped = query.strip()

    # Short queries that look like a code identifier
    if _CODE_RE.search(stripped) and len(stripped.split()) <= 6:
        return "keyword"

    first_word = stripped.split()[0].lower() if stripped else ""
    if first_word in _QUESTION_WORDS:
        return "answer"

    return "hybrid"


def _snippet(text: str | None, n: int = 200) -> str:
    if not text:
        return ""
    text = text.strip().replace("\n", " ")
    if len(text) <= n:
        return text
    return text[:n] + "..."


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SkillSearchRequest(BaseModel):
    query: str = Field(..., description="Free-text search query")
    mode: str | None = Field(
        default=None,
        description="Force a specific mode: keyword, answer, or hybrid. "
        "Omit to auto-select based on query characteristics.",
    )
    limit: int = Field(default=5, ge=1, le=20)
    filters: SearchFilters | None = None


class BatchQueryItem(BaseModel):
    query: str
    mode: str | None = None
    limit: int = Field(default=5, ge=1, le=20)
    filters: SearchFilters | None = None


class BatchSearchRequest(BaseModel):
    queries: list[BatchQueryItem] = Field(
        ..., min_length=1, max_length=20,
        description="Array of queries to run in parallel",
    )


# ---------------------------------------------------------------------------
# /skills/manifest
# ---------------------------------------------------------------------------

@router.get("/manifest")
async def manifest(request: Request) -> ForgeResult:
    """Return a JSON manifest describing ForgeRAG capabilities with live stats."""
    neo4j = request.app.state.neo4j

    # Gather live counts
    stats: dict[str, int] = {"documents": 0, "pages": 0, "entities": 0, "communities": 0}
    try:
        rows = await neo4j.run_query(
            """
            OPTIONAL MATCH (d:Document) WITH count(d) AS documents
            OPTIONAL MATCH (p:Page) WITH documents, count(p) AS pages
            OPTIONAL MATCH (e) WHERE e:Material OR e:Process OR e:Standard
                                  OR e:Equipment OR e:Clause
            WITH documents, pages, count(e) AS entities
            OPTIONAL MATCH (c:Community)
            RETURN documents, pages, entities, count(c) AS communities
            """
        )
        if rows:
            r = rows[0]
            stats = {
                "documents": r.get("documents", 0) or 0,
                "pages": r.get("pages", 0) or 0,
                "entities": r.get("entities", 0) or 0,
                "communities": r.get("communities", 0) or 0,
            }
    except Exception as exc:
        logger.warning("Failed to gather stats for manifest: %s", exc)

    capabilities = [
        {
            "name": "search_answer",
            "description": "Retrieve pages and synthesize an LLM answer with citations",
            "endpoint": "/search/answer",
            "method": "POST",
            "params": ["query", "limit", "search_mode", "use_vision", "use_graph"],
        },
        {
            "name": "search_keyword",
            "description": "Full-text keyword search on extracted page text",
            "endpoint": "/search/keyword",
            "method": "POST",
            "params": ["query", "limit"],
        },
        {
            "name": "search_semantic",
            "description": "Dense vector semantic search via BGE-M3 embeddings",
            "endpoint": "/search/semantic",
            "method": "POST",
            "params": ["query", "limit", "filters"],
        },
        {
            "name": "search_hybrid",
            "description": "Hybrid search combining text vectors and knowledge graph",
            "endpoint": "/search/hybrid",
            "method": "POST",
            "params": ["query", "strategy", "limit", "filters", "boost_weight", "rerank"],
        },
        {
            "name": "search_summaries",
            "description": "Search hierarchical TOC summaries (section/chapter/whole-document level) — for zoom-out questions about what books or chapters cover",
            "endpoint": "/search/summaries",
            "method": "POST",
            "params": ["query", "limit"],
        },
        {
            "name": "graph_query",
            "description": "Run a predefined graph query template (material_standards, process_materials, etc.)",
            "endpoint": "/graph/query",
            "method": "POST",
            # Field names MUST match models/graph.py GraphQueryRequest —
            # the old advertised names (template/params) 422'd every
            # manifest-following client.
            "params": ["query_type", "parameters", "limit"],
        },
        {
            "name": "list_documents",
            "description": "List all ingested documents with metadata",
            "endpoint": "/documents",
            "method": "GET",
            "params": ["collection"],
        },
    ]

    return ForgeResult(
        success=True,
        data={
            "name": "forgerag",
            "version": "1.0",
            "capabilities": capabilities,
            "stats": stats,
        },
    )


# ---------------------------------------------------------------------------
# /skills/search — unified auto-routing search
# ---------------------------------------------------------------------------

@router.post("/search")
async def skills_search(body: SkillSearchRequest, request: Request) -> ForgeResult:
    """Unified search endpoint that picks the best strategy automatically.

    If ``mode`` is omitted the strategy is selected by inspecting the query:
    - Code/standard references (e.g. "QW-451", "ASTM A36") -> keyword
    - Broad questions ("what", "how", "why", ...) -> answer (LLM-synthesized)
    - Everything else -> hybrid/rrf

    Returns a simplified result set optimised for LLM consumption (max 5
    results with short snippets).
    """
    # Import search endpoint functions from the search router so we can
    # reuse the existing implementations directly.
    from backend.routers.search import (
        hybrid_search,
        keyword_search,
        rag_answer,
    )
    from backend.routers.search import (
        AnswerRequest,
        KeywordSearchRequest,
    )

    mode = body.mode or _pick_strategy(body.query)
    results: list[dict[str, Any]] = []

    try:
        if mode == "keyword":
            resp = await keyword_search(
                KeywordSearchRequest(query=body.query, limit=body.limit),
                request,
            )
            for hit in (resp.data or [])[:body.limit]:
                results.append({
                    "title": hit.get("document_title", ""),
                    "page": hit.get("page_number"),
                    "snippet": _snippet(hit.get("text_snippet")),
                    "score": hit.get("score", 0.0),
                })

        elif mode == "answer":
            resp = await rag_answer(
                AnswerRequest(query=body.query, limit=body.limit),
                request,
            )
            # Answer mode returns {answer, sources, ...} — wrap it directly
            return ForgeResult(
                success=True,
                data={"strategy_used": "answer", "answer": resp.data},
            )

        else:
            # Default: hybrid/rrf
            resp = await hybrid_search(
                HybridSearchRequest(
                    query=body.query,
                    strategy="rrf",
                    limit=body.limit,
                    filters=body.filters,
                    rerank=True,
                ),
                request,
            )
            for hit in (resp.data or [])[:body.limit]:
                results.append({
                    "title": hit.get("document_title", ""),
                    "page": hit.get("page_number"),
                    "snippet": _snippet(
                        hit.get("text_snippet") or hit.get("extracted_text")
                    ),
                    "score": hit.get("score", 0.0),
                })

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("skills/search mode=%s failed", mode)
        return ForgeResult(
            success=False,
            reason=f"Search failed ({mode}): {exc}",
        )

    return ForgeResult(
        success=True,
        data={"strategy_used": mode, "results": results},
    )


# ---------------------------------------------------------------------------
# /skills/batch — parallel multi-query search
# ---------------------------------------------------------------------------

@router.post("/batch")
async def skills_batch(body: BatchSearchRequest, request: Request) -> ForgeResult:
    """Run multiple queries in parallel and return an array of results.

    Each query in the batch is routed through the same auto-selection
    logic as ``/skills/search``. Results are returned in the same order
    as the input queries.
    """

    async def _run_one(item: BatchQueryItem) -> dict[str, Any]:
        try:
            resp = await skills_search(
                SkillSearchRequest(
                    query=item.query,
                    mode=item.mode,
                    limit=item.limit,
                    filters=item.filters,
                ),
                request,
            )
            return {"query": item.query, "success": True, "data": resp.data}
        except Exception as exc:
            return {"query": item.query, "success": False, "reason": str(exc)}

    results = await asyncio.gather(*[_run_one(q) for q in body.queries])
    return ForgeResult(success=True, data=list(results))
