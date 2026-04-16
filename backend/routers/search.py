"""Search endpoints: semantic (text embeddings) and visual (ColPali)."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request

from backend.models.common import ForgeResult
from backend.models.search import SearchFilters, SemanticSearchRequest, VisualSearchRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


def _snippet(text: str | None, n: int = 240) -> str:
    if not text:
        return ""
    text = text.strip().replace("\n", " ")
    if len(text) <= n:
        return text
    return text[:n] + "…"


def _filter_clauses(filters: SearchFilters | None) -> tuple[str, dict]:
    """Build WHERE clauses + params from a SearchFilters."""
    if not filters:
        return "", {}
    parts: list[str] = []
    params: dict = {}
    if filters.categories:
        parts.append(
            "ALL(cn IN $cats WHERE EXISTS { (d)-[:IN_CATEGORY]->(:Category {name: cn}) })"
        )
        params["cats"] = filters.categories
    if filters.tags:
        parts.append(
            "ALL(tn IN $tags WHERE EXISTS { (d)-[:TAGGED_WITH]->(:Tag {name: tn}) })"
        )
        params["tags"] = filters.tags
    if filters.document_ids:
        parts.append("d.doc_id IN $docs")
        params["docs"] = filters.document_ids
    if filters.source_type:
        parts.append("d.source_type = $stype")
        params["stype"] = filters.source_type
    return " AND ".join(parts), params


@router.post("/semantic")
async def semantic_search(body: SemanticSearchRequest, request: Request) -> ForgeResult:
    """Semantic text search via Neo4j vector index on Page.text_embedding.

    Uses nomic-embed-text to embed the query, then Neo4j's native vector
    similarity search (cosine) to find top-K pages. Applies category/tag
    filters as a second pass (after vector search) since Neo4j's procedure
    doesn't combine them directly.
    """
    text_emb = getattr(request.app.state, "text_embedding", None)
    neo4j = request.app.state.neo4j
    if text_emb is None:
        raise HTTPException(503, "Text embedding service not available")

    gpu = request.app.state.gpu
    async with gpu.load_scope("text_embedding"):
        query_vec = await asyncio.to_thread(text_emb.embed_query, body.query)

    # Over-fetch from the vector index so we can apply filters without
    # falling short of the requested limit.
    topk = max(body.limit * 3, body.limit)

    filter_where, filter_params = _filter_clauses(body.filters)
    where = f" WHERE {filter_where}" if filter_where else ""

    cypher = f"""
        CALL db.index.vector.queryNodes('page_text_embedding', $topk, $query_vec)
        YIELD node AS p, score
        MATCH (d:Document)-[:HAS_PAGE]->(p){where}
        RETURN p.page_id AS page_id,
               p.page_number AS page_number,
               p.extracted_text AS extracted_text,
               d.doc_id AS doc_id,
               d.title AS document_title,
               d.filename AS filename,
               d.file_hash AS file_hash,
               score AS score,
               [(d)-[:IN_CATEGORY]->(c) | c.name] AS categories,
               [(d)-[:TAGGED_WITH]->(t) | t.name] AS tags
        ORDER BY score DESC
        LIMIT $limit
    """
    params = {"topk": topk, "query_vec": query_vec.tolist(), "limit": body.limit}
    params.update(filter_params)

    rows = await neo4j.run_query(cypher, params)

    hits = []
    for r in rows:
        hits.append({
            "page_id": r["page_id"],
            "doc_id": r["doc_id"],
            "document_title": r["document_title"],
            "filename": r["filename"],
            "page_number": r["page_number"],
            "score": float(r["score"]),
            "text_snippet": _snippet(r["extracted_text"]),
            "image_url": f"/images/{r['file_hash']}/{r['page_number']}",
            "reduced_image_url": f"/images/{r['file_hash']}/{r['page_number']}/reduced",
            "categories": r["categories"],
            "tags": r["tags"],
        })
    return ForgeResult(success=True, data=hits)


@router.post("/visual")
async def visual_search(body: VisualSearchRequest, request: Request) -> ForgeResult:
    """Two-stage visual search using ColPali.

    Stage 1: text-vector search retrieves candidate_pool pages using the text
    embedding of the query (cheap, uses existing vector index).

    Stage 2: load ColPali, embed the query as multi-vector, compute MaxSim
    scores against each candidate's stored ColPali vectors, and return top-K.
    """
    text_emb = getattr(request.app.state, "text_embedding", None)
    colpali = getattr(request.app.state, "colpali", None)
    neo4j = request.app.state.neo4j
    gpu = request.app.state.gpu

    if text_emb is None:
        raise HTTPException(503, "Text embedding service not available")
    if colpali is None:
        raise HTTPException(503, "ColPali service not available")

    # Stage 1: coarse candidates via text vector search
    async with gpu.load_scope("text_embedding"):
        tvec = await asyncio.to_thread(text_emb.embed_query, body.query)

    filter_where, filter_params = _filter_clauses(body.filters)
    where = f" WHERE {filter_where}" if filter_where else ""

    cand_cypher = f"""
        CALL db.index.vector.queryNodes('page_text_embedding', $pool, $vec)
        YIELD node AS p, score AS coarse_score
        MATCH (d:Document)-[:HAS_PAGE]->(p){where}
        WHERE p.colpali_vector_count IS NOT NULL AND p.colpali_vector_count > 0
        RETURN p.page_id AS page_id,
               p.page_number AS page_number,
               p.extracted_text AS extracted_text,
               p.colpali_vectors AS colpali_vectors,
               p.colpali_vector_count AS colpali_count,
               p.colpali_vector_dim AS colpali_dim,
               coarse_score,
               d.doc_id AS doc_id,
               d.title AS document_title,
               d.filename AS filename,
               d.file_hash AS file_hash,
               [(d)-[:IN_CATEGORY]->(c) | c.name] AS categories,
               [(d)-[:TAGGED_WITH]->(t) | t.name] AS tags
    """
    params = {"pool": body.candidate_pool, "vec": tvec.tolist()}
    params.update(filter_params)
    candidates = await neo4j.run_query(cand_cypher, params)

    if not candidates:
        return ForgeResult(success=True, data=[])

    # Stage 2: MaxSim rerank in Python
    from backend.services.colpali_service import deserialize_colpali, maxsim_score

    async with gpu.load_scope("colpali"):
        qvec = await asyncio.to_thread(colpali.embed_query, body.query)

    scored = []
    for c in candidates:
        blob = c["colpali_vectors"]
        if blob is None:
            continue
        try:
            doc_vecs = deserialize_colpali(
                bytes(blob), int(c["colpali_count"]), int(c["colpali_dim"] or 128)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to deserialize colpali for %s: %s", c["page_id"], exc)
            continue
        score = maxsim_score(qvec, doc_vecs)
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    hits = []
    for score, r in scored[: body.limit]:
        hits.append({
            "page_id": r["page_id"],
            "doc_id": r["doc_id"],
            "document_title": r["document_title"],
            "filename": r["filename"],
            "page_number": r["page_number"],
            "score": float(score),
            "coarse_score": float(r["coarse_score"]),
            "text_snippet": _snippet(r["extracted_text"]),
            "image_url": f"/images/{r['file_hash']}/{r['page_number']}",
            "reduced_image_url": f"/images/{r['file_hash']}/{r['page_number']}/reduced",
            "categories": r["categories"],
            "tags": r["tags"],
        })
    return ForgeResult(success=True, data=hits)
