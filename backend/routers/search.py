"""Search endpoints: semantic (text embeddings) and visual (ColPali)."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request

from backend.models.common import ForgeResult
from backend.models.search import (
    HybridSearchRequest,
    SearchFilters,
    SemanticSearchRequest,
    VisualSearchRequest,
)

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


# ============================================================================
# Hybrid search (Phase 5)
# ============================================================================

@router.post("/hybrid")
async def hybrid_search(body: HybridSearchRequest, request: Request) -> ForgeResult:
    """Hybrid search combining text vectors and the knowledge graph.

    Strategies:
      graph_boosted  — text vector candidates, boosted by # of matching
                       entities from the query in the page's graph links.
      vector_first   — text vector results enriched with graph context
                       (related entities + communities).
      graph_first    — find entities in the query (simple keyword match),
                       traverse to connected pages, rank by vector similarity.
      community      — search the :Community summary vector index; return
                       community summaries + their member pages for broad
                       "what do we know about X" queries.
    """
    text_emb = getattr(request.app.state, "text_embedding", None)
    neo4j = request.app.state.neo4j
    gpu = request.app.state.gpu
    if text_emb is None:
        raise HTTPException(503, "Text embedding service not available")

    # Every strategy needs the query embedding.
    async with gpu.load_scope("text_embedding"):
        import asyncio as _asyncio
        qvec = await _asyncio.to_thread(text_emb.embed_query, body.query)

    filter_where, filter_params = _filter_clauses(body.filters)

    if body.strategy == "community":
        # Search community summaries
        cypher = """
            CALL db.index.vector.queryNodes('community_summary_embedding', $topk, $vec)
            YIELD node AS c, score
            OPTIONAL MATCH (p:Page)-[:IN_COMMUNITY]->(c)
            OPTIONAL MATCH (d:Document)-[:HAS_PAGE]->(p)
            RETURN c.community_id AS community_id,
                   c.level AS level,
                   c.summary AS summary,
                   c.member_count AS member_count,
                   score,
                   collect(DISTINCT {doc_id: d.doc_id, title: d.title,
                                     page_number: p.page_number,
                                     file_hash: d.file_hash})[..10] AS sample_pages
            ORDER BY score DESC
            LIMIT $limit
        """
        rows = await neo4j.run_query(
            cypher, {"topk": body.limit, "vec": qvec.tolist(), "limit": body.limit}
        )
        return ForgeResult(success=True, data=rows)

    where = f" WHERE {filter_where}" if filter_where else ""

    if body.strategy == "graph_boosted":
        # Text vector candidates, then boost score by number of entity
        # names (or common_names / aliases) from the query appearing on the
        # page's graph edges.
        query_terms = [t.lower() for t in body.query.split() if len(t) >= 3]

        cypher = f"""
            CALL db.index.vector.queryNodes('page_text_embedding', $pool, $vec)
            YIELD node AS p, score AS base_score
            MATCH (d:Document)-[:HAS_PAGE]->(p){where}
            // Gather linked entity keys plus their aliases in a single list
            OPTIONAL MATCH (p)-[r]->(e)
            WHERE type(r) IN ['MENTIONS_MATERIAL','DESCRIBES_PROCESS',
                              'REFERENCES_STANDARD','MENTIONS_EQUIPMENT']
            WITH p, d, base_score,
                 collect(DISTINCT toLower(coalesce(e.name, e.code))) AS ent_names,
                 collect(coalesce(e.common_names, [])) AS nested_aliases
            WITH p, d, base_score, ent_names,
                 // flatten list-of-lists to a single list of lowercased aliases
                 reduce(acc = [], sub IN nested_aliases |
                        acc + [x IN sub | toLower(x)]) AS ent_aliases
            WITH p, d, base_score, ent_names, ent_aliases,
                 ent_names + ent_aliases AS all_names
            WITH p, d, base_score, all_names,
                 size([t IN $terms WHERE
                       any(n IN all_names WHERE n CONTAINS t)
                 ]) AS entity_hits
            RETURN p.page_id AS page_id,
                   p.page_number AS page_number,
                   p.extracted_text AS extracted_text,
                   d.doc_id AS doc_id,
                   d.title AS document_title,
                   d.filename AS filename,
                   d.file_hash AS file_hash,
                   base_score,
                   entity_hits,
                   base_score + $boost * entity_hits AS final_score,
                   all_names[..10] AS matched_entities,
                   [(d)-[:IN_CATEGORY]->(c) | c.name] AS categories,
                   [(d)-[:TAGGED_WITH]->(t) | t.name] AS tags
            ORDER BY final_score DESC
            LIMIT $limit
        """
        params = {
            "pool": body.candidate_pool, "vec": qvec.tolist(),
            "terms": query_terms, "boost": body.boost_weight,
            "limit": body.limit,
        }
        params.update(filter_params)
        rows = await neo4j.run_query(cypher, params)
        hits = [_format_hit_with_boost(r) for r in rows]
        return ForgeResult(success=True, data=hits)

    if body.strategy == "vector_first":
        # Pure vector search, enriched with entities and community membership.
        # Entity/community collects use list comprehensions so null results
        # from OPTIONAL MATCH become empty lists, not [{name: null}] sentinels.
        cypher = f"""
            CALL db.index.vector.queryNodes('page_text_embedding', $pool, $vec)
            YIELD node AS p, score
            MATCH (d:Document)-[:HAS_PAGE]->(p){where}
            RETURN p.page_id AS page_id,
                   p.page_number AS page_number,
                   p.extracted_text AS extracted_text,
                   d.doc_id AS doc_id,
                   d.title AS document_title,
                   d.filename AS filename,
                   d.file_hash AS file_hash,
                   score,
                   [(p)-[r]->(e) WHERE type(r) IN
                    ['MENTIONS_MATERIAL','DESCRIBES_PROCESS',
                     'REFERENCES_STANDARD','MENTIONS_EQUIPMENT']
                    | {{kind: type(r), name: coalesce(e.name, e.code)}}
                   ] AS entities,
                   [(p)-[:IN_COMMUNITY]->(c:Community) |
                    {{level: c.level, community_id: c.community_id,
                      summary: c.summary}}
                   ] AS communities,
                   [(d)-[:IN_CATEGORY]->(c2) | c2.name] AS categories,
                   [(d)-[:TAGGED_WITH]->(t) | t.name] AS tags
            ORDER BY score DESC
            LIMIT $limit
        """
        params = {"pool": body.candidate_pool, "vec": qvec.tolist(), "limit": body.limit}
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
                "entities": r["entities"],
                "communities": r["communities"],
                "categories": r["categories"],
                "tags": r["tags"],
            })
        return ForgeResult(success=True, data=hits)

    if body.strategy == "graph_first":
        # Find entity nodes whose names or common_names contain any query
        # term, then pages that mention them.
        query_terms = [t.lower() for t in body.query.split() if len(t) >= 3]
        cypher = f"""
            MATCH (e)
            WHERE any(l IN labels(e) WHERE l IN ['Material','Process','Standard','Equipment'])
              AND (
                any(t IN $terms WHERE toLower(coalesce(e.name, e.code, '')) CONTAINS t)
                OR any(alias IN coalesce(e.common_names, []) WHERE
                       any(t IN $terms WHERE toLower(alias) CONTAINS t))
              )
            MATCH (p:Page)-[r]->(e)
            WHERE type(r) IN ['MENTIONS_MATERIAL','DESCRIBES_PROCESS',
                              'REFERENCES_STANDARD','MENTIONS_EQUIPMENT']
            MATCH (d:Document)-[:HAS_PAGE]->(p){where}
            WITH p, d, collect(DISTINCT coalesce(e.name, e.code))[..10] AS matched
            WITH p, d, matched, size(matched) AS match_count
            ORDER BY match_count DESC
            LIMIT $pool
            RETURN p.page_id AS page_id,
                   p.page_number AS page_number,
                   p.extracted_text AS extracted_text,
                   p.text_embedding AS emb,
                   d.doc_id AS doc_id,
                   d.title AS document_title,
                   d.filename AS filename,
                   d.file_hash AS file_hash,
                   matched,
                   match_count,
                   [(d)-[:IN_CATEGORY]->(c) | c.name] AS categories,
                   [(d)-[:TAGGED_WITH]->(t) | t.name] AS tags
        """
        params = {"terms": query_terms, "pool": body.candidate_pool}
        params.update(filter_params)
        rows = await neo4j.run_query(cypher, params)

        # Score each candidate by cosine similarity between query and text embedding
        import numpy as np
        hits = []
        for r in rows:
            emb = r.get("emb")
            if emb:
                v = np.asarray(emb, dtype=np.float32)
                sim = float(np.dot(qvec, v))
            else:
                sim = 0.0
            hits.append({
                "page_id": r["page_id"],
                "doc_id": r["doc_id"],
                "document_title": r["document_title"],
                "filename": r["filename"],
                "page_number": r["page_number"],
                "score": float(r["match_count"]) + sim,
                "vector_similarity": sim,
                "match_count": r["match_count"],
                "matched_entities": r["matched"],
                "text_snippet": _snippet(r["extracted_text"]),
                "image_url": f"/images/{r['file_hash']}/{r['page_number']}",
                "reduced_image_url": f"/images/{r['file_hash']}/{r['page_number']}/reduced",
                "categories": r["categories"],
                "tags": r["tags"],
            })
        hits.sort(key=lambda h: h["score"], reverse=True)
        return ForgeResult(success=True, data=hits[: body.limit])

    raise HTTPException(400, f"Unknown strategy: {body.strategy}")


def _format_hit_with_boost(r: dict) -> dict:
    return {
        "page_id": r["page_id"],
        "doc_id": r["doc_id"],
        "document_title": r["document_title"],
        "filename": r["filename"],
        "page_number": r["page_number"],
        "score": float(r["final_score"]),
        "base_score": float(r["base_score"]),
        "entity_hits": int(r["entity_hits"]),
        "matched_entities": r["matched_entities"],
        "text_snippet": _snippet(r["extracted_text"]),
        "image_url": f"/images/{r['file_hash']}/{r['page_number']}",
        "reduced_image_url": f"/images/{r['file_hash']}/{r['page_number']}/reduced",
        "categories": r["categories"],
        "tags": r["tags"],
    }
