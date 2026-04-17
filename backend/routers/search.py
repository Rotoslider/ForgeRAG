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
from pydantic import BaseModel, Field

from backend.services.graph_reasoning import GraphContext, explore_from_query

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


# ============================================================================
# Keyword search — the missing Ctrl+F equivalent
# ============================================================================

class KeywordSearchRequest(BaseModel):
    query: str = Field(..., description="Exact text to search for (case-insensitive)")
    limit: int = Field(default=20, ge=1, le=100)


@router.post("/keyword")
async def keyword_search(body: KeywordSearchRequest, request: Request) -> ForgeResult:
    """Full-text keyword search on extracted page text.

    Uses Neo4j's Lucene-backed full-text index (page_text_fulltext) for
    O(1) lookup at any collection size. Falls back to CONTAINS scan if
    the index doesn't exist yet. This is the equivalent of Ctrl+F in a
    PDF viewer — use it for specific codes, alloy designations, clause
    IDs, etc.
    """
    neo4j = request.app.state.neo4j

    # Try the full-text index first (scales to 100K+ pages).
    # Wrap in double quotes for Lucene phrase matching — "ASTM A 709"
    # matches as a phrase, not as three individual words (which would
    # return every page mentioning "A" or "709" separately).
    try:
        # Escape any existing quotes in the query
        escaped = body.query.replace('"', '\\"')
        phrase_query = f'"{escaped}"'
        rows = await neo4j.run_query(
            """
            CALL db.index.fulltext.queryNodes('page_text_fulltext', $q)
            YIELD node AS p, score AS ft_score
            MATCH (d:Document)-[:HAS_PAGE]->(p)
            RETURN p.page_id AS page_id,
                   p.page_number AS page_number,
                   p.extracted_text AS extracted_text,
                   d.doc_id AS doc_id,
                   d.title AS document_title,
                   d.filename AS filename,
                   d.file_hash AS file_hash,
                   ft_score,
                   [(d)-[:IN_CATEGORY]->(c) | c.name] AS categories,
                   [(d)-[:TAGGED_WITH]->(t) | t.name] AS tags
            ORDER BY ft_score DESC
            LIMIT $limit
            """,
            {"q": phrase_query, "limit": body.limit},
        )
    except Exception:
        # Fallback: CONTAINS scan (works without the index, slower at scale)
        rows = await neo4j.run_query(
            """
            MATCH (d:Document)-[:HAS_PAGE]->(p:Page)
            WHERE toLower(p.extracted_text) CONTAINS toLower($q)
            RETURN p.page_id AS page_id,
                   p.page_number AS page_number,
                   p.extracted_text AS extracted_text,
                   d.doc_id AS doc_id,
                   d.title AS document_title,
                   d.filename AS filename,
                   d.file_hash AS file_hash,
                   1.0 AS ft_score,
                   [(d)-[:IN_CATEGORY]->(c) | c.name] AS categories,
                   [(d)-[:TAGGED_WITH]->(t) | t.name] AS tags
            ORDER BY d.title, p.page_number
            LIMIT $limit
            """,
            {"q": body.query, "limit": body.limit},
        )
    hits = []
    for r in rows:
        # Extract a context snippet around the match
        text = r["extracted_text"] or ""
        idx = text.lower().find(body.query.lower())
        if idx >= 0:
            start = max(0, idx - 80)
            end = min(len(text), idx + len(body.query) + 120)
            snippet = ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")
        else:
            snippet = _snippet(text)

        hits.append({
            "page_id": r["page_id"],
            "doc_id": r["doc_id"],
            "document_title": r["document_title"],
            "filename": r["filename"],
            "page_number": r["page_number"],
            "score": 1.0,  # exact match, all equally relevant
            "text_snippet": snippet,
            "image_url": f"/images/{r['file_hash']}/{r['page_number']}",
            "reduced_image_url": f"/images/{r['file_hash']}/{r['page_number']}/reduced",
            "categories": r["categories"],
            "tags": r["tags"],
        })
    return ForgeResult(success=True, data=hits)


# ============================================================================
# RAG Answer — synthesize an answer from retrieved pages
# ============================================================================

class AnswerRequest(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=20, description="Pages to retrieve and read")
    search_mode: str = Field(
        default="auto",
        description="auto (keyword+visual+graph), keyword, visual, semantic, or hybrid",
    )
    use_vision: bool = Field(
        default=True,
        description="Send page IMAGES to the LLM instead of extracted text. "
        "Much more accurate (LLM reads the actual page) but slower.",
    )
    use_graph: bool = Field(
        default=True,
        description="Traverse the knowledge graph from query-matched entities to "
        "find related pages the user didn't ask about. Enables cross-document "
        "reasoning (material→process→standard chains).",
    )
    include_adjacent: bool = Field(
        default=True,
        description="Include pages N-1 and N+1 for each retrieved page. Catches "
        "tables and content that span page boundaries. Slightly slower but "
        "much more accurate for handbook-style content.",
    )


@router.post("/answer")
async def rag_answer(body: AnswerRequest, request: Request) -> ForgeResult:
    """Retrieve relevant pages and ask the LLM to synthesize an answer.

    The core RAG loop:
    1. Retrieve pages via keyword + ColPali visual search (or chosen mode)
    2. Send page IMAGES (not extracted text) to the vision LLM — the LLM
       reads the actual page including tables, diagrams, and formatting
    3. Return the LLM's answer with [Page N] citations

    This mirrors the accuracy of the original ColPali + Qwen-VL pipeline
    while adding the knowledge graph context.
    """
    neo4j = request.app.state.neo4j
    llm = getattr(request.app.state, "llm", None)
    colpali = getattr(request.app.state, "colpali", None)
    text_emb = getattr(request.app.state, "text_embedding", None)
    gpu = request.app.state.gpu
    settings = request.app.state.settings

    if llm is None:
        raise HTTPException(503, "LLM service not available")

    # Step 1: Retrieve pages
    # "auto" mode: run keyword + visual in parallel, merge results
    pages: list[dict] = []
    mode = body.search_mode

    if mode == "auto":
        # Keyword search for exact matches
        kw_result = await keyword_search(
            KeywordSearchRequest(query=body.query, limit=body.limit), request
        )
        kw_pages = kw_result.data or []

        # ColPali visual search for conceptual matches
        vis_pages: list[dict] = []
        if colpali is not None and text_emb is not None:
            try:
                vis_result = await visual_search(
                    VisualSearchRequest(
                        query=body.query, limit=body.limit, candidate_pool=30
                    ),
                    request,
                )
                vis_pages = vis_result.data or []
            except Exception:
                pass  # visual not available, keyword only

        # Merge: keyword hits first (exact match = highest priority), then
        # visual hits not already in keyword results
        seen_page_ids = {p["page_id"] for p in kw_pages}
        pages = list(kw_pages)
        for vp in vis_pages:
            if vp["page_id"] not in seen_page_ids:
                pages.append(vp)
                seen_page_ids.add(vp["page_id"])
        pages = pages[: body.limit]
    elif mode == "keyword":
        result = await keyword_search(
            KeywordSearchRequest(query=body.query, limit=body.limit), request
        )
        pages = result.data or []
    elif mode == "visual":
        if colpali is None or text_emb is None:
            raise HTTPException(503, "ColPali or text embedding not available")
        result = await visual_search(
            VisualSearchRequest(query=body.query, limit=body.limit, candidate_pool=30),
            request,
        )
        pages = result.data or []
    elif mode == "semantic":
        if text_emb is None:
            raise HTTPException(503, "Text embedding not available")
        result = await semantic_search(
            SemanticSearchRequest(query=body.query, limit=body.limit), request
        )
        pages = result.data or []
    elif mode == "hybrid":
        result = await hybrid_search(
            HybridSearchRequest(
                query=body.query, strategy="graph_boosted", limit=body.limit
            ),
            request,
        )
        pages = result.data or []

    # Step 1b: Graph exploration — traverse the knowledge graph for related pages
    graph_ctx = GraphContext()
    if body.use_graph:
        try:
            graph_ctx = await explore_from_query(
                body.query, neo4j, max_pages=body.limit * 2
            )
            # Add graph-discovered pages (that aren't already in search results)
            search_page_ids = {p["page_id"] for p in pages}
            for pid, reason in graph_ctx.page_ids.items():
                if pid not in search_page_ids:
                    # Create a minimal hit dict for this page
                    page_row = await neo4j.run_query(
                        """
                        MATCH (d:Document)-[:HAS_PAGE]->(p:Page {page_id: $pid})
                        RETURN p.page_id AS page_id, p.page_number AS page_number,
                               d.doc_id AS doc_id, d.title AS document_title,
                               d.filename AS filename, d.file_hash AS file_hash
                        """,
                        {"pid": pid},
                    )
                    if page_row:
                        r = page_row[0]
                        pages.append({
                            "page_id": r["page_id"],
                            "doc_id": r["doc_id"],
                            "document_title": r["document_title"],
                            "filename": r["filename"],
                            "page_number": r["page_number"],
                            "score": 0.5,  # graph-discovered
                            "text_snippet": f"[Graph: {reason}]",
                            "image_url": f"/images/{r['file_hash']}/{r['page_number']}",
                            "reduced_image_url": f"/images/{r['file_hash']}/{r['page_number']}/reduced",
                        })
                        search_page_ids.add(pid)
        except Exception as exc:
            logger.warning("Graph exploration failed (continuing without): %s", exc)

    pages = pages[: body.limit]

    if not pages:
        return ForgeResult(
            success=True,
            data={
                "answer": "No relevant pages found for this query.",
                "sources": [],
            },
        )

    # Step 2: Build LLM context — page images (vision) or extracted text
    # For each retrieved page, also include adjacent pages (N-1, N+1) so the
    # VLM can read tables and content that span page boundaries. This fixes
    # the common "table starts on page 997 but the row I need is on 998"
    # problem. We deduplicate so a page isn't sent twice if two search hits
    # are adjacent.
    import base64
    from pathlib import Path

    data_dir = Path(settings.server.data_dir)
    messages_content: list[dict] = []
    sources = []
    sent_pages: set[tuple[str, int]] = set()  # (doc_hash, page_number) dedup

    async def _add_page_image(doc_hash: str, pn: int, title: str, label: str) -> bool:
        """Add a page image to the LLM context. Returns True if added."""
        key = (doc_hash, pn)
        if key in sent_pages or pn < 1:
            return False
        img_path = data_dir / "reduced_images" / doc_hash / f"page_{str(pn).zfill(4)}.jpg"
        if not img_path.exists():
            return False
        sent_pages.add(key)
        img_bytes = img_path.read_bytes()
        b64 = base64.b64encode(img_bytes).decode("ascii")
        messages_content.append(
            {"type": "text", "text": f"[{label} — Page {pn} from {title}]"}
        )
        messages_content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )
        return True

    for p in pages[: body.limit]:
        page_rows = await neo4j.run_query(
            """
            MATCH (d:Document)-[:HAS_PAGE]->(pg:Page {page_id: $pid})
            RETURN pg.extracted_text AS text, pg.page_number AS pn,
                   pg.reduced_image_path AS img_path,
                   d.title AS title, d.file_hash AS hash
            """,
            {"pid": p["page_id"]},
        )
        if not page_rows:
            continue
        pr = page_rows[0]
        doc_hash = pr["hash"]
        pn = pr["pn"]
        title = pr["title"]
        sources.append({
            "document_title": title,
            "page_number": pn,
            "image_url": p.get("image_url", ""),
            "score": p.get("score", 0),
        })

        if body.use_vision:
            # Send the matched page, optionally with adjacent pages
            if body.include_adjacent:
                await _add_page_image(doc_hash, pn - 1, title, "Context (prev page)")
            await _add_page_image(doc_hash, pn, title, "Source")
            if body.include_adjacent:
                await _add_page_image(doc_hash, pn + 1, title, "Context (next page)")
            continue

        # Fallback: send extracted text
        text = (pr["text"] or "")[:3000]
        messages_content.append(
            {
                "type": "text",
                "text": f"[Page {pr['pn']} from {pr['title']}]\n{text}",
            }
        )

    # Step 3: Build graph context summary for the LLM
    graph_summary = ""
    if body.use_graph and (
        graph_ctx.reasoning_chains
        or graph_ctx.materials
        or graph_ctx.processes
        or graph_ctx.standards
    ):
        parts = ["KNOWLEDGE GRAPH CONTEXT (relationships discovered from the engineering database):"]
        if graph_ctx.reasoning_chains:
            parts.append("Relationship chains:")
            for chain in graph_ctx.reasoning_chains[:15]:
                parts.append(f"  • {chain}")
        if graph_ctx.materials:
            mat_names = [m["name"] for m in graph_ctx.materials[:10]]
            parts.append(f"Related materials: {', '.join(mat_names)}")
        if graph_ctx.processes:
            proc_names = [p["name"] for p in graph_ctx.processes[:10]]
            parts.append(f"Related processes: {', '.join(proc_names)}")
        if graph_ctx.standards:
            std_names = [s["name"] for s in graph_ctx.standards[:10]]
            parts.append(f"Related standards: {', '.join(std_names)}")
        if graph_ctx.community_summaries:
            parts.append("Topic summaries:")
            for cs in graph_ctx.community_summaries[:2]:
                parts.append(f"  {cs[:300]}")
        graph_summary = "\n".join(parts)

    # Build the question with graph context
    question_text = f"Question: {body.query}\n\n"
    if graph_summary:
        question_text += (
            f"{graph_summary}\n\n"
            "Use the knowledge graph context above to inform your answer. "
            "The graph shows how materials, processes, and standards relate "
            "to each other — mention relevant connections even if the user "
            "didn't specifically ask about them.\n\n"
        )
    question_text += (
        "Answer the question based on the source pages and graph context. "
        "Cite page numbers inline as [Page N]. Be precise with numbers, "
        "codes, and specifications. If you notice relevant cross-references "
        "or related considerations (e.g., applicable standards, compatible "
        "processes, known limitations), mention them proactively.\n\n/no_think"
    )

    messages_content.append({"type": "text", "text": question_text})

    messages = [
        {
            "role": "system",
            "content": "You are an engineering reference assistant with access to "
            "a knowledge graph of materials, processes, standards, and equipment. "
            "You can see page images from engineering handbooks. Read ALL pages "
            "carefully including tables, diagrams, and figure captions. "
            "Pages labeled 'Context (prev page)' or 'Context (next page)' are "
            "adjacent pages included because tables and data often span page "
            "boundaries — check them for continuation of tables, footnotes, "
            "and additional data. "
            "Answer questions based on the source pages AND the knowledge graph "
            "context. When you see a relationship chain (e.g., Material → "
            "GOVERNED_BY → Standard), use it to provide comprehensive answers "
            "that connect information across different sections of the handbook. "
            "Always cite the specific page number where you found each fact.",
        },
        {"role": "user", "content": messages_content},
    ]

    try:
        answer = await llm.chat(messages, max_tokens=2048, temperature=0.1)
    except Exception as exc:
        answer = f"LLM error: {exc}"

    return ForgeResult(
        success=True,
        data={
            "answer": answer,
            "sources": sources,
            "query": body.query,
            "search_mode": mode,
            "used_vision": body.use_vision,
            "used_graph": body.use_graph,
            "graph_context": {
                "materials_found": len(graph_ctx.materials),
                "processes_found": len(graph_ctx.processes),
                "standards_found": len(graph_ctx.standards),
                "reasoning_chains": graph_ctx.reasoning_chains[:10],
                "pages_from_graph": len([
                    p for p in pages
                    if isinstance(p.get("text_snippet"), str)
                    and p["text_snippet"].startswith("[Graph:")
                ]),
            } if body.use_graph else None,
        },
    )


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
    from backend.services.nemotron_service import deserialize_nemotron
    # Both deserializers have the same interface: (blob, K, D) -> ndarray
    # MaxSim is model-agnostic — works with any (K, D) embeddings

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
