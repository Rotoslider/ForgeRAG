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
    if filters.collection:
        parts.append("coalesce(d.collection, 'default') = $collection")
        params["collection"] = filters.collection
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
    # Build a Lucene query that starts with a boosted phrase match and
    # falls back to OR-of-terms. This catches exact-hit cases like
    # "ASTM A 709" while still returning pages that share most of the
    # query's tokens (e.g., "wire gauge for a 20 amp circuit" no longer
    # gets zero results just because the full string isn't in any doc).
    escaped = body.query.replace('"', '\\"')
    tokens = [t for t in body.query.split() if t]
    escaped_tokens = [t.replace('"', '\\"') for t in tokens if len(t) >= 2]
    # Phrase match ^4 (higher weight) OR any-of-terms
    if escaped_tokens:
        or_clause = " OR ".join(escaped_tokens)
        phrase_query = f'"{escaped}"^4 OR ({or_clause})'
    else:
        phrase_query = f'"{escaped}"'
    try:
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
    # "auto" mode: chunk-aware RRF hybrid (BGE-M3 dense + BM25 + reranker)
    # fused with Nemotron visual results. This is a meaningful upgrade over
    # the earlier "keyword + visual" auto mode — hybrid catches semantic
    # paraphrases that plain keyword misses, and the reranker cleans up
    # ordering. Visual retrieval still runs alongside to catch chart/diagram
    # matches that aren't in the text.
    pages: list[dict] = []
    mode = body.search_mode

    if mode == "auto":
        # Primary: chunk-level RRF hybrid
        try:
            rrf_result = await hybrid_search(
                HybridSearchRequest(
                    query=body.query,
                    strategy="rrf",
                    limit=body.limit,
                    filters=body.filters if hasattr(body, "filters") else None,
                    rerank=True,
                ),
                request,
            )
            primary_pages = rrf_result.data or []
        except Exception as exc:
            logger.warning("RRF retrieval failed, falling back to keyword: %s", exc)
            kw_result = await keyword_search(
                KeywordSearchRequest(query=body.query, limit=body.limit), request
            )
            primary_pages = kw_result.data or []

        # Secondary: visual search (charts, diagrams, tables a reranker
        # might miss because the textual form doesn't match).
        vis_pages: list[dict] = []
        if colpali is not None and text_emb is not None:
            try:
                vis_result = await visual_search(
                    VisualSearchRequest(
                        query=body.query, limit=max(3, body.limit // 2),
                        candidate_pool=30,
                    ),
                    request,
                )
                vis_pages = vis_result.data or []
            except Exception:
                pass

        # Merge: RRF hits first (text-precise), then visual hits not
        # already covered.
        seen_page_ids = {p["page_id"] for p in primary_pages}
        pages = list(primary_pages)
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
        # Default the hybrid strategy to `rrf` (chunk-aware) for the best
        # general-purpose retrieval. Callers can still POST to /search/hybrid
        # directly to pick another strategy.
        result = await hybrid_search(
            HybridSearchRequest(
                query=body.query, strategy="rrf", limit=body.limit,
                rerank=True,
            ),
            request,
        )
        pages = result.data or []

    # Step 1b: Graph exploration — traverse the knowledge graph for related pages.
    # Seeded from the entities actually mentioned on our top retrieval hits,
    # not from query-term keyword matching. This prevents noise words like
    # "wire" or "amp" in a general-English query from flooding the context
    # with welding chains pulled in via substring collisions.
    graph_ctx = GraphContext()
    seed_pids = [p["page_id"] for p in pages if p.get("page_id")][: body.limit]
    if body.use_graph:
        try:
            graph_ctx = await explore_from_query(
                body.query, neo4j,
                max_pages=body.limit * 2,
                seed_page_ids=seed_pids if seed_pids else None,
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
        # Each image is introduced by a distinctive ID token ("#NNN") that
        # cannot be confused with the printed page number in the PDF
        # header/footer. The LLM is instructed in the system prompt to
        # cite using "#NNN" so we get a clean signal independent of the
        # book's internal page numbering.
        messages_content.append(
            {
                "type": "text",
                "text": (
                    f"[{label} | IMG_ID #{pn} | {title}] "
                    f"— cite facts from this image as [#{pn}]. "
                    f"Do NOT use the page number printed in the header or footer."
                ),
            }
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

        # If the retrieval returned a specific matching chunk on this
        # page, surface its summary + section path so the VLM knows which
        # slice of the image matters. RRF retrieval populates these fields;
        # older retrievers (keyword, visual) don't — in those cases we
        # just label the page as "Source".
        chunk_hint_parts: list[str] = []
        if p.get("summary"):
            chunk_hint_parts.append(f"Relevant chunk summary: {p['summary']}")
        if p.get("section_path"):
            chunk_hint_parts.append(
                "Section: " + " › ".join(p["section_path"])
            )
        if p.get("chunk_type") and p["chunk_type"] not in (None, "text"):
            chunk_hint_parts.append(f"Chunk type: {p['chunk_type']}")
        if chunk_hint_parts:
            messages_content.append(
                {"type": "text", "text": "\n".join(chunk_hint_parts)}
            )

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

    # Step 3: Build graph context summary for the LLM.
    #
    # Capped tightly to keep topical drift out of the prompt — a flood of
    # tangentially-related chains (e.g. material → process → standard
    # walks on entities that only appear once on a single retrieved page)
    # biases the LLM more than it helps it. We show at most 5 chains and
    # the top ~6 entities per category.
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
            for chain in graph_ctx.reasoning_chains[:5]:
                parts.append(f"  • {chain}")
        if graph_ctx.materials:
            mat_names = [m["name"] for m in graph_ctx.materials[:6]]
            parts.append(f"Related materials: {', '.join(mat_names)}")
        if graph_ctx.processes:
            proc_names = [p["name"] for p in graph_ctx.processes[:6]]
            parts.append(f"Related processes: {', '.join(proc_names)}")
        if graph_ctx.standards:
            std_names = [s["name"] for s in graph_ctx.standards[:6]]
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
        "Answer the question based on the source pages and graph context.\n"
        "\n"
        "CITATION RULES — CRITICAL. The user's page viewer only works with "
        "the IMG_ID tokens below.\n"
        "- Each page image is introduced by a label like "
        "'[Source | IMG_ID #98 | Metal Forming Handbook]'.\n"
        "- When referring to a fact from an image, cite it as '[#98]' using "
        "that IMG_ID integer — NOT 'Page 80', NOT the number printed in "
        "the page header or footer. The printed number is usually wrong "
        "because of front matter or chapter-relative numbering.\n"
        "- For multiple images, write '[#98, #99]' or '[#98-#99]'.\n"
        "- Do NOT write 'Page 80' or 'p.80' anywhere — always use the "
        "'#N' form with the IMG_ID.\n"
        "\n"
        "Be precise with numbers, codes, and specifications. If you notice "
        "relevant cross-references or related considerations (e.g., applicable "
        "standards, compatible processes, known limitations), mention them "
        "proactively.\n\n/no_think"
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
            "CITE IMAGES USING ONLY THE IMG_ID from the label that introduces "
            "each image (e.g. [Source | IMG_ID #98 | <title>]). Write the "
            "citation as '[#98]'. Never write 'Page 80' or use the number "
            "printed in the page's header or footer — that number is usually "
            "wrong for the viewer's linking because of front matter and "
            "chapter-relative numbering.",
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
    """Semantic text search via the chunk vector index.

    Queries chunk_embedding (BGE-M3 1024-dim) and dedupes to unique
    pages. Falls back to page_text_embedding only if the chunk index
    returns nothing — useful for docs that haven't been rebuilt into
    chunks yet (though those also need 1024-dim page embeddings for
    the fallback to work; currently page embeddings are stale 768-dim,
    so the fallback is a no-op until we rerun text embedding on all
    pages — the rebuild_chunks script populates chunks but not pages).
    """
    text_emb = getattr(request.app.state, "text_embedding", None)
    neo4j = request.app.state.neo4j
    if text_emb is None:
        raise HTTPException(503, "Text embedding service not available")

    gpu = request.app.state.gpu
    async with gpu.load_scope("text_embedding"):
        query_vec = await asyncio.to_thread(text_emb.embed_query, body.query)

    # Over-fetch from the vector index so dedupe-to-page filtering still
    # leaves us with enough results.
    topk = max(body.limit * 5, 20)

    filter_where, filter_params = _filter_clauses(body.filters)
    where = f" WHERE {filter_where}" if filter_where else ""

    # Primary: chunk vector search, grouped by page (best chunk wins)
    cypher = f"""
        CALL db.index.vector.queryNodes('chunk_embedding', $topk, $query_vec)
        YIELD node AS c, score
        MATCH (p:Page)-[:HAS_CHUNK]->(c)
        MATCH (d:Document)-[:HAS_PAGE]->(p){where}
        WITH p, d, max(score) AS score,
             collect(c)[0] AS best_chunk
        RETURN p.page_id AS page_id,
               p.page_number AS page_number,
               p.extracted_text AS extracted_text,
               best_chunk.summary AS summary,
               best_chunk.chunk_type AS chunk_type,
               d.doc_id AS doc_id,
               d.title AS document_title,
               d.filename AS filename,
               d.file_hash AS file_hash,
               score AS score,
               [(d)-[:IN_CATEGORY]->(c2) | c2.name] AS categories,
               [(d)-[:TAGGED_WITH]->(t) | t.name] AS tags
        ORDER BY score DESC
        LIMIT $limit
    """
    params = {"topk": topk, "query_vec": query_vec.tolist(), "limit": body.limit}
    params.update(filter_params)

    rows = await neo4j.run_query(cypher, params)

    hits = []
    for r in rows:
        # Prefer chunk summary as snippet if available (more precise),
        # else fall back to extracted page text.
        snippet_source = r.get("summary") or r.get("extracted_text")
        hits.append({
            "page_id": r["page_id"],
            "doc_id": r["doc_id"],
            "document_title": r["document_title"],
            "filename": r["filename"],
            "page_number": r["page_number"],
            "score": float(r["score"]),
            "text_snippet": _snippet(snippet_source),
            "summary": r.get("summary"),
            "chunk_type": r.get("chunk_type"),
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

    # Stage 1: coarse candidates. Prefer the chunk vector index (populated
    # and correct dim) so text-relevant pages surface first. Fall back to
    # full-text BM25 if chunk search returns nothing — this way visual
    # retrieval still finds pages in docs that haven't been chunked yet.
    async with gpu.load_scope("text_embedding"):
        tvec = await asyncio.to_thread(text_emb.embed_query, body.query)

    filter_where, filter_params = _filter_clauses(body.filters)
    where = f" WHERE {filter_where}" if filter_where else ""

    cand_cypher = f"""
        CALL db.index.vector.queryNodes('chunk_embedding', $pool, $vec)
        YIELD node AS c, score AS coarse_score
        MATCH (p:Page)-[:HAS_CHUNK]->(c)
        MATCH (d:Document)-[:HAS_PAGE]->(p){where}
        WHERE p.colpali_vector_count IS NOT NULL AND p.colpali_vector_count > 0
        WITH p, d, max(coarse_score) AS coarse_score
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
               [(d)-[:IN_CATEGORY]->(c2) | c2.name] AS categories,
               [(d)-[:TAGGED_WITH]->(t) | t.name] AS tags
        ORDER BY coarse_score DESC
        LIMIT $pool
    """
    params = {"pool": body.candidate_pool, "vec": tvec.tolist()}
    params.update(filter_params)
    candidates = await neo4j.run_query(cand_cypher, params)

    # Fallback: if no chunks matched, cast a wide net via page full-text
    # BM25 (useful for non-chunked docs). ColPali MaxSim in stage 2 will
    # still rank by visual similarity.
    if not candidates:
        terms = [t for t in body.query.split() if t]
        ft_query = (
            " OR ".join(t.replace('"', '\\"') for t in terms if len(t) >= 2)
            if len(terms) > 3
            else f'"{body.query.replace(chr(34), chr(92) + chr(34))}"'
        )
        fallback_cypher = f"""
            CALL db.index.fulltext.queryNodes('page_text_fulltext', $q)
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
                   [(d)-[:IN_CATEGORY]->(c2) | c2.name] AS categories,
                   [(d)-[:TAGGED_WITH]->(t) | t.name] AS tags
            ORDER BY coarse_score DESC
            LIMIT $pool
        """
        fallback_params = {"q": ft_query, "pool": body.candidate_pool}
        fallback_params.update(filter_params)
        try:
            candidates = await neo4j.run_query(fallback_cypher, fallback_params)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Visual-search fallback BM25 failed: %s", exc)
            candidates = []

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
# Chunk-level retrieval (Phase 9)
# ============================================================================


class ChunkSearchRequest(BaseModel):
    """Chunk-level retrieval with BGE-M3 dense + BM25 + reranker.

    Returns raw chunk records (text + summary + section_path + page/doc
    linkage) so agents can quote precisely rather than re-reading whole
    pages. Skips the VLM — use /search/answer when you want a synthesized
    answer.
    """

    query: str
    limit: int = Field(default=10, ge=1, le=50)
    filters: SearchFilters | None = None
    rerank: bool = True
    rerank_pool: int = Field(default=50, ge=5, le=200)
    chunk_type: str | None = Field(
        None,
        description="Optional: filter to a specific chunk type ("
        "'text', 'table', 'figure', 'equation', 'list', 'caption').",
    )


@router.post("/chunks")
async def search_chunks(body: ChunkSearchRequest, request: Request) -> ForgeResult:
    """Chunk-level retrieval — BGE-M3 dense + Lucene BM25 + bge-reranker.

    Queries only Chunk nodes (not Page), so docs that haven't been
    rebuilt into chunks yet won't surface here — use /search/keyword or
    /search/hybrid (strategy=rrf) for mixed coverage. This endpoint is
    designed for agents that want paragraph/table-precise evidence.
    """
    text_emb = getattr(request.app.state, "text_embedding", None)
    neo4j = request.app.state.neo4j
    gpu = request.app.state.gpu
    reranker = getattr(request.app.state, "reranker", None)
    if text_emb is None:
        raise HTTPException(503, "Text embedding service not available")

    async with gpu.load_scope("text_embedding"):
        qvec = await asyncio.to_thread(text_emb.embed_query, body.query)

    filter_where, filter_params = _filter_clauses(body.filters)
    where_parts = [filter_where] if filter_where else []
    if body.chunk_type:
        where_parts.append("c.chunk_type = $chunk_type")
        filter_params["chunk_type"] = body.chunk_type
    where = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""

    pool = max(body.rerank_pool, body.limit * 3)
    K_RRF = 60

    # Build the Lucene query
    terms = [t for t in body.query.split() if t]
    if len(terms) <= 3:
        ft_query = f'"{body.query.replace(chr(34), chr(92) + chr(34))}"'
    else:
        ft_query = " OR ".join(t.replace('"', '\\"') for t in terms if len(t) >= 2)

    dense_cypher = f"""
        CALL db.index.vector.queryNodes('chunk_embedding', $pool, $vec)
        YIELD node AS c, score AS dense_score
        MATCH (p:Page)-[:HAS_CHUNK]->(c)
        MATCH (d:Document)-[:HAS_PAGE]->(p){where}
        RETURN c.chunk_id AS chunk_id, dense_score,
               c.text AS text, c.summary AS summary,
               c.chunk_type AS chunk_type, c.section_path AS section_path,
               p.page_id AS page_id, p.page_number AS page_number,
               d.doc_id AS doc_id, d.title AS document_title,
               d.filename AS filename, d.file_hash AS file_hash,
               [(d)-[:IN_CATEGORY]->(cat) | cat.name] AS categories,
               [(d)-[:TAGGED_WITH]->(t) | t.name] AS tags
        ORDER BY dense_score DESC
    """
    ft_cypher = f"""
        CALL db.index.fulltext.queryNodes('chunk_text_fulltext', $q)
        YIELD node AS c, score AS ft_score
        MATCH (p:Page)-[:HAS_CHUNK]->(c)
        MATCH (d:Document)-[:HAS_PAGE]->(p){where}
        RETURN c.chunk_id AS chunk_id, ft_score
        ORDER BY ft_score DESC
        LIMIT $pool
    """
    dense_params = {"pool": pool, "vec": qvec.tolist()}
    dense_params.update(filter_params)
    ft_params = {"q": ft_query, "pool": pool}
    ft_params.update(filter_params)

    async def _safe(cypher: str, params: dict) -> list:
        try:
            return await neo4j.run_query(cypher, params)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chunk retrieval subquery failed: %s", exc)
            return []

    dense_rows, ft_rows = await asyncio.gather(
        _safe(dense_cypher, dense_params),
        _safe(ft_cypher, ft_params),
    )

    records: dict[str, dict] = {r["chunk_id"]: dict(r) for r in dense_rows}
    rrf_scores: dict[str, float] = {}
    for rank, r in enumerate(dense_rows, start=1):
        cid = r["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (K_RRF + rank)

    # Hydrate chunks that only came in via BM25
    ft_misses = [r["chunk_id"] for r in ft_rows if r["chunk_id"] not in records]
    if ft_misses:
        hyd = await neo4j.run_query(
            """
            MATCH (c:Chunk)<-[:HAS_CHUNK]-(p:Page)
            WHERE c.chunk_id IN $ids
            MATCH (d:Document)-[:HAS_PAGE]->(p)
            RETURN c.chunk_id AS chunk_id, c.text AS text, c.summary AS summary,
                   c.chunk_type AS chunk_type, c.section_path AS section_path,
                   p.page_id AS page_id, p.page_number AS page_number,
                   d.doc_id AS doc_id, d.title AS document_title,
                   d.filename AS filename, d.file_hash AS file_hash,
                   [(d)-[:IN_CATEGORY]->(cat) | cat.name] AS categories,
                   [(d)-[:TAGGED_WITH]->(t) | t.name] AS tags
            """,
            {"ids": ft_misses},
        )
        for row in hyd:
            records[row["chunk_id"]] = dict(row)
    for rank, r in enumerate(ft_rows, start=1):
        cid = r["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (K_RRF + rank)

    fused = sorted(
        records.values(),
        key=lambda r: rrf_scores.get(r["chunk_id"], 0.0),
        reverse=True,
    )

    # Cross-encoder rerank
    if body.rerank and reranker is not None and fused:
        cands = fused[: body.rerank_pool]
        passages = [(r.get("text") or "")[:1800] for r in cands]
        try:
            async with gpu.load_scope("reranker"):
                scores = await asyncio.to_thread(
                    reranker.score_pairs, body.query, passages
                )
            for rec, s in zip(cands, scores):
                rec["rerank_score"] = float(s)
            cands.sort(
                key=lambda r: r.get("rerank_score", float("-inf")), reverse=True,
            )
            fused = cands + fused[body.rerank_pool:]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reranker failed: %s", exc)

    hits = []
    for r in fused[: body.limit]:
        cid = r["chunk_id"]
        hits.append({
            "chunk_id": cid,
            "page_id": r.get("page_id"),
            "page_number": r.get("page_number"),
            "chunk_type": r.get("chunk_type"),
            "section_path": r.get("section_path") or [],
            "text": r.get("text") or "",
            "summary": r.get("summary"),
            "doc_id": r.get("doc_id"),
            "document_title": r.get("document_title"),
            "filename": r.get("filename"),
            "score": float(r.get("rerank_score", rrf_scores.get(cid, 0.0))),
            "rrf_score": float(rrf_scores.get(cid, 0.0)),
            "rerank_score": r.get("rerank_score"),
            "dense_score": r.get("dense_score"),
            "image_url": f"/images/{r.get('file_hash')}/{r.get('page_number')}",
            "reduced_image_url": f"/images/{r.get('file_hash')}/{r.get('page_number')}/reduced",
            "categories": r.get("categories") or [],
            "tags": r.get("tags") or [],
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
        # Chunk vector candidates (deduped to unique pages), then boost
        # score by number of entity names / aliases from the query
        # appearing on the page's graph edges. Previously used the stale
        # page_text_embedding index and returned nothing post-BGE-M3
        # migration — rewritten to go through chunk_embedding.
        query_terms = [t.lower() for t in body.query.split() if len(t) >= 3]

        cypher = f"""
            CALL db.index.vector.queryNodes('chunk_embedding', $pool, $vec)
            YIELD node AS c, score AS chunk_score
            MATCH (p:Page)-[:HAS_CHUNK]->(c)
            WITH p, max(chunk_score) AS base_score
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
        # Pure vector search (via chunks), enriched with entities and
        # community membership. Chunks dedupe to unique pages; the page's
        # best chunk score is used as the page score.
        cypher = f"""
            CALL db.index.vector.queryNodes('chunk_embedding', $pool, $vec)
            YIELD node AS c, score AS chunk_score
            MATCH (p:Page)-[:HAS_CHUNK]->(c)
            WITH p, max(chunk_score) AS score
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

    if body.strategy == "rrf":
        # Reciprocal Rank Fusion over THREE retrievers:
        #
        #   (1) Chunk dense   — BGE-M3 embeddings on (summary + text) via
        #                       the chunk_embedding vector index
        #   (2) Chunk BM25    — Lucene full-text on chunk_text_fulltext
        #                       (covers chunk.text AND chunk.summary)
        #   (3) Page BM25     — Lucene full-text on page_text_fulltext
        #                       (fallback for docs that haven't been
        #                       rebuilt into chunks yet)
        #
        # k=60 is the standard RRF constant. Each retriever contributes
        # 1/(k+rank) to the fused score of whichever Page the hit maps
        # to. Chunks are always deduped-to-page for the final response
        # — the VLM reads page images, not isolated chunks — but the
        # winning chunk's summary and section path are attached as
        # context so the LLM knows *which* slice of the page matters.
        #
        # Optional cross-encoder reranking runs over the fused pool
        # using the chunk text (if available) or page text as the
        # candidate passage.
        K_RRF = 60
        pool = max(body.rerank_pool, body.limit * 3)

        # Filter-where operates on the Document; we reuse filter_params
        # verbatim for all three Cypher calls.
        where = f" WHERE {filter_where}" if filter_where else ""

        # Lucene query construction (same for both BM25 retrievers).
        terms = [t for t in body.query.split() if t]
        if len(terms) <= 3:
            escaped = body.query.replace('"', '\\"')
            ft_query = f'"{escaped}"'
        else:
            escaped_terms = [t.replace('"', '\\"') for t in terms if len(t) >= 2]
            ft_query = " OR ".join(escaped_terms)

        # (1) Chunk dense
        chunk_dense_cypher = f"""
            CALL db.index.vector.queryNodes('chunk_embedding', $pool, $vec)
            YIELD node AS c, score AS dense_score
            MATCH (p:Page)-[:HAS_CHUNK]->(c)
            MATCH (d:Document)-[:HAS_PAGE]->(p){where}
            RETURN c.chunk_id AS chunk_id, dense_score,
                   c.text AS text, c.summary AS summary,
                   c.chunk_type AS chunk_type, c.section_path AS section_path,
                   p.page_id AS page_id, p.page_number AS page_number,
                   p.extracted_text AS page_text,
                   d.doc_id AS doc_id, d.title AS document_title,
                   d.filename AS filename, d.file_hash AS file_hash,
                   [(d)-[:IN_CATEGORY]->(cat) | cat.name] AS categories,
                   [(d)-[:TAGGED_WITH]->(t) | t.name] AS tags
            ORDER BY dense_score DESC
        """
        chunk_dense_params = {"pool": pool, "vec": qvec.tolist()}
        chunk_dense_params.update(filter_params)

        # (2) Chunk BM25
        chunk_ft_cypher = f"""
            CALL db.index.fulltext.queryNodes('chunk_text_fulltext', $q)
            YIELD node AS c, score AS ft_score
            MATCH (p:Page)-[:HAS_CHUNK]->(c)
            MATCH (d:Document)-[:HAS_PAGE]->(p){where}
            RETURN c.chunk_id AS chunk_id, ft_score
            ORDER BY ft_score DESC
            LIMIT $pool
        """
        chunk_ft_params = {"q": ft_query, "pool": pool}
        chunk_ft_params.update(filter_params)

        # (3) Page BM25 — fallback for non-chunked docs
        page_ft_cypher = f"""
            CALL db.index.fulltext.queryNodes('page_text_fulltext', $q)
            YIELD node AS p, score AS ft_score
            MATCH (d:Document)-[:HAS_PAGE]->(p){where}
            RETURN p.page_id AS page_id, ft_score
            ORDER BY ft_score DESC
            LIMIT $pool
        """
        page_ft_params = {"q": ft_query, "pool": pool}
        page_ft_params.update(filter_params)

        import asyncio as _asyncio
        async def _safe(cypher: str, params: dict) -> list:
            try:
                return await neo4j.run_query(cypher, params)
            except Exception as exc:  # noqa: BLE001
                logger.warning("RRF retriever failed (continuing): %s", exc)
                return []

        chunk_dense_rows, chunk_ft_rows, page_ft_rows = await _asyncio.gather(
            _safe(chunk_dense_cypher, chunk_dense_params),
            _safe(chunk_ft_cypher, chunk_ft_params),
            _safe(page_ft_cypher, page_ft_params),
        )

        # Build a page-id → record map. The "best" chunk for each page
        # (first one seen, since chunks are ranked) populates the
        # chunk-level fields. Pages that only came in via page_text
        # fallback get empty chunk fields.
        records: dict[str, dict] = {}
        rrf_scores: dict[str, float] = {}

        def _ensure_record(page_id: str) -> dict:
            return records.setdefault(page_id, {"page_id": page_id})

        # (1) Chunk dense — populates per-page record
        for rank, r in enumerate(chunk_dense_rows, start=1):
            pid = r["page_id"]
            rec = _ensure_record(pid)
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (K_RRF + rank)
            # Only overwrite chunk fields from a higher-ranked chunk on
            # the same page. Since rows are already sorted by dense_score,
            # the first arrival is the best.
            if "chunk_id" not in rec:
                for key in ("chunk_id", "chunk_type", "text", "summary",
                            "section_path", "page_number", "doc_id",
                            "document_title", "filename", "file_hash",
                            "categories", "tags", "dense_score"):
                    rec[key] = r.get(key)

        # (2) Chunk BM25 — may promote chunks not in the dense top-K
        # into the pool. Need a second query to hydrate their metadata.
        chunk_ft_hydrate: list[str] = []
        for rank, r in enumerate(chunk_ft_rows, start=1):
            chunk_id = r["chunk_id"]
            # Increment RRF score via page_id, but we need to resolve
            # chunk_id → page_id first. Do that in bulk after.
            chunk_ft_hydrate.append((rank, chunk_id))  # type: ignore[arg-type]
        if chunk_ft_hydrate:
            hydrate_rows = await neo4j.run_query(
                """
                MATCH (c:Chunk)<-[:HAS_CHUNK]-(p:Page)
                WHERE c.chunk_id IN $ids
                MATCH (d:Document)-[:HAS_PAGE]->(p)
                RETURN c.chunk_id AS chunk_id,
                       c.text AS text, c.summary AS summary,
                       c.chunk_type AS chunk_type, c.section_path AS section_path,
                       p.page_id AS page_id, p.page_number AS page_number,
                       p.extracted_text AS page_text,
                       d.doc_id AS doc_id, d.title AS document_title,
                       d.filename AS filename, d.file_hash AS file_hash,
                       [(d)-[:IN_CATEGORY]->(cat) | cat.name] AS categories,
                       [(d)-[:TAGGED_WITH]->(t) | t.name] AS tags
                """,
                {"ids": [cid for _, cid in chunk_ft_hydrate]},
            )
            hydrate_by_chunk = {r["chunk_id"]: r for r in hydrate_rows}
            for rank, chunk_id in chunk_ft_hydrate:
                row = hydrate_by_chunk.get(chunk_id)
                if not row:
                    continue
                pid = row["page_id"]
                rec = _ensure_record(pid)
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (K_RRF + rank)
                if "chunk_id" not in rec:
                    for key in ("chunk_id", "chunk_type", "text", "summary",
                                "section_path", "page_number", "doc_id",
                                "document_title", "filename", "file_hash",
                                "categories", "tags"):
                        rec[key] = row.get(key)

        # (3) Page BM25 — fallback for pages that never got chunked
        page_ft_hydrate: list[str] = []
        for rank, r in enumerate(page_ft_rows, start=1):
            pid = r["page_id"]
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (K_RRF + rank)
            rec = _ensure_record(pid)
            if "doc_id" not in rec:  # needs hydration
                page_ft_hydrate.append(pid)
        if page_ft_hydrate:
            hydrate_rows = await neo4j.run_query(
                """
                MATCH (d:Document)-[:HAS_PAGE]->(p:Page)
                WHERE p.page_id IN $ids
                RETURN p.page_id AS page_id, p.page_number AS page_number,
                       p.extracted_text AS page_text,
                       d.doc_id AS doc_id, d.title AS document_title,
                       d.filename AS filename, d.file_hash AS file_hash,
                       [(d)-[:IN_CATEGORY]->(cat) | cat.name] AS categories,
                       [(d)-[:TAGGED_WITH]->(t) | t.name] AS tags
                """,
                {"ids": page_ft_hydrate},
            )
            for row in hydrate_rows:
                rec = _ensure_record(row["page_id"])
                for key in ("page_number", "page_text", "doc_id",
                            "document_title", "filename", "file_hash",
                            "categories", "tags"):
                    if rec.get(key) is None:
                        rec[key] = row.get(key)

        # Fused ranking.
        fused = sorted(
            records.values(),
            key=lambda r: rrf_scores.get(r["page_id"], 0.0),
            reverse=True,
        )

        # Cross-encoder reranking (optional).
        reranker = getattr(request.app.state, "reranker", None)
        if body.rerank and reranker is not None and fused:
            rerank_candidates = fused[: body.rerank_pool]
            # Prefer chunk text (more precise) over page text; cap at
            # 1800 chars so the cross-encoder's 512-token context limit
            # doesn't truncate mid-sentence after tokenization.
            passages = [
                ((r.get("text") or r.get("page_text") or "")[:1800])
                for r in rerank_candidates
            ]
            try:
                async with gpu.load_scope("reranker"):
                    import asyncio as _asyncio2
                    scores = await _asyncio2.to_thread(
                        reranker.score_pairs, body.query, passages
                    )
                for rec, s in zip(rerank_candidates, scores):
                    rec["rerank_score"] = float(s)
                rerank_candidates.sort(
                    key=lambda r: r.get("rerank_score", float("-inf")),
                    reverse=True,
                )
                tail = fused[body.rerank_pool:]
                fused = rerank_candidates + tail
            except Exception as exc:  # noqa: BLE001
                logger.warning("Reranker failed, using RRF order: %s", exc)

        hits = []
        for r in fused[: body.limit]:
            pid = r["page_id"]
            chunk_id = r.get("chunk_id")
            # Use chunk text for the snippet when available, else page
            # text. Summary is surfaced as a separate field so callers
            # (especially /search/answer) can use it as LLM context.
            snippet_source = r.get("text") or r.get("page_text")
            hits.append({
                "page_id": pid,
                "chunk_id": chunk_id,
                "chunk_type": r.get("chunk_type"),
                "section_path": r.get("section_path") or [],
                "summary": r.get("summary"),
                "doc_id": r.get("doc_id"),
                "document_title": r.get("document_title"),
                "filename": r.get("filename"),
                "page_number": r.get("page_number"),
                "score": float(r.get("rerank_score", rrf_scores.get(pid, 0.0))),
                "rrf_score": float(rrf_scores.get(pid, 0.0)),
                "rerank_score": r.get("rerank_score"),
                "dense_score": r.get("dense_score"),
                "text_snippet": _snippet(snippet_source),
                "image_url": f"/images/{r.get('file_hash')}/{r.get('page_number')}",
                "reduced_image_url": f"/images/{r.get('file_hash')}/{r.get('page_number')}/reduced",
                "categories": r.get("categories") or [],
                "tags": r.get("tags") or [],
                "has_chunks": chunk_id is not None,
            })
        return ForgeResult(success=True, data=hits)

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
