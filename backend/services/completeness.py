"""Document completeness audit.

Derives per-document pipeline-step status from the graph itself — every
step leaves a fingerprint on :Page / :Chunk nodes, so no re-ingestion is
needed to know what's missing:

  pages            :Page node count vs Document.page_count
  text_embedding   p.text_embedding present AND size() == configured dim
  visual_embedding p.colpali_vector_count > 0 AND colpali_vector_dim ==
                   configured dim (blank pages excluded — never embedded)
  chunks           (p)-[:HAS_CHUNK]->(:Chunk) coverage over text pages
  entities         (p)-[:MENTIONS_*|...]->() coverage over text pages

"Ran and found nothing" vs "never ran" is disambiguated by markers the
pipeline stamps after doing the work:

  p.entities_extracted_at   set on a page after extraction, even when the
                            LLM found zero entities on it
  d.chunks_built_at         set on a document after a chunk build wrote
                            chunks — some pages legitimately yield none

Pages/docs processed before these markers existed fall back to coverage
heuristics; one repair pass stamps them and the audit then converges.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# One pass over all pages, aggregated per document. Blob properties are
# never returned — text_embedding is only touched via IS NOT NULL / size(),
# and the visual embedding is judged by its small count/dim metadata.
AUDIT_QUERY = """
MATCH (d:Document)
OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Page)
WITH d, p,
     CASE WHEN p IS NULL THEN 0 ELSE 1 END AS is_page,
     CASE WHEN p.text_char_count > 0 THEN 1 ELSE 0 END AS has_text,
     CASE WHEN coalesce(p.is_blank, false) THEN 1 ELSE 0 END AS is_blank,
     CASE WHEN p.text_embedding IS NOT NULL THEN 1 ELSE 0 END AS has_temb,
     CASE WHEN p.text_embedding IS NOT NULL
               AND size(p.text_embedding) = $text_dim THEN 1 ELSE 0 END AS temb_ok,
     CASE WHEN coalesce(p.colpali_vector_count, 0) > 0 THEN 1 ELSE 0 END AS has_vemb,
     CASE WHEN coalesce(p.colpali_vector_count, 0) > 0
               AND coalesce(p.colpali_vector_dim, 0) = $visual_dim
          THEN 1 ELSE 0 END AS vemb_ok,
     CASE WHEN p IS NOT NULL AND EXISTS { (p)-[:HAS_CHUNK]->(:Chunk) }
          THEN 1 ELSE 0 END AS has_chunks,
     CASE WHEN p IS NOT NULL AND EXISTS {
              (p)-[:MENTIONS_MATERIAL|DESCRIBES_PROCESS|REFERENCES_STANDARD|MENTIONS_EQUIPMENT]->()
          } THEN 1 ELSE 0 END AS has_entities,
     CASE WHEN p IS NOT NULL AND (p.entities_extracted_at IS NOT NULL OR EXISTS {
              (p)-[:MENTIONS_MATERIAL|DESCRIBES_PROCESS|REFERENCES_STANDARD|MENTIONS_EQUIPMENT]->()
          }) THEN 1 ELSE 0 END AS extraction_done,
     CASE WHEN p.topic_tags IS NOT NULL AND size(p.topic_tags) > 0
          THEN 1 ELSE 0 END AS has_topic_tags
RETURN d.doc_id AS doc_id,
       (d.chunks_built_at IS NOT NULL) AS chunks_built,
       d.title AS title,
       d.filename AS filename,
       d.page_count AS declared_pages,
       coalesce(d.collection, 'default') AS collection,
       d.source_type AS source_type,
       sum(is_page) AS pages,
       sum(has_text) AS pages_with_text,
       sum(is_blank) AS blank_pages,
       sum(has_temb) AS text_embedded,
       sum(temb_ok) AS text_embedded_ok,
       sum(has_vemb) AS visual_embedded,
       sum(vemb_ok) AS visual_embedded_ok,
       sum(has_chunks) AS pages_with_chunks,
       sum(has_entities) AS pages_with_entities,
       sum(extraction_done) AS pages_extraction_done,
       sum(has_topic_tags) AS pages_with_topic_tags
ORDER BY d.title
"""

CHUNK_COUNT_QUERY = """
MATCH (d:Document)-[:HAS_PAGE]->(:Page)-[:HAS_CHUNK]->(c:Chunk)
RETURN d.doc_id AS doc_id, count(DISTINCT c) AS chunk_count
"""

AspectStatus = str  # "done" | "partial" | "missing" | "error" | "na"

# Below this fraction of text pages, chunk/entity coverage is reported as
# partial rather than done. Deliberately loose: not every text page yields
# a chunk or an entity, so 100% coverage is not the bar.
_COVERAGE_PARTIAL_THRESHOLD = 0.5


def _aspect(status: AspectStatus, done: int, needed: int, detail: str | None = None) -> dict:
    return {"status": status, "done": done, "needed": needed, "detail": detail}


def derive_doc_audit(
    row: dict[str, Any],
    chunk_count: int,
    *,
    text_dim: int,
    visual_dim: int,
) -> dict:
    """Turn one AUDIT_QUERY row into per-aspect statuses.

    Pure function — unit-testable without Neo4j.
    """
    pages = row["pages"] or 0
    declared = row["declared_pages"] or 0
    pages_with_text = row["pages_with_text"] or 0
    blank = row["blank_pages"] or 0
    aspects: dict[str, dict] = {}

    # --- pages (register + rasterize + extract_text all leave Page nodes)
    if pages == 0:
        aspects["pages"] = _aspect(
            "error", 0, declared,
            "no Page nodes — ingestion died early; delete and re-ingest this PDF",
        )
    elif declared and pages != declared:
        aspects["pages"] = _aspect(
            "partial", pages, declared,
            f"{pages} Page nodes but PDF has {declared} pages",
        )
    else:
        aspects["pages"] = _aspect("done", pages, declared or pages)

    # --- text extraction (informational: scanned docs legitimately have none)
    if pages > 0 and pages_with_text == 0 and row.get("source_type") == "digital_native":
        aspects["text"] = _aspect(
            "partial", 0, pages,
            "digital-native PDF but no page has extracted text",
        )
    else:
        aspects["text"] = _aspect(
            "done", pages_with_text, pages,
            None if pages_with_text else "no text pages (scanned document)",
        )

    # --- text embedding: every text page should carry a correct-dim vector
    t_needed = pages_with_text
    t_any = row["text_embedded"] or 0
    t_ok = row["text_embedded_ok"] or 0
    if t_any > t_ok:
        aspects["text_embedding"] = _aspect(
            "error", t_ok, t_needed,
            f"{t_any - t_ok} embeddings have wrong dimensions (expected {text_dim}) "
            "— needs re-embed, not fill",
        )
    elif t_needed == 0:
        aspects["text_embedding"] = _aspect("na", 0, 0, "no text pages")
    elif t_ok >= t_needed:
        aspects["text_embedding"] = _aspect("done", t_ok, t_needed)
    elif t_ok == 0:
        aspects["text_embedding"] = _aspect("missing", 0, t_needed)
    else:
        aspects["text_embedding"] = _aspect(
            "partial", t_ok, t_needed, f"{t_needed - t_ok} pages missing"
        )

    # --- visual embedding: every non-blank page should carry vectors
    v_needed = max(0, pages - blank)
    v_any = row["visual_embedded"] or 0
    v_ok = row["visual_embedded_ok"] or 0
    if v_any > v_ok:
        aspects["visual_embedding"] = _aspect(
            "error", v_ok, v_needed,
            f"{v_any - v_ok} embeddings have wrong dimensions (expected {visual_dim}) "
            "— needs re-embed, not fill",
        )
    elif v_needed == 0:
        aspects["visual_embedding"] = _aspect("na", 0, 0, "no non-blank pages")
    elif v_ok >= v_needed:
        aspects["visual_embedding"] = _aspect("done", v_ok, v_needed)
    elif v_ok == 0:
        aspects["visual_embedding"] = _aspect("missing", 0, v_needed)
    else:
        aspects["visual_embedding"] = _aspect(
            "partial", v_ok, v_needed, f"{v_needed - v_ok} pages missing"
        )

    # --- chunks (Phase 9): trust the chunks_built_at marker when present —
    # Docling assigns chunks to the pages that yield them, so whatever
    # coverage a completed build produced IS complete. Docs built before the
    # marker existed fall back to a coverage heuristic over text pages.
    c_pages = row["pages_with_chunks"] or 0
    chunks_built = bool(row.get("chunks_built"))
    # Chunks can legitimately exist on more pages than have extracted text
    # (Docling parses the PDF itself — e.g. scanned docs with a text layer
    # PyMuPDF missed), so the denominator is the larger of the two.
    c_needed = max(pages_with_text, c_pages)
    if pages_with_text == 0 and chunk_count == 0:
        aspects["chunks"] = _aspect("na", 0, 0, "no text pages")
    elif chunks_built and chunk_count > 0:
        aspects["chunks"] = _aspect(
            "done", c_pages, c_needed,
            f"{chunk_count} chunks (build completed"
            + (f"; {c_needed - c_pages} pages yielded none)" if c_needed > c_pages else ")"),
        )
    elif chunk_count == 0:
        aspects["chunks"] = _aspect(
            "missing", 0, c_needed,
            "no chunks — ingested before Phase 9 or chunking failed",
        )
    elif c_pages / c_needed < _COVERAGE_PARTIAL_THRESHOLD:
        aspects["chunks"] = _aspect(
            "partial", c_pages, c_needed,
            f"{chunk_count} chunks cover only {c_pages} of {c_needed} pages",
        )
    else:
        aspects["chunks"] = _aspect(
            "done", c_pages, c_needed, f"{chunk_count} chunks"
        )

    # --- entities: a page counts as done when it has entity relationships OR
    # the entities_extracted_at marker (extraction ran, found nothing).
    # Pages extracted before the marker existed and holding no entities look
    # like gaps — one repair pass stamps them and the audit converges.
    e_pages = row["pages_with_entities"] or 0
    done_pages = min(row.get("pages_extraction_done") or e_pages, pages_with_text)
    tt_pages = row["pages_with_topic_tags"] or 0
    empty_note = (
        f" ({done_pages - e_pages} extracted with no entities found)"
        if done_pages > e_pages else ""
    )
    if pages_with_text == 0:
        aspects["entities"] = _aspect("na", 0, 0, "no text pages")
    elif done_pages >= pages_with_text:
        aspects["entities"] = _aspect(
            "done", done_pages, pages_with_text,
            f"{e_pages} pages with entities{empty_note}",
        )
    elif done_pages == 0:
        aspects["entities"] = _aspect(
            "missing", 0, pages_with_text, "no page has been extracted"
        )
    elif done_pages / pages_with_text < _COVERAGE_PARTIAL_THRESHOLD:
        aspects["entities"] = _aspect(
            "partial", done_pages, pages_with_text,
            f"only {done_pages} of {pages_with_text} text pages extracted"
            f"{empty_note} ({tt_pages} have topic tags)",
        )
    else:
        aspects["entities"] = _aspect(
            "done", done_pages, pages_with_text,
            f"{e_pages} pages with entities{empty_note}; "
            f"{tt_pages} have topic tags",
        )

    statuses = {a["status"] for a in aspects.values()}
    if "error" in statuses or aspects["pages"]["status"] == "error":
        overall = "error"
    elif "missing" in statuses or "partial" in statuses:
        overall = "incomplete"
    else:
        overall = "complete"

    return {
        "doc_id": row["doc_id"],
        "title": row["title"] or row["filename"] or row["doc_id"],
        "collection": row["collection"],
        "source_type": row.get("source_type"),
        "pages": pages,
        "declared_pages": declared,
        "chunk_count": chunk_count,
        "overall": overall,
        "aspects": aspects,
    }


def summarize(docs: list[dict]) -> dict:
    """Roll the per-doc audits up into headline numbers."""
    summary: dict[str, Any] = {
        "documents": len(docs),
        "complete": sum(1 for d in docs if d["overall"] == "complete"),
        "incomplete": sum(1 for d in docs if d["overall"] == "incomplete"),
        "error": sum(1 for d in docs if d["overall"] == "error"),
        "total_pages": sum(d["pages"] for d in docs),
    }
    gaps: dict[str, dict[str, int]] = {}
    for aspect in ("pages", "text_embedding", "visual_embedding", "chunks", "entities"):
        bad = [d for d in docs if d["aspects"][aspect]["status"] in ("missing", "partial", "error")]
        gaps[aspect] = {
            "docs": len(bad),
            "pages_missing": sum(
                max(0, d["aspects"][aspect]["needed"] - d["aspects"][aspect]["done"])
                for d in bad
            ),
        }
    summary["gaps"] = gaps
    return summary


async def run_audit(neo4j, *, text_dim: int, visual_dim: int) -> dict:
    """Execute the audit against Neo4j and return the full report."""
    # The page-level scan touches every Page property record; on a ~100k-page
    # graph it can exceed the default 90s query timeout, so allow 10 minutes.
    rows = await neo4j.run_query(
        AUDIT_QUERY, {"text_dim": text_dim, "visual_dim": visual_dim},
        timeout=600.0,
    )
    chunk_rows = await neo4j.run_query(CHUNK_COUNT_QUERY, timeout=600.0)
    chunk_counts = {r["doc_id"]: r["chunk_count"] for r in chunk_rows}

    docs = [
        derive_doc_audit(
            r, chunk_counts.get(r["doc_id"], 0),
            text_dim=text_dim, visual_dim=visual_dim,
        )
        for r in rows
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "text_dim": text_dim,
        "visual_dim": visual_dim,
        "summary": summarize(docs),
        "documents": docs,
    }
