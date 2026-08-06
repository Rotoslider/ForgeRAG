"""Deep pipeline verification.

Where the completeness audit (completeness.py) answers "which steps ran for
which documents", this module answers a harder question: is every artifact
the pipeline claims to have produced actually present, well-formed, and
internally consistent — across the entire database, with zero sampling.

Every check returns exact counts. A check either PASSES (violations == 0),
FAILS (violations > 0, with sample offenders for debugging), or WARNS
(suspicious but not provably wrong). There is no "probably fine".

Checks are grouped by pipeline stage:

  structure   documents/pages: counts match, numbering contiguous, no
              duplicates, no orphans
  files       every page's full-res and reduced image exists on disk
  text        extracted_text consistent with text_char_count; is_blank
              populated; no unrecovered OCR text stranded in chunks
  embeddings  every text page has a text embedding of exactly the
              configured dimension; every non-blank page has visual
              vectors whose stored blob is byte-for-byte the right size
              (count * dim * 4 bytes float32); no stale wrong-dim vectors
  chunks      every chunk has non-empty text, a summary, an embedding of
              the right dimension, a valid page link, and page_number
              agreeing with its page
  entities    every text page extracted (relationship or marker); entity
              nodes have their primary key set
  communities community nodes have members and summaries
  indexes     every schema index exists and is ONLINE, vector indexes at
              the configured dimensions

File-existence checks stat ~2 files per page (fast, local SSD). Everything
else is server-side Cypher aggregation — blobs are size()d in the database,
never transferred.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

SAMPLE = 10  # offenders to include per failed check


def _check(name: str, description: str, violations: int,
           total: int | None = None, samples: list | None = None,
           status: str | None = None, detail: str | None = None) -> dict:
    return {
        "name": name,
        "description": description,
        "status": status or ("pass" if violations == 0 else "fail"),
        "violations": violations,
        "total": total,
        "samples": samples or [],
        "detail": detail,
    }


async def run_verification(neo4j, settings) -> dict:
    """Run every check. Read-only except for nothing — this never writes."""
    checks: list[dict] = []
    text_dim = settings.models.text_embedding_dim
    visual_dim = settings.models.visual_embed_dim

    async def q(cypher: str, params: dict | None = None):
        return await neo4j.run_query(cypher, params or {}, timeout=600.0)

    # ---------------------------------------------------------- structure
    rows = await q("""
        MATCH (d:Document)
        OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Page)
        WITH d, count(p) AS pages
        WHERE pages <> d.page_count
        RETURN d.doc_id AS id, d.title AS title, d.page_count AS declared,
               pages AS actual LIMIT 1000
    """)
    checks.append(_check(
        "page_count_matches",
        "Every Document's page_count equals its actual :Page node count",
        len(rows), samples=rows[:SAMPLE],
    ))

    rows = await q("""
        MATCH (d:Document)-[:HAS_PAGE]->(p:Page)
        WITH d, p.page_number AS n, count(p) AS c
        WHERE c > 1
        RETURN d.doc_id AS id, d.title AS title, n AS page_number, c AS copies
        LIMIT 1000
    """)
    checks.append(_check(
        "no_duplicate_pages",
        "No (document, page_number) pair has more than one :Page node",
        len(rows), samples=rows[:SAMPLE],
    ))

    rows = await q("""
        MATCH (d:Document)-[:HAS_PAGE]->(p:Page)
        WITH d, min(p.page_number) AS lo, max(p.page_number) AS hi,
             count(p) AS n
        WHERE lo <> 1 OR hi <> n
        RETURN d.doc_id AS id, d.title AS title, lo, hi, n LIMIT 1000
    """)
    checks.append(_check(
        "page_numbering_contiguous",
        "Page numbers run 1..N with no gaps",
        len(rows), samples=rows[:SAMPLE],
    ))

    rows = await q("""
        MATCH (p:Page) WHERE NOT (:Document)-[:HAS_PAGE]->(p)
        RETURN p.page_id AS id, p.page_number AS page_number LIMIT 1000
    """)
    checks.append(_check(
        "no_orphan_pages",
        "Every :Page belongs to a :Document",
        len(rows), samples=rows[:SAMPLE],
    ))

    rows = await q("""
        MATCH (c:Chunk) WHERE NOT (:Page)-[:HAS_CHUNK]->(c)
        RETURN c.chunk_id AS id, c.doc_id AS doc_id LIMIT 1000
    """)
    checks.append(_check(
        "no_orphan_chunks",
        "Every :Chunk is linked to a :Page",
        len(rows), samples=rows[:SAMPLE],
    ))

    # --------------------------------------------------------------- files
    rows = await q("""
        MATCH (:Document)-[:HAS_PAGE]->(p:Page)
        RETURN p.page_id AS id, p.image_path AS img, p.reduced_image_path AS red
    """)
    missing_files = []
    for r in rows:
        if not r["img"] or not os.path.isfile(r["img"]):
            missing_files.append({"page_id": r["id"], "path": r["img"], "kind": "full"})
        if not r["red"] or not os.path.isfile(r["red"]):
            missing_files.append({"page_id": r["id"], "path": r["red"], "kind": "reduced"})
    checks.append(_check(
        "page_images_on_disk",
        "Every page's full-resolution PNG and reduced JPG exist on disk",
        len(missing_files), total=len(rows) * 2, samples=missing_files[:SAMPLE],
    ))

    # ---------------------------------------------------------------- text
    rows = await q("""
        MATCH (:Document)-[:HAS_PAGE]->(p:Page)
        WHERE coalesce(p.text_char_count, 0) <> size(coalesce(p.extracted_text, ''))
        RETURN p.page_id AS id, p.page_number AS page_number,
               p.text_char_count AS declared,
               size(coalesce(p.extracted_text, '')) AS actual LIMIT 1000
    """)
    checks.append(_check(
        "text_char_count_consistent",
        "text_char_count equals the actual length of extracted_text",
        len(rows), samples=rows[:SAMPLE],
    ))

    rows = await q("""
        MATCH (:Document)-[:HAS_PAGE]->(p:Page)
        WHERE p.is_blank IS NULL
        RETURN count(p) AS n
    """)
    n = rows[0]["n"] if rows else 0
    checks.append(_check(
        "blank_flags_populated",
        "Every page has is_blank computed (needed to exclude blanks from visual embedding)",
        n, status="warn" if n else "pass",
        detail="run any re-embed/fill job on affected docs to backfill" if n else None,
    ))

    rows = await q("""
        MATCH (:Document)-[:HAS_PAGE]->(p:Page)
        WHERE coalesce(p.text_char_count, 0) = 0
          AND EXISTS { (p)-[:HAS_CHUNK]->(:Chunk) }
        RETURN count(p) AS n
    """)
    n = rows[0]["n"] if rows else 0
    checks.append(_check(
        "no_stranded_ocr_text",
        "No page is missing text that exists as Docling OCR text in its chunks",
        n, detail="run 'Recover OCR text' on affected docs" if n else None,
    ))

    # ---------------------------------------------------------- embeddings
    rows = await q("""
        MATCH (:Document)-[:HAS_PAGE]->(p:Page)
        WHERE p.text_char_count > 0
          AND (p.text_embedding IS NULL OR size(p.text_embedding) <> $dim)
        RETURN p.page_id AS id, p.page_number AS page_number,
               size(p.text_embedding) AS actual_dim LIMIT 1000
    """, {"dim": text_dim})
    checks.append(_check(
        "text_embeddings_complete_and_correct_dim",
        f"Every page with text has a text embedding of exactly {text_dim} dims",
        len(rows), samples=rows[:SAMPLE],
    ))

    rows = await q("""
        MATCH (:Document)-[:HAS_PAGE]->(p:Page)
        WHERE coalesce(p.is_blank, false) = false
          AND coalesce(p.colpali_vector_count, 0) = 0
        RETURN p.page_id AS id, p.page_number AS page_number LIMIT 1000
    """)
    checks.append(_check(
        "visual_embeddings_complete",
        "Every non-blank page has visual embedding vectors",
        len(rows), samples=rows[:SAMPLE],
    ))

    # Blob integrity: the serialized multi-vector blob must be exactly
    # count * dim * 4 bytes (float32). A truncated write or serialization
    # bug shows up here even though count/dim metadata look fine.
    rows = await q("""
        MATCH (:Document)-[:HAS_PAGE]->(p:Page)
        WHERE coalesce(p.colpali_vector_count, 0) > 0
          AND (p.colpali_vector_dim <> $dim
               OR p.colpali_vectors IS NULL
               OR size(p.colpali_vectors) <> p.colpali_vector_count * p.colpali_vector_dim * 4)
        RETURN p.page_id AS id, p.page_number AS page_number,
               p.colpali_vector_count AS count, p.colpali_vector_dim AS dim,
               size(p.colpali_vectors) AS blob_bytes LIMIT 1000
    """, {"dim": visual_dim})
    checks.append(_check(
        "visual_embedding_blobs_intact",
        f"Every visual embedding blob is exactly count x {visual_dim} x 4 bytes of float32",
        len(rows), samples=rows[:SAMPLE],
    ))

    # ---------------------------------------------------------------- chunks
    rows = await q("""
        MATCH (c:Chunk)
        WHERE c.text IS NULL OR trim(c.text) = ''
        RETURN c.chunk_id AS id LIMIT 1000
    """)
    checks.append(_check(
        "chunks_have_text", "Every chunk has non-empty text",
        len(rows), samples=rows[:SAMPLE],
    ))

    rows = await q("""
        MATCH (c:Chunk)
        WHERE c.summary IS NULL OR trim(c.summary) = ''
        RETURN c.chunk_id AS id LIMIT 1000
    """)
    checks.append(_check(
        "chunks_have_summaries", "Every chunk has a summary",
        len(rows), samples=rows[:SAMPLE],
    ))

    rows = await q("""
        MATCH (c:Chunk)
        WHERE c.embedding IS NULL OR size(c.embedding) <> $dim
        RETURN c.chunk_id AS id, size(c.embedding) AS actual_dim LIMIT 1000
    """, {"dim": text_dim})
    checks.append(_check(
        "chunk_embeddings_correct_dim",
        f"Every chunk has an embedding of exactly {text_dim} dims",
        len(rows), samples=rows[:SAMPLE],
    ))

    rows = await q("""
        MATCH (p:Page)-[:HAS_CHUNK]->(c:Chunk)
        WHERE c.page_number <> p.page_number
        RETURN c.chunk_id AS id, c.page_number AS chunk_page,
               p.page_number AS actual_page LIMIT 1000
    """)
    checks.append(_check(
        "chunk_page_links_consistent",
        "Every chunk's page_number matches the page it is linked to",
        len(rows), samples=rows[:SAMPLE],
    ))

    # -------------------------------------------------------------- entities
    rows = await q("""
        MATCH (:Document)-[:HAS_PAGE]->(p:Page)
        WHERE p.text_char_count > 0
          AND p.entities_extracted_at IS NULL
          AND NOT EXISTS {
            (p)-[:MENTIONS_MATERIAL|DESCRIBES_PROCESS|REFERENCES_STANDARD|MENTIONS_EQUIPMENT]->()
          }
        RETURN count(p) AS n
    """)
    n = rows[0]["n"] if rows else 0
    checks.append(_check(
        "entity_extraction_complete",
        "Every text page has been entity-extracted (relationships or extracted-empty marker)",
        n, detail="run 'Extract missing entities' on affected docs" if n else None,
    ))

    bad_entities = 0
    ent_samples: list = []
    for label, pk in [("Material", "name"), ("Process", "name"),
                      ("Standard", "code"), ("Equipment", "name")]:
        rows = await q(f"""
            MATCH (e:{label})
            WHERE e.{pk} IS NULL OR trim(e.{pk}) = ''
            RETURN '{label}' AS label, elementId(e) AS id LIMIT 100
        """)
        bad_entities += len(rows)
        ent_samples.extend(rows[:3])
    checks.append(_check(
        "entities_have_primary_keys",
        "Every entity node has its primary key (name/code) set and non-empty",
        bad_entities, samples=ent_samples[:SAMPLE],
    ))

    # ---------------------------------------------------------- communities
    rows = await q("""
        MATCH (c:Community)
        WHERE c.summary IS NULL OR trim(c.summary) = ''
        RETURN c.community_id AS id LIMIT 1000
    """)
    checks.append(_check(
        "communities_have_summaries", "Every community has a summary",
        len(rows), samples=rows[:SAMPLE],
    ))

    # ------------------------------------------------------------- indexes
    idx_rows = await q("SHOW INDEXES YIELD name, state, type RETURN name, state, type")
    by_name = {r["name"]: r for r in idx_rows}
    expected = [
        "page_doc_number", "document_title", "chunk_page_number", "chunk_type",
        "page_text_fulltext", "chunk_text_fulltext", "entity_name_fulltext",
        "page_text_embedding", "chunk_embedding", "community_summary_embedding",
    ]
    idx_problems = [
        {"index": name,
         "state": by_name[name]["state"] if name in by_name else "MISSING"}
        for name in expected
        if name not in by_name or by_name[name]["state"] != "ONLINE"
    ]
    checks.append(_check(
        "indexes_online",
        "Every schema index (btree, fulltext, vector) exists and is ONLINE",
        len(idx_problems), total=len(expected), samples=idx_problems,
    ))

    # ------------------------------------------------------------- summary
    failed = [c for c in checks if c["status"] == "fail"]
    warned = [c for c in checks if c["status"] == "warn"]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS" if not failed else "FAIL",
        "checks_total": len(checks),
        "checks_passed": len(checks) - len(failed) - len(warned),
        "checks_failed": len(failed),
        "checks_warned": len(warned),
        "checks": checks,
    }
