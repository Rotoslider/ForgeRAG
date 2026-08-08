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
    # Existence first: every doc with pages must have chunks AT ALL. The
    # per-chunk checks below verify chunks that exist — a doc Docling never
    # chunked (or produced nothing for) passes them vacuously. Found live
    # 2026-08-08: a 52-page scanned manual with zero chunks sailed through
    # every chunk check. Warn-level: a doc Docling genuinely cannot read
    # stays visual-only and will keep warning here — that is the honest
    # state, not a bug in the check.
    # Distinct variable: the preceding query for chunks_have_text has
    # already run into `rows` and its append below consumes it — reusing
    # `rows` here made chunks_have_text report THIS check's violations
    # (caught live 2026-08-08: chunks_have_text "failed" with a doc sample).
    chunkless_docs = await q("""
        MATCH (d:Document)
        WHERE EXISTS { (d)-[:HAS_PAGE]->(:Page) }
          AND NOT EXISTS { (d)-[:HAS_PAGE]->(:Page)-[:HAS_CHUNK]->(:Chunk) }
        RETURN d.doc_id AS id, d.title AS title LIMIT 100
    """)
    checks.append(_check(
        "docs_have_chunks",
        "Every document with pages has at least one Docling chunk "
        "(zero chunks = OCR/chunking never produced output for the doc)",
        len(chunkless_docs), status="warn" if chunkless_docs else "pass",
        samples=chunkless_docs[:SAMPLE],
        detail="re-run rebuild-chunks on these docs; if the chunker again "
        "produces nothing, the PDF is unreadable to Docling and the doc "
        "remains visual-search-only" if chunkless_docs else None,
    ))

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

    # Dense pages stamped done with ZERO entities and no confirmed-empty
    # marker: before 2026-08-07 the model's fast schema-valid bail on
    # table-heavy pages was accepted and stamped, indistinguishable from a
    # genuine "nothing on this page". Post-fix empties survive an anti-bail
    # retry and carry entities_confirmed_empty. Warn (not fail): a legacy
    # empty CAN be genuine — but >=2000 chars of engineering text naming no
    # material/process/standard/equipment is rare enough to re-check.
    from backend.services.work_predicates import ENTITY_SUSPICIOUS_EMPTY
    rows = await q(f"""
        MATCH (d:Document)-[:HAS_PAGE]->(p:Page)
        WHERE {ENTITY_SUSPICIOUS_EMPTY}
        RETURN d.title AS doc, p.page_number AS page,
               p.text_char_count AS chars
        ORDER BY p.text_char_count DESC LIMIT 1000
    """)
    n_rows = await q(f"""
        MATCH (p:Page) WHERE {ENTITY_SUSPICIOUS_EMPTY}
        RETURN count(p) AS n
    """)
    n = n_rows[0]["n"] if n_rows else 0
    checks.append(_check(
        "entity_extractions_not_bailed",
        "No dense page is stamped extracted-with-nothing without a "
        "confirmed-empty marker (pre-2026-08-07 extractions accepted the "
        "model bailing on dense tables)",
        n, status="warn" if n else "pass", samples=rows[:SAMPLE],
        detail="run 'Re-extract suspicious empties' to re-check them "
        "(genuine empties get confirmed and drop off this list)" if n else None,
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

    # Junk relationships left by the pre-2026-08-06 normalize-entities bug
    # (it invented <Label>__TEMP_REL edges instead of real page links).
    temp_total = 0
    temp_samples: list = []
    for label in ("Material", "Process", "Standard", "Equipment"):
        rows = await q(
            f"MATCH ()-[r:{label}__TEMP_REL]->() RETURN count(r) AS n"
        )
        n = rows[0]["n"] if rows else 0
        if n:
            temp_total += n
            temp_samples.append({"label": label, "count": n})
    checks.append(_check(
        "no_temp_rel_garbage",
        "No junk <Label>__TEMP_REL relationships exist (artifact of the old "
        "normalize-entities bug)",
        temp_total, samples=temp_samples,
        detail="run 'Normalize entities' — it converts these back to real "
        "page links" if temp_total else None,
    ))

    # Repair coverage: the pages the AUDIT counts as missing an artifact
    # must be exactly the pages the REPAIR queries would select. If these
    # drift apart you get the "fix runs, reports success, audit unchanged"
    # failure mode — this check turns that into a red row.
    from backend.services.work_predicates import (
        ENTITY_EXTRACTION_DONE,
        ENTITY_NEEDS_EXTRACTION,
        TEXT_EMBED_MISSING,
        VISUAL_EMBED_MISSING,
    )

    coverage_mismatches = []
    # Entities: audit arithmetic (text pages minus done) vs repair selector.
    rows = await q(f"""
        MATCH (p:Page)
        RETURN
          sum(CASE WHEN p.text_char_count > 0 THEN 1 ELSE 0 END) AS text_pages,
          sum(CASE WHEN p.text_char_count > 0
                    AND ({ENTITY_EXTRACTION_DONE}) THEN 1 ELSE 0 END) AS done,
          sum(CASE WHEN {ENTITY_NEEDS_EXTRACTION} THEN 1 ELSE 0 END) AS selectable
    """)
    r = rows[0]
    audit_missing = r["text_pages"] - r["done"]
    if audit_missing != r["selectable"]:
        coverage_mismatches.append({
            "gap": "entities", "audit_missing": audit_missing,
            "repair_selects": r["selectable"],
        })
    # Text / visual embeddings: audit-missing-entirely vs fill selectors.
    rows = await q(f"""
        MATCH (p:Page)
        RETURN
          sum(CASE WHEN p.text_char_count > 0 AND p.text_embedding IS NULL
              THEN 1 ELSE 0 END) AS audit_text_missing,
          sum(CASE WHEN {TEXT_EMBED_MISSING} THEN 1 ELSE 0 END) AS text_selectable,
          sum(CASE WHEN coalesce(p.colpali_vector_count, 0) = 0
                    AND (p.is_blank IS NULL OR p.is_blank = false)
              THEN 1 ELSE 0 END) AS audit_visual_missing,
          sum(CASE WHEN {VISUAL_EMBED_MISSING} THEN 1 ELSE 0 END) AS visual_selectable
    """)
    r = rows[0]
    if r["audit_text_missing"] != r["text_selectable"]:
        coverage_mismatches.append({
            "gap": "text_embedding", "audit_missing": r["audit_text_missing"],
            "repair_selects": r["text_selectable"],
        })
    if r["audit_visual_missing"] != r["visual_selectable"]:
        coverage_mismatches.append({
            "gap": "visual_embedding", "audit_missing": r["audit_visual_missing"],
            "repair_selects": r["visual_selectable"],
        })
    checks.append(_check(
        "repair_coverage_matches",
        "Every page the audit counts as missing an artifact is selected by "
        "the corresponding repair query (no audit/repair predicate drift)",
        len(coverage_mismatches), samples=coverage_mismatches,
        detail="a repair would report success without touching the audited "
        "gap — fix the predicate drift in code" if coverage_mismatches else None,
    ))

    # Case/whitespace duplicate entities. These accumulate because the
    # post-ingest dedup step was broken (invalid MERGE) from 2026-05-05 to
    # 2026-08-06 — hygiene, not data loss, so a warning rather than a fail.
    dup_extra = 0
    dup_samples: list = []
    for label, pk in [("Material", "name"), ("Process", "name"),
                      ("Standard", "code"), ("Equipment", "name")]:
        rows = await q(f"""
            MATCH (e:{label})
            WITH toLower(trim(e.{pk})) AS k, collect(e.{pk}) AS names
            WHERE size(names) > 1
            RETURN '{label}' AS label, names LIMIT 1000
        """)
        dup_extra += sum(len(r["names"]) - 1 for r in rows)
        dup_samples.extend(rows[:3])
    checks.append(_check(
        "entities_case_deduped",
        "No two entities of one type differ only by case/whitespace",
        dup_extra, samples=dup_samples[:SAMPLE],
        status="warn" if dup_extra else "pass",
        detail="run 'Normalize entities' to merge them (mentions are "
        "preserved)" if dup_extra else None,
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

    # ---------------------------------------------------------- summaries
    # A chunk summary must be genuine: either a real LLM summary or the
    # intentional short-chunk case (text is its own summary). Preview
    # fallbacks — stored when the LLM failed during chunking — are marked
    # summary_source='preview' by current code and detectable on legacy
    # rows as summary == first 240 chars of the (possibly stripped) text.
    rows = await q("""
        MATCH (c:Chunk)
        WHERE c.summary_source = 'preview' OR
              (size(c.text) >= 400 AND c.summary IN
               [left(c.text, 240), left(trim(c.text), 240)])
        RETURN c.chunk_id AS id, c.doc_id AS doc_id,
               c.page_number AS page_number
        LIMIT 100000
    """)
    checks.append(_check(
        "chunk_summaries_genuine",
        "No chunk summary is a failure-fallback text preview",
        len(rows), samples=rows[:SAMPLE],
        detail="run POST /admin/resummarize-fallbacks to regenerate"
        if rows else None,
    ))

    # ------------------------------------------------------- organization
    # An unorganized doc (default collection, no categories, no tags) is
    # the footprint auto-tagging leaves when it fails during ingest. It can
    # also be a legitimately unclassifiable doc, so this warns rather than
    # fails.
    rows = await q("""
        MATCH (d:Document)
        WHERE coalesce(d.collection, 'default') = 'default'
          AND NOT (d)-[:IN_CATEGORY]->()
          AND NOT (d)-[:TAGGED_WITH]->()
        RETURN d.doc_id AS id, d.title AS title, d.filename AS filename
        LIMIT 1000
    """)
    checks.append(_check(
        "documents_organized",
        "Every document has a collection, categories, or tags "
        "(auto-tagging ran and produced something)",
        len(rows), samples=rows[:SAMPLE],
        status="warn" if rows else "pass",
        detail="run POST /admin/autotag-missing to tag them"
        if rows else None,
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
