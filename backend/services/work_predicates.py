"""Single source of truth for "this page still needs work" predicates.

The completeness audit decides what counts as MISSING; the repair drains
decide what to SELECT for work. When those two definitions drift apart you
get the worst failure mode this project has had: a fix that runs, reports
success, and changes nothing the audit can see. Every repair query
references these constants, and deep verification's repair_coverage check
compares the audit's arithmetic against them — so a drift becomes a
failing check instead of a support mystery.

All predicates assume the page variable is bound as `p`.
"""

from __future__ import annotations

# The page->entity relationship types that mean "extraction ran and found
# something" (matches the audit and GraphBuilder output).
ENTITY_PAGE_RELS = (
    "MENTIONS_MATERIAL|DESCRIBES_PROCESS|REFERENCES_STANDARD|MENTIONS_EQUIPMENT"
)

# Pages entity extraction should process: has text, never stamped, and no
# entity relationships from a pre-stamp era run.
ENTITY_NEEDS_EXTRACTION = (
    "p.text_char_count > 0 "
    "AND p.entities_extracted_at IS NULL "
    f"AND NOT EXISTS {{ (p)-[:{ENTITY_PAGE_RELS}]->() }}"
)

# Pages counted as extraction-done by the audit: the stamp OR relationships.
ENTITY_EXTRACTION_DONE = (
    "p.entities_extracted_at IS NOT NULL "
    f"OR EXISTS {{ (p)-[:{ENTITY_PAGE_RELS}]->() }}"
)

# A page with at least this much text and ZERO entity relationships is
# suspicious: dense engineering pages nearly always name some material,
# process, standard, or equipment. Before 2026-08-07 the extractor accepted
# the model's fast schema-valid empty bail on table-heavy pages, so such
# pages were stamped done with nothing extracted. Post-fix, an empty result
# on a dense page survives an anti-bail retry and is stamped
# entities_confirmed_empty — those are trusted and excluded here.
# Keep equal to entity_extractor.BAIL_RETRY_MIN_CHARS.
SUSPICIOUS_EMPTY_MIN_CHARS = 2000

# Stamped pages whose empty extraction predates the anti-bail retry.
ENTITY_SUSPICIOUS_EMPTY = (
    f"p.text_char_count >= {SUSPICIOUS_EMPTY_MIN_CHARS} "
    "AND p.entities_extracted_at IS NOT NULL "
    "AND p.entities_confirmed_empty IS NULL "
    f"AND NOT EXISTS {{ (p)-[:{ENTITY_PAGE_RELS}]->() }}"
)

# Documents the TOC-summary builder should process: has chunks (the tree is
# built from chunk summaries) and no summaries stamp. Binds `d`, not `p` —
# summary coverage is a document-level property.
SUMMARIES_MISSING = (
    "EXISTS { (d)-[:HAS_PAGE]->(:Page)-[:HAS_CHUNK]->(:Chunk) } "
    "AND d.summaries_built_at IS NULL"
)

# Pages text embedding should process: has text, embedding absent entirely.
# (Wrong-dimension embeddings are a separate, destructive re-embed path and
# a separate verification check.)
TEXT_EMBED_MISSING = "p.text_char_count > 0 AND p.text_embedding IS NULL"

# Pages visual embedding should process: no vectors, and not flagged blank.
VISUAL_EMBED_MISSING = (
    "(p.colpali_vector_count IS NULL OR p.colpali_vector_count = 0) "
    "AND (p.is_blank IS NULL OR p.is_blank = false)"
)
