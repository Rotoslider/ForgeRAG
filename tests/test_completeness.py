"""Tests for the completeness-audit derivation logic (pure, no Neo4j)."""

from __future__ import annotations

from backend.services.completeness import derive_doc_audit, summarize


def _row(**overrides) -> dict:
    base = {
        "doc_id": "d1",
        "title": "Handbook",
        "filename": "handbook.pdf",
        "declared_pages": 100,
        "collection": "default",
        "source_type": "digital_native",
        "pages": 100,
        "pages_with_text": 95,
        "blank_pages": 5,
        "text_embedded": 95,
        "text_embedded_ok": 95,
        "visual_embedded": 95,
        "visual_embedded_ok": 95,
        "pages_with_chunks": 90,
        "pages_with_entities": 80,
        "pages_extraction_done": 80,
        "pages_text_recoverable": 0,
        "pages_with_topic_tags": 80,
        "chunks_built": False,
    }
    base.update(overrides)
    return base


def _audit(row, chunk_count=500):
    return derive_doc_audit(row, chunk_count, text_dim=1024, visual_dim=128)


def test_fully_complete_document():
    d = _audit(_row())
    assert d["overall"] == "complete"
    assert all(
        a["status"] in ("done", "na") for a in d["aspects"].values()
    ), d["aspects"]


def test_zero_pages_is_error():
    d = _audit(_row(pages=0, pages_with_text=0, blank_pages=0,
                    text_embedded=0, text_embedded_ok=0,
                    visual_embedded=0, visual_embedded_ok=0,
                    pages_with_chunks=0, pages_with_entities=0,
                    pages_with_topic_tags=0), chunk_count=0)
    assert d["overall"] == "error"
    assert d["aspects"]["pages"]["status"] == "error"
    assert "re-ingest" in d["aspects"]["pages"]["detail"]


def test_page_count_mismatch_is_partial():
    d = _audit(_row(pages=90))
    assert d["aspects"]["pages"]["status"] == "partial"
    assert d["overall"] == "incomplete"


def test_missing_text_embeddings():
    d = _audit(_row(text_embedded=0, text_embedded_ok=0))
    assert d["aspects"]["text_embedding"]["status"] == "missing"
    assert d["overall"] == "incomplete"


def test_partial_visual_embeddings_counts_missing():
    d = _audit(_row(visual_embedded=40, visual_embedded_ok=40))
    a = d["aspects"]["visual_embedding"]
    assert a["status"] == "partial"
    assert a["needed"] == 95  # 100 pages - 5 blank
    assert "55 pages missing" in a["detail"]


def test_wrong_dimension_embeddings_are_error_not_fillable():
    # 95 embedded but only 60 at the configured dim → the rest are stale
    # vectors from an old model. Filling won't fix them (they're not NULL).
    d = _audit(_row(text_embedded=95, text_embedded_ok=60))
    a = d["aspects"]["text_embedding"]
    assert a["status"] == "error"
    assert "wrong dimensions" in a["detail"]
    assert "re-embed" in a["detail"]
    assert d["overall"] == "error"


def test_scanned_doc_without_text_is_not_penalized():
    d = _audit(
        _row(source_type="scanned", pages_with_text=0,
             text_embedded=0, text_embedded_ok=0,
             pages_with_chunks=0, pages_with_entities=0,
             pages_with_topic_tags=0),
        chunk_count=0,
    )
    assert d["aspects"]["text_embedding"]["status"] == "na"
    assert d["aspects"]["chunks"]["status"] == "na"
    assert d["aspects"]["entities"]["status"] == "na"
    assert d["overall"] == "complete"


def test_no_chunks_is_missing():
    d = _audit(_row(pages_with_chunks=0), chunk_count=0)
    assert d["aspects"]["chunks"]["status"] == "missing"
    assert "Phase 9" in d["aspects"]["chunks"]["detail"]


def test_low_entity_coverage_is_partial():
    d = _audit(_row(pages_with_entities=10, pages_extraction_done=10))
    assert d["aspects"]["entities"]["status"] == "partial"


def test_extraction_marker_completes_entities_despite_empty_pages():
    """Pages where extraction ran but found nothing (stamped with
    entities_extracted_at) count as done — a scanned doc whose only text
    page has no entities must converge to green, not loop forever."""
    d = _audit(_row(pages_with_text=11, pages_with_entities=5,
                    pages_extraction_done=11))
    a = d["aspects"]["entities"]
    assert a["status"] == "done"
    assert a["done"] == 11
    assert "6 extracted with no entities found" in a["detail"]


def test_chunks_built_marker_completes_chunks():
    """A completed chunk build is final even if some pages yielded no
    chunks — Docling assigns chunks to the pages that have content."""
    d = _audit(
        _row(pages_with_text=7, pages_with_chunks=5, chunks_built=True),
        chunk_count=26,
    )
    a = d["aspects"]["chunks"]
    assert a["status"] == "done"
    assert "build completed" in a["detail"]
    assert "2 pages yielded none" in a["detail"]


def test_scanned_doc_with_docling_chunks_beyond_text_pages():
    """Scanned PDF: 1 text page per PyMuPDF but Docling chunked 391 pages.
    The chunk denominator must grow to the chunked pages (391/391 done,
    not the absurd 391/1), and one extraction-marked text page satisfies
    entities."""
    d = _audit(
        _row(pages=499, declared_pages=499, source_type="scanned",
             pages_with_text=1, blank_pages=0,
             text_embedded=1, text_embedded_ok=1,
             visual_embedded=499, visual_embedded_ok=499,
             pages_with_chunks=391, pages_with_entities=0,
             pages_extraction_done=1, pages_with_topic_tags=0),
        chunk_count=4922,
    )
    c = d["aspects"]["chunks"]
    assert c["status"] == "done"
    assert c["done"] == 391 and c["needed"] == 391
    e = d["aspects"]["entities"]
    assert e["status"] == "done"
    assert e["done"] == 1 and e["needed"] == 1
    assert d["overall"] == "complete"


def test_recoverable_ocr_text_flags_doc_incomplete():
    """A scanned doc whose chunks carry Docling OCR text that never made it
    onto the pages must be flagged — keyword search and entity extraction
    silently lose those pages until the text is recovered."""
    d = _audit(
        _row(source_type="scanned", pages=499, declared_pages=499,
             pages_with_text=1, blank_pages=0,
             text_embedded=1, text_embedded_ok=1,
             visual_embedded=499, visual_embedded_ok=499,
             pages_with_chunks=391, pages_with_entities=0,
             pages_extraction_done=1, pages_text_recoverable=390,
             pages_with_topic_tags=0),
        chunk_count=4922,
    )
    a = d["aspects"]["text"]
    assert a["status"] == "partial"
    assert a["needed"] == 391  # 1 text page + 390 recoverable
    assert "OCR text in chunks" in a["detail"]
    assert d["recoverable_text_pages"] == 390
    assert d["overall"] == "incomplete"


def test_scanned_doc_without_chunk_text_is_not_flagged():
    """A pure scan where Docling also got nothing has no recovery path —
    it must not be nagged as incomplete."""
    d = _audit(
        _row(source_type="scanned", pages_with_text=0, blank_pages=0,
             text_embedded=0, text_embedded_ok=0,
             visual_embedded=100, visual_embedded_ok=100,
             pages_with_chunks=0, pages_with_entities=0,
             pages_extraction_done=0, pages_text_recoverable=0,
             pages_with_topic_tags=0),
        chunk_count=0,
    )
    assert d["aspects"]["text"]["status"] == "done"
    assert d["overall"] == "complete"


def test_summarize_rolls_up_gaps():
    docs = [
        _audit(_row()),
        _audit(_row(doc_id="d2", text_embedded=0, text_embedded_ok=0)),
        _audit(_row(doc_id="d3", pages=0, pages_with_text=0, blank_pages=0,
                    text_embedded=0, text_embedded_ok=0,
                    visual_embedded=0, visual_embedded_ok=0,
                    pages_with_chunks=0, pages_with_entities=0,
                    pages_with_topic_tags=0), chunk_count=0),
    ]
    s = summarize(docs)
    assert s["documents"] == 3
    assert s["complete"] == 1
    assert s["incomplete"] == 1
    assert s["error"] == 1
    # d3 (zero pages) counts as a pages gap, not a text-embedding gap —
    # with no text pages it has nothing to embed.
    assert s["gaps"]["text_embedding"]["docs"] == 1
    assert s["gaps"]["text_embedding"]["pages_missing"] == 95
    assert s["gaps"]["pages"]["docs"] == 1
