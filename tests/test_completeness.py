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
        "pages_with_topic_tags": 80,
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
    d = _audit(_row(pages_with_entities=10))
    assert d["aspects"]["entities"]["status"] == "partial"


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
