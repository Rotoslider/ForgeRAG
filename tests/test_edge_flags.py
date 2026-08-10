"""N1 relations tier-1: suspect-edge flagging (flag, never delete).

An entity-entity edge asserted on exactly one page, between entities that
never co-occur on any other page, is statistically hallucination — but
some singletons are real rare facts, so they are FLAGGED (r.suspect),
excluded from reasoning chains, surfaced honestly in explicit graph
queries, and self-healed when independent re-assertion arrives. 31,914
flagged / 8,315 support-1 edges kept by the co-occurrence test
(2026-08-09, docs/noise-review-2026-08.md).
"""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GUARD = "coalesce(r{n}.suspect, false) = false"


def test_reasoning_chains_exclude_suspect_edges():
    src = (REPO / "backend" / "services" / "graph_reasoning.py").read_text()
    assert "coalesce(r1.suspect, false) = false" in src
    assert "coalesce(r2.suspect, false) = false" in src


def test_entity_rel_rewrite_self_heals():
    src = (REPO / "backend" / "ingestion" / "graph_builder.py").read_text()
    # ON MATCH means an independent page re-asserted the edge — that IS
    # the co-occurrence evidence the flag was waiting for.
    assert "r.suspect = null" in src


def test_explicit_graph_queries_surface_or_filter_the_flag():
    src = (REPO / "backend" / "routers" / "graph.py").read_text()
    # Per-row templates show the flag (honesty over hiding)...
    assert src.count("coalesce(r.suspect, false) AS suspect") == 3
    # ...bare name-list collects exclude it (no shape to carry a caveat).
    assert "coalesce(gr.suspect, false) = false" in src
