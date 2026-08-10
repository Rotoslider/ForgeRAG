"""Keyword-search `prefer` hint (table/figure re-rank).

Born from the 2026-08-09 Genesis vision run: the best TEXT match for a
designation is often the prose about a table, not the table page itself,
so vision tasks opened pages whose pixels lacked the asked-for table.
`prefer` stable-partitions results toward pages structurally containing
a table/figure chunk while text relevance orders within each group.
"""

from pathlib import Path

from backend.routers.search import KeywordSearchRequest

SEARCH_SRC = (Path(__file__).resolve().parent.parent
              / "backend" / "routers" / "search.py").read_text()


def test_prefer_field_accepted_and_optional():
    req = KeywordSearchRequest(query="C26000")
    assert req.prefer is None
    req = KeywordSearchRequest(query="C26000", prefer="table")
    assert req.prefer == "table"


def test_both_query_paths_flag_structural_content():
    # Main fulltext path AND the CONTAINS fallback must both return the
    # has_table/has_figure flags the re-rank depends on.
    assert SEARCH_SRC.count(
        "EXISTS { (p)-[:HAS_CHUNK]->(:Chunk {chunk_type: 'table'}) }"
    ) >= 2
    assert SEARCH_SRC.count(
        "EXISTS { (p)-[:HAS_CHUNK]->(:Chunk {chunk_type: 'figure'}) }"
    ) >= 2


def test_prefer_overfetches_for_promotion_headroom():
    # A preferred re-rank over exactly `limit` rows can promote nothing —
    # the table page for a code is often ranked 20-30 by pure text score.
    assert "min(body.limit * 4, 80)" in SEARCH_SRC
