"""RAPTOR-by-TOC tree building and bottom-up summarization."""

import pytest

from backend.ingestion.toc_summarizer import (
    MAX_ITEMS_PER_CALL,
    TocSummarizer,
    build_section_tree,
    iter_nodes_bottom_up,
    summary_id,
)



def _chunk(path, page, summary="s", ctype="text"):
    return {"section_path": path, "page_number": page,
            "summary": summary, "chunk_type": ctype, "text": summary}


class _LLM:
    def __init__(self):
        self.calls: list[str] = []
        self.finish = "stop"

    async def chat_with_finish_reason(self, messages, *, max_tokens=None,
                                      temperature=None):
        prompt = messages[-1]["content"]
        self.calls.append(prompt)
        return f"SUMMARY({len(self.calls)})", self.finish


def test_tree_groups_by_section_path_with_page_ranges():
    chunks = [
        _chunk(["Ch 1", "1.1 Bolts"], 10),
        _chunk(["Ch 1", "1.1 Bolts"], 12),
        _chunk(["Ch 1", "1.2 Welds"], 20),
        _chunk(["Ch 2"], 40),
    ]
    root = build_section_tree("Book", chunks)
    ch1 = root.children["Ch 1"]
    assert set(ch1.children) == {"1.1 Bolts", "1.2 Welds"}
    assert ch1.children["1.1 Bolts"].page_start == 10
    assert ch1.children["1.1 Bolts"].page_end == 12
    assert root.page_start == 10 and root.page_end == 40
    assert root.children["Ch 2"].level == 1


def test_depth_capped_at_three():
    chunks = [_chunk(["A", "B", "C", "D", "E"], 5)]
    root = build_section_tree("Book", chunks)
    node = root.children["A"].children["B"].children["C"]
    assert node.children == {}
    assert node.chunk_summaries  # the chunk landed at the capped depth


def test_structureless_doc_falls_back_to_page_windows():
    chunks = [_chunk(None, p) for p in range(1, 60)]
    root = build_section_tree("Scan", chunks)
    names = sorted(root.children)
    assert names == ["Pages 1–25", "Pages 26–50", "Pages 51–75"]


def test_bottom_up_iteration_yields_children_first():
    chunks = [_chunk(["Ch 1", "1.1"], 1), _chunk(["Ch 1"], 2)]
    root = build_section_tree("Book", chunks)
    order = [n.path for n in iter_nodes_bottom_up(root)]
    assert order.index(("Ch 1", "1.1")) < order.index(("Ch 1",))
    assert order[-1] == ()


def test_summary_ids_stable_and_distinct():
    a = summary_id("doc1", ("Ch 1",))
    assert a == summary_id("doc1", ("Ch 1",))
    assert a != summary_id("doc1", ("Ch 2",))
    assert a != summary_id("doc2", ("Ch 1",))


@pytest.mark.asyncio
async def test_single_short_item_skips_llm():
    llm = _LLM()
    s = TocSummarizer(llm)
    out = await s.summarize_items("Book", "Ch 1", ["only child summary"])
    assert out == "only child summary"
    assert llm.calls == []


@pytest.mark.asyncio
async def test_large_section_batches_then_merges():
    llm = _LLM()
    s = TocSummarizer(llm)
    items = [f"item {i}" for i in range(MAX_ITEMS_PER_CALL * 2 + 5)]
    out = await s.summarize_items("Book", "Ch 1", items)
    # 3 run-calls + 1 merge call
    assert len(llm.calls) == 4
    assert out.startswith("SUMMARY(")


@pytest.mark.asyncio
async def test_truncated_summary_retries_bigger():
    llm = _LLM()
    llm.finish = "length"
    s = TocSummarizer(llm)
    await s.summarize_items("Book", "Ch 1", ["a" * 100, "b" * 100])
    assert len(llm.calls) == 2  # original + one bigger-budget retry


@pytest.mark.asyncio
async def test_summarize_tree_feeds_children_into_parents():
    llm = _LLM()
    s = TocSummarizer(llm)
    chunks = [
        _chunk(["Ch 1", "1.1"], 1, "bolt torque data"),
        _chunk(["Ch 1", "1.1"], 2, "thread engagement"),
        _chunk(["Ch 1", "1.2"], 3, "weld symbols"),
    ]
    root = build_section_tree("Book", chunks)
    n = await s.summarize_tree("Book", root)
    assert n >= 3
    assert root.summary  # root got a summary
    # The parent call's prompt contains its children's summaries.
    parent_prompt = llm.calls[-1]
    assert "SUMMARY(" in parent_prompt


def test_flat_numbered_headings_gain_synthesized_hierarchy():
    # The SLAM Handbook case: Docling emits every heading at depth 1, so
    # "9.2.3 X" sat beside 400+ siblings. Numbering must rebuild nesting.
    chunks = [
        _chunk(["9 Radar Sensing"], 260),
        _chunk(["9.2 Radar Odometry"], 272),
        _chunk(["9.2.1 Doppler Odometry"], 273),
        _chunk(["9.2.3 Feature-based Odometry"], 276),
        _chunk(["10 Conclusions"], 300),
        _chunk(["Preface"], 5),
    ]
    root = build_section_tree("Handbook", chunks)
    ch9 = root.children["9 Radar Sensing"]
    sec92 = ch9.children["9.2 Radar Odometry"]
    assert set(sec92.children) == {
        "9.2.1 Doppler Odometry", "9.2.3 Feature-based Odometry",
    }
    # Unnumbered front matter stays a top-level leaf.
    assert "Preface" in root.children
    assert root.children["10 Conclusions"].level == 1


def test_synthesized_ancestor_placeholder_when_parent_unnamed():
    # No "9" heading exists — ancestry synthesizes a §9 placeholder.
    chunks = [
        _chunk(["9.1 Alpha"], 1), _chunk(["9.2 Beta"], 2),
        _chunk(["9.3 Gamma"], 3), _chunk(["9.4 Delta"], 4),
    ]
    root = build_section_tree("Book", chunks)
    assert "§9" in root.children
    assert set(root.children["§9"].children) == {
        "9.1 Alpha", "9.2 Beta", "9.3 Gamma", "9.4 Delta",
    }


def test_docling_hierarchy_left_untouched_when_not_flat():
    chunks = [
        _chunk(["Ch 1", "1.1 Bolts"], 1),
        _chunk(["Ch 1", "1.2 Welds"], 2),
        _chunk(["Ch 2", "2.1 Gears"], 3),
    ]
    root = build_section_tree("Book", chunks)
    assert set(root.children) == {"Ch 1", "Ch 2"}
