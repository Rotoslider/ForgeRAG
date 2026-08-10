"""Intermediate-level synthesis for wide-flat summary trees.

Reference volumes with unnumbered prose headings built as one root over
1,000+ leaf sections (ASM handbooks: {0:1, 1:1724}). Future ingests get
an intermediate level at build time (regroup_wide_flat); the existing
library is retrofitted by run_build_intermediates, which reuses leaf
summaries and pays only ~n/30 cluster summaries.
"""

from backend.ingestion.toc_summarizer import (
    CLUSTER_TARGET,
    WIDE_FLAT_THRESHOLD,
    SectionNode,
    chunk_evenly,
    cluster_label,
    regroup_wide_flat,
)


def _flat_root(n: int) -> SectionNode:
    root = SectionNode(path=(), title="Big Handbook")
    for i in range(n):
        t = f"Section {i:04d}"
        root.children[t] = SectionNode(
            path=(t,), title=t, page_start=i * 2 + 1, page_end=i * 2 + 2,
        )
    return root


def test_chunk_evenly_no_runt_cluster():
    bounds = chunk_evenly(2933)
    sizes = [e - s for s, e in bounds]
    assert sum(sizes) == 2933
    assert max(sizes) - min(sizes) <= 1  # near-equal, no 3-item runt
    assert all(abs(s - CLUSTER_TARGET) <= CLUSTER_TARGET // 2 for s in sizes)
    # Contiguous, ordered
    assert bounds[0][0] == 0 and bounds[-1][1] == 2933


def test_regroup_inserts_level_and_repaths_children():
    root = _flat_root(300)
    made = regroup_wide_flat(root)
    assert made == len(root.children) > 1
    total_kids = 0
    for mid in root.children.values():
        assert mid.level == 1 and mid.children
        assert mid.page_start is not None and mid.page_end is not None
        for kid in mid.children.values():
            # Child re-pathed under its cluster BEFORE summary ids are
            # derived — parent linkage in the write phase is path-based.
            assert kid.path == mid.path + (kid.title,)
            assert kid.level == 2
            total_kids += 1
    assert total_kids == 300


def test_regroup_leaves_small_and_deep_trees_alone():
    small = _flat_root(WIDE_FLAT_THRESHOLD)
    assert regroup_wide_flat(small) == 0

    deep = _flat_root(200)
    first = next(iter(deep.children.values()))
    first.children["sub"] = SectionNode(
        path=first.path + ("sub",), title="sub"
    )
    # Any existing depth means the book has real structure — never
    # reshuffle it.
    assert regroup_wide_flat(deep) == 0


def test_cluster_label_shape():
    lbl = cluster_label("Circuit Analysis", "Active Filters", 120, 184)
    assert lbl.startswith("pp 120–184: ")
    assert "Circuit Analysis" in lbl and "Active Filters" in lbl
    assert cluster_label("Only", "Only", 7, 7).startswith("p 7: ")
