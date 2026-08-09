"""RAPTOR-by-TOC: hierarchical document summaries guided by book structure.

RAPTOR (Sarthi et al., 2024) builds a summary tree by clustering chunks in
embedding space and summarizing each cluster recursively, so retrieval can
happen at any abstraction level. Clustering exists because most corpora
have no structure. Engineering books DO: Docling gives every chunk a
section_path (the heading hierarchy). This module builds the summary tree
from that structure instead — the table of contents is a better tree than
k-means will ever find for reference works — which removes the clustering
pass entirely and keeps every summary aligned with a section a human could
open the book to.

Tree shape: leaf nodes are the deepest section_path groups; parents are
path prefixes; the root (empty path) is the whole document. Summaries are
written bottom-up by the local LLM: leaves from their chunks' existing
summaries (figure/table chunk summaries included, so graphical content
reaches the tree), parents from their children's summaries. Documents
without heading structure (some scanned books) fall back to fixed
page-window sections so every document gets a tree.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Chunks whose section_path is empty get grouped into windows of this many
# pages so structureless (scanned) books still produce a usable tree.
FALLBACK_WINDOW_PAGES = 25

# Max child summaries fed into one LLM call; larger sections summarize in
# runs of this size and then merge the run-summaries.
MAX_ITEMS_PER_CALL = 35

SUMMARY_MAX_TOKENS = 450
SUMMARY_RETRY_MAX_TOKENS = 900


@dataclass
class SectionNode:
    path: tuple[str, ...]
    title: str
    chunk_summaries: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    children: dict[str, "SectionNode"] = field(default_factory=dict)
    summary: str = ""

    @property
    def level(self) -> int:
        return len(self.path)

    def observe_page(self, page: int | None) -> None:
        if page is None:
            return
        self.page_start = page if self.page_start is None else min(self.page_start, page)
        self.page_end = page if self.page_end is None else max(self.page_end, page)


def summary_id(doc_id: str, path: tuple[str, ...]) -> str:
    """Stable id per (doc, section path)."""
    key = doc_id + "\x00" + "\x1f".join(path)
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def build_section_tree(
    doc_title: str, chunks: list[dict]
) -> SectionNode:
    """Group chunks into a section tree by their Docling section_path.

    chunks: dicts with section_path (list[str] | None), page_number,
    summary (str | None), text (str | None), chunk_type.
    """
    root = SectionNode(path=(), title=doc_title or "(document)")

    structured = sum(1 for c in chunks if c.get("section_path"))
    use_fallback = structured < max(1, len(chunks)) * 0.4

    for c in chunks:
        raw_path = c.get("section_path") or []
        page = c.get("page_number")
        if use_fallback or not raw_path:
            if page is None:
                path: tuple[str, ...] = ("Front matter",)
            else:
                lo = ((int(page) - 1) // FALLBACK_WINDOW_PAGES) * FALLBACK_WINDOW_PAGES + 1
                path = (f"Pages {lo}–{lo + FALLBACK_WINDOW_PAGES - 1}",)
        else:
            # Cap depth: very deep heading stacks add tree levels without
            # adding retrieval value.
            path = tuple(str(p).strip() for p in raw_path if str(p).strip())[:3]
            if not path:
                path = ("Front matter",)

        node = root
        node.observe_page(page)
        for part in path:
            node = node.children.setdefault(
                part, SectionNode(path=node.path + (part,), title=part)
            )
            node.observe_page(page)

        label = ""
        ctype = c.get("chunk_type")
        if ctype and ctype not in ("text", "paragraph"):
            label = f"[{ctype}] "
        s = (c.get("summary") or c.get("text") or "").strip()
        if s:
            node.chunk_summaries.append(label + s[:600])

    return root


def iter_nodes_bottom_up(root: SectionNode):
    """Yield every node, children before parents (root last)."""
    for child in root.children.values():
        yield from iter_nodes_bottom_up(child)
    yield root


class TocSummarizer:
    """LLM summarization over a section tree, bottom-up."""

    def __init__(self, llm):
        self.llm = llm

    async def _call(self, prompt: str, max_tokens: int) -> tuple[str, str | None]:
        return await self.llm.chat_with_finish_reason(
            [
                {
                    "role": "system",
                    "content": (
                        "You summarize sections of engineering reference "
                        "books. Write a dense factual summary (120-200 "
                        "words) of what this section covers: topics, "
                        "materials, methods, standards, notable tables or "
                        "figures. No preamble, no meta-commentary — just "
                        "the summary text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )

    async def summarize_items(
        self, doc_title: str, section_title: str, items: list[str]
    ) -> str:
        """Summarize a list of child summaries into one section summary."""
        if not items:
            return ""
        if len(items) == 1 and len(items[0]) < 700:
            # A single short child IS the summary — no LLM needed.
            return items[0]

        async def one(batch: list[str]) -> str:
            body = "\n".join(f"- {s}" for s in batch)
            prompt = (
                f"Book: {doc_title}\nSection: {section_title or '(whole document)'}\n"
                f"Content summaries from this section:\n{body}"
            )
            text, finish = await self._call(prompt, SUMMARY_MAX_TOKENS)
            if finish == "length":
                text, _ = await self._call(prompt, SUMMARY_RETRY_MAX_TOKENS)
            return (text or "").strip()

        if len(items) <= MAX_ITEMS_PER_CALL:
            return await one(items)
        # Large section: summarize runs, then merge the run-summaries.
        runs = [
            items[i : i + MAX_ITEMS_PER_CALL]
            for i in range(0, len(items), MAX_ITEMS_PER_CALL)
        ]
        run_summaries = [await one(r) for r in runs]
        return await one(run_summaries)

    async def summarize_tree(
        self, doc_title: str, root: SectionNode, checkpoint=None
    ) -> int:
        """Fill node.summary for every node, bottom-up. Returns count."""
        n = 0
        for node in iter_nodes_bottom_up(root):
            if checkpoint is not None:
                await checkpoint()
            items = list(node.chunk_summaries)
            items.extend(
                f"({child.title}) {child.summary}"
                for child in node.children.values()
                if child.summary
            )
            node.summary = await self.summarize_items(
                doc_title, " / ".join(node.path), items
            )
            if node.summary:
                n += 1
        return n
