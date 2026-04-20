"""Docling-based structural chunker.

Replaces the whole-page text chunks the original pipeline produced. A
design handbook page often contains multiple distinct topics (a paragraph
on bearing selection, a table of shaft fits, a figure caption about seal
geometry). Whole-page embeddings average those together and bury fine
retrieval signal. Structural chunking emits one chunk per paragraph,
table, figure caption, or equation block — each independently embedded,
summarized, and searchable.

The chunker wraps Docling's HybridChunker: layout-aware parsing from the
Docling core (tables become coherent blocks, headings carry section
context) with semantic grouping on top. Output is a flat list of chunks
with traceable (document, page, bbox) references.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass
class StructuralChunk:
    """One structural chunk from a PDF.

    - chunk_id is deterministic from (doc_hash, page, index, content hash),
      so re-running the chunker on the same PDF reproduces the same ids.
    - chunk_type: "text", "table", "figure", "equation", "list", "code",
      "heading", "caption" — matches Docling's item labels, collapsed a bit.
    - section_path: the heading hierarchy leading to this chunk, e.g.
      ["12. Fasteners", "12.3 Threaded Fasteners"].
    - bbox is optional; tables and figures always have one, text chunks
      sometimes don't when they span multiple blocks.
    """

    chunk_id: str
    page_number: int
    chunk_index: int
    chunk_type: str
    text: str
    section_path: list[str] = field(default_factory=list)
    bbox: tuple[float, float, float, float] | None = None
    # Reserved for Phase 3: topic tags, formula references, table captions.
    meta: dict = field(default_factory=dict)


# Docling item labels we care about and how we map them onto chunk_type.
# Everything else gets dropped.
_LABEL_TO_TYPE = {
    "text": "text",
    "paragraph": "text",
    "section_header": "heading",
    "title": "heading",
    "list_item": "list",
    "caption": "caption",
    "table": "table",
    "picture": "figure",
    "figure": "figure",
    "formula": "equation",
    "code": "code",
}


class StructuralChunker:
    """Layout-aware chunker built on Docling.

    Initialized once per process. Thread-safe for concurrent PDFs because
    Docling's DocumentConverter is stateless per convert() call.
    """

    def __init__(
        self,
        *,
        max_tokens: int = 512,
        merge_small_text: bool = True,
        min_text_chars: int = 40,
    ):
        self.max_tokens = max_tokens
        self.merge_small_text = merge_small_text
        self.min_text_chars = min_text_chars
        self._converter = None
        self._chunker = None

    def _ensure_loaded(self) -> None:
        if self._converter is not None:
            return
        # Imported here so the module imports cleanly even when Docling
        # isn't installed — unit tests that only exercise StructuralChunk
        # shouldn't require the full dependency chain.
        from docling.document_converter import DocumentConverter
        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

        logger.info("Initializing Docling converter + HybridChunker")
        self._converter = DocumentConverter()
        self._chunker = HybridChunker(max_tokens=self.max_tokens)

    def chunk_pdf(self, pdf_path: str | Path, doc_hash: str) -> list[StructuralChunk]:
        """Parse a PDF into structural chunks. doc_hash is used in the
        deterministic chunk_id so the same PDF always yields the same ids."""
        self._ensure_loaded()
        pdf_path = Path(pdf_path)
        logger.info("Chunking PDF %s", pdf_path.name)

        result = self._converter.convert(str(pdf_path))  # type: ignore[union-attr]
        doc = result.document

        chunks: list[StructuralChunk] = []
        per_page_counter: dict[int, int] = {}

        for raw in self._chunker.chunk(doc):  # type: ignore[union-attr]
            text = (raw.text or "").strip()
            if not text:
                continue

            # Resolve page number + type + section path from the chunk's
            # source items in the original document. HybridChunker merges
            # items, so one chunk can span multiple items — we report the
            # first item's page and type.
            doc_items = getattr(raw.meta, "doc_items", None) or []
            if not doc_items:
                continue
            first = doc_items[0]
            page_number = _extract_page_number(first)
            if page_number is None:
                continue

            label = _extract_label(first)
            chunk_type = _LABEL_TO_TYPE.get(label, "text")
            bbox = _extract_bbox(first)
            section_path = _extract_section_path(raw.meta)

            # Skip tiny text chunks that are almost certainly header/footer
            # fragments. Tables and figures can be short legitimately.
            if (
                self.merge_small_text
                and chunk_type == "text"
                and len(text) < self.min_text_chars
            ):
                continue

            idx = per_page_counter.get(page_number, 0)
            per_page_counter[page_number] = idx + 1

            chunk_id = _chunk_id(doc_hash, page_number, idx, text)
            chunks.append(
                StructuralChunk(
                    chunk_id=chunk_id,
                    page_number=page_number,
                    chunk_index=idx,
                    chunk_type=chunk_type,
                    text=text,
                    section_path=section_path,
                    bbox=bbox,
                )
            )

        logger.info(
            "Chunker produced %d chunks across %d pages for %s",
            len(chunks), len(per_page_counter), pdf_path.name,
        )
        return chunks


# --- Docling helpers -------------------------------------------------------
# Docling's object model changes between minor versions; these helpers
# extract what we need defensively so a small upstream schema tweak doesn't
# break ingestion.

def _extract_page_number(item) -> int | None:
    # Path 1: item.prov[0].page_no (docling_core newer API)
    prov = getattr(item, "prov", None)
    if prov:
        first_prov = prov[0] if isinstance(prov, list) else prov
        page_no = getattr(first_prov, "page_no", None)
        if page_no is not None:
            return int(page_no)
    # Path 2: item.page (older API)
    page = getattr(item, "page", None)
    if isinstance(page, int):
        return page
    return None


def _extract_label(item) -> str:
    label = getattr(item, "label", None)
    if label is None:
        return "text"
    # Label may be a string or an enum with .value
    if hasattr(label, "value"):
        return str(label.value).lower()
    return str(label).lower()


def _extract_bbox(item) -> tuple[float, float, float, float] | None:
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    first_prov = prov[0] if isinstance(prov, list) else prov
    bbox = getattr(first_prov, "bbox", None)
    if bbox is None:
        return None
    # docling BoundingBox has l, t, r, b (or similar). Handle both patterns.
    try:
        l = getattr(bbox, "l", None) or bbox[0]
        t = getattr(bbox, "t", None) or bbox[1]
        r = getattr(bbox, "r", None) or bbox[2]
        b = getattr(bbox, "b", None) or bbox[3]
        return (float(l), float(t), float(r), float(b))
    except Exception:
        return None


def _extract_section_path(meta) -> list[str]:
    headings = getattr(meta, "headings", None)
    if not headings:
        return []
    out = []
    for h in headings:
        text = getattr(h, "text", None) or (h if isinstance(h, str) else None)
        if text:
            out.append(str(text).strip())
    return out


def _chunk_id(doc_hash: str, page: int, idx: int, text: str) -> str:
    """Stable chunk id — re-running the chunker on the same PDF produces
    the same ids. Based on (doc_hash, page, index, content prefix hash).
    Using a content hash means two paragraphs with identical text on the
    same page still get distinct ids via their distinct `idx`."""
    h = hashlib.sha1()
    h.update(doc_hash.encode("utf-8"))
    h.update(f"|{page}|{idx}|".encode("utf-8"))
    h.update(text[:200].encode("utf-8", errors="ignore"))
    return f"ch_{h.hexdigest()[:20]}"


def chunks_by_page(chunks: list[StructuralChunk]) -> Iterator[tuple[int, list[StructuralChunk]]]:
    """Group chunks by page_number. Yields (page_number, [chunk, ...])."""
    by_page: dict[int, list[StructuralChunk]] = {}
    for c in chunks:
        by_page.setdefault(c.page_number, []).append(c)
    for page_no in sorted(by_page):
        yield page_no, by_page[page_no]
