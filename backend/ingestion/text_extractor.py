"""Text extraction from PDFs.

Strategy:
1. Use PyMuPDF (fitz) to extract text page-by-page from the source PDF.
   This gives high-quality text for digital-native PDFs at zero GPU cost.
2. For each page, classify as 'digital_native' if extracted text exceeds a
   char threshold, else 'scanned'. Scanned pages are left empty here —
   Phase 3 wires in the VLM (Qwen2.5-VL) to OCR those page images.
3. Return per-page text + source_type, and a document-level aggregate.

PyMuPDF's .get_text("text") preserves reading order reasonably well for
single-column documents and handles multi-column via layout analysis. For
heavily tabular content, future work can upgrade to "blocks" or "dict"
modes to preserve structure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from backend.models.documents import SourceType

logger = logging.getLogger(__name__)


@dataclass
class PageText:
    """Extracted text for a single page."""
    page_number: int          # 1-indexed
    text: str = ""
    char_count: int = 0
    source_type: SourceType = "unknown"
    needs_ocr: bool = False   # True if the page looked scanned


@dataclass
class DocumentTextExtraction:
    """Aggregate extraction result for a whole PDF."""
    page_count: int = 0
    pages: list[PageText] = field(default_factory=list)
    document_source_type: SourceType = "unknown"
    ocr_page_numbers: list[int] = field(default_factory=list)  # 1-indexed


class TextExtractor:
    """Extracts text from a PDF using PyMuPDF and classifies per-page source type."""

    def __init__(self, scanned_text_threshold_chars: int = 50):
        self.threshold = scanned_text_threshold_chars

    def extract_sync(self, pdf_path: Path) -> DocumentTextExtraction:
        """Synchronous extraction. Wrap in asyncio.to_thread from async callers."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        result = DocumentTextExtraction()
        with fitz.open(str(pdf_path)) as doc:
            result.page_count = doc.page_count
            digital_count = 0
            scanned_count = 0

            for i, page in enumerate(doc, start=1):
                # "text" mode preserves reading order for most layouts.
                raw = page.get_text("text").strip()
                char_count = len(raw)

                if char_count >= self.threshold:
                    source = "digital_native"
                    digital_count += 1
                    needs_ocr = False
                else:
                    source = "scanned"
                    scanned_count += 1
                    needs_ocr = True
                    result.ocr_page_numbers.append(i)

                result.pages.append(
                    PageText(
                        page_number=i,
                        text=raw,
                        char_count=char_count,
                        source_type=source,
                        needs_ocr=needs_ocr,
                    )
                )

            # Document-level classification:
            #   digital_native if >= 90% of pages have text
            #   scanned        if >= 90% of pages are empty
            #   hybrid         otherwise
            if result.page_count == 0:
                result.document_source_type = "unknown"
            elif digital_count / result.page_count >= 0.9:
                result.document_source_type = "digital_native"
            elif scanned_count / result.page_count >= 0.9:
                result.document_source_type = "scanned"
            else:
                result.document_source_type = "hybrid"

        logger.info(
            "Text extraction: %s — %d pages (%d digital, %d scanned) → %s",
            pdf_path.name,
            result.page_count,
            digital_count,
            scanned_count,
            result.document_source_type,
        )
        return result
