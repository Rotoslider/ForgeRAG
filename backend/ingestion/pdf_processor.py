"""PDF to page image conversion.

Adapted from /home/nuc1/projects/2025-03-20 wag RAG_Vision/v0.15.6_colpali/pdf_manager.py
but reorganized for ForgeRAG's doc-hash-based storage layout:

    data/
        page_images/{doc_hash}/page_0001.png     <- full resolution (for ColPali)
        reduced_images/{doc_hash}/page_0001.jpg  <- reduced (for thumbnails / VLM OCR)

Uses pdf2image (poppler backend) for rasterization. Runs synchronously inside
a thread to avoid blocking the event loop — callers should wrap in asyncio.to_thread.
"""

from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path
from typing import Iterable

from pdf2image import convert_from_path
from PIL import Image

logger = logging.getLogger(__name__)


def _pad_page_num(n: int, width: int = 4) -> str:
    """Zero-pad so file listings sort naturally (page_0001, page_0012, page_0123)."""
    return str(n).zfill(width)


def _resize_for_reduced(image: Image.Image, reduction_pct: int, min_dimension: int) -> Image.Image:
    """Resize to (reduction_pct / 100) of original, but not below min_dimension on short side."""
    width, height = image.size
    if width < min_dimension or height < min_dimension:
        return image

    scale = reduction_pct / 100.0
    new_w = int(width * scale)
    new_h = int(height * scale)

    # Ensure shorter side doesn't fall below min_dimension
    if min(new_w, new_h) < min_dimension:
        adjust = min_dimension / min(new_w, new_h)
        new_w = math.ceil(new_w * adjust)
        new_h = math.ceil(new_h * adjust)

    if (new_w, new_h) == (width, height):
        return image
    return image.resize((new_w, new_h), Image.LANCZOS)


class PDFProcessor:
    """Converts PDFs to page images on disk.

    Config values come from IngestionSettings (pdf_dpi, reduction_percentage,
    reduction_min_dimension, max_concurrent_pdf_conversions).
    """

    def __init__(
        self,
        *,
        data_dir: Path,
        dpi: int = 300,
        reduction_pct: int = 50,
        reduction_min_dimension: int = 768,
    ):
        self.data_dir = Path(data_dir)
        self.page_images_dir = self.data_dir / "page_images"
        self.reduced_images_dir = self.data_dir / "reduced_images"
        self.dpi = dpi
        self.reduction_pct = reduction_pct
        self.reduction_min_dimension = reduction_min_dimension

        self.page_images_dir.mkdir(parents=True, exist_ok=True)
        self.reduced_images_dir.mkdir(parents=True, exist_ok=True)

    def doc_folder(self, doc_hash: str) -> Path:
        return self.page_images_dir / doc_hash

    def reduced_doc_folder(self, doc_hash: str) -> Path:
        return self.reduced_images_dir / doc_hash

    def page_image_path(self, doc_hash: str, page_number: int) -> Path:
        return self.doc_folder(doc_hash) / f"page_{_pad_page_num(page_number)}.png"

    def reduced_image_path(self, doc_hash: str, page_number: int) -> Path:
        return self.reduced_doc_folder(doc_hash) / f"page_{_pad_page_num(page_number)}.jpg"

    def clear_doc(self, doc_hash: str) -> None:
        """Delete all page images for a given doc_hash (idempotent)."""
        for folder in (self.doc_folder(doc_hash), self.reduced_doc_folder(doc_hash)):
            if folder.exists():
                shutil.rmtree(folder)

    def count_existing_pages(self, doc_hash: str) -> int:
        """Return the number of full-resolution PNGs already on disk for this doc."""
        folder = self.doc_folder(doc_hash)
        if not folder.exists():
            return 0
        return len(list(folder.glob("page_*.png")))

    def convert_pdf_sync(
        self,
        pdf_path: Path,
        doc_hash: str,
        *,
        progress_cb=None,
        resume: bool = True,
    ) -> list[Path]:
        """Synchronous PDF-to-images conversion. Returns list of full-resolution PNG paths.

        Call from within asyncio.to_thread to avoid blocking.

        progress_cb: optional callable(pages_done: int, pages_total: int) for progress updates.
        resume: if True (default), skips pages that already exist on disk (both PNG
            and reduced JPG must be present). Set False to force full re-render.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if not resume:
            self.clear_doc(doc_hash)
        self.doc_folder(doc_hash).mkdir(parents=True, exist_ok=True)
        self.reduced_doc_folder(doc_hash).mkdir(parents=True, exist_ok=True)

        # Determine total page count cheaply without rasterizing all pages
        import fitz

        with fitz.open(str(pdf_path)) as doc:
            total = doc.page_count

        # Collect the set of page numbers still to render
        to_render: list[int] = []
        saved_paths: list[Path] = []
        for idx in range(1, total + 1):
            full_path = self.page_image_path(doc_hash, idx)
            reduced_path = self.reduced_image_path(doc_hash, idx)
            if resume and full_path.exists() and reduced_path.exists():
                saved_paths.append(full_path)
                if progress_cb is not None:
                    try:
                        progress_cb(idx, total)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("progress_cb raised: %s", exc)
                continue
            to_render.append(idx)

        if not to_render:
            logger.info("All %d pages of %s already rendered — skipping rasterization", total, pdf_path.name)
            return saved_paths

        logger.info(
            "Rasterizing %s at %d DPI (%d of %d pages needed, %d already on disk)",
            pdf_path.name, self.dpi, len(to_render), total, total - len(to_render),
        )

        # Render in chunks so we don't load the entire PDF into memory at once.
        # pdf2image uses first_page/last_page (1-indexed).
        chunk_size = 50
        i = 0
        while i < len(to_render):
            chunk = to_render[i:i + chunk_size]
            first = chunk[0]
            last = chunk[-1]
            # Render a contiguous range, then pick out pages we actually need
            # (in case the chunk spans a discontinuous gap).
            images: list[Image.Image] = convert_from_path(
                str(pdf_path),
                dpi=self.dpi,
                fmt="png",
                first_page=first,
                last_page=last,
            )
            for offset, img in enumerate(images):
                page_num = first + offset
                if page_num not in chunk:
                    continue
                full_path = self.page_image_path(doc_hash, page_num)
                reduced_path = self.reduced_image_path(doc_hash, page_num)

                img.save(full_path, "PNG", optimize=True)
                reduced_img = _resize_for_reduced(
                    img, self.reduction_pct, self.reduction_min_dimension
                )
                if reduced_img.mode == "RGBA":
                    reduced_img = reduced_img.convert("RGB")
                reduced_img.save(reduced_path, "JPEG", quality=85, optimize=True)
                saved_paths.append(full_path)

                if progress_cb is not None:
                    try:
                        progress_cb(page_num, total)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("progress_cb raised: %s", exc)

                if page_num % 25 == 0 or page_num == total:
                    logger.info("Rasterized %d/%d pages of %s", page_num, total, pdf_path.name)
            i += chunk_size

        return saved_paths

    def iter_existing_pages(self, doc_hash: str) -> Iterable[tuple[int, Path]]:
        """Yield (page_number, full_image_path) for existing images under doc_hash,
        sorted by page number."""
        folder = self.doc_folder(doc_hash)
        if not folder.exists():
            return
        for p in sorted(folder.glob("page_*.png")):
            try:
                n = int(p.stem.split("_", 1)[1])
                yield n, p
            except (IndexError, ValueError):
                continue
