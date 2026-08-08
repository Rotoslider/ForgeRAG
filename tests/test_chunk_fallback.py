"""Chunking fallback for text-less PDFs (the Camplux class).

Vector-outline PDFs (fonts converted to curves) render perfectly but hold
zero extractable text, so Docling trusts the empty text layer and produces
no chunks. The pipeline must rebuild a rasterized PDF from the page images
it already rendered and retry OCR on that — automatically, not via a
manual rasterize-and-reingest.
"""

import asyncio
from pathlib import Path

import pytest
from PIL import Image

from backend.ingestion.pipeline import IngestionPipeline

pytestmark = pytest.mark.asyncio


class _ProcessorStub:
    def __init__(self, img_dir: Path):
        self._dir = img_dir

    def doc_folder(self, doc_hash: str) -> Path:
        return self._dir


class _JobsStub:
    async def checkpoint(self, job_id):
        pass

    async def update(self, job_id, **kwargs):
        pass

    async def update_step(self, job_id, step, status, detail=None):
        pass


class _ChunkerStub:
    """Returns no chunks for the original path; records every call."""

    def __init__(self, rebuilt_result=None):
        self.calls: list[str] = []
        self.rebuilt_result = rebuilt_result or []

    def chunk_pdf(self, pdf_path, doc_hash):
        self.calls.append(str(pdf_path))
        if len(self.calls) == 1:
            return []
        return self.rebuilt_result


def _pipeline_with(img_dir: Path, chunker: _ChunkerStub) -> IngestionPipeline:
    p = object.__new__(IngestionPipeline)
    p.pdf_processor = _ProcessorStub(img_dir)
    p.structural_chunker = chunker
    p.jobs = _JobsStub()
    return p


def _write_pages(d: Path, n: int) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        Image.new("RGB", (200, 300), "white").save(d / f"page_{i:04d}.png")


async def test_no_chunks_triggers_rasterized_retry(tmp_path):
    _write_pages(tmp_path / "imgs", 3)
    chunker = _ChunkerStub()
    p = _pipeline_with(tmp_path / "imgs", chunker)

    result = await p._build_chunks("job1", "doc1", "hash1", "/orig/doc.pdf")

    # Both empty -> clean no-chunks result, but the retry MUST have run
    # against a temp rasterized rebuild, which is cleaned up afterwards.
    assert result == {"chunks": 0}
    assert len(chunker.calls) == 2
    assert chunker.calls[0] == "/orig/doc.pdf"
    assert chunker.calls[1].endswith(".pdf") and "rechunk_" in chunker.calls[1]
    assert not Path(chunker.calls[1]).exists()


async def test_no_page_images_skips_retry(tmp_path):
    chunker = _ChunkerStub()
    p = _pipeline_with(tmp_path / "missing", chunker)

    result = await p._build_chunks("job1", "doc1", "hash1", "/orig/doc.pdf")

    assert result == {"chunks": 0}
    assert len(chunker.calls) == 1  # nothing to rebuild from


async def test_rasterized_pdf_is_valid_and_streams(tmp_path):
    _write_pages(tmp_path / "imgs", 4)
    p = object.__new__(IngestionPipeline)
    p.pdf_processor = _ProcessorStub(tmp_path / "imgs")

    out = p._rasterized_pdf_from_page_images("hash1")

    try:
        assert out is not None
        data = Path(out).read_bytes()
        assert data.startswith(b"%PDF")
        # 4 pages present (PIL writes one /Type /Page per appended image)
        assert data.count(b"/Type /Page") >= 4
    finally:
        if out:
            Path(out).unlink(missing_ok=True)


async def test_gapped_page_images_skip_rebuild(tmp_path):
    """A missing render means rebuilt-PDF positions would misnumber every
    later page — the fallback must refuse rather than misattribute OCR."""
    d = tmp_path / "imgs"
    _write_pages(d, 4)
    (d / "page_0003.png").unlink()  # gap
    chunker = _ChunkerStub()
    p = _pipeline_with(d, chunker)

    result = await p._build_chunks("job1", "doc1", "hash1", "/orig/doc.pdf")

    assert result == {"chunks": 0}
    assert len(chunker.calls) == 1  # no retry against a misnumbered rebuild
