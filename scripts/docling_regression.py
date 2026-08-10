"""N4 Docling version regression diff.

Runs the SAME conversion + chunking the backend does (DocumentConverter +
HybridChunker(max_tokens=512)) against golden PDFs and emits comparable
stats. Run once in the current venv and once in a candidate venv, then
diff the JSON — chunk counts, section-path population, and table/picture
extraction are the axes that decide whether a bump is safe to adopt
(ROADMAP N4). Deliberately imports only docling, never the backend, so a
throwaway candidate venv needs nothing else.

Usage:
    <venv>/bin/python scripts/docling_regression.py out.json golden1.pdf [...]
"""

from __future__ import annotations

import json
import sys
import time
from importlib.metadata import version
from pathlib import Path


def stats_for(pdf_path: Path) -> dict:
    from docling.document_converter import DocumentConverter
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

    t0 = time.time()
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    chunker = HybridChunker(max_tokens=512)
    n_chunks = 0
    text_chars = 0
    with_sections = 0
    section_paths: set[tuple] = set()
    for raw in chunker.chunk(doc):
        text = (raw.text or "").strip()
        if not text:
            continue
        n_chunks += 1
        text_chars += len(text)
        headings = tuple(getattr(raw.meta, "headings", None) or [])
        if headings:
            with_sections += 1
            section_paths.add(headings)

    return {
        "file": pdf_path.name,
        "pages": doc.num_pages() if callable(getattr(doc, "num_pages", None))
        else getattr(doc, "num_pages", None),
        "chunks": n_chunks,
        "text_chars": text_chars,
        "chunks_with_section_path": with_sections,
        "distinct_section_paths": len(section_paths),
        "tables": len(getattr(doc, "tables", []) or []),
        "pictures": len(getattr(doc, "pictures", []) or []),
        "convert_secs": round(time.time() - t0, 1),
    }


def main() -> None:
    out_path, pdfs = sys.argv[1], sys.argv[2:]
    report = {"docling_version": version("docling"), "books": []}
    for p in pdfs:
        print(f"[{report['docling_version']}] converting {Path(p).name}...",
              flush=True)
        report["books"].append(stats_for(Path(p)))
        Path(out_path).write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
