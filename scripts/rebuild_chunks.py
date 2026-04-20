#!/usr/bin/env python3
"""Re-process already-ingested PDFs through the Phase 5 pipeline.

For every Document in Neo4j, this script:
  1. Locates the original PDF in data/uploads/ by file_hash.
  2. Runs Docling's structural chunker.
  3. Generates per-chunk summaries via the LLM.
  4. Embeds (summary + text) pairs with BGE-M3.
  5. Writes Chunk nodes linked to the existing Page nodes via HAS_CHUNK.
  6. Re-runs entity extraction per page so the new Formula / Table /
     topic_tags fields get populated in the graph (Phase 3 content).
  7. Marks the document as rebuilt.

Design notes:
- IDEMPOTENT. Re-running the script on an already-rebuilt doc updates
  chunks in place (via MERGE on chunk_id) and overwrites embeddings.
  Entity extraction is skipped on pages that already have the new topic_tags
  field populated, so a resumed rebuild doesn't re-pay for Qwen calls.
- Resumable. --only-missing skips docs that already have chunks.
- Cost-aware. --skip-extract skips the per-page entity re-extraction when
  you only want the retrieval upgrade (chunks+summaries+embeddings).
- The Neo4j schema is auto-applied at the top so the new Chunk / Formula /
  RefTable / fulltext indexes are present before writes begin.

Usage:
    # Full rebuild — chunks AND re-extract entities with new Formula/Table/tag fields.
    # Hours of work; run overnight.
    NEO4J_PASSWORD=... python scripts/rebuild_chunks.py

    # Just chunks+summaries+embeddings (skips LLM re-extraction).
    NEO4J_PASSWORD=... python scripts/rebuild_chunks.py --skip-extract

    # Resume: only process docs that don't have chunks yet.
    NEO4J_PASSWORD=... python scripts/rebuild_chunks.py --only-missing

    # Limit to one doc for testing.
    NEO4J_PASSWORD=... python scripts/rebuild_chunks.py --doc-id DOC_XXX
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_settings  # noqa: E402
from backend.db.neo4j_schema import apply_schema  # noqa: E402
from backend.ingestion.chunk_summarizer import ChunkSummarizer  # noqa: E402
from backend.ingestion.chunker import StructuralChunker  # noqa: E402
from backend.ingestion.entity_extractor import EntityExtractor  # noqa: E402
from backend.ingestion.graph_builder import GraphBuilder  # noqa: E402
from backend.services.gpu_manager import GPUManager  # noqa: E402
from backend.services.llm_service import create_llm_service  # noqa: E402
from backend.services.neo4j_service import Neo4jService  # noqa: E402
from backend.services.text_embedding_service import (
    create_text_embedding_service,
)  # noqa: E402


log = logging.getLogger("rebuild_chunks")


async def find_pdf_path(svc: Neo4jService, doc_id: str) -> Path | None:
    """Locate the source PDF in data/uploads/ for a given Document node."""
    settings = get_settings()
    upload_dir = Path(settings.server.data_dir) / "uploads"
    rows = await svc.run_query(
        "MATCH (d:Document {doc_id: $d}) RETURN d.filename AS fn, d.file_hash AS h",
        {"d": doc_id},
    )
    if not rows:
        return None
    fn = rows[0]["fn"]
    h = rows[0]["h"]
    # Uploads are stored as "{hash}_{original_filename}" (first 32 chars of
    # the sha256, then an underscore, then the user's filename). Look for a
    # file that starts with the hash.
    candidates = list(upload_dir.glob(f"{h[:32]}_*"))
    if candidates:
        return candidates[0]
    # Fallback: exact filename match
    candidates = list(upload_dir.glob(f"*{fn}"))
    if candidates:
        return candidates[0]
    return None


async def has_chunks(svc: Neo4jService, doc_id: str) -> bool:
    rows = await svc.run_query(
        """
        MATCH (d:Document {doc_id: $d})-[:HAS_PAGE]->(:Page)-[:HAS_CHUNK]->(c:Chunk)
        RETURN count(c) AS n
        """,
        {"d": doc_id},
    )
    return bool(rows) and int(rows[0]["n"]) > 0


async def pages_needing_extraction(svc: Neo4jService, doc_id: str) -> list[dict]:
    """Return pages whose extracted_text is populated but whose topic_tags
    field isn't — i.e. pages that predate Phase 3 entity extraction and
    would benefit from a re-run."""
    rows = await svc.run_query(
        """
        MATCH (d:Document {doc_id: $d})-[:HAS_PAGE]->(p:Page)
        WHERE p.extracted_text IS NOT NULL
          AND (p.topic_tags IS NULL OR size(p.topic_tags) = 0)
          AND coalesce(p.is_blank, false) = false
        RETURN p.page_id AS page_id, p.page_number AS page_number,
               p.extracted_text AS text
        ORDER BY p.page_number
        """,
        {"d": doc_id},
    )
    return list(rows)


async def rebuild_one_doc(
    doc_id: str, title: str, *,
    svc: Neo4jService,
    chunker: StructuralChunker,
    summarizer: ChunkSummarizer,
    text_embedding,
    gpu: GPUManager,
    extractor: EntityExtractor | None,
    graph_builder: GraphBuilder,
    batch_size: int,
    skip_extract: bool,
    extract_only: bool = False,
) -> dict:
    """Full rebuild of one document. Returns stats dict.

    Two opt-outs let you scope the work:
    - skip_extract: run chunking/summarizing/embedding but NOT entity
      re-extraction. Fast Phase 2-only pass.
    - extract_only: skip chunking/summarizing/embedding, only re-run entity
      extraction on pages that still need it. Cheap retry after an
      extractor bug fix.
    """
    t0 = time.time()
    stats = {"chunks": 0, "extracted_pages": 0, "failed": 0}

    pdf_path = await find_pdf_path(svc, doc_id)
    if pdf_path is None:
        log.warning("PDF not found for %s — skipping", doc_id)
        stats["failed"] = 1
        return stats

    if not extract_only:
        # 1-3. Chunks
        log.info("[%s] chunking %s", doc_id, pdf_path.name)
        file_hash = (await svc.run_query(
            "MATCH (d:Document {doc_id: $d}) RETURN d.file_hash AS h", {"d": doc_id}
        ))[0]["h"]

        chunks = await asyncio.to_thread(chunker.chunk_pdf, pdf_path, file_hash)
        if not chunks:
            log.warning("[%s] no chunks produced — skipping", doc_id)
            return stats

        log.info("[%s] %d chunks; summarizing", doc_id, len(chunks))
        summaries = await summarizer.summarize_batch(chunks, concurrency=4)

        log.info("[%s] embedding %d chunk pairs", doc_id, len(chunks))
        embed_inputs = [f"{s}\n\n{c.text[:2000]}" for s, c in zip(summaries, chunks)]
        async with gpu.load_scope("text_embedding"):
            vectors = await asyncio.to_thread(
                text_embedding.embed_documents, embed_inputs, batch_size=batch_size,
            )
    else:
        log.info("[%s] extract-only mode — skipping chunk rebuild", doc_id)
        chunks = []
        summaries = []
        vectors = []

    # Write chunks in batches (no-op when extract_only and chunks == [])
    BATCH = 200
    for i in range(0, len(chunks), BATCH):
        end = min(i + BATCH, len(chunks))
        rows = []
        for ch, summ, vec in zip(
            chunks[i:end], summaries[i:end], vectors[i:end]
        ):
            rows.append({
                "chunk_id": ch.chunk_id,
                "page_number": ch.page_number,
                "chunk_index": ch.chunk_index,
                "chunk_type": ch.chunk_type,
                "text": ch.text,
                "summary": summ,
                "section_path": ch.section_path,
                "embedding": vec.tolist(),
                "bbox": list(ch.bbox) if ch.bbox is not None else None,
            })
        await svc.run_write(
            """
            UNWIND $rows AS row
            MATCH (d:Document {doc_id: $doc_id})-[:HAS_PAGE]->(p:Page {page_number: row.page_number})
            MERGE (c:Chunk {chunk_id: row.chunk_id})
            ON CREATE SET c.page_number = row.page_number,
                          c.chunk_index = row.chunk_index,
                          c.chunk_type = row.chunk_type,
                          c.text = row.text,
                          c.summary = row.summary,
                          c.section_path = row.section_path,
                          c.embedding = row.embedding,
                          c.bbox = row.bbox,
                          c.doc_id = $doc_id
            ON MATCH SET  c.text = row.text,
                          c.summary = row.summary,
                          c.section_path = row.section_path,
                          c.chunk_type = row.chunk_type,
                          c.embedding = row.embedding,
                          c.bbox = row.bbox
            MERGE (p)-[:HAS_CHUNK]->(c)
            """,
            {"doc_id": doc_id, "rows": rows},
        )
    stats["chunks"] = len(chunks)

    # 4-6. Re-run page-level entity extraction with the Phase 3 schema
    # (Formula, Table, topic_tags). Skip pages that already have topic_tags.
    if not skip_extract and extractor is not None:
        todo = await pages_needing_extraction(svc, doc_id)
        log.info(
            "[%s] re-extracting entities for %d pages (skipping %d already done)",
            doc_id, len(todo),
            await _page_count(svc, doc_id) - len(todo),
        )
        for i, p in enumerate(todo, 1):
            try:
                extraction = await extractor.extract_page(
                    document_title=title,
                    page_number=p["page_number"],
                    page_text=p["text"],
                )
                await graph_builder.write_page(
                    page_id=p["page_id"], extraction=extraction,
                )
                stats["extracted_pages"] += 1
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[%s] page %d extraction failed: %s",
                    doc_id, p["page_number"], exc,
                )
                stats["failed"] += 1
            if i % 50 == 0:
                log.info(
                    "[%s] re-extracted %d/%d pages", doc_id, i, len(todo),
                )

    elapsed = time.time() - t0
    log.info(
        "[%s] done in %.0fs — chunks=%d, re-extracted pages=%d",
        doc_id, elapsed, stats["chunks"], stats["extracted_pages"],
    )
    return stats


async def _page_count(svc: Neo4jService, doc_id: str) -> int:
    rows = await svc.run_query(
        "MATCH (d:Document {doc_id: $d})-[:HAS_PAGE]->(p:Page) RETURN count(p) AS n",
        {"d": doc_id},
    )
    return int(rows[0]["n"]) if rows else 0


async def _find_stale_vector_indexes(
    svc: Neo4jService, target_dim: int,
) -> list[tuple[str, int]]:
    """Return (name, actual_dim) for any VECTOR index whose stored
    dimension disagrees with target_dim. We need to drop these before
    applying the schema at the new dim since IF NOT EXISTS would skip
    them otherwise."""
    rows = await svc.run_query(
        """
        SHOW VECTOR INDEXES YIELD name, options
        RETURN name, options
        """
    )
    stale: list[tuple[str, int]] = []
    for r in rows:
        name = r.get("name")
        opts = r.get("options") or {}
        cfg = opts.get("indexConfig") or {}
        dim = cfg.get("vector.dimensions") or cfg.get("vector_dimensions")
        if isinstance(dim, int) and dim != target_dim:
            stale.append((name, dim))
    return stale


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--doc-id", default=None,
        help="Process a single document by doc_id (useful for testing)",
    )
    parser.add_argument(
        "--only-missing", action="store_true",
        help="Skip documents that already have Chunk nodes",
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Skip the Phase 3 entity re-extraction (chunks only)",
    )
    parser.add_argument(
        "--extract-only", action="store_true",
        help="Inverse of --skip-extract: skip chunking/summarizing/embedding "
        "and only re-run entity extraction on pages missing topic_tags. "
        "Cheap resume after fixing an extractor bug.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="BGE-M3 embedding batch size",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = get_settings()
    if not os.environ.get(settings.neo4j.password_env):
        log.error("Env var %s not set.", settings.neo4j.password_env)
        return 1

    # 1. Neo4j + schema
    svc = Neo4jService(settings.neo4j)
    await svc.connect()
    try:
        if not await svc.verify_connectivity():
            log.error("Cannot reach Neo4j at %s", settings.neo4j.uri)
            return 2
        # If the embedding dim changed (Nomic 768 → BGE-M3 1024), the old
        # page_text_embedding vector index is incompatible with new writes.
        # Drop any stale vector indexes whose dim doesn't match the target.
        # Schema re-application (with IF NOT EXISTS) will then recreate them
        # at the new dim. Existing 768-dim vectors in p.text_embedding
        # stay in place as orphaned data — they're ignored by a 1024-dim
        # index and get overwritten by future page-level embedding runs.
        target_dim = settings.models.text_embedding_dim
        stale = await _find_stale_vector_indexes(svc, target_dim)
        for idx_name, actual_dim in stale:
            log.warning(
                "Dropping vector index %s (dim=%d; target=%d)",
                idx_name, actual_dim, target_dim,
            )
            await svc.run_write(f"DROP INDEX {idx_name} IF EXISTS")

        log.info(
            "Applying schema (may create new Chunk/Formula/RefTable constraints + "
            "indexes at dim=%d)", target_dim,
        )
        await apply_schema(svc, embedding_dim=target_dim)

        # 2. Services
        gpu = GPUManager(idle_unload_seconds=settings.gpu.model_idle_unload_seconds)
        await gpu.start()
        text_embedding = create_text_embedding_service(settings, gpu)

        llm = create_llm_service(settings)
        await llm.start()
        if not await llm.health():
            log.error(
                "LLM endpoint %s not reachable — cannot summarize or re-extract. "
                "Start LM Studio / vLLM first.", settings.llm.endpoint,
            )
            return 3

        chunker = StructuralChunker()
        summarizer = ChunkSummarizer(llm)
        extractor = EntityExtractor(llm) if not args.skip_extract else None
        graph_builder = GraphBuilder(svc)

        if args.extract_only and args.skip_extract:
            log.error("--extract-only and --skip-extract are mutually exclusive")
            return 5
        if args.extract_only and extractor is None:
            log.error("--extract-only requires entity extraction enabled")
            return 5

        # 3. Doc list
        if args.doc_id:
            rows = await svc.run_query(
                "MATCH (d:Document {doc_id: $d}) RETURN d.doc_id AS id, d.title AS title",
                {"d": args.doc_id},
            )
        else:
            rows = await svc.run_query(
                "MATCH (d:Document) RETURN d.doc_id AS id, d.title AS title ORDER BY d.title"
            )
        if not rows:
            log.error("No documents found.")
            return 4

        totals = {"chunks": 0, "extracted_pages": 0, "failed": 0}
        for i, doc in enumerate(rows, 1):
            doc_id = doc["id"]
            title = doc["title"] or doc_id

            if args.only_missing and await has_chunks(svc, doc_id):
                log.info("[%s] already has chunks — skipping", doc_id)
                continue

            log.info("===== [%d/%d] %s =====", i, len(rows), title)
            try:
                result = await rebuild_one_doc(
                    doc_id, title,
                    svc=svc, chunker=chunker, summarizer=summarizer,
                    text_embedding=text_embedding, gpu=gpu,
                    extractor=extractor, graph_builder=graph_builder,
                    batch_size=args.batch_size,
                    skip_extract=args.skip_extract,
                    extract_only=args.extract_only,
                )
                for k, v in result.items():
                    totals[k] = totals.get(k, 0) + v
            except Exception as exc:  # noqa: BLE001
                log.exception("[%s] rebuild failed: %s", doc_id, exc)
                totals["failed"] += 1

        print()
        print("=" * 60)
        print("Rebuild complete")
        print("=" * 60)
        print(f"Documents processed: {len(rows)}")
        print(f"Total chunks written: {totals['chunks']}")
        print(f"Pages re-extracted:  {totals['extracted_pages']}")
        print(f"Failures:            {totals['failed']}")
        print()
        print("Next: trigger a community rebuild from the Admin page "
              "(or POST /graph/build-communities) so the new Formula/"
              "Table/Chunk nodes are clustered into the community layer.")

        await llm.stop()
        await gpu.stop()
        return 0
    finally:
        await svc.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
