"""Depth-sample the bulk-era entity extractions (READ-ONLY).

The ~68k pages extracted April-July 2026 carry >=1 entity relationship each
(complete-JSON parses), but were extracted under older schemas and models
(Gemma 4 -> Qwen 3.5 -> Qwen 3.6) without the anti-bail retry. This script
answers "how much did those runs miss?" with data: it samples random
bulk-era pages, re-extracts them IN MEMORY with today's extractor, and
compares new entity counts against what the graph holds. Nothing is
written — no stamps, no relationships, no flags.

Usage (LLM must be idle or lightly loaded; each page costs a generation):
    NEO4J_PASSWORD=... FORGERAG_CONFIG=config/forgerag.toml \
        venv/bin/python scripts/depth_sample.py [n_pages]

Interpretation:
- ratio ~1x: era gap is cosmetic; the bulk extractions are sound.
- ratio >=2-3x on table-heavy pages: schedule the full re-extraction drain
  (delete rels + unstamp + queue, same convergent pattern as the
  suspicious-empty drain).
"""

from __future__ import annotations

import asyncio
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neo4j import AsyncGraphDatabase  # noqa: E402

from backend.config import get_settings  # noqa: E402
from backend.ingestion.entity_extractor import EntityExtractor  # noqa: E402
from backend.services.llm_service import create_llm_service  # noqa: E402

RELS = "MENTIONS_MATERIAL|DESCRIBES_PROCESS|REFERENCES_STANDARD|MENTIONS_EQUIPMENT"

SAMPLE_QUERY = f"""
MATCH (d:Document)-[:HAS_PAGE]->(p:Page)
WHERE p.entities_extracted_at IS NULL
  AND p.text_char_count >= 1000
  AND EXISTS {{ (p)-[:{RELS}]->() }}
WITH d, p, rand() AS r ORDER BY r LIMIT $n
RETURN d.title AS doc, p.page_number AS page, p.extracted_text AS text,
       p.text_char_count AS chars,
       COUNT {{ (p)-[:{RELS}]->() }} AS old_rels
"""


async def main(n: int) -> None:
    settings = get_settings()
    password = os.environ.get("NEO4J_PASSWORD", "")
    driver = AsyncGraphDatabase.driver(
        settings.neo4j.uri, auth=(settings.neo4j.user, password)
    )
    llm = create_llm_service(settings)
    await llm.start()
    extractor = EntityExtractor(llm)

    async with driver.session(database=settings.neo4j.database) as s:
        result = await s.run(SAMPLE_QUERY, n=n)
        rows = [r.data() async for r in result]
    print(f"Sampled {len(rows)} bulk-era pages (rels, no stamp)\n")

    ratios: list[float] = []
    failures = 0
    for i, r in enumerate(rows, 1):
        try:
            ex = await extractor.extract_page(
                document_title=r["doc"] or "(untitled)",
                page_number=r["page"],
                page_text=r["text"] or "",
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[{i:>3}] {r['doc'][:40]:<40} p{r['page']:<5} "
                  f"old={r['old_rels']:<4} new=FAILED ({str(exc)[:60]})")
            continue
        new = (len(ex.materials) + len(ex.processes)
               + len(ex.standards) + len(ex.equipment))
        old = r["old_rels"] or 0
        # old counts distinct rel edges; new counts mentions pre-dedup —
        # close enough for a ratio, exact only in aggregate.
        ratio = new / old if old else float(new or 0)
        ratios.append(ratio)
        print(f"[{i:>3}] {r['doc'][:40]:<40} p{r['page']:<5} "
              f"chars={r['chars']:<6} old={old:<4} new={new:<4} "
              f"ratio={ratio:.1f}x")

    await llm.stop()
    await driver.close()

    if ratios:
        print(f"\n=== {len(ratios)} pages compared, {failures} failed ===")
        print(f"mean ratio:   {statistics.mean(ratios):.2f}x")
        print(f"median ratio: {statistics.median(ratios):.2f}x")
        gain = sum(1 for x in ratios if x >= 2.0)
        print(f"pages where today's extractor finds >=2x: "
              f"{gain}/{len(ratios)} ({100 * gain / len(ratios):.0f}%)")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 50))
