"""Neo4j schema definition for ForgeRAG.

Defines all constraints and vector indexes needed by the knowledge graph.
Idempotent — uses IF NOT EXISTS on every statement so it can be re-run
safely during development or after schema migrations.

Entity types:
- Document, Page: source material with extracted content
- Category, Tag: organization (hierarchical categories + flat tags)
- Material, Process, Standard, Clause, Equipment: extracted engineering entities
- Community: GraphRAG hierarchical summary nodes
"""

from __future__ import annotations

import logging
from typing import Iterable

from backend.services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)


CONSTRAINTS: list[str] = [
    "CREATE CONSTRAINT doc_id_unique IF NOT EXISTS "
    "FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",

    "CREATE CONSTRAINT doc_hash_unique IF NOT EXISTS "
    "FOR (d:Document) REQUIRE d.file_hash IS UNIQUE",

    "CREATE CONSTRAINT page_id_unique IF NOT EXISTS "
    "FOR (p:Page) REQUIRE p.page_id IS UNIQUE",

    "CREATE CONSTRAINT category_name_unique IF NOT EXISTS "
    "FOR (c:Category) REQUIRE c.name IS UNIQUE",

    "CREATE CONSTRAINT tag_name_unique IF NOT EXISTS "
    "FOR (t:Tag) REQUIRE t.name IS UNIQUE",

    "CREATE CONSTRAINT material_name_unique IF NOT EXISTS "
    "FOR (m:Material) REQUIRE m.name IS UNIQUE",

    "CREATE CONSTRAINT process_name_unique IF NOT EXISTS "
    "FOR (p:Process) REQUIRE p.name IS UNIQUE",

    "CREATE CONSTRAINT standard_code_unique IF NOT EXISTS "
    "FOR (s:Standard) REQUIRE s.code IS UNIQUE",

    "CREATE CONSTRAINT clause_id_unique IF NOT EXISTS "
    "FOR (c:Clause) REQUIRE c.clause_id IS UNIQUE",

    "CREATE CONSTRAINT equipment_name_unique IF NOT EXISTS "
    "FOR (e:Equipment) REQUIRE e.name IS UNIQUE",

    "CREATE CONSTRAINT community_id_unique IF NOT EXISTS "
    "FOR (c:Community) REQUIRE c.community_id IS UNIQUE",
]

# Standard B-tree indexes for common lookup patterns
INDEXES: list[str] = [
    "CREATE INDEX page_doc_number IF NOT EXISTS "
    "FOR (p:Page) ON (p.page_number)",

    "CREATE INDEX document_title IF NOT EXISTS "
    "FOR (d:Document) ON (d.title)",

    "CREATE INDEX material_type IF NOT EXISTS "
    "FOR (m:Material) ON (m.material_type)",

    "CREATE INDEX process_type IF NOT EXISTS "
    "FOR (p:Process) ON (p.process_type)",

    "CREATE INDEX standard_organization IF NOT EXISTS "
    "FOR (s:Standard) ON (s.organization)",
]


# Full-text index for keyword search — Lucene-backed, handles 100K+ pages
# without scanning every row. Supports fuzzy matching, phrase queries, etc.
FULLTEXT_INDEXES: list[str] = [
    """CREATE FULLTEXT INDEX page_text_fulltext IF NOT EXISTS
       FOR (p:Page) ON EACH [p.extracted_text]""",
]


def vector_indexes(dim: int) -> list[str]:
    """Vector indexes — parameterized by embedding dimension."""
    return [
        f"""CREATE VECTOR INDEX page_text_embedding IF NOT EXISTS
           FOR (p:Page) ON (p.text_embedding)
           OPTIONS {{ indexConfig: {{
               `vector.dimensions`: {dim},
               `vector.similarity_function`: 'cosine'
           }} }}""",

        f"""CREATE VECTOR INDEX community_summary_embedding IF NOT EXISTS
           FOR (c:Community) ON (c.summary_embedding)
           OPTIONS {{ indexConfig: {{
               `vector.dimensions`: {dim},
               `vector.similarity_function`: 'cosine'
           }} }}""",
    ]


async def apply_schema(svc: Neo4jService, embedding_dim: int = 768) -> dict[str, int]:
    """Apply all constraints and indexes. Returns counts of each type applied.

    Idempotent — safe to run multiple times.
    """
    counts = {"constraints": 0, "indexes": 0, "vector_indexes": 0}

    async def _run_all(statements: Iterable[str], counter_key: str) -> None:
        for stmt in statements:
            try:
                await svc.run_write(stmt)
                counts[counter_key] += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Schema statement failed:\n%s\n%s", stmt, exc)
                raise

    logger.info("Applying Neo4j constraints...")
    await _run_all(CONSTRAINTS, "constraints")

    logger.info("Applying Neo4j indexes...")
    await _run_all(INDEXES, "indexes")

    logger.info("Applying Neo4j vector indexes (dim=%d)...", embedding_dim)
    await _run_all(vector_indexes(embedding_dim), "vector_indexes")

    logger.info("Applying Neo4j full-text indexes...")
    await _run_all(FULLTEXT_INDEXES, "fulltext_indexes")
    counts["fulltext_indexes"] = len(FULLTEXT_INDEXES)

    logger.info(
        "Schema applied: %d constraints, %d indexes, %d vector indexes",
        counts["constraints"], counts["indexes"], counts["vector_indexes"],
    )
    return counts
