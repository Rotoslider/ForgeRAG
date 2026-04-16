"""Write extracted entities + relationships into Neo4j.

Takes PageExtraction objects from entity_extractor and MERGEs them into the
knowledge graph:

- Materials, Processes, Standards, Clauses, Equipment are MERGEd on their
  unique-constraint keys (name, code, clause_id, etc.) so re-running extraction
  doesn't create duplicates.
- Page relationships (page MENTIONS_MATERIAL, DESCRIBES_PROCESS, etc.) are
  created per page with a `context` property for the supporting text.
- Entity-to-entity relationships (material COMPATIBLE_WITH_PROCESS, etc.) are
  MERGEd so the same edge from multiple pages counts once but accumulates
  `support_count` and the first `context` seen.

Name normalization is intentionally lightweight. We trim whitespace, collapse
internal whitespace, and uppercase known abbreviations (UNS numbers). We do
NOT try to fuzzy-match "304 SS" to "304 Stainless Steel" — that's a Phase 5+
concern once we see how messy the extractions are in practice.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

from backend.ingestion.entity_extractor import (
    PageExtraction,
    Relationship,
    RelationshipType,
)
from backend.services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)


# Relationship type -> Cypher relationship name on the Page->entity side
_PAGE_REL_MAP: dict[RelationshipType, str] = {
    "mentions_material": "MENTIONS_MATERIAL",
    "describes_process": "DESCRIBES_PROCESS",
    "references_standard": "REFERENCES_STANDARD",
    "mentions_equipment": "MENTIONS_EQUIPMENT",
}

# Entity->entity relationships: (neo4j rel, src label, dst label)
_ENTITY_REL_MAP: dict[RelationshipType, tuple[str, str, str]] = {
    "material_compatible_with_process": ("COMPATIBLE_WITH_PROCESS", "Material", "Process"),
    "material_governed_by_standard":    ("GOVERNED_BY",             "Material", "Standard"),
    "process_governed_by_standard":     ("GOVERNED_BY",             "Process",  "Standard"),
    "equipment_governed_by_standard":   ("GOVERNED_BY",             "Equipment","Standard"),
    "standard_references_standard":     ("REFERENCES",              "Standard", "Standard"),
}

_WS = re.compile(r"\s+")


def _norm_name(name: str) -> str:
    """Light normalization: collapse whitespace, strip outer whitespace."""
    return _WS.sub(" ", name).strip()


class GraphBuilder:
    """Writes a PageExtraction into Neo4j for a given page_id."""

    def __init__(self, neo4j: Neo4jService):
        self.neo4j = neo4j

    async def write_page(
        self, *, page_id: str, extraction: PageExtraction
    ) -> dict[str, int]:
        """Persist all entities + relationships from one page. Returns counts."""
        counts = {
            "materials": 0,
            "processes": 0,
            "standards": 0,
            "clauses": 0,
            "equipment": 0,
            "page_rels": 0,
            "entity_rels": 0,
        }

        # ---- Materials ---------------------------------------------------
        mat_rows = []
        for m in extraction.materials:
            name = _norm_name(m.name)
            if not name:
                continue
            mat_rows.append({
                "name": name,
                "material_type": m.material_type,
                "uns_number": m.uns_number,
                "common_names": [_norm_name(x) for x in m.common_names if x.strip()],
                "tensile_strength_ksi": m.tensile_strength_ksi,
                "yield_strength_ksi": m.yield_strength_ksi,
                "hardness": m.hardness,
            })
        if mat_rows:
            await self.neo4j.run_write(
                """
                UNWIND $rows AS row
                MERGE (m:Material {name: row.name})
                ON CREATE SET
                    m.material_type = row.material_type,
                    m.uns_number = row.uns_number,
                    m.common_names = row.common_names,
                    m.tensile_strength_ksi = row.tensile_strength_ksi,
                    m.yield_strength_ksi = row.yield_strength_ksi,
                    m.hardness = row.hardness
                ON MATCH SET
                    m.material_type = CASE WHEN m.material_type IN [NULL, 'unknown', 'other']
                                           THEN row.material_type ELSE m.material_type END,
                    m.uns_number = coalesce(m.uns_number, row.uns_number),
                    m.tensile_strength_ksi = coalesce(m.tensile_strength_ksi, row.tensile_strength_ksi),
                    m.yield_strength_ksi = coalesce(m.yield_strength_ksi, row.yield_strength_ksi),
                    m.hardness = coalesce(m.hardness, row.hardness)
                """,
                {"rows": mat_rows},
            )
            counts["materials"] = len(mat_rows)

        # ---- Processes ---------------------------------------------------
        proc_rows = []
        for p in extraction.processes:
            name = _norm_name(p.name)
            if not name:
                continue
            proc_rows.append({
                "name": name,
                "process_type": p.process_type,
                "process_number": p.process_number,
            })
        if proc_rows:
            await self.neo4j.run_write(
                """
                UNWIND $rows AS row
                MERGE (p:Process {name: row.name})
                ON CREATE SET p.process_type = row.process_type,
                              p.process_number = row.process_number
                ON MATCH SET  p.process_type = CASE WHEN p.process_type IN [NULL, 'other']
                                                    THEN row.process_type ELSE p.process_type END,
                              p.process_number = coalesce(p.process_number, row.process_number)
                """,
                {"rows": proc_rows},
            )
            counts["processes"] = len(proc_rows)

        # ---- Standards + their Clauses -----------------------------------
        std_rows = []
        clause_rows = []
        for s in extraction.standards:
            code = _norm_name(s.code)
            if not code:
                continue
            std_rows.append({
                "code": code,
                "organization": s.organization,
                "number": _norm_name(s.number),
                "section": s.section,
            })
            if s.clause_id:
                clause_rows.append({
                    "standard_code": code,
                    "clause_id": _norm_name(s.clause_id),
                })

        if std_rows:
            await self.neo4j.run_write(
                """
                UNWIND $rows AS row
                MERGE (s:Standard {code: row.code})
                ON CREATE SET s.organization = row.organization,
                              s.number = row.number,
                              s.section = row.section
                ON MATCH SET  s.organization = CASE WHEN s.organization IN [NULL, 'other']
                                                    THEN row.organization ELSE s.organization END,
                              s.section = coalesce(s.section, row.section)
                """,
                {"rows": std_rows},
            )
            counts["standards"] = len(std_rows)

        if clause_rows:
            await self.neo4j.run_write(
                """
                UNWIND $rows AS row
                MERGE (c:Clause {clause_id: row.clause_id})
                WITH c, row
                MATCH (s:Standard {code: row.standard_code})
                MERGE (s)-[:CONTAINS_CLAUSE]->(c)
                """,
                {"rows": clause_rows},
            )
            counts["clauses"] = len(clause_rows)

        # ---- Equipment ---------------------------------------------------
        eq_rows = []
        for e in extraction.equipment:
            name = _norm_name(e.name)
            if not name:
                continue
            eq_rows.append({"name": name, "equipment_type": e.equipment_type})
        if eq_rows:
            await self.neo4j.run_write(
                """
                UNWIND $rows AS row
                MERGE (e:Equipment {name: row.name})
                ON CREATE SET e.equipment_type = row.equipment_type
                ON MATCH SET  e.equipment_type = CASE WHEN e.equipment_type IN [NULL, 'other']
                                                      THEN row.equipment_type ELSE e.equipment_type END
                """,
                {"rows": eq_rows},
            )
            counts["equipment"] = len(eq_rows)

        # ---- Relationships -----------------------------------------------
        page_rels, entity_rels = self._split_rels(extraction.relationships)
        if page_rels:
            counts["page_rels"] = await self._write_page_rels(page_id, page_rels)
        if entity_rels:
            counts["entity_rels"] = await self._write_entity_rels(entity_rels)

        return counts

    # ----------------------------------------------------------------

    @staticmethod
    def _split_rels(
        rels: Iterable[Relationship],
    ) -> tuple[list[Relationship], list[Relationship]]:
        page_rels, entity_rels = [], []
        for r in rels:
            if r.type in _PAGE_REL_MAP:
                page_rels.append(r)
            elif r.type in _ENTITY_REL_MAP:
                entity_rels.append(r)
        return page_rels, entity_rels

    async def _write_page_rels(
        self, page_id: str, rels: list[Relationship]
    ) -> int:
        """Page->entity relationships. Group by rel type because Cypher labels
        can't be parameterized."""
        by_type: dict[str, list[dict]] = {}
        for r in rels:
            rel_name = _PAGE_REL_MAP[r.type]  # type: ignore[index]
            target_label = _target_label_for_page_rel(r.type)
            target_name = _norm_name(r.object)
            if not target_name:
                continue
            by_type.setdefault((rel_name, target_label), []).append(  # type: ignore[arg-type]
                {"target_name": target_name, "context": r.context}
            )

        written = 0
        for (rel_name, target_label), rows in by_type.items():
            # Note: rel_name and target_label are from a fixed whitelist above,
            # safe to interpolate.
            cypher = f"""
                UNWIND $rows AS row
                MATCH (p:Page {{page_id: $page_id}})
                MATCH (t:{target_label} {{{_pk_for_label(target_label)}: row.target_name}})
                MERGE (p)-[r:{rel_name}]->(t)
                ON CREATE SET r.context = row.context, r.support_count = 1
                ON MATCH SET  r.context = coalesce(r.context, row.context),
                              r.support_count = coalesce(r.support_count, 0) + 1
            """
            await self.neo4j.run_write(cypher, {"page_id": page_id, "rows": rows})
            written += len(rows)
        return written

    async def _write_entity_rels(self, rels: list[Relationship]) -> int:
        """Entity->entity relationships, grouped by (rel_name, src_label, dst_label)."""
        by_type: dict[tuple, list[dict]] = {}
        for r in rels:
            rel_name, src_label, dst_label = _ENTITY_REL_MAP[r.type]  # type: ignore[index]
            subj = _norm_name(r.subject)
            obj = _norm_name(r.object)
            if not subj or not obj:
                continue
            by_type.setdefault((rel_name, src_label, dst_label), []).append(
                {"src_name": subj, "dst_name": obj, "context": r.context}
            )

        written = 0
        for (rel_name, src_label, dst_label), rows in by_type.items():
            src_pk = _pk_for_label(src_label)
            dst_pk = _pk_for_label(dst_label)
            cypher = f"""
                UNWIND $rows AS row
                MATCH (s:{src_label} {{{src_pk}: row.src_name}})
                MATCH (d:{dst_label} {{{dst_pk}: row.dst_name}})
                MERGE (s)-[r:{rel_name}]->(d)
                ON CREATE SET r.context = row.context, r.support_count = 1
                ON MATCH SET  r.context = coalesce(r.context, row.context),
                              r.support_count = coalesce(r.support_count, 0) + 1
            """
            await self.neo4j.run_write(cypher, {"rows": rows})
            written += len(rows)
        return written


def _target_label_for_page_rel(rel_type: str) -> str:
    return {
        "mentions_material": "Material",
        "describes_process": "Process",
        "references_standard": "Standard",
        "mentions_equipment": "Equipment",
    }[rel_type]


def _pk_for_label(label: str) -> str:
    """Primary-key property name for each label (matches neo4j_schema constraints)."""
    return {
        "Material": "name",
        "Process": "name",
        "Standard": "code",
        "Clause": "clause_id",
        "Equipment": "name",
    }[label]
