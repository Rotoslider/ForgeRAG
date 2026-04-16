"""LLM-driven engineering entity + relationship extraction.

Sends page text to an instruction-tuned LLM and asks for a structured JSON
payload describing the materials, processes, standards/codes, clauses, and
equipment mentioned, plus the relationships between them.

Design:
- Single-page extraction keeps prompts small and latency predictable. We pay
  some redundancy cost (same entity extracted from many pages) but graph_builder
  deduplicates on MERGE.
- Ontology is intentionally tight (limited relationship types) so the graph
  stays queryable. The prompt lists the exact labels and relationship verbs
  we accept; everything else gets dropped by Pydantic validation.
- Properties are plain strings except for a few typed ranges (tensile, yield,
  hardness). We accept strings with units and let graph_builder normalize.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from backend.services.llm_service import LLMFatalError, LLMService, LLMTransientError

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- schema

MaterialType = Literal[
    "carbon_steel",
    "alloy_steel",
    "stainless_steel",
    "aluminum_alloy",
    "nickel_alloy",
    "copper_alloy",
    "titanium_alloy",
    "cast_iron",
    "polymer",
    "ceramic",
    "composite",
    "other",
    "unknown",
]

ProcessType = Literal[
    "welding",
    "brazing",
    "soldering",
    "heat_treatment",
    "machining",
    "casting",
    "forming",
    "forging",
    "nde",                # non-destructive examination
    "surface_treatment",
    "joining",
    "other",
]

EquipmentType = Literal[
    "pressure_vessel",
    "heat_exchanger",
    "piping",
    "tank",
    "structural",
    "rotating",
    "other",
]

StandardOrg = Literal[
    "ASME", "ASTM", "API", "AWS", "ISO", "NACE", "MSS", "ANSI", "other"
]

RelationshipType = Literal[
    "mentions_material",
    "describes_process",
    "references_standard",
    "mentions_equipment",
    # Entity-to-entity relationships inferred from the same page:
    "material_compatible_with_process",
    "material_governed_by_standard",
    "process_governed_by_standard",
    "equipment_governed_by_standard",
    "standard_references_standard",
]


class MaterialMention(BaseModel):
    name: str = Field(..., description="Canonical name or designation, e.g. 'ASTM A36' or 'Alloy 625'")
    material_type: MaterialType = "unknown"
    uns_number: str | None = Field(None, description="UNS designation like S30400, if mentioned")
    common_names: list[str] = Field(default_factory=list, description="Aliases mentioned on the page")
    tensile_strength_ksi: str | None = None    # free-form with units, e.g. "58-80 ksi"
    yield_strength_ksi: str | None = None
    hardness: str | None = None


class ProcessMention(BaseModel):
    name: str = Field(..., description="Process name, e.g. 'GTAW', 'SMAW', 'quench and temper'")
    process_type: ProcessType = "other"
    process_number: str | None = Field(None, description="AWS/ASME designation like 'P-1' or 'F-4'")


class StandardMention(BaseModel):
    code: str = Field(..., description="Full code, e.g. 'ASME BPVC Section IX'")
    organization: StandardOrg = "other"
    number: str = Field(..., description="Standard number, e.g. 'BPVC-IX' or 'A370'")
    section: str | None = None
    clause_id: str | None = Field(None, description="If a specific clause is cited, e.g. 'QW-451.1'")


class EquipmentMention(BaseModel):
    name: str
    equipment_type: EquipmentType = "other"


class Relationship(BaseModel):
    type: RelationshipType
    subject: str = Field(..., description="Name of the 'from' entity")
    object: str = Field(..., description="Name of the 'to' entity")
    context: str | None = Field(
        None, description="Short phrase from the page that supports this relationship"
    )


class PageExtraction(BaseModel):
    materials: list[MaterialMention] = Field(default_factory=list)
    processes: list[ProcessMention] = Field(default_factory=list)
    standards: list[StandardMention] = Field(default_factory=list)
    equipment: list[EquipmentMention] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)


# ----------------------------------------------------------------- prompt

SYSTEM_PROMPT = """\
You are an engineering knowledge extraction assistant. Given text from a single
page of an engineering document (handbook, standard, specification, or procedure),
extract the technical entities and relationships present.

Only include entities that are ACTUALLY MENTIONED in the text — do not invent
or infer entities from prior knowledge. If nothing relevant is on the page
(e.g. a table of contents, a blank page, a reference list), return empty arrays.

You must output a single JSON object matching this structure:

{
  "materials": [ { "name", "material_type", "uns_number", "common_names",
                   "tensile_strength_ksi", "yield_strength_ksi", "hardness" } ],
  "processes": [ { "name", "process_type", "process_number" } ],
  "standards": [ { "code", "organization", "number", "section", "clause_id" } ],
  "equipment": [ { "name", "equipment_type" } ],
  "relationships": [ { "type", "subject", "object", "context" } ]
}

Entity naming rules:
- Materials: use canonical designations. "ASTM A36" not "A-36" or "A 36".
  Alloys: "Alloy 625" or "Inconel 625" (use exactly what's on the page).
- Processes: use standard abbreviations (GTAW, SMAW, GMAW, FCAW, SAW, etc.)
  for welding, or the exact name for heat treatments ("normalizing",
  "quench and temper", "stress relief annealing").
- Standards: full code. "ASME BPVC Section IX" not just "IX". If a specific
  clause is referenced (e.g. "QW-451.1"), include it as clause_id.
- Equipment: generic type when possible ("pressure vessel", "pipe",
  "heat exchanger").

Allowed material_type values: carbon_steel, alloy_steel, stainless_steel,
aluminum_alloy, nickel_alloy, copper_alloy, titanium_alloy, cast_iron,
polymer, ceramic, composite, other, unknown.

Allowed process_type values: welding, brazing, soldering, heat_treatment,
machining, casting, forming, forging, nde, surface_treatment, joining, other.

Allowed equipment_type values: pressure_vessel, heat_exchanger, piping,
tank, structural, rotating, other.

Allowed organization values: ASME, ASTM, API, AWS, ISO, NACE, MSS, ANSI, other.

Allowed relationship types:
- mentions_material, describes_process, references_standard, mentions_equipment
  (these link the PAGE to the entity — use subject="page" for these)
- material_compatible_with_process, material_governed_by_standard,
  process_governed_by_standard, equipment_governed_by_standard,
  standard_references_standard
  (these link two entities mentioned on the same page)

Output JSON only. No prose, no code fences, no commentary.
"""

USER_PROMPT_TEMPLATE = """\
Extract entities and relationships from this page.

Document: {document_title}
Page: {page_number}

--- BEGIN PAGE TEXT ---
{page_text}
--- END PAGE TEXT ---

/no_think
"""
# The /no_think trailer is a Qwen3-family directive that disables the
# model's chain-of-thought deliberation. Without it, Qwen3.5 spends
# hundreds of tokens thinking before emitting JSON and takes 30+ s per
# page instead of ~8 s. Models that don't recognize the directive
# (Gemma 4, GLM Flash, Llama, etc.) ignore it as plain text — harmless.


# ----------------------------------------------------------------- extractor

class EntityExtractor:
    """Per-page extraction using the LLM service."""

    def __init__(self, llm: LLMService, *, max_page_chars: int = 12000):
        self.llm = llm
        self.max_page_chars = max_page_chars  # truncate overly long pages

    async def extract_page(
        self, *, document_title: str, page_number: int, page_text: str
    ) -> PageExtraction:
        """Extract entities from a single page. Returns empty PageExtraction on empty text."""
        if not page_text or not page_text.strip():
            return PageExtraction()

        truncated = page_text[: self.max_page_chars]
        user_msg = USER_PROMPT_TEMPLATE.format(
            document_title=document_title,
            page_number=page_number,
            page_text=truncated,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        try:
            extraction = await self.llm.chat_json_structured(
                messages, PageExtraction
            )
            return extraction
        except LLMTransientError as exc:
            logger.warning(
                "Transient LLM error extracting page %d: %s", page_number, exc
            )
            return PageExtraction()  # skip this page; graph remains valid
        except LLMFatalError as exc:
            logger.error(
                "Fatal LLM error extracting page %d: %s", page_number, exc
            )
            return PageExtraction()
