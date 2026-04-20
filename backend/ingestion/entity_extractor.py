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
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from backend.services.llm_service import LLMFatalError, LLMService, LLMTransientError

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- validators
#
# Defensive sanitation for LLM output. The model sometimes emits garbage for
# specific fields — JSON structural debris, the extraction prompt itself,
# paragraph-length bibliographic references as entity names, etc. These
# helpers catch the known failure modes and return a sentinel (empty string
# for names, None for quantity strings) so graph_builder skips the bad item.
# We never raise — a single bad entity should not cost us the rest of the
# page's extraction.

# Signature phrases from the extraction prompt. If any of these appear in an
# LLM-returned value, the model hallucinated the prompt as the answer.
_PROMPT_SIGNATURES = (
    "you are an engineering",
    "extraction assistant",
    "output json only",
    "allowed material_type",
    "allowed process_type",
    "allowed equipment_type",
    "allowed organization",
    "allowed relationship",
    "entity naming rules",
)

# Characters that only ever show up in garbled JSON output, never in a real
# entity name or quantity string.
_JSON_DEBRIS_CHARS = set('{}[]"')

_WS_RE = re.compile(r"\s+")


def _looks_like_prompt_leakage(s: str) -> bool:
    low = s.lower()
    return any(sig in low for sig in _PROMPT_SIGNATURES)


def _longest_nondigit_run(s: str) -> int:
    """Used by the quantity-string validator. Long spans of non-digits
    indicate prose rather than a quantity like "45 ksi" or "100 to 325 HB"."""
    cur = best = 0
    for ch in s:
        if ch.isdigit():
            cur = 0
        else:
            cur += 1
            if cur > best:
                best = cur
    return best


def clean_entity_name(v: str | None, *, max_len: int = 80) -> str:
    """Validate and normalize an entity name. Returns empty string on any
    signal that the value is prose, JSON debris, or prompt leakage rather
    than a canonical engineering name.

    max_len default is 80. Any engineering entity name that needs more than
    80 characters is almost certainly prose — the longest legitimate names
    we see are things like "austenitic stainless steel" (27 chars) or
    "gray cast iron, Grade 40A" (25). Callers that want tighter limits
    (e.g. UNS numbers, process numbers, clause IDs) pass smaller values.
    """
    if not v:
        return ""
    s = _WS_RE.sub(" ", str(v)).strip()
    if not s:
        return ""
    if len(s) > max_len:
        logger.warning("rejecting overly-long entity name (%d chars): %.80s...", len(s), s)
        return ""
    if any(c in _JSON_DEBRIS_CHARS for c in s):
        logger.warning("rejecting entity name with JSON debris chars: %.80s", s)
        return ""
    if _looks_like_prompt_leakage(s):
        logger.warning("rejecting entity name containing prompt text: %.80s", s)
        return ""
    # Multi-sentence prose: sentence-ending period followed by a capital.
    # Tolerates "Grade 3.5" and "No. 1" since those have digits on either side.
    if re.search(r"[a-z]\. [A-Z]", s):
        logger.warning("rejecting entity name that reads as prose: %.80s", s)
        return ""
    # Bibliographic-reference pattern: 3+ commas indicate a citation with
    # multiple fields (author, title, edition, year). Engineering entity
    # names are short phrases with at most one comma (e.g. "gray cast
    # iron, Grade 40A"). Catches refs like "Alloy CF-8 J. M. SVOBODA,
    # HIGH-ALLOY STEELS, CASTING, 9TH ED., VOL 15, METALS HANDBOOK..."
    if s.count(",") >= 3:
        logger.warning("rejecting bibliographic-style entity name: %.80s", s)
        return ""
    return s


def clean_quantity_str(v: str | None) -> str | None:
    """Validate a numeric-field string (tensile/yield/hardness). Returns
    None if the value is JSON debris, prose, prompt leakage, or otherwise
    not a plausible quantity. The rule matches cleanup_numeric_garbage.py
    so ingestion produces the same shape the cleanup script would accept."""
    if v is None or v == "":
        return None
    s = str(v).strip()
    if not s:
        return None
    if any(c in _JSON_DEBRIS_CHARS for c in s):
        return None
    if s.startswith(","):
        # Parser-misfile marker — neighbouring-field content leaked in.
        return None
    if not any(c.isdigit() for c in s):
        # Pure prose like "various", "high", "annealed".
        return None
    if _looks_like_prompt_leakage(s):
        return None
    if _longest_nondigit_run(s) > 30:
        # Prose with an occasional digit (A572 descriptive text).
        return None
    return s


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
    "pure_element",       # nickel, lead, iron, tungsten, etc. — not an alloy
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
    # Mechanical / materials / welding / petroleum — the original set
    "ASME", "ASTM", "API", "AWS", "ISO", "NACE", "MSS", "ANSI",
    # Electrical, electronics, safety
    "IEEE", "IEC", "NEMA", "NFPA", "UL", "IPC", "JEDEC",
    # Regional bodies frequently cited in design handbooks
    "DIN", "JIS", "BS", "EN",
    # HVAC & building
    "ASHRAE",
    # Fallback
    "other",
]

RelationshipType = Literal[
    "mentions_material",
    "describes_process",
    "references_standard",
    "mentions_equipment",
    "mentions_formula",
    "mentions_table",
    # Entity-to-entity relationships inferred from the same page:
    "material_compatible_with_process",
    "material_governed_by_standard",
    "process_governed_by_standard",
    "equipment_governed_by_standard",
    "standard_references_standard",
    "formula_uses_material",
    "table_describes_material",
]


# Phase 3: Formula and reference-Table types. Both are optional in the
# schema — the LLM emits them only when the chunk actually contains one.
FormulaKind = Literal[
    "stress", "strength", "deflection", "torque", "power",
    "thermodynamics", "fluid", "electrical", "geometry",
    "fatigue", "heat_transfer", "chemistry", "other",
]

TableKind = Literal[
    "properties",         # material / fluid property table
    "dimensions",         # thread, fastener, pipe, gear dimensions
    "specifications",     # bolt torque, wire gauge, tap drill
    "conversion",         # unit conversion tables
    "selection",          # part / material selection charts
    "performance",        # motor, pump, compressor performance curves
    "other",
]


class MaterialMention(BaseModel):
    name: str = Field(..., description="Canonical name or designation, e.g. 'ASTM A36' or 'Alloy 625'")
    material_type: MaterialType = "unknown"
    uns_number: str | None = Field(None, description="UNS designation like S30400, if mentioned")
    common_names: list[str] = Field(default_factory=list, description="Aliases mentioned on the page")
    tensile_strength_ksi: str | None = None    # free-form with units, e.g. "58-80 ksi"
    yield_strength_ksi: str | None = None
    hardness: str | None = None

    @field_validator("name")
    @classmethod
    def _v_name(cls, v: str) -> str:
        return clean_entity_name(v)

    @field_validator("uns_number")
    @classmethod
    def _v_uns(cls, v: str | None) -> str | None:
        # UNS codes are short (e.g. "S30400"); reject prose-length strings.
        cleaned = clean_entity_name(v, max_len=40)
        return cleaned or None

    @field_validator("common_names")
    @classmethod
    def _v_common_names(cls, v: list[str]) -> list[str]:
        out = []
        for name in v or []:
            cleaned = clean_entity_name(name)
            if cleaned:
                out.append(cleaned)
        return out

    @field_validator("tensile_strength_ksi", "yield_strength_ksi", "hardness")
    @classmethod
    def _v_quantity(cls, v: str | None) -> str | None:
        return clean_quantity_str(v)


class ProcessMention(BaseModel):
    name: str = Field(..., description="Process name, e.g. 'GTAW', 'SMAW', 'quench and temper'")
    process_type: ProcessType = "other"
    process_number: str | None = Field(None, description="AWS/ASME designation like 'P-1' or 'F-4'")
    common_names: list[str] = Field(default_factory=list, description="Aliases mentioned on the page")

    @field_validator("name")
    @classmethod
    def _v_name(cls, v: str) -> str:
        return clean_entity_name(v, max_len=80)

    @field_validator("process_number")
    @classmethod
    def _v_num(cls, v: str | None) -> str | None:
        cleaned = clean_entity_name(v, max_len=30)
        return cleaned or None

    @field_validator("common_names")
    @classmethod
    def _v_common_names(cls, v: list[str]) -> list[str]:
        return [c for c in (clean_entity_name(x) for x in v or []) if c]


class StandardMention(BaseModel):
    code: str = Field(
        ...,
        description="SHORT canonical designation only, e.g. 'ASME BPVC IX', "
                    "'ASTM A36', 'NFPA 70', 'SEMI S2', 'IEEE 519'. "
                    "Never include the descriptive title.",
    )
    organization: StandardOrg = "other"
    number: str = Field(..., description="Standard number, e.g. 'BPVC-IX' or 'A370'")
    title: str | None = Field(
        None,
        description="Full descriptive title if present on the page, e.g. "
                    "'Environmental, Health, and Safety Guideline for "
                    "Semiconductor Manufacturing Equipment'. Separate from "
                    "code; both are searchable.",
    )
    section: str | None = None
    clause_id: str | None = Field(None, description="If a specific clause is cited, e.g. 'QW-451.1'")
    common_names: list[str] = Field(default_factory=list, description="Alternate code spellings on the page")

    @field_validator("code")
    @classmethod
    def _v_code(cls, v: str) -> str:
        # Codes are short structured designators. 40-char cap rejects the
        # full descriptive titles the LLM sometimes drops in here — those
        # belong in `title`. Book titles / journal references still get
        # blocked by the prose-shape filters in clean_entity_name.
        return clean_entity_name(v, max_len=40)

    @field_validator("number")
    @classmethod
    def _v_number(cls, v: str | None) -> str:
        # number is declared str (required). If the LLM omits it or emits
        # something that fails sanitation, return "" rather than None —
        # graph_builder skips rows whose name/code is empty.
        if v is None:
            return ""
        return clean_entity_name(v, max_len=40)

    @field_validator("title")
    @classmethod
    def _v_title(cls, v: str | None) -> str | None:
        # Full titles can run long (SEMI, IEEE, and NEMA conventions are
        # descriptive). 150 chars easily fits the longest real titles
        # while still rejecting prose paragraphs if one slips in here.
        if v is None:
            return None
        s = _WS_RE.sub(" ", str(v)).strip()
        if not s:
            return None
        if len(s) > 150:
            logger.warning(
                "rejecting overly-long standard title (%d chars): %.80s...",
                len(s), s,
            )
            return None
        if any(c in _JSON_DEBRIS_CHARS for c in s):
            return None
        if _looks_like_prompt_leakage(s):
            return None
        # Bibliographic-style references (multiple commas) are still
        # unwelcome here — the title of ONE standard rarely has 4+ commas.
        if s.count(",") >= 5:
            return None
        return s

    @field_validator("section", "clause_id")
    @classmethod
    def _v_short(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = clean_entity_name(v, max_len=40)
        return cleaned or None

    @field_validator("common_names")
    @classmethod
    def _v_common_names(cls, v: list[str]) -> list[str]:
        return [c for c in (clean_entity_name(x) for x in v or []) if c]


class EquipmentMention(BaseModel):
    name: str
    equipment_type: EquipmentType = "other"
    common_names: list[str] = Field(default_factory=list, description="Aliases mentioned on the page")

    @field_validator("name")
    @classmethod
    def _v_name(cls, v: str) -> str:
        return clean_entity_name(v, max_len=80)

    @field_validator("common_names")
    @classmethod
    def _v_common_names(cls, v: list[str]) -> list[str]:
        return [c for c in (clean_entity_name(x) for x in v or []) if c]


class Relationship(BaseModel):
    type: RelationshipType
    subject: str = Field(..., description="Name of the 'from' entity")
    object: str = Field(..., description="Name of the 'to' entity")
    context: str | None = Field(
        None, description="Short phrase from the page that supports this relationship"
    )

    @field_validator("subject", "object")
    @classmethod
    def _v_endpoint(cls, v: str) -> str:
        # "page" and "chunk" are valid subject sentinels for content-to-entity
        # edges. Anything else must be a real canonical entity name.
        if v and v.strip().lower() in ("page", "chunk"):
            return v.strip().lower()
        return clean_entity_name(v)

    @field_validator("context")
    @classmethod
    def _v_context(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = _WS_RE.sub(" ", v).strip()
        if not s:
            return None
        # Context is a "short phrase" — cap it and strip prompt leakage.
        if len(s) > 500:
            s = s[:500]
        if _looks_like_prompt_leakage(s):
            return None
        return s


class FormulaMention(BaseModel):
    """A named engineering formula extracted from the text.

    Not every equation is worth extracting — just the ones with a name
    or a clear purpose (beam deflection, power loss, torque required,
    etc.). Nameless transient equations in a derivation should be left
    alone. The prompt guides the LLM to this judgment.
    """
    name: str = Field(..., description="Short name, e.g. 'cantilever beam deflection (end load)'")
    kind: FormulaKind = "other"
    expression: str | None = Field(
        None, description="The equation as written, with original symbols."
    )
    variables: list[str] = Field(
        default_factory=list,
        description="Variable definitions, e.g. ['P = applied load', 'L = length', 'E = elastic modulus']",
    )
    context: str | None = Field(
        None, description="Short phrase of where/how it's used on the page."
    )

    @field_validator("name")
    @classmethod
    def _v_name(cls, v: str) -> str:
        return clean_entity_name(v, max_len=120)

    @field_validator("expression")
    @classmethod
    def _v_expr(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        if any(c in _JSON_DEBRIS_CHARS for c in s):
            return None
        if _looks_like_prompt_leakage(s):
            return None
        if len(s) > 500:
            return None
        return s

    @field_validator("variables")
    @classmethod
    def _v_vars(cls, v: list[str]) -> list[str]:
        out = []
        for item in v or []:
            s = _WS_RE.sub(" ", str(item)).strip()
            if not s or len(s) > 200 or _looks_like_prompt_leakage(s):
                continue
            out.append(s)
        return out[:20]  # cap absurd variable lists

    @field_validator("context")
    @classmethod
    def _v_ctx(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = _WS_RE.sub(" ", v).strip()
        if not s or len(s) > 500 or _looks_like_prompt_leakage(s):
            return None
        return s


class TableMention(BaseModel):
    """A reference table extracted from the text — the kind of thing a
    design handbook has hundreds of. E.g. "Table 12-3: Tap drill sizes
    for UNC and UNF threads". We capture the title, its topic, and a
    short natural-language description of what the table contains so it's
    findable by topic rather than just by literal text match."""
    title: str = Field(..., description="Table title as printed, e.g. 'Table 12-3: Tap Drill Sizes'")
    kind: TableKind = "other"
    description: str | None = Field(
        None,
        description="1-2 sentences describing what the table contains "
        "in natural language (for retrieval). E.g. 'Tap drill diameters "
        "in inches for UNC/UNF threads from #0 through 1-1/2 inch.'",
    )
    subject_entities: list[str] = Field(
        default_factory=list,
        description="Named entities the table describes, e.g. ['UNC thread',"
        " '4140 steel', 'ISO metric thread'].",
    )

    @field_validator("title")
    @classmethod
    def _v_title(cls, v: str) -> str:
        return clean_entity_name(v, max_len=150)

    @field_validator("description")
    @classmethod
    def _v_desc(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = _WS_RE.sub(" ", v).strip()
        if not s or len(s) > 500 or _looks_like_prompt_leakage(s):
            return None
        return s

    @field_validator("subject_entities")
    @classmethod
    def _v_subjects(cls, v: list[str]) -> list[str]:
        return [s for s in (clean_entity_name(x) for x in v or []) if s][:10]


class PageExtraction(BaseModel):
    materials: list[MaterialMention] = Field(default_factory=list)
    processes: list[ProcessMention] = Field(default_factory=list)
    standards: list[StandardMention] = Field(default_factory=list)
    equipment: list[EquipmentMention] = Field(default_factory=list)
    formulas: list[FormulaMention] = Field(default_factory=list)
    tables: list[TableMention] = Field(default_factory=list)
    topic_tags: list[str] = Field(
        default_factory=list,
        description="1-5 page-level topic tags in kebab-case, "
        "e.g. ['tap-drill-chart', 'thread-specifications', "
        "'fastener-torque']. Used as fast filters for retrieval.",
    )
    relationships: list[Relationship] = Field(default_factory=list)

    @field_validator("topic_tags")
    @classmethod
    def _v_tags(cls, v: list[str]) -> list[str]:
        out = []
        seen: set[str] = set()
        for raw in v or []:
            s = _WS_RE.sub("-", str(raw).strip().lower())
            # Normalize: lowercase kebab-case, alphanumerics + hyphen.
            s = re.sub(r"[^a-z0-9-]+", "", s)
            s = s.strip("-")
            if not s or len(s) > 60 or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out[:8]  # cap hallucinated tag lists


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
  "materials":  [ { "name", "material_type", "uns_number", "common_names",
                    "tensile_strength_ksi", "yield_strength_ksi", "hardness" } ],
  "processes":  [ { "name", "process_type", "process_number", "common_names" } ],
  "standards":  [ { "code", "organization", "number", "title", "section",
                    "clause_id", "common_names" } ],
  "equipment":  [ { "name", "equipment_type", "common_names" } ],
  "formulas":   [ { "name", "kind", "expression", "variables", "context" } ],
  "tables":     [ { "title", "kind", "description", "subject_entities" } ],
  "topic_tags": [ "kebab-case", "page", "topics" ],
  "relationships": [ { "type", "subject", "object", "context" } ]
}

`common_names` lists aliases/alternate spellings the text uses for the same
entity (e.g. "Inconel 718" and "Alloy 718" and "IN-718" all refer to one
material — emit the canonical name plus the other spellings in common_names).
Omit or leave empty if only one spelling appears on the page.

`formulas` captures NAMED engineering formulas — equations with a purpose
(beam deflection, torque required, heat transfer coefficient, etc.). Skip
nameless transient equations in a derivation. Prefer a short name like
"cantilever beam deflection (end load)" over a literal LaTeX rendering.

`tables` captures the reference tables that design handbooks are full of:
tap drill charts, thread dimension tables, bolt torque specs, wire gauge
tables, beam property tables, motor performance curves. Emit one entry per
distinct table, with `title` as printed and `description` in natural
language describing what the table contains ("Tap drill diameters in
inches for UNC and UNF threads from #0 through 1-1/2 inch"). This makes
tables findable by topic, not just by a literal title match.

`topic_tags` are 1-5 short kebab-case tags describing what the page is
ABOUT — used as fast filters at retrieval time. Examples:
"tap-drill-chart", "thread-specifications", "bolt-torque",
"beam-deflection-formula", "wire-gauge-ampacity", "gear-tooth-form",
"bearing-selection", "pipe-schedule-dimensions".
Emit between 0 and 5 tags. Don't invent tags that aren't supported by
the content — leave empty if the page is boilerplate or unclear.

Entity naming rules:
- Entity names must be short canonical designations, not sentences.
  Keep names under ~60 characters. Never copy bibliographic references,
  book titles, journal names, or multi-sentence descriptions as names.
- Materials: use canonical designations. "ASTM A36" not "A-36" or "A 36".
  Alloys: "Alloy 625" or "Inconel 625" (use exactly what's on the page).
- Processes: use standard abbreviations (GTAW, SMAW, GMAW, FCAW, SAW, etc.)
  for welding, or the exact name for heat treatments ("normalizing",
  "quench and temper", "stress relief annealing").
- Standards: `code` is the SHORT canonical designation only. Put the
  descriptive title (if any) in `title`, NOT in `code`.
    * "ASME BPVC IX" → code="ASME BPVC IX", title="Welding, Brazing, and
      Fusing Qualifications" (if the title is printed on the page).
    * "ASTM A36" → code="ASTM A36", title="Carbon Structural Steel".
    * "SEMI S2" → code="SEMI S2", title="Environmental, Health, and Safety
      Guideline for Semiconductor Manufacturing Equipment".
    * "NFPA 70" → code="NFPA 70", title="National Electrical Code".
  If the page shows ONLY the full title and never the short code, still
  put a best-effort short code (e.g. "SEMI S2") in `code` and the full
  title in `title`. If you cannot determine a short code, omit the entry.
  If a specific clause is referenced (e.g. "QW-451.1"), include it as
  clause_id. Do NOT emit book titles or journal names as standards.
- Equipment: generic type when possible ("pressure vessel", "pipe",
  "heat exchanger").

Quantity-field rules (tensile_strength_ksi, yield_strength_ksi, hardness):
- Emit ONLY a numeric value with an optional unit, e.g. "58-80 ksi",
  "450 MPa", "100 to 325 HB", "QT: 90-115 HB".
- If you don't know the value, use null. Never emit prose, never copy
  field schema text like "yield_strength_ksi: 50" into the string.
- Never leave a leading comma or JSON fragment in the value.

Allowed material_type values: carbon_steel, alloy_steel, stainless_steel,
aluminum_alloy, nickel_alloy, copper_alloy, titanium_alloy, cast_iron,
pure_element, polymer, ceramic, composite, other, unknown.

Use `pure_element` for unalloyed pure metals (nickel, iron, lead, zinc,
magnesium, tungsten, molybdenum, copper, aluminum, titanium, etc. when
referenced as the pure element — NOT as "Aluminum alloy 6061" or
"Titanium Grade 5").

Allowed process_type values: welding, brazing, soldering, heat_treatment,
machining, casting, forming, forging, nde, surface_treatment, joining, other.

Allowed equipment_type values: pressure_vessel, heat_exchanger, piping,
tank, structural, rotating, other.

Allowed organization values:
- Mechanical/materials/welding/petroleum: ASME, ASTM, API, AWS, ISO, NACE, MSS, ANSI
- Electrical/electronics/safety: IEEE, IEC, NEMA, NFPA, UL, IPC, JEDEC
- Regional: DIN, JIS, BS, EN
- HVAC/building: ASHRAE
- Fallback: other

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
            # Standards-heavy pages (especially NFPA / ASME cross-reference
            # appendices) can emit long extractions listing dozens of
            # standards/clauses. Default max_tokens (4096) truncates the
            # JSON mid-field. 8192 clears the vast majority of cases at
            # a modest cost per page.
            extraction = await self.llm.chat_json_structured(
                messages, PageExtraction, max_tokens=8192,
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
