"""N2 extraction-time noise valve.

The N1 review (docs/noise-review-2026-08.md) established that generic-noun
entities ("steel", "welding", "motor") are noise at retrieval time: they
carry no discriminating power and their fanout drowns designation matches.
N1 cleaned the stock retroactively; this valve keeps future ingests clean
at the source, in three moves:

1. REROUTE — an extracted entity whose (label, name) is on the banked
   blocklist (backend/resources/noise_blocklist.json, casefolded match)
   is removed from its entity lane and appended to the page's topic_tags,
   which is where page-level "this page is about steel" signals belong.
   No entity node, no mention edge, no new fanout.

2. VALIDATE RELATIONS — a model-declared relationship whose subject or
   object does not resolve against the entities extracted on the SAME
   page (primary names, aliases, UNS numbers, standard codes/numbers —
   all casefolded) is dropped as a logged validator decision. Previously
   these died silently in graph_builder's MATCH; worse, a hallucinated
   link to an entity from some other page would succeed. This valve
   deliberately tightens that: the relationship list adds context to
   what the page actually contains, nothing more. Relationships whose
   endpoint was rerouted in step 1 drop the same way.

3. FLAG, DON'T GUESS — a single lowercase/titlecase dictionary-looking
   word that is NOT on the blocklist ("bracket", "flange") is logged as
   a generic-candidate for the next N1 review round but is NOT dropped:
   single words can be real discriminators ("martensite", "austenite"),
   and degree — not wordform — is what proves noise. Standards are
   exempt (codes are structured).

The valve is pure (extraction in, extraction + report out) and is applied
by the pipeline immediately after every successful extract_page call,
before graph_builder.write_page.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from backend.ingestion.graph_builder import _ENTITY_REL_MAP, _PAGE_REL_MAP

logger = logging.getLogger(__name__)

_BLOCKLIST_PATH = (
    Path(__file__).resolve().parent.parent / "resources" / "noise_blocklist.json"
)

_blocklist_cache: dict[str, set[str]] | None = None

# Lanes: (extraction attr, graph label, primary-key attr on the mention)
_LANES = [
    ("materials", "Material", "name"),
    ("processes", "Process", "name"),
    ("standards", "Standard", "code"),
    ("equipment", "Equipment", "name"),
]

_GENERIC_WORD_RE = re.compile(r"^[A-Za-z]{4,}$")
_TAG_JUNK_RE = re.compile(r"[^a-z0-9-]+")
_MAX_TOPIC_TAGS = 12


def get_blocklist() -> dict[str, set[str]]:
    """Load the banked N1 blocklist once: {label: {casefolded names}}.

    A missing or malformed file is an empty blocklist (the valve's other
    two duties are independent of it), logged loudly once.
    """
    global _blocklist_cache
    if _blocklist_cache is None:
        table: dict[str, set[str]] = {}
        try:
            data = json.loads(_BLOCKLIST_PATH.read_text())
            for entry in data["stop_tier"]:
                table.setdefault(entry["label"], set()).add(
                    entry["name"].casefold()
                )
        except (OSError, KeyError, ValueError) as exc:
            logger.error(
                "noise-valve: blocklist unreadable at %s (%s) — "
                "reroute disabled this run", _BLOCKLIST_PATH, exc,
            )
        _blocklist_cache = table
    return _blocklist_cache


@dataclass
class ValveReport:
    rerouted: list[tuple[str, str]] = field(default_factory=list)
    dropped_rels: list[tuple[str, str, str, str]] = field(default_factory=list)
    generic_candidates: list[tuple[str, str]] = field(default_factory=list)

    @property
    def acted(self) -> bool:
        return bool(self.rerouted or self.dropped_rels)


def _kebab(name: str) -> str:
    s = _TAG_JUNK_RE.sub("-", name.strip().casefold()).strip("-")
    return s[:60]


def _looks_generic(name: str) -> bool:
    return bool(
        _GENERIC_WORD_RE.match(name)
        and (name.islower() or name.istitle())
    )


def apply_noise_valve(extraction, page_id: str = "?"):
    """Return (cleaned extraction, ValveReport). Pure — no I/O but logging."""
    blocklist = get_blocklist()
    report = ValveReport()

    updates: dict[str, list] = {}
    onpage: set[str] = set()

    for attr, label, pk in _LANES:
        blocked = blocklist.get(label, set())
        kept = []
        for mention in getattr(extraction, attr):
            primary = getattr(mention, pk)
            if primary.casefold() in blocked:
                report.rerouted.append((label, primary))
                continue
            kept.append(mention)
            # Every identifier this mention answers to can appear as a
            # relationship endpoint.
            onpage.add(primary.casefold())
            for alias in getattr(mention, "common_names", []) or []:
                onpage.add(alias.casefold())
            for extra_attr in ("uns_number", "process_number", "number"):
                extra = getattr(mention, extra_attr, None)
                if extra:
                    onpage.add(str(extra).casefold())
            if label != "Standard" and _looks_generic(primary):
                report.generic_candidates.append((label, primary))
        if len(kept) != len(getattr(extraction, attr)):
            updates[attr] = kept

    # Formulas and tables aren't noise-tiered, but their names are legal
    # relationship endpoints (formula_uses_material, table_describes_...).
    for f in getattr(extraction, "formulas", []) or []:
        onpage.add(f.name.casefold())
    for t in getattr(extraction, "tables", []) or []:
        onpage.add(t.title.casefold())

    kept_rels = []
    for rel in extraction.relationships:
        # Page-level rels carry the page as their implicit subject —
        # graph_builder only reads rel.object for them. Entity-entity
        # rels need both endpoints on-page. Unknown types pass through
        # untouched (graph_builder's _split_rels already ignores them).
        if rel.type in _PAGE_REL_MAP:
            endpoints = (rel.object,)
        elif rel.type in _ENTITY_REL_MAP:
            endpoints = (rel.subject, rel.object)
        else:
            kept_rels.append(rel)
            continue
        missing = [e for e in endpoints if e.casefold() not in onpage]
        if missing:
            rerouted_names = {n.casefold() for _, n in report.rerouted}
            why = ", ".join(
                f"{e} (rerouted to topic_tags)"
                if e.casefold() in rerouted_names
                else f"{e} (not extracted on page)"
                for e in missing
            )
            report.dropped_rels.append(
                (rel.type, rel.subject, rel.object, why)
            )
            continue
        kept_rels.append(rel)
    if len(kept_rels) != len(extraction.relationships):
        updates["relationships"] = kept_rels

    if report.rerouted:
        tags = list(extraction.topic_tags)
        for _, name in report.rerouted:
            tag = _kebab(name)
            if tag and tag not in tags and len(tags) < _MAX_TOPIC_TAGS:
                tags.append(tag)
        updates["topic_tags"] = tags

    if updates:
        extraction = extraction.model_copy(update=updates)

    if report.acted or report.generic_candidates:
        logger.info(
            "noise-valve page %s: rerouted=%s dropped_rels=%d%s "
            "generic_candidates=%s",
            page_id,
            [n for _, n in report.rerouted],
            len(report.dropped_rels),
            "".join(
                f" [{t}:{s}->{o}: {why}]"
                for t, s, o, why in report.dropped_rels[:5]
            ),
            [n for _, n in report.generic_candidates],
        )
    return extraction, report
