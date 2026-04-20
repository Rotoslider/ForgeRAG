#!/usr/bin/env python3
"""Dry-run: propose Tier 1 canonicalization merges for any entity label.

Same logic as canonicalize_materials_dryrun.py, parameterized by label.
Supports Material, Equipment, Process, Standard. Each label's primary key
and the label-specific plural-singularization set are configured below.

Usage:
    NEO4J_PASSWORD=... python scripts/canonicalize_entity_dryrun.py \
        --label Equipment
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_settings  # noqa: E402
from backend.services.neo4j_service import Neo4jService  # noqa: E402


# --- Label configs ---------------------------------------------------------

# Per-label config: primary-key field, the incoming-edge type that counts
# as a "mention" (for sizing the node's weight), and a label-specific set
# of plural→singular class nouns.
LABEL_CONFIG: dict[str, dict] = {
    "Material": {
        "pk": "name",
        "mention_rel": "MENTIONS_MATERIAL",
        "mention_source": "Page",
        "plurals": {
            "steels": "steel", "alloys": "alloy", "irons": "iron",
            "bronzes": "bronze", "brasses": "brass", "coppers": "copper",
        },
    },
    "Equipment": {
        "pk": "name",
        "mention_rel": "MENTIONS_EQUIPMENT",
        "mention_source": "Page",
        "plurals": {
            "pipes": "pipe", "tanks": "tank", "valves": "valve",
            "flanges": "flange", "bolts": "bolt", "welds": "weld",
            "joints": "joint", "fittings": "fitting", "nozzles": "nozzle",
            "vessels": "vessel", "exchangers": "exchanger",
            "compressors": "compressor", "pumps": "pump",
            "gaskets": "gasket", "bearings": "bearing",
            "gears": "gear", "shafts": "shaft", "couplings": "coupling",
            "springs": "spring", "seals": "seal", "screws": "screw",
            "nuts": "nut", "washers": "washer",
        },
    },
    "Process": {
        "pk": "name",
        "mention_rel": "DESCRIBES_PROCESS",
        "mention_source": "Page",
        "plurals": {
            # Process names are mostly gerunds ("welding", "machining") that
            # don't naturally pluralize, but a few do.
            "treatments": "treatment", "tests": "test",
            "inspections": "inspection", "operations": "operation",
        },
    },
    "Standard": {
        "pk": "code",
        "mention_rel": "REFERENCES_STANDARD",
        "mention_source": "Page",
        # Standard codes shouldn't be pluralized.
        "plurals": {},
    },
}


_CHEMICAL_FORMULA_RE = re.compile(r"^([A-Z][a-z]?\d*)+$")


def looks_like_chemical_formula(s: str) -> bool:
    if not s:
        return False
    has_upper = any(c.isupper() for c in s)
    has_lower = any(c.islower() for c in s)
    if not (has_upper and has_lower):
        return False
    return bool(_CHEMICAL_FORMULA_RE.match(s))


def build_canonicalizer(label: str):
    """Return a canonicalize() function closed over the label's plurals."""
    plurals = LABEL_CONFIG[label]["plurals"]
    if plurals:
        pat = re.compile(r"\b(" + "|".join(plurals) + r")\b", re.IGNORECASE)

        def _sing(m):
            return plurals[m.group(1).lower()]
    else:
        pat = None
        _sing = None

    def canonicalize(raw: str) -> str:
        s = (raw or "").strip()
        if not s:
            return ""
        # Chemical-formula guard: only relevant for Material, but cheap to
        # apply across labels — formulas shouldn't appear in other labels
        # so the guard is effectively a no-op there.
        if looks_like_chemical_formula(s):
            return s
        s = s.lower()
        s = re.sub(r"[\s_\-]+", " ", s).strip()
        if pat is not None:
            s = pat.sub(_sing, s)
        return s

    return canonicalize


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True,
                        choices=list(LABEL_CONFIG.keys()))
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "canonicalization")
    parser.add_argument("--min-mentions", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("canonicalize_dryrun")

    cfg = LABEL_CONFIG[args.label]
    pk = cfg["pk"]
    mention_rel = cfg["mention_rel"]
    mention_src = cfg["mention_source"]

    settings = get_settings()
    if not os.environ.get(settings.neo4j.password_env):
        log.error("Env var %s not set.", settings.neo4j.password_env)
        return 1

    svc = Neo4jService(settings.neo4j)
    await svc.connect()
    try:
        if not await svc.verify_connectivity():
            log.error("Cannot reach Neo4j at %s", settings.neo4j.uri)
            return 2

        fetch_cypher = f"""
            MATCH (m:{args.label})
            OPTIONAL MATCH (s:{mention_src})-[:{mention_rel}]->(m)
            WITH m, count(DISTINCT s) AS mentions
            OPTIONAL MATCH (m)-[r_out]->()
            WITH m, mentions, collect(type(r_out)) AS out_types
            RETURN m.{pk} AS name,
                   mentions,
                   out_types,
                   properties(m) AS props
        """
        log.info("Fetching all %s nodes...", args.label)
        rows = await svc.run_query(fetch_cypher)
        log.info("Fetched %d %s nodes", len(rows), args.label)
    finally:
        await svc.close()

    canonicalize = build_canonicalizer(args.label)

    groups: dict[str, list[dict]] = defaultdict(list)
    formula_guarded: list[dict] = []
    for r in rows:
        if r["mentions"] < args.min_mentions:
            continue
        raw = r["name"] or ""
        canon = canonicalize(raw)
        if not canon:
            continue
        if looks_like_chemical_formula(raw.strip()):
            formula_guarded.append({"name": raw, "mentions": r["mentions"]})
        groups[canon].append({
            "name": r["name"],
            "mentions": r["mentions"],
            "out_type_counts": dict(Counter(r["out_types"] or [])),
            "props": r["props"] or {},
        })

    multi = {
        canon: variants for canon, variants in groups.items()
        if len({v["name"] for v in variants}) > 1
    }

    naive_buckets = {c for c in groups if not looks_like_chemical_formula(c)}
    prevented_merges = [
        f for f in formula_guarded
        if f["name"].lower().strip() in naive_buckets
    ]

    def score(v: dict, canon: str) -> tuple:
        exact = 1 if v["name"] == canon else 0
        return (-v["mentions"], -exact, v["name"])

    plan: list[dict] = []
    total_loser_nodes = 0
    total_loser_mentions = 0
    total_conflicts = 0
    for canon, variants in multi.items():
        ranked = sorted(variants, key=lambda v: score(v, canon))
        winner, losers = ranked[0], ranked[1:]
        prop_conflicts: list[str] = []
        # Detect conflicts on a few common enum-ish properties.
        for prop_key in ("material_type", "uns_number", "process_type",
                         "equipment_type", "organization"):
            seen = {
                v["props"].get(prop_key)
                for v in variants
                if v["props"].get(prop_key) not in (None, "")
            }
            if len(seen) > 1:
                prop_conflicts.append(f"{prop_key}: {sorted(seen)}")
        total_conflicts += 1 if prop_conflicts else 0
        total_loser_nodes += len(losers)
        total_loser_mentions += sum(l_["mentions"] for l_ in losers)
        plan.append({
            "canonical": canon,
            "winner": winner,
            "losers": losers,
            "total_mentions": sum(v["mentions"] for v in variants),
            "prop_conflicts": prop_conflicts,
        })

    plan.sort(key=lambda g: -g["total_mentions"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lbl = args.label.lower()
    json_path = args.output_dir / f"tier1_plan_{lbl}_{stamp}.json"
    with json_path.open("w") as f:
        json.dump({
            "generated": datetime.now().isoformat(),
            "label": args.label,
            "pk": pk,
            "total_nodes": len(rows),
            "merge_groups": len(multi),
            "loser_nodes": total_loser_nodes,
            "redirected_mention_edges": total_loser_mentions,
            "property_conflict_groups": total_conflicts,
            "plan": plan,
        }, f, indent=2)

    print()
    print("=" * 60)
    print(f"Tier 1 {args.label} canonicalization — DRY RUN summary")
    print("=" * 60)
    print(f"Total {args.label} nodes:          {len(rows):>6}")
    print(f"Distinct canonical forms:          {len(groups):>6}")
    print(f"Merge groups (multi-variant):      {len(multi):>6}")
    print(f"Loser nodes to delete:             {total_loser_nodes:>6}")
    print(f"Mention edges to redirect:         {total_loser_mentions:>6}")
    print(f"Groups with property conflicts:    {total_conflicts:>6}")
    print(f"Formula-guarded nodes:             {len(formula_guarded):>6}")
    print(f"  ...merges prevented by guard:    {len(prevented_merges):>6}")
    print()
    print(f"Plan: {json_path}")
    print()
    print("Top 10 groups by total mentions:")
    for g in plan[:10]:
        variants = [f"{g['winner']['name']!r}({g['winner']['mentions']})"]
        variants += [
            f"{l_['name']!r}({l_['mentions']})" for l_ in g["losers"][:4]
        ]
        more = len(g["losers"]) - 4
        if more > 0:
            variants.append(f"+{more} more")
        print(f"  [{g['total_mentions']:>5}]  {g['canonical']!r}  "
              f"← {', '.join(variants)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
