#!/usr/bin/env python3
"""Dry-run: propose Tier 1 Material-node canonicalization merges.

Tier 1 rules are the zero-risk ones: case-fold + whitespace/hyphen collapse +
singularization of a closed set of class-noun plurals. Variants that collapse
to the same canonical string are grouped; the highest-mention variant becomes
the winner.

This script NEVER writes to Neo4j. It emits:
  - a summary to stdout
  - a human-readable plan file (one section per merge group)
  - a machine-readable JSON plan for a future --apply step

Usage:
    NEO4J_PASSWORD=... python scripts/canonicalize_materials_dryrun.py
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


# --- Tier 1 canonicalization ------------------------------------------------

_CLASS_PLURALS = {
    "steels": "steel",
    "alloys": "alloy",
    "irons": "iron",
    "bronzes": "bronze",
    "brasses": "brass",
    "coppers": "copper",
}
_PLURAL_RE = re.compile(
    r"\b(" + "|".join(_CLASS_PLURALS) + r")\b", re.IGNORECASE
)

# Chemical-formula guard: strings like "TiN", "Al2O3", "Fe3C", "NaCl" must not
# be case-folded together with the English words they collide with ("tin",
# "fen", etc.). A formula here = element-symbol chunks (uppercase optionally
# followed by one lowercase, optionally followed by digits), with BOTH an
# upper- and lower-case letter present anywhere in the string. The
# mixed-case requirement excludes all-caps designations like "AISI" or
# "SUS304" while still catching the dangerous case-fold collisions.
_CHEMICAL_FORMULA_RE = re.compile(r"^([A-Z][a-z]?\d*)+$")


def looks_like_chemical_formula(s: str) -> bool:
    if not s:
        return False
    has_upper = any(c.isupper() for c in s)
    has_lower = any(c.islower() for c in s)
    if not (has_upper and has_lower):
        return False
    return bool(_CHEMICAL_FORMULA_RE.match(s))


def canonicalize_tier1(raw: str) -> str:
    """Deterministic, zero-risk normalization.

    lowercase -> collapse [whitespace|_|-] to single space -> strip ->
    singularize closed-set class-noun plurals.

    Exception: strings that look like chemical formulas (e.g. "TiN", "Al2O3")
    are returned unchanged — they must keep their case so they don't collide
    with English words that happen to share their letters ("tin", "fen").
    """
    s = raw.strip()
    if looks_like_chemical_formula(s):
        return s
    s = s.lower()
    s = re.sub(r"[\s_\-]+", " ", s).strip()
    s = _PLURAL_RE.sub(lambda m: _CLASS_PLURALS[m.group(1).lower()], s)
    return s


# --- Main ------------------------------------------------------------------

FETCH_CYPHER = """
MATCH (m:Material)
OPTIONAL MATCH (p:Page)-[:MENTIONS_MATERIAL]->(m)
WITH m, count(DISTINCT p) AS mentions
OPTIONAL MATCH (m)-[r_out]->()
WITH m, mentions, collect(type(r_out)) AS out_types
RETURN m.name AS name,
       mentions,
       out_types,
       properties(m) AS props
"""


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "canonicalization",
        help="Where to write plan files (default: data/canonicalization/)",
    )
    parser.add_argument(
        "--min-mentions",
        type=int,
        default=0,
        help="Only group nodes with at least this many page mentions "
             "(default 0 = include everything).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("canonicalize_dryrun")

    settings = get_settings()
    if not os.environ.get(settings.neo4j.password_env):
        log.error(
            "Env var %s not set. This script only reads from Neo4j but still "
            "needs to authenticate.",
            settings.neo4j.password_env,
        )
        return 1

    svc = Neo4jService(settings.neo4j)
    await svc.connect()
    try:
        if not await svc.verify_connectivity():
            log.error("Cannot reach Neo4j at %s", settings.neo4j.uri)
            return 2

        log.info("Fetching all Material nodes with edge/mention data...")
        rows = await svc.run_query(FETCH_CYPHER)
        log.info("Fetched %d Material nodes", len(rows))
    finally:
        await svc.close()

    # Group by Tier 1 canonical form.
    groups: dict[str, list[dict]] = defaultdict(list)
    # Track nodes where the formula guard fired, and nodes whose naive
    # lowercase would have collided with a non-formula canonical form —
    # i.e., merges the guard actually prevented.
    formula_guarded: list[dict] = []
    for r in rows:
        if r["mentions"] < args.min_mentions:
            continue
        raw = r["name"] or ""
        canon = canonicalize_tier1(raw)
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

    # Prevented-merge check: formula nodes whose naive lowercase would have
    # landed in some other (non-formula) group's bucket.
    naive_buckets = {c for c in groups if not looks_like_chemical_formula(c)}
    prevented_merges = [
        f for f in formula_guarded
        if f["name"].lower().strip() in naive_buckets
    ]

    multi = {
        canon: variants
        for canon, variants in groups.items()
        if len({v["name"] for v in variants}) > 1
    }

    # Pick winner per group: highest mention count, ties broken by exact match
    # to canonical form, then lexicographic.
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
        # Detect property conflicts on merge (Tier 1 should be clean; flag anyway).
        for prop_key in ("material_type", "uns_number"):
            seen = {
                v["props"].get(prop_key)
                for v in variants
                if v["props"].get(prop_key) not in (None, "")
            }
            if len(seen) > 1:
                prop_conflicts.append(f"{prop_key}: {sorted(seen)}")
        total_conflicts += 1 if prop_conflicts else 0
        total_loser_nodes += len(losers)
        total_loser_mentions += sum(l["mentions"] for l in losers)
        plan.append({
            "canonical": canon,
            "winner": winner,
            "losers": losers,
            "total_mentions": sum(v["mentions"] for v in variants),
            "prop_conflicts": prop_conflicts,
        })

    plan.sort(key=lambda g: -g["total_mentions"])

    # --- Write outputs ----------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = args.output_dir / f"tier1_plan_{stamp}.txt"
    json_path = args.output_dir / f"tier1_plan_{stamp}.json"

    with txt_path.open("w") as f:
        f.write("Tier 1 Material canonicalization plan — DRY RUN\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total Material nodes considered: {len(rows)}\n")
        f.write(f"Merge groups (≥2 distinct variants): {len(multi)}\n")
        f.write(f"Nodes to be deleted (losers): {total_loser_nodes}\n")
        f.write(f"Page-mention edges to be redirected: {total_loser_mentions}\n")
        f.write(f"Groups with property conflicts flagged: {total_conflicts}\n")
        f.write("\n" + "=" * 72 + "\n\n")

        for g in plan:
            f.write(f"canonical: {g['canonical']!r}\n")
            f.write(f"  winner:  {g['winner']['name']!r}  "
                    f"(mentions={g['winner']['mentions']}, "
                    f"type={g['winner']['props'].get('material_type','?')})\n")
            for loser in g["losers"]:
                out_summary = ", ".join(
                    f"{t}={c}" for t, c in sorted(loser["out_type_counts"].items())
                ) or "(no outgoing edges)"
                f.write(f"  loser:   {loser['name']!r}  "
                        f"(mentions={loser['mentions']}, "
                        f"type={loser['props'].get('material_type','?')})\n")
                f.write(f"           outgoing: {out_summary}\n")
            if g["prop_conflicts"]:
                f.write(f"  ⚠ property conflicts: "
                        f"{'; '.join(g['prop_conflicts'])}\n")
            f.write("\n")

    with json_path.open("w") as f:
        json.dump({
            "generated": datetime.now().isoformat(),
            "total_material_nodes": len(rows),
            "merge_groups": len(multi),
            "loser_nodes": total_loser_nodes,
            "redirected_mention_edges": total_loser_mentions,
            "property_conflict_groups": total_conflicts,
            "plan": plan,
        }, f, indent=2)

    # --- Stdout summary ---------------------------------------------------
    print()
    print("=" * 60)
    print("Tier 1 Material canonicalization — DRY RUN summary")
    print("=" * 60)
    print(f"Total Material nodes:              {len(rows):>6}")
    print(f"Distinct canonical forms:          {len(groups):>6}")
    print(f"Merge groups (multi-variant):      {len(multi):>6}")
    print(f"Loser nodes to delete:             {total_loser_nodes:>6}")
    print(f"Mention edges to redirect:         {total_loser_mentions:>6}")
    print(f"Groups flagged for property review:{total_conflicts:>6}")
    print(f"Formula-guarded nodes:             {len(formula_guarded):>6}")
    print(f"  ...merges prevented by guard:    {len(prevented_merges):>6}")
    if prevented_merges:
        for pm in prevented_merges:
            print(f"    kept separate: {pm['name']!r} ({pm['mentions']} mentions) "
                  f"— would have collided with {pm['name'].lower().strip()!r}")
    print()
    print(f"Plan written to:")
    print(f"  {txt_path}")
    print(f"  {json_path}")
    print()
    print("Top 10 groups by total mentions:")
    for g in plan[:10]:
        variants = [f"{g['winner']['name']!r}({g['winner']['mentions']})"]
        variants += [
            f"{l['name']!r}({l['mentions']})" for l in g["losers"][:4]
        ]
        more = len(g["losers"]) - 4
        if more > 0:
            variants.append(f"+{more} more")
        print(f"  [{g['total_mentions']:>5}]  {g['canonical']!r}  "
              f"← {', '.join(variants)}")
    print()
    print("No changes were made. Review the plan, then we'll write the "
          "--apply step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
