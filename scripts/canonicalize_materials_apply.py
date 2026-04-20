#!/usr/bin/env python3
"""Apply a Tier 1 Material-node canonicalization plan.

Consumes the JSON plan produced by canonicalize_materials_dryrun.py and
executes the merges — one Neo4j transaction per group. Every merge:

  1. Redirects incoming MENTIONS_MATERIAL edges from losers to winner,
     deduplicating via MERGE (if a page already linked to the winner, the
     loser's duplicate edge is dropped).
  2. Redirects outgoing COMPATIBLE_WITH_PROCESS and GOVERNED_BY edges,
     summing `support_count` when merging onto an existing edge.
  3. Resolves the winner's scalar properties via Policy B (mention-weighted
     majority for enums like material_type; winner-first then most-mentioned
     for numerics).
  4. Appends loser names + their prior common_names (deduped) into the
     winner's common_names, so original spellings remain searchable.
  5. DETACH DELETEs the losers.
  6. Renames the winner to the canonical form if they differ.

Usage:
    # Dry-run summary (no writes):
    NEO4J_PASSWORD=... python scripts/canonicalize_materials_apply.py \
        --plan data/canonicalization/tier1_plan_YYYYMMDD_HHMMSS.json

    # Execute the plan:
    NEO4J_PASSWORD=... python scripts/canonicalize_materials_apply.py \
        --plan data/canonicalization/tier1_plan_YYYYMMDD_HHMMSS.json \
        --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_settings  # noqa: E402
from backend.services.neo4j_service import Neo4jService  # noqa: E402


# --- Policy B: mention-weighted property resolution ------------------------

ENUM_PROPS = ("material_type", "uns_number")
# NOTE: numeric properties (tensile_strength_ksi, yield_strength_ksi,
# hardness) are intentionally NOT resolved here. A separate audit found
# that these fields in the current graph contain LLM-output fragments
# stored as raw strings (JSON debris like ', yield_strength_ksi": 70 } ] }')
# rather than clean numeric values. Propagating that garbage onto the
# merged winner adds no value. The winner keeps whatever it had; losers'
# numeric values vanish with the losers on DETACH DELETE. Cleanup of the
# numeric fields is a separate re-extraction pass.


def resolve_properties(group: dict) -> tuple[dict[str, Any], list[str]]:
    """Apply Policy B to enum properties only. Returns (resolved_props,
    decision_log)."""
    variants = [group["winner"]] + group["losers"]
    resolved: dict[str, Any] = {}
    log: list[str] = []

    for prop in ENUM_PROPS:
        votes: Counter = Counter()
        for v in variants:
            val = v["props"].get(prop)
            if val in (None, ""):
                continue
            # Weight by mentions; give singletons 1 vote so they still count.
            votes[val] += max(v["mentions"], 1)
        if not votes:
            continue
        ranked = votes.most_common()
        winner_val, winner_votes = ranked[0]
        resolved[prop] = winner_val
        if len(ranked) > 1 and ranked[1][1] > 0:
            log.append(
                f"{prop}: {winner_val!r} ({winner_votes} votes) "
                f"beat {ranked[1][0]!r} ({ranked[1][1]} votes)"
            )

    return resolved, log


def build_common_names(group: dict) -> list[str]:
    """Union of loser names + losers' prior common_names + winner's prior
    common_names, minus the winner's current name. Deduped, stable order."""
    winner_name = group["winner"]["name"]
    seen: set[str] = set()
    out: list[str] = []
    existing = group["winner"]["props"].get("common_names") or []
    for name in existing:
        if name and name != winner_name and name not in seen:
            seen.add(name)
            out.append(name)
    for loser in group["losers"]:
        if loser["name"] != winner_name and loser["name"] not in seen:
            seen.add(loser["name"])
            out.append(loser["name"])
        for cn in loser["props"].get("common_names") or []:
            if cn and cn != winner_name and cn not in seen:
                seen.add(cn)
                out.append(cn)
    return out


# --- Cypher ----------------------------------------------------------------

# Incoming MENTIONS_MATERIAL: deduplicate via MERGE. This intentionally
# drops the loser edge's properties if the winner already has an edge from
# the same page — that's the dedup we want.
REDIRECT_MENTIONS = """
MATCH (w:Material {name: $winner_name})
UNWIND $loser_names AS ln
MATCH (l:Material {name: ln})
OPTIONAL MATCH (p:Page)-[r:MENTIONS_MATERIAL]->(l)
WITH w, p, r WHERE r IS NOT NULL
MERGE (p)-[:MENTIONS_MATERIAL]->(w)
DELETE r
"""

# Outgoing edges: sum support_count on merge so aggregate weight is
# preserved across duplicates.
REDIRECT_OUTGOING_TEMPLATE = """
MATCH (w:Material {{name: $winner_name}})
UNWIND $loser_names AS ln
MATCH (l:Material {{name: ln}})
OPTIONAL MATCH (l)-[r:{rel}]->(t:{tgt_label})
WITH w, t, r WHERE r IS NOT NULL
MERGE (w)-[newr:{rel}]->(t)
  ON CREATE SET
    newr.support_count = coalesce(r.support_count, 0),
    newr.context = r.context
  ON MATCH SET
    newr.support_count = coalesce(newr.support_count, 0)
                       + coalesce(r.support_count, 0)
DELETE r
"""

REDIRECT_COMPATIBLE_WITH_PROCESS = REDIRECT_OUTGOING_TEMPLATE.format(
    rel="COMPATIBLE_WITH_PROCESS", tgt_label="Process"
)
REDIRECT_GOVERNED_BY = REDIRECT_OUTGOING_TEMPLATE.format(
    rel="GOVERNED_BY", tgt_label="Standard"
)

DETACH_DELETE_LOSERS = """
UNWIND $loser_names AS ln
MATCH (l:Material {name: ln})
DETACH DELETE l
"""

RENAME_WINNER = """
MATCH (w:Material {name: $old_name})
SET w.name = $new_name
"""


def build_update_winner_cypher(resolved: dict[str, Any]) -> tuple[str, dict]:
    """Build a SET clause covering only the properties we decided on.

    common_names is always set (even if empty list, so any prior value is
    replaced by our merged-and-deduped list)."""
    sets = ["w.common_names = $common_names"]
    params: dict[str, Any] = {}
    for k, v in resolved.items():
        if v is None:
            continue
        sets.append(f"w.{k} = ${k}")
        params[k] = v
    cypher = f"""
        MATCH (w:Material {{name: $winner_name}})
        SET {", ".join(sets)}
    """
    return cypher, params


# --- Pre-flight ------------------------------------------------------------

async def preflight_check(svc: Neo4jService, plan: list[dict]) -> list[str]:
    """Verify every winner + loser named in the plan still exists in the
    graph. Returns a list of error strings (empty = plan is fresh)."""
    all_names = set()
    for g in plan:
        all_names.add(g["winner"]["name"])
        for l_ in g["losers"]:
            all_names.add(l_["name"])
    rows = await svc.run_query(
        "MATCH (m:Material) WHERE m.name IN $names RETURN m.name AS name",
        {"names": sorted(all_names)},
    )
    present = {r["name"] for r in rows}
    missing = all_names - present
    errors = []
    if missing:
        sample = sorted(missing)[:10]
        errors.append(
            f"Plan references {len(missing)} Material nodes that no longer "
            f"exist in the graph. Sample: {sample}. "
            f"Regenerate the dry-run plan."
        )
    return errors


# --- Apply one group -------------------------------------------------------

async def apply_group(svc: Neo4jService, group: dict) -> dict[str, int]:
    """Run the full merge for one group in a single transaction.

    Returns a small stats dict (edges redirected, losers deleted)."""
    winner_name = group["winner"]["name"]
    canonical = group["canonical"]
    loser_names = [l_["name"] for l_ in group["losers"]]
    resolved, _ = resolve_properties(group)
    common_names = build_common_names(group)
    update_cypher, update_params = build_update_winner_cypher(resolved)

    db = svc.settings.database

    async def _tx(tx):
        await tx.run(REDIRECT_MENTIONS,
                     winner_name=winner_name, loser_names=loser_names)
        await tx.run(REDIRECT_COMPATIBLE_WITH_PROCESS,
                     winner_name=winner_name, loser_names=loser_names)
        await tx.run(REDIRECT_GOVERNED_BY,
                     winner_name=winner_name, loser_names=loser_names)
        await tx.run(update_cypher,
                     winner_name=winner_name,
                     common_names=common_names,
                     **update_params)
        await tx.run(DETACH_DELETE_LOSERS, loser_names=loser_names)
        if winner_name != canonical:
            # Losers deleted above, so the canonical name (if it was one of
            # them) is free. Safe to rename.
            await tx.run(RENAME_WINNER,
                         old_name=winner_name, new_name=canonical)

    async with svc.driver.session(database=db) as session:
        await session.execute_write(_tx)

    return {
        "losers_deleted": len(loser_names),
        "mentions_redirected": sum(l_["mentions"] for l_ in group["losers"]),
    }


# --- Main ------------------------------------------------------------------

async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True,
                        help="JSON plan produced by the dry-run script.")
    parser.add_argument("--apply", action="store_true",
                        help="Actually execute the plan. Without this flag, "
                             "the script only prints what it would do.")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Abort on the first failed group (default: "
                             "log and continue).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("canonicalize_apply")

    plan_doc = json.load(args.plan.open())
    plan = plan_doc["plan"]
    log.info("Loaded plan: %d groups, %d loser nodes, %d edges to redirect",
             plan_doc["merge_groups"],
             plan_doc["loser_nodes"],
             plan_doc["redirected_mention_edges"])

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

        # Pre-apply stats snapshot.
        before = await svc.run_query(
            "MATCH (m:Material) RETURN count(m) AS n"
        )
        before_materials = before[0]["n"] if before else 0
        before_edges = await svc.run_query(
            "MATCH (:Page)-[r:MENTIONS_MATERIAL]->(:Material) "
            "RETURN count(r) AS n"
        )
        before_edge_count = before_edges[0]["n"] if before_edges else 0
        log.info("BEFORE: %d Material nodes, %d MENTIONS_MATERIAL edges",
                 before_materials, before_edge_count)

        errors = await preflight_check(svc, plan)
        if errors:
            for e in errors:
                log.error(e)
            return 3

        if not args.apply:
            print()
            print("=" * 60)
            print("DRY-RUN — add --apply to execute")
            print("=" * 60)
            print(f"Plan file: {args.plan}")
            print(f"Groups to merge:           {len(plan):>6}")
            print(f"Loser nodes to delete:     {plan_doc['loser_nodes']:>6}")
            print(f"Mention edges to redirect: "
                  f"{plan_doc['redirected_mention_edges']:>6}")
            print(f"Current Material nodes:    {before_materials:>6}")
            print(f"Expected Material nodes after apply: "
                  f"{before_materials - plan_doc['loser_nodes']:>6}")
            print()
            # Sample a few groups with their resolved properties so the
            # user can eyeball Policy B outcomes before committing.
            print("Sample resolved properties (top 10 by total mentions):")
            for g in plan[:10]:
                resolved, decisions = resolve_properties(g)
                rename = ("" if g["winner"]["name"] == g["canonical"]
                          else f"  RENAME {g['winner']['name']!r}→{g['canonical']!r}")
                print(f"  [{g['total_mentions']:>5}]  canon={g['canonical']!r}"
                      f"{rename}")
                for k, v in resolved.items():
                    if v is not None:
                        print(f"           {k}={v!r}")
                for d in decisions:
                    print(f"           decision: {d}")
            return 0

        # --- APPLY ----------------------------------------------------
        log.info("Applying plan — %d groups, one transaction per group",
                 len(plan))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_log_path = args.plan.parent / f"apply_errors_{stamp}.log"
        decisions_log_path = args.plan.parent / f"apply_decisions_{stamp}.log"

        total_losers = 0
        total_edges = 0
        failures: list[tuple[str, str]] = []
        decisions: list[str] = []

        for idx, g in enumerate(plan, 1):
            try:
                stats = await apply_group(svc, g)
                total_losers += stats["losers_deleted"]
                total_edges += stats["mentions_redirected"]
                _, decision_log = resolve_properties(g)
                for d in decision_log:
                    decisions.append(f"{g['canonical']}: {d}")
                if idx % 200 == 0:
                    log.info("Progress: %d/%d groups (%d losers, %d edges)",
                             idx, len(plan), total_losers, total_edges)
            except Exception as exc:  # noqa: BLE001
                failures.append((g["canonical"], str(exc)))
                log.warning("Failed group %r: %s", g["canonical"], exc)
                if args.stop_on_error:
                    log.error("Aborting due to --stop-on-error")
                    break

        if failures:
            with error_log_path.open("w") as f:
                for canon, err in failures:
                    f.write(f"{canon}\t{err}\n")
            log.warning("%d groups failed — see %s",
                        len(failures), error_log_path)

        if decisions:
            with decisions_log_path.open("w") as f:
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Plan: {args.plan}\n\n")
                for d in decisions:
                    f.write(d + "\n")
            log.info("Policy B decision log: %s", decisions_log_path)

        # Post-apply stats.
        after = await svc.run_query(
            "MATCH (m:Material) RETURN count(m) AS n"
        )
        after_materials = after[0]["n"] if after else 0
        after_edges = await svc.run_query(
            "MATCH (:Page)-[r:MENTIONS_MATERIAL]->(:Material) "
            "RETURN count(r) AS n"
        )
        after_edge_count = after_edges[0]["n"] if after_edges else 0

        print()
        print("=" * 60)
        print("APPLY COMPLETE")
        print("=" * 60)
        print(f"Groups processed:              {len(plan):>6}")
        print(f"Groups succeeded:              {len(plan) - len(failures):>6}")
        print(f"Groups failed:                 {len(failures):>6}")
        print(f"Losers deleted:                {total_losers:>6}")
        print(f"Material nodes: {before_materials} → {after_materials} "
              f"(Δ={after_materials - before_materials:+d})")
        print(f"MENTIONS_MATERIAL edges: {before_edge_count} → "
              f"{after_edge_count} "
              f"(Δ={after_edge_count - before_edge_count:+d})")
        print()
        print("Spot-check queries:")
        for name in ("4140", "AISI 4140", "steel", "stainless steel", "TiN"):
            rows = await svc.run_query(
                "MATCH (m:Material {name: $n}) "
                "OPTIONAL MATCH (p:Page)-[:MENTIONS_MATERIAL]->(m) "
                "RETURN m.name AS name, m.material_type AS type, "
                "       size(coalesce(m.common_names, [])) AS alias_count, "
                "       count(DISTINCT p) AS mentions",
                {"n": name},
            )
            if rows and rows[0]["name"]:
                r = rows[0]
                print(f"  {r['name']!r}: mentions={r['mentions']}, "
                      f"type={r['type']}, aliases={r['alias_count']}")
            else:
                print(f"  {name!r}: not present")
        return 0 if not failures else 4
    finally:
        await svc.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
