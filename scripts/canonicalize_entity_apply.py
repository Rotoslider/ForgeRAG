#!/usr/bin/env python3
"""Apply a Tier 1 entity-canonicalization plan for any entity label.

Same mechanics as canonicalize_materials_apply.py, but label-agnostic:

  - Discovers the set of incoming/outgoing relationship types touching the
    label at runtime, so it works without hardcoded edge knowledge.
  - For each merge group, runs one transaction that:
      1. Redirects each incoming edge type (MERGE-deduped onto the winner).
      2. Redirects each outgoing edge type, summing `support_count` when
         an edge already exists.
      3. Updates the winner's enum props via Policy B
         (mention-weighted majority).
      4. Appends loser names to the winner's common_names (deduped).
      5. DETACH DELETEs the losers.
    The winner's own name is NOT renamed — we keep whatever casing the
    most-mentioned variant uses. Grouping was done via the canonical form,
    but the stored name stays as the winner's (preserves "ASME..." etc.).

Usage:
    NEO4J_PASSWORD=... python scripts/canonicalize_entity_apply.py \
        --plan data/canonicalization/tier1_plan_equipment_YYYYMMDD.json \
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


# Enum-like properties that make sense to resolve via Policy B if both
# variants disagree. Any property name not in this list is left as the
# winner's value (no merge policy applied).
ENUM_PROPS_BY_LABEL: dict[str, tuple[str, ...]] = {
    "Material": ("material_type", "uns_number"),
    "Equipment": ("equipment_type",),
    "Process": ("process_type",),
    "Standard": ("organization",),
}


def resolve_properties(group: dict, label: str) -> tuple[dict[str, Any], list[str]]:
    variants = [group["winner"]] + group["losers"]
    resolved: dict[str, Any] = {}
    log: list[str] = []
    for prop in ENUM_PROPS_BY_LABEL.get(label, ()):
        votes: Counter = Counter()
        for v in variants:
            val = v["props"].get(prop)
            if val in (None, ""):
                continue
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


def build_common_names(group: dict, pk: str) -> list[str]:
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


async def discover_rel_types(svc: Neo4jService, label: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Find every (incoming_rel_type) and every (outgoing_rel_type, target_label)
    actually present in the graph for this label."""
    incoming_rows = await svc.run_query(
        f"MATCH ()-[r]->(m:{label}) RETURN DISTINCT type(r) AS t"
    )
    incoming = [r["t"] for r in incoming_rows]
    outgoing_rows = await svc.run_query(
        f"MATCH (m:{label})-[r]->(t) "
        "RETURN DISTINCT type(r) AS rt, labels(t)[0] AS tl"
    )
    outgoing = [(r["rt"], r["tl"]) for r in outgoing_rows]
    return incoming, outgoing


def redirect_incoming_cypher(label: str, pk: str, rel: str) -> str:
    # Dedup via MERGE — if source already has an edge to winner, drop the
    # duplicate from the loser.
    return f"""
        MATCH (w:{label} {{{pk}: $winner_name}})
        UNWIND $loser_names AS ln
        MATCH (l:{label} {{{pk}: ln}})
        OPTIONAL MATCH (src)-[r:{rel}]->(l)
        WITH w, src, r WHERE r IS NOT NULL
        MERGE (src)-[:{rel}]->(w)
        DELETE r
    """


def redirect_outgoing_cypher(label: str, pk: str, rel: str, tgt_label: str) -> str:
    return f"""
        MATCH (w:{label} {{{pk}: $winner_name}})
        UNWIND $loser_names AS ln
        MATCH (l:{label} {{{pk}: ln}})
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


def build_update_winner_cypher(label: str, pk: str,
                               resolved: dict[str, Any]) -> tuple[str, dict]:
    sets = ["w.common_names = $common_names"]
    params: dict[str, Any] = {}
    for k, v in resolved.items():
        if v is None:
            continue
        sets.append(f"w.{k} = ${k}")
        params[k] = v
    cypher = f"""
        MATCH (w:{label} {{{pk}: $winner_name}})
        SET {", ".join(sets)}
    """
    return cypher, params


def detach_delete_cypher(label: str, pk: str) -> str:
    return f"""
        UNWIND $loser_names AS ln
        MATCH (l:{label} {{{pk}: ln}})
        DETACH DELETE l
    """


async def preflight_check(svc: Neo4jService, label: str, pk: str,
                          plan: list[dict]) -> list[str]:
    all_names = set()
    for g in plan:
        all_names.add(g["winner"]["name"])
        for l_ in g["losers"]:
            all_names.add(l_["name"])
    rows = await svc.run_query(
        f"MATCH (m:{label}) WHERE m.{pk} IN $names RETURN m.{pk} AS name",
        {"names": sorted(all_names)},
    )
    present = {r["name"] for r in rows}
    missing = all_names - present
    errors = []
    if missing:
        sample = sorted(missing)[:10]
        errors.append(
            f"Plan references {len(missing)} {label} nodes that no longer "
            f"exist. Sample: {sample}. Regenerate the dry-run plan."
        )
    return errors


async def apply_group(svc: Neo4jService, label: str, pk: str,
                      incoming_rels: list[str],
                      outgoing_rels: list[tuple[str, str]],
                      group: dict) -> dict[str, int]:
    winner_name = group["winner"]["name"]
    loser_names = [l_["name"] for l_ in group["losers"]]
    resolved, _ = resolve_properties(group, label)
    common_names = build_common_names(group, pk)
    update_cypher, update_params = build_update_winner_cypher(
        label, pk, resolved
    )

    db = svc.settings.database

    async def _tx(tx):
        for rel in incoming_rels:
            await tx.run(
                redirect_incoming_cypher(label, pk, rel),
                winner_name=winner_name, loser_names=loser_names,
            )
        for rel, tgt in outgoing_rels:
            await tx.run(
                redirect_outgoing_cypher(label, pk, rel, tgt),
                winner_name=winner_name, loser_names=loser_names,
            )
        await tx.run(update_cypher,
                     winner_name=winner_name,
                     common_names=common_names,
                     **update_params)
        await tx.run(detach_delete_cypher(label, pk),
                     loser_names=loser_names)

    async with svc.driver.session(database=db) as session:
        await session.execute_write(_tx)

    return {
        "losers_deleted": len(loser_names),
        "mentions_redirected": sum(l_["mentions"] for l_ in group["losers"]),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("canonicalize_apply")

    plan_doc = json.load(args.plan.open())
    plan = plan_doc["plan"]
    label = plan_doc["label"]
    pk = plan_doc["pk"]
    log.info("Plan: label=%s pk=%s groups=%d losers=%d",
             label, pk, plan_doc["merge_groups"], plan_doc["loser_nodes"])

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

        incoming_rels, outgoing_rels = await discover_rel_types(svc, label)
        log.info("Discovered edge types — incoming: %s, outgoing: %s",
                 incoming_rels, outgoing_rels)

        before = await svc.run_query(
            f"MATCH (m:{label}) RETURN count(m) AS n"
        )
        before_n = before[0]["n"] if before else 0
        log.info("BEFORE: %d %s nodes", before_n, label)

        errors = await preflight_check(svc, label, pk, plan)
        if errors:
            for e in errors:
                log.error(e)
            return 3

        if not args.apply:
            print()
            print(f"DRY-RUN {label} — add --apply to execute")
            print(f"  Groups: {len(plan)}")
            print(f"  Losers: {plan_doc['loser_nodes']}")
            print(f"  Expected after: {before_n - plan_doc['loser_nodes']}")
            return 0

        log.info("Applying %s plan — %d groups", label, len(plan))
        total_losers = 0
        total_edges = 0
        failures: list[tuple[str, str]] = []

        for idx, g in enumerate(plan, 1):
            try:
                stats = await apply_group(
                    svc, label, pk, incoming_rels, outgoing_rels, g
                )
                total_losers += stats["losers_deleted"]
                total_edges += stats["mentions_redirected"]
                if idx % 500 == 0:
                    log.info("Progress: %d/%d groups (%d losers)",
                             idx, len(plan), total_losers)
            except Exception as exc:  # noqa: BLE001
                failures.append((g["canonical"], str(exc)))
                log.warning("Failed group %r: %s", g["canonical"], exc)
                if args.stop_on_error:
                    break

        after = await svc.run_query(
            f"MATCH (m:{label}) RETURN count(m) AS n"
        )
        after_n = after[0]["n"] if after else 0

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if failures:
            err_path = args.plan.parent / f"apply_errors_{label.lower()}_{stamp}.log"
            with err_path.open("w") as f:
                for c, e in failures:
                    f.write(f"{c}\t{e}\n")
            log.warning("%d failures logged to %s", len(failures), err_path)

        print()
        print(f"APPLY {label} COMPLETE")
        print(f"  Groups processed: {len(plan)}")
        print(f"  Succeeded:        {len(plan) - len(failures)}")
        print(f"  Failed:           {len(failures)}")
        print(f"  Losers deleted:   {total_losers}")
        print(f"  {label} nodes: {before_n} → {after_n} "
              f"(Δ={after_n - before_n:+d})")
        return 0 if not failures else 4
    finally:
        await svc.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
