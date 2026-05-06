#!/usr/bin/env python3
"""Apply a Tier 2 fuzzy entity-deduplication plan.

This is a thin wrapper around canonicalize_entity_apply.py — the merge
mechanics (edge redirection, property resolution, common_names, delete)
are identical for Tier 1 and Tier 2 plans.

The wrapper adds:
  - Tier-2-specific summary output (similarity stats, match types)
  - A reminder of what Tier 2 merges look like before applying

Usage:
    NEO4J_PASSWORD=... python scripts/dedup_entities_apply.py \\
        --plan data/canonicalization/tier2_plan_material_XXXX.json [--apply]

Alternatively, the existing apply script works directly:
    NEO4J_PASSWORD=... python scripts/canonicalize_entity_apply.py \\
        --plan data/canonicalization/tier2_plan_material_XXXX.json [--apply]
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_settings  # noqa: E402
from backend.services.neo4j_service import Neo4jService  # noqa: E402

# Import all the apply machinery from the existing script
from scripts.canonicalize_entity_apply import (  # noqa: E402
    apply_group,
    discover_rel_types,
    preflight_check,
)


def _tier2_preview(plan: list[dict], label: str) -> None:
    """Print a Tier 2 specific preview with similarity scores."""
    print()
    print("=" * 60)
    print(f"Tier 2 {label} fuzzy deduplication — merge preview")
    print("=" * 60)

    # Aggregate stats
    match_types: Counter = Counter()
    all_sims: list[float] = []
    for g in plan:
        for loser in g["losers"]:
            mt = loser.get("match_type", "unknown")
            match_types[mt] += 1
            sim = loser.get("similarity", 0.0)
            all_sims.append(sim)

    total_losers = sum(len(g["losers"]) for g in plan)
    print(f"Merge groups:          {len(plan):>6}")
    print(f"Loser nodes to merge:  {total_losers:>6}")
    print()

    if all_sims:
        print("Similarity distribution:")
        print(f"  Min:    {min(all_sims):.3f}")
        print(f"  Median: {sorted(all_sims)[len(all_sims)//2]:.3f}")
        print(f"  Max:    {max(all_sims):.3f}")
        print()

    if match_types:
        print("Match types:")
        for mt, count in match_types.most_common():
            print(f"  {mt:25s} {count:>5}")
        print()

    show = min(10, len(plan))
    print(f"Top {show} merge groups by total mentions:")
    for g in plan[:show]:
        w = g["winner"]
        losers_str = ", ".join(
            f"{l['name']!r} (sim={l.get('similarity', 0):.2f})"
            for l in g["losers"][:3]
        )
        more = len(g["losers"]) - 3
        if more > 0:
            losers_str += f", +{more} more"
        print(f"  {w['name']!r} ({w['mentions']} mentions) <- {losers_str}")
    print()

    # Flag groups with property conflicts
    conflict_groups = [g for g in plan if g.get("prop_conflicts")]
    if conflict_groups:
        print(f"WARNING: {len(conflict_groups)} groups have property conflicts:")
        for g in conflict_groups[:5]:
            print(f"  {g['winner']['name']!r}: {g['prop_conflicts']}")
        print()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True,
                        help="Path to tier2_plan_*.json from dedup_entities_dryrun.py")
    parser.add_argument("--apply", action="store_true",
                        help="Actually execute the merges (without this, dry-run only)")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Stop on first failed merge group")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("dedup_apply")

    plan_doc = json.load(args.plan.open())
    plan = plan_doc["plan"]
    label = plan_doc["label"]
    pk = plan_doc["pk"]
    tier = plan_doc.get("tier", 2)
    threshold = plan_doc.get("threshold", "unknown")

    log.info(
        "Plan: label=%s pk=%s tier=%d threshold=%s groups=%d losers=%d",
        label, pk, tier, threshold,
        plan_doc["merge_groups"], plan_doc["loser_nodes"],
    )

    if tier != 2:
        log.warning(
            "This plan has tier=%d. Use canonicalize_entity_apply.py for "
            "Tier 1 plans. Proceeding anyway (the format is compatible).",
            tier,
        )

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

        # Print tier2 preview
        _tier2_preview(plan, label)

        if not args.apply:
            print(
                f"DRY-RUN {label} — add --apply to execute\n"
                f"  Groups:         {len(plan)}\n"
                f"  Losers:         {plan_doc['loser_nodes']}\n"
                f"  Expected after: {before_n - plan_doc['loser_nodes']}\n"
            )
            return 0

        log.info("Applying Tier 2 %s plan — %d groups", label, len(plan))
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
                if idx % 100 == 0:
                    log.info("Progress: %d/%d groups (%d losers merged)",
                             idx, len(plan), total_losers)
            except Exception as exc:  # noqa: BLE001
                failures.append((g.get("canonical", "?"), str(exc)))
                log.warning("Failed group %r: %s",
                            g.get("canonical", "?"), exc)
                if args.stop_on_error:
                    break

        after = await svc.run_query(
            f"MATCH (m:{label}) RETURN count(m) AS n"
        )
        after_n = after[0]["n"] if after else 0

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if failures:
            err_path = (args.plan.parent
                        / f"tier2_apply_errors_{label.lower()}_{stamp}.log")
            with err_path.open("w") as f:
                for c, e in failures:
                    f.write(f"{c}\t{e}\n")
            log.warning("%d failures logged to %s", len(failures), err_path)

        print()
        print(f"TIER 2 APPLY {label} COMPLETE")
        print(f"  Groups processed: {len(plan)}")
        print(f"  Succeeded:        {len(plan) - len(failures)}")
        print(f"  Failed:           {len(failures)}")
        print(f"  Losers deleted:   {total_losers}")
        print(f"  {label} nodes: {before_n} -> {after_n} "
              f"(delta={after_n - before_n:+d})")
        return 0 if not failures else 4
    finally:
        await svc.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
