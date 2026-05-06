#!/usr/bin/env python3
"""Dry-run: propose Tier 2 fuzzy deduplication merges for any entity label.

Unlike Tier 1 canonicalization (which handles case/whitespace/plural
normalization), this uses SequenceMatcher similarity and containment checks
to find entities that refer to the same thing but have different names
(e.g. "Inconel 625" vs "IN625", "ASTM A36" vs "A36 steel").

Uses a blocking strategy (shared prefixes) to avoid O(n^2) all-pairs
comparison, then applies safety guards to avoid false merges.

The output plan is compatible with canonicalize_entity_apply.py, so the
same apply script can be used for both Tier 1 and Tier 2 plans.

Usage:
    NEO4J_PASSWORD=... python scripts/dedup_entities_dryrun.py \\
        --label Material [--threshold 0.90] [--min-mentions 2]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_settings  # noqa: E402
from backend.services.neo4j_service import Neo4jService  # noqa: E402

# --- Label configs (reused from tier 1) ------------------------------------

LABEL_CONFIG: dict[str, dict] = {
    "Material": {
        "pk": "name",
        "mention_rel": "MENTIONS_MATERIAL",
        "mention_source": "Page",
        "conflict_props": ("material_type", "uns_number"),
    },
    "Equipment": {
        "pk": "name",
        "mention_rel": "MENTIONS_EQUIPMENT",
        "mention_source": "Page",
        "conflict_props": ("equipment_type",),
    },
    "Process": {
        "pk": "name",
        "mention_rel": "DESCRIBES_PROCESS",
        "mention_source": "Page",
        "conflict_props": ("process_type",),
    },
    "Standard": {
        "pk": "code",
        "mention_rel": "REFERENCES_STANDARD",
        "mention_source": "Page",
        "conflict_props": ("organization",),
    },
}

# --- Normalization (same as EntityMatcher._normalize) ----------------------

_STRIP_CHARS_RE = re.compile(r"[®©™°\-–—\s]+")


def _normalize(name: str) -> str:
    """Normalize for comparison: strip special chars, collapse whitespace,
    lowercase.  Matches backend/services/entity_matcher.py:_normalize."""
    return _STRIP_CHARS_RE.sub("", name).lower()


# --- Standard-specific safety: extract numeric portion ---------------------

_STANDARD_NUM_RE = re.compile(r"(\d[\d./-]+)")
_STANDARD_ORG_RE = re.compile(
    r"^(ASTM|ASME|AWS|ISO|EN|DIN|JIS|API|NACE|SAE|NFPA|AISI|UNS|AMS|MIL)",
    re.IGNORECASE,
)


def _extract_standard_number(name: str) -> str | None:
    """Return the numeric portion of a standard code, or None."""
    m = _STANDARD_NUM_RE.search(name)
    return m.group(1) if m else None


def _extract_standard_org(name: str) -> str | None:
    """Return the organization prefix of a standard code, or None."""
    m = _STANDARD_ORG_RE.match(name.strip())
    return m.group(1).upper() if m else None


# --- Blocking strategy -----------------------------------------------------

def _blocking_keys(normalized: str, original: str) -> set[str]:
    """Generate multiple blocking keys for a normalized entity name.

    This is how we avoid O(n^2): entities only compare against others
    that share at least one blocking key.

    Keys generated:
      - First 3 characters of normalized form
      - First 4 characters of normalized form
      - First word (whitespace-split) of the ORIGINAL lowered name
      - Sorted character bag of first 4 chars (anagram-tolerant)
    """
    keys: set[str] = set()
    if len(normalized) >= 3:
        keys.add(f"p3:{normalized[:3]}")
    if len(normalized) >= 4:
        keys.add(f"p4:{normalized[:4]}")

    # First word of original (lowered, stripped of special chars)
    words = original.lower().split()
    if words:
        w0 = re.sub(r"[^a-z0-9]", "", words[0])
        if len(w0) >= 3:
            keys.add(f"w0:{w0}")

    # Sorted char bag of first 4 normalized chars (catches reorderings)
    if len(normalized) >= 4:
        bag = "".join(sorted(normalized[:4]))
        keys.add(f"bg:{bag}")

    return keys


# --- Safety guards ----------------------------------------------------------

def _check_safety(
    a: dict,
    b: dict,
    label: str,
    conflict_props: tuple[str, ...],
) -> str | None:
    """Return a reason string if merging a and b should be blocked,
    or None if the merge is safe.

    Safety rules:
      1. Both entities have >50 mentions => both well-established, likely distinct
      2. Conflicting enum-ish properties (material_type, etc.)
      3. For Standards: different numeric portions or different organizations
      4. Different numeric designations (Alloy 625 vs Alloy 718, Type 304 vs Type 316)
    """
    # Rule 1: both well-established
    if a["mentions"] > 50 and b["mentions"] > 50:
        return (
            f"both well-established ({a['name']!r}={a['mentions']} mentions, "
            f"{b['name']!r}={b['mentions']} mentions)"
        )

    # Rule 2: conflicting properties
    for prop in conflict_props:
        va = a["props"].get(prop)
        vb = b["props"].get(prop)
        if va and vb and va != vb:
            return f"conflicting {prop}: {va!r} vs {vb!r}"

    # Rule 3: Standard-specific checks
    if label == "Standard":
        num_a = _extract_standard_number(a["name"])
        num_b = _extract_standard_number(b["name"])
        if num_a and num_b and num_a != num_b:
            return f"different standard numbers: {num_a!r} vs {num_b!r}"

        org_a = _extract_standard_org(a["name"])
        org_b = _extract_standard_org(b["name"])
        if org_a and org_b and org_a != org_b:
            return f"different organizations: {org_a!r} vs {org_b!r}"

    # Rule 4: Different numeric designations — catches "Alloy 625" vs "Alloy 718",
    # "Type 304" vs "Type 316", "Table 1" vs "Table 41", "Section I" vs "Section IX".
    # For Standards, also match single-digit numbers and roman numerals since
    # "Table 1", "Vol 2", "Section IX" are common patterns.
    if label == "Standard":
        # Match all numbers (including single digit) AND roman numerals
        _rom = re.compile(r"\b(I{1,3}|IV|V|VI{0,3}|IX|X{1,3}|XI{0,3})\b")
        nums_a = set(re.findall(r"\d+", a["name"])) | set(_rom.findall(a["name"]))
        nums_b = set(re.findall(r"\d+", b["name"])) | set(_rom.findall(b["name"]))
    else:
        nums_a = set(re.findall(r"\d{2,}", a["name"]))
        nums_b = set(re.findall(r"\d{2,}", b["name"]))
    if nums_a and nums_b and nums_a != nums_b:
        return f"different numeric designations: {sorted(nums_a)} vs {sorted(nums_b)}"

    return None


# --- Core matching ----------------------------------------------------------

def _similarity(norm_a: str, norm_b: str) -> tuple[float, str]:
    """Return (score, match_type) for two normalized entity names.

    Tries containment first (cheaper), then SequenceMatcher.
    """
    if not norm_a or not norm_b:
        return 0.0, "none"

    # Exact normalized match (shouldn't happen if tier 1 already ran,
    # but handle it)
    if norm_a == norm_b:
        return 1.0, "exact_normalized"

    # Containment check
    shorter, longer = (norm_a, norm_b) if len(norm_a) <= len(norm_b) else (norm_b, norm_a)
    if shorter in longer:
        score = len(shorter) / len(longer)
        if score >= 0.5:  # at least half the length
            return score, "containment"

    # SequenceMatcher
    score = SequenceMatcher(None, norm_a, norm_b).ratio()
    return score, "sequence_matcher"


def build_merge_groups(
    entities: list[dict],
    label: str,
    threshold: float,
    conflict_props: tuple[str, ...],
) -> tuple[list[dict], int, int]:
    """Find fuzzy-duplicate groups among entities.

    Returns: (plan, comparisons_made, safety_blocked_count)
    """
    t0 = time.monotonic()

    # Build normalized forms and blocking keys
    entries: list[dict] = []
    for e in entities:
        norm = _normalize(e["name"])
        if len(norm) < 3:
            continue  # skip very short entities
        e["_norm"] = norm
        e["_blocks"] = _blocking_keys(norm, e["name"])
        entries.append(e)

    # Build inverted index: blocking_key -> list of entry indices
    block_index: dict[str, list[int]] = defaultdict(list)
    for idx, e in enumerate(entries):
        for bk in e["_blocks"]:
            block_index[bk].append(idx)

    # Find candidate pairs (share at least one block)
    candidate_pairs: set[tuple[int, int]] = set()
    for bk, members in block_index.items():
        if len(members) > 500:
            # Very large blocks (common prefixes like "a36", "steel") —
            # skip to avoid blowup. These are handled by tier 1 anyway.
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a_idx, b_idx = members[i], members[j]
                if a_idx < b_idx:
                    candidate_pairs.add((a_idx, b_idx))
                else:
                    candidate_pairs.add((b_idx, a_idx))

    logging.getLogger("dedup_dryrun").info(
        "Blocking produced %d candidate pairs from %d entities "
        "(%d blocking keys, %.1fs)",
        len(candidate_pairs), len(entries), len(block_index),
        time.monotonic() - t0,
    )

    # Compare all candidate pairs — direct matching only, NO transitive
    # closure. Each loser must individually match the winner above threshold.
    # This prevents chains like A→B→C where A and C are unrelated
    # (e.g., "Alloy 625"→"Alloy 625 wrought"→"Alloy 718 wrought").
    comparisons = 0
    safety_blocked = 0

    # edges[a_idx] -> list of (b_idx, score, match_type)
    edges: dict[int, list[tuple[int, float, str]]] = defaultdict(list)
    blocked_pairs: list[tuple[dict, dict, str]] = []

    for a_idx, b_idx in candidate_pairs:
        a, b = entries[a_idx], entries[b_idx]
        comparisons += 1

        score, match_type = _similarity(a["_norm"], b["_norm"])
        if score < threshold:
            continue

        # Safety check
        reason = _check_safety(a, b, label, conflict_props)
        if reason:
            safety_blocked += 1
            blocked_pairs.append((a, b, reason))
            continue

        edges[a_idx].append((b_idx, score, match_type))
        edges[b_idx].append((a_idx, score, match_type))

    # Build groups: star topology around the highest-mention entity.
    # Each entity can only be in one group (first-claimed wins).
    claimed: set[int] = set()
    plan: list[dict] = []

    # Sort potential winners by mention count descending
    candidates_with_edges = sorted(
        edges.keys(), key=lambda i: -entries[i]["mentions"]
    )

    for winner_idx in candidates_with_edges:
        if winner_idx in claimed:
            continue

        # Collect all unclaimed entities that directly match this winner
        losers_info: list[tuple[int, float, str]] = []
        for other_idx, score, match_type in edges[winner_idx]:
            if other_idx in claimed:
                continue
            losers_info.append((other_idx, score, match_type))

        if not losers_info:
            continue

        claimed.add(winner_idx)
        winner = entries[winner_idx]
        losers_with_meta = []
        for other_idx, score, match_type in losers_info:
            claimed.add(other_idx)
            losers_with_meta.append((entries[other_idx], score, match_type))

        # Compute similarity of each loser to the winner
        losers_with_sim = []
        all_group = [winner] + [lm[0] for lm in losers_with_meta]
        for loser_entry, score, match_type in losers_with_meta:
            losers_with_sim.append({
                "name": loser_entry["name"],
                "mentions": loser_entry["mentions"],
                "props": loser_entry["props"],
                "similarity": round(score, 3),
                "match_type": match_type,
            })

        # Detect property conflicts within the group
        prop_conflicts: list[str] = []
        for prop_key in conflict_props:
            seen = {
                e["props"].get(prop_key)
                for e in all_group
                if e["props"].get(prop_key) not in (None, "")
            }
            if len(seen) > 1:
                prop_conflicts.append(f"{prop_key}: {sorted(seen)}")

        # Determine the dominant match type
        match_types = {mt for _, _, mt in losers_with_meta}
        dominant_match = (
            "containment" if "containment" in match_types
            else "sequence_matcher" if "sequence_matcher" in match_types
            else "exact_normalized"
        )

        plan.append({
            "canonical": winner["_norm"],
            "winner": {
                "name": winner["name"],
                "mentions": winner["mentions"],
                "props": winner["props"],
            },
            "losers": losers_with_sim,
            "total_mentions": sum(e["mentions"] for e in all_group),
            "match_type": dominant_match,
            "prop_conflicts": prop_conflicts,
        })

    plan.sort(key=lambda g: -g["total_mentions"])

    elapsed = time.monotonic() - t0
    log = logging.getLogger("dedup_dryrun")
    log.info(
        "Matching complete: %d comparisons, %d merge groups, "
        "%d safety-blocked, %.1fs",
        comparisons, len(plan), safety_blocked, elapsed,
    )
    if blocked_pairs:
        log.info("Top 5 safety-blocked pairs:")
        for a, b, reason in blocked_pairs[:5]:
            log.info("  %r <-> %r: %s", a["name"], b["name"], reason)

    return plan, comparisons, safety_blocked


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label", required=True, choices=list(LABEL_CONFIG.keys()),
        help="Entity label to deduplicate",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.90,
        help="Minimum similarity score for a merge (default: 0.85)",
    )
    parser.add_argument(
        "--min-mentions", type=int, default=2,
        help="Skip entities with fewer than this many mentions (default: 2)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "data" / "canonicalization",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("dedup_dryrun")

    cfg = LABEL_CONFIG[args.label]
    pk = cfg["pk"]
    mention_rel = cfg["mention_rel"]
    mention_src = cfg["mention_source"]
    conflict_props = cfg["conflict_props"]

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
            RETURN m.{pk} AS name,
                   mentions,
                   properties(m) AS props
        """
        log.info("Fetching all %s nodes...", args.label)
        rows = await svc.run_query(fetch_cypher)
        log.info("Fetched %d %s nodes", len(rows), args.label)
    finally:
        await svc.close()

    # Filter by min-mentions and short names
    entities: list[dict] = []
    for r in rows:
        name = r["name"]
        if not name or len(name.strip()) < 3:
            continue
        if r["mentions"] < args.min_mentions:
            continue
        props = r["props"] or {}
        entities.append({
            "name": name,
            "mentions": r["mentions"],
            "props": props,
        })

    log.info(
        "After filtering: %d entities (min_mentions=%d)",
        len(entities), args.min_mentions,
    )

    plan, comparisons, safety_blocked = build_merge_groups(
        entities, args.label, args.threshold, conflict_props,
    )

    total_loser_nodes = sum(len(g["losers"]) for g in plan)

    # Write plan JSON
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lbl = args.label.lower()
    json_path = args.output_dir / f"tier2_plan_{lbl}_{stamp}.json"
    with json_path.open("w") as f:
        json.dump({
            "generated": datetime.now().isoformat(),
            "label": args.label,
            "pk": pk,
            "tier": 2,
            "threshold": args.threshold,
            "total_nodes": len(rows),
            "merge_groups": len(plan),
            "loser_nodes": total_loser_nodes,
            "safety_blocked": safety_blocked,
            "comparisons": comparisons,
            "plan": plan,
        }, f, indent=2)

    # Print summary
    print()
    print("=" * 60)
    print(f"Tier 2 {args.label} fuzzy deduplication — DRY RUN summary")
    print("=" * 60)
    print(f"Total {args.label} nodes fetched:    {len(rows):>6}")
    print(f"Entities after filtering:          {len(entities):>6}")
    print(f"Candidate pairs (via blocking):    {comparisons:>6}")
    print(f"Merge groups found:                {len(plan):>6}")
    print(f"Loser nodes to merge:              {total_loser_nodes:>6}")
    print(f"Safety-blocked merges:             {safety_blocked:>6}")
    print(f"Threshold:                         {args.threshold:>6.2f}")
    print()
    print(f"Plan: {json_path}")
    print()

    if plan:
        show = min(20, len(plan))
        print(f"Top {show} proposed merges:")
        for g in plan[:show]:
            winner = g["winner"]
            losers_str = ", ".join(
                f"{l['name']!r}({l['mentions']}, sim={l['similarity']:.2f})"
                for l in g["losers"][:4]
            )
            more = len(g["losers"]) - 4
            if more > 0:
                losers_str += f", +{more} more"
            print(
                f"  [{g['total_mentions']:>5}]  {winner['name']!r}"
                f"({winner['mentions']}) <- {losers_str}"
            )
    else:
        print("No merge groups found at this threshold.")

    print()
    print(
        "To apply, run:\n"
        f"  NEO4J_PASSWORD=... python scripts/canonicalize_entity_apply.py \\\n"
        f"      --plan {json_path} --apply"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
