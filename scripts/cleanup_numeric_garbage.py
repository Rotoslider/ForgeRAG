#!/usr/bin/env python3
"""Null out Material numeric-field values that are LLM-output debris.

Three fields — tensile_strength_ksi, yield_strength_ksi, hardness — are
supposed to hold numeric or simple-range quantity strings. An audit of the
live graph found ~157 values containing JSON fragment debris, prompt-text
leakage, or wrong-field contamination from a past extraction-pipeline bug.

Classification rule:
  - "clean"  : bare number or `number [unit]` (e.g. "45", "45 ksi")
  - "range"  : `a-b [unit]` or `a to b [unit]` (e.g. "100 to 325 HB")
  - "loose"  : plausible human quantity string (e.g. "280 HV (max), 22 HRC")
  - "garbage": contains any of { } [ ] " : — the characters that only
               appear in LLM output debris, never in a legitimate quantity
               string. This is the ONLY bucket we null.

"loose" values are preserved — they're not machine-parseable but they're
accurate and useful to humans. A future structured-property pass can
re-parse them. This script does not touch them.

Before writing, a full log of every `(node_name, field, old_value)` triple
is dumped so the change is reversible from the log if needed.

Usage:
    NEO4J_PASSWORD=... python scripts/cleanup_numeric_garbage.py
    NEO4J_PASSWORD=... python scripts/cleanup_numeric_garbage.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_settings  # noqa: E402
from backend.services.neo4j_service import Neo4jService  # noqa: E402


NUMERIC_FIELDS = ("tensile_strength_ksi", "yield_strength_ksi", "hardness")

# JSON-structural characters that should never appear in a legitimate
# quantity string. Colon is NOT in this set — it's common in valid
# condition-prefixed values like "QT: 90-115 HB" (Quenched and Tempered).
GARBAGE_CHARS = set('{}[]"')

# The strongest signal that a value is prose rather than a quantity is a
# long stretch of consecutive non-digit characters. Legit quantity strings
# interleave letters and digits frequently (units, parentheticals, range
# separators). Prose has long letter-only spans. Tuned against the actual
# graph data: 22 catches ASTM A572 descriptive text and prompt leakage
# while preserving multi-scale hardness readings like
# "55 HRC (tempered at 315 °C), 96 HRB (annealed)" (longest non-digit
# run = 18 chars).
MAX_NONDIGIT_RUN = 30

_CLEAN_NUMERIC_RE = re.compile(r"^\s*-?\d+(\.\d+)?\s*[A-Za-z]{0,6}\s*$")
_CLEAN_RANGE_RE = re.compile(
    r"^\s*-?\d+(\.\d+)?\s*(?:-|–|—|to)\s*-?\d+(\.\d+)?\s*[A-Za-z]{0,6}\s*$",
    re.IGNORECASE,
)


def _longest_nondigit_run(s: str) -> int:
    cur = best = 0
    for ch in s:
        if ch.isdigit():
            cur = 0
        else:
            cur += 1
            if cur > best:
                best = cur
    return best


def classify(val) -> str:
    """Four-way classification of a numeric-field value.

    - null:    no value / empty
    - clean:   bare number or simple range, optionally with a short unit
    - garbage: clearly LLM debris — JSON-structure chars, leading comma
               (parser-misfile marker), no digits at all, or a long stretch
               of non-digit characters (prose).
    - loose:   plausible human quantity string — condition-prefixed values
               ("QT: 90-115 HB"), multi-scale readings ("55 HRC, 96 HRB"),
               parentheticals. Preserved: semantically correct even if not
               machine-readable.
    """
    if val is None or val == "":
        return "null"
    s = str(val).strip()
    if not s:
        return "null"
    if any(c in GARBAGE_CHARS for c in s):
        return "garbage"
    if s.startswith(","):
        # Parser-misfile pattern: "tensile" field captured ", yield: X, ..."
        # The leading comma means the ingestion parser started mid-field.
        # Whatever number follows is in the wrong column and can't be
        # trusted — null it.
        return "garbage"
    if not any(c.isdigit() for c in s):
        # Pure prose — descriptive text that leaked into a quantity field.
        return "garbage"
    if _longest_nondigit_run(s) > MAX_NONDIGIT_RUN:
        return "garbage"
    if _CLEAN_NUMERIC_RE.match(s) or _CLEAN_RANGE_RE.match(s):
        return "clean"
    return "loose"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually null the garbage values. Without this flag, the "
             "script prints the plan and exits.",
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
    log = logging.getLogger("cleanup_numeric")

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

        log.info("Scanning Material nodes for garbage numeric values...")
        rows = await svc.run_query("""
            MATCH (m:Material)
            RETURN m.name AS name,
                   m.tensile_strength_ksi AS tensile_strength_ksi,
                   m.yield_strength_ksi AS yield_strength_ksi,
                   m.hardness AS hardness
        """)
        log.info("Scanned %d Material nodes", len(rows))

        # Collect (field -> list of (name, old_value))
        per_field: dict[str, list[tuple[str, str]]] = {f: [] for f in NUMERIC_FIELDS}
        for r in rows:
            for f in NUMERIC_FIELDS:
                if classify(r[f]) == "garbage":
                    per_field[f].append((r["name"], r[f]))

        total = sum(len(v) for v in per_field.values())
        affected_nodes = len({
            name for vals in per_field.values() for name, _ in vals
        })

        print()
        print("=" * 60)
        print("Numeric-field garbage cleanup"
              f" — {'APPLY' if args.apply else 'DRY RUN'}")
        print("=" * 60)
        print(f"Material nodes scanned:          {len(rows):>6}")
        print(f"Nodes with ≥1 garbage field:     {affected_nodes:>6}")
        print(f"Total field values to null:      {total:>6}")
        for f, items in per_field.items():
            print(f"  {f:<24} {len(items):>5}")
        print()
        print("Sample garbage values (first 5 per field):")
        for f, items in per_field.items():
            if not items:
                continue
            print(f"  {f}:")
            for name, val in items[:5]:
                vs = str(val)[:80].replace('\n', ' ')
                print(f"    {name!r:30}  {vs!r}")
        print()

        if not args.apply:
            print("DRY RUN — add --apply to null these values.")
            return 0

        # Write full log before mutating anything.
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = args.output_dir / f"numeric_cleanup_{stamp}.log"
        with log_path.open("w") as f:
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Nulled fields (reversible from this log):\n\n")
            for field, items in per_field.items():
                for name, old in items:
                    # Tab-separated, raw old value last — simplest recovery
                    # format. Old values may contain newlines; escape them.
                    safe_old = str(old).replace("\\", "\\\\").replace(
                        "\n", "\\n").replace("\t", "\\t")
                    f.write(f"{field}\t{name}\t{safe_old}\n")
        log.info("Change log written: %s", log_path)

        # Execute: one UNWIND per field. Idempotent — SET to null is a no-op
        # if the value was already nulled.
        for field, items in per_field.items():
            if not items:
                continue
            names = [n for n, _ in items]
            await svc.run_write(
                f"""
                UNWIND $names AS n
                MATCH (m:Material {{name: n}})
                SET m.{field} = null
                """,
                {"names": names},
            )
            log.info("Nulled %s on %d nodes", field, len(names))

        # Post-apply re-scan to confirm.
        rows2 = await svc.run_query("""
            MATCH (m:Material)
            RETURN m.name AS name,
                   m.tensile_strength_ksi AS tensile_strength_ksi,
                   m.yield_strength_ksi AS yield_strength_ksi,
                   m.hardness AS hardness
        """)
        remaining = 0
        for r in rows2:
            for f in NUMERIC_FIELDS:
                if classify(r[f]) == "garbage":
                    remaining += 1
        print(f"Remaining garbage values after cleanup: {remaining:>6}  "
              f"(expected 0)")
        return 0 if remaining == 0 else 3
    finally:
        await svc.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
