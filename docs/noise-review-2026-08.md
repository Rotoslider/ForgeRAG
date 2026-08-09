# N1 Graph Noise Review — Ledger (August 2026)

Roadmap item N1, executed 2026-08-08/09. This document is the audit record:
who judged what, what was vetoed and why, and exactly what was written to
the graph. Nothing was deleted.

## Process

1. **Candidates** — 306 top-degree entities across the four labels
   (Material / Process / Standard / Equipment), extracted with a
   no-designation-pattern filter (single dictionary words, no numbers).
2. **First pass** — Genesis (researcher Choom, DeepSeek v4 Flash) reviewed
   all 306 against the DELETE / STOP / KEEP protocol via the delegation
   API, no-writes rule enforced. Returned 305 line-item verdicts plus a
   torn-calls memo.
3. **Veto pass** — Claude (supervising assistant) reviewed all DELETEs
   against live graph facts (degrees, canonical twins), reconciled the one
   missing verdict, and executed the approved ledger.

## Verdict summary

| Verdict | Genesis | After veto | Action taken |
|---------|---------|-----------|--------------|
| KEEP    | 213 (+1 by intent, see below) | 214 | none |
| STOP    | 85      | **92**    | `e.noise_tier = 'stop'` |
| DELETE  | 7       | **0**     | nothing deleted |
| MERGE (held) | —  | 3 (also stop-marked) | awaiting owner ack |

Genesis's closing prose said "7 DELETE, 60 STOP, 238 KEEP" — a miscount in
their summary note; the 305 line items (213/85/7) are the deliverable and
were used verbatim. The 306th candidate, `Equipment | inductor`, got no
line item, but Genesis's memo says "the equipment instances (both KEEP)" —
ruled KEEP per their stated intent.

## The seven DELETE vetoes

Deletion is the irreversible verdict, so every DELETE was checked against
the live graph. All seven were downgraded — each one severs real
page→entity mention edges that STOP preserves while achieving the same
retrieval exclusion:

| Entity | Degree | Genesis's reason | Veto ruling |
|--------|-------:|------------------|-------------|
| Material \| water | 867 | not an engineering entity | **STOP** — it's a real material (coolant, quench medium, working fluid); Genesis themselves STOPped `steam` and `oil`, and consistency wins |
| Material \| air | 658 | same | **STOP** — same reasoning |
| Material \| resistor | 336 | type misclassification (component, not material) | **STOP** — mislabeled but real; deletion severs 336 page links; Genesis offered this flip as the conservative option |
| Material \| inductor | 205 | same | **STOP** — same reasoning |
| Process \| Normalize | 232 | lexical variant | **MERGE → `normalizing` (743)** — canonical twin exists; merge is lossless where deletion severs 232 links. Held for owner ack; stop-marked meanwhile |
| Process \| Austenitize | 162 | lexical variant | **MERGE → `austenitizing` (377)** — same; held, stop-marked |
| Process \| annealed | 167 | lexical variant | **MERGE → `Annealing` (2,336)** — same; held, stop-marked |

The held merges are pure cleanup, not retrieval-critical: both endpoints
of each pair are stop-tiered, so retrieval behavior is already final.

**Merge addendum (2026-08-09, owner-approved):** all three merges
executed via `merge_entity` under `GRAPH_MERGE_LOCK` with pre/post degree
accounting — annealed(167)→Annealing (2,336→2,472), Normalize(232)→
normalizing (743→948), Austenitize(162)→austenitizing (377→539); losers
deleted, names preserved as `common_names` aliases. `austenitizing` was
never itself a review candidate, but the merge moved excluded mentions
under it, so it inherits its variant's stop ruling. Ledger total:
**90 stop-tier entities** (92 − 3 merged variants + austenitizing).

## What was written (2026-08-09)

- `SET e.noise_tier = 'stop'` on exactly **92** entities — dry-run first
  (every label+name pair must resolve to exactly one node; zero
  mismatches), then count-verified write (92 marked, 92 total in graph).
- Blocklist banked at `backend/resources/noise_blocklist.json` (the N2
  extraction-valve input).
- **No deletions. No merges. No edge changes.** Reversal is one query:
  `MATCH (e) WHERE e.noise_tier='stop' REMOVE e.noise_tier`.

## Consumption wiring (code, live at next restart)

Stop-tier entities remain in the graph, remain explorable via the entity
endpoints, and still appear in page enrichment — they are excluded from
exactly three places:

1. `EntityMatcher.refresh()` — stop names no longer enter fuzzy query
   expansion (`backend/services/entity_matcher.py`).
2. `graph_first` seeding — the entity fulltext seeds skip stop-tier
   (`backend/routers/search.py`).
3. `graph_boosted` scoring — stop-tier matches no longer add boosts
   (`backend/routers/search.py`).

Measured effect (live Cypher, pre/post guard): for a "steel OR quenching
OR 4140" seed set, the generic `quenching` node's 1,020-page fanout
disappears while every 4140 designation (AISI 4140, SAE 4140, 4140
steel…) survives untouched. That is the N1 motivating failure — generic
fanout drowning designation precision — fixed.

Guards are pinned by `tests/test_noise_tier.py`; verification check #30
(`noise_stop_tier_matches_ledger`) proves graph marks and banked
blocklist agree exactly, both directions, forever.

## Genesis's torn calls (recorded for posterity)

- Heat-treating verbs (Annealing 1,822 pages, tempering, quenching,
  hardening…) → STOP per the steel precedent. Upheld.
- `Aluminium` → STOP mirroring `Aluminum`. Upheld.
- `PV panel` → STOP as redundant vs `PV module`. Upheld.
- Alloy families (Brass, cast iron, carbon steel) → KEEP (real retrieval
  discriminators at family level). Upheld.
- `Nitrogen` STOPped despite sub-500 count — "an element this generic
  earns the tier on principle". Upheld; STOP is cheap and reversible.
- `IS` (Indian Standards) flagged ambiguous with English "is" → KEEP with
  a note. Upheld; Standard codes are matched exactly.

## Appendix — full line-item verdicts (Genesis, 2026-08-08)

```
DELETE | Material | water          <- vetoed to STOP
DELETE | Material | air            <- vetoed to STOP
DELETE | Material | resistor       <- vetoed to STOP
DELETE | Material | inductor       <- vetoed to STOP
DELETE | Process | Normalize       <- vetoed to MERGE->normalizing (held; stop-marked)
DELETE | Process | Austenitize     <- vetoed to MERGE->austenitizing (held; stop-marked)
DELETE | Process | annealed        <- vetoed to MERGE->Annealing (held; stop-marked)
(no line) | Equipment | inductor   <- KEEP per Genesis's memo ("both KEEP")
```

The 85 STOP and 213 KEEP line items are recorded verbatim in
`backend/resources/noise_blocklist.json` (STOP set) and in the full raw
transcript of Genesis's review session (including the torn-calls memo
quoted above): `docs/noise-review-2026-08-genesis-raw.md`.

### STOP (85, all approved)

Material (36): steel, Copper, Aluminum, Nickel, Iron, Titanium, Zinc,
Silicon, Lead, Hydrogen, Chromium, concrete, Tin, Silver, Magnesium,
Tungsten, glass, Molybdenum, Platinum, Carbon, Gold, wood, Oxygen,
Nitrogen, rubber, metal, ceramics, ceramic, plastic, coal, metals,
Aluminium, plastics, oil, polymers, steam.

Process (22): Annealing, welding, tempering, quenching, machining,
grinding, normalizing, drilling, aging, casting, hardening, turning,
milling, rolling, drawing, polishing, heat treating, forming, cutting,
bending, plating, painting.

Equipment (27): shaft, motor, battery, generator, transformer, gear,
condenser, pump, pipe, bearing, furnace, compressor, boiler, turbine,
cylinder, valve, Fastener, tank, batteries, bearings, controller,
PV modules, PV panel, gears, inverters, spur gears, engine.

### KEEP (214)

Everything else on the 306-candidate list, including all Standards
(nothing with a designation pattern was ever tiered), alloy families,
and the correctly-typed Equipment resistor/inductor pair.
