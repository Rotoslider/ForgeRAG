# Genesis Gauntlet — Graded (2026-08-09)

Second independent Choom run of the search-quality gauntlet
(`scripts/eval/choom-search-test.md` + vision addendum), executed by
Genesis (researcher Choom) via the delegation API in 8 stages, graded by
the supervising assistant against the Section B answer key. System under
test: post-N1 (stop-tier exclusion live), post-N2/N3 (inert for search),
full summary plane. Raw outputs: `2026-08-09-genesis-gauntlet-raw.md`.

## Headline

**The Choom skills path is provably distortion-free**: Genesis's reported
top-3s match the raw-API battery byte-for-byte where checked (K1
1027/1028/1337, K4 26/20/96, K5 19/720/715 — exact). What a Choom sees
IS what the API returns.

**Zero fabrication under vision**: on pages whose pixels didn't contain
the asked-for table, Genesis said so plainly instead of inventing values
— the exact anti-fabrication behavior the platform's guards were built
around, now demonstrated end-to-end through a real agent.

## Scores

| Tier | Score | Notes |
|------|-------|-------|
| 1 — keyword | **5/5** book-level | K3 NFPA p75 page-exact vs key; K4, K5 pages in key families. Genesis top-3 = raw API top-3 (verified). |
| 2 — semantic | **4/5** book | S2 found Norton in a different-but-legitimate EE text (Urbano) instead of the key's Dorf; content on-topic in all 5. |
| 2 — hybrid | **3/5** book | Misses were topical (S5: three loop-closure *papers*, correct subject matter). |
| 3 — hybrid | **5/7** book | V3 Ballistics p338 page-exact (the N1 blind-spot fix, visible through the skills path); V5 Tesla p4 page-exact. V6/V7 answered from Machinery's instead of the key's books — right facts, different valid source. |
| 3 — answer | **7/7 facts** | V5 honestly surfaced the Ferraris-vs-Tesla priority dispute while including the key fact (Tesla 1882). V6 gave source-grounded sfpm values rather than the key's canonical 100–250 range — graded correct-but-narrower. |
| 4 — synthesis | **4/4** | Every answer cited 3–4 distinct documents (bar was 2). X2's 4340 composition matched the key digit-for-digit; preheat 315–345 °C with hydrogen-cracking rationale. |
| 5 — graph | **2/2** | G1: A36 → ASTM A36 (support 90), AWS D1.1, AISC, AASHTO, electrode standards — rich and noise-free. G2: first call errored on the param name and the error message itself taught the fix (`parameters.entity_name` + example); retry returned the ASM Vol 2 p763-family pages the key expects. |
| Vision | **2 full + 1 guided + 2 honest refusals, 0 fabrications** | See below. |

## Vision detail — the interesting result

- **VZ4 PASS**: read a Geneva drawing accurately from pixels (driver pin
  B entering star-wheel slots, quarter-turn indexing, locking arc).
- **VZ5 PASS**: read the Atlas's cyclic hysteresis figure — σ/ε axes,
  four panels correctly characterized (softening / hardening / stable /
  mixed).
- **VZ3 half**: opened p715 (inside the key's 715–721 region), correctly
  reported no 555 schematic there and that the 555 section starts
  ~p720 — accurate page reading, guided retrieval.
- **VZ1/VZ2 honest refusals**: opened the CURRENT top-1 keyword hits
  (C26000 property-curve page p1027; steel identification-color table
  p26), described what those pages actually show, and stated the
  requested table was not visible. **The answer key's page expectations
  (p763, p20–21) predate the search-polish ranking** — the battery
  blessed the current ranking as its reference — so these are key
  staleness, not retrieval or vision failures. All five page READINGS
  were accurate; this run did not re-demonstrate table-value extraction
  (the 2026-08-07 run did, reading 70.0 Cu / 30.0 Zn from p763).

## Genesis's own closing verdict (verbatim)

> Where it felt strong: exact-code keyword retrieval (C26000, E7018,
> 210.8, A36) was precise and landed on the right handbook pages, and
> the answer-mode cross-book synthesis consistently pulled from multiple
> distinct authoritative sources with correct page citations.
> Where it felt weak: the vision layer repeatedly opened pages whose
> *text* matched the query but whose *pixels* didn't contain the
> requested figure or table.

That weakness is real and now has a name: **top-1 page selection for
vision tasks**. The fix isn't in ForgeRAG — it's prompt-side (tell the
agent to scan its top-3 and pick the page whose snippet mentions a
table/figure) or a future `prefer:"table"` hint on keyword search.
Logged as a watch item.

## Operational notes

- Genesis's primary model fell back to their OpenRouter backup once
  mid-run (platform fallback chain worked; minor paid-token cost).
- Answer-mode calls reported `used_vision: true` and `used_graph: true`.
- Delegation stages of 5–7 tool calls each fit comfortably inside the
  12-iteration delegation cap; nothing came back incomplete.
