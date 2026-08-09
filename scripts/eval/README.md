# Search evaluation harness

Frozen from the August 2026 search audit. Run after any material change
(model swap, Docling bump, store migration, retrieval refactor) and compare
against the answer key — regressions show up as changed top-3 results.

- `search_battery.py` — 45-test scripted battery across every mode
  (keyword incl. metachar/fuzzy probes, semantic, visual, chunks, all
  hybrid strategies, graph queries, answer-mode cross-book synthesis).
  Writes compact JSONL; diff two runs to see exactly what moved.

Reference runs live in `results/` (date-named). The 2026-08-09 run is the
post-N1 reference: 39/39 ok, 33/39 identical top-3 vs the 08-08 baseline,
and the six that moved include the two blind-spot fixes N1 promised
(projectile-stability now resolves to the Ballistics text, lathe-speed to
the machining-data cluster). Known cosmetic wobble: the "3/16 weld
fitting spec" probe gained ASME B31.1 (the actual governing spec) in its
top-3 but also admitted one off-topic page at #2 — graded criteria still
pass; watch it in future runs.
  `venv/bin/python scripts/eval/search_battery.py` (edit OUT path inside).
- `choom-search-test.md` — the same question set as agent-pasteable
  instructions (Section A) with the page-grounded answer key (Section B),
  plus the vision addendum. Exercises the skills-API path my direct
  battery cannot.

Page numbers in the key are PDF positions. Where the corpus holds several
valid sources, the key lists the expected one; grade honest alternatives
as passes.
