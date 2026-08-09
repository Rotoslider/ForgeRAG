# Search evaluation harness

Frozen from the August 2026 search audit. Run after any material change
(model swap, Docling bump, store migration, retrieval refactor) and compare
against the answer key — regressions show up as changed top-3 results.

- `search_battery.py` — 45-test scripted battery across every mode
  (keyword incl. metachar/fuzzy probes, semantic, visual, chunks, all
  hybrid strategies, graph queries, answer-mode cross-book synthesis).
  Writes compact JSONL; diff two runs to see exactly what moved.
  `venv/bin/python scripts/eval/search_battery.py` (edit OUT path inside).
- `choom-search-test.md` — the same question set as agent-pasteable
  instructions (Section A) with the page-grounded answer key (Section B),
  plus the vision addendum. Exercises the skills-API path my direct
  battery cannot.

Page numbers in the key are PDF positions. Where the corpus holds several
valid sources, the key lists the expected one; grade honest alternatives
as passes.
