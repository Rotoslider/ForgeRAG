# ForgeRAG Roadmap — the 100k → 1M Page Guide

*Written August 2026, while the summary-tree build churned through the library.
This is a trigger-based map, not a schedule: ForgeRAG is a research instrument
used when needed, so every move below is gated on a measurable condition, not
a date.*

## Standing constraints (unchanged from day one)

1. The corpus is private and stays on local hardware.
2. Every answer terminates in a page image a human can open.
3. The index must be able to prove its own completeness with exact counts.
4. New constraint acknowledged 2026: **plan for ~1,000,000 pages** within a
   few years, served to one human and a small number of autonomous agents
   with bursty, curiosity-driven usage.

The doctrine that follows from (3) and governs everything below: **any new
store or plane ships with its verification checks and repair drains on day
one.** Derived data that cannot prove parity with its source is rot waiting
to be discovered.

---

## NOW (worth doing while the summary build runs)

### N1. Graph noise cleanup — retroactive, cheap, high leverage
The bulk-era extractor emitted generic-noun entities ("steel", "fitting",
"resistor" as standalone nodes) and occasional hallucinated relations. This
noise is what makes graph_first drag a Gaussian-splatting paper into a
weld-fitting query.

- **Identify:** noise identifies itself by degree. Query top-degree entities
  per label; generic nouns dominate the head. Add a no-designation-pattern
  filter (no numbers, single dictionary word).
- **Review:** a few hundred candidates; a Choom can do the first pass with
  human veto.
- **Purge:** delete entity + mention edges (pages keep their real entities);
  bank the blocklist for N2.
- **Relations tier 1 (free):** drop/flag edges with support_count = 1 whose
  endpoints never co-occur elsewhere.
- **Relations tier 2 (deferred):** LLM re-adjudication of low-support edges —
  only if graph-strategy quality still warrants after tier 1.
- **Cost:** hours of Cypher + one review session. Zero LLM. **Full era
  re-extraction remains the nuclear option and is NOT justified** (the depth
  sample showed parity on real designations).
- **Design refinement (2026-08, after extracting 306 candidates):** degree
  alone conflates two populations needing different remedies. "steel"
  (5,556 pages) is a useless retrieval discriminator but so is deleting it
  wrong; "water"/"air" are not engineering entities at all. Therefore two
  verdicts: **DELETE** (not a real entity) and **STOP-TIER** (real but
  ubiquitous — kept, marked `e.noise_tier = 'stop'`, excluded from query
  expansion and graph_first seeding; reversible, no per-query degree
  computation). Review executed by Genesis (researcher Choom) with human/
  assistant veto before any write.
- **Status: entity pass DONE 2026-08-09** — Genesis judged all 306, every
  DELETE was vetoed to STOP or held-merge (nothing deleted), 92 entities
  stop-tiered with count-verified writes, exclusion wired into the matcher
  + graph_first + graph_boosted (live at next restart), blocklist banked
  for N2, verification check #30 pins graph↔ledger parity. Full audit
  trail: `docs/noise-review-2026-08.md`. Remaining: the relations tier-1
  edge pass (support_count = 1, endpoints never co-occur) — unstarted.

### N2. Extraction-time noise valve — future ingests — DONE (2026-08-09)
Shipped as `backend/ingestion/noise_valve.py`, applied by both pipeline
extraction lanes after every extract_page: blocklist matches reroute to
page topic_tags (no entity node, no mention edge); relationships whose
endpoints aren't among the page's own extracted identifiers (names,
aliases, UNS/process/standard numbers, formula/table names) drop as
logged validator decisions — this deliberately tightens the old
graph-wide MATCH, which would let a hallucinated link to an off-page
entity succeed; single generic-looking words are LOGGED as candidates
for the next N1 round but never dropped on wordform alone (degree, not
wordform, proves noise — "martensite" is one word and a great
discriminator). Battery after shipping: byte-identical to the post-N1
reference, as expected for a future-ingest-only change.

### N3. Bearer-token auth — DONE (2026-08-09)
Shipped as `backend/services/api_auth.py` + `[server] api_token` in the
toml (FORGERAG_API_TOKEN env wins). Localhost clients, /health, and the
static UI shell are exempt — the Chooms and on-box use need zero changes;
non-localhost API requests 401 without the bearer header. OFF while no
token is configured (loud startup warning) — the owner enables it by
pasting `openssl rand -hex 24` output into the toml and restarting. The
Choom client sends the header automatically when FORGERAG_API_TOKEN is
set. As designed: one static token, no users/roles/OAuth; remote access
is Tailscale's job.

### N4. Docling version bump — new ingests only — IN EVALUATION (2026-08-09)
- `docling_version` stamp on new chunks: SHIPPED (chunker captures the
  installed version; both pipeline chunk writes persist it; pre-stamp
  chunks read as empty — the mix is auditable).
- Regression harness: SHIPPED as `scripts/docling_regression.py` —
  docling-only imports so it runs in a throwaway candidate venv; golden
  books = EGR_450 (81pp), SLAM Handbook (660pp, numbered hierarchy),
  Atlas of Stress-Strain Curves (808pp, figure-dense). 2.90.0 baseline
  reproduces the graph's historical chunk counts (±1), validating the
  harness. Candidate 2.118.1 diff pending; adopt only if chunk counts,
  section-path population, and table extraction hold. **Never**
  retroactively re-chunk (cascades into re-summarize + re-embed for
  marginal gain); old and new chunks coexist, rebuilds upgrade
  opportunistically.

---

## NEXT (the real 1M-page gate: ingestion throughput)

### T1. Batched LLM serving for the extractor — the single cheapest big win
*(Amended after the shared-GPU discussion: one card serves Whisper,
Chatterbox, Stable Diffusion, the Chooms, and ForgeRAG.)*

**Step 0 — llama-server first.** LM Studio is llama.cpp underneath but
serves requests one at a time. Running `llama-server` directly with
`--parallel 8` and a larger context gives continuous batching on the SAME
GGUF file already on disk — zero new downloads, LM-Studio-like memory
behavior, one-evening benchmark. If this yields 2-3x, vLLM may never be
needed.

**Step 1 — vLLM as campaign mode, not a resident.** Facts that matter on a
shared GPU: one vLLM instance serves ONE model to any number of clients
(ForgeRAG and the Qwen Chooms would share the same endpoint — you never run
two copies of Qwen); it is OpenAI-compatible, so both clients change only a
URL; but it PRE-ALLOCATES VRAM (gpu_memory_utilization, default 0.9) and
does not idle-unload, so a resident vLLM competes with SD/Whisper/TTS all
day; and it prefers AWQ/GPTQ/FP8 quants (GGUF support is second-class), so
expect one new quant download. Sleep/wake endpoints exist for VRAM
handoffs but are manual orchestration. Conclusion: run vLLM as a
**campaign profile** — during heavy ingestion, stop LM Studio, start vLLM
with the extractor model, point ForgeRAG AND the Qwen-based Chooms at it,
raise max_concurrent_requests to 8-12 and ingest parallelism to match;
swap back when the campaign ends. A systemd profile pair makes the swap
one command. Non-Qwen Chooms (Gemma) either idle during campaigns or stay
on a slimmed LM Studio if VRAM allows.
At 8–10 s/page of entity extraction, 900k new pages ≈ **100+ days of GPU
nights**. Storage is not the 1M gate; this is.

- vLLM with continuous batching typically yields 3–5× throughput on exactly
  this workload (many concurrent structured-output calls against one model).
- OpenAI-compatible: the client changes a URL. The existing semaphore
  machinery already manages concurrency; raise `max_concurrent_requests`
  to exploit batching.
- Verify structured-output (JSON schema) support for the chosen model under
  vLLM before switching; keep LM Studio as the fallback profile.

### T2. Plane policy per collection — DECLINED by owner (2026-08)
Owner's call: the deferred-entity hit is not worth the later repayment;
every ingested document gets the full treatment at ingest time. Kept here
because the math may change (a 5x faster extractor makes this moot; a 10x
corpus growth spike revives it). Superseded in priority by T1.

*(Original proposal below for the record.)*
Not every document deserves every plane. Proposal:

| Tier | Planes | Example collections |
|------|--------|---------------------|
| Reference | pixels + text + entities + summaries | ASM handbooks, codes, machine design |
| Research  | pixels + text + summaries (entities deferred) | paper collections, one-off PDFs |
| Archive   | pixels + text | bulk scans awaiting promotion |

- Per-collection setting consumed by ingest and the drains; completeness
  audit reports against the doc's *policy*, not a universal bar (a Research
  doc without entities is complete BY POLICY, visibly so).
- Promotion is just running the missing drains later — everything is already
  incremental and convergent.
- This turns the 100-day extraction problem into a knob.

---

## LATER (trigger-gated)

### L1. Qdrant sidecar for vectors
**Trigger: ~250–300k pages, OR a re-embed/index rebuild no longer fits in a
night, OR Neo4j heap pressure visibly degrades graph queries.** Not before.

- **Choice: Qdrant** over a Milvus return: single local Rust binary, strong
  filtering, scalar/binary quantization, and native **dense + sparse +
  multivector (MaxSim)** in one collection — it absorbs the
  dense/sparse/late-interaction ambition in the same move.
- **Architecture: sidecar, never migration.** Neo4j keeps the provenance
  spine, markers, graph, and full-text (Lucene). Qdrant holds derived
  vectors keyed by chunk/page/summary id.
- **Day-one obligations:** cross-store parity checks in the verification
  suite (counts per plane, spot-vector checksums); a rebuild-from-Neo4j
  drain (Qdrant must be reconstructible, i.e., disposable); backup story
  (Qdrant snapshots beside the Neo4j dump).
- **Sequence:** chunks first, parity proven, then pages/summaries, then drop
  Neo4j vector indexes to reclaim heap. Visual multivectors move last or
  never — the two-stage coarse+MaxSim rerank works and its Qdrant
  multivector footprint at 1M pages (~258 vectors/page) needs quantization
  math done first.

### L2. Learned sparse vectors (BGE-M3 sparse output)
BGE-M3 already computes dense + sparse + ColBERT weights in one forward
pass; today only dense is kept. Learned sparse beats BM25 on vocabulary
mismatch while keeping exact-term strength. **Do it WITH L1** (Qdrant hosts
sparse natively; wedging sparse into Neo4j is not worth it). Until then,
Lucene BM25 is honestly adequate for this user count. Text ColBERT
multivectors: watchlist only — storage jump, marginal gain over
RRF + reranker.

### L3. Backup at scale
28 GB of page images today → ~250 GB at 1M pages. The current copy loop
(already flagged in review) needs: event-loop offload, incremental
rsync-style file sync, and Qdrant snapshots once L1 lands. Trigger: before
the library doubles again.

### L4. Eval harness as a repo fixture — DONE (2026-08)
Frozen at `scripts/eval/`: the 45-test battery, the agent-pasteable
question set, and the page-grounded answer key. Run after any material
change; diff JSONL outputs between runs.

---

## REJECTED — with reasons, so future-us doesn't relitigate on vibes

### R1. Rust port of the Python core
The Python layer orchestrates; it waits on the GPU (extraction, embeddings),
the JVM (Lucene), and Neo4j. FastAPI overhead is microseconds against
8-second LLM calls. Every real stall found in the 2026 audit was
sync-work-on-the-event-loop architecture, fixable in-place. A port would
reset the system's actual value — the verification suite, the convergent
drains, the failure doctrine encoding two years of incidents — to buy
single-digit latency percentages. **Escape hatch:** if profiling ever shows
a genuine Python hotspot >20% of a user-visible path, write that one kernel
as a PyO3 extension (candidates: in-process MaxSim at 1M pages; the trigram
matcher past ~1M entity names — currently 40 ms, fine).

### R2. pdf_oxide (or any parser swap) for ingest speed
Wrong bottleneck. Pipeline cost order: LLM extraction ≫ Docling ≫ summaries
> embeddings > rendering > text parsing. A 10× faster parser changes total
ingest time by ~nothing, while trading away PyMuPDF's accumulated edge-case
hardening (encrypted PDFs, vector-outline exports — both live incidents).
**Watchlist condition:** if T1/T2 collapse LLM cost to where parsing shows
up in profiles, benchmark alternatives on 1k pages — one afternoon.

### R3. Full re-extraction of bulk-era pages
Depth-sampled at parity on real designations; the surplus is generic-noun
noise, which N1 removes for free. Re-extraction is days of GPU for negligible
recall gain. Revisit only alongside a step-change in local extractor quality
(and then via T2 tiers, reference collections first).

---

## Watchlist (things that could change the map)

- **Local model step-changes** — a materially better small extractor model
  re-prices T1/T2/R3 at once; re-evaluate the tier policy each major local
  model generation.
- **Neo4j vector quantization maturity** — could extend the L1 trigger
  outward. Checked 2026-08: running 5.26.28 Community (LTS line — current
  and supported; vector quantization landed in the 2025.x series, NOT in
  5.26). Re-check when considering L1: upgrading Neo4j vs adding Qdrant is
  the actual fork.
- **Handwritten material** (the 1980s notes from v1): Docling will not OCR
  handwriting well, but the pixel plane + VLM answer mode reads handwritten
  pages natively. If that corpus returns, it enters as an Archive-tier
  collection and is served by visual retrieval — no new machinery required.
- **Choom usage patterns** — if autonomous exploration becomes heavy, the
  summary plane and graph queries absorb it cheaply; watch the answer-mode
  (VLM) load, which is the only expensive path.

## One-line philosophy check

Every fork above resolves the same way the paper's Section 2 argued: prefer
the simplest tool that fixes a demonstrated failure, adopt the new thing only
where it earns its seams, and never ship a store that cannot prove what it
holds.
