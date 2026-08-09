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

### N2. Extraction-time noise valve — future ingests
- Validator rejects standalone generic-noun entities (blocklist from N1 +
  designation-pattern heuristic); reroute generic concepts to topic_tags,
  which is where they belong.
- Drop model-declared relations whose subject/object was not extracted as an
  entity on the same page (currently dropped silently by MATCH; make it a
  logged validator decision).

### N3. Bearer-token auth
It is already a REST API; it lacks authentication, not REST. One static
token: FastAPI middleware + `Authorization` header in the Choom client
(~50 lines total). Bind stays LAN so the Chooms keep working. Remote access,
if ever wanted, is Tailscale — not an auth framework. Fifty-engineer
deployment machinery (users, roles, OAuth) is explicitly out of scope for a
one-human instrument.

### N4. Docling version bump — new ingests only
- Branch, bump, regression-diff three golden books (chunk counts, section
  paths, table extraction — the CSC and SLAM summary trees are ready
  fixtures).
- Stamp `docling_version` on chunks. Adopt for new ingests; **never**
  retroactively re-chunk (it cascades into re-summarize + re-embed for
  marginal gain). Old and new chunks coexist; rebuilds upgrade
  opportunistically.

---

## NEXT (the real 1M-page gate: ingestion throughput)

### T1. vLLM serving for the extractor — the single cheapest big win
At 8–10 s/page of entity extraction, 900k new pages ≈ **100+ days of GPU
nights**. Storage is not the 1M gate; this is.

- vLLM with continuous batching typically yields 3–5× throughput on exactly
  this workload (many concurrent structured-output calls against one model).
- OpenAI-compatible: the client changes a URL. The existing semaphore
  machinery already manages concurrency; raise `max_concurrent_requests`
  to exploit batching.
- Verify structured-output (JSON schema) support for the chosen model under
  vLLM before switching; keep LM Studio as the fallback profile.

### T2. Plane policy per collection
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

### L4. Eval harness as a repo fixture
The 27-question audited battery + answer key currently lives in scratch.
Move to `scripts/eval/`, runnable after any material change (model swap,
Docling bump, L1 migration). The Choom-executed variant doubles as an
integration test of the skills path. Cheap insurance against silent
regressions during every migration above.

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
  outward; verify against the running Neo4j version before committing to
  the sidecar.
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
