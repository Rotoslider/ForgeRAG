# ForgeRAG Search Quality Test — Choom Edition

## SECTION A — PASTE EVERYTHING BELOW THIS LINE INTO THE CHOOM'S CHAT

---

I want you to run a structured search-quality test against ForgeRAG (your
engineering library). Use your ForgeRAG search tools. There are five ways to
search — exact keyword, semantic (meaning-based), hybrid (graph-aware; it has
strategies like rrf, graph_boosted, graph_first if your tool exposes them),
answer mode (reads the pages and writes a cited answer), and graph queries
(query_type: material_standards, process_materials, page_entities,
entity_pages with a parameters dict).

For EVERY question below:
1. Run the search mode(s) named for that question.
2. Record: the mode used, the top 3 results (document title + page number),
   and for answer mode the full answer text with its citations.
3. Mark HIT if the expected book appears in your top 3 (you don't know the
   expected book — just record honestly; I'll grade).
4. If a search errors or returns something weird, quote the error exactly —
   that's valuable data, not a failure on your part.

Do not use outside knowledge to answer — only report what ForgeRAG returns.

### Tier 1 — exact codes and terms (use KEYWORD search)
K1. Search: C26000
K2. Search: E7018
K3. Search: 210.8 ground-fault
K4. Search: A36 yield point
K5. Search: 555 timer astable

### Tier 2 — technical phrases (run each in SEMANTIC, then again in HYBRID)
S1. Fick's first law of diffusion
S2. Norton equivalent circuit
S3. austempering of ductile iron
S4. kinematics of external Geneva wheels
S5. loop closure detection in SLAM

### Tier 3 — vague questions a person would actually ask
(run each in HYBRID, then in ANSWER mode)
V1. Which brass is best for making cartridge cases and why?
V2. What connection types are used for three-phase transformer circuits?
V3. How do you keep a spinning projectile stable in flight?
V4. How far away do I need to stand from an electrical arc hazard?
V5. Who discovered the rotating magnetic field and when?
V6. How fast should a lathe run when cutting with high-speed steel tooling?
V7. What friction coefficient applies to a body at rest on an incline?

### Tier 4 — cross-book synthesis (ANSWER mode only — these need multiple
books; note in your report WHICH different documents the citations came from)
X1. What copper alloy should I use to deep-draw cartridge cases, and what are
    its composition and mechanical properties?
X2. What is the composition of AISI 4340 steel, and what preheat
    considerations apply when welding hardenable low-alloy steels like it?
X3. How do I estimate the endurance limit for a rotating steel shaft, and
    what factors modify it?
X4. Where does the electrical code require GFCI protection, and what does a
    ground-fault interrupter actually do?

### Tier 5 — knowledge graph (use graph_query)
G1. graph_query with query_type "material_standards" and parameters
    {"material": "A36"} — list what comes back.
G2. graph_query with query_type "entity_pages" and parameters
    {"name": "C26000"} (if that errors, try {"entity": "C26000"}) — list the
    top pages returned.

### Final report format
End with a table: question ID | mode(s) | top result (doc + page) | HIT/MISS/
ERROR — then three sentences on where ForgeRAG felt strong and where it felt
weak.

---
## END OF CHOOM PASTE

## SECTION B — ANSWER KEY (for Donny; grade against this, or paste it to the
## Choom AFTER it finishes so it can self-grade)

Page numbers are PDF page positions (what ForgeRAG's citations and page links
use), not the numbers printed in the books.

| ID | Expected source (top-3 should include) | Expected fact |
|----|----------------------------------------|---------------|
| K1 | ASM Handbook Vol 2, p763 / p772 | C26000 = cartridge brass, 70% — 70.0 Cu / 30.0 Zn; excellent cold workability. (Schuler Metal Forming p472 also cross-references it — bonus hit) |
| K2 | ASM Handbook Vol 6, p154 / p156 | E7018 = low-hydrogen iron-powder SMAW electrode (coating constituents table) |
| K3 | NFPA 70 NEC, p75 | Section 210.8, GFCI protection for personnel, 210.8(A)–(D) |
| K4 | Structural Steel Designers Handbook, p20–21 | A36: min yield 36 ksi, tensile 58–80 ksi |
| K5 | Practical Electronics for Inventors, p715 (TOC p19) | 555 timer IC — astable/monostable operation |
| S1 | Callister Materials Science 8th, p155 | Diffusion flux ∝ concentration gradient; J = −D dC/dx |
| S2 | Electrical Engineering Handbook (Dorf), p71 | Norton equivalent = current source ∥ impedance; ZT = ZN |
| S3 | ASM Handbook Vol 4, p11 (deeper article follows) | Austempering of ductile iron — dedicated heat-treating article |
| S4 | Mechanisms & Mechanical Devices Sourcebook, p190 | Kinematics of External Geneva Wheels (ch. 7) |
| S5 | SLAM Handbook, p19 (also p11) | Front-end detects loop closures; back-end estimates pose + map |
| V1 | ASM Vol 2 p772 (and/or Schuler p472) | C26000 cartridge brass — excellent cold workability, used for cartridge cases |
| V2 | Delmar Electricity, p624–632 / p716 | Wye and delta connections (+ delta–wye w/ neutral, T, Scott, zig-zag) |
| V3 | Ballistics (Carlucci), p337–338 | Gyroscopic stability requires P² − 4M > 0; necessary but not sufficient (dynamic stability too) |
| V4 | Electrical Safety Handbook, p295 (also p243) | Arc flash boundary — NFPA 70E Annex D calculation methods |
| V5 | Tesla Lectures/Patents, p4 | February 1882, Budapest — Tesla discovered the rotating magnetic field |
| V6 | Machining & Metalworking (Cormier), p329 | HSS: 100–250 surface feet per minute |
| V7 | Marks' Standard Handbook, p125 (example p118) | Static friction: f₀ = tan α₀ (coefficient of friction of rest) |
| X1 | ASM Vol 2 (p763 properties, p772 workability) + Schuler (p472 designation cross-ref; deep drawing process ~p156+) | C26000 70/30 brass; 303–896 MPa tensile range by temper; excellent cold workability; Schuler links C26000 = CuZn28/cartridge brass for deep drawing |
| X2 | Engineering Properties of Steels p81 + ASM Vol 6 (p43 cooling-rate example; preheat discussion) | 4340: 0.38–0.43 C, 0.60–0.80 Mn, 1.65–2.00 Ni, 0.70–0.90 Cr, 0.20–0.30 Mo; preheat slows cooling rate to avoid crack-prone martensite in hardenable steels |
| X3 | Shigley 11th, ch. 6 (~p302–320) + a properties source (Eng. Properties of Steels or ASM) | Se′ ≈ 0.5 Su_t for steels (Su_t < 200 ksi); modified by Marin factors (surface, size, load, temperature, reliability) |
| X4 | NFPA 70 NEC p75 + Delmar / Electrical Safety Handbook | NEC 210.8 lists required GFCI locations; a GFCI trips on line/neutral current imbalance (a few mA) |
| G1 | Knowledge graph | Standards nodes governing A36 (expect ASTM-family standards; any non-empty, relevant result is a pass) |
| G2 | Knowledge graph | Pages mentioning C26000 — should include ASM Vol 2 pages (763/772 family) |

Scoring guide: Tier 1 should be near-perfect (5/5). Tiers 2–3 strong (≥80%
with the right book in top 3). Tier 4 passes if the answer cites ≥2 different
documents and gets the facts right. Tier 5 passes if the graph returns
relevant non-empty results without errors.

---
# VISION ADDENDUM (Choom paste + key)
See conversation message of 2026-08-09 for the paste block and answer key —
same content as delivered inline.
