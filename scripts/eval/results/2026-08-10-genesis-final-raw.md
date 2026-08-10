# Genesis FINAL gauntlet — raw stage outputs (2026-08-10, campaign-complete system)

## Stage 1

All five keyword searches returned clean, no errors. Stage 1 table:

| ID | Top (doc + page) | 2nd | 3rd |
|---|---|---|---|
| K1 | ASM_Handbook_Vol_02 Nonferrous — p.1027 | ASM Vol.02 — p.1028 | ASM Vol.02 — p.1337 |
| K2 | Structural Steel Designers Handbook 3rd ed. — p.286 | ASM_Handbook_Vol_06 Welding — p.1608 | Machinery's Handbook 27th — p.1415 |
| K3 | NFPA 70HB-14 NEC — p.75 | NFPA 70HB-14 NEC — p.76 | NFPA 70HB-14 NEC — p.28 |
| K4 | Structural Steel Designers Handbook 3rd ed. — p.26 | Structural Steel Designers Handbook 3rd ed. — p.20 | Guide to Design Criteria for Bolted & Riveted Joints — p.96 |
| K5 | Practical Electronics for Inventors 4th — p.19 | PEfI — p.720 | PEfI — p.715 |

Mode used for all: `keyword`. Stage 1 matched the prior baseline run exactly. Ready for Stage 2.
## Stage 2

| ID | Top | 2nd | 3rd |
|---|---|---|---|
| S1 | materials-science-and-engineering-8th-edition-callister — p.156 | same doc — p.171 | same doc — p.155 |
| S2 | Introductory Electrical Engineering With Math Explained In Accessible Language (Magno Urbano) — p.273 | same doc — p.277 | same doc — p.268 |
| S3 | ASM Practical Heat Treating (Boyer/Dossett) — p.230 | ASM Handbook Vol.01 — p.71 | ASM Handbook Vol.01 — p.77 |
| S4 | Mechanisms and Mechanical Devices Sourcebook (Sclater/Chironis) — p.199 | Machine Design Databook (Lingaiah) — p.1045 | Machine Design Databook (Lingaiah) — p.1044 |
| S5 | SLAM Handbook — p.197 | SLAM-past-present-future — p.7 | Handheld LiDAR SLAM Guidance (Tersus) — p.13 |

Mode: `semantic` for all. No errors. Ready for Stage 3.
## Stage 3

| ID | Top | 2nd | 3rd |
|---|---|---|---|
| S1 | materials-science-and-engineering-8th-edition-callister — p.156 | materials-science-and-engineering-8th-edition-callister — p.960 | materials-science-and-engineering-8th-edition-callister — p.171 |
| S2 | Introductory Electrical Engineering With Math Explained In Accessible Language by Magno Urbano — p.267 | Practical Electronics for Inventors, 4th Edition by Paul Scherz, Simon Monk — p.106 | *(2nd pair missing — returning what API returned)* |
| S3 | ASM_Handbook_Vol_04 Heat_Treating (1991) — p.1492 | J. R. Davis Metals Handbook Desk Edition 2nd (1998) — p.1794 | ASM_Handbook_Vol_01 Properties_and_Selection Irons_Steels_and_High_Performance_Alloys (1997) — p.77 |
| S4 | Machine analysis with computer applications for mechanical engineers by Doane, James — p.167 | Mechanisms and Mechanical Devices Sourcebook by Neil Sclater, Nicholas Chironis — p.7 | *(2nd pair missing)* |
| S5 | DTD: Density Triangle Descriptor for 3D LiDAR Loop Closure Detection — p.1 | Loop Closure with 3D Gaussian Splatting for Dynamic SLAM — p.1 | Lidar–Vision Depth Fusion for Robust Loop Closure Detection — p.14 |

Mode: `hybrid` for all, no errors. Ready for Stage 4.
## Stage 4

Running all seven hybrid searches in parallel.



| ID | Top | 2nd | 3rd |
|---|---|---|---|
| V1 | Shigley's Mechanical Engineering Design (11th ed) — p.173 | Shigley's (9th ed) — p.82 | ASM Handbook Vol.02 — p.1023 |
| V2 | Delmar's Standard Textbook of Electricity 6th ed — p.743 | Delmar's — p.723 | Delmar's — p.733 |
| V3 | Ballistics: Theory and Design of Guns and Ammunition 3rd ed — p.345 | same doc — p.365 | same doc — p.338 |
| V4 | Electrical Safety Handbook 4th ed — p.192 | same doc — p.434 | same doc — p.274 |
| V5 | Power Electronics — p.121 | Tesla, Nikola — Lectures Patents Articles (1956) — p.4 | Tesla, Nikola — Tribute (1962) — p.234 |
| V6 | ASM Handbook Vol.05 Surface Engineering — p.190 | Machinery's Handbook 27th — p.1081 | Handbook of Mechanical Engineering Calculations — p.1620 |
| V7 | Machinery's Handbook 27th — p.184 | Statics and Dynamics Demystified — p.109 | Machinery's Handbook 27th — p.164 |

All 7 in `hybrid` mode, no errors. Identical to the baseline run. Ready for Stage 5.
## Stage 5

Running V1 first, one at a time as instructed.



V1 done. Running V2.



V2 done. Running V3.



V3 done. Running V4.





V4 done. Running V5.





V5 done. Running V6.



V6 done. Running V7.





All seven answer-mode calls returned clean. Stage 5 summary:

**V1 — Cartridge brass.** Key claim: cartridge brass (70–30, C26000) is best for cartridge cases — optimal ductility+strength, cold-worked through repeated deep draws with annealing; prone to season cracking. **Cited:** Shigley's ME Design 11th (p.172–174), Shigley's 9th (p.81–83), ASM Handbook Vol.02 (p.1022–1024), ASM Handbook Vol.06 (p.2156–58, 1288–90), plus graph-chain standards AMS 4505/4507, ASTM B129, DIN 8582.

**V2 — Three-phase transformer connections.** Key claim: wye & delta configurations (wye-delta, delta-wye, wye-wye caution), plus specialized open-delta (86.6% capacity), T, Scott, and zig-zag for grounding. **Cited:** Delmar's Standard Textbook of Electricity 6th (p.722–724, 732–734, 742–744), ASM Vol.06 (p.932–34, 578–80), graph-chain NFPA 70 clauses 430.7, 690.31(B), 690.35, 250.166.

**V3 — Spinning projectile stability.** Key claim: keep gyroscopic stability factor Sg>1 (P²>4M), dynamic stability 0<Sd<2, avoid roll resonance (e.g., EFP yaw divergence ~300 rad/s). **Cited:** Carlucci & Jacobson, *Ballistics* 3rd ed. (p.337–39, 344–46, 363–66, 393–95) — single book for this one.

**V4 — Arc flash stand-off.** Key claim: no fixed distance — it's the calculated flash-hazard boundary where incident energy reaches 1.2 cal/cm² (NFPA 70E / IEEE 1584), plus Limited/Restricted approach boundaries. **Cited:** *Electrical Safety Handbook* 4th (p.191–93, 273–75, 433–35), NFPA 79-15A (p.13–15, 55–57), graph-chain NFPA 70E & IEEE 1584.

**V5 — Rotating magnetic field.** Key claim: theory attributed to Galileo Ferraris (1885) in Power Electronics, but Tesla (1882 discovery, 1887 practical motors, 1888 patents) with priority established in America per the Tesla documents. **Cited:** *Power Electronics* (p.120–122), *Tesla Lectures Patents Articles* (p.3–5), *Tribute to Nikola Tesla* (p.233–35) — a genuine cross-book historical synthesis.

**V6 — HSS lathe speed.** Key claim: speed depends on material/operation — finish turning 400–3000 ft/min, HSS broaching 3–50 fpm by material (AISI 1040 10–30, superalloys 3–10), Hi-E optimum ~33 ft/min. **Cited:** ASM Handbook Vol.05 (p.189–91), *Machinery's Handbook* 27th (p.1080–82), *Handbook of Mechanical Engineering Calculations* (p.1619–21), ASM Vol.06 (p.918–20), plus graph-chain carbide/brazing standards AWS B2.2, ASME B31.1, ASME BPVC IX.

**V7 — Friction on incline.** Key claim: coefficient of static friction μs applies to a body at rest, with μ=tanθ at the angle of repose; e.g., clean steel-on-steel μs=0.8. **Cited:** *Machinery's Handbook* 27th (p.163–65, 183–85), *Statics and Dynamics Demystified* (p.108–110), *Design Engineers Reference Guide* (p.216–20).

All ran in answer mode, `used_vision: true`, `used_graph: true`, no errors. Ready for Stage 6.
## Stage 6

Running X1 first, one at a time.



X1 done — 4 distinct documents. Running X2.



X2 done — 3 distinct documents. Running X3.



X3 done — 3 distinct primary documents in the returned sources. Running X4.



G1 returned 54 rows. Now running G2 with the requested `name` parameter first.



The first G2 call returned this error verbatim; retrying with the tool-specified `entity_name` parameter.



## X1

**Key facts:** The answer identifies cartridge brass, C26000, as the recommended deep-drawing alloy: approximately 70% Cu / 30% Zn, with a strong ductility–strength combination; cases are formed through repeated cold draws with intermediate annealing. It also reports poor corrosion resistance and susceptibility to residual-stress cracking.

**Cited documents/pages:**
- *Shigleys Mechanical Engineering Design*, 11th ed.: pp. 172–174
- *Shigley’s Mechanical Engineering Design*, 9th ed.: pp. 81–83
- *Textbook of Machine Design* — Khurmi & Gupta: pp. 59–61
- *ASM Handbook Vol. 06 — Welding, Brazing, and Soldering*: pp. 1288–1290, 2156–2158

## X2

**Key facts:** AISI 4340 is reported as C 0.38–0.43%, Mn 0.60–0.80%, Si 0.20–0.35%, Cr 0.70–0.90%, Ni 1.65–2.00%, and Mo 0.20–0.30%. Because it is a highly hardenable low-alloy steel, the answer emphasizes hydrogen-assisted cold-cracking risk, low-hydrogen procedures, maintained preheat/interpass temperature, and postweld heat treatment. The returned answer gives multiple temperature considerations, including above 290 °C / 550 °F, below 205 °C / 400 °F, and a 315–345 °C / 600–650 °F transformation range.

**Cited documents/pages:**
- *Metals Handbook Desk Edition*, 2nd ed.: pp. 597–599
- *ASM Handbook Vol. 06 — Welding, Brazing, and Soldering*: pp. 1061–1063, 1647–1649, 1651–1653
- *Engineering Properties of Steels*: pp. 135–137

## X3

**Key facts:** Estimate the unmodified steel endurance limit as approximately \(S'_e = 0.5S_{ut}\) for steels below the stated strength limit, then apply load, size, surface, temperature, reliability, and material factors. The returned example uses a 1.75-inch SAE 1050 cold-drawn shaft: \(S'_e=38\) ksi, size factor approximately 0.83, reliability factor approximately 0.81, producing an actual endurance limit of 25.5 ksi.

**Cited documents/pages:**
- *Machine Elements in Mechanical Design* — Mott: pp. 200–202
- *Machine Design Solutions Manual* — Norton: pp. 501–504, 628–632, 700–706
- *Shigley’s Mechanical Engineering Design*, 9th ed.: pp. 342–344
- *ASM Handbook Vol. 06 — Welding, Brazing, and Soldering*: pp. 416–418, 793–795

## X4

**Key facts:** The returned answer identifies NEC requirements for 125 V, single-phase, 15- and 20-A receptacles in locations including dwelling bathrooms, garages, outdoors, swimming-pool-related outlets, construction sites, and other wet or high-risk areas. A GFCI compares current in the ungrounded and grounded conductors, detects an imbalance of approximately 5 mA ± 1 mA, and opens the circuit in typically less than 25 ms; the returned answer also notes that it does not protect against every phase-to-neutral or phase-to-phase shock.

**Cited documents/pages:**
- *Delmar’s Standard Textbook of Electricity*, 6th ed.: pp. 45–50, 53–55
- *Electrical Safety Handbook*, 4th ed.: pp. 134–137, 234–236
- *NFPA 70HB-14 NEC Standard*: pp. 74–76, 904–906
- *Marks’ Standard Handbook for Mechanical Engineers*: pp. 1505–1507

## G1 — `material_standards`, `{"material":"A36"}`

The tool returned **54 rows**. The main returned relationships were:

- ASTM A36 → ASTM A36 — support 90
- A36 → ASTM A36 — support 15
- A36 steel → ASTM A36 — support 12
- ASTM A36 → AWS D1.1 — support 7
- Grade 36 steel → AASHTO — support 4
- A36 steel → AISC ASD — support 4
- A36 steel → AISC LRFD — support 4
- ASTM A36 → ASTM A 36 — support 4
- ASTM A36 → AASHTO — support 3
- A36 → AISC ASD — support 2
- ASTM A36 → AISC LRFD — support 2
- Grade 50 Steel → AISC Steel Construction Manual — support 2
- A36 structural steel → ASTM A36 — support 2
- ASTM A36 → ASTM A6 — support 2
- A36 → AWS A5.1 — support 2
- A36 → AWS A5.17 — support 2
- A36 → AWS A5.20 — support 2
- A36 → AWS A5.5 — support 2
- Grade 36 steel → HS20-44 — support 2

The remaining lower-support rows included links to:

- AASHTO, 1972 Specifications, A325, AASHTO ASD
- AISC, AISC 360, AISC LRFD Manual, AISC LRFD Specification
- AISC Manual of Steel Construction, AISC Specifications, AISI Specification
- AREMA Manual, ASME B31.1
- ASTM, ASTM A6, ASTM A36, ASTM A 36, ASTM A373
- AWS A5.1, AWS A5.23, AWS A5.29
- AWS D1.1, AWS D1.1-92, AWS D1.5
- Eurocode 4, HS25, HS20-44, LRFD
- Table 10.15

The tool marked several low-support relationships as `suspect: true`; the high-support primary relationships were marked `suspect: false`.

## G2 — `entity_pages`, requested parameter sequence

The first call with `{"name":"C26000"}` returned this error verbatim:

> `query_knowledge_graph(query_type="entity_pages") needs parameters.entity_name. Example: query_knowledge_graph({ query_type: "entity_pages", parameters: { entity_name: "C12000" } }). This tool looks up specific named engineering entities (codes, standards, alloys). For discovery, news, or "latest/weirdest/recent" queries, use web_search or scrape_page_content instead.`

The retry with `{"entity_name":"C26000"}` succeeded. The returned top pages were:

- *ASM Handbook Vol. 02 — Properties and Selection — Nonferrous Alloys and Special-Purpose Materials*: pp. 763, 772, 815, 847, 849, 856, 858, 860, 907, 1023, 1025–1030, 1035, 1056, 1125–1126, 1337
- *ASM Handbook Vol. 03 — Alloy Phase Diagrams*: p. 23
- *ASM Handbook Vol. 04 — Heat Treating*: pp. 1963, 1966–1967, 1969, 1971
- *ASM Handbook Vol. 06 — Welding, Brazing, and Soldering*: pp. 1874, 2396
- *Denis Cormier — Machining and Metalworking Handbook*: pp. 207, 213, 227
- *Metals Handbook Desk Edition*, 2nd ed.: pp. 198–199, 1369, 1386, 1394, 1411, 1413, 1415–1416, 1487, 1512, 1517–1518, 2070, 2072
- *Machine Design Databook*, 2nd ed. — K. Lingaiah: pp. 56, 76
## Stage 7

Let me re-run both with `prefer="table"` as instructed to get the table-marked hits.



The prefer=table hint worked — it surfaced table-marked preferred matches. For VZ1 the top ASM Vol.2 table hit is p.1337 (C26000 composition row); for VZ2 the best Structural Steel table is p.31 (Table 1.7, Specified Minimum Mechanical Properties, with yield point + tensile columns). Fetching both page images.



**VZ1.** Page opened: *ASM Handbook Vol. 02 — Nonferrous Alloys* — p.1337 (table-marked preferred hit) | What I saw: The table's C26000 row lists **30 Zn**; with the footnote that alloying elements + copper total 100%, copper is **70%**. The row sits under the family heading **"Brass"** in this composition table. | Confidence: High (composition); the table gives the family name "Brass" — the fuller common name "Cartridge brass" appears on other pages (p.772, p.1369) but is not the label on this particular row.

**VZ2.** Page opened: *Structural Steel Designers Handbook* — p.31 (Table 1.7, table-marked preferred hit) | What I saw: This table lists **specified minimum mechanical properties of structural TUBING** — designations A500, A501, A618, A847. **A36 is not present in this table**, so neither the yield-point nor tensile-range values could be read from it. | Confidence: High — honest negative. The prefer=table hint surfaced a genuine table, but it's the tubing table, not the A36 plate/shape table. The A36 yield-point table (Table 1.2, "yield point of 1.2") is elsewhere in the same book (around p.26–27).

---

Notably, the new `prefer="table"` hint worked as designed — it flipped both queries to table-marked preferred matches instead of the figure pages the prior run opened. The tradeoff surfaced in VZ2: it returned a *real* table, just not the A36 one. That's a useful data point for the watch item you logged. Ready for the final vision stage.
## Stage 8

Top results identified. For VZ3 I'll open PEfI p.716 (the 555 astable block diagram/schematic page), VZ4 the Sourcebook p.19 Geneva drawing, and for VZ5 the Atlas p.17 (the tangent-modulus curve diagram — the top Atlas hits are preface text, so this is the first actual curve figure). Fetching all three page images.



All three page images saved. Analyzing them in parallel.



All three pages opened and analyzed from the pixels. Report:

**VZ3.** Page opened: *Practical Electronics for Inventors, 4th ed.* — p.716 | What I saw: A 555 IC in **astable mode** (heading "10.2.1 How a 555 Works — Astable Operation"). Timing is set by **resistors R₁ and R₂ plus capacitor C** — R₁+R₂ charge C toward Vcc, R₂+C discharge it via the pin-7 discharge transistor, producing a self-sustaining square wave at pin 3; there's also a 0.01 µF bypass capacitor on pin 5 (control voltage). | Confidence: High

**VZ4.** Page opened: *Mechanisms and Mechanical Devices Sourcebook* — p.19 | What I saw: Geneval wheel (Fig. 19). A continuously rotating driving wheel A carries pin B, which enters one of the four radial slots in star wheel C, driving it a **¼-revolution step**, then exits; a dwell holds the star stationary by an arc until the next pin entry. Continuous rotary input → discrete intermittent **quarter-turn stepping** output. (Also notes the figures for quick-return mechanisms on the same spread.) | Confidence: High

**VZ5.** Page opened: *Atlas of Stress-Strain Curves* (Tamarin, ASM) — p.17 | What I saw: Figs. 21–22, **hysteresis (cyclic) stress-strain loops**. Axes: **stress σ (vertical) vs strain ε (horizontal)**. At higher values the curves go **nonlinear**, round off at the yield point, and in cyclic loading form **hysteresis loops that widen and flatten** at high strain amplitude — diminishing stiffness, elastic-plastic behavior, and Bauschinger-effect asymmetry between the tension and compression branches. | Confidence: High

---

**Gauntlet close — where ForgeRAG felt strong and weak:** It felt strongest at the retrieval layer — exact-code keyword hits landing on the precise handbook pages, and answer-mode cross-book synthesis reliably pulling multiple distinct authoritative sources with correct citations, with full reproducibility run-over-run. It felt weakest in the vision layer, exactly as flagged in the prior run: pages whose *text* matches the query but whose *pixels* contain a different figure than expected remain the main gap, though the new `prefer="table"` hint was a clear improvement in Stage 7. The one persistent wrinkle worth noting from this final run is that the graph `entity_pages` query wants `entity_name` rather than the intuitive `name`/`entity` key, which is worth fixing in the client tools so future callers don't hit the verbatim error I did.
