# Graph Noise Review — Genesis Briefing (paste everything below into Genesis's chat)

---
Genesis, I need your researcher judgment on ForgeRAG's knowledge graph. The
entity extractor's early generations created some noise entities, and we've
mechanically extracted the 306 highest-traffic suspects below. Your job is to
classify each one. Do NOT delete or modify anything — you are producing a
recommendation list that will be vetoed/approved before any change is made.

For each entity, assign exactly one verdict:

- **DELETE** — not a legitimate engineering entity at all for a materials/
  mechanical/electrical reference library (examples of the kind of thing
  that earns DELETE: "water", "air", "food", "thing", verbs, adjectives,
  publishing artifacts). Deleting it loses nothing.
- **STOP** — a real, legitimate entity (real material, real process, real
  equipment class) that is simply too ubiquitous to help search ("steel"
  appears on 5,556 pages — it discriminates nothing). It will be KEPT in
  the graph but excluded from query expansion. When in doubt between
  DELETE and STOP, choose STOP — it is reversible and destroys nothing.
- **KEEP** — a legitimate entity that is fine as-is (specific enough that
  its high count reflects genuine importance, e.g. a specific alloy family
  or named process). Use your engineering knowledge; you may use ForgeRAG
  itself to check what pages an entity actually connects to
  (query_knowledge_graph entity_pages) if you are unsure.

Work through ALL of them. Output format — one line each, no commentary:
VERDICT | Label | name
Then finish with a short paragraph: any entities you were genuinely torn
on, and why.

The candidates (Label | name | pages that mention it):

Material|steel|5556
Material|Copper|3199
Material|Aluminum|2906
Material|Nickel|1425
Material|Iron|1414
Material|cast iron|1393
Material|Titanium|1151
Material|carbon steel|1117
Material|Zinc|1100
Material|Brass|1090
Material|Silicon|981
Material|Lead|903
Material|water|817
Material|Hydrogen|790
Material|Chromium|771
Material|concrete|756
Material|Tin|742
Material|Silver|738
Material|Magnesium|695
Material|Tungsten|678
Material|glass|672
Material|Molybdenum|671
Material|Platinum|665
Material|air|630
Material|Bronze|609
Material|Carbon|584
Material|Gold|577
Material|alloy steel|529
Material|wood|516
Material|Oxygen|509
Material|Graphite|501
Material|Nitrogen|474
Material|Cadmium|466
Material|mild steel|425
Material|Mercury|396
Material|Cobalt|382
Material|rubber|381
Material|Tantalum|378
Material|Niobium|374
Material|Zirconium|353
Material|tool steel|351
Material|Helium|350
Material|resistor|335
Material|Alumina|323
Material|metal|322
Material|ceramics|320
Material|ceramic|314
Material|plastic|314
Material|Manganese|304
Material|coal|293
Material|Vanadium|292
Material|copper alloy|289
Material|ferrite|288
Material|Argon|280
Material|Nylon|276
Material|gray iron|276
Material|ductile iron|276
Material|Polyethylene|275
Material|metals|271
Material|Diamond|269
Material|methane|267
Material|natural gas|267
Material|Beryllium|248
Material|Aluminium|247
Material|Palladium|246
Material|Polypropylene|238
Material|Boron|238
Material|plastics|229
Material|PVC|216
Material|Germanium|214
Material|oil|210
Material|austenite|209
Material|Epoxy|207
Material|inductor|205
Material|Polystyrene|199
Material|nickel alloy|198
Material|Antimony|197
Material|Ammonia|196
Material|Uranium|186
Material|fiberglass|184
Material|Phosphorus|180
Material|Polycarbonate|179
Material|propane|176
Material|Indium|175
Material|cast steel|173
Material|Bismuth|173
Material|methanol|172
Material|martensite|172
Material|polymers|170
Material|sulfuric acid|167
Material|Sulfur|166
Material|steam|166
Material|wrought iron|163
Process|Annealing|1822
Process|welding|1570
Process|tempering|1155
Process|quenching|1020
Process|machining|931
Process|forging|745
Process|soldering|740
Process|Brazing|729
Process|grinding|658
Process|normalizing|573
Process|carburizing|547
Process|drilling|512
Process|aging|506
Process|sand casting|483
Process|casting|466
Process|hardening|451
Process|turning|422
Process|milling|415
Process|Electroplating|386
Process|cold working|377
Process|extrusion|354
Process|GMAW|343
Process|sintering|339
Process|hot rolling|334
Process|GTAW|327
Process|cold drawing|323
Process|austenitizing|320
Process|oil quenching|315
Process|nitriding|300
Process|die casting|282
Process|rolling|280
Process|cold rolling|277
Process|SMAW|257
Process|Reaming|241
Process|hot working|240
Process|Pickling|233
Process|drawing|225
Process|tapping|223
Process|arc welding|215
Process|polishing|209
Process|Anodizing|205
Process|shot peening|195
Process|blanking|184
Process|heat treating|181
Process|stress relief|178
Process|forming|174
Process|boring|165
Process|cold forming|163
Process|cutting|159
Process|SAW|157
Process|bending|156
Process|combustion|154
Process|carbonitriding|151
Process|etching|150
Process|age hardening|149
Process|fillet weld|148
Process|lapping|147
Process|FCAW|147
Process|Normalize|143
Process|preheating|140
Process|deep drawing|137
Process|Electrodeposition|135
Process|austempering|130
Process|plating|125
Process|air cooling|125
Process|riveting|124
Process|Electrolysis|111
Process|Austenitize|106
Process|painting|105
Process|annealed|103
Process|spot welding|101
Process|stamping|99
Process|electropolishing|98
Process|martempering|96
Process|through-hardening|95
Process|laser welding|89
Process|Buffing|89
Process|punching|89
Process|Phosphating|89
Process|solidification|89
Equipment|shaft|968
Equipment|inverter|889
Equipment|motor|702
Equipment|battery|681
Equipment|generator|670
Equipment|transformer|669
Equipment|gear|623
Equipment|condenser|619
Equipment|pump|599
Equipment|pipe|534
Equipment|bearing|523
Equipment|furnace|484
Equipment|compressor|472
Equipment|boiler|464
Equipment|flywheel|431
Equipment|turbine|409
Equipment|capacitor|380
Equipment|wind turbine|355
Equipment|beam|339
Equipment|pulley|325
Equipment|fuel cell|323
Equipment|cylinder|323
Equipment|gas turbine|320
Equipment|valve|317
Equipment|bolt|303
Equipment|LiDAR|302
Equipment|pinion|298
Equipment|piston|292
Equipment|thermocouple|281
Equipment|resistor|274
Equipment|steam turbine|269
Equipment|Fastener|263
Equipment|crankshaft|257
Equipment|spur gears|256
Equipment|inductor|255
Equipment|lathe|252
Equipment|die|239
Equipment|PV module|239
Equipment|regenerator|234
Equipment|ball bearing|228
Equipment|PV array|226
Equipment|tank|225
Equipment|diesel engine|218
Equipment|robot|217
Equipment|batteries|212
Equipment|PLC|202
Equipment|spur gear|197
Equipment|column|195
Equipment|spring|193
Equipment|battery bank|193
Equipment|solar panel|190
Equipment|PV system|189
Equipment|steam engine|189
Equipment|cam|188
Equipment|coil|183
Equipment|voltmeter|180
Equipment|worm gear|178
Equipment|DC motor|175
Equipment|alternator|174
Equipment|switch|173
Equipment|sprocket|173
Equipment|IMU|167
Equipment|fan|165
Equipment|storage tank|164
Equipment|evaporator|163
Equipment|press|161
Equipment|nozzle|160
Equipment|bearings|158
Equipment|sheave|158
Equipment|controller|155
Equipment|PV modules|155
Equipment|PV panel|155
Equipment|Conduit|152
Equipment|fuse|151
Equipment|microcontroller|151
Equipment|helical gears|150
Equipment|gearbox|149
Equipment|punch|148
Equipment|transducer|148
Equipment|machine tool|147
Equipment|gears|142
Equipment|power supply|142
Equipment|bevel gears|139
Equipment|Anode|139
Equipment|nut|138
Equipment|camera|137
Equipment|drum|136
Equipment|receiver|135
Equipment|oscilloscope|133
Equipment|piping|132
Equipment|microprocessor|130
Equipment|ammeter|129
Equipment|inverters|128
Equipment|electrolyzer|127
Equipment|relay|127
Equipment|armature|127
Equipment|UAV|127
Equipment|receptacle|125
Equipment|rectifier|125
Equipment|transistor|124
Equipment|engine|120
Standard|NEC|309
Standard|ASTM|292
Standard|ISO|197
Standard|AGMA|191
Standard|SAE|155
Standard|ANSI|154
Standard|UL|153
Standard|OSHA|130
Standard|ASME BPVC|101
Standard|AASHTO|93
Standard|ASME|93
Standard|AISC|92
Standard|AISI|92
Standard|AISC LRFD|77
Standard|UNS|75
Standard|AWS|73
Standard|EPA|53
Standard|AISC ASD|53
Standard|AASHTO LRFD|43
Standard|Energy Star|41
Standard|ASM Handbook|40
Standard|BS|40
Standard|NABCEP|40
Standard|DIN|39
Standard|FIPS|38
Standard|NEMA|36
Standard|IEC|36
Standard|SI|34
Standard|IS|34
Standard|UBC|34
Standard|ASM|30
Standard|IEEE|30
---
# END OF GENESIS PASTE
