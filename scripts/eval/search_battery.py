"""ForgeRAG search-quality battery — runs every mode against the graded
question set plus edge/bug probes. Sequential (one LLM at a time), compact
JSONL output for post-analysis."""
import json
import time
import urllib.request

BASE = "http://localhost:8200"
OUT = "battery_results.jsonl"


def post(path, body, timeout=300):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r), round(time.time() - t0, 1)
    except Exception as exc:
        body_text = ""
        if hasattr(exc, "read"):
            try:
                body_text = exc.read().decode()[:400]
            except Exception:
                pass
        return {"success": False, "reason": f"{type(exc).__name__}: {exc} {body_text}"}, round(time.time() - t0, 1)


def top3(data):
    hits = data if isinstance(data, list) else (data or {}).get("hits", data or [])
    out = []
    for h in (hits or [])[:3]:
        if isinstance(h, dict):
            out.append({
                "doc": (h.get("document_title") or h.get("title") or "?")[:60],
                "page": h.get("page_number"),
                "score": round(h.get("score", 0), 3) if isinstance(h.get("score"), (int, float)) else h.get("score"),
            })
    return out


TESTS = []
# Tier 1 — keyword
for qid, q in [("K1", "C26000"), ("K2", "E7018"), ("K3", "210.8 ground-fault"),
               ("K4", "A36 yield point"), ("K5", "555 timer astable")]:
    TESTS.append((qid, "/search/keyword", {"query": q, "limit": 5}))
# keyword edge probes
TESTS.append(("K6-slash", "/search/keyword", {"query": "3/16 weld", "limit": 5}))
TESTS.append(("K7-fuzzy", "/search/keyword", {"query": "weldlng electrode", "limit": 5, "fuzzy": True}))
TESTS.append(("K8-paren", "/search/keyword", {"query": "weld(ing test", "limit": 5}))
# Tier 2 — semantic + hybrid rrf
for qid, q in [("S1", "Fick's first law of diffusion"),
               ("S2", "Norton equivalent circuit"),
               ("S3", "austempering of ductile iron"),
               ("S4", "kinematics of external Geneva wheels"),
               ("S5", "loop closure detection in SLAM")]:
    TESTS.append((qid + "-sem", "/search/semantic", {"query": q, "limit": 5}))
    TESTS.append((qid + "-rrf", "/search/hybrid", {"query": q, "limit": 5, "strategy": "rrf"}))
# Tier 3 — vague via hybrid strategies
V = [("V1", "Which brass is best for making cartridge cases and why?"),
     ("V2", "What connection types are used for three-phase transformer circuits?"),
     ("V3", "How do you keep a spinning projectile stable in flight?"),
     ("V4", "How far away do I need to stand from an electrical arc hazard?"),
     ("V5", "Who discovered the rotating magnetic field and when?"),
     ("V6", "How fast should a lathe run when cutting with high-speed steel tooling?"),
     ("V7", "What friction coefficient applies to a body at rest on an incline?")]
for i, (qid, q) in enumerate(V):
    strat = ["rrf", "graph_boosted", "graph_first"][i % 3]
    TESTS.append((f"{qid}-{strat}", "/search/hybrid", {"query": q, "limit": 5, "strategy": strat}))
# graph_first regression (metachars)
TESTS.append(("GF-slash", "/search/hybrid", {"query": "3/16 weld fitting spec", "limit": 5, "strategy": "graph_first"}))
TESTS.append(("GB-alloy", "/search/hybrid", {"query": "4340 steel properties", "limit": 5, "strategy": "graph_boosted"}))
# visual
for qid, q in [("VIS1", "stress strain curve diagram"),
               ("VIS2", "555 timer circuit schematic"),
               ("VIS3", "geneva wheel mechanism drawing")]:
    TESTS.append((qid, "/search/visual", {"query": q, "limit": 5}))
# chunks
TESTS.append(("CH1", "/search/chunks", {"query": "tap drill size for threads", "limit": 5}))
TESTS.append(("CH2", "/search/chunks", {"query": "arc flash boundary calculation", "limit": 5}))
# graph queries
TESTS.append(("G1", "/graph/query", {"query_type": "material_standards", "parameters": {"material": "A36"}, "limit": 10}))
TESTS.append(("G2", "/graph/query", {"query_type": "entity_pages", "parameters": {"entity_name": "C26000"}, "limit": 10}))
# Tier 4 — answer mode (cross-book) + one vague
ANSWERS = [
    ("X1", "What copper alloy should I use to deep-draw cartridge cases, and what are its composition and mechanical properties?"),
    ("X2", "What is the composition of AISI 4340 steel, and what preheat considerations apply when welding hardenable low-alloy steels like it?"),
    ("X3", "How do I estimate the endurance limit for a rotating steel shaft, and what factors modify it?"),
    ("X4", "Where does the electrical code require GFCI protection, and what does a ground-fault interrupter actually do?"),
    ("XV3", "How do you keep a spinning projectile stable in flight?"),
]

with open(OUT, "w") as f:
    for qid, path, body in TESTS:
        res, dt = post(path, body, timeout=180)
        row = {"id": qid, "path": path, "q": body.get("query") or body.get("query_type"),
               "ok": res.get("success"), "secs": dt}
        if res.get("success"):
            row["top3"] = top3(res.get("data"))
            if not row["top3"]:
                row["raw_keys"] = list((res.get("data") or {}).keys()) if isinstance(res.get("data"), dict) else f"list[{len(res.get('data') or [])}]"
                row["raw_sample"] = str(res.get("data"))[:300]
        else:
            row["error"] = str(res.get("reason"))[:300]
        f.write(json.dumps(row) + "\n")
        f.flush()
        print(f"{qid}: ok={row['ok']} {dt}s")
    for qid, q in ANSWERS:
        res, dt = post("/search/answer", {"query": q, "limit": 5, "use_graph": True}, timeout=420)
        row = {"id": qid, "path": "/search/answer", "q": q[:60], "ok": res.get("success"), "secs": dt}
        if res.get("success"):
            d = res.get("data") or {}
            row["answer"] = (d.get("answer") or "")[:900]
            row["sources"] = [
                {"doc": (s.get("document_title") or "?")[:55], "page": s.get("page_number"),
                 "adj": s.get("adjacent", False)}
                for s in (d.get("sources") or [])]
            row["distinct_docs"] = len({s["doc"] for s in row["sources"]})
        else:
            row["error"] = str(res.get("reason"))[:300]
        f.write(json.dumps(row) + "\n")
        f.flush()
        print(f"{qid}: ok={row['ok']} {dt}s docs={row.get('distinct_docs')}")

print("BATTERY COMPLETE")
