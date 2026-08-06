#!/usr/bin/env python3
"""End-to-end smoke test against the LIVE ForgeRAG service.

Drives the real system the way a user does and verifies every claim it
makes: ingests a synthetic PDF through the actual pipeline (priority lane,
so it runs even while jobs are paused and never disturbs the queued
backlog), asserts every step-ledger entry genuinely succeeded, confirms
the completeness audit sees the document as complete, proves retrieval
finds it (keyword, semantic, hybrid; visual endpoint functional), checks
the knowledge graph received its entities, verifies a repair on a
complete document honestly reports "nothing missing" instead of inventing
work, then deletes the document and confirms it is gone.

The design principle, learned the hard way: state audits can't catch
behavioral lies ("reported success without doing the thing"), so this
suite asserts *outcomes* at every hop of the click -> job -> state ->
retrieval chain.

Usage:
    ./venv/bin/python scripts/smoke_e2e.py [--base http://localhost:8200]
        [--skip-answer]   skip the VLM answer check (the slowest step)
        [--skip-verify]   skip the deep-verification spot checks
        [--json PATH]     also write a machine-readable report

Exit code 0 = every check passed. Anything else = at least one failure.
Schedule it inside your processing window (it needs the LLM + GPU briefly),
e.g. nightly after resume: it costs ~3 pages of pipeline work.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import httpx

MARKER = "ZIRCONIUM CALIBRATION SMOKE"  # stable, unique-in-library search hook
TIMEOUT_INGEST_S = 20 * 60
POLL_S = 5


class Suite:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.failed = 0

    def record(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"name": name, "ok": ok, "detail": detail})
        icon = "PASS" if ok else "FAIL"
        print(f"  [{icon}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.failed += 1
        return ok

    def require(self, name: str, ok: bool, detail: str = "") -> None:
        """Record; abort the suite on failure (later checks depend on it)."""
        if not self.record(name, ok, detail):
            raise SystemExit(self.finish())

    def finish(self) -> int:
        total = len(self.checks)
        passed = total - self.failed
        print(f"\n{'=' * 60}")
        print(f"SMOKE {'PASS' if self.failed == 0 else 'FAIL'}: "
              f"{passed}/{total} checks passed")
        for c in self.checks:
            if not c["ok"]:
                print(f"  FAILED: {c['name']} — {c['detail']}")
        return 0 if self.failed == 0 else 1


def make_pdf(path: Path, nonce: str) -> None:
    import fitz

    pages = [
        f"ForgeRAG end-to-end smoke test document. Run nonce {nonce}. "
        f"Unique retrieval marker: {MARKER}. This page discusses AISI 4140 "
        "alloy steel, a chromium-molybdenum steel quenched and tempered per "
        "ASTM A29. Typical tensile strength after tempering is 150 ksi.",
        f"Page two of the smoke test ({MARKER}). TIG welding (GTAW) of 4140 "
        "steel uses ER80S-D2 filler metal with 300 F preheat per AWS D1.1. "
        "Post-weld heat treatment reduces hydrogen cracking risk.",
        f"Page three of the smoke test ({MARKER}). Summary table: material "
        "AISI 4140; process GTAW; standard ASTM A29; equipment TIG torch.",
    ]
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(72, 72, 540, 720), text, fontsize=13)
    doc.save(str(path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8200")
    ap.add_argument("--skip-answer", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    s = Suite()
    c = httpx.Client(base_url=args.base, timeout=300.0)
    nonce = uuid.uuid4().hex[:12]
    doc_id: str | None = None
    print(f"ForgeRAG E2E smoke — {args.base} — nonce {nonce}\n")

    try:
        # ------------------------------------------------------- preflight
        r = c.get("/health")
        s.require("service healthy", r.status_code == 200,
                  f"GET /health -> {r.status_code}")

        # --------------------------------------------------------- ingest
        pdf = Path(f"/tmp/forgerag_smoke_{nonce}.pdf")
        make_pdf(pdf, nonce)
        with pdf.open("rb") as fh:
            r = c.post("/ingest", files={"file": (f"smoke_{nonce}.pdf", fh,
                                                  "application/pdf")},
                       data={"collection": "default", "priority": "true"})
        s.require("upload accepted (priority lane)", r.status_code == 200
                  and r.json().get("success") is True,
                  f"POST /ingest -> {r.status_code}")
        job_id = r.json()["data"]["job_id"]

        deadline = time.time() + TIMEOUT_INGEST_S
        job = None
        while time.time() < deadline:
            job = c.get(f"/ingest/jobs/{job_id}").json()["data"]
            if job["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(POLL_S)
        s.require("ingest job completed",
                  job is not None and job["status"] == "completed",
                  f"status={job and job['status']}, "
                  f"err={job and job.get('error_message')}")
        doc_id = job["doc_id"]

        # Every step must have genuinely succeeded. A warning or error in
        # ANY step fails the suite — that is how the dedup bug hid for
        # three months.
        bad_steps = [
            f"{st['name']}={st['status']} ({st.get('detail')})"
            for st in job["steps"]
            if st["status"] not in ("done", "skipped")
        ]
        skipped = [st for st in job["steps"] if st["status"] == "skipped"]
        s.record("every pipeline step clean (no errors/warnings)",
                 not bad_steps, "; ".join(bad_steps))
        s.record("no unexpected skips (LLM/GPU services were all available)",
                 not skipped,
                 "; ".join(f"{st['name']}: {st.get('detail')}" for st in skipped))

        # ---------------------------------------------------------- audit
        r = c.get("/admin/audit/completeness", timeout=600.0)
        docs = r.json()["data"]["documents"]
        mine = next((d for d in docs if d.get("doc_id") == doc_id), None)
        s.require("document appears in completeness audit", mine is not None)
        gaps = {k: v for k, v in mine["aspects"].items()
                if v["status"] not in ("done", "na")}
        s.record("audit shows document fully complete", not gaps,
                 json.dumps(gaps)[:200])

        # ------------------------------------------------------ retrieval
        r = c.post("/search/keyword", json={"query": MARKER, "limit": 10})
        hits = r.json().get("data") or []
        hit_docs = {h.get("doc_id") for h in hits}
        s.record("keyword search finds the document", doc_id in hit_docs,
                 f"{len(hits)} hits")

        r = c.post("/search/semantic", json={
            "query": "smoke test document with a zirconium calibration marker",
            "limit": 10})
        hits = r.json().get("data") or []
        s.record("semantic search finds the document",
                 doc_id in {h.get("doc_id") for h in hits},
                 f"{len(hits)} hits")

        r = c.post("/search/hybrid", json={"query": f"{MARKER} 4140 welding",
                                           "limit": 10})
        hits = r.json().get("data") or []
        s.record("hybrid search finds the document",
                 doc_id in {h.get("doc_id") for h in hits},
                 f"{len(hits)} hits")

        r = c.post("/search/visual", json={"query": "smoke test calibration",
                                           "limit": 5})
        s.record("visual search endpoint functional",
                 r.status_code == 200 and r.json().get("success") is True,
                 f"-> {r.status_code}")

        # ------------------------------------------------- answer (VLM)
        if not args.skip_answer:
            r = c.post("/search/answer", json={
                "query": f"According to the document with the marker "
                         f"'{MARKER}', what filler metal is used for GTAW "
                         "welding of 4140 steel?",
                "limit": 3, "search_mode": "keyword"}, timeout=600.0)
            body = json.dumps(r.json())
            s.record("answer mode reads the page and cites the fact",
                     r.status_code == 200 and "ER80S" in body,
                     f"-> {r.status_code}, ER80S in answer: {'ER80S' in body}")

        # ---------------------------------------------------------- graph
        r = c.get(f"/documents/{doc_id}/pages")
        pages = r.json()["data"]
        page_id = (pages[0].get("page_id")
                   if isinstance(pages, list) else pages["pages"][0]["page_id"])
        r = c.post("/graph/query", json={"query_type": "page_entities",
                                         "parameters": {"page_id": page_id},
                                         "limit": 10})
        rows = r.json().get("data") or []
        found = rows and any(
            rows[0].get(k) for k in ("materials", "processes", "standards",
                                     "equipment"))
        s.record("knowledge graph has the document's entities", bool(found),
                 json.dumps(rows)[:160])

        # ------------------------------------------- repair honesty check
        # A repair on a COMPLETE document must say "nothing missing" and do
        # zero work — not invent success.
        r = c.post("/admin/fill-missing", json={
            "doc_ids": [doc_id], "text": False, "visual": False,
            "entities": True, "priority": True})
        fix_job_id = r.json()["data"]["jobs"][0]["job_id"]
        deadline = time.time() + 120
        fix = None
        while time.time() < deadline:
            fix = c.get(f"/ingest/jobs/{fix_job_id}").json()["data"]
            if fix["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(2)
        ent_step = next((st for st in (fix or {}).get("steps", [])
                         if st["name"] == "extracting_entities"), None)
        s.record("repair on complete doc reports 'nothing missing'",
                 fix is not None and fix["status"] == "completed"
                 and ent_step is not None
                 and "nothing missing" in (ent_step.get("detail") or ""),
                 f"status={fix and fix['status']}, "
                 f"detail={ent_step and ent_step.get('detail')}")

        # ------------------------------------- deep-verify spot checks
        if not args.skip_verify:
            r = c.get("/admin/verify", timeout=600.0)
            checks = {ch["name"]: ch for ch in r.json()["data"]["checks"]}
            for name in ("repair_coverage_matches", "no_temp_rel_garbage"):
                ch = checks.get(name)
                s.record(f"deep verify: {name}",
                         ch is not None and ch["status"] == "pass",
                         ch and f"violations={ch['violations']}")

    finally:
        # -------------------------------------------------------- cleanup
        if doc_id:
            r = c.delete(f"/documents/{doc_id}")
            s.record("document deleted", r.status_code == 200,
                     f"-> {r.status_code}")
            time.sleep(2)
            r = c.post("/search/keyword", json={"query": MARKER, "limit": 5})
            leftovers = [h for h in (r.json().get("data") or [])
                         if h.get("doc_id") == doc_id]
            s.record("document gone from search after delete", not leftovers)
        try:
            Path(f"/tmp/forgerag_smoke_{nonce}.pdf").unlink(missing_ok=True)
        except OSError:
            pass

    code = s.finish()
    if args.json:
        args.json.write_text(json.dumps(
            {"passed": code == 0, "checks": s.checks}, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
