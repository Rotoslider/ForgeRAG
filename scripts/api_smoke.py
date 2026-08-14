#!/usr/bin/env python3
"""Full-surface API smoke test against a LIVE ForgeRAG service.

Verifies the API the way a remote agent (e.g. the Raspberry Pi harness)
hits it: bearer-token auth over the LAN, the standard envelope, and every
public read/search/graph/skills/documents endpoint. It is strictly
read-only — it never ingests, mutates, or deletes anything — so it is safe
to run against a production library any time.

Auth semantics checked:
  - /health and the SPA shell are exempt; everything else requires a token
    from a non-localhost client.
  - Admin token: full access.
  - Read-only token (if provided with --readonly-token):
      * GET/HEAD everywhere -> 200
      * read-only POSTs (/search/*, /skills/*, /graph/query|explore,
        /ingest/check-duplicates) -> 200
      * any other write -> 403

Auth checks run through --auth-base (derived from the box's LAN IP when
--base is loopback) so the token path is exercised exactly as a remote
client experiences it — localhost is exempt by design.

Usage:
    ./venv/bin/python scripts/api_smoke.py [--base http://localhost:8200]
        [--token TOKEN] [--readonly-token TOKEN]
        [--skip-answer]   skip the VLM answer check (slow, needs LLM+GPU)
        [--skip-visual]   skip the visual retrieval check (loads the
                          visual model on GPU, freeing it on demand)
        [--skip-verify]   skip the deep-verification spot check
        [--json PATH]     also write a machine-readable report

Exit code 0 = every check passed, 1 = at least one failure.

Tokens: --token / --readonly-token flags win, then FORGERAG_API_TOKEN /
FORGERAG_API_TOKEN_READONLY env vars, then config/forgerag.toml.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import tomllib
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "forgerag.toml"


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
        if not self.record(name, ok, detail):
            raise SystemExit(self.finish())

    def finish(self) -> int:
        total = len(self.checks)
        passed = total - self.failed
        print(f"\n{'=' * 60}")
        print(f"API SMOKE {'PASS' if self.failed == 0 else 'FAIL'}: "
              f"{passed}/{total} checks passed")
        for c in self.checks:
            if not c["ok"]:
                print(f"  FAILED: {c['name']} — {c['detail']}")
        return 0 if self.failed == 0 else 1


def lan_ip() -> str | None:
    """Primary outbound interface address (the one remote devices reach)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return None


def _config_token() -> str:
    try:
        with open(CONFIG_PATH, "rb") as fh:
            cfg = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return ""
    return (cfg.get("server") or {}).get("api_token", "") or ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8200")
    ap.add_argument("--auth-base", default=None,
                    help="URL used for auth checks (remote semantics). Defaults "
                         "to the box's LAN IP + --base port when --base is loopback.")
    ap.add_argument("--token", default=None)
    ap.add_argument("--readonly-token", default=None)
    ap.add_argument("--skip-answer", action="store_true")
    ap.add_argument("--skip-visual", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    token = args.token or os.environ.get("FORGERAG_API_TOKEN") or _config_token()
    if not token:
        print("No API token: pass --token, set FORGERAG_API_TOKEN, or configure "
              "api_token in config/forgerag.toml")
        return 2
    readonly = args.readonly_token or os.environ.get("FORGERAG_API_TOKEN_READONLY") or ""

    # The auth checks must go through a non-loopback address, otherwise the
    # middleware's localhost exemption hides the token path entirely.
    base = args.base.rstrip("/")
    auth_base = args.auth_base
    if not auth_base:
        if base.replace("http://", "").replace("https://", "").startswith(("127.", "localhost")):
            ip = lan_ip()
            port = base.rsplit(":", 1)[-1] if ":" in base else "8200"
            auth_base = f"http://{ip}:{port}" if ip else base
        else:
            auth_base = base

    s = Suite()
    c = httpx.Client(base_url=base, timeout=120.0)
    a = httpx.Client(base_url=auth_base, timeout=120.0)
    H = {"Authorization": f"Bearer {token}"}
    RH = {"Authorization": f"Bearer {readonly}"} if readonly else None

    print(f"ForgeRAG API smoke — base {base} — auth {auth_base}\n")

    def ok(body) -> bool:
        """True if a response parsed to the standard envelope with success=True."""
        return bool(body) and body.get("success") is True

    try:
        # ------------------------------------------------------------------
        # Discovery (OpenAPI)
        # ------------------------------------------------------------------
        r = c.get("/openapi.json", headers=H, timeout=30.0)
        spec = r.json() if r.status_code == 200 else None
        s.require("openapi.json served (with token)", spec is not None,
                  f"-> {r.status_code}")
        if spec is not None:
            want = ["/health", "/documents", "/ingest", "/search/keyword",
                    "/search/answer", "/skills/manifest", "/graph/stats",
                    "/ingest/jobs", "/admin/verify"]
            missing = [p for p in want if p not in spec["paths"]]
            s.record("openapi advertises the full public surface", not missing,
                     "missing: " + ", ".join(missing))
            s.record("openapi is a standard schema (3.x)",
                     spec.get("openapi", "").startswith("3."),
                     spec.get("openapi", ""))

        # ------------------------------------------------------------------
        # Auth semantics (remote / non-localhost path)
        # ------------------------------------------------------------------
        s.record("health exempt from auth", a.get("/health").status_code == 200)
        s.record("remote GET without token -> 401",
                 a.get("/collections").status_code == 401)
        s.record("remote GET with wrong token -> 401",
                 a.get("/collections",
                       headers={"Authorization": "Bearer nope"}).status_code == 401)
        r = a.get("/collections", headers=H)
        s.record("remote GET with admin token -> 200", r.status_code == 200,
                 f"-> {r.status_code}")
        r = a.get("/documents", headers=H)
        s.require("admin token reads documents", r.status_code == 200 and ok(r.json()))

        if RH:
            r = a.get("/documents", headers=RH)
            s.record("readonly token GET -> 200", r.status_code == 200 and ok(r.json()))
            r = a.post("/search/keyword", headers=RH, json={"query": "welding", "limit": 3})
            s.record("readonly token read-only POST /search/keyword -> 200",
                     r.status_code == 200 and ok(r.json()))
            r = a.post("/graph/query", headers=RH,
                       json={"query_type": "page_entities",
                             "parameters": {"page_id": "none"}, "limit": 1})
            s.record("readonly token read-only POST /graph/query -> 200",
                     r.status_code == 200 and ok(r.json()))
            r = a.post("/ingest", headers=RH)
            s.record("readonly token write POST /ingest -> 403",
                     r.status_code == 403 and "readonly-token" in r.text)
            r = a.delete("/documents/definitely-not-real", headers=RH)
            s.record("readonly token DELETE -> 403",
                     r.status_code == 403 and "readonly-token" in r.text)

        # ------------------------------------------------------------------
        # Health / envelope
        # ------------------------------------------------------------------
        r = a.get("/health")
        body = r.json()
        d = body.get("data") or {}
        s.record("health reports success + envelope", ok(body) and
                 set(body) >= {"success", "reason", "data"})
        s.record("neo4j connected", d.get("neo4j_connected") is True)
        s.record("library non-empty", (d.get("document_count") or 0) > 0,
                 f"{d.get('document_count')} documents")

        # ------------------------------------------------------------------
        # Documents, collections, categories, tags
        # ------------------------------------------------------------------
        r = c.get("/documents?limit=5", headers=H)
        docs = (r.json().get("data") or {}).get("documents") or []
        s.require("documents list", ok(r.json()) and len(docs) > 0,
                  f"{len(docs)} docs returned")
        fields = {"doc_id", "title", "collection", "categories", "tags", "page_count"}
        s.record("document row shape complete",
                 fields <= set(docs[0].keys()) if docs else False,
                 f"keys={sorted(docs[0])}")

        doc = None
        if docs:
            r = c.get(f"/documents/{docs[0]['doc_id']}", headers=H)
            doc = r.json().get("data")
            s.record("document detail", ok(r.json()) and isinstance(doc, dict),
                     f"-> {r.status_code}")
        else:
            s.record("document detail", False, "no documents to inspect")

        for name, path in [("collections", "/collections"),
                           ("categories", "/categories"),
                           ("tags", "/tags")]:
            r = c.get(path, headers=H)
            s.record(f"GET {path}", r.status_code == 200 and ok(r.json()),
                     f"-> {r.status_code}")

        # ------------------------------------------------------------------
        # Images: resolve a relative image_url from a real search hit
        # ------------------------------------------------------------------
        r = c.post("/search/keyword", headers=H,
                   json={"query": "welding", "limit": 1})
        hits = r.json().get("data") or []
        img_hit = hits[0] if hits else None
        if img_hit is not None:
            rel = img_hit.get("reduced_image_url") or "/images/x/1/reduced"
            r = c.get(rel, headers=H)
            ctype = r.headers.get("content-type", "")
            s.record("search-hit reduced image resolves", r.status_code == 200
                     and ctype.startswith("image/"), f"-> {r.status_code} {ctype}")
        else:
            s.record("search-hit reduced image resolves", False, "no hits")

        # ------------------------------------------------------------------
        # Search modes
        # ------------------------------------------------------------------
        for path, payload in [
            ("/search/keyword", {"query": "welding steel", "limit": 3}),
            ("/search/semantic", {"query": "welding procedure for steel", "limit": 3}),
            ("/search/hybrid", {"query": "welding procedure for steel", "limit": 3}),
            ("/search/chunks", {"query": "welding steel", "limit": 3}),
            ("/search/summaries", {"query": "welding", "limit": 3}),
        ]:
            r = c.post(path, headers=H, json=payload)
            hits = r.json().get("data") or []
            shaped = all(isinstance(h, dict) and h.get("doc_id") and h.get("page_number")
                         for h in hits) if hits else True
            s.record(f"POST {path}", r.status_code == 200 and ok(r.json())
                     and shaped, f"{len(hits)} hits -> {r.status_code}")

        if not args.skip_visual:
            r = c.post("/search/visual", headers=H,
                       json={"query": "welding diagram", "limit": 2,
                             "candidate_pool": 10}, timeout=300.0)
            s.record("POST /search/visual", r.status_code == 200 and ok(r.json()),
                     f"-> {r.status_code}, {len(r.json().get('data') or [])} hits")

        if not args.skip_answer:
            r = c.post("/search/answer", headers=H,
                       json={"query": "How is alloy C12000 welded?",
                             "limit": 2, "search_mode": "keyword"},
                       timeout=600.0)
            body = r.json()
            ans = body.get("data") or {}
            s.record("POST /search/answer (VLM)", ok(body) and
                     bool((ans.get("answer") if isinstance(ans, dict) else "")),
                     f"-> {r.status_code}")

        # ------------------------------------------------------------------
        # Skills
        # ------------------------------------------------------------------
        r = c.get("/skills/manifest", headers=H)
        man = r.json().get("data") or {}
        man_stats = man.get("stats") or {}
        s.record("GET /skills/manifest", ok(r.json()) and man_stats.get("documents"),
                 f"documents={man_stats.get('documents')}")

        r = c.post("/skills/search", headers=H,
                   json={"query": "welding procedure", "limit": 3})
        s.record("POST /skills/search (auto-routing)", r.status_code == 200 and ok(r.json()),
                 f"-> {r.status_code}")

        r = c.post("/skills/search", headers=H,
                   json={"query": "SA-516.70", "mode": "keyword", "limit": 3})
        s.record("POST /skills/search (forced keyword mode)",
                 r.status_code == 200 and ok(r.json()))

        r = c.post("/skills/batch", headers=H,
                   json={"queries": [{"query": "welding"}, {"query": "cast iron"}]})
        batch = r.json().get("data") or []
        s.record("POST /skills/batch", r.status_code == 200 and ok(r.json())
                 and isinstance(batch, list) and len(batch) == 2,
                 f"{len(batch)} result sets")

        # ------------------------------------------------------------------
        # Knowledge graph
        # ------------------------------------------------------------------
        r = c.get("/graph/stats", headers=H)
        stats = r.json().get("data") or {}
        s.record("GET /graph/stats", ok(r.json()) and bool(stats),
                 f"{len(stats)} labels")

        r = c.get("/graph/entities/material?limit=1", headers=H)
        ents = r.json().get("data") or []
        ent_name = ""
        if ents and isinstance(ents, list) and ents:
            ent = ents[0]
            ent_name = (ent.get("key") or ent.get("properties", {}).get("name") or "")
        s.record("GET /graph/entities/material", r.status_code == 200 and ok(r.json()),
                 f"sample entity: {ent_name or '(none)'}")

        if ent_name:
            r = c.post("/graph/query", headers=H,
                       json={"query_type": "entity_pages",
                             "parameters": {"entity": ent_name}, "limit": 3})
            s.record("POST /graph/query (entity_pages)",
                     r.status_code == 200 and ok(r.json()))
            r = c.post("/graph/explore", headers=H,
                       json={"entity_type": "material", "entity_name": ent_name,
                             "depth": 1, "limit": 3})
            s.record("POST /graph/explore", r.status_code == 200 and ok(r.json()),
                     f"-> {r.status_code}")
        else:
            s.record("POST /graph/query + explore", False, "no material entities")

        r = c.get("/graph/communities?limit=3", headers=H)
        s.record("GET /graph/communities", r.status_code == 200 and ok(r.json()),
                 f"-> {r.status_code}")

        # ------------------------------------------------------------------
        # Jobs / schedule / system
        # ------------------------------------------------------------------
        r = c.get("/ingest/jobs?limit=3", headers=H)
        s.record("GET /ingest/jobs", r.status_code == 200 and ok(r.json()))
        r = c.get("/ingest/jobs/controls", headers=H)
        s.record("GET /ingest/jobs/controls", r.status_code == 200 and ok(r.json()))
        r = c.get("/schedule", headers=H)
        s.record("GET /schedule", r.status_code == 200 and ok(r.json()))
        r = c.get("/system/gpu", headers=H)
        s.record("GET /system/gpu", r.status_code == 200 and ok(r.json()))

        # ------------------------------------------------------------------
        # Admin read-only checks
        # ------------------------------------------------------------------
        r = c.get("/admin/audit/completeness", headers=H, timeout=300.0)
        s.record("GET /admin/audit/completeness", r.status_code == 200 and ok(r.json()),
                 f"-> {r.status_code}, {len(r.json().get('data') or {}).__class__.__name__}")
        r = c.get("/admin/backup/manifest", headers=H)
        s.record("GET /admin/backup/manifest", r.status_code == 200 and ok(r.json()))
        if not args.skip_verify:
            r = c.get("/admin/verify", headers=H, timeout=600.0)
            body = r.json()
            vdata = body.get("data") or {}
            violations = 0
            if isinstance(vdata, dict):
                vlist = vdata.get("violations") or []
                violations = vlist if isinstance(vlist, int) else len(vlist)
            s.record("GET /admin/verify (deep)", ok(body),
                     f"violations={violations} -> {r.status_code}")

        # ------------------------------------------------------------------
        # A document's pages + page text (the "learn from a page" path)
        # ------------------------------------------------------------------
        if doc and docs:
            doc_id = docs[0]["doc_id"]
            r = c.get(f"/documents/{doc_id}/pages?limit=2", headers=H)
            pages = r.json().get("data")
            if isinstance(pages, dict):
                items = pages.get("pages") or []
            elif isinstance(pages, list):
                items = pages
            else:
                items = []
            s.record("GET /documents/{id}/pages", ok(r.json()) and len(items) > 0,
                     f"{len(items)} pages")
            if items:
                pn = items[0].get("page_number")
                r = c.get(f"/documents/{doc_id}/pages/{pn}", headers=H)
                pg = r.json().get("data") or {}
                text = pg.get("extracted_text") or pg.get("text") or ""
                s.record("GET /documents/{id}/pages/{n} returns text",
                         r.status_code == 200 and ok(r.json()) and len(text) > 0,
                         f"-> {r.status_code}, {len(text)} chars")
        else:
            s.record("document pages/text", False, "no document to inspect")
    finally:
        c.close()
        a.close()

    if args.json is not None:
        args.json.write_text(json.dumps({
            "base": base, "auth_base": auth_base,
            "checks": s.checks, "passed": len(s.checks) - s.failed,
            "total": len(s.checks),
        }, indent=2))
        print(f"\nReport written to {args.json}")
    return s.finish()


if __name__ == "__main__":
    sys.exit(main())
