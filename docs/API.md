# ForgeRAG HTTP API — Full Guide

ForgeRAG ships a complete JSON HTTP API on top of its web GUI. Anything the
UI can do, an agent can do over the network: search the library, ask
questions, read document pages, explore the knowledge graph, ingest PDFs,
monitor jobs, and run maintenance. The API is what the browser GUI is built
on — every button in the UI is a call to one of these endpoints.

This guide is written for external clients: scripts, agent harnesses, and
other devices on your network (e.g. the Raspberry Pi harness that gives a
learning agent read access to the library). It covers authentication, the
response format, every endpoint group, worked examples, and how to verify
the API yourself.

- [Quickstart](#quickstart)
- [Base URL & network](#base-url--network)
- [Authentication](#authentication)
- [Response envelope](#response-envelope)
- [Discovering the API](#discovering-the-api)
- [Endpoint reference](#endpoint-reference)
- [Agent cookbook](#agent-cookbook)
- [Error handling](#error-handling)
- [Verifying the API](#verifying-the-api)
- [Security notes](#security-notes)

---

## Quickstart

```bash
# The server listens on 0.0.0.0:8200 on the ForgeRAG machine.
# From the LAN, use its IP; from the box itself, localhost works.

# 1. Liveness + library stats (no token required)
curl http://192.168.1.23:8200/health

# 2. Everything else needs a bearer token when called from a remote host
TOKEN="paste-the-api-token-here"

# 3. Discover what the library can do (live stats, capability manifest)
curl -H "Authorization: Bearer $TOKEN" http://192.168.1.23:8200/skills/manifest

# 4. Ask a question — retrieval + an LLM-written answer with page citations
curl -X POST http://192.168.1.23:8200/search/answer \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"query":"What is alloy C12000 used for and how do I weld it?","limit":3}'

# 5. Full endpoint list is served by the API itself
curl -H "Authorization: Bearer $TOKEN" http://192.168.1.23:8200/openapi.json
```

---

## Base URL & network

| Thing | Value |
|-------|-------|
| Bind address | `0.0.0.0` (all interfaces) |
| Port | `8200` (set `[server] port` in `config/forgerag.toml`) |
| LAN URL | `http://<server-lan-ip>:8200` |
| Local URL | `http://localhost:8200` (and `http://127.0.0.1:8200`) |

- The server binds all interfaces, so any device on your LAN can reach it.
  Keep it on a trusted network; see [Security notes](#security-notes).
- Requests from the ForgeRAG machine itself (loopback) are treated as
  trusted and need no token. Anything arriving on a network interface must
  authenticate. This is exactly the path a Raspberry Pi on the LAN takes.
- **Firewall**: if a client can't connect, allow inbound TCP 8200 (e.g.
  `sudo ufw allow from 192.168.1.0/24 to any port 8200 proto tcp`).
- **`/images/...` URLs in search results are relative paths** (`/images/…`).
  Resolve them against the same base URL you called. See
  [Image URLs](#images--page-images).
- Optional remote-path hardening: put both machines on a Tailscale tailnet
  and point clients at the private `100.x.y.z:8200` address instead of the
  raw LAN, so the token never travels beyond the tailnet.

---

## Authentication

ForgeRAG uses static bearer tokens. There are two, with different scopes:

| Token | Config key / env var | Scope |
|-------|----------------------|-------|
| Admin | `[server] api_token` / `FORGERAG_API_TOKEN` | Everything: search, ingest, delete, rebuild, backup/restore, admin |
| Read-only *(optional)* | `[server] api_token_readonly` / `FORGERAG_API_TOKEN_READONLY` | Search, read documents/pages, explore the graph. **Cannot modify anything.** |

For a learning agent you almost certainly want the **read-only** token: it can
learn from the whole library without ever being able to delete, re-ingest, or
modify anything.

### How auth works

- No token sent / wrong token → **401** `{"success": false, "reason": "unauthorized"}`.
- Read-only token on an operation it can't do → **403**
  `{"success": false, "reason": "readonly-token: write operation not permitted"}`.
- Exempt from auth entirely: `GET /`, `GET /health`, and the `/app*` SPA
  shell (the UI itself still calls authenticated endpoints).
- OPTIONS preflight requests pass (browser CORS).
- Comparisons are constant-time (`secrets.compare_digest`).

### Read-only token scope — exactly what it may call

| Allowed | Detail |
|---------|--------|
| All `GET` | Every read endpoint: documents, collections, categories, tags, pages, jobs, schedule, system/gpu, graph stats/entities/communities, skills/manifest, admin audit/verify/manifest/backup-list/progress/settings/restore-status |
| `POST /search/*` | All search modes (`keyword`, `semantic`, `hybrid`, `chunks`, `visual`, `summaries`, `answer`) — read-only by design |
| `POST /skills/*` | Unified auto-routing search and batch search |
| `POST /graph/query`, `POST /graph/explore` | Predefined graph queries and neighborhood exploration |
| `POST /ingest/check-duplicates` | Look up which SHA-256 hashes already exist |

Everything else — ingest, rebuild, re-embed, delete, pause/resume/cancel,
settings, schedule updates, backup/restore, admin repair jobs — returns 403
for the read-only token and requires the admin token.

### Setting the tokens

```bash
# Generate tokens on the ForgeRAG machine
openssl rand -hex 24          # -> paste into forgerag.toml and restart
```

```toml
# config/forgerag.toml
[server]
api_token = "<24-hex-bytes admin>"
api_token_readonly = "<24-hex-bytes read-only>"
```

Restart the service after changing the file:

```bash
sudo systemctl restart forgerag-api
```

Environment variables override the file (useful for systemd
`EnvironmentFile=` setups, e.g. under `/etc/forgerag/env`):
`FORGERAG_API_TOKEN`, `FORGERAG_API_TOKEN_READONLY`.

### Auth in curl

```bash
curl -H "Authorization: Bearer $TOKEN" http://192.168.1.23:8200/documents
```

---

## Response envelope

Every endpoint returns the same envelope:

| Field | Type | Meaning |
|-------|------|---------|
| `success` | bool | `true` on success, `false` on failure |
| `reason` | string \| null | Human-readable error message when `success` is `false` |
| `data` | dict \| list \| null | The payload when `success` is `true` |

```json
{
  "success": true,
  "reason": null,
  "data": { ... }
}
```

- Success carries `success: true` and the payload in `data`.
- Failures carry `success: false` and a `reason` string (e.g.
  `"unauthorized"`, `"readonly-token: write operation not permitted"`,
  `"Document not found"`, an LLM-unavailable diagnostic). HTTP status is
  set accordingly (401/403/404/503) — agents should branch on `success` and
  read `reason` for the message.
- A small number of endpoints return `data` as a **list** (search hits,
  collections, graph/explore); most return a dict; some admin endpoints
  return counts. Check the endpoint reference for the exact shape.

---

## Discovering the API

| Endpoint | What you get |
|----------|--------------|
| `GET /openapi.json` | Machine-readable OpenAPI 3.1 schema for **all 86 routes** with request/response models. Requires a token from a remote host. |
| `GET /docs` | Swagger UI for humans. Requires a token from a remote host. |
| `GET /skills/manifest` | Capability manifest with live library stats (documents, pages, entities, communities) plus the search endpoints and their parameter names. |

An agent should start from `/openapi.json`: it is always in sync with the
code, whereas this document is a curated snapshot.

---

## Endpoint reference

Legend for the **Access** column:
- **RO** — callable with the read-only token (safe/read-only).
- **Admin** — requires the admin token (writes or maintenance).

### Health & system

| Method | Path | Access | Description |
|--------|------|--------|-------------|
| GET | `/` | exempt | Alias of `/health`. |
| GET | `/health` | exempt | Liveness: Neo4j connectivity, document/page counts, GPU status, LLM circuit breaker, restore flag. |
| GET | `/system/gpu` | RO | VRAM usage + currently loaded models. |
| POST | `/system/models/{name}/unload` | Admin | Force-unload a loaded model to free VRAM. `name` = model key (`text_embedding`, `reranker`, `visual_embed`). |

```bash
curl http://192.168.1.23:8200/health
```

```json
{
  "success": true,
  "data": {
    "status": "ok",
    "service": "forgerag",
    "version": "0.1.0",
    "neo4j_connected": true,
    "document_count": 541,
    "page_count": 117030,
    "gpu_available": true,
    "details": {
      "neo4j_healthy": true,
      "llm_circuit_breaker": {"state": "closed", "consecutive_failures": 0, "is_open": false},
      "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
      "vram_total_gb": 101.9,
      "vram_free_gb": 36.9
    }
  }
}
```

---

### Skills (agent-facing discovery & unified search)

These are the endpoints an agent should use first: the manifest advertises
capabilities, and `/skills/search` picks the right strategy automatically so
you don't have to know the retrieval internals.

| Method | Path | Access | Description |
|--------|------|--------|-------------|
| GET | `/skills/manifest` | RO | Capability advertisement + live stats. |
| POST | `/skills/search` | RO | Unified search. Auto-routes by query shape, or force `mode`. |
| POST | `/skills/batch` | RO | Run up to 20 queries in parallel. |

**`POST /skills/search` body:**

```json
{
  "query": "What is alloy C12000 used for and how do I weld it?",
  "mode": null,          // optional: "keyword" | "answer" | "hybrid"
  "limit": 5,            // 1..20
  "filters": null        // optional: {collection, categories, tags, document_ids, source_type}
}
```

Auto-routing: short queries that look like a code/standard identifier
(e.g. `SA-516.70`, `ASTM A36`) → keyword; queries starting with a question
word → answer; anything else → hybrid/RRF.

```bash
curl -X POST http://192.168.1.23:8200/skills/search \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"query":"SA-516.70","limit":3}'
```

**`GET /skills/manifest`** returns the capability list and stats:

```json
{
  "success": true,
  "data": {
    "name": "forgerag",
    "version": "0.1.0",
    "capabilities": [
      {"name": "search_answer", "endpoint": "/search/answer", "method": "POST",
       "description": "Retrieve pages and synthesize an LLM answer with citations",
       "params": ["query", "limit", "search_mode", "use_graph", "use_vision"]},
      ...
    ],
    "stats": {"documents": 541, "pages": 117030, "entities": 206799, "communities": 432}
  }
}
```

---

### Search

All search endpoints return hits in `data` as a **list**. Hits share a common
core shape (fields may differ slightly per mode):

```json
{
  "page_id": "63f2be26-…",
  "doc_id": "1273b439-…",
  "document_title": "Steel Structure Design",
  "filename": "Steel Structure Design.pdf",
  "page_number": 638,
  "score": 4.71,
  "text_snippet": "…introductory text…",
  "image_url": "/images/6f9f226b…/638",
  "reduced_image_url": "/images/6f9f226b…/638/reduced",
  "categories": ["Welding Codes", "…"],
  "tags": ["AWS", "welding", "…"]
}
```

`image_url` / `reduced_image_url` are **relative** — prefix them with your
base URL to fetch the page image (or OCR proof), e.g.
`http://192.168.1.23:8200` + `/images/…`.

| Method | Path | Access | Mode & best for |
|--------|------|--------|-----------------|
| POST | `/search/keyword` | RO | Lucene full-text on extracted page text with fuzzy-tolerance option. Exact alloy codes, clause IDs, standard numbers. |
| POST | `/search/semantic` | RO | BGE-M3 vector search over chunk text. Meaning/paraphrase search. |
| POST | `/search/hybrid` | RO | Vector + knowledge-graph. Strategies: `rrf` (default), `graph_boosted`, `graph_first`, `vector_first`, `community`. Optional `rerank`. |
| POST | `/search/chunks` | RO | Chunk-level retrieval (dense + BM25 + reranker). Returns `chunk_id`, `chunk_type`, `section_path`, `summary`, `text_snippet`. |
| POST | `/search/visual` | RO | Two-stage Nemotron/ColPali visual retrieval — searches page *images* (diagrams, tables, figures). |
| POST | `/search/summaries` | RO | RAPTOR-by-TOC summary search at section/chapter/document abstraction levels. |
| POST | `/search/answer` | RO | Full RAG: retrieve pages, then the VLM reads them and writes a cited answer. |

All except `/search/answer` are fast (~100–400 ms once models are warm). The
first call after a model idles out reloads it (a few seconds); `/search/answer`
loads the vision LLM and reads pages, so allow 10–60 s. See
[Model lifecycle](models-not-loaded) note.

**`POST /search/answer` body:**

```json
{
  "query": "What is alloy C12000 used for and how do I weld it?",
  "limit": 5,                        // pages to retrieve and read, 1..20
  "search_mode": "hybrid",           // "keyword" | "hybrid" | "visual"
  "use_graph": true,                 // include graph reasoning context
  "use_vision": true                 // read page images for figures/tables
}
```

**Response** — `data` is a dict:

```json
{
  "success": true,
  "data": {
    "query": "What is alloy C12000 used for and how do I weld it?",
    "search_mode": "keyword",
    "answer": "Based on the provided engineering handbook pages…C12000 is a phosphorus-deoxidized copper…",
    "sources": [
      {"page_number": 1887, "document_title": "ASM_Handbook_Vol_06…",
       "image_url": "/images/6f9f226b…/1887", "score": 1.0},
      ...
    ],
    "used_vision": true,
    "used_graph": true,
    "graph_context": {
      "materials_found": 13, "processes_found": 2, "standards_found": 3,
      "reasoning_chains": ["GTAW (Process) → GOVERNED_BY → AWS D1.1 (Standard) → …"]
    }
  }
}
```

The `sources` array is the citation list — each item's `image_url` points at
the page the model actually read. `graph_context` shows the entity graph
chains that influenced the answer (when `use_graph: true`).

---

### Documents

#### Reads (RO)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/collections` | All collections with document counts and page totals. `data` = list of `{collection, document_count, total_pages}`. |
| GET | `/documents` | List with filters: `collection`, `category`, `tag`, `source_type`, `limit` (default 20, max 100), `offset`. `data` = `{documents: [...], total: N}`. |
| GET | `/documents/{doc_id}` | Full metadata for one document. |
| GET | `/documents/{doc_id}/pages` | Page index. `data` = list of `{page_id, page_number, text_char_count, source_type, image_path, reduced_image_path}`. |
| GET | `/documents/{doc_id}/pages/{n}` | Page detail **including full `extracted_text`** — the "read a page" endpoint. |
| GET | `/categories` | Category tree. |
| GET | `/tags` | All tags with document counts. |

`GET /documents` row shape:

```json
{
  "doc_id": "58c483c1-…",
  "title": "2605.23904v2",
  "filename": "2605.23904v2.pdf",
  "file_hash": "87f7f0f3…",
  "page_count": 27,
  "file_size_bytes": 879864,
  "source_type": "digital_native",
  "collection": "artificial_intelligence",
  "ingested_at": "2026-08-12T21:43:04Z",
  "categories": ["Large Language Models", "Agent Systems"],
  "tags": ["skill-optimization", "benchmarking", "…"]
}
```

The "learn from a page" flow:

```bash
# 1. Find pages about a topic
curl -X POST http://192.168.1.23:8200/search/keyword \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"query":"C12000","limit":1}' | python3 -m json.tool

# 2. Take doc_id + page_number from a hit and read the page text
curl -H "Authorization: Bearer $TOKEN" \
  http://192.168.1.23:8200/documents/1273b439-…/pages/638 | python3 -m json.tool
```

#### Writes (Admin)

| Method | Path | Description |
|--------|------|-------------|
| PUT | `/documents/{doc_id}/collection?collection=…` | Move a document between collections (metadata only; no re-ingest). |
| POST | `/documents/{doc_id}/tags` | Add a tag. Body `{name}`. |
| DELETE | `/documents/{doc_id}/tags/{tag}` | Remove a tag. |
| POST | `/documents/{doc_id}/categories` | Add a category. Body `{name, parent_name?, description?}`. |
| DELETE | `/documents/{doc_id}/categories/{cat}` | Remove a category. |
| POST | `/documents/{doc_id}/suggest-tags` | LLM proposes `{collection, categories, tags}` from the doc's content (read-only; **returns** the suggestion without applying). |
| POST | `/documents/{doc_id}/apply-tags` | Apply tags/categories/collection. Body `{collection?, categories?, tags?, mode: "merge"\|"replace"}`. |
| POST | `/documents/{doc_id}/extract-entities` | Re-run LLM entity extraction for this document (background job). |
| POST | `/documents/{doc_id}/rebuild-chunks` | Phase-9 rebuild: chunks + summaries + embeddings + entity re-extraction. Query params `extract_only`, `skip_extract`. |
| POST | `/documents/{doc_id}/reembed` | Re-run text + visual embeddings (background job). |
| DELETE | `/documents/{doc_id}` | Delete the document and everything derived from it (pages, chunks, images, entities). |
| POST | `/categories` | Create/update a category. Body `{name, parent_name?, description?}`. |
| DELETE | `/categories/{name}` | Delete a category. |
| POST | `/tags` | Create a tag. Body `{name}`. |
| DELETE | `/tags/{name}` | Delete a tag. |

---

### Images & page images

| Method | Path | Access | Description |
|--------|------|--------|-------------|
| GET | `/images/{doc_hash}/{page_number}` | RO | Full-resolution PNG of a page. |
| GET | `/images/{doc_hash}/{page_number}/reduced` | RO | Reduced JPG thumbnail (fast; used in the UI). |
| GET | `/images/{doc_hash}/{page_number}/highlighted` | RO | Query-time highlighted image (requires an active highlight session; see `/search/visual` flow in the UI). |

`doc_hash` is the document's SHA-256 `file_hash` (returned on every search
hit and document row). Image URLs come back **relative** (`/images/…`) — a
remote client must prefix its base URL:

```bash
curl -o /tmp/page.jpg -H "Authorization: Bearer $TOKEN" \
  http://192.168.1.23:8200/images/6f9f226b…/1887/reduced
```

---

### Knowledge graph

| Method | Path | Access | Description |
|--------|------|--------|-------------|
| GET | `/graph/stats` | RO | Per-label node counts (`data` = `{labels: [{label, count}...]}`). |
| GET | `/graph/entities/{type}` | RO | List entities of a type. Type one of `material`, `process`, `standard`, `equipment`, `clause`. `data` = list of `{key, page_mentions, properties}`. |
| GET | `/graph/communities` | RO | GraphRAG communities. Query: `level` (0–3), `limit`, `search`. |
| POST | `/graph/query` | RO | Run a predefined query template. |
| POST | `/graph/explore` | RO | N-hop neighborhood of an entity (`depth` 1–3). `data` = list of connected nodes/edges. |
| POST | `/graph/build-communities` | Admin | Rebuild the hierarchical community layer (Leiden + LLM summaries). Heavy background job. |

**`POST /graph/query`** — templates:

`material_standards`, `process_materials`, `standard_cross_references`,
`material_properties`, `equipment_requirements`, `page_entities`,
`entity_pages`

```bash
curl -X POST http://192.168.1.23:8200/graph/query \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"query_type":"entity_pages","parameters":{"entity":"C12000"},"limit":10}'
```

**`POST /graph/explore`** body:

```json
{
  "entity_type": "material",
  "entity_name": "steel",
  "depth": 1,
  "limit": 50
}
```

Templates and entity lookup are alias-aware: after canonicalization,
"4140 steel" lives in the common-names list of the `4140` node, so you can
query by any historical spelling.

---

### Ingestion

| Method | Path | Access | Description |
|--------|------|--------|-------------|
| POST | `/ingest` | Admin | Upload a PDF (multipart form: `file` required; `collection`, `priority` optional). Starts a background job; returns `{job_id, doc_id}`. |
| POST | `/ingest/check-duplicates` | RO | Body `{file_hashes: [...]}` → which hashes already exist as documents. |
| GET | `/ingest/jobs` | RO | List jobs. Query `status`: a concrete status, `active`, or `terminal`. |
| GET | `/ingest/jobs/{job_id}` | RO | Poll one job: status, phase, pages processed, per-step status ledger, live `current_item`, `error_message`. |
| GET | `/ingest/jobs/{job_id}/logs` | RO | Captured log lines for a job (live-tails running jobs). |
| GET | `/ingest/jobs/controls` | RO | Global control state: `{pause_all, counts, active}`. |
| POST | `/ingest/jobs/pause-all` | Admin | Pause every running/queued job (the "free the GPU" switch; persists across restarts). |
| POST | `/ingest/jobs/resume-all` | Admin | Clear the global pause and all per-job pauses. |
| POST | `/ingest/jobs/{job_id}/pause` | Admin | Pause one job at its next checkpoint. |
| POST | `/ingest/jobs/{job_id}/resume` | Admin | Resume one paused job (stays held if pause-all is on). |
| POST | `/ingest/jobs/{job_id}/cancel` | Admin | Stop a job (queued stop immediately; running ones finish the current page/batch). |
| POST | `/ingest/jobs/{job_id}/restart` | Admin | Re-launch a finished job as a new job (re-checks what's missing). |

**Upload example** (multipart/form-data):

```bash
curl -X POST http://192.168.1.23:8200/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@manual.pdf" \
  -F "collection=mechanical_design" \
  -F "priority=true"
```

```json
{"success": true, "data": {"job_id": "job_…", "doc_id": "…"}}
```

**Polling a job:**

```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://192.168.1.23:8200/ingest/jobs/job_… | python3 -m json.tool
```

`data` includes `status` (`queued|processing|completed|failed|cancelled`),
`doc_id`, the step ledger `steps` (each `{name, status: done|running|failed|skipped, detail}`),
and errors. Terminal states are `completed`, `failed`, `cancelled`. Pause
and stop are cooperative — nothing is ever left half-written, and repair
jobs recompute what's missing when restarted.

---

### Schedule & automation

| Method | Path | Access | Description |
|--------|------|--------|-------------|
| GET | `/schedule` | RO | Current processing-window + watch-folder config and live status (window open/closed, next boundary, inbox counts, recent events). |
| PUT | `/schedule` | Admin | Set the processing window. Body `{enabled, start "HH:MM", end "HH:MM", days [0-6, Mon=0]}`. Overnight windows supported. |
| PUT | `/schedule/watch` | Admin | Configure the auto-ingest inbox. Body `{enabled, path, collection}` (empty path → default inbox). |
| POST | `/schedule/watch/scan-now` | Admin | Scan the inbox immediately. |
| GET | `/schedule/browse` | RO | List server-side directories (backs the GUI folder picker). Query `path`. |
| POST | `/schedule/watch/open-folder` | Admin | Open the inbox in the file manager on the ForgeRAG machine. |

---

### Admin maintenance (Admin token)

These repair/cleanup jobs are long-running background work. Use them through
the UI unless you know what you're doing; the deep-verification endpoints
are read-only and safe.

| Method | Path | Reads / Writes | Description |
|--------|------|----------------|-------------|
| GET | `/admin/audit/completeness` | read | Which pipeline step is missing per document, from graph state. A 100k-page library audits in under ten seconds. |
| GET | `/admin/verify` | read | Deep verification: ~30 exact-count integrity checks (images on disk, embedding dims, blob integrity, duplicates/orphans, entity hygiene, index health). PASS requires zero violations. |
| GET | `/admin/step-issues` | read | Recent step failures (`?days=N`). |
| GET | `/admin/backup/manifest` | read | Document manifest for backup verification. |
| POST | `/admin/extract-missing-entities` | write | Queue entity extraction for every doc with unextracted text pages. |
| POST | `/admin/build-missing-summaries` | write | Queue RAPTOR-by-TOC summary trees for chunked docs without one. Body `{doc_ids?}`. |
| POST | `/admin/build-intermediate-levels` | write | Intermediate summary levels for wide/flat trees. |
| POST | `/admin/resummarize-fallbacks` | write | Regenerate chunk summaries that fell back to text previews + re-embed. |
| POST | `/admin/autotag-missing` | write | Auto-tag every unorganized document (default collection, no tags/categories). |
| POST | `/admin/recover-stranded-text` | write | OCR text recovery for pages whose text exists only in chunks. |
| POST | `/admin/backfill-blank-flags` | write | Compute `is_blank` on pages missing it. |
| POST | `/admin/reextract-suspicious-empties` | write | Re-check dense pages stamped extracted-with-nothing. |
| POST | `/admin/fill-missing` | write | Incremental gap-filling. Body `{doc_ids, text?, visual?, entities?, recover_text?, priority?}`. `priority: true` = run-now lane. |
| POST | `/admin/normalize-entities` | write | Merge duplicate entities differing only by case/whitespace. |
| POST | `/admin/bulk-reembed` | write | Queue re-embed jobs for every document. |
| POST | `/admin/reembed-text` | write | Text-only re-embed. Body `{doc_id?}`. |
| POST | `/admin/rebuild-chunks-bulk` | write | Queue chunk rebuilds for `{doc_ids, extract_only?, skip_extract?, only_missing?}`. |
| POST | `/admin/cleanup-uploads` | write | Delete staged upload files. |
| POST | `/admin/dedup-pages` | write | Remove duplicate `:Page` nodes per `(doc_id, page_number)`. |
| POST | `/admin/purge-orphan-summaries` | write | Delete `:SectionSummary` nodes whose document no longer exists. |

### Backup & restore (Admin token)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/admin/backup` | Hot graph export (JSON, all metadata, no embeddings). |
| POST | `/admin/backup/full` | Full backup: Neo4j dump + images + PDFs + graph JSON + optional Drive upload. |
| GET | `/admin/backup/progress` | Backup progress (`running, percent, current_file, bytes_copied`). |
| GET | `/admin/backup/list` | Available backups from local + destination directories. |
| GET | `/admin/backup/settings` | Current backup configuration. |
| POST | `/admin/backup/settings` | Update configuration. Body `{destination, include_images, include_pdfs, gdrive_enabled, gdrive_dump}`. |
| GET | `/admin/restore/status` | Whether the DB is empty and a restore is needed; lists local backups. |
| POST | `/admin/restore` | Returns the CLI commands for restoring from a local dump or Drive. |

---

## Agent cookbook

### 1. Bootstrap — the first three calls

```bash
BASE=http://192.168.1.23:8200
TOKEN="…"

# Health + what's in the library
curl -s $BASE/health

# Full schema = the contract
curl -s -H "Authorization: Bearer $TOKEN" $BASE/openapi.json > /tmp/forgerag.json

# Capability manifest + live stats
curl -s -H "Authorization: Bearer $TOKEN" $BASE/skills/manifest
```

### 2. Search, step by step

```bash
# Unified (auto-routed) — start here, don't micro-manage modes
curl -X POST $BASE/skills/search -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"How is cast iron welded?","limit":5}'

# Exact identifiers -> keyword
curl -X POST $BASE/search/keyword -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"ASTM A36","limit":10}'

# Meaning/paraphrase -> semantic
curl -X POST $BASE/search/semantic -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"fatigue life of welded joints","limit":10}'

# Need a synthesized, cited answer -> answer
curl -X POST $BASE/search/answer -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"What preheat is required for GTAW of 4140?","limit":5}'
```

### 3. Follow a hit back to the source page text

```bash
# Save the first hit, grab doc_id + page_number
hit=$(curl -s -X POST $BASE/search/keyword -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{"query":"C12000","limit":1}')
did=$(echo "$hit" | jq -r '.data[0].doc_id')
pn=$(echo "$hit" | jq -r '.data[0].page_number')

# Full extracted text of that page — this is what the model reads
curl -s -H "Authorization: Bearer $TOKEN" $BASE/documents/$did/pages/$pn \
  | jq -r '.data.extracted_text'

# Or fetch the page image via the relative image_url
curl -s -o page.jpg -H "Authorization: Bearer $TOKEN" \
  $BASE$(echo "$hit" | jq -r '.data[0].reduced_image_url')
```

### 4. Explore the knowledge graph

```bash
# What standards govern material C12000?
curl -X POST $BASE/graph/query -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query_type":"material_standards","parameters":{"material":"C12000"},"limit":20}'

# 1-hop neighborhood
curl -X POST $BASE/graph/explore -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"entity_type":"material","entity_name":"C12000","depth":1,"limit":20}'

# Per-label stats
curl -s -H "Authorization: Bearer $TOKEN" $BASE/graph/stats | jq '.data.labels'
```

### 5. Drive an ingestion + watch it complete

```bash
curl -X POST $BASE/ingest -H "Authorization: Bearer $TOKEN" \
  -F "file=@manual.pdf" -F "collection=mechanical_design" \
  | jq .   # -> {job_id, doc_id}

# Poll until terminal
while true; do
  st=$(curl -s -H "Authorization: Bearer $TOKEN" $BASE/ingest/jobs/$JOB | jq -r .data.status)
  echo "status=$st"; [ "$st" = completed ] && break; [ "$st" = failed ] && break
  sleep 5
done
```

### 6. Client code examples

**Python (`httpx`):**

```python
import httpx

BASE = "http://192.168.1.23:8200"
HEADERS = {"Authorization": "Bearer <token>"}
c = httpx.Client(base_url=BASE, headers=HEADERS, timeout=60)

health = c.get("/health").json()["data"]
print(f"{health['document_count']} documents, {health['page_count']} pages")

ans = c.post("/search/answer", json={"query": "How do I weld C12000?",
                                     "limit": 3}).json()["data"]
print(ans["answer"])
for s in ans["sources"]:
    print(f"  p.{s['page_number']} {s['document_title']} {s['image_url']}")
```

**Node (`fetch`):**

```js
const BASE = "http://192.168.1.23:8200";
const HEADERS = { Authorization: "Bearer <token>", "Content-Type": "application/json" };

const health = await (await fetch(`${BASE}/health`)).json();
const hits = await (await fetch(`${BASE}/search/semantic`, {
  method: "POST", headers: HEADERS,
  body: JSON.stringify({ query: "welding distortion control", limit: 5 }),
})).json();
console.log(hits.data[0]?.document_title, hits.data[0]?.page_number);
```

### 7. Verifying from a headless device (the Pi)

```bash
TOKEN="<read-only-token>" BASE=http://192.168.1.23:8200

# The read-only token proves it can learn but not break things:
curl -H "Authorization: Bearer $TOKEN" $BASE/documents          # 200
curl -X POST $BASE/search/answer -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{"query":"…","limit":3}'   # 200
curl -X POST $BASE/ingest -H "Authorization: Bearer $TOKEN" -F "file=@x.pdf" # 403
```

---

## Error handling

| HTTP | `reason` (examples) | Meaning |
|------|---------------------|---------|
| 401 | `unauthorized` | No/invalid bearer token (remote client). |
| 403 | `readonly-token: write operation not permitted` | A read-only token tried to mutate state. |
| 404 | descriptive | Resource not found (doc, page, job, entity). |
| 400 | descriptive | Bad request body / unknown template / invalid params. |
| 422 | Pydantic detail | Request body failed schema validation. |
| 503 | descriptive | A required dependency is unavailable (e.g. LLM endpoint down for suggest-tags/answer). |
| 200-with-`success:false` | descriptive | Endpoint handled the failure itself (e.g. `suggest-tags` on a doc with no usable text, `rebuild` job not enqueued). |

Always branch on `success` (and inspect `reason`), not just HTTP status.

---

## Verifying the API

Two committed scripts drive the real system:

```bash
# Full-surface API smoke — auth (admin + optional read-only), every read /
# search / graph / skills / documents endpoint, jobs, schedule, system,
# images, deep verify. Strictly read-only. Safe to run anytime.
./venv/bin/python scripts/api_smoke.py --base http://127.0.0.1:8200 \
    --token "$TOKEN" --readonly-token "$RO_TOKEN"

# Deeper pipeline E2E — actually ingests a synthetic PDF through the whole
# pipeline, then searches for it and deletes it (uses GPU + LLM briefly).
./venv/bin/python scripts/smoke_e2e.py
```

`api_smoke.py` runs auth checks through a non-loopback address (it derives
your LAN IP when `--base` is localhost) so the exact remote path is
exercised — the same path a Raspberry Pi client takes.

---

## Security notes

- This is a single-human instrument on a trusted LAN. The static-token model
  is intentional; OAuth/roles are out of scope by design.
- **Give remote/learning agents the read-only token, never the admin token.**
  It can search and read the entire library but cannot delete, re-ingest,
  back up, or run maintenance.
- Tokens travel in the `Authorization` header, which on a plain-HTTP LAN is
  visible to anything sniffing the segment. Keep the API on a trusted
  network (home LAN or a Tailscale tailnet); don't expose port 8200 to the
  open internet.
- Rotate tokens by editing `config/forgerag.toml` (or the env file) and
  restarting the service.
- The web GUI, on-box scripts, and monitoring get free localhost access —
  only remote clients need tokens.
