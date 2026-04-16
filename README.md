# ForgeRAG

Local engineering knowledge graph for processing and querying large corpora of engineering PDFs
(ASM handbooks, ASME/ASTM/API/AWS codes, process manuals, company specs).

Designed to serve both a web GUI for direct research and a team of AI agents ("Chooms") via a
skill module. Runs entirely on local hardware — no cloud APIs.

## Status

**Phase 1 complete** — foundation scaffolding in place.

- [x] Project structure and Python packaging
- [x] FastAPI service with standardized response envelope
- [x] Neo4j Community 5.x installed with schema (constraints, indexes, vector indexes)
- [x] Health endpoint reporting service + Neo4j + GPU status
- [x] systemd unit file ready for boot-time startup
- [ ] Phase 2: PDF ingestion pipeline (upload → page images → text → Neo4j)
- [ ] Phase 3: ColPali visual embeddings + text embeddings + search
- [ ] Phase 4: LLM entity/relationship extraction + graph queries
- [ ] Phase 5: Community detection + hybrid search + page highlighting
- [ ] Phase 6: React/Vite frontend
- [ ] Phase 7: Choom skill integration
- [ ] Phase 8: Bulk ingestion + polish

See `/home/nuc1/.claude/plans/staged-zooming-cascade.md` for the full plan.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   React/Vite GUI                      │
├──────────────────────────────────────────────────────┤
│              FastAPI REST API (:8200)                  │
├───────────┬──────────────┬───────────────────────────┤
│  Neo4j    │  ColPali     │  Page Image Store         │
│  Graph +  │  Visual      │  (PNGs on disk)           │
│  Vector   │  (in Python) │                           │
└───────────┴──────────────┴───────────────────────────┘
```

Hardware:
- **NUC i7 (96GB DDR5)** — Neo4j, FastAPI, frontend, page images
- **RTX PRO 6000 (96GB VRAM)** — ColPali, text embeddings, VLM OCR, 70B entity extraction LLM
- Flux/Forge coexists on the RTX 6000 for image generation (~20-30GB when active)

## Setup (from a fresh clone)

### 1. Install Neo4j Community 5.x

```bash
cd /home/nuc1/projects/ForgeRAG
./scripts/install_neo4j.sh
```

The script installs Neo4j, locks it to localhost, and enables the systemd service.

### 2. Change the default password

Neo4j starts with `neo4j` / `neo4j`. Change it:

```bash
cypher-shell -u neo4j -p neo4j -d system \
    "ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO 'YOUR_STRONG_PASSWORD'"
```

> **Note:** Community Edition only supports the default `neo4j` and `system` databases.
> ForgeRAG uses the default `neo4j` database — no custom database creation needed.
>
> **Note:** cypher-shell may print Java version warnings on newer JVMs. These are harmless
> and do not affect operation.

### 3. Create the Python venv

```bash
python3 -m venv venv
./venv/bin/pip install --upgrade pip setuptools wheel
./venv/bin/pip install -r requirements.txt
```

### 4. Copy config template

```bash
cp config/forgerag.toml.example config/forgerag.toml
```

Edit `config/forgerag.toml` if you need to change ports, paths, or model names.

### 5. Export the Neo4j password

For the current shell:
```bash
export NEO4J_PASSWORD='YOUR_STRONG_PASSWORD'
```

To run as a systemd service later:
```bash
sudo mkdir -p /etc/forgerag
echo "NEO4J_PASSWORD='YOUR_STRONG_PASSWORD'" | sudo tee /etc/forgerag/env > /dev/null
sudo chmod 600 /etc/forgerag/env
```

### 6. Seed the Neo4j schema

```bash
./venv/bin/python scripts/seed_schema.py
```

Expected output:
```
Schema seeding complete: {'constraints': 11, 'indexes': 5, 'vector_indexes': 2}
```

### 7. Run the service

Manual (development):
```bash
./venv/bin/python backend/run.py
```

As a systemd service (production):
```bash
sudo cp systemd/forgerag-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable forgerag-api
sudo systemctl start forgerag-api
```

### 8. Verify

```bash
curl -s http://localhost:8200/health | python3 -m json.tool
```

Expected (when Neo4j is up):
```json
{
    "success": true,
    "data": {
        "status": "ok",
        "service": "forgerag",
        "version": "0.1.0",
        "neo4j_connected": true,
        "document_count": 0,
        "page_count": 0,
        "gpu_available": false,
        "config_loaded": true
    }
}
```

`gpu_available: false` is expected in Phase 1 — torch is not yet installed. It gets added
in Phase 3 with the ML dependencies.

## Project Structure

```
ForgeRAG/
├── backend/                      FastAPI service
│   ├── main.py                   App factory, lifespan, CORS, static mount
│   ├── run.py                    Uvicorn entrypoint
│   ├── config.py                 Pydantic Settings loaded from TOML
│   ├── models/
│   │   └── common.py             ForgeResult{success, reason?, data?}
│   ├── routers/
│   │   └── health.py             GET / and /health
│   ├── services/
│   │   └── neo4j_service.py      Async driver wrapper + query helpers
│   ├── db/
│   │   └── neo4j_schema.py       Constraints + indexes + vector indexes
│   ├── ingestion/                (Phase 2+)
│   └── __init__.py
├── frontend/                     (Phase 6)
├── config/
│   ├── forgerag.toml             Active config (gitignored, copied from .example)
│   └── forgerag.toml.example     Template
├── systemd/
│   └── forgerag-api.service      systemd unit for boot-time startup
├── scripts/
│   ├── install_neo4j.sh          Install Neo4j Community 5.x (Debian/Ubuntu)
│   └── seed_schema.py            Apply Neo4j constraints + indexes
├── data/                         Runtime data (gitignored)
│   ├── page_images/              (Phase 2)
│   └── reduced_images/           (Phase 2)
├── tests/
├── pyproject.toml                Package metadata + optional dep groups
├── requirements.txt              Phase 1 dependencies (pinned separately for reproducibility)
└── venv/                         Python virtualenv (gitignored)
```

## Response Envelope

All API endpoints return a standard envelope (matching Choom's memory-server convention):

```json
{
    "success": true,
    "reason": null,
    "data": { ... }
}
```

On error:
```json
{
    "success": false,
    "reason": "human-readable error message",
    "data": null
}
```

## Neo4j Schema

Node labels: `Document`, `Page`, `Category`, `Tag`, `Material`, `Process`, `Standard`, `Clause`,
`Equipment`, `Community`.

Vector indexes (created in Phase 1):
- `page_text_embedding` — 768d cosine, on `Page.text_embedding`
- `community_summary_embedding` — 768d cosine, on `Community.summary_embedding`

ColPali embeddings (multi-vector) are stored as binary blobs on `Page` nodes and searched
via MaxSim reranking in Python over candidates retrieved from the text vector index
(two-stage search — same pattern as the previous Milvus-based system).

To inspect the schema in the Neo4j Browser (http://localhost:7474):
```cypher
SHOW CONSTRAINTS;
SHOW INDEXES;
```

## Config Reference

See `config/forgerag.toml.example` for all settings. Key ones:

| Section | Key | Default | Notes |
|---------|-----|---------|-------|
| server | port | 8200 | FastAPI listen port |
| neo4j | uri | bolt://localhost:7687 | |
| neo4j | database | neo4j | Community Edition: always `neo4j` |
| neo4j | password_env | NEO4J_PASSWORD | Env var name — password never in TOML |
| models | colpali_name | vidore/colpali-v1.3 | (Phase 3) |
| models | text_embedding_model | nomic-ai/nomic-embed-text-v1.5 | (Phase 3) |
| models | text_embedding_dim | 768 | Must match vector index dimension |
| llm | endpoint | http://localhost:8300/v1 | Local llama.cpp/vLLM for entity extraction |
| gpu | model_idle_unload_seconds | 300 | Auto-unload idle models to free VRAM |

## Troubleshooting

**`neo4j_connected: false` in /health response**
Either Neo4j is not running (`systemctl status neo4j`) or `NEO4J_PASSWORD` is not exported.

**`TypeError: 'async for' requires an object with __aiter__ method`**
You have a stale copy of `backend/services/neo4j_service.py`. Pull the latest — the neo4j
6.x driver requires `await tx.run(...)` inside transaction functions.

**`Unsupported administration command: CREATE DATABASE`**
Neo4j Community Edition doesn't support custom databases. Use the default `neo4j` database
(the config is already set up this way).

**cypher-shell Java warnings**
Harmless noise from newer JVMs. cypher-shell still works correctly.
