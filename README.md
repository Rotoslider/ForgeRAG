# ForgeRAG

Local engineering knowledge graph for processing and querying large corpora of engineering PDFs. Combines visual document retrieval (Nemotron ColEmbed / ColPali), a Neo4j knowledge graph, and vision-language model answer generation into a single system that can read engineering handbooks, extract entities and relationships, and answer technical questions with page-level citations.

Designed for personal/research use. Runs entirely on local hardware — no cloud APIs.

## What it does

Ask a question like *"What is alloy C12000 used for and how do I weld it?"* and ForgeRAG will:

1. **Find the right pages** across all your engineering handbooks (keyword + visual retrieval)
2. **Traverse the knowledge graph** to discover related materials, processes, and standards you didn't ask about
3. **Read the actual page images** using a vision LLM (not mangled OCR text)
4. **Synthesize an answer** with `[Page N]` citations linking to a built-in page viewer
5. **Include adjacent pages** automatically so tables spanning page boundaries aren't missed

## Status

Phases 1–6 complete. Phase 7 (Choom agent integration) next.

- [x] **Phase 1**: FastAPI service + Neo4j schema
- [x] **Phase 2**: PDF ingestion (rasterize → text extract → resume-friendly)
- [x] **Phase 3**: Visual embeddings (Nemotron ColEmbed 4B) + text embeddings (nomic)
- [x] **Phase 4**: LLM entity extraction (Qwen3.5 35B) + knowledge graph queries
- [x] **Phase 5**: GraphRAG communities, hybrid search, page highlighting
- [x] **Phase 6**: React/Vite frontend (Search, Ingest, Manage, Page Viewer)
- [ ] **Phase 7**: Choom agent skill integration
- [ ] **Phase 8**: Bulk ingestion, entity cleanup, auto-tagging

## Architecture

```
┌──────────────────────────────────────────────────────┐
│              React/Vite GUI (:8200/app/)              │
│  Search (Answer/Keyword/Visual) · Ingest · Manage    │
├──────────────────────────────────────────────────────┤
│              FastAPI REST API (:8200)                  │
│  29 endpoints · ForgeResult{success, reason, data}    │
├───────────┬──────────────┬───────────────────────────┤
│  Neo4j    │  Nemotron    │  Page Image Store         │
│  Graph +  │  ColEmbed 4B │  (PNGs + reduced JPGs,    │
│  Vector   │  + MaxSim    │   page viewer with nav)   │
│  + Lucene │  reranking   │                           │
└───────────┴──────────────┴───────────────────────────┘
```

**Hardware** (designed for this specific setup):
- **NUC i7, 96 GB DDR5** — Neo4j, FastAPI, frontend, page images
- **NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM** — Nemotron ColEmbed, text embeddings, Qwen3.5 VLM
- **M3 Ultra Mac, 256 GB RAM** — LM Studio for auxiliary LLMs (optional)

## Knowledge Graph

Documents are organized into domain-specific **collections** (e.g., `asm_references`, `mechanical_design`, `firearms`). Each collection contains:

```
(:Document)─[:HAS_PAGE]─►(:Page)
     │                      │
     ├─[:IN_CATEGORY]─►(:Category)
     │                      ├─[:MENTIONS_MATERIAL]─►(:Material)
     ├─[:TAGGED_WITH]─►(:Tag)   ├─[:DESCRIBES_PROCESS]─►(:Process)
     │                      ├─[:REFERENCES_STANDARD]─►(:Standard)
     └─collection property   └─[:MENTIONS_EQUIPMENT]─►(:Equipment)

(:Material)─[:GOVERNED_BY]─►(:Standard)
(:Material)─[:COMPATIBLE_WITH_PROCESS]─►(:Process)
(:Standard)─[:REFERENCES]─►(:Standard)
(:Standard)─[:CONTAINS_CLAUSE]─►(:Clause)
(:Page)─[:IN_COMMUNITY]─►(:Community)  ← GraphRAG summaries
```

**Current graph** (after ingesting ASM Handbooks Vol 1 & 6):
- 3 documents, 4,511 pages
- 13,556 materials, 5,162 processes, 1,918 standards, 6,046 equipment
- 40 GraphRAG communities with LLM-generated summaries

## Search Modes

| Mode | What it does | Best for |
|------|-------------|----------|
| **Answer** (default) | Keyword + ColPali visual retrieval + graph traversal → VLM reads page images → synthesized answer with citations | Questions: *"What preheat does ASME IX require for P-1 over 1 inch?"* |
| **Keyword** | Lucene full-text phrase search on extracted text | Specific codes: *"C12000"*, *"QW-451.1"*, *"ASTM A 709"* |
| **Visual** | ColPali/Nemotron two-stage retrieval (text vector coarse → MaxSim rerank) | Finding specific charts, tables, diagrams |
| **Hybrid** | Vector + knowledge graph entity boosting | Broad topics where graph connections matter |

Answer mode includes **adjacent pages** (N-1 and N+1) so the VLM can read tables that span page boundaries. It also feeds **knowledge graph context** (relationship chains, related entities, community summaries) into the LLM prompt so it can mention relevant standards and processes the user didn't specifically ask about.

## Visual Retrieval Models

| Model | Embedding dim | Tokens/page | VRAM | Storage/page |
|-------|-------------|-------------|------|-------------|
| **Nemotron ColEmbed 4B** (default) | 128 (projected from 2560) | 773 | ~12 GB | 396 KB |
| ColPali v1.3 (fallback) | 128 | 1031 | ~24 GB | 175 KB |

Configured via `visual_model_type` in `config/forgerag.toml`. Both use the same MaxSim late-interaction scoring and binary blob storage format on Neo4j Page nodes.

## Setup

### Prerequisites

- Python 3.12+, Node.js 24+
- Neo4j Community Edition 5.x
- NVIDIA GPU with CUDA 12.8+ (for Nemotron/ColPali embeddings)
- LM Studio or llama.cpp server (for entity extraction + answer generation)

### Install

```bash
# 1. Clone and set up Python
cd /home/nuc1/projects/ForgeRAG
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# 2. Install Neo4j (one-time)
./scripts/install_neo4j.sh
# Change password:
cypher-shell -u neo4j -p neo4j -d system \
    "ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO 'YOUR_PASSWORD'"

# 3. Set up secrets
sudo mkdir -p /etc/forgerag
echo "NEO4J_PASSWORD='YOUR_PASSWORD'" | sudo tee /etc/forgerag/env > /dev/null
sudo chmod 600 /etc/forgerag/env

# 4. Copy and edit config
cp config/forgerag.toml.example config/forgerag.toml

# 5. Seed Neo4j schema
export NEO4J_PASSWORD='YOUR_PASSWORD'
./venv/bin/python scripts/seed_schema.py

# 6. Build frontend
cd frontend && npm install && npm run build && cd ..

# 7. Install systemd service
sudo cp systemd/forgerag-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable forgerag-api
sudo systemctl start forgerag-api
```

### Verify

```bash
curl -s http://localhost:8200/health | python3 -m json.tool
# Expect: neo4j_connected: true, gpu_available: true
```

Web GUI: `http://localhost:8200/app/`

## Usage

### Ingest a PDF

1. Open `http://localhost:8200/app/` → **Ingest** tab
2. Select a PDF, choose a collection (or create a new one), add tags
3. Click **Start Ingestion** — pipeline runs automatically:
   - PDF → page images (300 DPI PNGs + reduced JPGs)
   - Text extraction (PyMuPDF, scanned detection)
   - Text embeddings (nomic-embed-text-v1.5, 768d)
   - Visual embeddings (Nemotron ColEmbed 4B, 128d projected)
   - Entity extraction (Qwen3.5 35B via LM Studio)
4. Monitor progress on the Ingest tab

### Search

- **Answer mode** (default): type a question, get a synthesized answer with page citations
- **Keyword**: exact match for alloy codes, clause IDs, standard numbers
- Click page thumbnails to expand. Use Prev/Next to browse adjacent pages.
- Source links open in the Page Viewer (dedicated full-page view with navigation)

### Manage

- **Documents table**: edit collection, tags, categories inline. Re-embed, extract entities, or delete.
- **Graph Stats**: live entity counts across the knowledge graph
- **GPU**: VRAM usage, loaded models, manual unload
- **Communities**: rebuild GraphRAG summaries from the entity graph
- **Entities**: browse Materials, Processes, Standards, Equipment with page mention counts

## API Endpoints

### Core
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service status, Neo4j, GPU, counts |
| GET | `/collections` | List collections with doc/page counts |

### Search
| Method | Path | Description |
|--------|------|-------------|
| POST | `/search/answer` | RAG answer (keyword+visual+graph → VLM reads pages) |
| POST | `/search/keyword` | Lucene full-text phrase search |
| POST | `/search/visual` | ColPali/Nemotron two-stage visual retrieval |
| POST | `/search/semantic` | Text embedding vector search |
| POST | `/search/hybrid` | Vector + graph-boosted / graph-first / community |

### Documents
| Method | Path | Description |
|--------|------|-------------|
| GET | `/documents` | List (filter by collection/category/tag) |
| GET | `/documents/{id}` | Detail |
| DELETE | `/documents/{id}` | Delete (cascade: pages, images, entities) |
| PUT | `/documents/{id}/collection` | Move to a different collection |
| POST | `/documents/{id}/tags` | Add a tag |
| DELETE | `/documents/{id}/tags/{name}` | Remove a tag |
| POST | `/documents/{id}/categories` | Add a category |
| DELETE | `/documents/{id}/categories/{name}` | Remove a category |
| POST | `/documents/{id}/reembed` | Re-run visual + text embeddings |
| POST | `/documents/{id}/extract-entities` | Re-run LLM entity extraction |
| GET | `/documents/{id}/pages` | List pages |
| GET | `/documents/{id}/pages/{n}` | Page detail with full text |

### Ingestion
| Method | Path | Description |
|--------|------|-------------|
| POST | `/ingest` | Upload PDF (multipart: file, collection, categories, tags) |
| GET | `/ingest/jobs/{id}` | Poll job progress |
| GET | `/ingest/jobs` | List recent jobs |

### Knowledge Graph
| Method | Path | Description |
|--------|------|-------------|
| POST | `/graph/query` | Predefined graph queries (material_standards, process_materials, etc.) |
| POST | `/graph/explore` | N-hop neighborhood of an entity |
| GET | `/graph/entities/{type}` | List extracted entities with mention counts |
| GET | `/graph/stats` | Per-label node counts |
| POST | `/graph/build-communities` | Rebuild GraphRAG community summaries |
| GET | `/graph/communities` | List communities |

### Images
| Method | Path | Description |
|--------|------|-------------|
| GET | `/images/{hash}/{page}` | Full-resolution PNG |
| GET | `/images/{hash}/{page}/reduced` | Reduced JPG thumbnail |

### System
| Method | Path | Description |
|--------|------|-------------|
| GET | `/system/gpu` | VRAM usage + loaded models |
| POST | `/system/models/{name}/unload` | Manually unload a model |

### Admin
| Method | Path | Description |
|--------|------|-------------|
| POST | `/admin/dedup-pages` | Remove duplicate Page nodes |
| POST | `/admin/cleanup-uploads` | Delete staged upload files |

## Configuration

See `config/forgerag.toml.example` for all settings. Key sections:

| Section | Key settings |
|---------|-------------|
| `[server]` | port (8200), data_dir |
| `[neo4j]` | uri, database (neo4j), password_env |
| `[models]` | visual_model_name, visual_model_type (nemotron/colpali), visual_embed_dim (128), text_embedding_model |
| `[llm]` | endpoint (LM Studio), model (qwen3.5-35b-a3b), use_json_schema, max_tokens |
| `[ingestion]` | pdf_dpi (300), batch sizes, scanned text threshold |
| `[gpu]` | device, model_idle_unload_seconds (300) |

## Project Structure

```
ForgeRAG/
├── backend/
│   ├── main.py                    FastAPI app, lifespan, router wiring
│   ├── config.py                  Pydantic Settings from TOML
│   ├── run.py                     Uvicorn entrypoint (loads /etc/forgerag/env)
│   ├── models/                    Pydantic request/response models
│   ├── routers/                   API route handlers
│   │   ├── search.py              Answer, keyword, visual, semantic, hybrid search
│   │   ├── documents.py           Document/collection/tag/category CRUD
│   │   ├── ingestion.py           PDF upload + job tracking
│   │   ├── graph.py               Knowledge graph queries + communities
│   │   ├── images.py              Page image serving + viewer
│   │   ├── system.py              GPU status + model management
│   │   └── admin.py               Dedup, cleanup utilities
│   ├── services/
│   │   ├── nemotron_service.py    Nemotron ColEmbed 4B (visual retrieval)
│   │   ├── colpali_service.py     ColPali v1.3 (legacy visual retrieval)
│   │   ├── text_embedding_service.py  nomic-embed-text (text vectors)
│   │   ├── llm_service.py         OpenAI-compatible LLM client
│   │   ├── gpu_manager.py         VRAM tracking, semaphore, idle unload
│   │   ├── graph_reasoning.py     Graph traversal for answer context
│   │   ├── image_service.py       Page highlight overlay (ColPali heatmap)
│   │   └── neo4j_service.py       Async Neo4j driver wrapper
│   ├── ingestion/
│   │   ├── pipeline.py            8-step ingestion orchestrator
│   │   ├── pdf_processor.py       PDF → PNGs (chunked, resume-friendly)
│   │   ├── text_extractor.py      PyMuPDF text extraction
│   │   ├── entity_extractor.py    LLM structured entity/relationship extraction
│   │   ├── graph_builder.py       Neo4j MERGE for entities + relationships
│   │   ├── community_detector.py  Leiden clustering + LLM summaries
│   │   └── job_manager.py         SQLite job queue
│   └── db/
│       └── neo4j_schema.py        Constraints, indexes, vector indexes, full-text
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Search.tsx         Answer/Keyword/Visual/Hybrid search
│   │   │   ├── Ingest.tsx         Upload form + job progress
│   │   │   ├── Manage.tsx         Documents, entities, GPU, communities
│   │   │   └── Viewer.tsx         Full-page viewer with navigation
│   │   ├── components/Layout.tsx  Sidebar nav with live health indicators
│   │   └── api/                   Typed client + types
│   └── vite.config.ts             Proxy + /app/ base path
├── config/forgerag.toml           Active config (gitignored)
├── config/forgerag.toml.example   Template
├── systemd/forgerag-api.service   systemd unit
├── scripts/
│   ├── install_neo4j.sh           Neo4j Community 5.x installer
│   └── seed_schema.py             Apply Neo4j schema (idempotent)
└── data/                          Runtime data (gitignored)
    ├── page_images/{hash}/        Full-resolution PNGs
    ├── reduced_images/{hash}/     Reduced JPGs
    ├── uploads/                   Staged PDFs (cleaned via admin endpoint)
    └── jobs.sqlite                Ingestion job queue
```

## LLM Model Notes

**Entity extraction** (Qwen3.5 35B MoE, 3B active):
- Requires `use_json_schema = true` and `/no_think` in the prompt
- Runs on RTX 6000 via LM Studio at ~135 tok/s, ~8-10s per page
- LM Studio "thinking" toggle should be OFF

**Gemma 4 26B MoE**: breaks under strict JSON schema grammar (degenerate repetition). Use `use_json_schema = false`.

**GLM 4.7 Flash**: reasoning model, too slow for batch extraction (~25 tok/s). Outputs to `reasoning_content` field — the LLM client handles this via fallback.

## License

Code: MIT. Models have their own licenses:
- Nemotron ColEmbed: CC-BY-NC-4.0 (non-commercial)
- ColPali v1.3: MIT
- nomic-embed-text: Apache 2.0
