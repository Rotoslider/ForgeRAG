# ForgeRAG

Local engineering knowledge graph for processing and querying large corpora of engineering PDFs. Combines visual document retrieval (Nemotron ColEmbed / ColPali), a Neo4j knowledge graph, and vision-language model answer generation into a single system that can read engineering handbooks, extract entities and relationships, and answer technical questions with page-level citations.

Designed for personal/research use. Runs entirely on local hardware — no cloud APIs.

## Screenshots

![Search — Answer mode with VLM-generated response and page citations](docs/ForgeRAG-search.png)

![Ingest — PDF upload with collection, category, and tag assignment](docs/ForgeRAG-ingest.png)

![Manage — Documents, entities, GPU status, and GraphRAG communities](docs/ForgeRAG-manage.png)

## What it does

Ask a question like *"What is alloy C12000 used for and how do I weld it?"* and ForgeRAG will:

1. **Find the right pages** across all your engineering handbooks (keyword + visual retrieval)
2. **Traverse the knowledge graph** to discover related materials, processes, and standards you didn't ask about
3. **Read the actual page images** using a vision LLM (not mangled OCR text)
4. **Synthesize an answer** with `[Page N]` citations linking to a built-in page viewer
5. **Include adjacent pages** automatically so tables spanning page boundaries aren't missed

## Status

All phases complete.

- [x] **Phase 1**: FastAPI service + Neo4j schema
- [x] **Phase 2**: PDF ingestion (rasterize, text extract, resume-friendly)
- [x] **Phase 3**: Visual embeddings (Nemotron ColEmbed 4B, hierarchical token pooling)
      + text embeddings (BGE-M3 1024d)
- [x] **Phase 4**: LLM entity extraction (Qwen 3.6 35B-A3B) + knowledge graph queries
- [x] **Phase 5**: GraphRAG communities, hybrid search, page highlighting
- [x] **Phase 6**: React/Vite frontend (Search, Ingest, Manage, Page Viewer)
- [x] **Phase 7**: Choom agent skill integration (manifest, auto-search, batch)
- [x] **Phase 8**: Auto-tagging, entity canonicalization, bulk re-embed
- [x] **Phase 9**: Structural chunking (Docling) + per-chunk LLM summaries
      + RRF hybrid (BM25 + dense) + BGE reranker + Formula/Table/topic-tag
      extraction + Standards `title` field + data-quality validators
- [x] **Phase 10**: Backup & Restore system (Neo4j dump, image copy, Google Drive upload)

## New Features

Recent additions since the Phase 9 baseline:

- **Search error boundary** — no more blank pages when switching between search modes. The React search view catches rendering errors and recovers gracefully.
- **Fuzzy entity matching** — EntityMatcher service loads entity names from Neo4j into memory and matches query text with difflib SequenceMatcher. Handles OCR-style typos, missing special characters, case mismatches, and spacing differences.
- **OCR typo tolerance** — keyword search now uses Lucene `~1` fuzzy operator so queries like "alumnum" still match "aluminum" in extracted text.
- **Community search weighted by member count** — community results are ranked by the number of entity members, surfacing the most connected communities first.
- **LLM circuit breaker** — 5 consecutive LLM failures trip the breaker open; all requests fail fast for 60 seconds, then a single probe request is allowed through. Prevents cascading timeouts during LM Studio restarts.
- **Neo4j health loop** — 30-second heartbeat with exponential backoff auto-reconnect. The service stays alive and recovers automatically when Neo4j restarts for a dump or update.
- **Choom skills integration** — `/skills/manifest` advertises ForgeRAG capabilities with live stats; `/skills/search` auto-routes queries (keyword vs answer vs hybrid) based on content; `/skills/batch` runs up to 20 queries in parallel.
- **Backup & Restore system** — GUI-driven and CLI-driven full backups with Neo4j dump, graph JSON export, page images, reduced images, source PDFs, and optional Google Drive upload. Incremental: subsequent backups skip unchanged files.

## Architecture

```
+------------------------------------------------------+
|              React/Vite GUI (:8200/app/)              |
|  Search (Answer/Keyword/Visual) . Ingest . Manage    |
+------------------------------------------------------+
|              FastAPI REST API (:8200)                  |
|  40+ endpoints . ForgeResult{success, reason, data}   |
+-------------+--------------+-------------------------+
|  Neo4j      |  Nemotron    |  Page Image Store        |
|  Graph +    |  ColEmbed 4B |  (PNGs + reduced JPGs,   |
|  Vector     |  + MaxSim    |   page viewer with nav)  |
|  + Lucene   |  reranking   |                          |
+-------------+--------------+-------------------------+
```

## Models & Hardware

### Models

| Model | Role | Why this model |
|-------|------|----------------|
| **BGE-M3** (1024d) | Text embedding | Multilingual, strong retrieval benchmarks on technical content, no query/doc prefix needed (unlike Nomic which requires `search_query:` / `search_document:` prefixes). Pairs naturally with bge-reranker-v2-m3 |
| **Nemotron ColEmbed VL-4B-v2** (128d projected) | Visual embedding | Best-in-class visual document retrieval. Projects from 2560d to 128d with 96.8% accuracy retention. Hierarchical token pooling (pool_factor=3) reduces storage 3x with negligible accuracy loss |
| **BGE Reranker v2-m3** | Cross-encoder reranker | Re-scores top-K candidates from hybrid retrieval. Significant accuracy improvement over raw vector cosine scores, especially for technical queries with rare tokens |
| **Qwen 3.6 35B-A3B** | Entity extraction LLM | MoE architecture (only 3B active parameters), fast JSON extraction via json_schema grammar, ~135 tok/s on RTX 6000 via LM Studio. Changed from Qwen 3.5 — 3.6 requires `chat_template_kwargs.enable_thinking=False` instead of the old `/no_think` directive (see LLM Model Notes) |
| **ColPali v1.3** (128d) | Visual embedding (legacy fallback) | Still supported via `visual_model_type = "colpali"` in config. Nemotron is default due to ~20-30% better accuracy on tables/charts |

### Minimum Hardware

- **GPU**: 24 GB VRAM minimum (Nemotron + text embedding model). 48+ GB recommended for concurrent model loading (visual + text + reranker loaded simultaneously)
- **RAM**: 32 GB minimum, 64 GB recommended (Neo4j heap + Python processes + Node.js frontend build)
- **Storage**: 100 GB minimum for a small corpus. Approximately 1 GB per 100 pages (page images + reduced images + embeddings). Current 83-document corpus uses ~80 GB
- **CPU**: 8+ cores recommended for PDF rasterization (parallel page conversion) and Neo4j query processing

### Recommended Hardware (what ForgeRAG was built on)

- **Intel NUC i7, 96 GB DDR5** — Neo4j, FastAPI, frontend, page images
- **NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM** — Nemotron ColEmbed, text embeddings, Qwen 3.6 VLM, BGE reranker
- **2 TB NVMe** for data directory (page images, reduced images, Neo4j database)

## Knowledge Graph

Documents are organized into domain-specific **collections** (e.g., `asm_references`, `mechanical_design`, `firearms`). Each document's pages are further split into **structural chunks** (paragraphs, tables, figures, equations) via Docling, with per-chunk LLM summaries and BGE-M3 embeddings. Entities extracted from page text populate the knowledge graph; chunks carry the retrieval embeddings.

```
(:Document)--[:HAS_PAGE]-->(:Page)--[:HAS_CHUNK]-->(:Chunk)
     |                      |                      +- text + summary + embedding
     +--[:IN_CATEGORY]-->(:Category)                  chunk_type (text/table/figure/...)
     +--[:TAGGED_WITH]-->(:Tag)                       section_path + bbox
     |
     |                      +--[:MENTIONS_MATERIAL]-->(:Material)
     |                      +--[:DESCRIBES_PROCESS]-->(:Process)
     |                      +--[:REFERENCES_STANDARD]-->(:Standard)
     |                      +--[:MENTIONS_EQUIPMENT]-->(:Equipment)
     |                      +--[:MENTIONS_FORMULA]-->(:Formula)
     |                      +--[:MENTIONS_TABLE]-->(:RefTable)
     |
     +- Page.topic_tags: ["tap-drill-chart", "fastener-torque", ...]

(:Material)--[:GOVERNED_BY]-->(:Standard)
(:Material)--[:COMPATIBLE_WITH_PROCESS]-->(:Process)
(:Standard)--[:REFERENCES]-->(:Standard)
(:Standard)--[:CONTAINS_CLAUSE]-->(:Clause)
(:Page)--[:IN_COMMUNITY]-->(:Community)  <-- GraphRAG summaries
```

**Node types added in Phase 9:**
- `Chunk` — paragraph/table/figure-level unit with BGE-M3 embedding + LLM summary. Primary retrieval target (replaces whole-page text as the search granularity).
- `Formula` — named engineering formulas (`kind`: stress, deflection, torque, power, electrical...) with expression + variable definitions.
- `RefTable` — design-handbook reference tables (`kind`: dimensions, specifications, conversion, selection...) with title + natural-language description.
- `Page.topic_tags` — page-level kebab-case topic classifier (`tap-drill-chart`, `conductor-ampacity`, `gear-tooth-geometry`, ...) as a fast retrieval filter.

**Entity deduplication** runs at two levels:

- **Tier 1 (canonicalization)**: case-fold + singularization + designation-prefix merging. Handles "Stainless Steel" vs "stainless steel" vs "stainless steels". Run retroactively via `scripts/canonicalize_entity_dryrun.py` + `canonicalize_entity_apply.py`.
- **Tier 2 (fuzzy dedup)**: SequenceMatcher similarity matching with safety guards (blocks different alloy numbers, conflicting properties, well-established entities). Handles "Ti-6Al-4V" vs "Ti6Al4V" vs "TI-6AL-4V", "gas-tungsten arc welding" vs "Gas Tungsten Arc Welding". Run via `scripts/dedup_entities_dryrun.py` + `dedup_entities_apply.py`.
- **Automatic (per-ingestion)**: after each document's entity extraction, a lightweight dedup pass merges near-duplicates created by that document against existing entities. Uses a 0.92 similarity threshold with numeric-designation guards. No manual intervention needed — new documents won't create "Stainless Steel" when "stainless steel" already exists.

Merged entity names are preserved as `common_names` on the surviving node, so fuzzy search still finds them. After bulk dedup, rebuild communities (`Manage → GraphRAG Communities → rebuild`) since the graph topology has changed.

**Standard codes vs titles**: `Standard.code` is the short designator (`ASME BPVC IX`, `NFPA 70`, `SEMI S2`); `Standard.title` is the full descriptive title. Both are alias-aware in queries.

## Search Modes

| Mode | What it does | Best for |
|------|-------------|----------|
| **Answer** (default) | RRF hybrid + BGE reranker + graph traversal, then VLM reads page images and synthesizes an answer with citations | Questions: *"What preheat does ASME IX require for P-1 over 1 inch?"* |
| **Keyword** | Lucene full-text phrase search on extracted text/chunks, with ~1 fuzzy tolerance for OCR typos | Specific codes: *"C12000"*, *"QW-451.1"*, *"ASTM A 709"* |
| **Visual** | ColPali/Nemotron two-stage retrieval (text-vector coarse, then MaxSim rerank) | Finding specific charts, tables, diagrams |
| **Hybrid** | Strategies: `rrf` (BM25 + dense + bge-reranker, default), `graph_boosted`, `vector_first`, `graph_first`, `community` | Tuned search behaviour per query type |

### RRF Hybrid (default)

`rrf` fuses two independent rankings with Reciprocal Rank Fusion (k=60):

1. **BM25** over chunk text + summary via Neo4j's Lucene full-text index — catches rare exact tokens (`QW-451.1`, `6061-T6`, `ER308LT-1`) that dense embeddings blur.
2. **Dense vector** similarity over BGE-M3 1024-dim chunk embeddings — catches semantic paraphrases.

The top ~50 fused candidates are then reranked by **`BAAI/bge-reranker-v2-m3`**, a cross-encoder that scores each (query, chunk) pair in one pass. Final top-K is returned to the caller. `rerank: false` on the request skips the cross-encoder if you want to inspect raw RRF order.

Answer mode includes **adjacent pages** (N-1 and N+1) so the VLM can read tables that span page boundaries. It also feeds **knowledge graph context** (relationship chains, related entities, community summaries) into the LLM prompt so it can mention relevant standards and processes the user didn't specifically ask about.

## Retrieval Models

**Text embeddings** — BGE-M3 (1024-dim) is the default. The older Nomic v1.5 (768-dim) is still supported by toggling `text_embedding_model` and `text_embedding_dim` in config. Query-time prefix handling is model-aware (Nomic uses `search_query: ` / `search_document: `; BGE-M3 doesn't).

**Reranker** — `BAAI/bge-reranker-v2-m3` cross-encoder (~1.2 GB VRAM, fp16). Lazy-loaded, auto-unloaded when idle. Re-scores top-K hybrid candidates.

**Visual embeddings:**

| Model | Embed dim | Native tokens/page | With pool_factor=3 | VRAM | Storage/page |
|-------|-----------|-------------------|-------------------|------|-------------|
| **Nemotron ColEmbed 4B** (default) | 128 (projected from 2560) | 773 | ~258 | ~12 GB | ~130 KB pooled |
| ColPali v1.3 (fallback) | 128 | 1031 | ~343 | ~24 GB | ~175 KB pooled |

Both visual models share a single `visual_pool_factor_storage` config knob (default 3) that applies `HierarchicalTokenPooler` at embed time — semantic clusters of patches (whitespace, uniform text, figure regions) collapse to one representative vector each. 3x reduction in storage and MaxSim compute with negligible accuracy loss. Set to 1 to disable.

Configured via `visual_model_type` in `config/forgerag.toml`. Both use MaxSim late-interaction scoring and the same binary blob storage format on Page nodes.

## Installation on Fresh Ubuntu

Step-by-step installation from a clean Ubuntu 24.04+ system. These instructions assume the project will live at `/home/nuc1/projects/ForgeRAG` — adjust paths as needed for your setup.

### 1. System Prerequisites

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    gnupg \
    lsb-release \
    apt-transport-https \
    ca-certificates \
    software-properties-common \
    python3-dev \
    python3-pip \
    python3-venv \
    pkg-config \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    libpng-dev \
    poppler-utils \
    netcat-openbsd
```

### 2. NVIDIA Driver + CUDA 12.8

If you don't already have NVIDIA drivers installed:

```bash
# Add the NVIDIA CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install the driver and CUDA toolkit
sudo apt-get install -y cuda-toolkit-12-8 nvidia-driver-570

# Reboot to load the new driver
sudo reboot
```

After reboot, verify:

```bash
nvidia-smi
# Should show your GPU with driver version and CUDA 12.8
```

### 3. Neo4j Community 5.x

ForgeRAG includes an install script that adds the official Neo4j APT repository, installs Neo4j Community Edition, locks listen addresses to localhost, and starts the service:

```bash
cd /home/nuc1/projects/ForgeRAG
./scripts/install_neo4j.sh
```

After installation, change the default password:

```bash
cypher-shell -u neo4j -p neo4j -d system \
    "ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO 'YOUR_STRONG_PASSWORD'"
```

Note: Neo4j Community Edition only supports the built-in `neo4j` and `system` databases. ForgeRAG uses the default `neo4j` database.

### 4. Python 3.12 Virtual Environment

```bash
# Install Python 3.12 if not already present
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev

# Create the virtual environment
cd /home/nuc1/projects/ForgeRAG
python3.12 -m venv venv
./venv/bin/pip install --upgrade pip setuptools wheel
```

### 5. Node.js 24

```bash
# Install via NodeSource
curl -fsSL https://deb.nodesource.com/setup_24.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify
node --version   # v24.x.x
npm --version
```

### 6. LM Studio

LM Studio provides the OpenAI-compatible LLM endpoint that ForgeRAG uses for entity extraction and answer generation.

1. Download LM Studio from https://lmstudio.ai/ (Linux AppImage or .deb)
2. Install and launch LM Studio
3. Download the model `Qwen/Qwen3.6-35B-A3B` (or search for `qwen3.6-35b-a3b`)
4. In LM Studio, go to the Local Server tab and start the server on port 1234 (the default)
5. Make sure "Thinking" is toggled OFF in model settings — ForgeRAG disables thinking via `chat_template_kwargs` in the API request

Verify the model is loaded:

```bash
curl -s http://localhost:1234/v1/models | python3 -m json.tool
# Should show the model ID (e.g., "qwen/qwen3.6-35b-a3b")
```

### 7. Clone the Repo and Install Dependencies

```bash
# Clone
cd /home/nuc1/projects
git clone https://github.com/YOUR_USER/ForgeRAG.git
cd ForgeRAG

# Python dependencies
./venv/bin/pip install -r requirements.txt

# Frontend dependencies
cd frontend
npm install
cd ..
```

### 8. Configuration

```bash
# Copy the example config
cp config/forgerag.toml.example config/forgerag.toml

# Edit as needed — the defaults work for a standard single-machine setup.
# Key things to verify:
#   [server] data_dir — where page images and job data live
#   [llm] endpoint — should point to LM Studio (default: http://localhost:1234/v1)
#   [llm] model — should match the model ID in LM Studio
```

### 9. Neo4j Password Setup

The Neo4j password is never stored in the config file. It lives in a protected environment file:

```bash
sudo mkdir -p /etc/forgerag
echo "NEO4J_PASSWORD='YOUR_STRONG_PASSWORD'" | sudo tee /etc/forgerag/env > /dev/null
sudo chmod 600 /etc/forgerag/env
```

The env file is also where you can set `HF_HUB_OFFLINE=1` to prevent the `transformers` library from contacting HuggingFace at runtime. All models are downloaded at install time, so there's no reason to phone home during ingestion — and transient HuggingFace API errors (500s) can crash long-running jobs. Both `run.py` and the systemd service default to offline mode, but the env file lets you override it (`HF_HUB_OFFLINE=0`) if you need to download a new model.

### 10. Seed the Neo4j Schema

This creates constraints, indexes, vector indexes, and full-text indexes. Idempotent — safe to run multiple times:

```bash
export NEO4J_PASSWORD='YOUR_STRONG_PASSWORD'
./venv/bin/python scripts/seed_schema.py
```

### 11. Build the Frontend

```bash
cd frontend
npm run build
cd ..
```

The built frontend is served by FastAPI at `/app/`.

### 12. systemd Service

```bash
# Copy the service file
sudo cp systemd/forgerag-api.service /etc/systemd/system/

# If you changed the project path, edit the service file:
# sudo nano /etc/systemd/system/forgerag-api.service
# Update WorkingDirectory, ExecStart, and FORGERAG_CONFIG paths

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable forgerag-api
sudo systemctl start forgerag-api
```

### 13. Verify

```bash
# Check service status
sudo systemctl status forgerag-api

# Check API health
curl -s http://localhost:8200/health | python3 -m json.tool
# Expect: neo4j_connected: true, gpu_available: true
```

Open the web GUI at `http://localhost:8200/app/` — you should see the Search page with an empty library.

## Adding Documents

### GUI Method

1. Open `http://localhost:8200/app/` and click the **Ingest** tab
2. Select a PDF file (drag-and-drop or file picker)
3. Choose a **collection** (or type a new one to create it)
4. Optionally add **categories** and **tags** (or let the auto-tagger suggest them later)
5. Click **Start Ingestion**

### Pipeline Phases

The ingestion pipeline processes the PDF through these phases automatically:

| Phase | What it does | Approximate time |
|-------|-------------|-----------------|
| **Rasterize** | Converts each PDF page to a 300 DPI PNG + a reduced JPG thumbnail | ~1-2 s/page |
| **Text Extract** | PyMuPDF text extraction from digital-native pages; scanned pages are flagged for OCR | ~0.5 s/page |
| **Auto-Tag** | LLM reads the first 10 chunks and suggests a collection, 2-4 categories, and 5-10 tags | ~10 s total |
| **Text Embed** | BGE-M3 encodes extracted text into 1024-dim vectors stored on Page and Chunk nodes | ~0.5 s/page |
| **Chunk** | Docling structural chunker splits pages into paragraphs, tables, figures, and equations. Per-chunk LLM summaries are generated for long chunks | ~2-5 s/page |
| **Visual Embed** | Nemotron ColEmbed 4B encodes page images into 128-dim multi-vector representations with hierarchical token pooling | ~2-3 s/page |
| **Entity Extract** | Qwen 3.6 reads each page and extracts Materials, Processes, Standards, Equipment, Formulas, and Tables as structured JSON | ~8-10 s/page |

For a 100-page PDF, expect roughly 30-60 minutes total depending on content density.

### Auto-Tagging

After ingestion (or on existing documents), you can click **suggest** on any document in the Manage tab. The LLM reads the document's content and proposes:
- A **collection** (the top-level grouping)
- 2-4 **categories** (domain classifications)
- 5-10 **tags** (specific topics covered)

Review the suggestions as editable chips, drop what you don't want, choose **merge** (keep existing metadata) or **replace** (start fresh), then **apply**.

### After Ingestion

- **Rebuild communities**: Go to the Manage page and click "Rebuild Communities" to regenerate GraphRAG community summaries. This uses Leiden clustering on the entity graph to create community nodes with LLM-generated summaries.
- **Bulk operations**: Select multiple documents on the Manage page with checkboxes, then use "rebuild (N)" to re-run chunking + entity extraction, "extract-only" to only re-extract entities, or "only-missing" to skip documents that already have chunks.

### Monitoring Progress

The Ingest tab shows real-time progress for each active job: current phase, pages processed, and estimated time remaining. Jobs run sequentially — queue multiple PDFs and they'll process one at a time.

## Usage

### Search

- **Answer mode** (default): type a question, get a synthesized answer with page citations
- **Keyword**: exact match for alloy codes, clause IDs, standard numbers
- Click page thumbnails to expand. Use Prev/Next to browse adjacent pages.
- Source links open in the Page Viewer (dedicated full-page view with navigation)

### Manage

- **Documents table**: edit collection, tags, categories inline. Sticky header row + bulk-action bar stay pinned as you scroll. Sticky-right "Actions" column so controls are always reachable on wide rows. Per-row actions: **edit** (inline collection/tag/category edit), **suggest** (LLM proposes collection/categories/tags, review-and-apply with merge/replace), **rebuild** (Phase 9 chunks + entity re-extraction), **... overflow menu** (extract-only, re-embed, extract, delete).
- **Suggest tags (LLM auto-tagger)**: click **suggest** on any doc to have the LLM read the document's first 10 chunks (falls back to page text if the doc has no chunks yet) and propose a collection, 2-4 categories, and 5-10 tags. The panel shows the suggestion as editable chips — drop what you don't like, add your own, choose **merge** (keep existing) or **replace** (drop existing first), then **apply**. No re-embedding, just metadata writes.
- **Multi-select + bulk actions**: checkboxes on each row. Select multiple documents, then "rebuild (N)", "extract-only", or "only-missing" (skip docs that already have chunks). Jobs queue sequentially — pick a handful to run now or queue the whole library overnight.
- **Backup & Restore**: configure backup destination, trigger full backups, monitor progress, and view available backups for restore.
- **Graph Stats**: live entity counts across the knowledge graph
- **GPU**: VRAM usage, loaded models, manual unload
- **Communities**: rebuild GraphRAG summaries from the entity graph
- **Entities**: browse Materials, Processes, Standards, Equipment with page mention counts

### Rebuild existing documents for Phase 9

Documents ingested before Phase 9 only have Page-level embeddings. To get them onto the new chunked + RRF retrieval path:

**GUI path** — Manage tab, select docs, click "rebuild". Progress in the Ingest tab.

**CLI path**:

```bash
# Full rebuild of every doc — runs overnight at scale
NEO4J_PASSWORD=... ./venv/bin/python scripts/rebuild_chunks.py

# Just the docs that don't have chunks yet (resume)
NEO4J_PASSWORD=... ./venv/bin/python scripts/rebuild_chunks.py --only-missing

# One specific doc
NEO4J_PASSWORD=... ./venv/bin/python scripts/rebuild_chunks.py --doc-id DOC_XXX

# Cheap retry: only re-extract entities on pages that failed
NEO4J_PASSWORD=... ./venv/bin/python scripts/rebuild_chunks.py --doc-id DOC_XXX --extract-only
```

Flags:
- `--only-missing` — skip docs that already have Chunk nodes
- `--skip-extract` — chunks + summaries + embeddings only (no entity re-extraction)
- `--extract-only` — only re-extract entities on pages missing `topic_tags` (inverse of `--skip-extract`)

### Backfill tags/categories on older documents

Documents ingested before the auto-tagger existed — or ingested with manual tags that were later deleted — can be backfilled without a full rebuild. Per-doc via the Manage UI's **suggest** button, or bulk via the script:

```bash
# Preview only — no writes
NEO4J_PASSWORD=... ./venv/bin/python scripts/bulk_autotag.py --dry-run

# Apply to every doc that has no categories AND no tags
NEO4J_PASSWORD=... ./venv/bin/python scripts/bulk_autotag.py

# Specific docs, or limit
NEO4J_PASSWORD=... ./venv/bin/python scripts/bulk_autotag.py --doc-ids DOC_a,DOC_b
NEO4J_PASSWORD=... ./venv/bin/python scripts/bulk_autotag.py --limit 5

# Replace existing tags/categories (default is merge)
NEO4J_PASSWORD=... ./venv/bin/python scripts/bulk_autotag.py --overwrite
```

Default is `merge` — never overwrites a non-default collection and never detaches existing Tag/Category edges. `--overwrite` widens the selection to every doc and detaches existing edges before writing the suggestion.

## Backup & Restore

### One-Time Setup: Neo4j Dump Helper

The full backup includes a Neo4j database dump (all nodes, relationships, and embeddings). This requires stopping Neo4j briefly, which needs root privileges. A helper script handles the stop/dump/restart cycle safely:

```bash
# Install the helper script
sudo cp scripts/neo4j-dump-helper.sh /usr/local/bin/forgerag-dump
sudo chmod 755 /usr/local/bin/forgerag-dump

# Allow the service user to run it without a password prompt
echo 'nuc1 ALL=(ALL) NOPASSWD: /usr/local/bin/forgerag-dump' | sudo tee /etc/sudoers.d/forgerag-dump
sudo chmod 440 /etc/sudoers.d/forgerag-dump
```

The systemd service also needs `NoNewPrivileges=false` so the backup process can invoke `sudo`:

```bash
sudo sed -i 's/NoNewPrivileges=true/NoNewPrivileges=false/' /etc/systemd/system/forgerag-api.service
sudo systemctl daemon-reload
```

### GUI Backup

1. Open the **Manage** page in the web GUI
2. Find the **Backup & Restore** panel
3. Set the **destination path** (e.g., `/mnt/nas/forgerag-backups`)
4. Click **Start Full Backup**
5. Monitor progress in the panel — it shows the current file, bytes copied, and percentage

### What the Backup Includes

| Component | Typical size | Description |
|-----------|-------------|-------------|
| **Neo4j dump** | ~19 GB | Full database dump including all embeddings (text + visual vectors). This is the critical piece — everything else can be regenerated from source PDFs |
| **Graph JSON** | ~100 MB | Lightweight metadata export (documents, pages, entities, relationships, categories, tags). Can be imported without a full Neo4j restore |
| **Manifest** | ~10 KB | Backup metadata: timestamp, file counts, sizes, what was included |
| **Page images** | ~38 GB | Full-resolution 300 DPI PNGs of every page |
| **Reduced images** | ~14 GB | Reduced JPG thumbnails for the page viewer |
| **Source PDFs** | ~2.4 GB | Original uploaded PDF files |

### Google Drive Upload

Optionally, the backup can upload JSON files (graph export, manifest) to Google Drive. The Neo4j dump is typically too large for free-tier Drive storage, but the graph JSON provides a lightweight recovery path.

Enable in backup settings:
- **gdrive_enabled**: upload graph JSON and manifest to Drive
- **gdrive_dump**: also upload the Neo4j dump to Drive (only if you have sufficient storage)

### Incremental Backups

Subsequent backups to the same destination skip files that haven't changed (same size and modification time). Page images and reduced images are stored in shared directories at the destination root, not per-timestamp, so only new/modified files are copied. This makes daily backups fast after the initial full backup.

### CLI Backup

The same backup can be triggered from the command line:

```bash
./scripts/backup.sh
```

This creates a timestamped Neo4j dump and JSON manifest in `data/backups/`, keeping the last 5 backups and pruning older ones.

### Restoring from Backup

From a local backup directory:

```bash
./scripts/restore.sh --from-local /path/to/backup/
```

From Google Drive:

```bash
./scripts/restore.sh --from-drive
```

The restore script:
1. Stops the ForgeRAG API service
2. Stops Neo4j
3. Loads the Neo4j dump into the database
4. Restarts Neo4j
5. Restarts the ForgeRAG API service

### First-Run Detection

When ForgeRAG starts with an empty database (0 documents, 0 pages), the GUI shows restore instructions. The API endpoint `GET /admin/restore/status` reports `needs_restore: true` and lists available local backup directories.

## API Endpoints

### Core
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service status, Neo4j, GPU, counts |
| GET | `/collections` | List collections with doc/page counts |

### Search
| Method | Path | Description |
|--------|------|-------------|
| POST | `/search/answer` | RAG answer (keyword+visual+graph, then VLM reads pages) |
| POST | `/search/keyword` | Lucene full-text phrase search with fuzzy tolerance |
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
| POST | `/documents/{id}/rebuild-chunks` | Phase 9 rebuild: chunks + summaries + embeddings + entity re-extraction. Query params: `extract_only=true` (only re-extract pages missing topic_tags), `skip_extract=true` (chunks only) |
| POST | `/documents/{id}/suggest-tags` | LLM suggests collection, categories, and tags from chunk/page text. Read-only; returns `{collection, categories, tags}`. 503 if LLM is unconfigured; `success=false` with a diagnostic reason if the doc has no usable text |
| POST | `/documents/{id}/apply-tags` | Write confirmed tags/categories/collection. Body: `{collection?, categories?, tags?, mode: "merge"|"replace"}`. `merge` preserves existing edges; `replace` detaches all Tag/Category edges first |
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
| POST | `/admin/normalize-entities` | Merge duplicate entities that differ only by case/whitespace |
| POST | `/admin/bulk-reembed` | Queue re-embed jobs for every document |
| POST | `/admin/reembed-text` | Text-only re-embed (no visual, no entity extraction). Body: `{doc_id?}` |
| POST | `/admin/rebuild-chunks-bulk` | Queue Phase 9 chunk rebuilds for a list of doc_ids. Body: `{doc_ids, extract_only?, skip_extract?, only_missing?}`. Jobs run sequentially |
| POST | `/admin/cleanup-uploads` | Delete staged upload files |
| POST | `/admin/backup` | Hot graph export (JSON with all metadata, no embeddings) |
| GET | `/admin/backup/manifest` | Document manifest for backup verification |
| GET | `/admin/backup/settings` | Current backup configuration |
| POST | `/admin/backup/settings` | Update backup configuration (destination, include_images, include_pdfs, gdrive_enabled, gdrive_dump) |
| POST | `/admin/backup/full` | Trigger full backup (Neo4j dump + images + PDFs + graph JSON + optional Drive upload) |
| GET | `/admin/backup/progress` | Backup progress (running, percent, current_file, bytes_copied) |
| GET | `/admin/backup/list` | List available backups from local and destination directories |
| GET | `/admin/restore/status` | Check if database is empty and restore is needed. Lists available local backups |
| POST | `/admin/restore` | Return CLI commands for restoring from a local dump or Google Drive |

### Skills (Choom Integration)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/skills/manifest` | Capability advertisement with live stats (documents, pages, entities, communities) |
| POST | `/skills/search` | Auto-routing unified search — picks keyword, answer, or hybrid based on query characteristics |
| POST | `/skills/batch` | Parallel multi-query search (up to 20 queries) |

## Configuration

See `config/forgerag.toml.example` for all settings. Key sections:

| Section | Key settings |
|---------|-------------|
| `[server]` | port (8200), data_dir |
| `[neo4j]` | uri, database (neo4j), password_env |
| `[models]` | `visual_model_name`, `visual_model_type` (nemotron/colpali), `visual_embed_dim` (128), `colpali_pool_factor_storage` (3, shared by both visual models), `text_embedding_model` (`BAAI/bge-m3` default), `text_embedding_dim` (1024), `reranker_model` (`BAAI/bge-reranker-v2-m3`) |
| `[llm]` | endpoint (LM Studio), model (qwen3.6-35b-a3b), use_json_schema, max_tokens (4096; entity extraction internally bumps to 8192 for standards-heavy pages), disable_thinking (True) |
| `[ingestion]` | pdf_dpi (300), batch sizes, scanned text threshold |
| `[gpu]` | device, model_idle_unload_seconds (300) |
| `[backup]` | destination, include_images, include_pdfs, gdrive_enabled, gdrive_dump |

## Project Structure

```
ForgeRAG/
+-- backend/
|   +-- main.py                    FastAPI app, lifespan, router wiring
|   +-- config.py                  Pydantic Settings from TOML
|   +-- run.py                     Uvicorn entrypoint (loads /etc/forgerag/env)
|   +-- models/                    Pydantic request/response models
|   +-- routers/
|   |   +-- search.py              Answer, keyword, visual, semantic, hybrid search
|   |   +-- documents.py           Document/collection/tag/category CRUD
|   |   +-- ingestion.py           PDF upload + job tracking
|   |   +-- graph.py               Knowledge graph queries + communities
|   |   +-- images.py              Page image serving + viewer
|   |   +-- system.py              GPU status + model management
|   |   +-- admin.py               Dedup, cleanup, backup/restore endpoints
|   |   +-- skills.py              Choom skill integration (manifest, search, batch)
|   +-- services/
|   |   +-- nemotron_service.py    Nemotron ColEmbed 4B + hierarchical token pooling
|   |   +-- colpali_service.py     ColPali v1.3 (legacy visual retrieval)
|   |   +-- text_embedding_service.py  BGE-M3 / Nomic (model-aware prefixes)
|   |   +-- reranker_service.py    bge-reranker-v2-m3 cross-encoder
|   |   +-- llm_service.py         OpenAI-compatible LLM client with circuit breaker
|   |   +-- entity_matcher.py      Fuzzy entity name matching (difflib SequenceMatcher)
|   |   +-- gpu_manager.py         VRAM tracking, semaphore, idle unload
|   |   +-- graph_reasoning.py     Graph traversal for answer context
|   |   +-- image_service.py       Page highlight overlay (ColPali heatmap)
|   |   +-- neo4j_service.py       Async Neo4j driver wrapper + 30s health loop
|   +-- ingestion/
|   |   +-- pipeline.py            Ingestion orchestrator (full + partial runs)
|   |   +-- pdf_processor.py       PDF -> PNGs (chunked, resume-friendly)
|   |   +-- text_extractor.py      PyMuPDF text extraction
|   |   +-- chunker.py             Docling structural chunker (para/table/fig/eq)
|   |   +-- chunk_summarizer.py    Per-chunk LLM summaries (short chunks bypass LLM)
|   |   +-- entity_extractor.py    LLM structured entity/relationship extraction
|   |   |                           with content validators (prompt-leak, JSON-debris,
|   |   |                           prose-as-name, bibliographic-reference filters)
|   |   +-- graph_builder.py       Neo4j MERGE for entities + relationships + chunks
|   |   +-- community_detector.py  Leiden clustering + LLM summaries
|   |   +-- job_manager.py         SQLite job queue
|   +-- db/
|       +-- neo4j_schema.py        Constraints, indexes, vector indexes, full-text
+-- frontend/
|   +-- src/
|   |   +-- pages/
|   |   |   +-- Search.tsx         Answer/Keyword/Visual/Hybrid search
|   |   |   +-- Ingest.tsx         Upload form + job progress
|   |   |   +-- Manage.tsx         Documents, entities, GPU, communities, backup
|   |   |   +-- Viewer.tsx         Full-page viewer with navigation
|   |   +-- components/Layout.tsx  Sidebar nav with live health indicators
|   |   +-- api/                   Typed client + types
|   +-- vite.config.ts             Proxy + /app/ base path
+-- config/forgerag.toml           Active config (gitignored)
+-- config/forgerag.toml.example   Template
+-- systemd/forgerag-api.service   systemd unit
+-- scripts/
|   +-- install_neo4j.sh                  Neo4j Community 5.x installer
|   +-- seed_schema.py                    Apply Neo4j schema (idempotent)
|   +-- rebuild_chunks.py                 Phase 9 CLI rebuild: chunks + summaries
|   |                                      + BGE-M3 embeddings + entity re-extraction.
|   |                                      Flags: --doc-id, --only-missing, --skip-extract,
|   |                                      --extract-only
|   +-- bulk_autotag.py                   LLM auto-tag/categorize docs missing tags.
|   |                                      Flags: --dry-run, --limit, --doc-ids, --overwrite
|   +-- backup.sh                         CLI full backup (Neo4j dump + manifest, keeps last 5)
|   +-- restore.sh                        Restore from local backup or Google Drive
|   +-- neo4j-dump-helper.sh              Privileged helper: stop Neo4j, dump, restart
|   +-- gdrive_backup.py                  Upload graph JSON + manifest to Google Drive
|   +-- gdrive_restore.py                 Download latest backup from Google Drive
|   +-- canonicalize_materials_dryrun.py  Plan Tier 1 Material canonicalization
|   +-- canonicalize_materials_apply.py   Apply the plan (idempotent, per-group tx)
|   +-- canonicalize_entity_dryrun.py     Generalized canonicalization for any label
|   +-- canonicalize_entity_apply.py      (--label Equipment|Process|Standard|Material)
|   +-- dedup_entities_dryrun.py          Tier 2 fuzzy dedup: SequenceMatcher + safety guards
|   +-- dedup_entities_apply.py           Apply Tier 2 plan (reuses canonicalize_entity_apply)
|   +-- cleanup_numeric_garbage.py        Null LLM-debris values in Material numeric fields
+-- data/                          Runtime data (gitignored)
    +-- page_images/{hash}/        Full-resolution PNGs
    +-- reduced_images/{hash}/     Reduced JPGs
    +-- uploads/                   Staged PDFs (cleaned via admin endpoint)
    +-- backups/                   Graph JSON exports + Neo4j dumps
    +-- jobs.sqlite                Ingestion job queue
```

## LLM Model Notes

**Entity extraction** (Qwen 3.6 35B-A3B MoE, 3B active):
- Requires `use_json_schema = true` in config
- Thinking is disabled via `chat_template_kwargs.enable_thinking = False` in the API request payload (not the old `/no_think` prompt directive — Qwen 3.6 dropped support for the soft `/think` and `/nothink` switches). The `disable_thinking = true` config flag controls this
- Without thinking disabled, the model deliberates for hundreds of tokens before emitting JSON, regressing per-page latency from ~8 s to 30+ s
- Runs on RTX 6000 via LM Studio at ~135 tok/s, ~8-10 s per page
- LM Studio "Thinking" toggle should be OFF (the API-level `chat_template_kwargs` is the authoritative control)
- The `reasoning_content` fallback in `llm_service.py` handles the case where Qwen 3.6 routes output through the reasoning channel even with thinking disabled — no action needed

**Gemma 4 26B MoE**: breaks under strict JSON schema grammar (degenerate repetition). Use `use_json_schema = false`.

**GLM 4.7 Flash**: reasoning model, too slow for batch extraction (~25 tok/s). Outputs to `reasoning_content` field — the LLM client handles this via fallback.

## License

Code: MIT. Models have their own licenses:
- Nemotron ColEmbed: CC-BY-NC-4.0 (non-commercial)
- ColPali v1.3: MIT
- BGE-M3 / bge-reranker-v2-m3: MIT
- nomic-embed-text: Apache 2.0
- Docling + docling-models: MIT / Apache 2.0
