"""Configuration loading for ForgeRAG.

Loads settings from a TOML file with env var overrides. The TOML file path
defaults to config/forgerag.toml in the project root but can be overridden
via the FORGERAG_CONFIG environment variable.

Secrets like the Neo4j password are never stored in the TOML — the TOML
references an env var name and we resolve it at load time.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "forgerag.toml"


class ServerSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8200
    data_dir: Path = PROJECT_ROOT / "data"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    # Bearer token required for non-localhost API requests (empty = auth
    # disabled, with a loud startup warning). FORGERAG_API_TOKEN env wins.
    api_token: str = ""
    # Optional second token with read-only scope: lets remote agents (e.g. a
    # learning harness on another machine) search, read documents/pages, and
    # explore the graph, but NOT mutate state (no ingest/delete/rebuild/admin).
    # Empty = disabled (read-only access then requires the full api_token).
    # FORGERAG_API_TOKEN_READONLY env wins.
    api_token_readonly: str = ""


class Neo4jSettings(BaseModel):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password_env: str = "NEO4J_PASSWORD"
    database: str = "forgerag"
    max_connection_pool_size: int = 50
    connection_acquisition_timeout: int = 60

    @property
    def password(self) -> str:
        """Resolve the Neo4j password from the configured environment variable."""
        pw = os.environ.get(self.password_env, "")
        return pw


class ModelSettings(BaseModel):
    # Visual retrieval model — the model that embeds page images for MaxSim scoring.
    # Default is Nemotron ColEmbed 4B (significant upgrade from ColPali v1.3).
    # Set to "vidore/colpali-v1.3" to use the original ColPali.
    visual_model_name: str = "nvidia/nemotron-colembed-vl-4b-v2"
    visual_model_type: str = "nemotron"  # "nemotron" or "colpali"
    visual_embed_dim: int = 128  # projection target (128 retains 96.8% accuracy)
    # Legacy ColPali settings (only used when visual_model_type = "colpali")
    colpali_name: str = "vidore/colpali-v1.3"
    colpali_pool_factor_storage: int = 3
    colpali_pool_factor_search: int = 24
    # Text embedding model — BGE-M3 (1024-dim). Strong on technical content
    # and integrates naturally with bge-reranker-v2-m3 as a post-retrieval
    # step. Switch back to "nomic-ai/nomic-embed-text-v1.5" (768-dim) if
    # you need the smaller memory footprint.
    text_embedding_model: str = "BAAI/bge-m3"
    text_embedding_dim: int = 1024
    # Cross-encoder reranker — runs over top-K results from hybrid retrieval.
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    vlm_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"


class LLMSettings(BaseModel):
    endpoint: str = "http://localhost:1234/v1"
    model: str = "qwen/qwen3.6-35b-a3b"
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout_seconds: int = 300
    # Cap on in-flight HTTP requests to the LLM server. Keep this at or
    # below the server's own concurrency (LM Studio "Max Concurrent
    # Predictions", llama-server --parallel): requests beyond that queue
    # server-side, where queue-wait counts against timeout_seconds and
    # requests time out before inference ever starts. Excess callers wait
    # client-side instead (no timeout while waiting for a slot). Default 2
    # is safe for an unconfigured single-stream server; raise via toml to
    # match the server (see config/forgerag.toml [llm]).
    max_concurrent_requests: int = 2
    # Some models (Gemma 4 MoE) break under strict JSON schema grammar
    # enforcement and produce repetitive junk. Others (GLM reasoning
    # variants) work great with it. Toggle per model.
    use_json_schema: bool = True
    # Qwen3.6 disables thinking via chat_template_kwargs.enable_thinking
    # rather than the legacy "/no_think" prompt directive. LM Studio
    # forwards this kwarg to llama.cpp's chat-template renderer; models
    # that don't recognize it (DeepSeek, GLM, Llama, Gemma) ignore it
    # silently, so leaving it on is safe across the board.
    disable_thinking: bool = True


class IngestionSettings(BaseModel):
    pdf_dpi: int = 300
    reduction_percentage: int = 50
    reduction_min_dimension: int = 768
    colpali_batch_size: int = 10
    text_embedding_batch_size: int = 32
    max_concurrent_pdf_conversions: int = 4
    scanned_text_threshold_chars: int = 50
    # How many ingestion jobs may run at once. Each upload starts its own
    # background task; without a cap, adding dozens of PDFs at once stampedes
    # the job DB and the GPU. Keep this small.
    max_concurrent_ingestions: int = 3


class GPUSettings(BaseModel):
    device: str = "cuda"
    model_idle_unload_seconds: int = 300
    max_vram_usage_pct: int = 80


class BackupSettings(BaseModel):
    destination: str = ""  # local path like /mnt/nas/forgerag-backups
    include_images: bool = True
    include_pdfs: bool = True
    gdrive_enabled: bool = True
    gdrive_dump: bool = False  # upload the large Neo4j dump to Drive (8-14 GB)


class Settings(BaseModel):
    server: ServerSettings = Field(default_factory=ServerSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    gpu: GPUSettings = Field(default_factory=GPUSettings)
    backup: BackupSettings = Field(default_factory=BackupSettings)

    @classmethod
    def from_toml(cls, path: Path | str) -> Settings:
        path = Path(path)
        if not path.exists():
            # Missing config file is not fatal — fall back to defaults so the
            # service can still boot for diagnostic purposes. Callers can warn.
            return cls()
        with path.open("rb") as f:
            data: dict[str, Any] = tomllib.load(f)
        return cls.model_validate(data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process (cached). Respects FORGERAG_CONFIG."""
    config_path = os.environ.get("FORGERAG_CONFIG", str(DEFAULT_CONFIG_PATH))
    return Settings.from_toml(config_path)
