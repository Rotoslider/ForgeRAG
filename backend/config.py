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
    # Text embedding model
    text_embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    text_embedding_dim: int = 768
    vlm_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"


class LLMSettings(BaseModel):
    endpoint: str = "http://localhost:8300/v1"
    model: str = "qwen2.5-72b-instruct-q4_k_m"
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout_seconds: int = 120
    # Some models (Gemma 4 MoE) break under strict JSON schema grammar
    # enforcement and produce repetitive junk. Others (GLM reasoning
    # variants) work great with it. Toggle per model.
    use_json_schema: bool = True


class IngestionSettings(BaseModel):
    pdf_dpi: int = 300
    reduction_percentage: int = 50
    reduction_min_dimension: int = 768
    colpali_batch_size: int = 10
    text_embedding_batch_size: int = 32
    max_concurrent_pdf_conversions: int = 4
    scanned_text_threshold_chars: int = 50


class GPUSettings(BaseModel):
    device: str = "cuda"
    model_idle_unload_seconds: int = 300
    max_vram_usage_pct: int = 80


class Settings(BaseModel):
    server: ServerSettings = Field(default_factory=ServerSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    gpu: GPUSettings = Field(default_factory=GPUSettings)

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
