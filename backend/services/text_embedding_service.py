"""Text embedding service.

Supports two model families with a common interface:

- **nomic-ai/nomic-embed-text-v1.5** (768-dim) — the original choice. Uses
  asymmetric prefixes ("search_document: ", "search_query: "). Small and
  fast (~600 MB VRAM).

- **BAAI/bge-m3** (1024-dim) — current default for Phase 2+. Stronger on
  technical content, produces dense + (optionally) learned sparse vectors.
  No prefixes needed.

Both produce L2-normalized float32 vectors suitable for cosine similarity
via dot product — matches what Neo4j's vector index expects.

Exposes:
- embed_documents(texts) — batched, returns float32 (N, dim) numpy array
- embed_query(text)     — single, returns float32 (dim,) numpy array
- dim                    — embedding dimension (model-dependent)

The model is loaded lazily on first use and kept resident until the GPU
manager decides to evict it. Implements the _Loadable protocol so it can
register with GPUManager.
"""

from __future__ import annotations

import logging
import threading
from typing import Iterable

import numpy as np

from backend.services.gpu_manager import GPUManager

logger = logging.getLogger(__name__)


def _estimate_vram_bytes(model_name: str) -> int:
    """Heuristic VRAM estimate per model family (fp16 weights + overhead)."""
    name = (model_name or "").lower()
    if "bge-m3" in name:
        return int(1.5 * 1024 * 1024 * 1024)  # ~1.5 GB
    if "nomic" in name:
        return 600 * 1024 * 1024  # ~600 MB
    return 1 * 1024 * 1024 * 1024  # fallback 1 GB


class TextEmbeddingService:
    """Wraps a sentence-transformers-compatible model for page text + queries.

    Safe to instantiate before the model is loaded — actual weights load
    on first embed call. Once loaded, stays in memory until GPUManager
    unloads it.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cuda",
        dim: int = 1024,
    ):
        self.model_name = model_name
        self.device = device
        self.dim = dim
        self._model = None
        self._lock = threading.Lock()  # protect lazy init
        # Prefix behavior varies by model family. Nomic requires asymmetric
        # prefixes; BGE-M3 doesn't need them and scores WORSE with them.
        name = model_name.lower()
        self._uses_nomic_prefixes = "nomic" in name
        self._trust_remote_code = "nomic" in name  # only nomic needs it

    # ------------------------------------------------------ GPUManager API

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        with self._lock:
            if self._model is None:
                return
            try:
                del self._model
                self._model = None
                import gc

                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
                logger.info("Unloaded text embedding model")
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to unload text embedding model: %s", exc)

    # ---------------------------------------------------------- lazy load

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:  # double-checked
                return
            from sentence_transformers import SentenceTransformer

            logger.info(
                "Loading text embedding model %s on %s",
                self.model_name, self.device,
            )
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=self._trust_remote_code,
            )
            # Verify the model's actual output dim matches what config says —
            # catches a silent mis-config where the vector index would be the
            # wrong size.
            actual_dim = self._model.get_sentence_embedding_dimension()
            if actual_dim is not None and actual_dim != self.dim:
                logger.warning(
                    "Config says text embedding dim=%d but %s reports dim=%d. "
                    "Using model's dim; check models.text_embedding_dim in "
                    "forgerag.toml.",
                    self.dim, self.model_name, actual_dim,
                )
                self.dim = actual_dim
            logger.info(
                "Text embedding model ready (dim=%d, nomic_prefix=%s)",
                self.dim, self._uses_nomic_prefixes,
            )

    # -------------------------------------------------------- embed ops

    def _apply_doc_prefix(self, texts: list[str]) -> list[str]:
        if self._uses_nomic_prefixes:
            return [f"search_document: {t}" for t in texts]
        return texts

    def _apply_query_prefix(self, text: str) -> str:
        if self._uses_nomic_prefixes:
            return f"search_query: {text}"
        return text

    def embed_documents(
        self, texts: Iterable[str], batch_size: int = 32
    ) -> np.ndarray:
        """Embed document texts. Returns (N, dim) float32.

        Empty strings get a zero vector — matches what Neo4j's vector
        index does at write time for missing embeddings.
        """
        self._ensure_loaded()
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        # Separate empty from non-empty so empties don't skew the model
        raw: list[str] = []
        keep_indices: list[int] = []
        for i, t in enumerate(texts):
            if t and t.strip():
                raw.append(t)
                keep_indices.append(i)

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        if raw:
            prefixed = self._apply_doc_prefix(raw)
            embs = self._model.encode(  # type: ignore[union-attr]
                prefixed,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # cosine-ready
            ).astype(np.float32)
            for src, dst in enumerate(keep_indices):
                out[dst] = embs[src]
        return out

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query. Returns (dim,) float32."""
        self._ensure_loaded()
        if not text or not text.strip():
            return np.zeros((self.dim,), dtype=np.float32)
        emb = self._model.encode(  # type: ignore[union-attr]
            [self._apply_query_prefix(text)],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)[0]
        return emb


def create_text_embedding_service(
    settings, gpu: GPUManager
) -> TextEmbeddingService:
    """Factory: create the service and register it with the GPU manager."""
    svc = TextEmbeddingService(
        model_name=settings.models.text_embedding_model,
        device=settings.gpu.device,
        dim=settings.models.text_embedding_dim,
    )
    gpu.register(
        name="text_embedding",
        handle=svc,
        est_vram_bytes=_estimate_vram_bytes(settings.models.text_embedding_model),
    )
    return svc
