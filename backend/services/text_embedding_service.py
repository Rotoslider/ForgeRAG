"""Text embedding service using nomic-embed-text-v1.5.

nomic-embed-text-v1.5 is a 768-dim embedding model that supports asymmetric
retrieval — different prefixes for documents ("search_document: ") and queries
("search_query: "). It's also small enough to keep resident (~500MB VRAM) and
strong on technical content.

Exposes:
- embed_documents(texts) — batched, returns float32 (N, 768) numpy array
- embed_query(text)     — single, returns float32 (768,) numpy array

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

_MODEL_VRAM_ESTIMATE = 600 * 1024 * 1024  # ~600 MB


class TextEmbeddingService:
    """Wraps a sentence-transformers model for page text + queries.

    Safe to instantiate before the model is loaded — actual weights load
    on first embed call. Once loaded, stays in memory until GPUManager
    unloads it.
    """

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        device: str = "cuda",
        dim: int = 768,
    ):
        self.model_name = model_name
        self.device = device
        self.dim = dim
        self._model = None
        self._lock = threading.Lock()  # protect lazy init

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

            logger.info("Loading text embedding model %s on %s", self.model_name, self.device)
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True,  # nomic requires trust_remote_code
            )
            logger.info("Text embedding model ready")

    # -------------------------------------------------------- embed ops

    def embed_documents(
        self, texts: Iterable[str], batch_size: int = 32
    ) -> np.ndarray:
        """Embed document texts with the 'search_document: ' prefix.

        Returns a float32 array of shape (N, dim). Empty strings get a
        zero vector (same behavior as Neo4j vector search expects).
        """
        self._ensure_loaded()
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        # Separate empty from non-empty so empties don't skew the model
        prefixed: list[str] = []
        keep_indices: list[int] = []
        for i, t in enumerate(texts):
            if t and t.strip():
                prefixed.append(f"search_document: {t}")
                keep_indices.append(i)

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        if prefixed:
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
        """Embed a query with the 'search_query: ' prefix. Returns (dim,)."""
        self._ensure_loaded()
        if not text or not text.strip():
            return np.zeros((self.dim,), dtype=np.float32)
        emb = self._model.encode(  # type: ignore[union-attr]
            [f"search_query: {text}"],
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
        est_vram_bytes=_MODEL_VRAM_ESTIMATE,
    )
    return svc
