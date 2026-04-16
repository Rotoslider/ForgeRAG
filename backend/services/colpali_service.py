"""ColPali visual embedding service.

Adapted from /home/nuc1/projects/2025-03-20 wag RAG_Vision/v0.15.6_colpali/colpali_manager.py

ColPali produces multi-vector embeddings: one ~128-dim vector per token patch
after pooling. With pool_factor=3 for storage, a page typically has 300-500
vectors; with pool_factor=24 for search, ~50-100.

Storage: serialized as a contiguous float32 numpy array on Page.colpali_vectors
(bytes) with Page.colpali_vector_count so we can reshape back during search.

Search: MaxSim late-interaction reranking — for each query token, find its
best-matching document token and sum. Adapted from v0.15.6 milvus_manager.py
rerank_single_doc (lines 202-227).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Iterable

import numpy as np

from backend.services.gpu_manager import GPUManager

logger = logging.getLogger(__name__)

_MODEL_VRAM_ESTIMATE = 24 * 1024 * 1024 * 1024  # ~24 GB for ColPali v1.3 in bfloat16 on Blackwell


class ColPaliService:
    """Wraps the ColPali model for image embedding + query embedding.

    Lazy-loading: call ensure_loaded() before processing. Unload via
    unload() or the GPUManager's idle watcher.
    """

    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.3",
        device: str = "cuda",
        storage_pool_factor: int = 3,
        search_pool_factor: int = 24,
    ):
        self.model_name = model_name
        self.device = device
        self.storage_pool_factor = storage_pool_factor
        self.search_pool_factor = search_pool_factor

        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    # ------------------------------------------------------- GPUManager API

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        with self._lock:
            if self._model is None:
                return
            try:
                del self._model
                del self._processor
                self._model = None
                self._processor = None
                import gc

                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
                logger.info("Unloaded ColPali")
            except Exception as exc:  # noqa: BLE001
                logger.error("ColPali unload failed: %s", exc)

    # ---------------------------------------------------------- lazy load

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from colpali_engine.models import ColPali
            from colpali_engine.models.paligemma.colpali.processing_colpali import (
                ColPaliProcessor,
            )

            logger.info("Loading ColPali %s on %s (bfloat16)", self.model_name, self.device)
            self._model = ColPali.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ).eval()
            self._processor = ColPaliProcessor.from_pretrained(self.model_name)
            logger.info("ColPali ready")

    # -------------------------------------------------------- embed ops

    def _pool(self, embeddings, pool_factor: int):
        """Apply HierarchicalTokenPooler to compress per-patch embeddings.

        The colpali-engine 0.3.15 API takes no __init__ args; pool_factor moved
        onto pool_embeddings(). See token_pooling/hierarchical_token_pooling.py.
        """
        from colpali_engine.compression.token_pooling import HierarchicalTokenPooler

        pooler = HierarchicalTokenPooler()
        return pooler.pool_embeddings(
            embeddings,
            pool_factor=pool_factor,
            padding=True,
            padding_side=self._processor.tokenizer.padding_side,  # type: ignore[union-attr]
        )

    def embed_images(
        self,
        image_paths: Iterable[Path | str],
        *,
        pool_factor: int | None = None,
        progress_cb=None,
    ) -> list[np.ndarray]:
        """Embed page images. Returns a list of (K, D) float32 arrays,
        one per image, where K is the pooled token count and D is usually 128.

        Runs synchronously (callers should wrap in asyncio.to_thread).
        """
        self._ensure_loaded()
        import torch
        import torch.nn.functional as F
        from PIL import Image

        pf = pool_factor if pool_factor is not None else self.storage_pool_factor
        paths = [Path(p) for p in image_paths]
        total = len(paths)
        logger.info("ColPali: embedding %d images with pool_factor=%d", total, pf)

        results: list[np.ndarray] = []
        for i, path in enumerate(paths, start=1):
            # Defensive reload: if something unloaded us between iterations
            # (e.g. an explicit /system/models/colpali/unload), we re-load
            # rather than crash.
            if self._model is None or self._processor is None:
                self._ensure_loaded()

            try:
                img = Image.open(path).convert("RGB")
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to open %s: %s", path, exc)
                results.append(np.zeros((0, 128), dtype=np.float32))
                if progress_cb is not None:
                    progress_cb(i, total)
                continue

            inputs = self._processor.process_images([img])  # type: ignore[union-attr]
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}  # type: ignore[union-attr]

            with torch.no_grad():
                emb = self._model(**inputs)  # type: ignore[union-attr, misc]
                if pf > 1:
                    emb = self._pool(emb, pf)
                emb = F.normalize(emb.to(dtype=torch.float32), p=2, dim=-1)
                # emb: (1, K, D) — squeeze batch dim
                arr = emb[0].cpu().numpy().astype(np.float32)
            results.append(arr)

            if progress_cb is not None:
                try:
                    progress_cb(i, total)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("progress_cb raised: %s", exc)

            if i % 25 == 0 or i == total:
                logger.info("ColPali: %d/%d images embedded", i, total)

        return results

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a text query. Returns (K, D) float32 for MaxSim scoring."""
        self._ensure_loaded()
        import torch
        import torch.nn.functional as F

        # ColPali queries use the processor's process_queries path
        inputs = self._processor.process_queries([query])  # type: ignore[union-attr]
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}  # type: ignore[union-attr]
        with torch.no_grad():
            emb = self._model(**inputs)  # type: ignore[union-attr, misc]
            emb = F.normalize(emb.to(dtype=torch.float32), p=2, dim=-1)
            arr = emb[0].cpu().numpy().astype(np.float32)
        return arr


# ------------------------------------------------------------ serialization

def serialize_colpali(arr: np.ndarray) -> tuple[bytes, int]:
    """Pack a (K, D) float32 array into bytes + K. D is implied (128)."""
    arr = np.ascontiguousarray(arr.astype(np.float32))
    return arr.tobytes(), int(arr.shape[0])


def deserialize_colpali(blob: bytes, K: int, D: int = 128) -> np.ndarray:
    """Unpack bytes back to a (K, D) float32 array."""
    return np.frombuffer(blob, dtype=np.float32).reshape(K, D)


# ---------------------------------------------------------- MaxSim scoring

def maxsim_score(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
    """ColBERT-style late interaction: for each query token, max over doc tokens, sum.

    Arrays assumed L2-normalized so dot products are cosine similarities.
    Adapted from the Milvus reranker in v0.15.6_colpali/milvus_manager.py
    but without the extra heuristics — we'll re-add those once we have real queries
    to tune against.
    """
    if query_vecs.size == 0 or doc_vecs.size == 0:
        return 0.0
    # (Kq, D) @ (D, Kd) -> (Kq, Kd)
    sim = query_vecs @ doc_vecs.T
    # for each query token, take max over doc tokens, then sum across query tokens
    return float(np.sum(sim.max(axis=1)))


# ------------------------------------------------------------- factory

def create_colpali_service(settings, gpu: GPUManager) -> ColPaliService:
    svc = ColPaliService(
        model_name=settings.models.colpali_name,
        device=settings.gpu.device,
        storage_pool_factor=settings.models.colpali_pool_factor_storage,
        search_pool_factor=settings.models.colpali_pool_factor_search,
    )
    gpu.register(name="colpali", handle=svc, est_vram_bytes=_MODEL_VRAM_ESTIMATE)
    return svc
