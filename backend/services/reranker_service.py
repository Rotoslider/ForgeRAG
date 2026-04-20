"""Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

Post-retrieval step: take top-N candidates from hybrid search, score each
(query, passage) pair with the cross-encoder, return the top-K by rerank
score. Cross-encoders see both sides of the pair together and produce
much better relevance than bi-encoder similarity — at the cost of extra
latency per candidate. Typical setup: retrieve top-50, rerank to top-10.

Lazy-loaded like TextEmbeddingService: the model only materializes on
first score call and unloads when GPUManager evicts it for idleness.
"""

from __future__ import annotations

import logging
import threading
from typing import Sequence

from backend.services.gpu_manager import GPUManager

logger = logging.getLogger(__name__)

_MODEL_VRAM_ESTIMATE = int(1.2 * 1024 * 1024 * 1024)  # ~1.2 GB fp16


class RerankerService:
    """bge-reranker-v2-m3 cross-encoder wrapper."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    # ---------------------------------------------------- GPUManager API

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        with self._lock:
            if self._model is None:
                return
            try:
                del self._model
                del self._tokenizer
                self._model = None
                self._tokenizer = None
                import gc
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
                logger.info("Unloaded reranker model")
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to unload reranker: %s", exc)

    # ---------------------------------------------------------- lazy load

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            logger.info(
                "Loading reranker %s on %s", self.model_name, self.device
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = (
                AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, torch_dtype=torch.float16,
                )
                .to(self.device)
                .eval()
            )
            logger.info("Reranker ready")

    # -------------------------------------------------------- score

    def score_pairs(
        self,
        query: str,
        passages: Sequence[str],
        batch_size: int = 32,
        max_length: int = 512,
    ) -> list[float]:
        """Score each (query, passage) pair. Higher score = more relevant.
        Output is raw logits, not normalized — only meaningful for
        ranking within a single query call."""
        if not passages:
            return []
        self._ensure_loaded()

        import torch

        pairs = [[query, p or ""] for p in passages]
        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i : i + batch_size]
                tok = self._tokenizer(  # type: ignore[misc]
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self._model(**tok).logits.view(-1).float()  # type: ignore[misc]
                scores.extend(logits.cpu().tolist())
        return scores


def create_reranker_service(settings, gpu: GPUManager) -> RerankerService:
    """Factory: build the service and register with the GPU manager.

    The reranker is mandatory for the hybrid RRF strategy but falls back
    gracefully (skipped) if its model isn't available at query time.
    """
    svc = RerankerService(
        model_name=getattr(
            settings.models, "reranker_model", "BAAI/bge-reranker-v2-m3"
        ),
        device=settings.gpu.device,
    )
    gpu.register(
        name="reranker",
        handle=svc,
        est_vram_bytes=_MODEL_VRAM_ESTIMATE,
    )
    return svc
