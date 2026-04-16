"""Page image rendering + highlight overlay.

For a given query, produces a heatmap overlay on the page image showing
which regions ColPali considers most relevant. This gives users an instant
visual check on whether a retrieved page actually contains what they asked
about — the original anti-hallucination workflow from the user's previous
Milvus setup.

Implementation:
1. Load the page image + ColPali model (via GPU manager).
2. Re-embed the page at pool_factor=1 (no pooling — we need per-patch tokens).
3. Embed the query.
4. Compute per-patch similarity (max over query tokens).
5. Upsample the (h/14 × w/14) patch grid to image size, render as a warm
   colormap, alpha-blend over the original.
6. Cache the result on disk keyed by (doc_hash, page, query_hash).

The initial embedding cost (~2-4 seconds per page on the RTX 6000) means
we don't pre-generate these — they're on-demand.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from backend.services.colpali_service import ColPaliService
from backend.services.gpu_manager import GPUManager

logger = logging.getLogger(__name__)


class ImageHighlighter:
    def __init__(
        self,
        *,
        data_dir: Path,
        colpali: ColPaliService,
        gpu: GPUManager,
        alpha: float = 0.45,
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "highlighted_images"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.colpali = colpali
        self.gpu = gpu
        self.alpha = alpha

    def cache_path(self, doc_hash: str, page_number: int, query: str) -> Path:
        qh = hashlib.sha256(query.encode()).hexdigest()[:16]
        return self.cache_dir / doc_hash / f"page_{page_number:04d}_{qh}.png"

    def is_cached(self, doc_hash: str, page_number: int, query: str) -> bool:
        return self.cache_path(doc_hash, page_number, query).exists()

    async def render(
        self,
        *,
        source_image_path: Path,
        doc_hash: str,
        page_number: int,
        query: str,
    ) -> Path:
        """Generate the highlighted image. Returns the path to the cached PNG."""
        out_path = self.cache_path(doc_hash, page_number, query)
        if out_path.exists():
            return out_path

        out_path.parent.mkdir(parents=True, exist_ok=True)

        import asyncio
        async with self.gpu.load_scope("colpali"):
            # Run the heavy stuff in a worker thread
            await asyncio.to_thread(
                self._render_sync,
                source_image_path=Path(source_image_path),
                out_path=out_path,
                query=query,
            )
        return out_path

    def _render_sync(
        self, *, source_image_path: Path, out_path: Path, query: str
    ) -> None:
        """Synchronous rendering — expects to already be inside GPU scope."""
        from colpali_engine.interpretability import (
            get_similarity_maps_from_embeddings,
        )
        import torch
        import torch.nn.functional as F

        self.colpali._ensure_loaded()
        model = self.colpali._model
        processor = self.colpali._processor
        assert model is not None and processor is not None

        img = Image.open(source_image_path).convert("RGB")

        # Image: process + embed WITHOUT pooling so we get per-patch tokens
        image_inputs = processor.process_images([img])
        image_inputs = {k: v.to(model.device) for k, v in image_inputs.items()}

        # Query: embed
        query_inputs = processor.process_queries([query])
        query_inputs = {k: v.to(model.device) for k, v in query_inputs.items()}

        with torch.no_grad():
            image_embeddings = model(**image_inputs)  # (1, N_img_tokens, D)
            query_embeddings = model(**query_inputs)  # (1, N_q_tokens, D)
            image_embeddings = F.normalize(image_embeddings.to(torch.float32), p=2, dim=-1)
            query_embeddings = F.normalize(query_embeddings.to(torch.float32), p=2, dim=-1)

            # Patch grid shape — use processor helpers when available, else
            # compute from the image processor's patch size.
            n_patches = _compute_n_patches(processor, img)
            image_mask = processor.get_image_mask(image_inputs)  # type: ignore[attr-defined]

            sim_maps = get_similarity_maps_from_embeddings(
                image_embeddings=image_embeddings,
                query_embeddings=query_embeddings,
                n_patches=n_patches,
                image_mask=image_mask,
            )
            # sim_maps: List[Tensor], one per batch item. Each (N_q_tokens, h_patches, w_patches)
            sm = sim_maps[0]  # (N_q_tokens, h, w)
            # Aggregate across query tokens — average, then max-clip to 0
            heatmap = sm.mean(dim=0).clamp(min=0).cpu().numpy()

        # Overlay onto the original image
        overlaid = _overlay_heatmap(img, heatmap, alpha=self.alpha)
        overlaid.save(out_path, "PNG", optimize=True)


def _compute_n_patches(processor, img: Image.Image) -> tuple[int, int]:
    """Figure out the patch grid shape for the image. Falls back to a
    reasonable default for ColPali v1.3 (SigLIP 448 with patch size 14
    -> 32×32 patches) if we can't introspect."""
    try:
        # Newer colpali versions expose this helper
        return processor.get_n_patches(image_size=img.size)  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        pass
    # Default for PaliGemma + SigLIP 448 (patch size 14)
    return (32, 32)


def _overlay_heatmap(
    img: Image.Image, heatmap: np.ndarray, alpha: float = 0.45
) -> Image.Image:
    """Upsample heatmap to image size, apply warm colormap, alpha-blend."""
    import matplotlib.cm as cm
    from PIL import Image as PILImage

    # Normalize heatmap to [0, 1]
    hm = heatmap.astype(np.float32)
    lo, hi = float(hm.min()), float(hm.max())
    if hi > lo:
        hm = (hm - lo) / (hi - lo)
    else:
        hm = np.zeros_like(hm)

    # Apply colormap -> RGBA in [0,1]
    colored = cm.get_cmap("inferno")(hm)  # (h, w, 4)
    colored = (colored[..., :3] * 255).astype(np.uint8)  # drop alpha channel

    # Upsample to the image size
    heat_img = PILImage.fromarray(colored, mode="RGB").resize(img.size, PILImage.BILINEAR)

    # Blend
    return PILImage.blend(img.convert("RGB"), heat_img, alpha=alpha)
