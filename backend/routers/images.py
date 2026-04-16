"""Serve page images from disk.

Routes:
    GET /images/{doc_hash}/{page_number}          -> full PNG
    GET /images/{doc_hash}/{page_number}/reduced  -> reduced JPG thumbnail

Path validation ensures doc_hash matches the SHA-256 format (hex chars only)
to prevent directory traversal attacks.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/images", tags=["images"])

_HASH_RE = re.compile(r"^[a-f0-9]{64}$")


def _validate_hash(doc_hash: str) -> None:
    if not _HASH_RE.match(doc_hash):
        raise HTTPException(status_code=400, detail="Invalid doc_hash format")


def _page_filename(page_number: int, ext: str) -> str:
    return f"page_{str(page_number).zfill(4)}.{ext}"


@router.get("/{doc_hash}/{page_number}")
async def get_page_image(
    doc_hash: str, page_number: int, request: Request
) -> FileResponse:
    """Return the full-resolution PNG for a page."""
    _validate_hash(doc_hash)
    if page_number < 1:
        raise HTTPException(status_code=400, detail="page_number must be >= 1")

    settings = request.app.state.settings
    data_dir = Path(settings.server.data_dir)
    path = data_dir / "page_images" / doc_hash / _page_filename(page_number, "png")

    if not path.exists():
        raise HTTPException(status_code=404, detail="Page image not found")

    return FileResponse(
        path,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/{doc_hash}/{page_number}/reduced")
async def get_reduced_image(
    doc_hash: str, page_number: int, request: Request
) -> FileResponse:
    """Return the reduced JPG for a page (thumbnail / preview)."""
    _validate_hash(doc_hash)
    if page_number < 1:
        raise HTTPException(status_code=400, detail="page_number must be >= 1")

    settings = request.app.state.settings
    data_dir = Path(settings.server.data_dir)
    path = data_dir / "reduced_images" / doc_hash / _page_filename(page_number, "jpg")

    if not path.exists():
        raise HTTPException(status_code=404, detail="Reduced page image not found")

    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/{doc_hash}/{page_number}/highlighted")
async def get_highlighted_image(
    doc_hash: str,
    page_number: int,
    request: Request,
    query: str = "",
) -> FileResponse:
    """Return the page image with a ColPali similarity heatmap overlay for the given query.

    First call for a (page, query) pair runs ColPali on the page image (~2-4s
    on RTX 6000). Result cached in data/highlighted_images/{hash}/.
    """
    from fastapi import Query as QueryParam  # local import to avoid shadowing

    _validate_hash(doc_hash)
    if page_number < 1:
        raise HTTPException(status_code=400, detail="page_number must be >= 1")
    if not query.strip():
        raise HTTPException(status_code=400, detail="query parameter is required")

    settings = request.app.state.settings
    data_dir = Path(settings.server.data_dir)
    source_path = data_dir / "page_images" / doc_hash / _page_filename(page_number, "png")
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Source page image not found")

    highlighter = getattr(request.app.state, "highlighter", None)
    if highlighter is None:
        raise HTTPException(
            status_code=503, detail="Highlighter not available — ColPali service not wired",
        )

    try:
        out_path = await highlighter.render(
            source_image_path=source_path,
            doc_hash=doc_hash,
            page_number=page_number,
            query=query,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Highlight render failed")
        raise HTTPException(status_code=500, detail=f"Highlight failed: {exc}") from exc

    return FileResponse(
        out_path,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )
