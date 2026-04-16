"""Health check endpoints."""

from __future__ import annotations

import logging
from fastapi import APIRouter, Request

from backend.models.common import ForgeResult, HealthPayload

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/")
async def root(request: Request) -> ForgeResult:
    """Root endpoint — basic liveness check."""
    return await health(request)


@router.get("/health")
async def health(request: Request) -> ForgeResult:
    """Detailed health check — reports Neo4j connectivity, document counts, GPU status."""
    neo4j_svc = getattr(request.app.state, "neo4j", None)
    settings = getattr(request.app.state, "settings", None)

    payload = HealthPayload()
    if settings is not None:
        payload.config_loaded = True
        payload.config_path = str(
            getattr(request.app.state, "config_path", "defaults")
        )

    # Neo4j connectivity (optional — service can boot without Neo4j running)
    if neo4j_svc is not None:
        try:
            connected = await neo4j_svc.verify_connectivity()
            payload.neo4j_connected = connected
            if connected:
                counts = await neo4j_svc.get_counts()
                payload.document_count = counts.get("documents", 0)
                payload.page_count = counts.get("pages", 0)
        except Exception as exc:
            logger.warning("Neo4j health check failed: %s", exc)
            payload.details["neo4j_error"] = str(exc)

    # GPU availability check — optional import so Phase 1 doesn't require torch
    try:
        import torch  # type: ignore

        payload.gpu_available = torch.cuda.is_available()
        if payload.gpu_available:
            payload.details["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        payload.details["gpu_available_note"] = "torch not installed yet"

    return ForgeResult(success=True, data=payload.model_dump())
