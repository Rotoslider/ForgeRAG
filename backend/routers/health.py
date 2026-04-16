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

    # GPU availability — prefer the GPU manager which also reports VRAM + models
    gpu = getattr(request.app.state, "gpu", None)
    if gpu is not None:
        info = gpu.gpu_info()
        payload.gpu_available = bool(info.get("available", False))
        if payload.gpu_available:
            payload.details["gpu_name"] = info.get("device_name")
            total = info.get("vram_total_bytes", 0) or 0
            free = info.get("vram_free_bytes", 0) or 0
            if total:
                payload.details["vram_total_gb"] = round(total / 1e9, 1)
                payload.details["vram_free_gb"] = round(free / 1e9, 1)
            payload.details["models"] = info.get("models", [])
    else:
        try:
            import torch  # type: ignore

            payload.gpu_available = torch.cuda.is_available()
            if payload.gpu_available:
                payload.details["gpu_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            payload.details["gpu_available_note"] = "torch not installed yet"

    return ForgeResult(success=True, data=payload.model_dump())
