"""System / GPU management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.models.common import ForgeResult

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/gpu")
async def get_gpu_status(request: Request) -> ForgeResult:
    """Report VRAM usage and loaded model list."""
    gpu = getattr(request.app.state, "gpu", None)
    if gpu is None:
        return ForgeResult(success=True, data={"available": False, "reason": "GPU manager not initialized"})
    return ForgeResult(success=True, data=gpu.gpu_info())


@router.post("/models/{name}/unload")
async def unload_model(name: str, request: Request) -> ForgeResult:
    """Explicitly unload a registered model to free VRAM."""
    gpu = getattr(request.app.state, "gpu", None)
    if gpu is None:
        raise HTTPException(503, "GPU manager not initialized")
    ok = await gpu.unload_model(name)
    return ForgeResult(
        success=True,
        data={"name": name, "unloaded": ok},
    )
