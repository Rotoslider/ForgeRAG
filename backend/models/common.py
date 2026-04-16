"""Standard response envelope used across all ForgeRAG endpoints.

Mirrors the pattern used by Choom's memory-server so that clients
(including the Choom skill) can reuse the same success/reason/data
structure everywhere.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ForgeResult(BaseModel):
    """Standard response envelope.

    - success: True if the operation succeeded
    - reason:  Human-readable error message (if success is False)
    - data:    Operation-specific payload (if success is True)
    """

    success: bool
    reason: str | None = None
    data: list[Any] | dict[str, Any] | None = None


class HealthPayload(BaseModel):
    """Payload returned by the /health endpoint."""

    status: str = "ok"
    service: str = "forgerag"
    version: str = "0.1.0"
    neo4j_connected: bool = False
    document_count: int = 0
    page_count: int = 0
    gpu_available: bool = False
    config_loaded: bool = True
    config_path: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
