"""Pydantic models for search requests and responses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    categories: list[str] | None = None
    tags: list[str] | None = None
    document_ids: list[str] | None = None
    source_type: Literal["digital_native", "scanned", "hybrid"] | None = None


class SemanticSearchRequest(BaseModel):
    query: str = Field(..., description="Free-text query")
    limit: int = Field(default=10, ge=1, le=100)
    filters: SearchFilters | None = None


class VisualSearchRequest(BaseModel):
    query: str = Field(..., description="Free-text query")
    limit: int = Field(default=5, ge=1, le=50)
    candidate_pool: int = Field(
        default=50,
        ge=5,
        le=500,
        description="Candidates from text vector search to rerank with ColPali",
    )
    filters: SearchFilters | None = None


HybridStrategy = Literal["graph_boosted", "vector_first", "graph_first", "community"]


class HybridSearchRequest(BaseModel):
    query: str
    strategy: HybridStrategy = "graph_boosted"
    limit: int = Field(default=10, ge=1, le=100)
    filters: SearchFilters | None = None
    # Strategy-specific tunables
    boost_weight: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Score bonus per graph-linked entity match (graph_boosted)",
    )
    candidate_pool: int = Field(
        default=50,
        ge=5,
        le=500,
        description="Initial vector candidates before graph re-ranking / enrichment",
    )


class SearchHit(BaseModel):
    """One result row."""

    page_id: str
    doc_id: str
    document_title: str
    filename: str
    page_number: int
    score: float
    text_snippet: str | None = None
    image_url: str
    reduced_image_url: str
    categories: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
