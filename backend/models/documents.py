"""Pydantic models for Document, Page, Category, and Tag.

These map to Neo4j nodes of the same labels. Request models are used by
routers; response models are serialized back to clients.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


SourceType = Literal["digital_native", "scanned", "hybrid", "unknown"]


class DocumentMeta(BaseModel):
    """Document metadata — what we know about a PDF independent of its pages."""

    doc_id: str
    title: str
    filename: str
    file_hash: str
    page_count: int = 0
    file_size_bytes: int = 0
    ingested_at: datetime
    source_type: SourceType = "unknown"
    edition: str | None = None
    publisher: str | None = None
    year: int | None = None

    # Organization
    categories: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class PageMeta(BaseModel):
    """Page metadata — what's stored on a :Page node."""

    page_id: str
    doc_id: str
    page_number: int
    image_path: str
    reduced_image_path: str | None = None
    extracted_text: str = ""
    text_char_count: int = 0
    source_type: SourceType = "unknown"
    has_text_embedding: bool = False
    has_colpali_embedding: bool = False


class Category(BaseModel):
    """Hierarchical category. Root categories have parent_name=None."""

    name: str
    parent_name: str | None = None
    description: str | None = None


class Tag(BaseModel):
    """Freeform tag applied to documents."""

    name: str


# Request bodies -----------------------------------------------------------

class CategoryCreate(BaseModel):
    name: str
    parent_name: str | None = None
    description: str | None = None


class TagCreate(BaseModel):
    name: str


class DocumentListFilter(BaseModel):
    """Query filter for GET /documents."""

    category: str | None = None
    tag: str | None = None
    source_type: SourceType | None = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
