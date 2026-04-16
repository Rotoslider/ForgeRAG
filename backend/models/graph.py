"""Pydantic models for graph queries and exploration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


QueryTemplate = Literal[
    "material_standards",         # Which standards govern a material?
    "process_materials",          # Which materials are compatible with a process?
    "standard_cross_references",  # Which standards does this one reference?
    "material_properties",        # All properties for a named material
    "equipment_requirements",     # Standards/materials/processes for an equipment type
    "page_entities",              # All entities on a given page
    "entity_pages",               # All pages mentioning a named entity
]


EntityType = Literal["material", "process", "standard", "equipment", "clause"]


class GraphQueryRequest(BaseModel):
    query_type: QueryTemplate
    parameters: dict[str, str] = Field(default_factory=dict)
    limit: int = Field(default=50, ge=1, le=500)


class GraphExploreRequest(BaseModel):
    entity_type: EntityType
    entity_name: str
    depth: int = Field(default=1, ge=1, le=3)
    limit: int = Field(default=50, ge=1, le=500)
