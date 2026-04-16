"""Graph query endpoints — predefined templates + neighborhood exploration."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request

from backend.models.common import ForgeResult
from backend.models.graph import (
    EntityType,
    GraphExploreRequest,
    GraphQueryRequest,
    QueryTemplate,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph", tags=["graph"])


_ENTITY_LABEL = {
    "material": "Material",
    "process": "Process",
    "standard": "Standard",
    "equipment": "Equipment",
    "clause": "Clause",
}

_ENTITY_PK = {
    "material": "name",
    "process": "name",
    "standard": "code",
    "equipment": "name",
    "clause": "clause_id",
}


# Predefined query templates. Each template is a (cypher, required_params) pair.
# Params are substituted via $param syntax — never string-interpolated.
_QUERY_TEMPLATES: dict[QueryTemplate, tuple[str, list[str]]] = {
    "material_standards": (
        """
        MATCH (m:Material {name: $material})-[r:GOVERNED_BY]->(s:Standard)
        RETURN m.name AS material, s.code AS standard, s.organization AS org,
               r.support_count AS support_count, r.context AS context
        ORDER BY r.support_count DESC, s.code
        LIMIT $limit
        """,
        ["material"],
    ),
    "process_materials": (
        """
        MATCH (m:Material)-[r:COMPATIBLE_WITH_PROCESS]->(p:Process {name: $process})
        RETURN p.name AS process, m.name AS material,
               m.material_type AS material_type,
               r.support_count AS support_count, r.context AS context
        ORDER BY r.support_count DESC, m.name
        LIMIT $limit
        """,
        ["process"],
    ),
    "standard_cross_references": (
        """
        MATCH (s1:Standard {code: $standard})-[r:REFERENCES]->(s2:Standard)
        RETURN s1.code AS standard, s2.code AS referenced, s2.organization AS org,
               r.support_count AS support_count, r.context AS context
        ORDER BY r.support_count DESC, s2.code
        LIMIT $limit
        """,
        ["standard"],
    ),
    "material_properties": (
        """
        MATCH (m:Material {name: $material})
        RETURN m.name AS name, m.material_type AS material_type,
               m.uns_number AS uns_number, m.common_names AS common_names,
               m.tensile_strength_ksi AS tensile_strength_ksi,
               m.yield_strength_ksi AS yield_strength_ksi,
               m.hardness AS hardness,
               size([(m)<-[:MENTIONS_MATERIAL]-(:Page) | 1]) AS page_mentions
        """,
        ["material"],
    ),
    "equipment_requirements": (
        """
        MATCH (e:Equipment {name: $equipment})
        OPTIONAL MATCH (e)-[:GOVERNED_BY]->(s:Standard)
        WITH e, collect(DISTINCT s.code) AS standards
        RETURN e.name AS equipment, e.equipment_type AS equipment_type,
               standards
        LIMIT $limit
        """,
        ["equipment"],
    ),
    "page_entities": (
        """
        MATCH (p:Page {page_id: $page_id})
        OPTIONAL MATCH (p)-[:MENTIONS_MATERIAL]->(m:Material)
        OPTIONAL MATCH (p)-[:DESCRIBES_PROCESS]->(pr:Process)
        OPTIONAL MATCH (p)-[:REFERENCES_STANDARD]->(s:Standard)
        OPTIONAL MATCH (p)-[:MENTIONS_EQUIPMENT]->(e:Equipment)
        RETURN p.page_id AS page_id, p.page_number AS page_number,
               collect(DISTINCT m.name) AS materials,
               collect(DISTINCT pr.name) AS processes,
               collect(DISTINCT s.code) AS standards,
               collect(DISTINCT e.name) AS equipment
        """,
        ["page_id"],
    ),
    "entity_pages": (
        """
        MATCH (t) WHERE t.name = $entity_name OR t.code = $entity_name
        MATCH (p:Page)-[r]->(t)
        MATCH (d:Document)-[:HAS_PAGE]->(p)
        RETURN d.doc_id AS doc_id, d.title AS document_title,
               p.page_number AS page_number, type(r) AS rel_type,
               r.context AS context,
               d.file_hash AS file_hash
        ORDER BY d.title, p.page_number
        LIMIT $limit
        """,
        ["entity_name"],
    ),
}


@router.post("/query")
async def graph_query(body: GraphQueryRequest, request: Request) -> ForgeResult:
    """Run a predefined graph query template."""
    neo4j = request.app.state.neo4j
    tpl = _QUERY_TEMPLATES.get(body.query_type)
    if tpl is None:
        raise HTTPException(status_code=400, detail=f"Unknown query_type: {body.query_type}")
    cypher, required = tpl

    for p in required:
        if p not in body.parameters:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required parameter '{p}' for query_type '{body.query_type}'",
            )

    params = dict(body.parameters)
    params["limit"] = body.limit
    rows = await neo4j.run_query(cypher, params)
    return ForgeResult(success=True, data=rows)


@router.post("/explore")
async def graph_explore(body: GraphExploreRequest, request: Request) -> ForgeResult:
    """Get the N-hop neighborhood of an entity.

    Returns a flat list of {direction, relationship, neighbor_label, neighbor_name}
    triples. Callers can render this however they want — we don't ship a graph
    visualizer, per the original scope.
    """
    neo4j = request.app.state.neo4j
    label = _ENTITY_LABEL.get(body.entity_type)
    pk = _ENTITY_PK.get(body.entity_type)
    if label is None or pk is None:
        raise HTTPException(status_code=400, detail=f"Unknown entity_type: {body.entity_type}")

    # Safe to interpolate label/pk from whitelist; name goes through a parameter.
    cypher = f"""
        MATCH (n:{label} {{{pk}: $name}})
        OPTIONAL MATCH path = (n)-[*1..{body.depth}]-(neighbor)
        WHERE neighbor <> n
        WITH n, neighbor, relationships(path) AS rels, path
        RETURN DISTINCT
            labels(neighbor)[0] AS neighbor_label,
            coalesce(neighbor.name, neighbor.code, neighbor.clause_id, neighbor.page_id) AS neighbor_key,
            neighbor.name AS neighbor_name,
            [r IN rels | type(r)] AS path_types,
            length(path) AS distance
        ORDER BY distance, neighbor_label, neighbor_key
        LIMIT $limit
    """
    rows = await neo4j.run_query(cypher, {"name": body.entity_name, "limit": body.limit})
    return ForgeResult(success=True, data=rows)


@router.get("/entities/{entity_type}")
async def list_entities(
    entity_type: str,
    request: Request,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> ForgeResult:
    """List extracted entities of a type, with mention counts."""
    neo4j = request.app.state.neo4j
    label = _ENTITY_LABEL.get(entity_type)
    pk = _ENTITY_PK.get(entity_type)
    if label is None or pk is None:
        raise HTTPException(status_code=400, detail=f"Unknown entity_type: {entity_type}")

    rel_mapping = {
        "material": "MENTIONS_MATERIAL",
        "process": "DESCRIBES_PROCESS",
        "standard": "REFERENCES_STANDARD",
        "equipment": "MENTIONS_EQUIPMENT",
        "clause": "REFERENCES_CLAUSE",
    }
    rel = rel_mapping.get(entity_type, "MENTIONS_MATERIAL")

    cypher = f"""
        MATCH (n:{label})
        OPTIONAL MATCH (p:Page)-[:{rel}]->(n)
        RETURN n.{pk} AS key, properties(n) AS properties,
               count(DISTINCT p) AS page_mentions
        ORDER BY page_mentions DESC, n.{pk}
        SKIP $offset LIMIT $limit
    """
    rows = await neo4j.run_query(cypher, {"limit": limit, "offset": offset})
    return ForgeResult(success=True, data=rows)


@router.post("/build-communities")
async def build_communities(request: Request) -> ForgeResult:
    """Rebuild the hierarchical community layer (GraphRAG summaries).

    Global operation — scans all ingested pages with extracted entities,
    runs Leiden community detection at three resolution levels, generates
    LLM summaries for each community, and embeds them. Existing Community
    nodes are wiped and recreated.
    """
    import asyncio
    jobs = request.app.state.job_manager
    pipeline = request.app.state.pipeline

    if pipeline.community_detector is None:
        raise HTTPException(
            status_code=503,
            detail="LLM or text embedding service not available — cannot detect communities",
        )

    job = await jobs.create(
        source_path="(build-communities global)",
        filename="(all documents)",
        categories=[],
        tags=[],
    )
    asyncio.create_task(pipeline.run_communities_only(job.job_id))
    return ForgeResult(
        success=True,
        data={"job_id": job.job_id, "status": "queued"},
    )


@router.get("/communities")
async def list_communities(
    request: Request,
    level: int | None = Query(None, ge=0, le=3, description="Filter by hierarchy level"),
    limit: int = Query(50, ge=1, le=500),
) -> ForgeResult:
    """List communities with their summaries and page counts."""
    neo4j = request.app.state.neo4j
    where = "WHERE c.level = $level" if level is not None else ""
    params: dict = {"limit": limit}
    if level is not None:
        params["level"] = level
    rows = await neo4j.run_query(
        f"""
        MATCH (c:Community) {where}
        OPTIONAL MATCH (p:Page)-[:IN_COMMUNITY]->(c)
        RETURN c.community_id AS community_id,
               c.level AS level,
               c.resolution AS resolution,
               c.summary AS summary,
               c.member_count AS member_count,
               count(DISTINCT p) AS actual_page_count
        ORDER BY c.level DESC, c.member_count DESC
        LIMIT $limit
        """,
        params,
    )
    return ForgeResult(success=True, data=rows)


@router.get("/stats")
async def graph_stats(request: Request) -> ForgeResult:
    """Summary of the knowledge graph — total counts per label."""
    neo4j = request.app.state.neo4j
    rows = await neo4j.run_query(
        """
        OPTIONAL MATCH (d:Document) WITH count(d) AS documents
        OPTIONAL MATCH (pg:Page) WITH documents, count(pg) AS pages
        OPTIONAL MATCH (m:Material) WITH documents, pages, count(m) AS materials
        OPTIONAL MATCH (pr:Process) WITH documents, pages, materials, count(pr) AS processes
        OPTIONAL MATCH (s:Standard) WITH documents, pages, materials, processes, count(s) AS standards
        OPTIONAL MATCH (c:Clause) WITH documents, pages, materials, processes, standards, count(c) AS clauses
        OPTIONAL MATCH (e:Equipment) WITH documents, pages, materials, processes, standards, clauses, count(e) AS equipment
        OPTIONAL MATCH (cat:Category) WITH documents, pages, materials, processes, standards, clauses, equipment, count(cat) AS categories
        OPTIONAL MATCH (t:Tag) WITH documents, pages, materials, processes, standards, clauses, equipment, categories, count(t) AS tags
        OPTIONAL MATCH (com:Community)
        RETURN documents, pages, materials, processes, standards, clauses, equipment, categories, tags, count(com) AS communities
        """
    )
    return ForgeResult(success=True, data=rows[0] if rows else {})
