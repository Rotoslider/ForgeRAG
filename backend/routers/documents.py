"""Document, Category, and Tag CRUD endpoints."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

from backend.models.common import ForgeResult
from backend.models.documents import CategoryCreate, TagCreate

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])


# Documents ----------------------------------------------------------------

@router.get("/documents")
async def list_documents(
    request: Request,
    category: str | None = Query(None, description="Filter by category name"),
    tag: str | None = Query(None, description="Filter by tag name"),
    source_type: str | None = Query(None, description="digital_native/scanned/hybrid"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> ForgeResult:
    """List documents with optional filtering. Includes page count and metadata."""
    neo4j = request.app.state.neo4j

    where_clauses = []
    params: dict = {"limit": limit, "offset": offset}

    if category:
        where_clauses.append("EXISTS { (d)-[:IN_CATEGORY]->(:Category {name: $category}) }")
        params["category"] = category
    if tag:
        where_clauses.append("EXISTS { (d)-[:TAGGED_WITH]->(:Tag {name: $tag}) }")
        params["tag"] = tag
    if source_type:
        where_clauses.append("d.source_type = $source_type")
        params["source_type"] = source_type

    where = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    query = f"""
        MATCH (d:Document){where}
        RETURN d.doc_id AS doc_id,
               d.title AS title,
               d.filename AS filename,
               d.file_hash AS file_hash,
               d.page_count AS page_count,
               d.file_size_bytes AS file_size_bytes,
               d.source_type AS source_type,
               toString(d.ingested_at) AS ingested_at,
               [(d)-[:IN_CATEGORY]->(c) | c.name] AS categories,
               [(d)-[:TAGGED_WITH]->(t) | t.name] AS tags
        ORDER BY d.ingested_at DESC
        SKIP $offset LIMIT $limit
    """
    rows = await neo4j.run_query(query, params)
    return ForgeResult(success=True, data=rows)


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str, request: Request) -> ForgeResult:
    """Get a single document with full metadata."""
    neo4j = request.app.state.neo4j
    rows = await neo4j.run_query(
        """
        MATCH (d:Document {doc_id: $doc_id})
        RETURN d.doc_id AS doc_id,
               d.title AS title,
               d.filename AS filename,
               d.file_hash AS file_hash,
               d.page_count AS page_count,
               d.file_size_bytes AS file_size_bytes,
               d.source_type AS source_type,
               toString(d.ingested_at) AS ingested_at,
               [(d)-[:IN_CATEGORY]->(c) | c.name] AS categories,
               [(d)-[:TAGGED_WITH]->(t) | t.name] AS tags
        """,
        {"doc_id": doc_id},
    )
    if not rows:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return ForgeResult(success=True, data=rows[0])


@router.post("/documents/{doc_id}/extract-entities")
async def extract_entities(doc_id: str, request: Request) -> ForgeResult:
    """Run only the LLM entity-extraction step for an existing document.

    Useful for documents ingested before Phase 4 or after changing the
    extraction prompt. Requires the LLM service to be configured and reachable.
    """
    import asyncio
    neo4j = request.app.state.neo4j
    jobs = request.app.state.job_manager
    pipeline = request.app.state.pipeline

    rows = await neo4j.run_query(
        "MATCH (d:Document {doc_id: $id}) RETURN d.filename AS f",
        {"id": doc_id},
    )
    if not rows:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    if pipeline.entity_extractor is None:
        raise HTTPException(
            status_code=503,
            detail="LLM service not configured or unreachable — cannot extract entities",
        )

    filename = rows[0]["f"]
    job = await jobs.create(
        source_path=f"(extract-entities of {doc_id})",
        filename=filename,
        categories=[],
        tags=[],
    )
    asyncio.create_task(pipeline.run_extraction_only(job.job_id, doc_id))
    return ForgeResult(
        success=True,
        data={"job_id": job.job_id, "doc_id": doc_id, "status": "queued"},
    )


@router.post("/documents/{doc_id}/reembed")
async def reembed_document(doc_id: str, request: Request) -> ForgeResult:
    """Re-run only the embedding steps (text + ColPali) for an existing document.

    Useful for documents ingested under earlier phases that don't yet have
    embeddings. Creates a new job for progress tracking.
    """
    import asyncio
    neo4j = request.app.state.neo4j
    jobs = request.app.state.job_manager
    pipeline = request.app.state.pipeline

    rows = await neo4j.run_query(
        "MATCH (d:Document {doc_id: $id}) RETURN d.filename AS f",
        {"id": doc_id},
    )
    if not rows:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    filename = rows[0]["f"]
    job = await jobs.create(
        source_path=f"(reembed of {doc_id})",
        filename=filename,
        categories=[],
        tags=[],
    )
    asyncio.create_task(pipeline.run_embeddings_only(job.job_id, doc_id))
    return ForgeResult(
        success=True,
        data={"job_id": job.job_id, "doc_id": doc_id, "status": "queued"},
    )


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, request: Request) -> ForgeResult:
    """Delete a document and all its pages. Also removes page images from disk."""
    neo4j = request.app.state.neo4j
    settings = request.app.state.settings

    # Look up the file_hash first (needed to find on-disk images)
    rows = await neo4j.run_query(
        "MATCH (d:Document {doc_id: $doc_id}) RETURN d.file_hash AS h",
        {"doc_id": doc_id},
    )
    if not rows:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    file_hash = rows[0]["h"]

    # Detach-delete the document and its pages (keeps category/tag nodes)
    await neo4j.run_write(
        """
        MATCH (d:Document {doc_id: $doc_id})
        OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Page)
        DETACH DELETE d, p
        """,
        {"doc_id": doc_id},
    )

    # Remove page images from disk (best-effort)
    data_dir = Path(settings.server.data_dir)
    removed = []
    for sub in ("page_images", "reduced_images"):
        folder = data_dir / sub / file_hash
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
            removed.append(str(folder))

    return ForgeResult(
        success=True,
        data={"doc_id": doc_id, "folders_removed": removed},
    )


@router.post("/documents/{doc_id}/tags")
async def add_document_tag(doc_id: str, body: TagCreate, request: Request) -> ForgeResult:
    """Add a tag to an existing document. Creates the tag if it doesn't exist."""
    neo4j = request.app.state.neo4j
    rows = await neo4j.run_query(
        "MATCH (d:Document {doc_id: $id}) RETURN d.doc_id", {"id": doc_id}
    )
    if not rows:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    await neo4j.run_write(
        """
        MERGE (t:Tag {name: $tag})
        WITH t
        MATCH (d:Document {doc_id: $doc_id})
        MERGE (d)-[:TAGGED_WITH]->(t)
        """,
        {"tag": body.name, "doc_id": doc_id},
    )
    return ForgeResult(success=True, data={"doc_id": doc_id, "tag": body.name})


@router.delete("/documents/{doc_id}/tags/{tag_name}")
async def remove_document_tag(doc_id: str, tag_name: str, request: Request) -> ForgeResult:
    """Remove a tag from a document (doesn't delete the tag node itself)."""
    neo4j = request.app.state.neo4j
    await neo4j.run_write(
        """
        MATCH (d:Document {doc_id: $doc_id})-[r:TAGGED_WITH]->(t:Tag {name: $tag})
        DELETE r
        """,
        {"doc_id": doc_id, "tag": tag_name},
    )
    return ForgeResult(success=True, data={"doc_id": doc_id, "removed_tag": tag_name})


@router.post("/documents/{doc_id}/categories")
async def add_document_category(doc_id: str, body: CategoryCreate, request: Request) -> ForgeResult:
    """Add a category to an existing document. Creates the category if it doesn't exist."""
    neo4j = request.app.state.neo4j
    rows = await neo4j.run_query(
        "MATCH (d:Document {doc_id: $id}) RETURN d.doc_id", {"id": doc_id}
    )
    if not rows:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    await neo4j.run_write(
        """
        MERGE (c:Category {name: $cat})
        ON CREATE SET c.description = $desc
        WITH c
        MATCH (d:Document {doc_id: $doc_id})
        MERGE (d)-[:IN_CATEGORY]->(c)
        """,
        {"cat": body.name, "doc_id": doc_id, "desc": body.description},
    )
    return ForgeResult(success=True, data={"doc_id": doc_id, "category": body.name})


@router.delete("/documents/{doc_id}/categories/{cat_name}")
async def remove_document_category(doc_id: str, cat_name: str, request: Request) -> ForgeResult:
    """Remove a category from a document."""
    neo4j = request.app.state.neo4j
    await neo4j.run_write(
        """
        MATCH (d:Document {doc_id: $doc_id})-[r:IN_CATEGORY]->(c:Category {name: $cat})
        DELETE r
        """,
        {"doc_id": doc_id, "cat": cat_name},
    )
    return ForgeResult(success=True, data={"doc_id": doc_id, "removed_category": cat_name})


@router.get("/documents/{doc_id}/pages")
async def list_document_pages(
    doc_id: str,
    request: Request,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> ForgeResult:
    """List pages of a document with their metadata (no full text in list view)."""
    neo4j = request.app.state.neo4j
    rows = await neo4j.run_query(
        """
        MATCH (d:Document {doc_id: $doc_id})-[:HAS_PAGE]->(p:Page)
        RETURN p.page_id AS page_id,
               p.page_number AS page_number,
               p.image_path AS image_path,
               p.reduced_image_path AS reduced_image_path,
               p.text_char_count AS text_char_count,
               p.source_type AS source_type
        ORDER BY p.page_number
        SKIP $offset LIMIT $limit
        """,
        {"doc_id": doc_id, "limit": limit, "offset": offset},
    )
    return ForgeResult(success=True, data=rows)


@router.get("/documents/{doc_id}/pages/{page_number}")
async def get_page_detail(
    doc_id: str, page_number: int, request: Request
) -> ForgeResult:
    """Get a single page with full extracted text. Useful for inspection and debugging."""
    neo4j = request.app.state.neo4j
    rows = await neo4j.run_query(
        """
        MATCH (d:Document {doc_id: $doc_id})-[:HAS_PAGE]->(p:Page {page_number: $pn})
        RETURN p.page_id AS page_id,
               p.page_number AS page_number,
               p.image_path AS image_path,
               p.reduced_image_path AS reduced_image_path,
               p.text_char_count AS text_char_count,
               p.source_type AS source_type,
               p.extracted_text AS extracted_text
        """,
        {"doc_id": doc_id, "pn": page_number},
    )
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"Page {page_number} of document {doc_id} not found",
        )
    return ForgeResult(success=True, data=rows[0])


# Categories ---------------------------------------------------------------

@router.get("/categories")
async def list_categories(request: Request) -> ForgeResult:
    """List all categories (hierarchy included via parent_name)."""
    neo4j = request.app.state.neo4j
    rows = await neo4j.run_query(
        """
        MATCH (c:Category)
        OPTIONAL MATCH (c)-[:SUBCATEGORY_OF]->(parent:Category)
        RETURN c.name AS name,
               c.description AS description,
               parent.name AS parent_name,
               size([(c)<-[:IN_CATEGORY]-(:Document) | 1]) AS document_count
        ORDER BY c.name
        """
    )
    return ForgeResult(success=True, data=rows)


@router.post("/categories")
async def create_category(body: CategoryCreate, request: Request) -> ForgeResult:
    """Create or update a category. Idempotent (MERGE)."""
    neo4j = request.app.state.neo4j
    await neo4j.run_write(
        """
        MERGE (c:Category {name: $name})
        ON CREATE SET c.description = $description
        ON MATCH  SET c.description = coalesce($description, c.description)
        """,
        {"name": body.name, "description": body.description},
    )
    if body.parent_name:
        await neo4j.run_write(
            """
            MATCH (c:Category {name: $name})
            MERGE (p:Category {name: $parent})
            MERGE (c)-[:SUBCATEGORY_OF]->(p)
            """,
            {"name": body.name, "parent": body.parent_name},
        )
    return ForgeResult(success=True, data={"name": body.name})


@router.delete("/categories/{name}")
async def delete_category(name: str, request: Request) -> ForgeResult:
    """Delete a category. Documents tagged with it lose the IN_CATEGORY relationship."""
    neo4j = request.app.state.neo4j
    await neo4j.run_write(
        "MATCH (c:Category {name: $name}) DETACH DELETE c",
        {"name": name},
    )
    return ForgeResult(success=True, data={"name": name})


# Tags ---------------------------------------------------------------------

@router.get("/tags")
async def list_tags(request: Request) -> ForgeResult:
    """List all tags with document counts."""
    neo4j = request.app.state.neo4j
    rows = await neo4j.run_query(
        """
        MATCH (t:Tag)
        RETURN t.name AS name,
               size([(t)<-[:TAGGED_WITH]-(:Document) | 1]) AS document_count
        ORDER BY t.name
        """
    )
    return ForgeResult(success=True, data=rows)


@router.post("/tags")
async def create_tag(body: TagCreate, request: Request) -> ForgeResult:
    """Create a tag. Idempotent (MERGE)."""
    neo4j = request.app.state.neo4j
    await neo4j.run_write(
        "MERGE (t:Tag {name: $name})",
        {"name": body.name},
    )
    return ForgeResult(success=True, data={"name": body.name})


@router.delete("/tags/{name}")
async def delete_tag(name: str, request: Request) -> ForgeResult:
    """Delete a tag. Documents tagged with it lose the TAGGED_WITH relationship."""
    neo4j = request.app.state.neo4j
    await neo4j.run_write(
        "MATCH (t:Tag {name: $name}) DETACH DELETE t",
        {"name": name},
    )
    return ForgeResult(success=True, data={"name": name})
