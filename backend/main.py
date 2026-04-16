"""FastAPI application entry point for ForgeRAG.

Wires together config, Neo4j service, routers, and CORS. Additional routers
(documents, ingestion, search, graph, etc.) are added in later phases.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import DEFAULT_CONFIG_PATH, get_settings
from backend.ingestion.job_manager import JobManager
from backend.ingestion.pipeline import IngestionPipeline
from backend.routers import admin, documents, graph, health, images, ingestion, search, system
from backend.services.colpali_service import create_colpali_service
from backend.services.image_service import ImageHighlighter
from backend.services.gpu_manager import GPUManager
from backend.services.llm_service import create_llm_service
from backend.services.neo4j_service import Neo4jService
from backend.services.text_embedding_service import create_text_embedding_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize shared resources on startup, clean up on shutdown."""
    settings = get_settings()
    app.state.settings = settings
    app.state.config_path = os.environ.get("FORGERAG_CONFIG", str(DEFAULT_CONFIG_PATH))

    # Ensure data directories exist
    data_dir = Path(settings.server.data_dir)
    (data_dir / "page_images").mkdir(parents=True, exist_ok=True)
    (data_dir / "reduced_images").mkdir(parents=True, exist_ok=True)
    (data_dir / "uploads").mkdir(parents=True, exist_ok=True)

    # Neo4j — connect but don't fail if unreachable (service can serve health)
    neo4j = Neo4jService(settings.neo4j)
    await neo4j.connect()
    app.state.neo4j = neo4j
    try:
        connected = await neo4j.verify_connectivity()
        if connected:
            logger.info("Neo4j reachable at %s", settings.neo4j.uri)
        else:
            logger.warning(
                "Neo4j unreachable at %s — service will start but DB operations will fail.",
                settings.neo4j.uri,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Neo4j connectivity check raised: %s", exc)

    # Job manager (SQLite-backed) for tracking ingestion progress
    job_manager = JobManager(data_dir / "jobs.sqlite")
    await job_manager.init()
    app.state.job_manager = job_manager

    # GPU manager + ML services (created regardless, actual model loading is lazy)
    gpu = GPUManager(idle_unload_seconds=settings.gpu.model_idle_unload_seconds)
    await gpu.start()
    app.state.gpu = gpu
    try:
        text_embedding = create_text_embedding_service(settings, gpu)
        app.state.text_embedding = text_embedding
    except Exception as exc:  # noqa: BLE001
        logger.warning("Text embedding service not wired: %s", exc)
        app.state.text_embedding = None

    try:
        colpali = create_colpali_service(settings, gpu)
        app.state.colpali = colpali
    except Exception as exc:  # noqa: BLE001
        logger.warning("ColPali service not wired: %s", exc)
        app.state.colpali = None

    if app.state.colpali is not None:
        app.state.highlighter = ImageHighlighter(
            data_dir=data_dir, colpali=app.state.colpali, gpu=gpu
        )
    else:
        app.state.highlighter = None

    # LLM service (for Phase 4 entity extraction). Lazy connection check —
    # the pipeline disables extraction if the endpoint isn't reachable at
    # step time, but we start the client so it's ready when it is.
    llm_service = create_llm_service(settings)
    await llm_service.start()
    app.state.llm = llm_service
    try:
        llm_ok = await llm_service.health()
        if llm_ok:
            logger.info("LLM endpoint reachable at %s", settings.llm.endpoint)
        else:
            logger.warning(
                "LLM endpoint %s not reachable — entity extraction will be skipped "
                "until the server is running.",
                settings.llm.endpoint,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM health check raised: %s", exc)

    # Ingestion pipeline (one instance, processes one job at a time via asyncio)
    pipeline = IngestionPipeline(
        settings=settings,
        neo4j=neo4j,
        job_manager=job_manager,
        gpu=gpu,
        text_embedding=app.state.text_embedding,
        colpali=app.state.colpali,
        llm=llm_service,
    )
    app.state.pipeline = pipeline

    logger.info("ForgeRAG startup complete on %s:%d", settings.server.host, settings.server.port)
    try:
        yield
    finally:
        try:
            await llm_service.stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning("llm.stop failed: %s", exc)
        try:
            await gpu.stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning("gpu.stop failed: %s", exc)
        await neo4j.close()
        logger.info("ForgeRAG shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="ForgeRAG",
        description="Local engineering knowledge graph (ColPali + Neo4j + GraphRAG)",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(ingestion.router)
    app.include_router(images.router)
    app.include_router(search.router)
    app.include_router(system.router)
    app.include_router(graph.router)
    app.include_router(admin.router)

    # Frontend static mount (production build). Skipped if not built yet.
    # We register a SPA fallback route that returns index.html for any /app/*
    # path that isn't a real file, so React Router deep-links survive a
    # browser refresh. The literal /app/assets/... paths are served by the
    # nested StaticFiles mount, which takes precedence over the fallback.
    frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        index_html = frontend_dist / "index.html"
        app.mount(
            "/app/assets",
            StaticFiles(directory=str(frontend_dist / "assets")),
            name="frontend-assets",
        )

        @app.get("/app", include_in_schema=False)
        async def _app_root() -> FileResponse:
            return FileResponse(index_html)

        @app.get("/app/{path:path}", include_in_schema=False)
        async def _spa_fallback(path: str, request: Request) -> FileResponse:
            # Serve static files from dist root if they exist; otherwise fall
            # back to index.html so react-router can handle client-side routing.
            candidate = frontend_dist / path
            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(index_html)

        logger.info("Frontend mounted from %s (SPA fallback enabled)", frontend_dist)

    return app


app = create_app()
