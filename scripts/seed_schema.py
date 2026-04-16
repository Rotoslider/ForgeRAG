#!/usr/bin/env python3
"""Seed the Neo4j schema for ForgeRAG.

Usage:
    python scripts/seed_schema.py

Requires NEO4J_PASSWORD env var to be set. The embedding dimension is
read from config/forgerag.toml (models.text_embedding_dim).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_settings  # noqa: E402
from backend.db.neo4j_schema import apply_schema  # noqa: E402
from backend.services.neo4j_service import Neo4jService  # noqa: E402


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("seed_schema")

    settings = get_settings()
    if not os.environ.get(settings.neo4j.password_env):
        log.error(
            "Environment variable %s is not set. Set it to your Neo4j password "
            "before running this script.",
            settings.neo4j.password_env,
        )
        return 1

    svc = Neo4jService(settings.neo4j)
    await svc.connect()
    try:
        if not await svc.verify_connectivity():
            log.error(
                "Cannot reach Neo4j at %s. Is the service running?", settings.neo4j.uri
            )
            return 2

        counts = await apply_schema(svc, embedding_dim=settings.models.text_embedding_dim)
        log.info("Schema seeding complete: %s", counts)
        return 0
    finally:
        await svc.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
