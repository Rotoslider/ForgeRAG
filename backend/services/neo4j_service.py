"""Neo4j driver wrapper for ForgeRAG.

Provides an async-friendly interface over the official neo4j Python driver.
Holds a single driver instance (with connection pooling handled by the driver)
and exposes helper methods for common operations used across the app.
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable

from backend.config import Neo4jSettings

logger = logging.getLogger(__name__)


class Neo4jService:
    """Async Neo4j driver wrapper.

    Construct with Neo4jSettings. Call connect() during app startup and
    close() during shutdown. Individual methods acquire sessions as needed.
    """

    def __init__(self, settings: Neo4jSettings):
        self.settings = settings
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Initialize the driver. Does not fail if Neo4j is not yet reachable —
        use verify_connectivity() to check."""
        if self._driver is not None:
            return
        password = self.settings.password
        if not password:
            logger.warning(
                "No Neo4j password resolved from env var %s — "
                "driver will be created but connections will fail.",
                self.settings.password_env,
            )

        self._driver = AsyncGraphDatabase.driver(
            self.settings.uri,
            auth=(self.settings.user, password),
            max_connection_pool_size=self.settings.max_connection_pool_size,
            connection_acquisition_timeout=self.settings.connection_acquisition_timeout,
        )
        logger.info("Neo4j driver initialized for %s", self.settings.uri)

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed")

    @property
    def driver(self) -> AsyncDriver:
        if self._driver is None:
            raise RuntimeError("Neo4jService not connected — call connect() first")
        return self._driver

    async def verify_connectivity(self) -> bool:
        """Check if Neo4j is reachable. Returns True/False; never raises."""
        if self._driver is None:
            return False
        try:
            await self._driver.verify_connectivity()
            return True
        except (ServiceUnavailable, Exception) as exc:  # noqa: BLE001
            logger.debug("Neo4j connectivity check failed: %s", exc)
            return False

    async def run_query(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
        database: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run a Cypher query and return results as a list of dicts."""
        db = database or self.settings.database
        async with self.driver.session(database=db) as session:
            result = await session.run(cypher, parameters or {})
            records = [dict(record) async for record in result]
            return records

    async def run_write(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
        database: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run a Cypher write query in a managed transaction.

        In the neo4j 6.x async driver, tx.run() returns a coroutine that must
        be awaited before iterating. We wrap that in an async transaction
        function passed to session.execute_write.
        """
        db = database or self.settings.database
        params = parameters or {}

        async def _tx(tx) -> list[dict[str, Any]]:
            result = await tx.run(cypher, params)
            return [dict(record) async for record in result]

        async with self.driver.session(database=db) as session:
            return await session.execute_write(_tx)

    async def get_counts(self) -> dict[str, int]:
        """Get document and page counts. Returns zeros if the database
        is empty or the schema is not yet seeded."""
        try:
            rows = await self.run_query(
                """
                OPTIONAL MATCH (d:Document) WITH count(d) AS documents
                OPTIONAL MATCH (p:Page) WITH documents, count(p) AS pages
                RETURN documents, pages
                """
            )
            if rows:
                return {"documents": rows[0].get("documents", 0) or 0,
                        "pages": rows[0].get("pages", 0) or 0}
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_counts failed: %s", exc)
        return {"documents": 0, "pages": 0}


