"""Neo4j driver wrapper for ForgeRAG.

Provides an async-friendly interface over the official neo4j Python driver.
Holds a single driver instance (with connection pooling handled by the driver)
and exposes helper methods for common operations used across the app.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable

from backend.config import Neo4jSettings

logger = logging.getLogger(__name__)

_HEALTH_INTERVAL: float = 30.0
_BACKOFF_INITIAL: float = 30.0
_BACKOFF_MAX: float = 300.0


class Neo4jService:
    """Async Neo4j driver wrapper.

    Construct with Neo4jSettings. Call connect() during app startup and
    close() during shutdown. Individual methods acquire sessions as needed.
    """

    def __init__(self, settings: Neo4jSettings):
        self.settings = settings
        self._driver: AsyncDriver | None = None
        self._healthy: bool = False
        self._health_task: asyncio.Task[None] | None = None
        self._backoff: float = _BACKOFF_INITIAL

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

    # ------------------------------------------------------------------
    # Health loop
    # ------------------------------------------------------------------

    @property
    def is_healthy(self) -> bool:
        """Whether the last health ping succeeded."""
        return self._healthy

    async def start_health_loop(self) -> None:
        """Start the background task that pings Neo4j periodically.

        Safe to call multiple times — a second call is a no-op while the
        task is already running.
        """
        if self._health_task is not None and not self._health_task.done():
            return
        # Run an initial check so is_healthy is set before the first interval
        self._healthy = await self.verify_connectivity()
        if self._healthy:
            logger.info("Neo4j health loop starting — initial check passed")
        else:
            logger.warning("Neo4j health loop starting — initial check FAILED")
        self._health_task = asyncio.create_task(
            self._health_loop(), name="neo4j-health-loop"
        )

    async def stop_health_loop(self) -> None:
        """Cancel the background health task (called on shutdown)."""
        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None

    async def _health_loop(self) -> None:
        """Background loop: ping Neo4j, track health, reconnect on failure."""
        while True:
            try:
                await asyncio.sleep(
                    _HEALTH_INTERVAL if self._healthy else self._backoff
                )
                reachable = await self.verify_connectivity()

                if reachable:
                    if not self._healthy:
                        logger.info(
                            "Neo4j recovered — marking healthy, "
                            "resetting backoff"
                        )
                        self._backoff = _BACKOFF_INITIAL
                    self._healthy = True
                else:
                    if self._healthy:
                        logger.warning("Neo4j health check failed — marking unhealthy")
                    self._healthy = False
                    # Attempt reconnect
                    await self._attempt_reconnect()

            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("Unexpected error in Neo4j health loop")
                self._healthy = False

    async def _attempt_reconnect(self) -> None:
        """Close and re-create the driver, then verify connectivity."""
        logger.info(
            "Neo4j attempting reconnect (backoff=%.0fs)", self._backoff
        )
        try:
            if self._driver is not None:
                await self._driver.close()
                self._driver = None
            # Re-create the driver
            self._driver = AsyncGraphDatabase.driver(
                self.settings.uri,
                auth=(self.settings.user, self.settings.password),
                max_connection_pool_size=self.settings.max_connection_pool_size,
                connection_acquisition_timeout=self.settings.connection_acquisition_timeout,
            )
            reachable = await self.verify_connectivity()
            if reachable:
                logger.info("Neo4j reconnect succeeded")
                self._healthy = True
                self._backoff = _BACKOFF_INITIAL
            else:
                logger.warning("Neo4j reconnect — driver created but not reachable")
                # Increase backoff for next attempt
                self._backoff = min(self._backoff * 2, _BACKOFF_MAX)
        except Exception:  # noqa: BLE001
            logger.exception("Neo4j reconnect failed")
            self._backoff = min(self._backoff * 2, _BACKOFF_MAX)

    async def run_query(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
        database: str | None = None,
        timeout: float | None = 90.0,
    ) -> list[dict[str, Any]]:
        """Run a Cypher query and return results as a list of dicts.

        timeout: max seconds for the query (default 90s). Prevents runaway
        Cypher from holding the HTTP connection until the browser kills it.
        Pass None to disable.
        """
        db = database or self.settings.database

        async def _execute() -> list[dict[str, Any]]:
            async with self.driver.session(database=db) as session:
                result = await session.run(cypher, parameters or {})
                return [dict(record) async for record in result]

        if timeout is not None:
            return await asyncio.wait_for(_execute(), timeout=timeout)
        return await _execute()

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


