"""Fuzzy entity-name matcher for search queries.

Loads entity names (Material.name, Standard.code, Process.name,
Equipment.name) from Neo4j into memory and provides fuzzy matching
against query text using difflib.SequenceMatcher. Handles common OCR
and user-input variations: missing special characters (registered
trademark, copyright), spacing differences, abbreviations, and
case mismatches.

The entity list is cached in memory and refreshed on demand (not every
query). Typical usage: call find_matches(query) before graph_first /
graph_boosted to expand the set of recognized entity names.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher

from backend.services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)

# Characters that are commonly missing in user input or OCR output
_STRIP_CHARS_RE = re.compile(r"[®©™°\-–—\s]+")


def _normalize(name: str) -> str:
    """Normalize an entity name for comparison.

    Strips special chars (registered, copyright, trademark, degree),
    collapses whitespace, and lowercases. This makes "Inconel® 625"
    compare well against "inconel 625" or "inconel625".
    """
    return _STRIP_CHARS_RE.sub("", name).lower()


@dataclass
class MatchResult:
    """A single fuzzy match result."""
    name: str           # Original entity name as stored in Neo4j
    entity_type: str    # Material, Standard, Process, Equipment
    score: float        # Similarity score [0, 1]


@dataclass
class _EntityEntry:
    """Internal: a cached entity with its normalized form."""
    name: str
    normalized: str
    entity_type: str


class EntityMatcher:
    """In-memory fuzzy matcher for known entity names.

    Loads entity names from Neo4j once (or on refresh), then matches
    query text against them using SequenceMatcher for edit-distance
    similarity, with a fast pre-filter based on normalized length.
    """

    def __init__(self, neo4j: Neo4jService):
        self._neo4j = neo4j
        self._entities: list[_EntityEntry] = []
        self._last_refresh: float = 0.0
        self._refresh_interval: float = 300.0  # 5 minutes

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    async def refresh(self) -> None:
        """Reload entity names from Neo4j."""
        rows = await self._neo4j.run_query(
            """
            MATCH (e)
            WHERE any(l IN labels(e) WHERE l IN ['Material', 'Process', 'Standard', 'Equipment'])
            WITH e,
                 CASE
                   WHEN 'Material' IN labels(e) THEN 'Material'
                   WHEN 'Standard' IN labels(e) THEN 'Standard'
                   WHEN 'Process' IN labels(e) THEN 'Process'
                   ELSE 'Equipment'
                 END AS etype
            RETURN coalesce(e.name, e.code) AS name,
                   etype,
                   coalesce(e.common_names, []) AS aliases
            """
        )
        entities: list[_EntityEntry] = []
        seen: set[str] = set()
        for r in rows:
            name = r.get("name")
            etype = r.get("etype", "")
            if not name:
                continue
            # Add the primary name
            key = (name.lower(), etype)
            if key not in seen:
                seen.add(key)
                entities.append(_EntityEntry(
                    name=name,
                    normalized=_normalize(name),
                    entity_type=etype,
                ))
            # Add aliases
            for alias in (r.get("aliases") or []):
                if not alias:
                    continue
                akey = (alias.lower(), etype)
                if akey not in seen:
                    seen.add(akey)
                    entities.append(_EntityEntry(
                        name=alias,
                        normalized=_normalize(alias),
                        entity_type=etype,
                    ))

        self._entities = entities
        self._last_refresh = time.monotonic()
        logger.info("EntityMatcher refreshed: %d entity names loaded", len(entities))

    async def _ensure_loaded(self) -> None:
        """Refresh if never loaded or stale."""
        if not self._entities or (time.monotonic() - self._last_refresh > self._refresh_interval):
            await self.refresh()

    def find_matches(
        self,
        query: str,
        threshold: float = 0.75,
    ) -> list[MatchResult]:
        """Find entity names that fuzzy-match tokens/windows in the query.

        Strategy:
        - Tokenize the query
        - For each bigram and trigram window (and individual tokens >= 3 chars),
          compare against known entity names using:
          1. Normalized exact match (catches "inconel625" vs "Inconel® 625")
          2. SequenceMatcher ratio for candidates within reasonable length range
          3. Token set overlap for multi-word entities

        Returns matches sorted by score descending.
        """
        if not self._entities:
            return []

        query_lower = query.lower()
        tokens = query_lower.split()
        results: dict[str, MatchResult] = {}  # name -> best result

        # Build candidate windows from the query
        windows: list[str] = []
        # Individual tokens (>= 3 chars)
        for t in tokens:
            if len(t) >= 3:
                windows.append(t)
        # Bigrams
        for i in range(len(tokens) - 1):
            windows.append(f"{tokens[i]} {tokens[i+1]}")
        # Trigrams
        for i in range(len(tokens) - 2):
            windows.append(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")
        # Full query as a window too (for short queries)
        if len(tokens) <= 5:
            windows.append(query_lower)

        # Normalize each window
        normalized_windows = [_normalize(w) for w in windows]

        for entry in self._entities:
            best_score = 0.0

            # Fast check: if entity is very long relative to all windows, skip
            elen = len(entry.normalized)
            if elen == 0:
                continue

            for i, window in enumerate(windows):
                nw = normalized_windows[i]
                wlen = len(nw)

                # Skip if length mismatch is too extreme (more than 3x difference)
                if wlen == 0:
                    continue
                ratio = max(elen, wlen) / max(min(elen, wlen), 1)
                if ratio > 3.0:
                    continue

                # 1. Normalized exact match
                if nw == entry.normalized:
                    best_score = 1.0
                    break

                # 2. Check if one contains the other (common for abbreviations)
                if entry.normalized in nw or nw in entry.normalized:
                    # Score based on how much of the entity the window covers
                    containment_score = min(elen, wlen) / max(elen, wlen)
                    if containment_score > best_score:
                        best_score = containment_score
                    continue

                # 3. SequenceMatcher for edit distance
                sim = SequenceMatcher(None, nw, entry.normalized).ratio()
                if sim > best_score:
                    best_score = sim

            if best_score >= threshold:
                # Keep the best score for this entity name
                existing = results.get(entry.name)
                if existing is None or best_score > existing.score:
                    results[entry.name] = MatchResult(
                        name=entry.name,
                        entity_type=entry.entity_type,
                        score=best_score,
                    )

        # Sort by score descending
        matched = sorted(results.values(), key=lambda m: m.score, reverse=True)
        return matched

    async def find_matches_async(
        self,
        query: str,
        threshold: float = 0.75,
    ) -> list[MatchResult]:
        """Async wrapper: ensures entities are loaded, then runs matching."""
        await self._ensure_loaded()
        return self.find_matches(query, threshold)
