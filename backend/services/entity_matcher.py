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

import asyncio
import logging
import re
import time
from collections import Counter
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
        # Indexes built by _build_index() after every refresh:
        # exact normalized-name lookup, and a character-trigram inverted
        # index for fuzzy candidate generation. Without them, matching was
        # an O(entities x windows) SequenceMatcher scan whose 5s budget
        # covered ~0.007% of a 276k-entity population (observed live) —
        # fuzzy expansion had silently become a lottery.
        self._by_norm: dict[str, list[int]] = {}
        self._trigram: dict[str, list[int]] = {}

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    async def refresh(self) -> None:
        """Reload entity names from Neo4j."""
        rows = await self._neo4j.run_query(
            """
            MATCH (e)
            WHERE any(l IN labels(e) WHERE l IN ['Material', 'Process', 'Standard', 'Equipment'])
              AND coalesce(e.noise_tier, '') <> 'stop'
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
        self._build_index()
        self._last_refresh = time.monotonic()
        logger.info("EntityMatcher refreshed: %d entity names loaded", len(entities))

    @staticmethod
    def _trigrams(s: str) -> set[str]:
        if len(s) < 3:
            return {s} if s else set()
        return {s[i:i + 3] for i in range(len(s) - 2)}

    def _build_index(self) -> None:
        """Build the exact-normalized and trigram indexes over _entities."""
        by_norm: dict[str, list[int]] = {}
        trigram: dict[str, list[int]] = {}
        for idx, e in enumerate(self._entities):
            if not e.normalized:
                continue
            by_norm.setdefault(e.normalized, []).append(idx)
            for g in self._trigrams(e.normalized):
                trigram.setdefault(g, []).append(idx)
        self._by_norm = by_norm
        self._trigram = trigram

    async def _ensure_loaded(self) -> None:
        """Refresh if never loaded or stale."""
        if not self._entities or (time.monotonic() - self._last_refresh > self._refresh_interval):
            await self.refresh()

    # Common English words that are noise for entity matching — they
    # produce millions of useless SequenceMatcher comparisons on long
    # natural-language queries without ever matching a real entity.
    _NOISE_TOKENS = frozenset({
        "the", "and", "for", "with", "from", "this", "that", "these",
        "those", "are", "was", "has", "have", "had", "can", "will",
        "would", "should", "could", "may", "might", "does", "did",
        "about", "into", "over", "what", "when", "where", "which",
        "who", "why", "how", "not", "but", "also", "than", "then",
        "between", "considering", "run", "size", "type", "using",
    })

    _MAX_WINDOWS = 25
    _TIME_BUDGET = 5.0  # safety net only — index-driven matching is ~ms
    # Per-window cap on fuzzy candidates scored with SequenceMatcher. A
    # 0.75-similar pair shares most of its trigrams, so ordering candidates
    # by shared-trigram count and scoring the top slice is near-exhaustive
    # for this threshold while bounding the work.
    _MAX_CANDIDATES = 300
    # Trigrams whose posting lists exceed this are too common to be
    # discriminative ("ste" in a steels library) — skipped for candidate
    # generation unless the window has no rarer trigram.
    _HOT_TRIGRAM_CAP = 5000

    def find_matches(
        self,
        query: str,
        threshold: float = 0.75,
    ) -> list[MatchResult]:
        """Find entity names that fuzzy-match tokens/windows in the query.

        Strategy (index-driven — full coverage of the entity population):
        - Tokenize the query, filter noise words, build token/bigram/trigram
          windows.
        - For each window: O(1) exact normalized lookup, then candidate
          generation via the character-trigram inverted index (entities
          sharing trigrams with the window, ranked by overlap), then the
          same containment/SequenceMatcher scoring as before — but only
          over the candidates instead of every entity in the graph.

        Scoring semantics are unchanged from the linear version: exact
        normalized match = 1.0, containment = min/max length ratio,
        otherwise SequenceMatcher ratio; results >= threshold.
        """
        if not self._entities:
            return []
        if not self._trigram and self._entities:
            # Entities injected without refresh() (tests) — build lazily.
            self._build_index()

        query_lower = query.lower()
        tokens = [
            t for t in query_lower.split()
            if len(t) >= 3 and t not in self._NOISE_TOKENS
        ]
        results: dict[str, MatchResult] = {}

        # Build candidate windows from filtered tokens
        windows: list[str] = []
        for t in tokens:
            windows.append(t)
        for i in range(len(tokens) - 1):
            windows.append(f"{tokens[i]} {tokens[i+1]}")
        for i in range(len(tokens) - 2):
            windows.append(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")
        if len(tokens) <= 5:
            windows.append(query_lower)
        if len(windows) > self._MAX_WINDOWS:
            windows.sort(key=len)
            windows = windows[: self._MAX_WINDOWS]

        deadline = time.monotonic() + self._TIME_BUDGET

        def _record(idx: int, score: float) -> None:
            entry = self._entities[idx]
            existing = results.get(entry.name)
            if existing is None or score > existing.score:
                results[entry.name] = MatchResult(
                    name=entry.name,
                    entity_type=entry.entity_type,
                    score=score,
                )

        for window in windows:
            nw = _normalize(window)
            wlen = len(nw)
            if wlen == 0:
                continue

            # 1. Exact normalized match — O(1), score 1.0.
            for idx in self._by_norm.get(nw, ()):
                _record(idx, 1.0)

            # 2. Candidate generation via trigram overlap.
            grams = self._trigrams(nw)
            postings = sorted(
                (self._trigram.get(g, ()) for g in grams), key=len,
            )
            counts: Counter[int] = Counter()
            used_any = False
            for plist in postings:
                if not plist:
                    continue
                if len(plist) > self._HOT_TRIGRAM_CAP and used_any:
                    continue  # too common to discriminate; rarer ones suffice
                used_any = True
                counts.update(plist)
            if not counts:
                continue

            # 3. Score the top candidates by shared-trigram count with the
            #    ORIGINAL semantics (containment, then SequenceMatcher).
            for idx, _shared in counts.most_common(self._MAX_CANDIDATES):
                entry = self._entities[idx]
                en = entry.normalized
                elen = len(en)
                if en == nw:
                    continue  # already recorded via exact lookup
                ratio = max(elen, wlen) / max(min(elen, wlen), 1)
                if ratio > 3.0:
                    continue
                if en in nw or nw in en:
                    score = min(elen, wlen) / max(elen, wlen)
                else:
                    score = SequenceMatcher(None, nw, en).ratio()
                if score >= threshold:
                    _record(idx, score)

            if time.monotonic() > deadline:
                logger.warning(
                    "EntityMatcher hit %.0fs safety budget mid-query "
                    "(%d windows processed) — should not happen with the "
                    "trigram index; investigate",
                    self._TIME_BUDGET, windows.index(window) + 1,
                )
                break

        matched = sorted(results.values(), key=lambda m: m.score, reverse=True)
        return matched

    async def find_matches_async(
        self,
        query: str,
        threshold: float = 0.75,
    ) -> list[MatchResult]:
        """Async wrapper: ensures entities are loaded, then runs matching
        in a thread to avoid blocking the event loop."""
        await self._ensure_loaded()
        return await asyncio.to_thread(self.find_matches, query, threshold)
