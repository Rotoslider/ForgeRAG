"""Shared entity-merge mechanics.

Merging one entity node into another sounds like one MERGE, but Cypher
requires a literal relationship type in MERGE — the naive dynamic-type
versions that used to live in the pipeline dedup step and in
/admin/normalize-entities were both broken (one was a syntax error, the
other invented junk ``<Label>__TEMP_REL`` edges and then deleted the
duplicate with its mentions). This module is the single correct
implementation both callers now use: discover the relationship types
actually touching the loser, redirect each with a literal type while
accumulating support_count, then delete the loser (same proven pattern as
scripts/canonicalize_entity_apply.py).
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

# Serializes entity merges against concurrent page-relationship writes.
# merge_entity's redirect passes and its final DETACH DELETE are separate
# transactions; a relationship MERGEd onto the loser node by a concurrent
# write_page between them is destroyed unmigrated (live incident
# 2026-08-08 16:29: a fill-missing job's REFERENCES_STANDARD rels were
# deleted seconds after being written by another job's dedup pass, leaving
# stamped pages with no rels and no marker). Every graph writer lives in
# this one process/event loop, so an asyncio lock is a complete exclusion:
# merge_entity holds it for the whole merge; GraphBuilder.write_page holds
# it while writing page rels. Merges are rare and fast (ms) relative to
# extraction, so the added latency is negligible.
GRAPH_MERGE_LOCK = asyncio.Lock()

# Page->entity relationship per entity label, used to recover the junk
# ``<Label>__TEMP_REL`` edges the old normalize-entities left behind.
PAGE_REL = {
    "Material": "MENTIONS_MATERIAL",
    "Process": "DESCRIBES_PROCESS",
    "Standard": "REFERENCES_STANDARD",
    "Equipment": "MENTIONS_EQUIPMENT",
}

ENTITY_LABELS = [
    ("Material", "name"),
    ("Process", "name"),
    ("Standard", "code"),
    ("Equipment", "name"),
]


async def merge_entity(neo4j, label: str, pk: str, winner: str, loser: str) -> None:
    """Merge the `loser` entity into `winner`: every relationship touching
    the loser is redirected to the winner (per discovered type, literal in
    the Cypher — type strings come from the graph's own type(r), never user
    input), support_count accumulates on collisions, the loser's name lands
    in winner.common_names, and the loser node is deleted.

    Holds GRAPH_MERGE_LOCK for the whole merge so a concurrent
    write_page can never land a relationship on the loser between the
    redirect passes and the final DETACH DELETE (which would destroy it
    unmigrated) — and so a concurrent MERGE-by-name can't resurrect the
    loser node mid-delete."""
    async with GRAPH_MERGE_LOCK:
        await _merge_entity_locked(neo4j, label, pk, winner, loser)


async def _merge_entity_locked(
    neo4j, label: str, pk: str, winner: str, loser: str
) -> None:
    params = {"winner": winner, "loser": loser}
    incoming = await neo4j.run_query(
        f"MATCH ()-[r]->(l:{label} {{{pk}: $loser}}) "
        "RETURN DISTINCT type(r) AS t",
        params,
    )
    for row in incoming:
        await neo4j.run_write(
            f"""
            MATCH (w:{label} {{{pk}: $winner}})
            MATCH (l:{label} {{{pk}: $loser}})
            OPTIONAL MATCH (src)-[r:{row['t']}]->(l)
            WITH w, src, r
            WHERE r IS NOT NULL AND src <> w
            MERGE (src)-[nr:{row['t']}]->(w)
            ON CREATE SET
                nr.support_count = coalesce(r.support_count, 0),
                nr.context = r.context
            ON MATCH SET
                nr.support_count = coalesce(nr.support_count, 0)
                                 + coalesce(r.support_count, 0)
            DELETE r
            """,
            params,
        )
    outgoing = await neo4j.run_query(
        f"MATCH (l:{label} {{{pk}: $loser}})-[r]->(t) "
        "RETURN DISTINCT type(r) AS t, labels(t)[0] AS tl",
        params,
    )
    for row in outgoing:
        await neo4j.run_write(
            f"""
            MATCH (w:{label} {{{pk}: $winner}})
            MATCH (l:{label} {{{pk}: $loser}})
            OPTIONAL MATCH (l)-[r:{row['t']}]->(tgt:{row['tl']})
            WITH w, tgt, r
            WHERE r IS NOT NULL AND tgt <> w
            MERGE (w)-[nr:{row['t']}]->(tgt)
            ON CREATE SET
                nr.support_count = coalesce(r.support_count, 0),
                nr.context = r.context
            ON MATCH SET
                nr.support_count = coalesce(nr.support_count, 0)
                                 + coalesce(r.support_count, 0)
            DELETE r
            """,
            params,
        )
    # Whatever is left on the loser (edges between the pair) goes with it.
    await neo4j.run_write(
        f"""
        MATCH (w:{label} {{{pk}: $winner}})
        MATCH (l:{label} {{{pk}: $loser}})
        SET w.common_names = coalesce(w.common_names, []) + [$loser]
        DETACH DELETE l
        """,
        params,
    )
    logger.info("Merged %s %r into %r", label, loser, winner)


async def recover_temp_rels(neo4j) -> int:
    """Convert junk ``<Label>__TEMP_REL`` edges (left by the old broken
    normalize-entities) back into the proper page->entity relationship for
    the label, then delete them. Returns how many were recovered."""
    recovered = 0
    for label, proper in PAGE_REL.items():
        rows = await neo4j.run_query(
            f"MATCH ()-[r:{label}__TEMP_REL]->() RETURN count(r) AS n"
        )
        n = rows[0]["n"] if rows else 0
        if not n:
            continue
        await neo4j.run_write(
            f"""
            MATCH (p)-[r:{label}__TEMP_REL]->(e:{label})
            MERGE (p)-[:{proper}]->(e)
            DELETE r
            """
        )
        # Any TEMP_REL whose endpoints didn't match the expected shape
        # (shouldn't exist, but junk is junk) is dropped outright.
        await neo4j.run_write(
            f"MATCH ()-[r:{label}__TEMP_REL]->() DELETE r"
        )
        recovered += n
        logger.info(
            "Recovered %d %s__TEMP_REL edge(s) to %s", n, label, proper
        )
    return recovered
