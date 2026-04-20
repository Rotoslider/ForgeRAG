"""Graph-aware retrieval — the reasoning layer.

Instead of just finding pages that match a query (what vector search does),
this module traverses the knowledge graph to find RELATED pages that the user
didn't know to ask about.

Example:
  User asks: "welding Alloy 625 in corrosive environment"

  Vector search alone returns: pages mentioning "welding Alloy 625"

  Graph-aware search also returns:
  - Pages about filler metals COMPATIBLE_WITH Alloy 625 (ERNiCrMo-3)
  - Pages about standards GOVERNING Alloy 625 (ASME IX, ASTM B443)
  - Pages about processes COMPATIBLE_WITH Alloy 625 (GTAW, SMAW)
  - Pages from the same community cluster (nickel-alloy welding)
  - Cross-referenced standards (NACE MR0175 for corrosive service)

The graph provides the reasoning path that a standard LLM doesn't have:
  material → compatible processes → governing standards → cross-references
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from backend.services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)


@dataclass
class GraphContext:
    """Context gathered by traversing the graph from query-matched entities."""

    # Pages found via graph traversal (page_id → reason it was included)
    page_ids: dict[str, str] = field(default_factory=dict)

    # Entities discovered during traversal (for LLM context)
    materials: list[dict] = field(default_factory=list)
    processes: list[dict] = field(default_factory=list)
    standards: list[dict] = field(default_factory=list)
    equipment: list[dict] = field(default_factory=list)

    # Relationship chains discovered (human-readable summaries)
    reasoning_chains: list[str] = field(default_factory=list)

    # Community summaries that cover the topic
    community_summaries: list[str] = field(default_factory=list)


# Short noise words / stopwords that produce catastrophic false matches
# when used in a CONTAINS-style entity name query. "for" hits forging,
# forming, formation. "amp" hits a dozen welding terms. "wire" hits
# every filler-wire and wire-feeder entity in the welding handbooks.
# Used only as a safety net for the legacy query-term-match fallback;
# the preferred exploration path now seeds entities from the retrieved
# pages rather than the query text.
_STOPWORDS = frozenset({
    "what", "when", "where", "which", "who", "why", "how",
    "the", "and", "for", "with", "from", "this", "that", "these", "those",
    "are", "was", "has", "have", "had", "can", "will", "would", "should",
    "could", "may", "might", "does", "did", "about", "into", "over",
    "amp", "wire", "feet", "foot",  # too noisy as bare tokens
    "size", "type", "kind", "data", "page", "book", "text",
    "please", "tell", "show", "give", "find", "look", "search",
})


async def explore_from_query(
    query: str,
    neo4j: Neo4jService,
    *,
    max_pages: int = 15,
    hop_depth: int = 2,
    seed_page_ids: list[str] | None = None,
) -> GraphContext:
    """Explore the knowledge graph, starting either from entities on a set
    of primary retrieval pages (preferred) or — as a fallback — from
    entities whose names keyword-match query terms.

    When `seed_page_ids` is provided, the exploration is GROUNDED in
    content that has already been validated as relevant by the primary
    retriever (RRF hybrid, keyword, or visual). This avoids the classic
    failure mode where a multi-word English question ("what gauge wire
    for a 20 amp circuit") keyword-matches dozens of welding entities
    via substring collisions on "wire" / "amp" / "for" and flood-fills
    the graph context with unrelated welding chains.

    Returns a GraphContext with page IDs, entities, reasoning chains,
    and community summaries that the answer endpoint feeds into the
    LLM prompt.
    """
    ctx = GraphContext()

    matched_entities: list[dict] = []

    if seed_page_ids:
        # PREFERRED PATH: entities mentioned by the primary-retrieval pages.
        # Rank by how many seed pages mention each entity — entities
        # appearing on multiple pages carry more signal than a one-off
        # mention that might be tangential.
        rows = await neo4j.run_query(
            """
            UNWIND $pids AS pid
            MATCH (p:Page {page_id: pid})-[r]->(e)
            WHERE type(r) IN ['MENTIONS_MATERIAL','DESCRIBES_PROCESS',
                              'REFERENCES_STANDARD','MENTIONS_EQUIPMENT']
              AND any(l IN labels(e) WHERE l IN
                      ['Material','Process','Standard','Equipment'])
            WITH e, count(DISTINCT p) AS page_hits
            RETURN labels(e)[0] AS label,
                   coalesce(e.name, e.code) AS name,
                   properties(e) AS props,
                   page_hits
            ORDER BY page_hits DESC, name
            LIMIT 20
            """,
            {"pids": seed_page_ids[:20]},  # cap to avoid unbounded UNWIND
        )
        matched_entities = [dict(r) for r in rows]
        logger.info(
            "Graph exploration seeded from %d retrieved pages: "
            "%d entities discovered",
            len(seed_page_ids), len(matched_entities),
        )
    else:
        # LEGACY FALLBACK: query-term keyword match. Kept for callers that
        # don't pre-retrieve (direct /graph endpoints, tests). Filters
        # stopwords and short noise tokens to limit false-positive explosion.
        terms = list({
            t.lower() for t in query.split()
            if len(t) >= 4 and t.lower() not in _STOPWORDS
        })
        if not terms:
            logger.debug("Query had no usable terms after stopword filtering: %s", query)
            return ctx

        for label in ["Material", "Process", "Standard", "Equipment"]:
            rows = await neo4j.run_query(
                f"""
                MATCH (e:{label})
                WHERE any(t IN $terms WHERE toLower(coalesce(e.name, e.code, '')) CONTAINS t)
                   OR any(alias IN coalesce(e.common_names, [])
                          WHERE any(t IN $terms WHERE toLower(alias) CONTAINS t))
                RETURN '{label}' AS label,
                       coalesce(e.name, e.code) AS name,
                       properties(e) AS props
                LIMIT 5
                """,
                {"terms": terms},
            )
            matched_entities.extend(rows)

    if not matched_entities:
        logger.debug("No graph entities to explore from")
        return ctx

    entity_names = [e["name"] for e in matched_entities]
    logger.info(
        "Graph exploration: %d entities matched (mode=%s): %s",
        len(entity_names),
        "seed_pages" if seed_page_ids else "query_terms",
        entity_names[:10],
    )

    # Categorize matched entities
    for e in matched_entities:
        info = {"name": e["name"], "matched_from": "query"}
        props = e.get("props", {})
        if e["label"] == "Material":
            info.update({
                "material_type": props.get("material_type"),
                "uns_number": props.get("uns_number"),
            })
            ctx.materials.append(info)
        elif e["label"] == "Process":
            info["process_type"] = props.get("process_type")
            ctx.processes.append(info)
        elif e["label"] == "Standard":
            info["organization"] = props.get("organization")
            ctx.standards.append(info)
        elif e["label"] == "Equipment":
            info["equipment_type"] = props.get("equipment_type")
            ctx.equipment.append(info)

    # Step 2: Traverse relationships from matched entities
    # Get pages that DIRECTLY mention the matched entities
    direct_pages = await neo4j.run_query(
        """
        UNWIND $names AS ename
        MATCH (e) WHERE coalesce(e.name, e.code) = ename
        MATCH (p:Page)-[r]->(e)
        WHERE type(r) IN ['MENTIONS_MATERIAL','DESCRIBES_PROCESS',
                          'REFERENCES_STANDARD','MENTIONS_EQUIPMENT']
        MATCH (d:Document)-[:HAS_PAGE]->(p)
        RETURN DISTINCT p.page_id AS page_id, p.page_number AS page_number,
               d.title AS doc_title, type(r) AS rel,
               coalesce(e.name, e.code) AS entity_name
        LIMIT $max
        """,
        {"names": entity_names, "max": max_pages * 2},
    )
    for row in direct_pages:
        reason = f"Page {row['page_number']} directly mentions {row['entity_name']}"
        ctx.page_ids[row["page_id"]] = reason

    # Step 3: Follow entity-to-entity relationships (1-2 hops)
    # e.g., Material → GOVERNED_BY → Standard, Material → COMPATIBLE_WITH → Process
    related = await neo4j.run_query(
        """
        UNWIND $names AS ename
        MATCH (e) WHERE coalesce(e.name, e.code) = ename
        MATCH (e)-[r1]->(neighbor)
        WHERE type(r1) IN ['GOVERNED_BY','COMPATIBLE_WITH_PROCESS',
                           'REFERENCES','CONTAINS_CLAUSE','REQUIRES_MATERIAL']
          AND any(l IN labels(neighbor) WHERE l IN
              ['Material','Process','Standard','Clause','Equipment'])
        OPTIONAL MATCH (neighbor)-[r2]->(hop2)
        WHERE type(r2) IN ['GOVERNED_BY','COMPATIBLE_WITH_PROCESS',
                           'REFERENCES','CONTAINS_CLAUSE']
          AND any(l IN labels(hop2) WHERE l IN
              ['Material','Process','Standard','Clause','Equipment'])
        RETURN coalesce(e.name, e.code) AS from_name,
               labels(e)[0] AS from_label,
               type(r1) AS rel1,
               coalesce(neighbor.name, neighbor.code, neighbor.clause_id) AS hop1_name,
               labels(neighbor)[0] AS hop1_label,
               type(r2) AS rel2,
               coalesce(hop2.name, hop2.code, hop2.clause_id) AS hop2_name,
               labels(hop2)[0] AS hop2_label
        LIMIT 50
        """,
        {"names": entity_names},
    )

    seen_entities = set(entity_names)
    for row in related:
        # Build reasoning chain
        chain = f"{row['from_name']} ({row['from_label']}) → {row['rel1']} → {row['hop1_name']} ({row['hop1_label']})"
        if row.get("hop2_name"):
            chain += f" → {row['rel2']} → {row['hop2_name']} ({row['hop2_label']})"
        ctx.reasoning_chains.append(chain)

        # Collect discovered entities
        for hop_name, hop_label in [
            (row["hop1_name"], row["hop1_label"]),
            (row.get("hop2_name"), row.get("hop2_label")),
        ]:
            if hop_name and hop_name not in seen_entities:
                seen_entities.add(hop_name)
                info = {"name": hop_name, "matched_from": "graph_traversal"}
                if hop_label == "Material":
                    ctx.materials.append(info)
                elif hop_label == "Process":
                    ctx.processes.append(info)
                elif hop_label == "Standard":
                    ctx.standards.append(info)

    # Step 4: Get pages connected to discovered entities (from hops)
    discovered_names = list(seen_entities - set(entity_names))
    if discovered_names:
        hop_pages = await neo4j.run_query(
            """
            UNWIND $names AS ename
            MATCH (e) WHERE coalesce(e.name, e.code) = ename
            MATCH (p:Page)-[r]->(e)
            WHERE type(r) IN ['MENTIONS_MATERIAL','DESCRIBES_PROCESS',
                              'REFERENCES_STANDARD','MENTIONS_EQUIPMENT']
            MATCH (d:Document)-[:HAS_PAGE]->(p)
            RETURN DISTINCT p.page_id AS page_id, p.page_number AS page_number,
                   d.title AS doc_title,
                   coalesce(e.name, e.code) AS entity_name
            LIMIT $max
            """,
            {"names": discovered_names[:20], "max": max_pages},
        )
        for row in hop_pages:
            if row["page_id"] not in ctx.page_ids:
                reason = f"Page {row['page_number']} mentions {row['entity_name']} (discovered via graph)"
                ctx.page_ids[row["page_id"]] = reason

    # Step 5: Community summaries
    community_rows = await neo4j.run_query(
        """
        UNWIND $pids AS pid
        MATCH (p:Page {page_id: pid})-[:IN_COMMUNITY]->(c:Community)
        WHERE c.summary IS NOT NULL AND c.summary <> ''
        RETURN DISTINCT c.community_id AS cid, c.summary AS summary, c.level AS level
        ORDER BY c.level DESC
        LIMIT 3
        """,
        {"pids": list(ctx.page_ids.keys())[:30]},
    )
    ctx.community_summaries = [r["summary"] for r in community_rows if r["summary"]]

    # Cap pages
    if len(ctx.page_ids) > max_pages:
        # Keep the first max_pages (direct matches have priority since they're added first)
        limited = dict(list(ctx.page_ids.items())[:max_pages])
        ctx.page_ids = limited

    logger.info(
        "Graph exploration complete: %d pages, %d materials, %d processes, "
        "%d standards, %d reasoning chains, %d community summaries",
        len(ctx.page_ids),
        len(ctx.materials),
        len(ctx.processes),
        len(ctx.standards),
        len(ctx.reasoning_chains),
        len(ctx.community_summaries),
    )
    return ctx
