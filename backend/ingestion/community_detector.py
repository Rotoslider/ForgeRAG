"""Community detection + hierarchical LLM summaries (the GraphRAG pattern).

Instead of (or in addition to) per-page retrieval, we cluster pages that share
engineering entities and summarize each cluster. A query like "what do we
know about GTAW welding of stainless steels?" can then hit a cluster summary
rather than dozens of individual pages, giving broader/global answers.

Algorithm:
1. Project the graph as Pages connected by shared Material/Process/Standard/
   Equipment entities. Edge weight = number of shared entities.
2. Run Leiden at multiple resolution levels (0.5, 1.0, 2.0) to get hierarchical
   communities. Higher resolution -> finer/smaller clusters.
3. For each community, concatenate (or sample) member page snippets and send
   to the LLM with a summarization prompt that emphasizes engineering content.
4. Embed the summary with the text embedding service.
5. Write :Community nodes + IN_COMMUNITY edges from pages + PARENT_COMMUNITY
   edges between levels.

This is a global operation — it rebuilds all :Community nodes each time.
For incremental updates later we can add a diff path.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

import igraph as ig

from backend.ingestion.entity_extractor import PageExtraction  # reuse enums
from backend.services.llm_service import LLMFatalError, LLMService, LLMTransientError
from backend.services.neo4j_service import Neo4jService
from backend.services.text_embedding_service import TextEmbeddingService

logger = logging.getLogger(__name__)


# Leiden resolution parameters — three levels, high to low.
# Higher resolution => more, smaller communities. level=0 is the finest.
DEFAULT_RESOLUTIONS = (2.0, 1.0, 0.5)


@dataclass
class CommunityResult:
    community_id: str
    level: int
    resolution: float
    page_ids: list[str]
    summary: str = ""


class CommunitySummary(dict):
    """Schema used for the structured summary prompt."""


class CommunityDetector:
    """Builds hierarchical communities over the page-entity graph."""

    def __init__(
        self,
        *,
        neo4j: Neo4jService,
        llm: LLMService,
        text_embedding: TextEmbeddingService,
        min_community_size: int = 3,
        max_summary_pages: int = 20,
        max_snippet_chars: int = 800,
    ):
        self.neo4j = neo4j
        self.llm = llm
        self.text_embedding = text_embedding
        self.min_community_size = min_community_size
        self.max_summary_pages = max_summary_pages
        self.max_snippet_chars = max_snippet_chars

    # ------------------------------------------------------------------

    async def build(
        self, resolutions: tuple[float, ...] = DEFAULT_RESOLUTIONS
    ) -> dict[str, int]:
        """Rebuild the Community layer. Returns counts per level."""
        counts: dict[str, int] = {}

        # 1. Wipe existing communities (idempotent rebuild)
        await self.neo4j.run_write(
            """
            MATCH (c:Community)
            DETACH DELETE c
            """
        )

        # 2. Pull pages + their entity mentions and build the igraph
        graph, page_index = await self._project_graph()
        if graph.vcount() == 0:
            logger.info("No pages with entity mentions — skipping community build")
            return counts

        logger.info(
            "Community graph: %d pages, %d shared-entity edges",
            graph.vcount(), graph.ecount(),
        )

        # 3. Run Leiden at each resolution, store the result
        level_to_communities: dict[int, list[CommunityResult]] = {}
        for level, resolution in enumerate(resolutions):
            partition = graph.community_leiden(
                objective_function="modularity",
                resolution=resolution,
                n_iterations=-1,
                weights="weight",
            )

            communities: list[CommunityResult] = []
            for cid, members in enumerate(partition):
                if len(members) < self.min_community_size:
                    continue
                page_ids = [page_index[v] for v in members]
                communities.append(CommunityResult(
                    community_id=str(uuid.uuid4()),
                    level=level,
                    resolution=resolution,
                    page_ids=page_ids,
                ))

            level_to_communities[level] = communities
            counts[f"level_{level}"] = len(communities)
            logger.info(
                "Level %d (res=%.1f): %d communities (min size %d)",
                level, resolution, len(communities), self.min_community_size,
            )

        # 4. Summarize each community via LLM and create nodes
        total_summarized = 0
        for level, communities in level_to_communities.items():
            for comm in communities:
                try:
                    await self._summarize_and_write(comm)
                    total_summarized += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Summarize/write failed for community %s: %s",
                        comm.community_id, exc,
                    )

        # 5. Wire PARENT_COMMUNITY edges across levels.
        # A community at level N is parent of a community at level N-1 if the
        # child's page set is a subset of the parent's page set.
        await self._link_parents(level_to_communities)

        counts["summarized"] = total_summarized
        return counts

    # ------------------------------------------------------------------

    async def _project_graph(self) -> tuple[ig.Graph, list[str]]:
        """Build an igraph where nodes are Pages and edges weight-by shared entities."""
        # Cypher: for each pair of pages that share any entity, count entities
        rows = await self.neo4j.run_query(
            """
            MATCH (p1:Page)-[:MENTIONS_MATERIAL|DESCRIBES_PROCESS|REFERENCES_STANDARD|MENTIONS_EQUIPMENT]->(e)
                 <-[:MENTIONS_MATERIAL|DESCRIBES_PROCESS|REFERENCES_STANDARD|MENTIONS_EQUIPMENT]-(p2:Page)
            WHERE id(p1) < id(p2)
            RETURN p1.page_id AS src, p2.page_id AS dst, count(e) AS weight
            """
        )
        if not rows:
            return ig.Graph(), []

        # Build unique page id list
        ids: list[str] = []
        idx: dict[str, int] = {}
        edges: list[tuple[int, int]] = []
        weights: list[int] = []
        for r in rows:
            s, d, w = r["src"], r["dst"], r["weight"]
            for p in (s, d):
                if p not in idx:
                    idx[p] = len(ids)
                    ids.append(p)
            edges.append((idx[s], idx[d]))
            weights.append(int(w))

        g = ig.Graph(n=len(ids), edges=edges, directed=False)
        g.es["weight"] = weights
        return g, ids

    # ------------------------------------------------------------------

    async def _summarize_and_write(self, comm: CommunityResult) -> None:
        """Fetch member page snippets, LLM-summarize, embed, write to Neo4j."""
        # Pull text from member pages. Cap the number of pages we send and
        # the chars per snippet to keep the prompt tractable.
        sample_ids = comm.page_ids[: self.max_summary_pages]
        rows = await self.neo4j.run_query(
            """
            MATCH (p:Page) WHERE p.page_id IN $ids
            MATCH (d:Document)-[:HAS_PAGE]->(p)
            RETURN d.title AS doc, p.page_number AS page,
                   p.extracted_text AS text
            """,
            {"ids": sample_ids},
        )

        snippets = []
        for r in rows:
            txt = (r["text"] or "")[: self.max_snippet_chars]
            if txt:
                snippets.append(f"[{r['doc']} p.{r['page']}] {txt}")

        if not snippets:
            logger.debug("Community %s has no text — skipping", comm.community_id)
            return

        # Also fetch the most common entities in this community so the summary
        # can anchor on them.
        entity_rows = await self.neo4j.run_query(
            """
            MATCH (p:Page)-[r]->(e)
            WHERE p.page_id IN $ids
              AND type(r) IN ['MENTIONS_MATERIAL','DESCRIBES_PROCESS',
                              'REFERENCES_STANDARD','MENTIONS_EQUIPMENT']
            WITH e, labels(e)[0] AS kind, count(DISTINCT p) AS mentions
            RETURN kind, coalesce(e.name, e.code) AS name, mentions
            ORDER BY mentions DESC
            LIMIT 20
            """,
            {"ids": sample_ids},
        )
        entity_summary = ", ".join(
            f"{r['kind']}:{r['name']} (×{r['mentions']})" for r in entity_rows
        )

        user_msg = (
            f"Summarize the engineering topic covered by this cluster of pages "
            f"in 3-5 sentences. Focus on materials, processes, standards, and "
            f"equipment. Cite entities by name inline.\n\n"
            f"Key entities: {entity_summary}\n\n"
            f"Sample page snippets (truncated):\n" + "\n\n".join(snippets)
        )

        messages = [
            {"role": "system", "content":
                "You are an engineering summarizer. Produce a concise, factual "
                "summary of the cluster of document pages provided. No "
                "preamble, no meta-commentary. Output plain text only."},
            {"role": "user", "content": user_msg},
        ]

        try:
            summary = await self.llm.chat(
                messages, max_tokens=1024, temperature=0.1
            )
        except (LLMTransientError, LLMFatalError) as exc:
            logger.warning(
                "LLM summarization failed for community %s: %s",
                comm.community_id, exc,
            )
            summary = ""

        summary = (summary or "").strip()
        if not summary:
            # No summary — still create the community so the graph knows it
            # exists, but leave the summary empty.
            logger.debug(
                "Community %s (level %d) got empty summary",
                comm.community_id, comm.level,
            )

        # Embed the summary (if non-empty)
        if summary:
            embedding = self.text_embedding.embed_documents([summary])[0]
            emb_list: list[float] = embedding.tolist()
        else:
            emb_list = [0.0] * self.text_embedding.dim

        # Write Community node + IN_COMMUNITY edges
        await self.neo4j.run_write(
            """
            MERGE (c:Community {community_id: $cid})
            SET c.level = $level,
                c.resolution = $resolution,
                c.summary = $summary,
                c.summary_embedding = $emb,
                c.member_count = $count
            WITH c
            UNWIND $page_ids AS pid
            MATCH (p:Page {page_id: pid})
            MERGE (p)-[:IN_COMMUNITY {level: $level}]->(c)
            """,
            {
                "cid": comm.community_id,
                "level": comm.level,
                "resolution": comm.resolution,
                "summary": summary,
                "emb": emb_list,
                "count": len(comm.page_ids),
                "page_ids": comm.page_ids,
            },
        )

        comm.summary = summary

    # ------------------------------------------------------------------

    async def _link_parents(
        self, levels: dict[int, list[CommunityResult]]
    ) -> None:
        """Connect finer communities (lower level) to coarser ones (higher level)
        via PARENT_COMMUNITY when the finer one's member set is contained in
        the coarser one.
        """
        sorted_levels = sorted(levels.keys())
        for i in range(len(sorted_levels) - 1):
            child_level = sorted_levels[i]
            parent_level = sorted_levels[i + 1]
            children = levels[child_level]
            parents = levels[parent_level]

            for child in children:
                child_pages = set(child.page_ids)
                best = None
                best_overlap = 0
                for parent in parents:
                    overlap = len(child_pages & set(parent.page_ids))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best = parent
                if best is not None and best_overlap >= max(1, len(child_pages) // 2):
                    await self.neo4j.run_write(
                        """
                        MATCH (child:Community {community_id: $child_id})
                        MATCH (parent:Community {community_id: $parent_id})
                        MERGE (child)-[:PARENT_COMMUNITY]->(parent)
                        """,
                        {
                            "child_id": child.community_id,
                            "parent_id": best.community_id,
                        },
                    )
