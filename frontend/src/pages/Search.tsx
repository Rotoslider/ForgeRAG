import { useEffect, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import {
  searchSemantic,
  searchVisual,
  searchHybrid,
  searchKeyword,
  searchAnswer,
  reducedImageUrl,
  pageImageUrl,
} from "../api/client";
import type { SearchHit, CommunityHit, HybridStrategy } from "../api/types";
import type { AnswerResult } from "../api/client";

type Mode = "semantic" | "keyword" | "visual" | "hybrid" | "answer";

export default function Search() {
  // Query, mode, and strategy are persisted in the URL so navigating
  // away and back (or following a page-viewer link and returning) keeps
  // the search in place. The limit slider and expanded-row state are
  // ephemeral and stay in component state.
  const [searchParams, setSearchParams] = useSearchParams();
  const [query, setQuery] = useState(searchParams.get("q") || "");
  const [mode, setMode] = useState<Mode>(
    (searchParams.get("m") as Mode) || "answer",
  );
  const [strategy, setStrategy] = useState<HybridStrategy>(
    (searchParams.get("s") as HybridStrategy) || "rrf",
  );
  const [limit, setLimit] = useState(10);
  const [expanded, setExpanded] = useState<string | null>(null);

  const searchMutation = useMutation({
    mutationFn: async (): Promise<Array<SearchHit | CommunityHit>> => {
      if (!query.trim()) return [];
      let result;
      if (mode === "semantic") {
        result = await searchSemantic(query, limit);
      } else if (mode === "keyword") {
        result = await searchKeyword(query, limit);
      } else if (mode === "visual") {
        result = await searchVisual(query, Math.min(limit, 10));
      } else if (mode === "answer") {
        return []; // answer mode uses its own mutation
      } else {
        result = await searchHybrid({ query, strategy, limit });
      }
      if (!result.success) throw new Error(result.reason || "Search failed");
      return (result.data ?? []) as Array<SearchHit | CommunityHit>;
    },
  });

  const answerMutation = useMutation({
    mutationFn: async (): Promise<AnswerResult | null> => {
      if (!query.trim()) return null;
      const result = await searchAnswer(query, Math.min(limit, 10));
      if (!result.success) throw new Error(result.reason || "Answer failed");
      return result.data ?? null;
    },
  });

  const hits = searchMutation.data ?? [];
  const isCommunity = mode === "hybrid" && strategy === "community";
  const answerData = answerMutation.data;

  // Switching mode or strategy mid-session leaves stale results in the
  // mutation cache whose shape doesn't match the new view. Example: a
  // previous "keyword" search left SearchHit rows in `hits`; the user
  // switches to hybrid/community; CommunityResults tries to render
  // c.community_id.slice() on a SearchHit and the whole page crashes
  // into a black screen. Reset both mutations so the view falls back
  // to the empty state until the next search runs.
  useEffect(() => {
    searchMutation.reset();
    answerMutation.reset();
    setExpanded(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, strategy]);

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setExpanded(null);
    // Persist current state to URL before the search fires so a mid-
    // query navigation leaves a reproducible URL behind.
    const next = new URLSearchParams();
    if (query.trim()) next.set("q", query.trim());
    next.set("m", mode);
    if (mode === "hybrid") next.set("s", strategy);
    setSearchParams(next, { replace: true });
    if (mode === "answer") {
      answerMutation.mutate();
    } else {
      searchMutation.mutate();
    }
  };

  return (
    <div className="p-6 max-w-7xl">
      <h1 className="text-2xl font-bold mb-1">Search</h1>
      <p className="text-sm text-forge-muted mb-4">
        Ask a question — the LLM reads the actual page images and answers
        with citations. Use <strong>Keyword</strong> for exact codes (C12000,
        QW-451.1). Use <strong>Visual</strong> for ColPali page-image retrieval.
        Right-click thumbnails → Open in New Tab for full-size pages.
      </p>

      <form onSubmit={onSubmit} className="flex flex-wrap gap-3 items-end mb-6">
        <div className="flex-1 min-w-[300px]">
          <label className="block text-xs text-forge-muted mb-1">Query</label>
          <input
            className="w-full bg-forge-panel border border-forge-edge rounded px-3 py-2"
            placeholder="e.g. preheat requirements for P-1 materials"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <div>
          <label className="block text-xs text-forge-muted mb-1">Mode</label>
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as Mode)}
            className="bg-forge-panel border border-forge-edge rounded px-2 py-2"
          >
            <option value="answer">Answer (reads pages, cites sources)</option>
            <option value="keyword">Keyword (exact code/term match)</option>
            <option value="visual">Visual (ColPali page retrieval)</option>
            <optgroup label="Advanced">
              <option value="semantic">Semantic (text vectors)</option>
              <option value="hybrid">Hybrid (graph-aware)</option>
            </optgroup>
          </select>
        </div>
        {mode === "hybrid" && (
          <div>
            <label className="block text-xs text-forge-muted mb-1">Strategy</label>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value as HybridStrategy)}
              className="bg-forge-panel border border-forge-edge rounded px-2 py-2"
            >
              <option value="rrf">rrf (BM25 + dense + reranker)</option>
              <option value="graph_boosted">graph_boosted</option>
              <option value="vector_first">vector_first</option>
              <option value="graph_first">graph_first</option>
              <option value="community">community</option>
            </select>
          </div>
        )}
        <div>
          <label className="block text-xs text-forge-muted mb-1">Limit</label>
          <input
            type="number"
            value={limit}
            min={1}
            max={50}
            onChange={(e) => setLimit(parseInt(e.target.value) || 10)}
            className="bg-forge-panel border border-forge-edge rounded px-2 py-2 w-20"
          />
        </div>
        <button
          type="submit"
          className="bg-forge-accent text-black font-semibold rounded px-4 py-2 hover:brightness-110 disabled:opacity-50"
          disabled={searchMutation.isPending || answerMutation.isPending}
        >
          {(searchMutation.isPending || answerMutation.isPending) ? "Searching…" : mode === "answer" ? "Ask" : "Search"}
        </button>
      </form>

      {(searchMutation.isPending || answerMutation.isPending) && (
        <AnswerProgress inProgress isAnswer={mode === "answer"} />
      )}

      {(searchMutation.isError || answerMutation.isError) && (
        <div className="bg-rose-950 border border-rose-700 text-rose-200 rounded p-3 mb-4">
          {((searchMutation.error || answerMutation.error) as Error).message}
        </div>
      )}

      {mode === "answer" && answerData && (
        <div className="bg-forge-panel border border-forge-edge rounded-lg p-5 mb-6">
          <div className="text-sm font-semibold text-forge-primary mb-2">Answer</div>
          <div className="text-forge-fg whitespace-pre-wrap leading-relaxed">
            {answerData.answer}
          </div>
          {answerData.graph_context && answerData.graph_context.reasoning_chains.length > 0 && (
            <div className="mt-4 pt-3 border-t border-forge-edge">
              <div className="text-xs text-forge-muted mb-2">
                Knowledge graph reasoning ({answerData.graph_context.materials_found} materials,{" "}
                {answerData.graph_context.processes_found} processes,{" "}
                {answerData.graph_context.standards_found} standards found):
              </div>
              <div className="text-xs text-forge-muted/80 space-y-0.5 mb-3 font-mono">
                {answerData.graph_context.reasoning_chains.slice(0, 8).map((chain, i) => (
                  <div key={i}>→ {chain}</div>
                ))}
              </div>
            </div>
          )}
          {answerData.sources.length > 0 && (
            <div className={answerData.graph_context?.reasoning_chains.length ? "" : "mt-4 pt-3 border-t border-forge-edge"}>
              <div className="text-xs text-forge-muted mb-2">
                Sources ({answerData.sources.length} pages
                {answerData.graph_context?.pages_from_graph
                  ? `, ${answerData.graph_context.pages_from_graph} via graph traversal`
                  : ""}
                ):
              </div>
              <div className="flex flex-wrap gap-2">
                {answerData.sources.map((s, i) => {
                  // Extract hash from image_url: /images/{hash}/{page}
                  const parts = (s.image_url || "").split("/");
                  const srcHash = parts[2] || "";
                  const viewerUrl = `/app/view/${srcHash}/${s.page_number}?from=/search`;
                  return (
                    <a
                      key={i}
                      href={viewerUrl}
                      target="_blank"
                      rel="noopener"
                      className="text-xs bg-forge-edge rounded px-2 py-1 hover:text-forge-primary"
                      title="Open in page viewer with navigation"
                    >
                      {s.document_title.slice(0, 30)}… p.{s.page_number}
                    </a>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {mode !== "answer" && !searchMutation.isPending && hits.length === 0 && searchMutation.isSuccess && (
        <div className="text-forge-muted text-sm">No results.</div>
      )}

      {isCommunity ? (
        <CommunityResults hits={hits as CommunityHit[]} />
      ) : mode !== "answer" ? (
        <div className="space-y-4">
          {(hits as SearchHit[]).map((h) => (
            <HitCard
              key={h.page_id}
              hit={h}
              expanded={expanded === h.page_id}
              onToggleExpand={() =>
                setExpanded(expanded === h.page_id ? null : h.page_id)
              }
              mode={mode}
              strategy={strategy}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}

function HitCard({
  hit,
  expanded,
  onToggleExpand,
  mode,
  strategy,
}: {
  hit: SearchHit;
  expanded: boolean;
  onToggleExpand: () => void;
  mode: Mode;
  strategy: HybridStrategy;
}) {
  const [pageOffset, setPageOffset] = useState(0); // for forward/back navigation
  const hash = hit.image_url.split("/")[2];
  const currentPage = hit.page_number + pageOffset;
  const imgSrc = expanded
    ? pageImageUrl(hash, currentPage)
    : reducedImageUrl(hash, hit.page_number);

  // Reset page offset when collapsing
  const handleToggle = () => {
    if (expanded) setPageOffset(0);
    onToggleExpand();
  };

  return (
    <div className="bg-forge-panel border border-forge-edge rounded-lg overflow-hidden">
      <div className="p-4 flex gap-4">
        {!expanded && (
          <img
            src={reducedImageUrl(hash, hit.page_number)}
            alt={`page ${hit.page_number}`}
            className="border border-forge-edge bg-white cursor-pointer w-24 shrink-0 self-start hover:opacity-80"
            onClick={handleToggle}
            title="Click to expand page image"
          />
        )}
        <div className="flex-1 min-w-0">
          <div className="flex items-baseline gap-3 flex-wrap">
            <div className="font-semibold text-forge-fg truncate">
              {hit.document_title}
            </div>
            <div className="text-sm text-forge-muted">
              page {hit.page_number}
            </div>
            <div className="ml-auto font-mono text-sm text-forge-accent">
              {hit.score.toFixed(4)}
            </div>
          </div>
          {renderScoreDetails(hit, mode, strategy)}
          {hit.text_snippet && (
            <div className="text-sm text-forge-fg mt-2 line-clamp-3">
              {hit.text_snippet}
            </div>
          )}
          {renderEntities(hit)}
          {renderCommunities(hit)}
          {(hit.categories?.length || hit.tags?.length) ? (
            <div className="mt-2 flex flex-wrap gap-1">
              {hit.categories?.map((c) => (
                <span key={c} className="text-xs bg-sky-950 text-sky-300 px-2 py-0.5 rounded">
                  {c}
                </span>
              ))}
              {hit.tags?.map((t) => (
                <span key={t} className="text-xs bg-forge-edge text-forge-fg px-2 py-0.5 rounded">
                  #{t}
                </span>
              ))}
            </div>
          ) : null}
        </div>
      </div>
      {expanded && (
        <div className="border-t border-forge-edge">
          <div className="px-4 pt-3 pb-2 flex items-center gap-3 flex-wrap">
            {/* Page navigation */}
            <div className="flex items-center gap-1">
              <button
                className="text-xs border border-forge-edge rounded px-2 py-1 hover:bg-forge-edge disabled:opacity-30"
                onClick={() => setPageOffset(pageOffset - 1)}
                disabled={currentPage <= 1}
                title="Previous page"
              >
                ← Prev
              </button>
              <span className="text-sm font-mono px-2">
                p.{currentPage}
                {pageOffset !== 0 && (
                  <span className="text-forge-muted text-xs ml-1">
                    (match: p.{hit.page_number})
                  </span>
                )}
              </span>
              <button
                className="text-xs border border-forge-edge rounded px-2 py-1 hover:bg-forge-edge"
                onClick={() => setPageOffset(pageOffset + 1)}
                title="Next page"
              >
                Next →
              </button>
              {pageOffset !== 0 && (
                <button
                  className="text-xs text-forge-muted hover:text-forge-primary ml-1"
                  onClick={() => setPageOffset(0)}
                  title="Back to matched page"
                >
                  (reset)
                </button>
              )}
            </div>
            <div className="flex items-center gap-3 ml-auto">
              <a
                href={pageImageUrl(hash, currentPage)}
                target="_blank"
                rel="noopener"
                className="text-xs text-forge-secondary hover:underline"
              >
                open in new tab
              </a>
              <button
                className="text-xs border border-forge-edge rounded px-2 py-1 hover:bg-forge-edge"
                onClick={handleToggle}
              >
                close
              </button>
            </div>
          </div>
          <div className="px-4 pb-4">
            <img
              src={imgSrc}
              alt={`page ${currentPage}`}
              className="max-w-full max-h-[80vh] mx-auto border border-forge-edge bg-white"
              loading="lazy"
            />
          </div>
        </div>
      )}
    </div>
  );
}

function renderScoreDetails(hit: SearchHit, mode: string, strategy: string) {
  const bits: string[] = [];
  if (hit.base_score != null) bits.push(`base ${hit.base_score.toFixed(3)}`);
  if (hit.coarse_score != null) bits.push(`coarse ${hit.coarse_score.toFixed(3)}`);
  if (hit.entity_hits != null) bits.push(`ent_hits ${hit.entity_hits}`);
  if (hit.match_count != null) bits.push(`matches ${hit.match_count}`);
  if (hit.vector_similarity != null) bits.push(`sim ${hit.vector_similarity.toFixed(3)}`);
  if (!bits.length) return null;
  return (
    <div className="font-mono text-xs text-forge-muted mt-1">
      [{mode}{strategy ? `/${strategy}` : ""}] {bits.join(" · ")}
      {hit.matched_entities?.length ? (
        <span> · matched: {hit.matched_entities.slice(0, 5).join(", ")}</span>
      ) : null}
    </div>
  );
}

function renderEntities(hit: SearchHit) {
  if (!hit.entities || !hit.entities.length) return null;
  const valid = hit.entities.filter((e) => e.name);
  if (!valid.length) return null;
  return (
    <div className="mt-2 text-xs text-forge-muted">
      <span className="text-forge-muted">entities:</span>{" "}
      {valid.slice(0, 8).map((e, i) => (
        <span key={i} className="mr-2">
          <span className="text-forge-muted">{(e.kind || "").replace("MENTIONS_", "").replace("DESCRIBES_", "")}:</span>{" "}
          <span className="text-forge-fg">{e.name}</span>
        </span>
      ))}
    </div>
  );
}

function renderCommunities(hit: SearchHit) {
  if (!hit.communities || !hit.communities.length) return null;
  const valid = hit.communities.filter((c) => c.summary);
  if (!valid.length) return null;
  return (
    <div className="mt-2 text-xs text-forge-muted italic border-l-2 border-forge-edge pl-2">
      {valid[0].summary?.slice(0, 200)}
      {valid[0].summary && valid[0].summary.length > 200 ? "…" : ""}
    </div>
  );
}

function CommunityResults({ hits }: { hits: CommunityHit[] }) {
  // Defensively filter out rows that don't look like CommunityHits —
  // can happen if a stale useMutation result from another mode gets
  // rendered here before the mode-change effect clears it.
  const valid = (hits || []).filter(
    (c): c is CommunityHit => !!c && typeof c.community_id === "string",
  );
  if (valid.length === 0) {
    return (
      <div className="bg-forge-panel border border-forge-edge rounded-lg p-5 text-forge-muted">
        <p className="font-semibold mb-2">No communities found.</p>
        <p className="text-sm">
          Communities are built from entity relationships. Run{" "}
          <span className="font-mono text-forge-primary">Manage → GraphRAG Communities → rebuild</span>{" "}
          after entity extraction is complete for your documents. Try <strong>Answer</strong> or{" "}
          <strong>Keyword</strong> mode instead.
        </p>
      </div>
    );
  }
  return (
    <div className="space-y-4">
      {valid.map((c) => (
        <div key={c.community_id} className="bg-forge-panel border border-forge-edge rounded-lg p-4">
          <div className="flex items-baseline gap-3 mb-2">
            <div className="font-semibold">Community #{c.community_id.slice(0, 8)}</div>
            <div className="text-xs text-forge-muted">
              level {c.level} · {c.member_count} pages
            </div>
            <div className="ml-auto font-mono text-sm text-forge-accent">
              {c.score.toFixed(4)}
            </div>
          </div>
          <div className="text-sm text-forge-fg whitespace-pre-wrap">{c.summary}</div>
          {c.sample_pages && c.sample_pages.length > 0 && (
            <div className="mt-3 text-xs text-forge-muted">
              sample pages: {c.sample_pages.slice(0, 5).map((p) =>
                `${p.title} p.${p.page_number}`
              ).join(" · ")}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// Staged progress indicator for search/answer requests. Shows an animated
// dot, the current stage (time-based), and an elapsed counter so users
// know the request is alive. Important because the VLM synthesis step
// can legitimately take 30-90s on detailed engineering questions, and
// the previous "Searching…" button text wasn't enough feedback to
// distinguish "working" from "stuck".
function AnswerProgress({ inProgress, isAnswer }: { inProgress: boolean; isAnswer: boolean }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!inProgress) {
      setElapsed(0);
      return;
    }
    const started = Date.now();
    const t = setInterval(() => {
      setElapsed(Math.floor((Date.now() - started) / 1000));
    }, 500);
    return () => clearInterval(t);
  }, [inProgress]);

  let stage: string;
  if (!isAnswer) {
    stage = elapsed < 3 ? "Searching…" : "Searching (still working, index may be warming up)…";
  } else if (elapsed < 3) {
    stage = "Retrieving relevant chunks and pages…";
  } else if (elapsed < 10) {
    stage = "Running graph exploration and reranker…";
  } else if (elapsed < 30) {
    stage = "Reading page images with the vision LLM…";
  } else if (elapsed < 60) {
    stage = "Synthesizing the answer…";
  } else if (elapsed < 120) {
    stage = "Still working — the VLM takes longer on detailed questions…";
  } else {
    stage = "Taking unusually long. LM Studio may be reloading the model.";
  }

  return (
    <div className="bg-forge-panel border border-forge-edge rounded-lg p-4 mb-4 flex items-center gap-3">
      <div className="w-2 h-2 rounded-full bg-forge-accent animate-pulse" />
      <div className="flex-1">
        <div className="text-sm font-semibold text-forge-fg">{stage}</div>
        <div className="text-xs text-forge-muted mt-0.5 font-mono">
          elapsed {elapsed}s
        </div>
      </div>
    </div>
  );
}
