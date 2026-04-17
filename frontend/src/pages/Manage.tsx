import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  buildCommunities,
  deleteDocument,
  extractEntities,
  fetchHealth,
  getGpu,
  graphStats,
  listCommunities,
  listDocuments,
  listEntities,
  reembedDocument,
  unloadModel,
} from "../api/client";
import type { DocumentRow } from "../api/types";

export default function Manage() {
  return (
    <div className="p-6 max-w-7xl space-y-6">
      <h1 className="text-2xl font-bold">Manage</h1>
      <div className="grid md:grid-cols-3 gap-4">
        <StatsCard />
        <GpuCard />
        <CommunitiesCard />
      </div>
      <DocumentsTable />
      <EntitiesPanel />
    </div>
  );
}

function StatsCard() {
  const { data } = useQuery({
    queryKey: ["graph-stats"],
    queryFn: graphStats,
    refetchInterval: 5000,
  });
  const s = data?.data;
  const rows: Array<[string, number]> = s
    ? [
        ["Documents", s.documents],
        ["Pages", s.pages],
        ["Materials", s.materials],
        ["Processes", s.processes],
        ["Standards", s.standards],
        ["Clauses", s.clauses],
        ["Equipment", s.equipment],
        ["Communities", s.communities],
      ]
    : [];
  return (
    <div className="bg-forge-panel border border-forge-edge rounded-lg p-4">
      <h2 className="font-semibold mb-3">Graph Stats</h2>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
        {rows.map(([k, v]) => (
          <div key={k} className="flex items-baseline justify-between gap-3">
            <span className="text-forge-muted/80">{k}</span>
            <span className="font-mono tabular-nums">{v}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function GpuCard() {
  const qc = useQueryClient();
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: 3000,
  });
  const { data } = useQuery({
    queryKey: ["gpu"],
    queryFn: getGpu,
    refetchInterval: 3000,
  });
  const unload = useMutation({
    mutationFn: (name: string) => unloadModel(name),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["gpu"] }),
  });
  const g = data?.data;
  const h = health?.data;

  return (
    <div className="bg-forge-panel border border-forge-edge rounded-lg p-4">
      <h2 className="font-semibold mb-3">GPU</h2>
      {!g?.available && <div className="text-forge-muted text-sm">GPU not available.</div>}
      {g?.available && (
        <>
          <div className="text-xs text-forge-muted mb-1">{g.device_name}</div>
          <div className="text-sm mb-2">
            VRAM {(g.vram_free_bytes / 1e9).toFixed(1)} /{" "}
            {(g.vram_total_bytes / 1e9).toFixed(1)} GB free
          </div>
          <div className="h-1.5 bg-forge-bg rounded overflow-hidden mb-3">
            <div
              className="h-full bg-forge-accent"
              style={{
                width: `${
                  100 * (g.vram_used_bytes / Math.max(1, g.vram_total_bytes))
                }%`,
              }}
            />
          </div>
          {g.models.map((m) => (
            <div key={m.name} className="flex items-center gap-2 text-xs mb-1">
              <span className={`h-2 w-2 rounded-full ${m.loaded ? "bg-emerald-500" : "bg-forge-edge"}`} />
              <span className="font-mono">{m.name}</span>
              <span className="text-forge-muted">
                {m.loaded ? `loaded · idle ${m.last_used_s_ago}s` : "unloaded"}
              </span>
              {m.loaded && (
                <button
                  className="ml-auto text-xs text-forge-muted hover:text-forge-accent"
                  onClick={() => unload.mutate(m.name)}
                  disabled={unload.isPending}
                >
                  unload
                </button>
              )}
            </div>
          ))}
        </>
      )}
      {h?.neo4j_connected ? (
        <div className="mt-3 pt-3 border-t border-forge-edge text-xs text-forge-muted">
          Neo4j: <span className="text-emerald-400">connected</span>
        </div>
      ) : (
        <div className="mt-3 pt-3 border-t border-forge-edge text-xs text-rose-400">
          Neo4j: not connected
        </div>
      )}
    </div>
  );
}

function CommunitiesCard() {
  const qc = useQueryClient();
  const { data } = useQuery({
    queryKey: ["communities"],
    queryFn: () => listCommunities(undefined, 5),
    refetchInterval: 10000,
  });
  const build = useMutation({
    mutationFn: buildCommunities,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs"] }),
  });
  const comms = data?.data || [];
  return (
    <div className="bg-forge-panel border border-forge-edge rounded-lg p-4">
      <div className="flex items-center mb-3">
        <h2 className="font-semibold">GraphRAG Communities</h2>
        <button
          onClick={() => build.mutate()}
          disabled={build.isPending}
          className="ml-auto text-xs border border-forge-edge rounded px-2 py-1 hover:bg-forge-edge disabled:opacity-50"
          title="Rebuild hierarchical community summaries from the current graph"
        >
          {build.isPending ? "queuing…" : "rebuild"}
        </button>
      </div>
      {build.isSuccess && (
        <div className="text-xs text-emerald-400 mb-2">
          Queued as job {build.data?.data?.job_id.slice(0, 8)}…
        </div>
      )}
      {comms.length === 0 && (
        <div className="text-sm text-forge-muted">
          No communities yet. Run "rebuild" after you have extracted entities
          for enough documents.
        </div>
      )}
      <ol className="space-y-2">
        {comms.map((c) => (
          <li key={c.community_id} className="text-xs">
            <div className="font-mono text-forge-muted">
              L{c.level} · {c.actual_page_count} pages
            </div>
            <div className="text-forge-fg line-clamp-2">{c.summary}</div>
          </li>
        ))}
      </ol>
    </div>
  );
}

function DocumentsTable() {
  const qc = useQueryClient();
  const { data } = useQuery({
    queryKey: ["documents"],
    queryFn: () => listDocuments({ limit: 100 }),
    refetchInterval: 10000,
  });
  const docs = data?.data || [];

  // Track which actions were just triggered so we can show feedback
  const [actionFeedback, setActionFeedback] = useState<Record<string, string>>({});

  const showFeedback = (docId: string, msg: string) => {
    setActionFeedback((prev) => ({ ...prev, [docId]: msg }));
    setTimeout(() => setActionFeedback((prev) => {
      const next = { ...prev };
      delete next[docId];
      return next;
    }), 4000);
  };

  const reembed = useMutation({
    mutationFn: (id: string) => reembedDocument(id),
    onSuccess: (_data, id) => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
      showFeedback(id, "Re-embed queued — check Ingest tab for progress");
    },
    onError: (_err, id) => showFeedback(id, "Re-embed failed"),
  });
  const extract = useMutation({
    mutationFn: (id: string) => extractEntities(id),
    onSuccess: (_data, id) => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
      showFeedback(id, "Entity extraction queued — check Ingest tab");
    },
    onError: (_err, id) => showFeedback(id, "Extraction failed"),
  });
  const del = useMutation({
    mutationFn: (id: string) => deleteDocument(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["documents"] });
      qc.invalidateQueries({ queryKey: ["graph-stats"] });
    },
  });

  return (
    <div className="bg-forge-panel border border-forge-edge rounded-lg overflow-hidden">
      <div className="px-4 py-3 border-b border-forge-edge flex items-center">
        <h2 className="font-semibold">Documents ({docs.length})</h2>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-forge-bg text-forge-muted text-xs uppercase">
            <tr>
              <th className="text-left px-4 py-2">Title</th>
              <th className="text-right px-4 py-2">Pages</th>
              <th className="text-left px-4 py-2">Source</th>
              <th className="text-left px-4 py-2">Categories</th>
              <th className="text-left px-4 py-2">Tags</th>
              <th className="text-right px-4 py-2">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-forge-edge">
            {docs.map((d) => (
              <DocRow
                key={d.doc_id}
                doc={d}
                feedback={actionFeedback[d.doc_id]}
                onReembed={() => reembed.mutate(d.doc_id)}
                onExtract={() => extract.mutate(d.doc_id)}
                onDelete={() => {
                  if (confirm(`Delete "${d.title}" and all pages?`))
                    del.mutate(d.doc_id);
                }}
                reembedPending={reembed.isPending}
                extractPending={extract.isPending}
              />
            ))}
            {docs.length === 0 && (
              <tr>
                <td colSpan={6} className="px-4 py-6 text-center text-forge-muted">
                  No documents. Use the Ingest tab to upload PDFs.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DocRow({
  doc: d,
  feedback,
  onReembed,
  onExtract,
  onDelete,
  reembedPending,
  extractPending,
}: {
  doc: DocumentRow;
  feedback?: string;
  onReembed: () => void;
  onExtract: () => void;
  onDelete: () => void;
  reembedPending: boolean;
  extractPending: boolean;
}) {
  return (
    <>
      <tr>
        <td className="px-4 py-2 max-w-md truncate" title={d.filename}>
          {d.title}
        </td>
        <td className="px-4 py-2 text-right font-mono">{d.page_count}</td>
        <td className="px-4 py-2 text-xs text-forge-muted">{d.source_type}</td>
        <td className="px-4 py-2 text-xs text-forge-muted">
          {d.categories.join(", ") || <span className="text-forge-muted/50 italic">none</span>}
        </td>
        <td className="px-4 py-2 text-xs text-forge-muted">
          {d.tags.length > 0
            ? d.tags.map((t) => `#${t}`).join(" ")
            : <span className="text-forge-muted/50 italic">none</span>}
        </td>
        <td className="px-4 py-2 text-right">
          <div className="flex gap-1 justify-end">
            <ActionBtn
              onClick={onReembed}
              title="Re-generate visual embeddings (Nemotron). Check Ingest tab for progress."
              disabled={reembedPending}
            >
              {reembedPending ? "queuing…" : "re-embed"}
            </ActionBtn>
            <ActionBtn
              onClick={onExtract}
              title="Re-run LLM entity extraction. Check Ingest tab for progress."
              disabled={extractPending}
            >
              {extractPending ? "queuing…" : "extract"}
            </ActionBtn>
            <ActionBtn onClick={onDelete} title="Delete document and all pages" danger>
              delete
            </ActionBtn>
          </div>
        </td>
      </tr>
      {feedback && (
        <tr>
          <td colSpan={6} className="px-4 py-1">
            <div className="text-xs text-emerald-400 bg-emerald-950/30 rounded px-3 py-1.5 inline-block">
              {feedback}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

function ActionBtn({
  children,
  onClick,
  title,
  danger,
  disabled,
}: {
  children: React.ReactNode;
  onClick: () => void;
  title: string;
  danger?: boolean;
  disabled?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      disabled={disabled}
      className={`text-xs border rounded px-2 py-1 disabled:opacity-40 disabled:cursor-not-allowed ${
        danger
          ? "border-rose-800 hover:bg-rose-900/40 text-rose-300"
          : "border-forge-edge hover:bg-forge-edge"
      }`}
    >
      {children}
    </button>
  );
}

type EntityType = "material" | "process" | "standard" | "equipment";
const ENTITY_TABS: Array<{ key: EntityType; label: string }> = [
  { key: "material", label: "Materials" },
  { key: "process", label: "Processes" },
  { key: "standard", label: "Standards" },
  { key: "equipment", label: "Equipment" },
];

function EntitiesPanel() {
  const [tab, setTab] = useState<EntityType>("material");
  const { data } = useQuery({
    queryKey: ["entities", tab],
    queryFn: () => listEntities(tab, 100),
    refetchInterval: 10000,
  });
  const rows = data?.data || [];

  return (
    <div className="bg-forge-panel border border-forge-edge rounded-lg overflow-hidden">
      <div className="flex border-b border-forge-edge">
        {ENTITY_TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`px-4 py-2 text-sm ${
              tab === t.key
                ? "bg-forge-edge text-forge-accent"
                : "text-forge-muted hover:text-forge-fg"
            }`}
          >
            {t.label} ({tab === t.key ? rows.length : ""})
          </button>
        ))}
      </div>
      <div className="overflow-x-auto max-h-96">
        <table className="w-full text-sm">
          <thead className="bg-forge-bg text-forge-muted text-xs uppercase sticky top-0">
            <tr>
              <th className="text-left px-4 py-2">Name</th>
              <th className="text-left px-4 py-2">Details</th>
              <th className="text-right px-4 py-2">Pages</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-forge-edge">
            {rows.map((r) => (
              <tr key={r.key}>
                <td className="px-4 py-2 font-medium">{r.key}</td>
                <td className="px-4 py-2 text-xs text-forge-muted">
                  {renderEntityProps(r.properties)}
                </td>
                <td className="px-4 py-2 text-right font-mono">{r.page_mentions}</td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={3} className="px-4 py-4 text-center text-forge-muted">
                  No {tab} entities extracted yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function renderEntityProps(props: Record<string, unknown>): string {
  const bits: string[] = [];
  for (const k of ["material_type", "process_type", "organization", "equipment_type", "uns_number"]) {
    const v = props[k];
    if (v && v !== "unknown" && v !== "other") bits.push(`${k}: ${v}`);
  }
  const cn = props.common_names;
  if (Array.isArray(cn) && cn.length > 0) bits.push(`aliases: ${cn.join(", ")}`);
  return bits.join(" · ");
}
