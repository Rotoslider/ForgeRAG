import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  addDocumentTag,
  buildCommunities,
  deleteDocument,
  listCollections,
  moveDocument,
  removeDocumentTag,
  extractEntities,
  fetchHealth,
  getGpu,
  graphStats,
  listCommunities,
  listDocuments,
  listEntities,
  rebuildChunks,
  rebuildChunksBulk,
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

// Naive pluralization that handles the labels currently in the graph schema.
// Uncountable nouns are listed explicitly; the rule-based fallback covers
// future labels well enough (Widget → Widgets, Analysis → Analyses, etc.).
const UNCOUNTABLE = new Set(["Equipment", "Hardware", "Software"]);
function pluralize(label: string): string {
  if (UNCOUNTABLE.has(label)) return label;
  if (/[^aeiou]y$/i.test(label)) return label.slice(0, -1) + "ies";
  if (/(s|x|ch|sh|z|ss)$/i.test(label)) return label + "es";
  if (/is$/i.test(label)) return label.slice(0, -2) + "es";
  return label + "s";
}

function StatsCard() {
  const { data } = useQuery({
    queryKey: ["graph-stats"],
    queryFn: graphStats,
    refetchInterval: 5000,
  });
  const labels = data?.data?.labels || [];
  return (
    <div className="bg-forge-panel border border-forge-edge rounded-lg p-4">
      <h2 className="font-semibold mb-3">Graph Stats</h2>
      {labels.length === 0 ? (
        <div className="text-xs text-forge-muted">No nodes in the graph yet.</div>
      ) : (
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
          {labels.map(({ label, count }) => (
            <div key={label} className="flex items-baseline justify-between gap-3">
              <span className="text-forge-muted/80">{pluralize(label)}</span>
              <span className="font-mono tabular-nums">{count.toLocaleString()}</span>
            </div>
          ))}
        </div>
      )}
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
  const rebuild = useMutation({
    mutationFn: (args: { id: string; extractOnly: boolean }) =>
      rebuildChunks(args.id, { extract_only: args.extractOnly }),
    onSuccess: (_data, args) => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
      showFeedback(
        args.id,
        args.extractOnly
          ? "Entity re-extract queued — check Ingest tab"
          : "Chunk rebuild queued — check Ingest tab"
      );
    },
    onError: (_err, args) => showFeedback(args.id, "Rebuild failed"),
  });
  const del = useMutation({
    mutationFn: (id: string) => deleteDocument(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["documents"] });
      qc.invalidateQueries({ queryKey: ["graph-stats"] });
    },
  });

  // Multi-select state for bulk rebuild. Keeps the set of selected doc_ids.
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [bulkMsg, setBulkMsg] = useState<string | null>(null);

  const toggle = (id: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  const allSelected = docs.length > 0 && docs.every((d) => selected.has(d.doc_id));
  const toggleAll = () =>
    setSelected(allSelected ? new Set() : new Set(docs.map((d) => d.doc_id)));

  const bulkRebuild = useMutation({
    mutationFn: (opts: { extract_only?: boolean; skip_extract?: boolean; only_missing?: boolean }) =>
      rebuildChunksBulk({ doc_ids: Array.from(selected), ...opts }),
    onSuccess: (res) => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
      if (res.success && res.data) {
        const { queued, skipped, not_found } = res.data;
        setBulkMsg(
          `Queued ${queued}` +
            (skipped ? ` (skipped ${skipped} already done)` : "") +
            (not_found ? ` (${not_found} not found)` : "")
        );
        setSelected(new Set());
      } else {
        setBulkMsg(`Failed: ${res.reason ?? "unknown error"}`);
      }
      setTimeout(() => setBulkMsg(null), 6000);
    },
    onError: () => setBulkMsg("Bulk rebuild failed"),
  });

  return (
    <div className="bg-forge-panel border border-forge-edge rounded-lg overflow-hidden">
      <div className="px-4 py-3 border-b border-forge-edge flex items-center gap-3 flex-wrap">
        <h2 className="font-semibold">Documents ({docs.length})</h2>
        {selected.size > 0 && (
          <>
            <span className="text-xs text-forge-muted">{selected.size} selected</span>
            <button
              onClick={() => bulkRebuild.mutate({})}
              disabled={bulkRebuild.isPending}
              className="px-2 py-1 text-xs rounded border border-forge-edge hover:bg-forge-bg disabled:opacity-50"
              title="Full chunk rebuild + Phase 3 entity re-extraction"
            >
              {bulkRebuild.isPending ? "queuing…" : `rebuild (${selected.size})`}
            </button>
            <button
              onClick={() => bulkRebuild.mutate({ extract_only: true })}
              disabled={bulkRebuild.isPending}
              className="px-2 py-1 text-xs rounded border border-forge-edge hover:bg-forge-bg disabled:opacity-50"
              title="Only re-extract entities on pages missing topic_tags (cheap resume)"
            >
              extract-only
            </button>
            <button
              onClick={() => bulkRebuild.mutate({ only_missing: true })}
              disabled={bulkRebuild.isPending}
              className="px-2 py-1 text-xs rounded border border-forge-edge hover:bg-forge-bg disabled:opacity-50"
              title="Only rebuild docs that don't have chunks yet"
            >
              only-missing
            </button>
            <button
              onClick={() => setSelected(new Set())}
              className="px-2 py-1 text-xs rounded border border-forge-edge hover:bg-forge-bg"
            >
              clear
            </button>
          </>
        )}
        {bulkMsg && (
          <span className="text-xs text-emerald-400 bg-emerald-950/30 rounded px-2 py-1">
            {bulkMsg}
          </span>
        )}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-forge-bg text-forge-muted text-xs uppercase">
            <tr>
              <th className="px-3 py-2 w-8">
                <input
                  type="checkbox"
                  checked={allSelected}
                  onChange={toggleAll}
                  aria-label="Select all"
                />
              </th>
              <th className="text-left px-4 py-2">Title</th>
              <th className="text-right px-4 py-2">Pages</th>
              <th className="text-left px-4 py-2">Collection</th>
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
                selected={selected.has(d.doc_id)}
                onToggle={() => toggle(d.doc_id)}
                feedback={actionFeedback[d.doc_id]}
                onReembed={() => reembed.mutate(d.doc_id)}
                onExtract={() => extract.mutate(d.doc_id)}
                onRebuild={() => rebuild.mutate({ id: d.doc_id, extractOnly: false })}
                onExtractOnly={() => rebuild.mutate({ id: d.doc_id, extractOnly: true })}
                onDelete={() => {
                  if (confirm(`Delete "${d.title}" and all pages?`))
                    del.mutate(d.doc_id);
                }}
                reembedPending={reembed.isPending}
                extractPending={extract.isPending}
                rebuildPending={rebuild.isPending}
              />
            ))}
            {docs.length === 0 && (
              <tr>
                <td colSpan={7} className="px-4 py-6 text-center text-forge-muted">
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
  selected,
  onToggle,
  feedback,
  onReembed,
  onExtract,
  onRebuild,
  onExtractOnly,
  onDelete,
  reembedPending,
  extractPending,
  rebuildPending,
}: {
  doc: DocumentRow;
  selected: boolean;
  onToggle: () => void;
  feedback?: string;
  onReembed: () => void;
  onExtract: () => void;
  onRebuild: () => void;
  onExtractOnly: () => void;
  onDelete: () => void;
  reembedPending: boolean;
  extractPending: boolean;
  rebuildPending: boolean;
}) {
  const [editing, setEditing] = useState(false);
  const qc = useQueryClient();

  return (
    <>
      <tr className={selected ? "bg-forge-bg/40" : ""}>
        <td className="px-3 py-2 text-center">
          <input
            type="checkbox"
            checked={selected}
            onChange={onToggle}
            aria-label={`Select ${d.title}`}
          />
        </td>
        <td className="px-4 py-2 max-w-md truncate" title={d.filename}>
          {d.title}
        </td>
        <td className="px-4 py-2 text-right font-mono">{d.page_count}</td>
        <td className="px-4 py-2 text-xs text-forge-muted">
          {d.collection || "default"}
        </td>
        <td className="px-4 py-2 text-xs text-forge-muted">
          {d.categories.length > 0
            ? d.categories.join(", ")
            : <span className="text-forge-muted/50 italic">none</span>}
        </td>
        <td className="px-4 py-2 text-xs text-forge-muted">
          {d.tags.length > 0
            ? d.tags.map((t) => `#${t}`).join(" ")
            : <span className="text-forge-muted/50 italic">none</span>}
        </td>
        <td className="px-4 py-2 text-right">
          <div className="flex gap-1 justify-end">
            <ActionBtn onClick={() => setEditing(!editing)} title="Edit collection, tags, and categories">
              {editing ? "close" : "edit"}
            </ActionBtn>
            <ActionBtn onClick={onRebuild} title="Rebuild chunks + re-extract entities (Phase 5)" disabled={rebuildPending}>
              {rebuildPending ? "…" : "rebuild"}
            </ActionBtn>
            <ActionBtn onClick={onExtractOnly} title="Only re-extract entities on pages missing topic_tags" disabled={rebuildPending}>
              extract-only
            </ActionBtn>
            <ActionBtn onClick={onReembed} title="Re-embed (legacy)" disabled={reembedPending}>
              {reembedPending ? "…" : "re-embed"}
            </ActionBtn>
            <ActionBtn onClick={onExtract} title="Extract entities (legacy, page-level)" disabled={extractPending}>
              {extractPending ? "…" : "extract"}
            </ActionBtn>
            <ActionBtn onClick={onDelete} title="Delete" danger>delete</ActionBtn>
          </div>
        </td>
      </tr>
      {feedback && (
        <tr>
          <td colSpan={7} className="px-4 py-1">
            <div className="text-xs text-emerald-400 bg-emerald-950/30 rounded px-3 py-1.5 inline-block">
              {feedback}
            </div>
          </td>
        </tr>
      )}
      {editing && (
        <tr>
          <td colSpan={7} className="px-4 py-3 bg-forge-bg/50">
            <DocEditPanel doc={d} onDone={() => { setEditing(false); qc.invalidateQueries({ queryKey: ["documents"] }); }} />
          </td>
        </tr>
      )}
    </>
  );
}

function DocEditPanel({ doc }: { doc: DocumentRow; onDone: () => void }) {
  const qc = useQueryClient();
  const { data: collectionsResp } = useQuery({ queryKey: ["collections"], queryFn: listCollections });
  const collections = collectionsResp?.data || [];
  const currentCol = doc.collection || "default";

  const [col, setCol] = useState(currentCol);
  const [creatingCol, setCreatingCol] = useState(false);
  const [newCol, setNewCol] = useState("");
  const [newTag, setNewTag] = useState("");
  const [newCat, setNewCat] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["documents"] });
    qc.invalidateQueries({ queryKey: ["collections"] });
    qc.invalidateQueries({ queryKey: ["tags"] });
    qc.invalidateQueries({ queryKey: ["categories"] });
  };

  const doMove = async () => {
    const target = creatingCol ? newCol.trim() : col;
    if (!target || target === currentCol) return;
    setBusy(true);
    await moveDocument(doc.doc_id, target);
    setCreatingCol(false);
    setNewCol("");
    refresh();
    setBusy(false);
  };

  const doAddTag = async () => {
    if (!newTag.trim()) return;
    setBusy(true);
    await addDocumentTag(doc.doc_id, newTag.trim());
    setNewTag("");
    refresh();
    setBusy(false);
  };

  const doRemoveTag = async (tag: string) => {
    setBusy(true);
    await removeDocumentTag(doc.doc_id, tag);
    refresh();
    setBusy(false);
  };

  const doAddCat = async () => {
    if (!newCat.trim()) return;
    setBusy(true);
    // Use the documents/{id}/categories endpoint
    await fetch(`/documents/${doc.doc_id}/categories`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: newCat.trim() }),
    });
    setNewCat("");
    refresh();
    setBusy(false);
  };

  const doRemoveCat = async (cat: string) => {
    setBusy(true);
    await fetch(`/documents/${doc.doc_id}/categories/${encodeURIComponent(cat)}`, {
      method: "DELETE",
    });
    refresh();
    setBusy(false);
  };

  return (
    <div className="grid md:grid-cols-3 gap-4 text-sm">
      {/* Collection */}
      <div>
        <div className="text-xs text-forge-muted mb-1 font-semibold">Collection</div>
        <div className="flex gap-1">
          {!creatingCol ? (
            <select
              value={col}
              onChange={(e) => {
                if (e.target.value === "__new__") {
                  setCreatingCol(true);
                  setNewCol("");
                } else {
                  setCol(e.target.value);
                }
              }}
              className="bg-forge-panel border border-forge-edge rounded px-2 py-1 text-xs flex-1"
            >
              {collections.map((c) => (
                <option key={c.collection} value={c.collection}>{c.collection}</option>
              ))}
              <option value="__new__">+ New collection...</option>
            </select>
          ) : (
            <div className="flex gap-1 flex-1">
              <input
                value={newCol}
                onChange={(e) => setNewCol(e.target.value.replace(/\s+/g, "_").toLowerCase())}
                placeholder="collection_name"
                className="bg-forge-panel border border-forge-edge rounded px-2 py-1 text-xs flex-1"
                autoFocus
              />
              <button
                onClick={() => { setCreatingCol(false); setNewCol(""); }}
                className="text-xs text-forge-muted hover:text-forge-fg"
              >
                cancel
              </button>
            </div>
          )}
          <button
            onClick={doMove}
            disabled={busy || (creatingCol ? !newCol.trim() : col === currentCol)}
            className="text-xs bg-forge-primary/20 text-forge-primary border border-forge-primary/30 rounded px-2 py-1 hover:bg-forge-primary/30 disabled:opacity-30"
          >
            move
          </button>
        </div>
        <div className="text-xs text-forge-muted/60 mt-1">Current: {currentCol}</div>
      </div>

      {/* Tags */}
      <div>
        <div className="text-xs text-forge-muted mb-1 font-semibold">Tags</div>
        <div className="flex flex-wrap gap-1 mb-1">
          {doc.tags.map((t) => (
            <span
              key={t}
              className="text-xs bg-forge-edge rounded px-2 py-0.5 cursor-pointer hover:bg-forge-danger/20 group"
              onClick={() => doRemoveTag(t)}
              title="Click to remove"
            >
              #{t} <span className="text-forge-danger opacity-0 group-hover:opacity-100">×</span>
            </span>
          ))}
          {doc.tags.length === 0 && <span className="text-xs text-forge-muted/50 italic">none</span>}
        </div>
        <div className="flex gap-1">
          <input
            value={newTag}
            onChange={(e) => setNewTag(e.target.value)}
            placeholder="add tag"
            className="bg-forge-panel border border-forge-edge rounded px-2 py-1 text-xs flex-1"
            onKeyDown={(e) => { if (e.key === "Enter") doAddTag(); }}
          />
          <button
            onClick={doAddTag}
            disabled={busy || !newTag.trim()}
            className="text-xs border border-forge-edge rounded px-2 py-1 hover:bg-forge-edge disabled:opacity-30"
          >
            add
          </button>
        </div>
      </div>

      {/* Categories */}
      <div>
        <div className="text-xs text-forge-muted mb-1 font-semibold">Categories</div>
        <div className="flex flex-wrap gap-1 mb-1">
          {doc.categories.map((c) => (
            <span
              key={c}
              className="text-xs bg-forge-edge rounded px-2 py-0.5 cursor-pointer hover:bg-forge-danger/20 group"
              onClick={() => doRemoveCat(c)}
              title="Click to remove"
            >
              {c} <span className="text-forge-danger opacity-0 group-hover:opacity-100">×</span>
            </span>
          ))}
          {doc.categories.length === 0 && <span className="text-xs text-forge-muted/50 italic">none</span>}
        </div>
        <div className="flex gap-1">
          <input
            value={newCat}
            onChange={(e) => setNewCat(e.target.value)}
            placeholder="add category"
            className="bg-forge-panel border border-forge-edge rounded px-2 py-1 text-xs flex-1"
            onKeyDown={(e) => { if (e.key === "Enter") doAddCat(); }}
          />
          <button
            onClick={doAddCat}
            disabled={busy || !newCat.trim()}
            className="text-xs border border-forge-edge rounded px-2 py-1 hover:bg-forge-edge disabled:opacity-30"
          >
            add
          </button>
        </div>
      </div>
    </div>
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
