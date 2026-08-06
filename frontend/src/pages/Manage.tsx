import { Fragment, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  addDocumentTag,
  applyTags,
  auditCompleteness,
  backfillBlankFlags,
  buildCommunities,
  deepVerify,
  deleteDocument,
  extractMissingEntitiesAll,
  fillMissingBulk,
  recoverStrandedTextAll,
  getBackupProgress,
  getBackupSettings,
  listBackups,
  listCollections,
  listJobs,
  moveDocument,
  normalizeEntities,
  removeDocumentTag,
  extractEntities,
  browseDirectories,
  fetchHealth,
  getGpu,
  getJobControls,
  getSchedule,
  graphStats,
  openWatchFolder,
  listCommunities,
  listDocuments,
  listEntities,
  rebuildChunks,
  rebuildChunksBulk,
  reembedDocument,
  scanWatchNow,
  suggestTags,
  triggerFullBackup,
  unloadModel,
  updateBackupSettings,
  updateSchedule,
  updateWatch,
} from "../api/client";
import type {
  AuditReport,
  BackupSettingsData,
  BrowseListing,
  DocAudit,
  ScheduleConfig,
  WatchConfig,
} from "../api/client";
import type { DocumentRow, JobRow } from "../api/types";

export default function Manage() {
  return (
    <div className="p-6 max-w-7xl space-y-6">
      <h1 className="text-2xl font-bold">Manage</h1>
      <div className="grid md:grid-cols-3 gap-4">
        <StatsCard />
        <GpuCard />
        <CommunitiesCard />
      </div>
      <ScheduleCard />
      <BackupRestoreCard />
      <CompletenessCard />
      <VerificationCard />
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
      <h2 className="font-semibold mb-3" title="Live node counts in the Neo4j knowledge graph — refreshes every 5 s">Graph Stats</h2>
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
      <h2 className="font-semibold mb-3" title="Models load into VRAM on demand and auto-unload after sitting idle — an empty list here is normal">GPU</h2>
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
                  title="Free this model's VRAM now (it reloads automatically next time it's needed)"
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
    queryFn: () => listCommunities(undefined, 100),
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
          title="Re-clusters the whole entity graph (Leiden) and writes fresh LLM topic summaries. Run after ingesting or extracting a batch of documents — takes a while on a large graph"
        >
          {build.isPending ? "queuing…" : "rebuild"}
        </button>
      </div>
      {build.isSuccess && (
        <div className="text-xs text-emerald-400 mb-2">
          Queued as job {build.data?.data?.job_id.slice(0, 8)}… (progress in Ingest tab)
        </div>
      )}
      {comms.length === 0 && !build.isPending && !build.isSuccess && (
        <div className="text-sm text-forge-muted">
          No communities yet. Run "rebuild" after you have extracted entities
          for enough documents.
        </div>
      )}
      {/* Scrollable list — communities can run into the dozens after a big rebuild */}
      <ol className="space-y-2 max-h-80 overflow-y-auto pr-1">
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

const DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

function fmtWhen(iso: string): string {
  const d = new Date(iso);
  const today = new Date();
  const day =
    d.toDateString() === today.toDateString()
      ? "today"
      : d.toLocaleDateString(undefined, { weekday: "short" });
  return `${day} ${d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })}`;
}

function FolderBrowser({
  initialPath,
  onPick,
  onClose,
}: {
  initialPath?: string;
  onPick: (path: string) => void;
  onClose: () => void;
}) {
  const [listing, setListing] = useState<BrowseListing | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const navigate = async (path?: string) => {
    const res = await browseDirectories(path);
    if (res.success && res.data) {
      setListing(res.data);
      setErr(null);
    } else {
      setErr(res.reason || "cannot open folder");
    }
  };
  useEffect(() => {
    navigate(initialPath || undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-6">
      <div className="bg-forge-panel border border-forge-edge rounded-lg p-4 w-full max-w-lg">
        <div className="flex items-center gap-2 mb-2">
          <h3 className="text-sm font-semibold flex-1">Choose a folder</h3>
          <button
            type="button"
            onClick={() => listing && navigate(listing.home)}
            className="text-xs border border-forge-edge rounded px-2 py-1 hover:bg-forge-edge"
            title="Jump to your home folder"
          >
            home
          </button>
          <button
            type="button"
            onClick={onClose}
            className="text-xs border border-forge-edge rounded px-2 py-1 hover:bg-forge-edge"
          >
            close
          </button>
        </div>
        <div className="text-xs font-mono text-forge-muted mb-2 truncate" title={listing?.path}>
          {listing?.path || "…"}
        </div>
        {err && <div className="text-xs text-rose-400 mb-2">{err}</div>}
        <div className="max-h-64 overflow-y-auto border border-forge-edge rounded bg-forge-bg mb-3">
          {listing?.parent && (
            <button
              type="button"
              onClick={() => navigate(listing.parent!)}
              className="w-full text-left px-3 py-1.5 text-sm hover:bg-forge-edge font-mono"
            >
              ..
            </button>
          )}
          {listing?.dirs.map((d) => (
            <button
              key={d.path}
              type="button"
              onClick={() => navigate(d.path)}
              className="w-full text-left px-3 py-1.5 text-sm hover:bg-forge-edge font-mono truncate"
              title={d.path}
            >
              {d.name}/
            </button>
          ))}
          {listing && listing.dirs.length === 0 && (
            <div className="px-3 py-2 text-xs text-forge-muted">no subfolders</div>
          )}
        </div>
        <button
          type="button"
          disabled={!listing}
          onClick={() => listing && onPick(listing.path)}
          className="bg-forge-accent text-black font-semibold rounded px-3 py-1.5 text-sm hover:brightness-110 disabled:opacity-50"
        >
          Use this folder
        </button>
      </div>
    </div>
  );
}

function ScheduleCard() {
  const qc = useQueryClient();
  const { data } = useQuery({
    queryKey: ["schedule"],
    queryFn: getSchedule,
    refetchInterval: 5000,
  });
  const { data: collectionsResp } = useQuery({
    queryKey: ["collections"],
    queryFn: listCollections,
  });
  const payload = data?.data;
  const status = payload?.status;

  // Form state, seeded once from the server so live polling doesn't stomp
  // on half-edited fields.
  const [sched, setSched] = useState<ScheduleConfig | null>(null);
  const [watch, setWatch] = useState<WatchConfig | null>(null);
  const [note, setNote] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  useEffect(() => {
    if (payload && sched === null) setSched(payload.schedule);
    if (payload && watch === null) setWatch(payload.watch);
  }, [payload, sched, watch]);

  const refresh = () => qc.invalidateQueries({ queryKey: ["schedule"] });
  const saveSched = useMutation({
    mutationFn: (cfg: ScheduleConfig) => updateSchedule(cfg),
    onSuccess: (res) => {
      setNote(
        res.success
          ? { kind: "ok", text: "Schedule saved — takes effect within seconds." }
          : { kind: "err", text: res.reason || "save failed" }
      );
      if (res.success && res.data) setSched(res.data.schedule);
    },
    onSettled: refresh,
  });
  const saveWatch = useMutation({
    mutationFn: (cfg: WatchConfig) => updateWatch(cfg),
    onSuccess: (res) => {
      setNote(
        res.success
          ? { kind: "ok", text: "Watch folder saved." }
          : { kind: "err", text: res.reason || "save failed" }
      );
      if (res.success && res.data) setWatch(res.data.watch);
    },
    onSettled: refresh,
  });
  const scanNow = useMutation({
    mutationFn: scanWatchNow,
    onSuccess: (res) => {
      setNote(
        res.success
          ? {
              kind: "ok",
              text: `Scan done: ${res.data?.queued ?? 0} queued, ${res.data?.duplicates ?? 0} duplicate(s), ${res.data?.waiting ?? 0} waiting.`,
            }
          : { kind: "err", text: res.reason || "scan failed" }
      );
    },
    onSettled: refresh,
  });

  const collections = collectionsResp?.data || [];
  const busy = saveSched.isPending || saveWatch.isPending;
  const [browsing, setBrowsing] = useState(false);
  const openFolder = useMutation({
    mutationFn: openWatchFolder,
    onSuccess: (res) =>
      setNote(
        res.success
          ? { kind: "ok", text: "Opened in the file manager (on the ForgeRAG machine)." }
          : { kind: "err", text: res.reason || "could not open folder" }
      ),
  });

  return (
    <div className="bg-forge-panel border border-forge-edge rounded-lg p-5">
      <div className="flex items-center gap-3 mb-1 flex-wrap">
        <h2
          className="font-semibold"
          title="Automate the Pause all / Resume all switch on a daily window, and auto-ingest PDFs dropped into an inbox folder"
        >
          Schedule &amp; Automation
        </h2>
        {status && (
          <span
            className={`text-xs px-2 py-0.5 rounded ${
              status.pause_all
                ? "bg-amber-500/15 text-amber-300"
                : "bg-emerald-500/15 text-emerald-300"
            }`}
            title="Live state of the global job switch (same one as the Ingest tab's Pause/Resume all button)"
          >
            jobs {status.pause_all ? "paused" : "running"}
          </span>
        )}
        {status?.next_boundary && (
          <span className="text-xs text-forge-muted">
            next: {status.next_boundary.action === "resume" ? "resume" : "pause"}{" "}
            {fmtWhen(status.next_boundary.at)}
          </span>
        )}
      </div>
      <p className="text-xs text-forge-muted mb-4">
        Time-shift heavy work: keep the GPU free during the day and let repairs,
        re-embeds, and inbox ingests run at night. The schedule drives the same
        Pause all / Resume all switch as the Ingest tab — manual clicks still
        work and simply hold until the next scheduled boundary.
      </p>

      <div className="grid md:grid-cols-2 gap-6">
        {/* -------- processing window -------- */}
        <div>
          <h3 className="text-sm font-semibold mb-2">Processing window</h3>
          {sched === null ? (
            <div className="text-xs text-forge-muted">loading…</div>
          ) : (
            <>
              <label className="flex items-center gap-2 text-sm mb-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={sched.enabled}
                  onChange={(e) => setSched({ ...sched, enabled: e.target.checked })}
                />
                <span title="When enabled, jobs are automatically resumed at the start time and paused at the end time. Enabling applies the current window state immediately.">
                  Only run jobs during this window
                </span>
              </label>
              <div className="flex items-center gap-2 text-sm mb-3">
                <span className="text-forge-muted text-xs">from</span>
                <input
                  type="time"
                  value={sched.start}
                  onChange={(e) => setSched({ ...sched, start: e.target.value })}
                  className="bg-forge-bg border border-forge-edge rounded px-2 py-1 text-sm"
                  title="Window start — jobs resume here (e.g. 21:00 for overnight work)"
                />
                <span className="text-forge-muted text-xs">to</span>
                <input
                  type="time"
                  value={sched.end}
                  onChange={(e) => setSched({ ...sched, end: e.target.value })}
                  className="bg-forge-bg border border-forge-edge rounded px-2 py-1 text-sm"
                  title="Window end — jobs pause here. May be past midnight (21:00 → 06:30 runs overnight)."
                />
                <span className="text-[10px] text-forge-muted">(overnight OK)</span>
              </div>
              <div className="flex flex-wrap gap-1.5 mb-3">
                {DAY_LABELS.map((label, d) => {
                  const on = sched.days.includes(d);
                  return (
                    <button
                      key={label}
                      type="button"
                      onClick={() =>
                        setSched({
                          ...sched,
                          days: on
                            ? sched.days.filter((x) => x !== d)
                            : [...sched.days, d].sort(),
                        })
                      }
                      className={`text-xs rounded px-2 py-1 border ${
                        on
                          ? "border-forge-accent bg-forge-accent/20 text-forge-accent"
                          : "border-forge-edge text-forge-muted hover:bg-forge-edge"
                      }`}
                      title={`Window starting on ${label} (an overnight window started ${label} evening runs into the next morning)`}
                    >
                      {label}
                    </button>
                  );
                })}
              </div>
              <button
                type="button"
                disabled={busy}
                onClick={() => saveSched.mutate(sched)}
                className="bg-forge-accent text-black font-semibold rounded px-3 py-1.5 text-sm hover:brightness-110 disabled:opacity-50"
              >
                Save schedule
              </button>
            </>
          )}
        </div>

        {/* -------- watch folder -------- */}
        <div>
          <h3 className="text-sm font-semibold mb-2">Watch folder (auto-ingest inbox)</h3>
          {watch === null ? (
            <div className="text-xs text-forge-muted">loading…</div>
          ) : (
            <>
              <label className="flex items-center gap-2 text-sm mb-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={watch.enabled}
                  onChange={(e) => setWatch({ ...watch, enabled: e.target.checked })}
                />
                <span title="PDFs dropped into the inbox (subfolders included) are ingested automatically whenever job processing isn't paused — with a schedule on, they wait for the window. Files are picked up only once fully copied; duplicates are filed to duplicates/ untouched; ingested files move to ingested/ keeping their folder structure.">
                  Auto-ingest PDFs from an inbox folder
                </span>
              </label>
              <div className="flex items-center gap-2 mb-2">
                <input
                  type="text"
                  value={watch.path}
                  onChange={(e) => setWatch({ ...watch, path: e.target.value })}
                  placeholder={status?.watch.default_path || "absolute folder path"}
                  className="bg-forge-bg border border-forge-edge rounded px-2 py-1 text-xs font-mono flex-1"
                  title="Absolute path of the inbox folder. Leave empty to use the default (created for you)."
                />
                <button
                  type="button"
                  onClick={() => setBrowsing(true)}
                  className="text-xs border border-forge-edge rounded px-2 py-1 hover:bg-forge-edge"
                  title="Browse the ForgeRAG machine's folders and pick one"
                >
                  browse…
                </button>
                <button
                  type="button"
                  onClick={() => setWatch({ ...watch, path: "" })}
                  className="text-xs border border-forge-edge rounded px-2 py-1 hover:bg-forge-edge"
                  title={`Use the default inbox: ${status?.watch.default_path || ""}`}
                >
                  default
                </button>
                <button
                  type="button"
                  disabled={!payload?.watch.enabled || !status?.watch.path_ok || openFolder.isPending}
                  onClick={() => openFolder.mutate()}
                  className="text-xs border border-forge-edge rounded px-2 py-1 hover:bg-forge-edge disabled:opacity-50"
                  title="Open the saved inbox folder in the file manager (window appears on the ForgeRAG machine's screen)"
                >
                  open
                </button>
              </div>
              {browsing && (
                <FolderBrowser
                  initialPath={watch.path || status?.watch.default_path}
                  onPick={(p) => {
                    setWatch({ ...watch, path: p });
                    setBrowsing(false);
                  }}
                  onClose={() => setBrowsing(false)}
                />
              )}
              <div className="flex items-center gap-2 mb-3">
                <span className="text-xs text-forge-muted">collection</span>
                <input
                  type="text"
                  list="watch-collections"
                  value={watch.collection}
                  onChange={(e) => setWatch({ ...watch, collection: e.target.value })}
                  className="bg-forge-bg border border-forge-edge rounded px-2 py-1 text-xs flex-1"
                  title='Collection for inbox ingests. Leave as "default" to let auto-tagging organize each document.'
                />
                <datalist id="watch-collections">
                  {collections.map((c: { collection: string }) => (
                    <option key={c.collection} value={c.collection} />
                  ))}
                </datalist>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => saveWatch.mutate(watch)}
                  className="bg-forge-accent text-black font-semibold rounded px-3 py-1.5 text-sm hover:brightness-110 disabled:opacity-50"
                >
                  Save watch folder
                </button>
                <button
                  type="button"
                  disabled={!payload?.watch.enabled || scanNow.isPending}
                  onClick={() => scanNow.mutate()}
                  className="text-xs border border-forge-edge rounded px-2 py-1.5 hover:bg-forge-edge disabled:opacity-50"
                  title="Scan the inbox right now (skips the usual wait for files to finish copying). Queued jobs still respect Pause all."
                >
                  scan now
                </button>
              </div>
              {payload?.watch.enabled && status && (
                <div className="text-[11px] text-forge-muted mt-2 font-mono">
                  {status.watch.path_ok
                    ? `${status.watch.pending_files} PDF(s) in inbox`
                    : "inbox folder missing!"}
                  {status.watch.last_scan_at
                    ? ` · last scan ${fmtWhen(status.watch.last_scan_at)} (${status.watch.last_scan_note})`
                    : " · not scanned yet"}
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {note && (
        <div
          className={`text-xs mt-3 ${note.kind === "ok" ? "text-emerald-400" : "text-rose-400"}`}
        >
          {note.text}
        </div>
      )}

      {status && status.events.length > 0 && (
        <div className="mt-4">
          <h3
            className="text-xs font-semibold text-forge-muted mb-1"
            title="What the scheduler has done recently (window opened/closed, inbox files queued, config changes)"
          >
            Recent automation events
          </h3>
          <div className="max-h-28 overflow-y-auto border border-forge-edge rounded bg-forge-bg p-2 font-mono text-[11px] leading-4">
            {status.events.map((e, i) => (
              <div key={i}>
                <span className="text-forge-muted/70">
                  {new Date(e.ts).toLocaleString(undefined, {
                    month: "short",
                    day: "numeric",
                    hour: "2-digit",
                    minute: "2-digit",
                  })}{" "}
                </span>
                {e.message}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function BackupRestoreCard() {
  const qc = useQueryClient();

  // Load settings
  const { data: settingsResp } = useQuery({
    queryKey: ["backup-settings"],
    queryFn: getBackupSettings,
  });
  const currentSettings = settingsResp?.data;

  // Load backup list
  const { data: backupsResp } = useQuery({
    queryKey: ["backup-list"],
    queryFn: listBackups,
    refetchInterval: 15000,
  });
  const backups = backupsResp?.data?.backups || [];

  // Poll progress when a backup is running
  const { data: progressResp } = useQuery({
    queryKey: ["backup-progress"],
    queryFn: getBackupProgress,
    refetchInterval: 2000,
  });
  const progress = progressResp?.data;

  // Local state for editable fields
  const [destination, setDestination] = useState("");
  const [includeImages, setIncludeImages] = useState(true);
  const [includePdfs, setIncludePdfs] = useState(true);
  const [gdriveEnabled, setGdriveEnabled] = useState(true);
  const [gdriveDump, setGdriveDump] = useState(false);
  const [settingsLoaded, setSettingsLoaded] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);

  // Sync local state when settings load for the first time
  useEffect(() => {
    if (currentSettings && !settingsLoaded) {
      setDestination(currentSettings.destination || "");
      setIncludeImages(currentSettings.include_images);
      setIncludePdfs(currentSettings.include_pdfs);
      setGdriveEnabled(currentSettings.gdrive_enabled);
      setGdriveDump(currentSettings.gdrive_dump ?? false);
      setSettingsLoaded(true);
    }
  }, [currentSettings, settingsLoaded]);

  const saveMutation = useMutation({
    mutationFn: (body: BackupSettingsData) => updateBackupSettings(body),
    onSuccess: (res) => {
      if (res.success) {
        qc.invalidateQueries({ queryKey: ["backup-settings"] });
        setSaveMsg("Settings saved");
        setTimeout(() => setSaveMsg(null), 3000);
      } else {
        setSaveMsg(`Error: ${res.reason}`);
        setTimeout(() => setSaveMsg(null), 5000);
      }
    },
  });

  const backupMutation = useMutation({
    mutationFn: triggerFullBackup,
    onSuccess: (res) => {
      if (res.success) {
        qc.invalidateQueries({ queryKey: ["backup-progress"] });
      } else {
        setSaveMsg(`Backup failed: ${res.reason}`);
        setTimeout(() => setSaveMsg(null), 5000);
      }
    },
  });

  const handleSave = () => {
    saveMutation.mutate({
      destination,
      include_images: includeImages,
      include_pdfs: includePdfs,
      gdrive_enabled: gdriveEnabled,
      gdrive_dump: gdriveDump,
    });
  };

  const isRunning = progress?.running === true;

  return (
    <div className="bg-forge-panel border border-forge-edge rounded-lg p-4">
      <h2 className="font-semibold mb-4">Backup & Restore</h2>

      <div className="grid md:grid-cols-3 gap-6">
        {/* Column 1: Destination + Options */}
        <div className="space-y-3">
          <div>
            <label className="text-xs text-forge-muted font-semibold block mb-1">
              Backup Destination
            </label>
            <div className="flex gap-1">
              <input
                value={destination}
                onChange={(e) => setDestination(e.target.value)}
                placeholder="/mnt/nas/forgerag-backups"
                className="bg-forge-bg border border-forge-edge rounded px-2 py-1 text-xs flex-1 font-mono"
              />
              <button
                onClick={handleSave}
                disabled={saveMutation.isPending}
                className="text-xs border border-forge-edge rounded px-2 py-1 hover:bg-forge-edge disabled:opacity-50"
              >
                {saveMutation.isPending ? "..." : "save"}
              </button>
            </div>
            {!destination && (
              <div className="text-xs text-forge-muted/60 mt-1 italic">
                Not configured — set a path to enable backups
              </div>
            )}
          </div>

          <div className="space-y-1.5">
            <label className="text-xs text-forge-muted font-semibold block">
              Options
            </label>
            <label className="flex items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={includeImages}
                onChange={(e) => setIncludeImages(e.target.checked)}
              />
              Include page images
            </label>
            <label className="flex items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={includePdfs}
                onChange={(e) => setIncludePdfs(e.target.checked)}
              />
              Include source PDFs
            </label>
            <label className="flex items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={gdriveEnabled}
                onChange={(e) => setGdriveEnabled(e.target.checked)}
              />
              Upload to Google Drive
            </label>
            {gdriveEnabled && (
              <label className="flex items-center gap-2 text-xs ml-5">
                <input
                  type="checkbox"
                  checked={gdriveDump}
                  onChange={(e) => setGdriveDump(e.target.checked)}
                />
                Include Neo4j dump (~8-14 GB)
              </label>
            )}
          </div>

          {saveMsg && (
            <div className="text-xs text-emerald-400">{saveMsg}</div>
          )}
        </div>

        {/* Column 2: Backup action + progress */}
        <div className="space-y-3">
          <button
            onClick={() => backupMutation.mutate()}
            disabled={!destination || isRunning || backupMutation.isPending}
            className="w-full text-sm bg-forge-primary/20 text-forge-primary border border-forge-primary/30 rounded px-3 py-2 hover:bg-forge-primary/30 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {isRunning ? "Backup Running..." : backupMutation.isPending ? "Starting..." : "Start Full Backup"}
          </button>
          {!isRunning && (
            <div className="text-xs text-forge-muted">
              Creates a complete Neo4j dump (includes all embeddings), graph JSON,
              and selected files. Neo4j pauses briefly (~30s) during the dump.
            </div>
          )}

          {isRunning && progress && (
            <div className="space-y-1.5">
              <div className="h-2 bg-forge-bg rounded overflow-hidden">
                <div
                  className="h-full bg-forge-accent transition-all duration-300"
                  style={{ width: `${progress.percent || 0}%` }}
                />
              </div>
              <div className="flex justify-between text-xs text-forge-muted">
                <span className="truncate max-w-[12rem]" title={progress.current_file}>
                  {progress.current_file}
                </span>
                <span>{progress.percent || 0}%</span>
              </div>
              {progress.bytes_copied != null && (
                <div className="text-xs text-forge-muted">
                  {(progress.bytes_copied / 1e9).toFixed(2)} GB copied
                </div>
              )}
            </div>
          )}

          {!isRunning && progress?.finished_at && !progress.error && (
            <div className="text-xs text-emerald-400 bg-emerald-950/30 rounded px-2 py-1.5">
              Last backup complete: {((progress.total_bytes || 0) / 1e9).toFixed(2)} GB
              {progress.backup_path && (
                <div className="font-mono mt-0.5 text-forge-muted truncate" title={progress.backup_path}>
                  {progress.backup_path}
                </div>
              )}
              {progress.dump_skipped && (
                <div className="mt-1 text-amber-400">
                  Neo4j dump skipped: {progress.dump_skipped}
                </div>
              )}
            </div>
          )}

          {!isRunning && progress?.error && (
            <div className="text-xs text-rose-400 bg-rose-950/30 rounded px-2 py-1.5">
              Backup failed: {progress.error}
            </div>
          )}
        </div>

        {/* Column 3: Backup history */}
        <div>
          <div className="text-xs text-forge-muted font-semibold mb-2">
            Backup History ({backups.length})
          </div>
          {backups.length === 0 ? (
            <div className="text-xs text-forge-muted/50 italic">No backups found.</div>
          ) : (
            <div className="max-h-48 overflow-y-auto space-y-1.5 pr-1">
              {backups.slice(0, 10).map((b, i) => (
                <div
                  key={i}
                  className="text-xs border border-forge-edge/50 rounded px-2 py-1.5 bg-forge-bg/30"
                >
                  <div className="flex justify-between items-baseline">
                    <span className="font-mono text-forge-fg">
                      {b.timestamp.length > 15
                        ? b.timestamp.slice(0, 15)
                        : b.timestamp}
                    </span>
                    <span className="text-forge-muted">{b.size_mb} MB</span>
                  </div>
                  <div className="flex gap-2 mt-0.5 text-forge-muted/70">
                    <span>{b.source}</span>
                    {b.has_dump && <span>dump</span>}
                    {b.has_images && <span>images</span>}
                    {b.has_manifest && <span>manifest</span>}
                    <span className="ml-auto">{b.type === "full_backup" ? "full" : "graph"}</span>
                  </div>
                  {b.type === "full_backup" && (
                    <div className="mt-1 text-forge-muted/60 font-mono text-[10px] truncate" title={b.path}>
                      restore: ./scripts/restore.sh --from-local {b.path}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------ verification

// Failing verification checks that have a one-click server-side drain.
// label may reference the check's violation count; note warns about cost.
const VERIFY_FIXES: Record<string, {
  label: (violations: number) => string;
  note: string;
  run: () => Promise<ForgeResultShape>;
}> = {
  entity_extraction_complete: {
    label: (v) => `Extract missing entities now (${v.toLocaleString()} pages)`,
    note: "Background LLM work at roughly 8–10 s/page — days for a large backlog. Fully resumable; jobs appear on the Ingest page and every page is stamped so nothing is ever re-paid.",
    run: extractMissingEntitiesAll,
  },
  no_stranded_ocr_text: {
    label: (v) => `Recover OCR text now (${v.toLocaleString()} pages)`,
    note: "Copies Docling OCR text from chunks onto pages and embeds it — fast GPU work.",
    run: recoverStrandedTextAll,
  },
  blank_flags_populated: {
    label: (v) => `Backfill blank flags now (${v.toLocaleString()} pages)`,
    note: "Computes is_blank from the reduced images — a few minutes of CPU.",
    run: () => backfillBlankFlags(),
  },
  no_temp_rel_garbage: {
    label: (v) => `Normalize entities now (${v.toLocaleString()} junk edges)`,
    note: "Converts the junk __TEMP_REL edges back into real page links, then merges case/whitespace duplicate entities. Runs synchronously — usually well under a minute.",
    run: () => normalizeEntities(),
  },
  entities_case_deduped: {
    label: (v) => `Normalize entities now (${v.toLocaleString()} duplicates)`,
    note: "Merges entities that differ only by case/whitespace onto the most-mentioned spelling — every relationship is redirected, nothing is lost. Runs synchronously.",
    run: () => normalizeEntities(),
  },
};

// Loose response shape shared by the verify-fix drains — queued is a count
// for the job-per-doc endpoints and a boolean for the blank-flag backfill.
interface ForgeResultShape {
  success: boolean;
  reason?: string | null;
  data?: {
    queued?: number | boolean;
    pages?: number;
    merged?: number;
    temp_rels_recovered?: number;
  };
}

function VerificationCard() {
  const qc = useQueryClient();
  const verify = useQuery({
    queryKey: ["deep-verify"],
    queryFn: deepVerify,
    enabled: false, // full scans incl. on-disk file checks — on demand only
    staleTime: Infinity,
    retry: false,
  });
  const report = verify.data?.data;
  const [showPassed, setShowPassed] = useState(false);
  const [fixMsgs, setFixMsgs] = useState<Record<string, string>>({});
  const fix = useMutation({
    mutationFn: (checkName: string) => VERIFY_FIXES[checkName].run(),
    onSuccess: (res, checkName) => {
      const d = res.data as { queued?: number | boolean; pages?: number } | undefined;
      setFixMsgs((prev) => ({
        ...prev,
        [checkName]:
          `Queued — ${typeof d?.queued === "number" ? `${d.queued} jobs, ` : ""}` +
          `${(d?.pages ?? 0).toLocaleString()} pages. Watch progress on the Ingest ` +
          `page, then re-run verification once the queue drains.`,
      }));
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: (err, checkName) => {
      setFixMsgs((prev) => ({
        ...prev,
        [checkName]: `Failed to queue: ${(err as Error).message}`,
      }));
    },
  });

  return (
    <div className="bg-forge-panel border border-forge-edge rounded-lg p-5">
      <PausedJobsWarning />
      <div className="flex items-center gap-3 flex-wrap mb-2">
        <h2 className="font-semibold">Deep Verification</h2>
        <button
          onClick={() => verify.refetch()}
          disabled={verify.isFetching}
          className="bg-forge-accent text-black font-semibold rounded px-3 py-1.5 text-sm hover:brightness-110 disabled:opacity-50"
          title="The strictest check: proves every stored artifact is present and well-formed, with exact counts. Read-only. PASS means zero violations anywhere"
        >
          {verify.isFetching ? "Verifying… (full scans, ~1-2 min)" : "Run verification"}
        </button>
        {report && (
          <span
            className={`text-sm font-bold px-3 py-0.5 rounded border ${
              report.verdict === "PASS"
                ? "text-emerald-300 border-emerald-500/60 bg-emerald-500/10"
                : "text-rose-300 border-rose-500/60 bg-rose-500/10"
            }`}
          >
            {report.verdict} — {report.checks_passed}/{report.checks_total} checks
            {report.checks_warned > 0 ? ` (${report.checks_warned} warnings)` : ""}
            {report.verdict === "FAIL" && (
              <span className="font-normal">
                {" · failing: "}
                {report.checks
                  .filter((c) => c.status === "fail")
                  .map((c) => c.name)
                  .join(", ")}
              </span>
            )}
          </span>
        )}
        {report && (
          <span className="text-xs text-forge-muted">
            at {new Date(report.generated_at).toLocaleTimeString()}
          </span>
        )}
      </div>
      <p className="text-xs text-forge-muted mb-3">
        Verifies every pipeline invariant with exact counts and zero sampling:
        page counts and numbering, duplicates and orphans, images on disk, text
        consistency, embedding dimensions and blob byte-integrity, chunk
        completeness, entity extraction coverage, community summaries, and
        index health. PASS means zero violations anywhere.
      </p>
      {verify.isError && (
        <div className="text-sm text-rose-400 mb-3">
          Verification failed to run: {(verify.error as Error).message}
        </div>
      )}
      {report && !verify.isFetching && (
        <>
          <label className="flex items-center gap-1.5 cursor-pointer text-xs text-forge-muted mb-2">
            <input
              type="checkbox"
              checked={showPassed}
              onChange={(e) => setShowPassed(e.target.checked)}
            />
            show passed checks too
          </label>
          <div className="space-y-1">
            {report.checks
              .filter((c) => showPassed || c.status !== "pass")
              .map((c) => (
                <div
                  key={c.name}
                  className="flex items-start gap-2 text-xs border border-forge-edge/60 rounded px-2 py-1.5"
                >
                  <span
                    className={`h-2.5 w-2.5 rounded-full shrink-0 mt-0.5 ${
                      c.status === "pass"
                        ? "bg-emerald-500"
                        : c.status === "warn"
                        ? "bg-amber-500"
                        : "bg-rose-500"
                    }`}
                  />
                  <div className="min-w-0">
                    <span className="font-mono">{c.name}</span>
                    <span className="text-forge-muted"> — {c.description}</span>
                    {c.violations > 0 && (
                      <span className={c.status === "warn" ? "text-amber-400" : "text-rose-400"}>
                        {" "}({c.violations.toLocaleString()} violation{c.violations === 1 ? "" : "s"})
                      </span>
                    )}
                    {c.detail && (
                      <div className="text-forge-muted">{c.detail}</div>
                    )}
                    {c.status !== "pass" && VERIFY_FIXES[c.name] && (
                      <div className="mt-1.5">
                        {fixMsgs[c.name] ? (
                          <span
                            className={
                              fixMsgs[c.name].startsWith("Failed")
                                ? "text-rose-400"
                                : "text-emerald-400"
                            }
                          >
                            {fixMsgs[c.name]}
                          </span>
                        ) : (
                          <>
                            <button
                              disabled={fix.isPending}
                              onClick={() => fix.mutate(c.name)}
                              className="border border-forge-edge rounded px-2.5 py-1 hover:bg-forge-edge disabled:opacity-50 font-semibold"
                            >
                              {fix.isPending && fix.variables === c.name
                                ? "Queueing…"
                                : VERIFY_FIXES[c.name].label(c.violations)}
                            </button>
                            <div className="text-forge-muted mt-1">
                              {VERIFY_FIXES[c.name].note}
                            </div>
                          </>
                        )}
                      </div>
                    )}
                    {c.samples.length > 0 && (
                      <details className="text-forge-muted">
                        <summary className="cursor-pointer">sample offenders</summary>
                        <pre className="whitespace-pre-wrap break-all text-[10px] mt-1">
                          {JSON.stringify(c.samples, null, 1)}
                        </pre>
                      </details>
                    )}
                  </div>
                </div>
              ))}
            {report.verdict === "PASS" &&
              report.checks_warned === 0 &&
              !showPassed && (
                <div className="text-sm text-emerald-400">
                  Every check passed with zero violations.
                </div>
              )}
          </div>
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- audit

const AUDIT_ASPECTS: Array<{ key: string; label: string; tip: string }> = [
  { key: "pages", label: "pages",
    tip: "Was every PDF page registered and rendered? Shows Page nodes vs pages in the PDF. Red = ingestion died early; delete and re-ingest." },
  { key: "text", label: "page text",
    tip: "Does each page have extracted text? Scanned PDFs have none from the PDF itself — amber here means OCR text exists in chunks and can be recovered with one click." },
  { key: "text_embedding", label: "text embed",
    tip: "Every page with text should have a 1024-dim BGE-M3 vector (used by semantic/hybrid search). Counts show embedded / needed." },
  { key: "visual_embedding", label: "visual embed",
    tip: "Every non-blank page should have Nemotron 128-dim visual vectors (used by Visual search). Counts show embedded / needed." },
  { key: "chunks", label: "chunks",
    tip: "Docling structural chunks with LLM summaries — the primary retrieval unit for semantic and hybrid search. Counts show pages with chunks / pages that should have them." },
  { key: "entities", label: "entities",
    tip: "LLM-extracted Materials, Processes, Standards, and Equipment per page — this feeds the knowledge graph. Counts show pages extracted / text pages. Some pages legitimately contain no entities; those still count once extraction has run on them." },
];

const AUDIT_CIRCLE: Record<string, string> = {
  done: "bg-emerald-500",
  partial: "bg-amber-500",
  missing: "bg-rose-500",
  error: "bg-rose-500 ring-2 ring-rose-300/50",
  na: "border border-forge-muted/40 bg-transparent",
};

function AuditCell({ doc, aspect }: { doc: DocAudit; aspect: string }) {
  const a = doc.aspects[aspect];
  if (!a) return <td className="px-2 py-1" />;
  const counts =
    a.status === "na"
      ? "—"
      : a.needed > 0
      ? `${a.done}/${a.needed}`
      : `${a.done}`;
  return (
    <td
      className="px-2 py-1"
      title={`${aspect}: ${a.status}${a.detail ? ` — ${a.detail}` : ""}`}
    >
      <span className="flex items-center gap-1.5">
        <span
          className={`h-2.5 w-2.5 rounded-full shrink-0 ${
            AUDIT_CIRCLE[a.status] || AUDIT_CIRCLE.na
          }`}
        />
        <span
          className={`tabular-nums ${
            a.status === "missing" || a.status === "error"
              ? "text-rose-400"
              : a.status === "partial"
              ? "text-amber-400"
              : "text-forge-muted"
          }`}
        >
          {counts}
        </span>
      </span>
    </td>
  );
}

// Per-doc repair panel shown when a row's "fix" button is expanded. Renders
// one button per detected gap; each queues a job for just this document.
// New aspect types added later only need a branch here to become fixable.
function DocFixButtons({
  doc,
  busy,
  queuedLabel,
  onFill,
  onChunks,
  onReembed,
}: {
  doc: DocAudit;
  busy: boolean;
  queuedLabel: string | null;
  onFill: (opts: {
    text: boolean;
    visual: boolean;
    entities: boolean;
    recover_text?: boolean;
  }) => void;
  onChunks: () => void;
  onReembed: () => void;
}) {
  if (queuedLabel) {
    return (
      <div className="text-xs text-emerald-400 py-1">
        ✓ {queuedLabel} queued — its live status shows in this row; re-run the
        audit after it reads “done”.
      </div>
    );
  }
  const a = doc.aspects;
  const gap = (k: string) => ["missing", "partial"].includes(a[k]?.status);
  const missingCount = (k: string) =>
    Math.max(0, (a[k]?.needed ?? 0) - (a[k]?.done ?? 0));
  const textGap = gap("text_embedding");
  const visualGap = gap("visual_embedding");
  const entityGap =
    a.entities?.status !== "na" &&
    (a.entities?.needed ?? 0) > (a.entities?.done ?? 0);
  const chunksMissing = a.chunks?.status === "missing";
  const chunksPartial = a.chunks?.status === "partial";
  const dimError =
    a.text_embedding?.status === "error" || a.visual_embedding?.status === "error";
  const broken = a.pages?.status === "error";

  const btn =
    "text-xs border border-forge-edge rounded px-2.5 py-1 hover:bg-forge-edge disabled:opacity-50";

  return (
    <div className="flex items-center gap-2 flex-wrap py-1">
      {broken && (
        <span className="text-xs text-rose-400">
          No pages exist — delete this document in the table below and re-ingest
          the PDF; nothing can be repaired in place.
        </span>
      )}
      {doc.recoverable_text_pages > 0 && (
        <button
          disabled={busy}
          onClick={() =>
            onFill({ text: true, visual: false, entities: true, recover_text: true })
          }
          className={btn}
          title="Copies Docling OCR text from this document's chunks onto its textless pages, then embeds and extracts entities from the recovered text — one job"
        >
          Recover OCR text + embed + extract ({doc.recoverable_text_pages} pages)
        </button>
      )}
      {(textGap || visualGap) && (
        <button
          disabled={busy}
          onClick={() => onFill({ text: textGap, visual: visualGap, entities: false })}
          className={btn}
          title="Embeds only the pages that have no embedding — existing work untouched"
        >
          Fill missing embeddings
          {" "}({[
            textGap ? `${missingCount("text_embedding")} text` : null,
            visualGap ? `${missingCount("visual_embedding")} visual` : null,
          ].filter(Boolean).join(", ")} pages)
        </button>
      )}
      {entityGap && (
        <button
          disabled={busy}
          onClick={() => onFill({ text: false, visual: false, entities: true })}
          className={btn}
          title="Runs LLM extraction only on pages without entity relationships"
        >
          Extract missing entities (~{missingCount("entities")} pages)
        </button>
      )}
      {(chunksMissing || chunksPartial) && (
        <button
          disabled={busy}
          onClick={onChunks}
          className={btn}
          title={
            chunksPartial
              ? "Re-runs Docling + summaries + embeddings for ALL of this document's chunks"
              : "Builds chunks + summaries + embeddings for this document"
          }
        >
          {chunksMissing ? "Build chunks" : "Rebuild chunks (redoes all summaries)"}
        </button>
      )}
      {dimError && (
        <button
          disabled={busy}
          onClick={onReembed}
          className={`${btn} border-rose-500/60 text-rose-300`}
          title="Clears ALL embeddings for this document and regenerates with the current models — the only fix for wrong-dimension vectors"
        >
          Re-embed (clears + regenerates)
        </button>
      )}
      {!broken && !textGap && !visualGap && !entityGap && !chunksMissing && !chunksPartial && !dimError && doc.recoverable_text_pages === 0 && (
        <span className="text-xs text-forge-muted">nothing fixable — all gaps resolved</span>
      )}
    </div>
  );
}

// Amber banner shown on the repair cards while the global pause is on —
// without it, "queue a fix" looks like "run a fix" and nothing visibly
// happens (the exact confusion behind the 2026-08-06 "fixed but not
// fixed" reports).
function PausedJobsWarning() {
  const { data } = useQuery({
    queryKey: ["job-controls"],
    queryFn: getJobControls,
    refetchInterval: 5000,
  });
  if (!data?.data?.pause_all) return null;
  return (
    <div className="mb-3 text-xs rounded border border-amber-500/50 bg-amber-500/10 text-amber-300 px-3 py-2">
      Job processing is <b>paused</b> — repairs you queue here are created but
      hold at "⏸ held" and the audit numbers will NOT change until you click
      “Resume all” on the Ingest page (or your schedule window opens).
    </div>
  );
}

// Live status of a repair queued from the audit table, resolved by polling
// the jobs list. Every status is spelled out — a job that is held, stopped,
// or killed must NEVER read as done (that exact confusion is how "fixed"
// jobs silently went nowhere during the 2026-08-06 incident).
function RepairStatus({ label, job }: { label: string; job: JobRow | null }) {
  if (!job || job.status === "queued") {
    return (
      <span className="text-forge-muted" title={`${label} queued — waiting for a slot; drains a few at a time`}>
        {label}: queued…
      </span>
    );
  }
  if (job.status === "paused") {
    return (
      <span
        className="text-amber-400"
        title={`${label} is held because job processing is paused. Click "Resume all" on the Ingest page (or wait for the schedule window) and it will run.`}
      >
        ⏸ held — jobs are paused
      </span>
    );
  }
  if (job.status === "processing") {
    return (
      <span className="text-sky-300 animate-pulse" title={`${label} running`}>
        {job.current_step} {Math.round(job.progress_pct)}%
      </span>
    );
  }
  if (job.status === "failed") {
    return (
      <span
        className="text-rose-400"
        title={job.error_message || `${label} failed — see its logs on the Ingest page`}
      >
        ✗ failed — see logs
      </span>
    );
  }
  if (job.status === "cancelled") {
    return (
      <span className="text-amber-400" title={`${label} was stopped before finishing`}>
        ■ stopped
      </span>
    );
  }
  return (
    <span className="text-emerald-400" title={`${label} finished`}>
      ✓ done — re-run audit
    </span>
  );
}

function CompletenessCard() {
  const qc = useQueryClient();
  const [showAll, setShowAll] = useState(false);
  const [queuedMsg, setQueuedMsg] = useState<string | null>(null);
  const [expandedDoc, setExpandedDoc] = useState<string | null>(null);
  // The expanded fix panel can open below the fold of the table's scroll
  // area (always, for the last row) — bring it into view on expand.
  const expandedRowRef = useRef<HTMLTableRowElement | null>(null);
  useEffect(() => {
    // Instant, not smooth — smooth scrolls can be silently cancelled by
    // competing scroll activity, leaving the panel hidden below the fold.
    expandedRowRef.current?.scrollIntoView({ block: "nearest" });
  }, [expandedDoc]);
  // doc_id -> repair queued for it this session. Tracked jobs are polled so
  // each audit row shows queued → running → done live, and a banner offers a
  // one-click re-audit when everything has finished (the table's numbers are
  // frozen at the last audit run and would otherwise silently go stale).
  const [tracked, setTracked] = useState<Record<string, { jobId: string | null; label: string }>>({});
  const audit = useQuery({
    queryKey: ["completeness"],
    queryFn: auditCompleteness,
    enabled: false, // full page scan — run only on demand
    staleTime: Infinity,
    retry: false,
  });
  const report: AuditReport | undefined = audit.data?.data;

  const trackJobs = (
    jobs: Array<{ doc_id: string; job_id: string }> | undefined,
    label: string
  ) => {
    if (!jobs?.length) return;
    setTracked((prev) => {
      const next = { ...prev };
      for (const j of jobs) next[j.doc_id] = { jobId: j.job_id, label };
      return next;
    });
    qc.invalidateQueries({ queryKey: ["jobs"] });
  };

  const fill = useMutation({
    mutationFn: fillMissingBulk,
    onSuccess: (res, vars) => {
      setQueuedMsg(`Queued ${res.data?.queued ?? 0} fill-missing job(s).`);
      trackJobs(
        res.data?.jobs,
        vars.recover_text
          ? "OCR text recovery"
          : vars.entities
          ? "entity extraction"
          : "embedding fill"
      );
    },
  });
  const chunkFix = useMutation({
    mutationFn: (ids: string[]) =>
      rebuildChunksBulk({ doc_ids: ids, skip_extract: true, only_missing: true }),
    onSuccess: (res) => {
      setQueuedMsg(`Queued ${res.data?.queued ?? 0} chunk-rebuild job(s).`);
      trackJobs(res.data?.jobs, "chunk rebuild");
    },
  });

  // ---- per-document repairs (queued from a row's expanded fix panel)
  const fillOne = useMutation({
    mutationFn: (args: { doc_id: string; text: boolean; visual: boolean; entities: boolean }) =>
      fillMissingBulk({
        doc_ids: [args.doc_id],
        text: args.text,
        visual: args.visual,
        entities: args.entities,
      }),
    onSuccess: (res, args) =>
      trackJobs(
        res.data?.jobs,
        args.entities ? "entity extraction" : "embedding fill"
      ),
  });
  const chunkOne = useMutation({
    mutationFn: (docId: string) => rebuildChunks(docId, { skip_extract: true }),
    onSuccess: (res, docId) =>
      trackJobs(
        res.data ? [{ doc_id: docId, job_id: res.data.job_id }] : undefined,
        "chunk rebuild"
      ),
  });
  const reembedOne = useMutation({
    mutationFn: (docId: string) => reembedDocument(docId),
    onSuccess: (res, docId) =>
      trackJobs(
        res.data ? [{ doc_id: docId, job_id: res.data.job_id }] : undefined,
        "full re-embed"
      ),
  });
  const rowBusy = fillOne.isPending || chunkOne.isPending || reembedOne.isPending;

  // Poll job status while any tracked repair is outstanding.
  const trackedIds = Object.values(tracked)
    .map((t) => t.jobId)
    .filter(Boolean) as string[];
  const jobsPoll = useQuery({
    queryKey: ["completeness-jobs"],
    queryFn: () => listJobs(undefined, 300),
    refetchInterval: 4000,
    enabled: trackedIds.length > 0,
  });
  const jobById = new Map<string, JobRow>(
    (jobsPoll.data?.data ?? []).map((j) => [j.job_id, j])
  );
  const trackedJobStatus = (docId: string): JobRow | null => {
    const t = tracked[docId];
    if (!t?.jobId) return null;
    return jobById.get(t.jobId) ?? null;
  };
  const outstanding = Object.keys(tracked).filter((docId) => {
    const j = trackedJobStatus(docId);
    return !j || j.status === "queued" || j.status === "processing";
  });
  const allRepairsFinished =
    Object.keys(tracked).length > 0 && outstanding.length === 0;

  const docsWhere = (aspect: string, statuses: string[]) =>
    (report?.documents ?? [])
      .filter((d) => statuses.includes(d.aspects[aspect]?.status))
      .map((d) => d.doc_id);

  const embedGapDocs = report
    ? Array.from(
        new Set([
          ...docsWhere("text_embedding", ["missing", "partial"]),
          ...docsWhere("visual_embedding", ["missing", "partial"]),
        ])
      )
    : [];
  // Any deficit counts, not just below-threshold statuses — a doc at 80%
  // entity coverage still has unextracted pages, and 100% completeness
  // (the deep-verification bar) requires draining those too.
  const entityGapDocs = (report?.documents ?? [])
    .filter((d) => {
      const a = d.aspects.entities;
      return a && a.status !== "na" && a.needed > a.done;
    })
    .map((d) => d.doc_id);
  const chunkGapDocs = docsWhere("chunks", ["missing"]);
  const recoverDocs = (report?.documents ?? [])
    .filter((d) => d.recoverable_text_pages > 0)
    .map((d) => d.doc_id);
  const wrongDimDocs = report
    ? (report.documents ?? []).filter(
        (d) =>
          d.aspects.text_embedding?.status === "error" ||
          d.aspects.visual_embedding?.status === "error"
      )
    : [];
  const brokenDocs = docsWhere("pages", ["error"]);

  const problems = (report?.documents ?? []).filter((d) => d.overall !== "complete");
  const shown = showAll ? report?.documents ?? [] : problems;
  const busy = fill.isPending || chunkFix.isPending;

  return (
    <div className="bg-forge-panel border border-forge-edge rounded-lg p-5">
      <div className="flex items-center gap-3 flex-wrap mb-2">
        <h2 className="font-semibold">Pipeline Completeness</h2>
        <button
          onClick={() => {
            setQueuedMsg(null);
            audit.refetch();
          }}
          disabled={audit.isFetching}
          className="bg-forge-accent text-black font-semibold rounded px-3 py-1.5 text-sm hover:brightness-110 disabled:opacity-50"
          title="Scans the whole database (~10 s) and shows which pipeline steps are complete per document. Read-only — changes nothing"
        >
          {audit.isFetching ? "Auditing… (full page scan)" : report ? "Re-run audit" : "Run audit"}
        </button>
        {report && (
          <span className="text-xs text-forge-muted">
            audited {report.summary.documents} docs / {report.summary.total_pages.toLocaleString()} pages
            {" at "}{new Date(report.generated_at).toLocaleTimeString()}
            {" · "}dims verified: text {report.text_dim}, visual {report.visual_dim}
          </span>
        )}
      </div>
      <PausedJobsWarning />
      {allRepairsFinished && (
        <div className="border border-emerald-500/50 bg-emerald-500/10 rounded p-3 mb-3 text-xs flex items-center gap-3 flex-wrap">
          <span className="text-emerald-300 font-semibold">
            All queued repairs have finished.
          </span>
          <span className="text-forge-muted">
            The numbers below are from the last audit and are now stale.
          </span>
          <button
            onClick={() => {
              setTracked({});
              setQueuedMsg(null);
              audit.refetch();
            }}
            className="bg-forge-accent text-black font-semibold rounded px-2.5 py-1 hover:brightness-110"
          >
            Re-run audit now
          </button>
        </div>
      )}
      <p className="text-xs text-forge-muted mb-3">
        Checks every document against the graph itself — no re-processing. Each
        step leaves a fingerprint (embeddings, chunks, entity links), so missing
        work is visible without re-ingesting. Repairs are incremental: only
        pages missing an artifact get processed, existing work is never redone.
      </p>

      {audit.isError && (
        <div className="text-sm text-rose-400 mb-3">
          Audit failed: {(audit.error as Error).message}
        </div>
      )}
      {report && !audit.isFetching && (
        <>
          <div className="flex items-center gap-2 flex-wrap mb-3 text-xs">
            <span className="border border-emerald-500/50 text-emerald-400 rounded px-2 py-0.5" title="Documents where every pipeline step is fully done">
              {report.summary.complete} complete
            </span>
            <span className="border border-amber-500/50 text-amber-400 rounded px-2 py-0.5" title="Documents with at least one step missing or partial — expand a row's fix button to repair">
              {report.summary.incomplete} incomplete
            </span>
            <span className="border border-rose-500/50 text-rose-400 rounded px-2 py-0.5" title="Documents with unrepairable problems (no pages, wrong-dimension embeddings) — these need re-ingest or re-embed">
              {report.summary.error} error
            </span>
            <label className="ml-auto flex items-center gap-1.5 cursor-pointer text-forge-muted">
              <input
                type="checkbox"
                checked={showAll}
                onChange={(e) => setShowAll(e.target.checked)}
              />
              <span title="By default only documents with problems are listed">show complete docs too</span>
            </label>
          </div>

          <div className="flex gap-2 flex-wrap mb-3">
            {recoverDocs.length > 0 && (
              <button
                disabled={busy}
                onClick={() =>
                  fill.mutate({
                    doc_ids: recoverDocs,
                    text: true,
                    visual: false,
                    entities: true,
                    recover_text: true,
                  })
                }
                className="text-xs border border-forge-edge rounded px-3 py-1.5 hover:bg-forge-edge disabled:opacity-50"
                title="Scanned PDFs: copy Docling OCR text from chunks onto pages, then embed and extract entities from it"
              >
                Recover OCR text + embed + extract ({recoverDocs.length} docs)
              </button>
            )}
            {embedGapDocs.length > 0 && (
              <button
                disabled={busy}
                onClick={() =>
                  fill.mutate({ doc_ids: embedGapDocs, text: true, visual: true })
                }
                className="text-xs border border-forge-edge rounded px-3 py-1.5 hover:bg-forge-edge disabled:opacity-50"
              >
                Fill missing embeddings ({embedGapDocs.length} docs)
              </button>
            )}
            {entityGapDocs.length > 0 && (
              <button
                disabled={busy}
                onClick={() =>
                  fill.mutate({
                    doc_ids: entityGapDocs,
                    text: false,
                    visual: false,
                    entities: true,
                  })
                }
                className="text-xs border border-forge-edge rounded px-3 py-1.5 hover:bg-forge-edge disabled:opacity-50"
              >
                Extract missing entities ({entityGapDocs.length} docs)
              </button>
            )}
            {chunkGapDocs.length > 0 && (
              <button
                disabled={busy}
                onClick={() => chunkFix.mutate(chunkGapDocs)}
                className="text-xs border border-forge-edge rounded px-3 py-1.5 hover:bg-forge-edge disabled:opacity-50"
              >
                Build missing chunks ({chunkGapDocs.length} docs)
              </button>
            )}
          </div>

          {queuedMsg && (
            <div className="text-xs text-emerald-400 mb-3">{queuedMsg}</div>
          )}
          {(fill.isError || chunkFix.isError) && (
            <div className="text-xs text-rose-400 mb-3">
              {((fill.error || chunkFix.error) as Error)?.message}
            </div>
          )}
          {wrongDimDocs.length > 0 && (
            <div className="border border-rose-500/50 bg-rose-500/10 rounded p-3 mb-3 text-xs">
              <span className="text-rose-300 font-semibold">
                {wrongDimDocs.length} document(s) have wrong-dimension embeddings
              </span>{" "}
              <span className="text-forge-muted">
                — stale vectors from an older model. Filling can't fix these
                (the vectors aren't empty); use the per-document re-embed
                actions below, which clear and regenerate.
              </span>
            </div>
          )}
          {brokenDocs.length > 0 && (
            <div className="border border-rose-500/50 bg-rose-500/10 rounded p-3 mb-3 text-xs">
              <span className="text-rose-300 font-semibold">
                {brokenDocs.length} document(s) have no pages at all
              </span>{" "}
              <span className="text-forge-muted">
                — ingestion died before page creation. Delete them below and
                re-ingest the PDFs; nothing can be filled.
              </span>
            </div>
          )}

          {shown.length === 0 ? (
            <div className="text-sm text-emerald-400">
              Every document has all pipeline steps complete.
            </div>
          ) : (
            <div className="max-h-96 overflow-y-auto border border-forge-edge rounded">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-forge-panel">
                  <tr className="text-left text-forge-muted border-b border-forge-edge">
                    <th className="px-2 py-1.5 font-normal">document</th>
                    <th className="px-2 py-1.5 font-normal">collection</th>
                    {AUDIT_ASPECTS.map((a) => (
                      <th
                        key={a.key}
                        className="px-2 py-1.5 font-normal cursor-help underline decoration-dotted decoration-forge-muted/40 underline-offset-2"
                        title={a.tip}
                      >
                        {a.label}
                      </th>
                    ))}
                    <th className="px-2 py-1.5 font-normal" />
                  </tr>
                </thead>
                <tbody>
                  {shown.map((d) => (
                    <Fragment key={d.doc_id}>
                      <tr className="border-b border-forge-edge/50 last:border-b-0">
                        <td className="px-2 py-1 max-w-[16rem]">
                          <span className="flex items-center gap-1.5">
                            <span
                              className={`h-2 w-2 rounded-full shrink-0 ${
                                d.overall === "complete"
                                  ? "bg-emerald-500"
                                  : d.overall === "error"
                                  ? "bg-rose-500"
                                  : "bg-amber-500"
                              }`}
                            />
                            <span className="truncate" title={d.title}>
                              {d.title}
                            </span>
                          </span>
                        </td>
                        <td className="px-2 py-1 text-forge-muted">{d.collection}</td>
                        {AUDIT_ASPECTS.map((a) => (
                          <AuditCell key={a.key} doc={d} aspect={a.key} />
                        ))}
                        <td className="px-2 py-1 text-right whitespace-nowrap">
                          {d.overall !== "complete" &&
                            (tracked[d.doc_id] ? (
                              <RepairStatus
                                label={tracked[d.doc_id].label}
                                job={trackedJobStatus(d.doc_id)}
                              />
                            ) : (
                              <button
                                onClick={() =>
                                  setExpandedDoc(
                                    expandedDoc === d.doc_id ? null : d.doc_id
                                  )
                                }
                                className={`border border-forge-edge rounded px-2 py-0.5 hover:bg-forge-edge ${
                                  expandedDoc === d.doc_id ? "bg-forge-edge" : ""
                                }`}
                              >
                                fix {expandedDoc === d.doc_id ? "▾" : "▸"}
                              </button>
                            ))}
                        </td>
                      </tr>
                      {expandedDoc === d.doc_id && !tracked[d.doc_id] && (
                        <tr
                          ref={expandedRowRef}
                          className="border-b border-forge-edge/50 bg-forge-bg/50"
                        >
                          <td colSpan={AUDIT_ASPECTS.length + 3} className="px-3 py-1">
                            <DocFixButtons
                              doc={d}
                              busy={rowBusy}
                              queuedLabel={null}
                              onFill={(opts) =>
                                fillOne.mutate({ doc_id: d.doc_id, ...opts })
                              }
                              onChunks={() => chunkOne.mutate(d.doc_id)}
                              onReembed={() => reembedOne.mutate(d.doc_id)}
                            />
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}

const DOCS_PAGE_SIZE = 100;

function DocumentsTable() {
  const qc = useQueryClient();
  // Search + pagination. searchInput is what the user is typing; search is
  // the debounced value that actually hits the API (300ms after the last
  // keystroke) so we don't fire a request per character.
  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => {
      setSearch(searchInput.trim());
      setPage(0); // a new search always starts at the first page
    }, 300);
    return () => clearTimeout(t);
  }, [searchInput]);

  const { data } = useQuery({
    queryKey: ["documents", search, page],
    queryFn: () =>
      listDocuments({
        search: search || undefined,
        limit: DOCS_PAGE_SIZE,
        offset: page * DOCS_PAGE_SIZE,
      }),
    refetchInterval: 10000,
  });
  const docs = data?.data?.documents || [];
  const total = data?.data?.total ?? 0;
  const pageCount = Math.max(1, Math.ceil(total / DOCS_PAGE_SIZE));
  // If deletions (or a narrower search) shrink the result set below the
  // current page, snap back to the last page that still exists.
  useEffect(() => {
    if (page > 0 && page >= pageCount) setPage(pageCount - 1);
  }, [page, pageCount]);

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
    <div className="bg-forge-panel border border-forge-edge rounded-lg">
      {/* Bulk-action bar is sticky to the top of the viewport so it stays
          usable as the document list scrolls. Sits just above the sticky
          table header row. */}
      <div className="sticky top-0 z-30 bg-forge-panel px-4 py-3 border-b border-forge-edge rounded-t-lg flex items-center gap-3 flex-wrap">
        <h2 className="font-semibold">Documents ({total})</h2>
        <input
          value={searchInput}
          onChange={(e) => setSearchInput(e.target.value)}
          placeholder="search title or filename…"
          title="Case-insensitive search across every document's title and filename — not just the current page"
          className="bg-forge-bg border border-forge-edge rounded px-2 py-1 text-xs w-56"
        />
        {total > DOCS_PAGE_SIZE && (
          <span className="flex items-center gap-1 text-xs text-forge-muted">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              className="px-2 py-1 rounded border border-forge-edge hover:bg-forge-bg disabled:opacity-40"
            >
              ‹ prev
            </button>
            <span className="px-1">
              {page * DOCS_PAGE_SIZE + 1}–{Math.min(total, (page + 1) * DOCS_PAGE_SIZE)} of {total}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
              disabled={page >= pageCount - 1}
              className="px-2 py-1 rounded border border-forge-edge hover:bg-forge-bg disabled:opacity-40"
            >
              next ›
            </button>
          </span>
        )}
        {search && (
          <span className="text-xs text-forge-muted">
            {total} match{total === 1 ? "" : "es"} for “{search}”
          </span>
        )}
        {selected.size > 0 && (
          <>
            <span className="text-xs text-forge-muted">{selected.size} selected</span>
            <button
              onClick={() => bulkRebuild.mutate({})}
              disabled={bulkRebuild.isPending}
              className="px-2 py-1 text-xs rounded border border-forge-edge hover:bg-forge-bg disabled:opacity-50"
              title="For each selected document: re-run Docling chunking, summaries, and embeddings, then re-extract entities on pages missing topic tags. The heavy option"
            >
              {bulkRebuild.isPending ? "queuing…" : `rebuild (${selected.size})`}
            </button>
            <button
              onClick={() => bulkRebuild.mutate({ extract_only: true })}
              disabled={bulkRebuild.isPending}
              className="px-2 py-1 text-xs rounded border border-forge-edge hover:bg-forge-bg disabled:opacity-50"
              title="Skip chunking — only re-run LLM entity extraction on pages missing topic tags. Cheap way to resume after extraction failures"
            >
              extract-only
            </button>
            <button
              onClick={() => bulkRebuild.mutate({ only_missing: true })}
              disabled={bulkRebuild.isPending}
              className="px-2 py-1 text-xs rounded border border-forge-edge hover:bg-forge-bg disabled:opacity-50"
              title="Rebuild only the selected documents that have no chunks at all — documents that already have chunks are skipped"
            >
              only-missing
            </button>
            <button
              onClick={() => setSelected(new Set())}
              className="px-2 py-1 text-xs rounded border border-forge-edge hover:bg-forge-bg"
              title="Deselect all documents"
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
      {/* No overflow-x wrapper: CSS makes `overflow-x: auto` imply a
          vertical scroll container too, which breaks sticky-top-to-window
          for the thead. The Actions column is sticky-right and the
          tag/category cells wrap, so we don't need horizontal scroll. */}
      <table className="w-full text-sm">
        <thead className="text-forge-muted text-xs uppercase">
          <tr>
            <th className="px-3 py-2 w-8 sticky top-[3.25rem] z-10 bg-forge-bg">
              <input
                type="checkbox"
                checked={allSelected}
                onChange={toggleAll}
                aria-label="Select all"
              />
            </th>
            <th className="text-left px-4 py-2 sticky top-[3.25rem] z-10 bg-forge-bg">Title</th>
            <th className="text-right px-4 py-2 sticky top-[3.25rem] z-10 bg-forge-bg">Pages</th>
            <th className="text-left px-4 py-2 sticky top-[3.25rem] z-10 bg-forge-bg">Collection</th>
            <th className="text-left px-4 py-2 sticky top-[3.25rem] z-10 bg-forge-bg">Categories</th>
            <th className="text-left px-4 py-2 sticky top-[3.25rem] z-10 bg-forge-bg">Tags</th>
            <th className="text-right px-4 py-2 sticky top-[3.25rem] right-0 z-20 bg-forge-bg shadow-[-8px_0_8px_-8px_rgba(0,0,0,0.6)]">
              Actions
            </th>
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
  const [suggesting, setSuggesting] = useState(false);
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
        <td className="px-4 py-2 text-xs text-forge-muted align-top max-w-[12rem] break-words">
          {d.categories.length > 0
            ? d.categories.join(", ")
            : <span className="text-forge-muted/50 italic">none</span>}
        </td>
        <td className="px-4 py-2 text-xs text-forge-muted align-top max-w-[16rem] break-words leading-relaxed">
          {d.tags.length > 0
            ? d.tags.map((t) => `#${t}`).join(" ")
            : <span className="text-forge-muted/50 italic">none</span>}
        </td>
        <td
          className={`px-4 py-2 text-right align-top whitespace-nowrap sticky right-0 shadow-[-8px_0_8px_-8px_rgba(0,0,0,0.5)] ${
            selected ? "bg-forge-bg" : "bg-forge-panel"
          }`}
        >
          <div className="inline-flex gap-1 justify-end items-center">
            <ActionBtn onClick={() => setEditing(!editing)} title="Edit collection, tags, and categories">
              {editing ? "close" : "edit"}
            </ActionBtn>
            <ActionBtn onClick={() => setSuggesting(!suggesting)} title="LLM-suggest collection, categories, and tags">
              {suggesting ? "close" : "suggest"}
            </ActionBtn>
            <ActionBtn onClick={onRebuild} title="Re-run Docling chunking + summaries + embeddings for this document, then re-extract entities on pages missing topic tags" disabled={rebuildPending}>
              {rebuildPending ? "…" : "rebuild"}
            </ActionBtn>
            <OverflowMenu>
              <MenuItem onClick={onExtractOnly} disabled={rebuildPending}>
                extract-only · re-run LLM entity extraction on pages missing topic tags (no chunking)
              </MenuItem>
              <MenuItem onClick={onReembed} disabled={reembedPending}>
                re-embed · clear and regenerate ALL embeddings for this doc — hours of GPU; only needed after an embedding-model change
              </MenuItem>
              <MenuItem onClick={onExtract} disabled={extractPending}>
                extract · run entity extraction on pages not yet extracted (skips finished pages)
              </MenuItem>
              <MenuItem onClick={onDelete} danger>
                delete · permanently remove this document, its pages, and its images — the PDF on your disk is untouched
              </MenuItem>
            </OverflowMenu>
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
          <td colSpan={7} className="px-0 py-0 bg-forge-bg/50">
            <div className="sticky left-0 px-4 py-3 max-w-[min(100vw,80rem)]">
              <DocEditPanel doc={d} onDone={() => { setEditing(false); qc.invalidateQueries({ queryKey: ["documents"] }); }} />
            </div>
          </td>
        </tr>
      )}
      {suggesting && (
        <tr>
          <td colSpan={7} className="px-0 py-0 bg-forge-bg/50">
            {/* sticky-left keeps the panel anchored to the visible viewport
                when the table has scrolled horizontally (long tag lists can
                push the table wider than the window). */}
            <div className="sticky left-0 px-4 py-3 max-w-[min(100vw,80rem)]">
              <SuggestTagsPanel
                doc={d}
                onDone={() => {
                  setSuggesting(false);
                  qc.invalidateQueries({ queryKey: ["documents"] });
                  qc.invalidateQueries({ queryKey: ["collections"] });
                  qc.invalidateQueries({ queryKey: ["tags"] });
                  qc.invalidateQueries({ queryKey: ["categories"] });
                }}
              />
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

function SuggestTagsPanel({
  doc,
  onDone,
}: {
  doc: DocumentRow;
  onDone: () => void;
}) {
  // Load suggestion on mount — there's nothing to tweak before asking.
  const suggest = useMutation({
    mutationFn: () => suggestTags(doc.doc_id),
  });
  const [collection, setCollection] = useState<string>("");
  const [categories, setCategories] = useState<string[]>([]);
  const [tags, setTags] = useState<string[]>([]);
  const [newCat, setNewCat] = useState("");
  const [newTag, setNewTag] = useState("");
  const [mode, setMode] = useState<"merge" | "replace">("merge");

  // Fire the suggestion exactly once when the panel opens. useRef is the
  // standard "run once even under StrictMode double-invoke" guard for
  // effect-kicked async work.
  const firedRef = useRef(false);
  useEffect(() => {
    if (firedRef.current) return;
    firedRef.current = true;
    suggest.mutate(undefined, {
      onSuccess: (res) => {
        if (res.success && res.data) {
          setCollection(res.data.collection || "");
          setCategories(res.data.categories || []);
          setTags(res.data.tags || []);
        }
      },
    });
    // suggest is stable across renders for react-query; lint disabled
    // intentionally.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const apply = useMutation({
    mutationFn: () =>
      applyTags(doc.doc_id, {
        // Only send collection if it's non-empty AND differs from current
        // (or we're replacing). Avoids overwriting a manually-set collection.
        collection:
          collection && (mode === "replace" || collection !== doc.collection)
            ? collection
            : undefined,
        categories,
        tags,
        mode,
      }),
    onSuccess: (res) => {
      if (res.success) onDone();
    },
  });

  const removeCat = (c: string) =>
    setCategories((xs) => xs.filter((x) => x !== c));
  const removeTag = (t: string) => setTags((xs) => xs.filter((x) => x !== t));
  const addCat = () => {
    const v = newCat.trim();
    if (!v) return;
    if (!categories.includes(v)) setCategories([...categories, v]);
    setNewCat("");
  };
  const addTag = () => {
    const v = newTag.trim().toLowerCase().replace(/\s+/g, "-");
    if (!v) return;
    if (!tags.includes(v)) setTags([...tags, v]);
    setNewTag("");
  };

  if (suggest.isPending) {
    return (
      <div className="text-xs text-forge-muted">Asking the LLM for suggestions…</div>
    );
  }
  if (suggest.isError || (suggest.data && !suggest.data.success)) {
    const reason =
      suggest.data && !suggest.data.success
        ? suggest.data.reason
        : suggest.error instanceof Error
        ? suggest.error.message
        : "unknown error";
    return (
      <div className="text-xs text-rose-400">
        Suggestion failed: {reason}
      </div>
    );
  }

  const hasChanges =
    collection !== (doc.collection || "default") ||
    categories.length > 0 ||
    tags.length > 0;

  return (
    <div className="grid md:grid-cols-3 gap-4 text-sm">
      {/* Collection */}
      <div>
        <div className="text-xs text-forge-muted mb-1 font-semibold">
          Collection
        </div>
        <input
          value={collection}
          onChange={(e) =>
            setCollection(e.target.value.replace(/\s+/g, "_").toLowerCase())
          }
          placeholder="collection_name"
          className="bg-forge-panel border border-forge-edge rounded px-2 py-1 text-xs w-full"
        />
        <div className="text-xs text-forge-muted/60 mt-1">
          Current: {doc.collection || "default"}
        </div>
      </div>

      {/* Categories */}
      <div>
        <div className="text-xs text-forge-muted mb-1 font-semibold">
          Categories
        </div>
        <div className="flex flex-wrap gap-1 mb-1">
          {categories.map((c) => (
            <span
              key={c}
              className="text-xs bg-forge-edge rounded px-2 py-0.5 cursor-pointer hover:bg-forge-danger/20 group"
              onClick={() => removeCat(c)}
              title="Click to remove"
            >
              {c} <span className="text-forge-danger opacity-0 group-hover:opacity-100">×</span>
            </span>
          ))}
          {categories.length === 0 && (
            <span className="text-xs text-forge-muted/50 italic">none</span>
          )}
        </div>
        <div className="flex gap-1">
          <input
            value={newCat}
            onChange={(e) => setNewCat(e.target.value)}
            placeholder="add category"
            className="bg-forge-panel border border-forge-edge rounded px-2 py-1 text-xs flex-1"
            onKeyDown={(e) => { if (e.key === "Enter") addCat(); }}
          />
          <button
            onClick={addCat}
            disabled={!newCat.trim()}
            className="text-xs border border-forge-edge rounded px-2 py-1 hover:bg-forge-edge disabled:opacity-30"
          >
            add
          </button>
        </div>
      </div>

      {/* Tags */}
      <div>
        <div className="text-xs text-forge-muted mb-1 font-semibold">Tags</div>
        <div className="flex flex-wrap gap-1 mb-1">
          {tags.map((t) => (
            <span
              key={t}
              className="text-xs bg-forge-edge rounded px-2 py-0.5 cursor-pointer hover:bg-forge-danger/20 group"
              onClick={() => removeTag(t)}
              title="Click to remove"
            >
              #{t} <span className="text-forge-danger opacity-0 group-hover:opacity-100">×</span>
            </span>
          ))}
          {tags.length === 0 && (
            <span className="text-xs text-forge-muted/50 italic">none</span>
          )}
        </div>
        <div className="flex gap-1">
          <input
            value={newTag}
            onChange={(e) => setNewTag(e.target.value)}
            placeholder="add tag"
            className="bg-forge-panel border border-forge-edge rounded px-2 py-1 text-xs flex-1"
            onKeyDown={(e) => { if (e.key === "Enter") addTag(); }}
          />
          <button
            onClick={addTag}
            disabled={!newTag.trim()}
            className="text-xs border border-forge-edge rounded px-2 py-1 hover:bg-forge-edge disabled:opacity-30"
          >
            add
          </button>
        </div>
      </div>

      {/* Apply row spans all three columns. Stays left-aligned (no
          ml-auto) so the table's horizontal overflow can't push apply off
          the right edge of the viewport. */}
      <div className="md:col-span-3 flex flex-wrap items-center gap-x-4 gap-y-2 pt-2 border-t border-forge-edge/60">
        <button
          onClick={() => apply.mutate()}
          disabled={apply.isPending || !hasChanges}
          className="text-xs bg-forge-primary/20 text-forge-primary border border-forge-primary/30 rounded px-3 py-1 hover:bg-forge-primary/30 disabled:opacity-30"
        >
          {apply.isPending ? "applying…" : "apply"}
        </button>
        <button
          onClick={() => onDone()}
          className="text-xs border border-forge-edge rounded px-3 py-1 hover:bg-forge-edge"
        >
          cancel
        </button>
        <label className="text-xs flex items-center gap-2 ml-2">
          <input
            type="radio"
            checked={mode === "merge"}
            onChange={() => setMode("merge")}
          />
          merge (keep existing)
        </label>
        <label className="text-xs flex items-center gap-2">
          <input
            type="radio"
            checked={mode === "replace"}
            onChange={() => setMode("replace")}
          />
          replace (drop existing first)
        </label>
      </div>
    </div>
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

function OverflowMenu({ children }: { children: React.ReactNode }) {
  // Dropdown with the less-frequently-used row actions tucked out of the
  // primary button row. The menu is rendered into a portal with FIXED
  // positioning anchored to the trigger button. This is deliberate: the
  // trigger lives inside the documents table's scroll container (and a
  // `sticky` cell), so an `absolute` menu gets clipped by those ancestors'
  // overflow and loses the z-index race against the sticky header. A fixed
  // portal escapes every clipping/stacking context.
  const [open, setOpen] = useState(false);
  const btnRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ top: number; right: number }>({ top: 0, right: 0 });

  // Position the menu just below the button, right-aligned to it. Runs
  // before paint so the menu never flashes in the wrong spot.
  useLayoutEffect(() => {
    if (!open || !btnRef.current) return;
    const r = btnRef.current.getBoundingClientRect();
    setPos({ top: r.bottom + 4, right: window.innerWidth - r.right });
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onDocClick = (e: MouseEvent) => {
      const t = e.target as Node;
      if (btnRef.current?.contains(t) || menuRef.current?.contains(t)) return;
      setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    // Any scroll/resize would leave the fixed menu detached from its button,
    // so just close it.
    const onScroll = () => setOpen(false);
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKey);
    window.addEventListener("scroll", onScroll, true);
    window.addEventListener("resize", onScroll);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("scroll", onScroll, true);
      window.removeEventListener("resize", onScroll);
    };
  }, [open]);

  return (
    <>
      <button
        ref={btnRef}
        onClick={() => setOpen((v) => !v)}
        title="More actions"
        aria-haspopup="menu"
        aria-expanded={open}
        className="text-xs border border-forge-edge rounded px-2 py-1 hover:bg-forge-edge"
      >
        ⋯
      </button>
      {open &&
        createPortal(
          <div
            ref={menuRef}
            role="menu"
            style={{ position: "fixed", top: pos.top, right: pos.right }}
            className="min-w-[14rem] bg-forge-panel border border-forge-edge rounded shadow-lg z-[100] py-1 text-left"
            onClick={() => setOpen(false)}
          >
            {children}
          </div>,
          document.body
        )}
    </>
  );
}

function MenuItem({
  children,
  onClick,
  disabled,
  danger,
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  danger?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      role="menuitem"
      className={`w-full text-left text-xs px-3 py-1.5 disabled:opacity-40 disabled:cursor-not-allowed ${
        danger
          ? "text-rose-300 hover:bg-rose-900/30"
          : "hover:bg-forge-edge text-forge-fg"
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
