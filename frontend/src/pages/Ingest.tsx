import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  cancelJob,
  checkDuplicates,
  getJobControls,
  getJobLogs,
  listCategories,
  listCollections,
  listTags,
  listJobs,
  pauseAllJobs,
  pauseJob,
  restartJob,
  resumeAllJobs,
  resumeJob,
  sha256File,
  uploadPdf,
} from "../api/client";
import type { DuplicateInfo } from "../api/client";
import type { JobRow, JobStepRecord, StepStatus } from "../api/types";

// Folder picks include subfolders, so two different "manual.pdf" files can
// arrive in one selection — key and label by the relative path when the
// browser provides one, falling back to the bare name for single-file picks.
const fileLabel = (f: File) =>
  (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name;
const fileKey = (f: File) => `${fileLabel(f)}|${f.size}`;

export default function Ingest() {
  return (
    <div className="p-6 max-w-6xl">
      <h1 className="text-2xl font-bold mb-1">Ingest</h1>
      <p className="text-sm text-forge-muted mb-6">
        Upload engineering PDFs. The pipeline runs PDF → pages →
        text extraction → embeddings → entity extraction → graph.
        Progress is tracked per job. Re-uploading a PDF resumes where the
        previous run stopped.
      </p>
      <UploadForm />
      <ActiveJobs />
      <JobsList />
    </div>
  );
}

function UploadForm() {
  const qc = useQueryClient();
  const { data: catsResp } = useQuery({ queryKey: ["categories"], queryFn: listCategories });
  const { data: tagsResp } = useQuery({ queryKey: ["tags"], queryFn: listTags });
  const { data: collectionsResp } = useQuery({ queryKey: ["collections"], queryFn: listCollections });

  const [files, setFiles] = useState<File[]>([]);
  const [collection, setCollection] = useState("default");
  const [newCollection, setNewCollection] = useState("");
  const [selectedCats, setSelectedCats] = useState<string[]>([]);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [newTag, setNewTag] = useState("");
  const [uploadProgress, setUploadProgress] = useState<{ done: number; total: number } | null>(null);
  const [uploadErrors, setUploadErrors] = useState<Array<{ name: string; reason: string }>>([]);

  // Duplicate-check gate: when set, the user has selected files that include
  // PDFs whose SHA-256 already matches an existing :Document. We surface those
  // matches and let the user choose per-file: skip (default) or re-ingest.
  const [dupGate, setDupGate] = useState<{
    duplicates: Map<string, DuplicateInfo>; // fileKey -> existing doc info
    decisions: Map<string, "skip" | "force">; // fileKey -> action
  } | null>(null);
  const [hashing, setHashing] = useState<{ done: number; total: number } | null>(null);
  const [precheckError, setPrecheckError] = useState<string | null>(null);

  const filesInputRef = useRef<HTMLInputElement | null>(null);
  const folderInputRef = useRef<HTMLInputElement | null>(null);

  const collections = collectionsResp?.data || [];

  const dedupByName = (current: File[], incoming: File[]): File[] => {
    const seen = new Set(current.map(fileKey));
    const merged = [...current];
    for (const f of incoming) {
      if (!seen.has(fileKey(f))) {
        seen.add(fileKey(f));
        merged.push(f);
      }
    }
    return merged;
  };

  const addFiles = (incoming: FileList | null) => {
    if (!incoming || incoming.length === 0) return;
    const pdfs = Array.from(incoming).filter((f) =>
      f.name.toLowerCase().endsWith(".pdf")
    );
    if (pdfs.length === 0) return;
    setFiles((prev) => dedupByName(prev, pdfs));
  };

  const removeFile = (idx: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  // Sequential upload: one POST at a time so we can show per-file progress
  // and avoid firing N concurrent file-upload streams from the browser.
  // The backend starts each pipeline as a background task, so server-side
  // processing can still overlap — serialization here is just for upload I/O.
  const upload = useMutation({
    mutationFn: async (filesToUpload: File[]) => {
      if (filesToUpload.length === 0) throw new Error("Select at least one PDF first");
      const col = newCollection.trim() || collection;
      const errors: Array<{ name: string; reason: string }> = [];
      setUploadErrors([]);
      setUploadProgress({ done: 0, total: filesToUpload.length });
      for (let i = 0; i < filesToUpload.length; i++) {
        const f = filesToUpload[i];
        try {
          const res = await uploadPdf(f, col, selectedCats, selectedTags);
          if (!res.success) {
            errors.push({ name: f.name, reason: res.reason || "upload failed" });
          }
        } catch (e) {
          errors.push({ name: f.name, reason: (e as Error).message });
        }
        setUploadProgress({ done: i + 1, total: filesToUpload.length });
        // Refresh jobs list so queued items appear as they go
        qc.invalidateQueries({ queryKey: ["jobs"] });
      }
      setUploadErrors(errors);
      return { queued: filesToUpload.length - errors.length, failed: errors.length };
    },
    onSuccess: () => {
      setFiles([]);
      setSelectedCats([]);
      setSelectedTags([]);
      setNewCollection("");
      if (folderInputRef.current) folderInputRef.current.value = "";
      qc.invalidateQueries({ queryKey: ["jobs"] });
      qc.invalidateQueries({ queryKey: ["collections"] });
    },
  });

  // Hash all selected files, ask the backend which already exist. If any do,
  // open the gate; otherwise upload everything immediately.
  const startIngest = async () => {
    if (files.length === 0 || upload.isPending || hashing) return;
    setPrecheckError(null);
    setHashing({ done: 0, total: files.length });
    try {
      const hashes: string[] = [];
      for (let i = 0; i < files.length; i++) {
        hashes.push(await sha256File(files[i]));
        setHashing({ done: i + 1, total: files.length });
      }
      const res = await checkDuplicates(hashes);
      if (!res.success) {
        setPrecheckError(res.reason || "duplicate check failed");
        return;
      }
      const dupes = res.data?.duplicates || {};
      if (Object.keys(dupes).length === 0) {
        upload.mutate(files);
        return;
      }
      const duplicates = new Map<string, DuplicateInfo>();
      const decisions = new Map<string, "skip" | "force">();
      files.forEach((f, i) => {
        const info = dupes[hashes[i]];
        if (info) {
          const k = fileKey(f);
          duplicates.set(k, info);
          decisions.set(k, "skip");
        }
      });
      setDupGate({ duplicates, decisions });
    } catch (e) {
      setPrecheckError((e as Error).message);
    } finally {
      setHashing(null);
    }
  };

  const confirmDupGate = () => {
    if (!dupGate) return;
    const filtered = files.filter((f) => {
      const k = fileKey(f);
      if (!dupGate.duplicates.has(k)) return true;
      return dupGate.decisions.get(k) === "force";
    });
    setDupGate(null);
    if (filtered.length === 0) return;
    upload.mutate(filtered);
  };

  const setDupDecision = (key: string, action: "skip" | "force") => {
    if (!dupGate) return;
    const next = new Map(dupGate.decisions);
    next.set(key, action);
    setDupGate({ ...dupGate, decisions: next });
  };

  const categories = catsResp?.data || [];
  const tags = tagsResp?.data || [];

  return (
    <div className="bg-forge-panel border border-forge-edge rounded-lg p-5 mb-8">
      <h2 className="font-semibold mb-3">Upload PDFs</h2>

      <div className="grid md:grid-cols-2 gap-4 mb-4">
        <div>
          <label className="block text-xs text-forge-muted mb-1">
            Files ({files.length} selected)
          </label>
          <div className="flex flex-wrap gap-2 mb-2">
            {/* Explicit buttons that .click() their hidden inputs: Chromium
                forwards label activation to plain file inputs but not
                reliably to webkitdirectory ones, so the label-wrap pattern
                left "Add folder…" doing nothing. A programmatic click from
                the button's own handler opens both pickers dependably. */}
            <button
              type="button"
              onClick={() => filesInputRef.current?.click()}
              className="text-xs border border-forge-edge rounded px-3 py-1.5 cursor-pointer hover:bg-forge-edge"
              title="Pick one or more PDFs"
            >
              Add files…
            </button>
            <input
              ref={filesInputRef}
              type="file"
              accept="application/pdf"
              multiple
              onChange={(e) => {
                addFiles(e.target.files);
                e.target.value = "";
              }}
              className="hidden"
            />
            <button
              type="button"
              onClick={() => folderInputRef.current?.click()}
              className="text-xs border border-forge-edge rounded px-3 py-1.5 cursor-pointer hover:bg-forge-edge"
              title="Pick a folder — every PDF inside (including subfolders) is added"
            >
              Add folder…
            </button>
            <input
              ref={(el) => {
                folderInputRef.current = el;
                if (el) {
                  // webkitdirectory isn't in the standard React types but is
                  // the supported Chromium/Safari way to pick a directory.
                  el.setAttribute("webkitdirectory", "");
                  el.setAttribute("directory", "");
                }
              }}
              type="file"
              multiple
              onChange={(e) => {
                addFiles(e.target.files);
                e.target.value = "";
              }}
              className="hidden"
            />
            {files.length > 0 && (
              <button
                type="button"
                onClick={() => setFiles([])}
                className="text-xs text-forge-muted hover:text-forge-danger"
              >
                clear
              </button>
            )}
          </div>
          {files.length > 0 && (
            <div className="max-h-32 overflow-y-auto border border-forge-edge rounded bg-forge-bg">
              {files.map((f, i) => (
                <div
                  key={`${fileKey(f)}-${i}`}
                  className="flex items-center gap-2 px-2 py-1 text-xs border-b border-forge-edge last:border-b-0"
                >
                  <span className="flex-1 truncate font-mono" title={fileLabel(f)}>
                    {fileLabel(f)}
                  </span>
                  <span className="text-forge-muted tabular-nums">
                    {(f.size / 1e6).toFixed(1)} MB
                  </span>
                  <button
                    type="button"
                    onClick={() => removeFile(i)}
                    className="text-forge-muted hover:text-forge-danger"
                    title="remove"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
        <div>
          <label className="block text-xs text-forge-muted mb-1" title="The top-level shelf a document lives on (e.g. robotics, electronics). Every document belongs to exactly one collection; pick an existing one or create a new one.">Collection</label>
          <div className="flex gap-2">
            <select
              value={newCollection ? "__new__" : collection}
              onChange={(e) => {
                if (e.target.value === "__new__") {
                  setNewCollection(collection === "default" ? "" : collection);
                } else {
                  setCollection(e.target.value);
                  setNewCollection("");
                }
              }}
              className="bg-forge-bg border border-forge-edge rounded px-2 py-1.5 text-sm flex-1"
            >
              {collections.map((c) => (
                <option key={c.collection} value={c.collection}>
                  {c.collection} ({c.document_count} docs)
                </option>
              ))}
              {collections.length === 0 && <option value="default">default</option>}
              <option value="__new__">+ New collection...</option>
            </select>
            {newCollection !== "" && (
              <input
                placeholder="collection name"
                value={newCollection}
                onChange={(e) => setNewCollection(e.target.value.replace(/\s+/g, "_").toLowerCase())}
                className="bg-forge-bg border border-forge-edge rounded px-2 py-1.5 text-sm flex-1"
              />
            )}
          </div>
        </div>
      </div>


      <div className="grid md:grid-cols-2 gap-4 mb-4">
        <div>
          <label className="block text-xs text-forge-muted mb-1" title="Optional domain classifications (Ctrl-click for several). Leave empty on the default collection and the LLM auto-tagger will suggest some from the content.">
            Categories ({selectedCats.length} selected)
          </label>
          <select
            multiple
            value={selectedCats}
            onChange={(e) =>
              setSelectedCats(Array.from(e.target.selectedOptions, (o) => o.value))
            }
            className="w-full bg-forge-bg border border-forge-edge rounded px-2 py-2 h-24"
          >
            {categories.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name} ({c.document_count})
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs text-forge-muted mb-1" title="Optional free-form topic labels. Leave empty on the default collection and the auto-tagger fills them in; setting any manually turns auto-tagging off for this upload.">
            Tags ({selectedTags.length} selected)
          </label>
          <div className="flex gap-2 mb-2">
            <input
              placeholder="new tag"
              value={newTag}
              onChange={(e) => setNewTag(e.target.value)}
              className="flex-1 bg-forge-bg border border-forge-edge rounded px-2 py-1.5 text-sm"
              onKeyDown={(e) => {
                if (e.key === "Enter" && newTag.trim()) {
                  e.preventDefault();
                  if (!selectedTags.includes(newTag.trim())) {
                    setSelectedTags([...selectedTags, newTag.trim()]);
                  }
                  setNewTag("");
                }
              }}
            />
            <button
              type="button"
              className="px-3 py-1.5 text-sm border border-forge-edge rounded hover:bg-forge-edge"
              onClick={() => {
                if (newTag.trim() && !selectedTags.includes(newTag.trim())) {
                  setSelectedTags([...selectedTags, newTag.trim()]);
                  setNewTag("");
                }
              }}
            >
              add
            </button>
          </div>
          <div className="flex flex-wrap gap-1 mb-2">
            {selectedTags.map((t) => (
              <span
                key={t}
                className="text-xs bg-forge-bg border border-forge-edge rounded px-2 py-0.5 cursor-pointer"
                onClick={() => setSelectedTags(selectedTags.filter((x) => x !== t))}
                title="click to remove"
              >
                #{t} ×
              </span>
            ))}
          </div>
          {tags.length > 0 && (
            <details className="text-xs text-forge-muted">
              <summary className="cursor-pointer">Existing tags ({tags.length})</summary>
              <div className="flex flex-wrap gap-1 mt-2">
                {tags.map((t) => (
                  <span
                    key={t.name}
                    className="text-xs bg-forge-bg border border-forge-edge rounded px-2 py-0.5 cursor-pointer hover:border-forge-accent"
                    onClick={() =>
                      !selectedTags.includes(t.name) &&
                      setSelectedTags([...selectedTags, t.name])
                    }
                  >
                    #{t.name} ({t.document_count})
                  </span>
                ))}
              </div>
            </details>
          )}
        </div>
      </div>

      {dupGate && (
        <DuplicateGate
          files={files}
          duplicates={dupGate.duplicates}
          decisions={dupGate.decisions}
          onChange={setDupDecision}
          onConfirm={confirmDupGate}
          onCancel={() => setDupGate(null)}
        />
      )}

      <div className="flex items-center gap-3 flex-wrap">
        <button
          onClick={startIngest}
          disabled={files.length === 0 || upload.isPending || hashing !== null || dupGate !== null}
          className="bg-forge-accent text-black font-semibold rounded px-4 py-2 hover:brightness-110 disabled:opacity-50"
        >
          {hashing
            ? `Checking ${hashing.done}/${hashing.total}…`
            : upload.isPending
            ? uploadProgress
              ? `Uploading ${uploadProgress.done}/${uploadProgress.total}…`
              : "Uploading…"
            : files.length > 1
            ? `Start Ingestion (${files.length} files)`
            : "Start Ingestion"}
        </button>
        {precheckError && (
          <span className="text-rose-400 text-sm">{precheckError}</span>
        )}
        {upload.isError && (
          <span className="text-rose-400 text-sm">
            {(upload.error as Error).message}
          </span>
        )}
        {upload.isSuccess && upload.data && (
          <span className="text-emerald-400 text-sm">
            Queued {upload.data.queued} file(s)
            {upload.data.failed > 0 ? ` · ${upload.data.failed} failed` : ""}
          </span>
        )}
      </div>
      {uploadErrors.length > 0 && (
        <div className="mt-3 text-xs text-rose-400 space-y-1">
          {uploadErrors.map((e) => (
            <div key={e.name} className="font-mono">
              ✗ {e.name}: {e.reason}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function DuplicateGate({
  files,
  duplicates,
  decisions,
  onChange,
  onConfirm,
  onCancel,
}: {
  files: File[];
  duplicates: Map<string, DuplicateInfo>;
  decisions: Map<string, "skip" | "force">;
  onChange: (key: string, action: "skip" | "force") => void;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const dupFiles = files.filter((f) => duplicates.has(fileKey(f)));
  const newCount = files.length - dupFiles.length;
  const forceCount = Array.from(decisions.values()).filter((d) => d === "force").length;
  const willUpload = newCount + forceCount;

  return (
    <div className="border border-amber-500/60 bg-amber-500/10 rounded p-4 mb-4">
      <div className="font-semibold text-amber-300 mb-2">
        {dupFiles.length} of {files.length} file(s) already in the database
      </div>
      <p className="text-xs text-forge-muted mb-3">
        These PDFs match an existing document by SHA-256. Re-ingesting reuses
        the existing doc_id but re-runs embeddings and entity extraction.
      </p>
      <div className="space-y-2 mb-3 max-h-64 overflow-y-auto">
        {dupFiles.map((f) => {
          const k = fileKey(f);
          const info = duplicates.get(k)!;
          const action = decisions.get(k) || "skip";
          return (
            <div key={k} className="bg-forge-bg border border-forge-edge rounded p-2 text-xs">
              <div className="font-mono truncate mb-1">{f.name}</div>
              <div className="text-forge-muted mb-2">
                already ingested as <span className="text-forge-fg">{info.title}</span>
                {" · "}{info.page_count} pages · collection: {info.collection}
              </div>
              <div className="flex gap-3">
                <label className="flex items-center gap-1 cursor-pointer">
                  <input
                    type="radio"
                    name={`dup-${k}`}
                    checked={action === "skip"}
                    onChange={() => onChange(k, "skip")}
                  />
                  <span>Skip</span>
                </label>
                <label className="flex items-center gap-1 cursor-pointer">
                  <input
                    type="radio"
                    name={`dup-${k}`}
                    checked={action === "force"}
                    onChange={() => onChange(k, "force")}
                  />
                  <span>Re-ingest anyway</span>
                </label>
              </div>
            </div>
          );
        })}
      </div>
      <div className="flex gap-2">
        <button
          onClick={onConfirm}
          disabled={willUpload === 0}
          className="bg-forge-accent text-black font-semibold rounded px-3 py-1.5 text-sm hover:brightness-110 disabled:opacity-50"
        >
          {willUpload === 0
            ? "Nothing to upload"
            : `Continue with ${willUpload} file${willUpload === 1 ? "" : "s"}`}
        </button>
        <button
          onClick={onCancel}
          className="border border-forge-edge rounded px-3 py-1.5 text-sm hover:bg-forge-edge"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}

// How many active jobs to render as full cards. Bulk drains can queue
// hundreds; everything past this cap is summarized in one footer line.
const ACTIVE_JOBS_SHOWN = 8;

function ActiveJobs() {
  const qc = useQueryClient();
  const { data: ctlResp } = useQuery({
    queryKey: ["job-controls"],
    queryFn: getJobControls,
    refetchInterval: 3000,
  });
  const { data: activeResp } = useQuery({
    queryKey: ["jobs-active"],
    queryFn: () => listJobs("active", 100),
    refetchInterval: 3000,
  });
  const controls = ctlResp?.data;
  const jobs = activeResp?.data || [];
  const activeTotal = controls?.active ?? jobs.length;
  const counts = controls?.counts || {};
  const pauseAll = !!controls?.pause_all;

  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["job-controls"] });
    qc.invalidateQueries({ queryKey: ["jobs-active"] });
    qc.invalidateQueries({ queryKey: ["jobs"] });
  };
  const toggleAll = useMutation({
    mutationFn: () => (pauseAll ? resumeAllJobs() : pauseAllJobs()),
    onSettled: refresh,
  });

  // Backend orders processing → paused → queued, so the cap always keeps
  // the jobs that are actually doing something.
  const shown = jobs.slice(0, ACTIVE_JOBS_SHOWN);
  const hidden = Math.max(0, activeTotal - shown.length);

  return (
    <div className="mb-8">
      <div className="flex items-center mb-3 gap-3 flex-wrap">
        <h2 className="font-semibold">Active Jobs</h2>
        <span className="text-xs text-forge-muted" title="Jobs currently running, paused, or waiting for a slot">
          {activeTotal === 0
            ? "none"
            : [
                counts.processing ? `${counts.processing} running` : null,
                counts.paused ? `${counts.paused} paused` : null,
                counts.queued ? `${counts.queued} queued` : null,
              ]
                .filter(Boolean)
                .join(" · ")}
        </span>
        {(activeTotal > 0 || pauseAll) && (
          <button
            type="button"
            onClick={() => toggleAll.mutate()}
            disabled={toggleAll.isPending}
            className={`ml-auto text-xs font-semibold rounded px-3 py-1.5 disabled:opacity-50 ${
              pauseAll
                ? "bg-emerald-600 text-white hover:brightness-110"
                : "bg-amber-500 text-black hover:brightness-110"
            }`}
            title={
              pauseAll
                ? "Continue all paused and held jobs where they left off"
                : "Every job holds after its current page/batch — frees the GPU for other work. Jobs stay paused (even across restarts) until you resume."
            }
          >
            {pauseAll ? "▶ Resume all" : "⏸ Pause all (free GPU)"}
          </button>
        )}
      </div>
      {pauseAll && (
        <div className="mb-3 text-xs rounded border border-amber-500/50 bg-amber-500/10 text-amber-300 px-3 py-2">
          All job processing is paused — nothing will use the GPU or LLM
          until you click “Resume all”. This survives service restarts.
        </div>
      )}
      {activeTotal === 0 && !pauseAll && (
        <div className="text-forge-muted text-sm">
          No active jobs. Repair jobs started from the Manage tab appear
          here with live progress and pause/stop controls.
        </div>
      )}
      <div className="space-y-2">
        {shown.map((j) => (
          <JobRowCard key={j.job_id} job={j} />
        ))}
      </div>
      {hidden > 0 && (
        <div className="text-xs text-forge-muted mt-2">
          …and {hidden} more active job{hidden === 1 ? "" : "s"} not shown
          (queued jobs drain a few at a time).
        </div>
      )}
    </div>
  );
}

function JobsList() {
  const { data, dataUpdatedAt, isFetching } = useQuery({
    queryKey: ["jobs"],
    queryFn: () => listJobs("terminal", 30),
    refetchInterval: 3000,
  });
  const jobs = data?.data || [];
  const updatedSec = dataUpdatedAt ? Math.round((Date.now() - dataUpdatedAt) / 1000) : null;

  return (
    <div>
      <div className="flex items-center mb-3 gap-3 flex-wrap">
        <h2 className="font-semibold" title="Jobs that finished — completed, failed, or stopped. Running jobs are in Active Jobs above.">Finished Jobs</h2>
        <span className="text-xs text-forge-muted">
          polling every 3s
          {updatedSec !== null ? ` · updated ${updatedSec}s ago` : ""}
        </span>
        {isFetching && (
          <span
            className="h-2 w-2 rounded-full bg-forge-primary animate-pulse"
            title="refetching"
          />
        )}
        <span className="text-[10px] text-forge-muted ml-auto flex items-center gap-2">
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-emerald-500 inline-block" /> <span title="Step finished successfully">done</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-amber-500 inline-block" /> <span title="Step finished but some pages inside it failed — open the logs for details">partial</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-rose-500 inline-block" /> <span title="Step failed — the reason is shown under the circles and in the logs">failed</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full border border-amber-500/70 inline-block" /> <span title="Step deliberately not run (e.g. LLM offline, manual tags provided) — hover the circle for the reason">skipped</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full border border-forge-muted/50 inline-block" /> <span title="Step has not started yet">not run</span>
          </span>
        </span>
      </div>
      {jobs.length === 0 && (
        <div className="text-forge-muted text-sm">No finished jobs yet.</div>
      )}
      <div className="space-y-2">
        {jobs.map((j) => (
          <JobRowCard key={j.job_id} job={j} />
        ))}
      </div>
    </div>
  );
}

// Human labels for pipeline step names (ledger + current_step values).
const STEP_LABELS: Record<string, string> = {
  registering: "register",
  rendering_pages: "render pages",
  extracting_text: "extract text",
  auto_tagging: "auto-tag",
  embedding_text: "text embed",
  building_chunks: "chunks",
  embedding_visual: "visual embed",
  extracting_entities: "entities",
  dedup_entities: "dedup",
  chunking: "chunking",
  summarizing: "summaries",
  embedding_chunks: "chunk embed",
  writing_chunks: "write chunks",
  building_graph: "communities",
};

const stepLabel = (name: string) => STEP_LABELS[name] || name.replace(/_/g, " ");

// What each job type actually does — shown on the job-type badge.
const JOB_TYPE_TIPS: Record<string, string> = {
  "ingest": "Full pipeline on an uploaded PDF: render pages, extract text, auto-tag, embed (text + visual), build chunks, extract entities, dedup",
  "re-embed": "Clears and regenerates BOTH text and visual embeddings for a document (used after switching embedding models)",
  "text-reembed": "Clears and regenerates only the text embeddings (visual untouched)",
  "extract-entities": "LLM entity extraction on pages that haven't been extracted yet — skips finished pages",
  "build-communities": "Rebuilds the GraphRAG topic communities across the whole library (Leiden clustering + LLM summaries)",
  "rebuild-chunks": "Re-runs Docling chunking + per-chunk summaries + embeddings for a document",
  "fill-missing": "Repair job from the completeness audit: processes ONLY missing artifacts (recovered text, embeddings, entities) — never redoes finished work",
  "resummarize": "Regenerates chunk summaries that fell back to raw text previews when the LLM failed — re-embeds each repaired chunk",
  "autotag": "Auto-tags every unorganized document (no collection/categories/tags) via the LLM",
  "blank-flags": "Backfills the is_blank flag on old pages so blank pages are skipped by visual embedding",
};

// Job types the backend's restart endpoint can re-launch (kept in sync
// with RESTARTABLE_JOB_TYPES in backend/routers/ingestion.py).
const RESTARTABLE = new Set([
  "ingest", "fill-missing", "extract-entities", "rebuild-chunks",
  "re-embed", "text-reembed", "resummarize", "autotag", "build-communities",
]);

const STEP_CIRCLE: Record<StepStatus, string> = {
  done: "bg-emerald-500",
  warning: "bg-amber-500",
  error: "bg-rose-500",
  running: "bg-sky-400 animate-pulse",
  skipped: "border border-amber-500/70 bg-transparent",
  pending: "border border-forge-muted/50 bg-transparent",
};

const STEP_STATUS_TEXT: Record<StepStatus, string> = {
  done: "completed",
  warning: "completed with errors",
  error: "failed",
  running: "running",
  skipped: "skipped",
  pending: "not run",
};

function StepCircles({ steps }: { steps: JobStepRecord[] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 mt-2">
      {steps.map((s) => (
        <span
          key={s.name}
          className="flex items-center gap-1.5"
          title={`${stepLabel(s.name)}: ${STEP_STATUS_TEXT[s.status] || s.status}${
            s.detail ? ` — ${s.detail}` : ""
          }`}
        >
          <span
            className={`h-2.5 w-2.5 rounded-full shrink-0 ${
              STEP_CIRCLE[s.status] || STEP_CIRCLE.pending
            }`}
          />
          <span
            className={`text-[10px] ${
              s.status === "error"
                ? "text-rose-400"
                : s.status === "warning" || s.status === "skipped"
                ? "text-amber-400"
                : s.status === "running"
                ? "text-sky-300"
                : "text-forge-muted"
            }`}
          >
            {stepLabel(s.name)}
          </span>
        </span>
      ))}
    </div>
  );
}

// Steps whose outcome deserves an explanation line under the circles —
// anything that didn't simply succeed.
function StepIssues({ steps }: { steps: JobStepRecord[] }) {
  const issues = steps.filter(
    (s) => (s.status === "error" || s.status === "warning" || s.status === "skipped") && s.detail
  );
  if (issues.length === 0) return null;
  return (
    <div className="mt-1.5 space-y-0.5">
      {issues.map((s) => (
        <div
          key={s.name}
          className={`text-[11px] font-mono ${
            s.status === "error" ? "text-rose-400" : "text-amber-400/90"
          }`}
        >
          {stepLabel(s.name)} {s.status === "skipped" ? "skipped" : s.status === "warning" ? "partial" : "failed"}: {s.detail}
        </div>
      ))}
    </div>
  );
}

function JobLogs({ jobId, isActive }: { jobId: string; isActive: boolean }) {
  const { data, isLoading } = useQuery({
    queryKey: ["job-logs", jobId],
    queryFn: () => getJobLogs(jobId),
    refetchInterval: isActive ? 3000 : false,
  });
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const pinnedToBottom = useRef(true);
  const lines = data?.data?.lines || [];

  // Follow the tail while the user hasn't scrolled up.
  useEffect(() => {
    const el = scrollRef.current;
    if (el && pinnedToBottom.current) el.scrollTop = el.scrollHeight;
  }, [lines.length]);

  const levelColor = (level: string) =>
    level === "ERROR" || level === "CRITICAL"
      ? "text-rose-400"
      : level === "WARNING"
      ? "text-amber-400"
      : "text-forge-muted";

  return (
    <div className="mt-2 border border-forge-edge rounded bg-forge-bg">
      {isLoading ? (
        <div className="text-xs text-forge-muted p-2">loading logs…</div>
      ) : lines.length === 0 ? (
        <div className="text-xs text-forge-muted p-2">
          No logs recorded for this job
          {" "}(jobs run before log capture was added have no stored logs).
        </div>
      ) : (
        <div
          ref={scrollRef}
          onScroll={(e) => {
            const el = e.currentTarget;
            pinnedToBottom.current =
              el.scrollHeight - el.scrollTop - el.clientHeight < 40;
          }}
          className="max-h-72 overflow-y-auto p-2 font-mono text-[11px] leading-4"
        >
          {lines.map((ln, i) => (
            <div key={i} className="whitespace-pre-wrap break-all">
              <span className="text-forge-muted/70">
                {new Date(ln.ts).toLocaleTimeString()}{" "}
              </span>
              <span className={levelColor(ln.level)}>{ln.level} </span>
              <span className="text-forge-muted/70">{ln.logger}: </span>
              <span className={ln.level === "ERROR" ? "text-rose-300" : ""}>
                {ln.message}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function JobControlButtons({ job }: { job: JobRow }) {
  const qc = useQueryClient();
  const [confirmStop, setConfirmStop] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["jobs-active"] });
    qc.invalidateQueries({ queryKey: ["jobs"] });
    qc.invalidateQueries({ queryKey: ["job-controls"] });
  };
  const run = useMutation({
    mutationFn: (fn: () => Promise<{ success: boolean; reason?: string | null }>) => fn(),
    onSuccess: (res) => setErr(res.success ? null : res.reason || "request failed"),
    onSettled: refresh,
  });

  const isActive =
    job.status === "processing" || job.status === "queued" || job.status === "paused";
  const canRestart = !isActive && RESTARTABLE.has(job.job_type);
  const btn =
    "text-xs border border-forge-edge rounded px-2 py-0.5 hover:bg-forge-edge disabled:opacity-50";

  return (
    <>
      {isActive && job.status !== "paused" && (
        <button
          type="button"
          className={btn}
          disabled={run.isPending}
          onClick={() => run.mutate(() => pauseJob(job.job_id))}
          title="Hold this job after its current page/batch. Nothing is lost — Resume continues where it left off."
        >
          ⏸ pause
        </button>
      )}
      {job.status === "paused" && (
        <button
          type="button"
          className={`${btn} text-emerald-400 border-emerald-500/50`}
          disabled={run.isPending}
          onClick={() => run.mutate(() => resumeJob(job.job_id))}
          title="Continue this job where it left off (if Pause all is on, it stays held until Resume all)"
        >
          ▶ resume
        </button>
      )}
      {isActive && (
        <button
          type="button"
          className={`${btn} ${confirmStop ? "bg-rose-600 text-white border-rose-600" : "text-rose-400"}`}
          disabled={run.isPending}
          onClick={() => {
            if (!confirmStop) {
              setConfirmStop(true);
              setTimeout(() => setConfirmStop(false), 4000);
              return;
            }
            setConfirmStop(false);
            run.mutate(() => cancelJob(job.job_id));
          }}
          title="Stop this job. Finished work is kept; a later Restart re-checks what's missing and continues from there."
        >
          {confirmStop ? "confirm stop?" : "■ stop"}
        </button>
      )}
      {canRestart && (
        <button
          type="button"
          className={btn}
          disabled={run.isPending}
          onClick={() => run.mutate(() => restartJob(job.job_id))}
          title="Launch this job again as a new job. Repair jobs re-check what's missing, so this continues rather than redoes."
        >
          ↻ restart
        </button>
      )}
      {err && (
        <span className="text-[10px] text-rose-400" title={err}>
          {err.length > 60 ? err.slice(0, 60) + "…" : err}
        </span>
      )}
    </>
  );
}

function JobRowCard({ job }: { job: JobRow }) {
  const [showLogs, setShowLogs] = useState(false);
  const colorMap: Record<string, string> = {
    queued: "bg-forge-muted/60",
    processing: "bg-forge-secondary",
    paused: "bg-amber-400",
    completed: "bg-emerald-500",
    failed: "bg-forge-danger",
    cancelled: "bg-amber-500",
  };
  const color = colorMap[job.status] || "bg-forge-muted/60";
  const pct = Math.min(100, Math.max(0, job.progress_pct));

  // Job type: recorded on the row for new jobs; derived from the
  // source_path pattern for jobs that predate the job_type column.
  const jobType =
    job.job_type ||
    (job.source_path?.startsWith("(reembed")
      ? "re-embed"
      : job.source_path?.startsWith("(text-reembed")
      ? "text-reembed"
      : job.source_path?.startsWith("(extract")
      ? "extract-entities"
      : job.source_path?.startsWith("(build-communities")
      ? "build-communities"
      : job.source_path?.startsWith("(rebuild-chunks")
      ? "rebuild-chunks"
      : job.source_path?.startsWith("(fill-missing")
      ? "fill-missing"
      : "ingest");

  const typeColors: Record<string, string> = {
    "ingest": "text-forge-secondary",
    "re-embed": "text-forge-primary",
    "text-reembed": "text-forge-primary",
    "extract-entities": "text-forge-accent",
    "build-communities": "text-emerald-400",
    "rebuild-chunks": "text-forge-accent",
    "fill-missing": "text-emerald-400",
    "resummarize": "text-forge-accent",
    "autotag": "text-forge-secondary",
    "blank-flags": "text-forge-muted",
  };

  const isActive =
    job.status === "processing" || job.status === "queued" || job.status === "paused";

  return (
    <div className="bg-forge-panel border border-forge-edge rounded p-3">
      <div className="flex items-center gap-3 mb-1">
        <span className={`h-2 w-2 rounded-full ${color}`} />
        <span className={`text-xs font-semibold uppercase ${typeColors[jobType] || ""}`} title={JOB_TYPE_TIPS[jobType] || jobType}>
          {jobType}
        </span>
        <span className="font-semibold truncate flex-1">{job.filename}</span>
        <span className="font-mono text-xs text-forge-muted" title="Job status and the pipeline step it is currently on">
          {job.status} · {job.current_step}
        </span>
        <span className="font-mono text-xs text-forge-muted" title="Units processed / total for the current phase (pages, or chunks during chunk building)">
          {job.pages_processed}
          {job.pages_total ? ` / ${job.pages_total}` : ""}
        </span>
        <JobControlButtons job={job} />
        <button
          type="button"
          onClick={() => setShowLogs((v) => !v)}
          className={`text-xs border border-forge-edge rounded px-2 py-0.5 hover:bg-forge-edge ${
            showLogs ? "bg-forge-edge" : ""
          }`}
          title="show captured log lines for this job"
        >
          logs {showLogs ? "▾" : "▸"}
        </button>
      </div>
      <div className="h-1.5 bg-forge-bg rounded overflow-hidden">
        <div
          className="h-full bg-forge-accent transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
      {isActive && job.current_item && (
        <div
          className="mt-1.5 text-[11px] font-mono text-sky-300 truncate"
          title="What this job is working on right now"
        >
          ▸ {job.current_item}
        </div>
      )}
      {job.status === "paused" && (
        <div className="mt-1 text-[11px] text-amber-400">
          paused — finished its current page/batch and is holding; Resume
          continues from here
        </div>
      )}
      {job.steps && job.steps.length > 0 && (
        <>
          <StepCircles steps={job.steps} />
          <StepIssues steps={job.steps} />
        </>
      )}
      {job.error_message && (
        <div className="text-xs text-rose-400 mt-2 font-mono">
          ERR: {job.error_message}
        </div>
      )}
      {showLogs && <JobLogs jobId={job.job_id} isActive={isActive} />}
    </div>
  );
}
