import { useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  checkDuplicates,
  listCategories,
  listCollections,
  listTags,
  listJobs,
  sha256File,
  uploadPdf,
} from "../api/client";
import type { DuplicateInfo } from "../api/client";
import type { JobRow } from "../api/types";

const fileKey = (f: File) => `${f.name}|${f.size}`;

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

  const folderInputRef = useRef<HTMLInputElement | null>(null);

  const collections = collectionsResp?.data || [];

  const dedupByName = (current: File[], incoming: File[]): File[] => {
    const seen = new Set(current.map((f) => `${f.name}|${f.size}`));
    const merged = [...current];
    for (const f of incoming) {
      const key = `${f.name}|${f.size}`;
      if (!seen.has(key)) {
        seen.add(key);
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
            <label className="text-xs border border-forge-edge rounded px-3 py-1.5 cursor-pointer hover:bg-forge-edge">
              Add files…
              <input
                type="file"
                accept="application/pdf"
                multiple
                onChange={(e) => {
                  addFiles(e.target.files);
                  e.target.value = "";
                }}
                className="hidden"
              />
            </label>
            <label className="text-xs border border-forge-edge rounded px-3 py-1.5 cursor-pointer hover:bg-forge-edge">
              Add folder…
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
            </label>
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
                  key={`${f.name}-${f.size}-${i}`}
                  className="flex items-center gap-2 px-2 py-1 text-xs border-b border-forge-edge last:border-b-0"
                >
                  <span className="flex-1 truncate font-mono">{f.name}</span>
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
          <label className="block text-xs text-forge-muted mb-1">Collection</label>
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
          <label className="block text-xs text-forge-muted mb-1">
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
          <label className="block text-xs text-forge-muted mb-1">
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

function JobsList() {
  const { data, dataUpdatedAt, isFetching } = useQuery({
    queryKey: ["jobs"],
    queryFn: () => listJobs(undefined, 30),
    refetchInterval: 3000,
  });
  const jobs = data?.data || [];
  const updatedSec = dataUpdatedAt ? Math.round((Date.now() - dataUpdatedAt) / 1000) : null;

  return (
    <div>
      <div className="flex items-center mb-3 gap-3">
        <h2 className="font-semibold">Recent Jobs</h2>
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
      </div>
      {jobs.length === 0 && (
        <div className="text-forge-muted text-sm">No jobs yet.</div>
      )}
      <div className="space-y-2">
        {jobs.map((j) => (
          <JobRowCard key={j.job_id} job={j} />
        ))}
      </div>
    </div>
  );
}

function JobRowCard({ job }: { job: JobRow }) {
  const colorMap: Record<string, string> = {
    queued: "bg-forge-muted/60",
    processing: "bg-forge-secondary",
    completed: "bg-emerald-500",
    failed: "bg-forge-danger",
    cancelled: "bg-amber-500",
  };
  const color = colorMap[job.status] || "bg-forge-muted/60";
  const pct = Math.min(100, Math.max(0, job.progress_pct));

  // Derive job type from source_path pattern
  const jobType = job.source_path?.startsWith("(reembed")
    ? "re-embed"
    : job.source_path?.startsWith("(extract")
    ? "extract-entities"
    : job.source_path?.startsWith("(build-communities")
    ? "build-communities"
    : "ingest";

  const typeColors: Record<string, string> = {
    "ingest": "text-forge-secondary",
    "re-embed": "text-forge-primary",
    "extract-entities": "text-forge-accent",
    "build-communities": "text-emerald-400",
  };

  return (
    <div className="bg-forge-panel border border-forge-edge rounded p-3">
      <div className="flex items-center gap-3 mb-1">
        <span className={`h-2 w-2 rounded-full ${color}`} />
        <span className={`text-xs font-semibold uppercase ${typeColors[jobType] || ""}`}>
          {jobType}
        </span>
        <span className="font-semibold truncate flex-1">{job.filename}</span>
        <span className="font-mono text-xs text-forge-muted">
          {job.status} · {job.current_step}
        </span>
        <span className="font-mono text-xs text-forge-muted">
          {job.pages_processed}
          {job.pages_total ? ` / ${job.pages_total}` : ""}
        </span>
      </div>
      <div className="h-1.5 bg-forge-bg rounded overflow-hidden">
        <div
          className="h-full bg-forge-accent transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
      {job.error_message && (
        <div className="text-xs text-rose-400 mt-2 font-mono">
          ERR: {job.error_message}
        </div>
      )}
    </div>
  );
}
