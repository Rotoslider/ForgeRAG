import { useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  listCategories,
  listCollections,
  listTags,
  listJobs,
  uploadPdf,
} from "../api/client";
import type { JobRow } from "../api/types";

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
    mutationFn: async () => {
      if (files.length === 0) throw new Error("Select at least one PDF first");
      const col = newCollection.trim() || collection;
      const errors: Array<{ name: string; reason: string }> = [];
      setUploadErrors([]);
      setUploadProgress({ done: 0, total: files.length });
      for (let i = 0; i < files.length; i++) {
        const f = files[i];
        try {
          const res = await uploadPdf(f, col, selectedCats, selectedTags);
          if (!res.success) {
            errors.push({ name: f.name, reason: res.reason || "upload failed" });
          }
        } catch (e) {
          errors.push({ name: f.name, reason: (e as Error).message });
        }
        setUploadProgress({ done: i + 1, total: files.length });
        // Refresh jobs list so queued items appear as they go
        qc.invalidateQueries({ queryKey: ["jobs"] });
      }
      setUploadErrors(errors);
      return { queued: files.length - errors.length, failed: errors.length };
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

      <div className="flex items-center gap-3 flex-wrap">
        <button
          onClick={() => upload.mutate()}
          disabled={files.length === 0 || upload.isPending}
          className="bg-forge-accent text-black font-semibold rounded px-4 py-2 hover:brightness-110 disabled:opacity-50"
        >
          {upload.isPending
            ? uploadProgress
              ? `Uploading ${uploadProgress.done}/${uploadProgress.total}…`
              : "Uploading…"
            : files.length > 1
            ? `Start Ingestion (${files.length} files)`
            : "Start Ingestion"}
        </button>
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
