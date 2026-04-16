import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  listCategories,
  listTags,
  listJobs,
  uploadPdf,
} from "../api/client";
import type { JobRow } from "../api/types";

export default function Ingest() {
  return (
    <div className="p-6 max-w-6xl">
      <h1 className="text-2xl font-bold mb-1">Ingest</h1>
      <p className="text-sm text-slate-400 mb-6">
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

  const [file, setFile] = useState<File | null>(null);
  const [selectedCats, setSelectedCats] = useState<string[]>([]);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [newTag, setNewTag] = useState("");

  const upload = useMutation({
    mutationFn: async () => {
      if (!file) throw new Error("Select a PDF first");
      const res = await uploadPdf(file, selectedCats, selectedTags);
      if (!res.success) throw new Error(res.reason || "Upload failed");
      return res.data!;
    },
    onSuccess: () => {
      setFile(null);
      setSelectedCats([]);
      setSelectedTags([]);
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const categories = catsResp?.data || [];
  const tags = tagsResp?.data || [];

  return (
    <div className="bg-forge-panel border border-forge-edge rounded-lg p-5 mb-8">
      <h2 className="font-semibold mb-3">Upload PDF</h2>

      <div className="mb-4">
        <label className="block text-xs text-slate-400 mb-1">File</label>
        <input
          type="file"
          accept="application/pdf"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          className="block w-full text-sm"
        />
      </div>

      <div className="grid md:grid-cols-2 gap-4 mb-4">
        <div>
          <label className="block text-xs text-slate-400 mb-1">
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
          <label className="block text-xs text-slate-400 mb-1">
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
            <details className="text-xs text-slate-400">
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

      <div className="flex items-center gap-3">
        <button
          onClick={() => upload.mutate()}
          disabled={!file || upload.isPending}
          className="bg-forge-accent text-black font-semibold rounded px-4 py-2 hover:brightness-110 disabled:opacity-50"
        >
          {upload.isPending ? "Uploading…" : "Start Ingestion"}
        </button>
        {upload.isError && (
          <span className="text-rose-400 text-sm">
            {(upload.error as Error).message}
          </span>
        )}
        {upload.isSuccess && upload.data && (
          <span className="text-emerald-400 text-sm">
            Queued: {upload.data.filename} (job {upload.data.job_id.slice(0, 8)}…)
          </span>
        )}
      </div>
    </div>
  );
}

function JobsList() {
  const { data } = useQuery({
    queryKey: ["jobs"],
    queryFn: () => listJobs(undefined, 30),
    refetchInterval: 3000,
  });
  const jobs = data?.data || [];

  return (
    <div>
      <h2 className="font-semibold mb-3">Recent Jobs</h2>
      {jobs.length === 0 && (
        <div className="text-slate-500 text-sm">No jobs yet.</div>
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
    queued: "bg-slate-500",
    processing: "bg-blue-500",
    completed: "bg-emerald-500",
    failed: "bg-rose-500",
    cancelled: "bg-amber-500",
  };
  const color = colorMap[job.status] || "bg-slate-500";
  const pct = Math.min(100, Math.max(0, job.progress_pct));

  return (
    <div className="bg-forge-panel border border-forge-edge rounded p-3">
      <div className="flex items-center gap-3 mb-1">
        <span className={`h-2 w-2 rounded-full ${color}`} />
        <span className="font-semibold truncate flex-1">{job.filename}</span>
        <span className="font-mono text-xs text-slate-400">
          {job.status} · {job.current_step}
        </span>
        <span className="font-mono text-xs text-slate-500">
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
