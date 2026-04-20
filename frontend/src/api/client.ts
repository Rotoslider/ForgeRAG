import type {
  CategoryRow,
  CommunityHit,
  DocumentRow,
  EntityRow,
  ForgeResult,
  GraphStats,
  HealthPayload,
  HybridStrategy,
  JobRow,
  PageDetail,
  PageMeta,
  SearchHit,
  TagRow,
} from "./types";

// Default fetch timeout for API requests. 5 minutes generously covers
// even slow VLM answer roundtrips (visual retrieval + image reading +
// synthesis can take 60-120s on a complex engineering question, and
// LM Studio can be further delayed by model reload). Short requests
// (health, list, etc.) complete in well under a second so the long
// default is harmless for them.
const DEFAULT_TIMEOUT_MS = 5 * 60 * 1000;

async function request<T>(
  path: string,
  opts: RequestInit & { timeoutMs?: number; retries?: number } = {}
): Promise<ForgeResult<T>> {
  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const retries = opts.retries ?? 1;
  const { timeoutMs: _t, retries: _r, ...fetchOpts } = opts;

  let lastErr: unknown;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(path, {
        headers: {
          "Content-Type": "application/json",
          ...(fetchOpts.headers || {}),
        },
        signal: AbortSignal.timeout(timeoutMs),
        ...fetchOpts,
      });
      if (!res.ok) {
        let reason: string;
        try {
          const body = await res.json();
          reason = body.detail || body.reason || res.statusText;
        } catch {
          reason = res.statusText;
        }
        // 5xx might be transient — retry once. 4xx is a real client
        // error, fail fast.
        if (res.status >= 500 && attempt < retries) {
          lastErr = new Error(`HTTP ${res.status}: ${reason}`);
          await new Promise((r) => setTimeout(r, 750));
          continue;
        }
        return { success: false, reason };
      }
      return (await res.json()) as ForgeResult<T>;
    } catch (err) {
      lastErr = err;
      const msg = err instanceof Error ? err.message : String(err);
      // Only retry transient-looking failures. True timeouts (AbortSignal)
      // don't retry because the server might still be working on the
      // previous request — a retry would stack. The user can explicitly
      // resubmit.
      const looksTransient =
        msg.includes("NetworkError") ||
        msg.includes("Failed to fetch") ||
        msg.includes("ECONNRESET") ||
        msg.includes("ECONNREFUSED");
      if (looksTransient && attempt < retries) {
        await new Promise((r) => setTimeout(r, 500 + 500 * attempt));
        continue;
      }
      // Normalize to a friendly message.
      if (msg.includes("aborted") || msg.includes("Timeout")) {
        return {
          success: false,
          reason:
            `Request timed out after ${Math.round(timeoutMs / 1000)}s. ` +
            `The LLM may be loading a model or the query is unusually complex — ` +
            `try again, or reduce the page limit.`,
        };
      }
      return {
        success: false,
        reason: `Network error: ${msg}. Is the ForgeRAG backend running?`,
      };
    }
  }
  // Exhausted retries on 5xx
  return {
    success: false,
    reason:
      lastErr instanceof Error
        ? `Request failed after retries: ${lastErr.message}`
        : "Request failed after retries",
  };
}

// ---- Health ----
export const fetchHealth = () => request<HealthPayload>("/health");

// ---- Documents ----
export function listDocuments(params: {
  category?: string;
  tag?: string;
  sourceType?: string;
  limit?: number;
  offset?: number;
} = {}) {
  const q = new URLSearchParams();
  if (params.category) q.set("category", params.category);
  if (params.tag) q.set("tag", params.tag);
  if (params.sourceType) q.set("source_type", params.sourceType);
  if (params.limit) q.set("limit", String(params.limit));
  if (params.offset) q.set("offset", String(params.offset));
  return request<DocumentRow[]>(`/documents${q.toString() ? "?" + q : ""}`);
}

export const getDocument = (id: string) => request<DocumentRow>(`/documents/${id}`);
export const deleteDocument = (id: string) =>
  request<{ doc_id: string; folders_removed: string[] }>(`/documents/${id}`, { method: "DELETE" });
export const listPages = (id: string, limit = 100, offset = 0) =>
  request<PageMeta[]>(`/documents/${id}/pages?limit=${limit}&offset=${offset}`);
export const getPage = (id: string, n: number) =>
  request<PageDetail>(`/documents/${id}/pages/${n}`);
export const reembedDocument = (id: string) =>
  request<{ job_id: string; doc_id: string; status: string }>(
    `/documents/${id}/reembed`,
    { method: "POST" }
  );
export const extractEntities = (id: string) =>
  request<{ job_id: string; doc_id: string; status: string }>(
    `/documents/${id}/extract-entities`,
    { method: "POST" }
  );
export function rebuildChunks(
  id: string,
  opts: { extract_only?: boolean; skip_extract?: boolean } = {}
) {
  const q = new URLSearchParams();
  if (opts.extract_only) q.set("extract_only", "true");
  if (opts.skip_extract) q.set("skip_extract", "true");
  return request<{ job_id: string; doc_id: string; status: string }>(
    `/documents/${id}/rebuild-chunks${q.toString() ? "?" + q : ""}`,
    { method: "POST" }
  );
}
export function rebuildChunksBulk(body: {
  doc_ids: string[];
  extract_only?: boolean;
  skip_extract?: boolean;
  only_missing?: boolean;
}) {
  return request<{
    queued: number;
    skipped: number;
    not_found: number;
    jobs: Array<{ doc_id: string; job_id: string; title: string }>;
  }>(`/admin/rebuild-chunks-bulk`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

// ---- Categories & Tags ----
export const listCategories = () => request<CategoryRow[]>("/categories");
export const createCategory = (body: { name: string; description?: string; parent_name?: string }) =>
  request<{ name: string }>("/categories", { method: "POST", body: JSON.stringify(body) });
export const deleteCategory = (name: string) =>
  request<{ name: string }>(`/categories/${encodeURIComponent(name)}`, { method: "DELETE" });

export const listTags = () => request<TagRow[]>("/tags");
export const createTag = (name: string) =>
  request<{ name: string }>("/tags", { method: "POST", body: JSON.stringify({ name }) });
export const deleteTag = (name: string) =>
  request<{ name: string }>(`/tags/${encodeURIComponent(name)}`, { method: "DELETE" });

// ---- Ingestion ----
export const listCollections = () =>
  request<Array<{ collection: string; document_count: number; total_pages: number }>>("/collections");

export const moveDocument = (docId: string, collection: string) =>
  request<{ doc_id: string; collection: string }>(
    `/documents/${docId}/collection?collection=${encodeURIComponent(collection)}`,
    { method: "PUT" }
  );

export const addDocumentTag = (docId: string, tag: string) =>
  request<{ doc_id: string; tag: string }>(
    `/documents/${docId}/tags`,
    { method: "POST", body: JSON.stringify({ name: tag }) }
  );

export const removeDocumentTag = (docId: string, tag: string) =>
  request<{ doc_id: string; removed_tag: string }>(
    `/documents/${docId}/tags/${encodeURIComponent(tag)}`,
    { method: "DELETE" }
  );

export interface DuplicateInfo {
  file_hash: string;
  doc_id: string;
  title: string;
  filename: string;
  collection: string;
  page_count: number;
  ingested_at: string;
}

export async function sha256File(file: File): Promise<string> {
  const buf = await file.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

export const checkDuplicates = (hashes: string[]) =>
  request<{ duplicates: Record<string, DuplicateInfo> }>(
    "/ingest/check-duplicates",
    { method: "POST", body: JSON.stringify({ hashes }) }
  );

export async function uploadPdf(
  file: File,
  collection: string,
  categories: string[],
  tags: string[]
): Promise<ForgeResult<{ job_id: string; status: string; filename: string }>> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("collection", collection);
  fd.append("categories", categories.join(","));
  fd.append("tags", tags.join(","));
  const res = await fetch("/ingest", { method: "POST", body: fd });
  if (!res.ok) {
    let reason: string;
    try {
      const body = await res.json();
      reason = body.detail || body.reason || res.statusText;
    } catch {
      reason = res.statusText;
    }
    return { success: false, reason };
  }
  return await res.json();
}

export const getJob = (id: string) => request<JobRow>(`/ingest/jobs/${id}`);
export const listJobs = (status?: string, limit = 50) => {
  const q = new URLSearchParams({ limit: String(limit) });
  if (status) q.set("status", status);
  return request<JobRow[]>(`/ingest/jobs?${q}`);
};

// ---- Search ----
export const searchSemantic = (query: string, limit = 10) =>
  request<SearchHit[]>("/search/semantic", {
    method: "POST",
    body: JSON.stringify({ query, limit }),
  });

export const searchKeyword = (query: string, limit = 20) =>
  request<SearchHit[]>("/search/keyword", {
    method: "POST",
    body: JSON.stringify({ query, limit }),
  });

export interface AnswerResult {
  answer: string;
  sources: Array<{ document_title: string; page_number: number; image_url: string; score: number }>;
  query: string;
  search_mode: string;
  used_vision?: boolean;
  used_graph?: boolean;
  graph_context?: {
    materials_found: number;
    processes_found: number;
    standards_found: number;
    reasoning_chains: string[];
    pages_from_graph: number;
  } | null;
}

export const searchAnswer = (query: string, limit = 5, search_mode = "semantic") =>
  request<AnswerResult>("/search/answer", {
    method: "POST",
    body: JSON.stringify({ query, limit, search_mode }),
  });

export const cleanupUploads = () =>
  request<{ deleted: number; freed_bytes: number; freed_mb: number }>(
    "/admin/cleanup-uploads",
    { method: "POST" }
  );

export const searchVisual = (query: string, limit = 5, candidate_pool = 30) =>
  request<SearchHit[]>("/search/visual", {
    method: "POST",
    body: JSON.stringify({ query, limit, candidate_pool }),
  });

export const searchHybrid = (body: {
  query: string;
  strategy: HybridStrategy;
  limit?: number;
  candidate_pool?: number;
  boost_weight?: number;
}) =>
  request<SearchHit[] | CommunityHit[]>("/search/hybrid", {
    method: "POST",
    body: JSON.stringify(body),
  });

// ---- Graph ----
export const graphStats = () => request<GraphStats>("/graph/stats");
export const listEntities = (type: string, limit = 100, offset = 0) =>
  request<EntityRow[]>(`/graph/entities/${type}?limit=${limit}&offset=${offset}`);
export const buildCommunities = () =>
  request<{ job_id: string; status: string }>("/graph/build-communities", { method: "POST" });
export const listCommunities = (level?: number, limit = 50) => {
  const q = new URLSearchParams({ limit: String(limit) });
  if (level !== undefined) q.set("level", String(level));
  return request<Array<{ community_id: string; level: number; summary: string; member_count: number; actual_page_count: number }>>(
    `/graph/communities?${q}`
  );
};

// ---- System ----
export const getGpu = () =>
  request<{ available: boolean; device_name: string | null; vram_total_bytes: number; vram_free_bytes: number; vram_used_bytes: number; models: Array<{ name: string; loaded: boolean; last_used_s_ago: number; est_vram_bytes: number }> }>(
    "/system/gpu"
  );

export const unloadModel = (name: string) =>
  request<{ name: string; unloaded: boolean }>(`/system/models/${name}/unload`, { method: "POST" });

// ---- Image URLs (direct, not JSON) ----
export const pageImageUrl = (hash: string, page: number) => `/images/${hash}/${page}`;
export const reducedImageUrl = (hash: string, page: number) => `/images/${hash}/${page}/reduced`;
export const highlightedImageUrl = (hash: string, page: number, query: string) =>
  `/images/${hash}/${page}/highlighted?query=${encodeURIComponent(query)}`;
