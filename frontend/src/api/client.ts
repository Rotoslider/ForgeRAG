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

async function request<T>(
  path: string,
  opts: RequestInit = {}
): Promise<ForgeResult<T>> {
  const res = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(opts.headers || {}),
    },
    ...opts,
  });
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
  return (await res.json()) as ForgeResult<T>;
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
