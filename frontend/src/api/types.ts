// Response envelope that matches the backend's ForgeResult model.
export interface ForgeResult<T = unknown> {
  success: boolean;
  reason?: string | null;
  data?: T;
}

// --- Document / Page ---
export type SourceType = "digital_native" | "scanned" | "hybrid" | "unknown";

export interface DocumentRow {
  doc_id: string;
  title: string;
  filename: string;
  file_hash: string;
  page_count: number;
  file_size_bytes: number;
  source_type: SourceType;
  collection: string;
  ingested_at: string;
  categories: string[];
  tags: string[];
}

export interface PageMeta {
  page_id: string;
  page_number: number;
  image_path: string;
  reduced_image_path: string | null;
  text_char_count: number;
  source_type: SourceType;
}

export interface PageDetail extends PageMeta {
  extracted_text: string;
}

// --- Ingest jobs ---
export type JobStatus = "queued" | "processing" | "completed" | "failed" | "cancelled";
export interface JobRow {
  job_id: string;
  status: JobStatus;
  current_step: string;
  progress_pct: number;
  created_at: string;
  updated_at: string;
  error_message: string | null;
  source_path: string;
  filename: string;
  pages_processed: number;
  pages_total: number;
  requested_categories: string[];
  requested_tags: string[];
  doc_id: string | null;
  file_hash: string | null;
}

// --- Search ---
export interface SearchHit {
  page_id: string;
  doc_id: string;
  document_title: string;
  filename: string;
  page_number: number;
  score: number;
  coarse_score?: number;
  base_score?: number;
  entity_hits?: number;
  match_count?: number;
  matched_entities?: string[];
  vector_similarity?: number;
  text_snippet: string | null;
  image_url: string;
  reduced_image_url: string;
  categories?: string[];
  tags?: string[];
  entities?: Array<{ kind: string | null; name: string | null }>;
  communities?: Array<{ level: number | null; community_id: string | null; summary: string | null }>;
}

export interface CommunityHit {
  community_id: string;
  level: number;
  summary: string;
  member_count: number;
  score: number;
  sample_pages: Array<{ doc_id: string; title: string; page_number: number; file_hash: string }>;
}

export type HybridStrategy = "graph_boosted" | "vector_first" | "graph_first" | "community";

// --- Graph ---
export interface GraphStats {
  documents: number;
  pages: number;
  materials: number;
  processes: number;
  standards: number;
  clauses: number;
  equipment: number;
  categories: number;
  tags: number;
  communities: number;
}

export interface EntityRow {
  key: string;
  properties: Record<string, unknown>;
  page_mentions: number;
}

export interface CategoryRow {
  name: string;
  description: string | null;
  parent_name: string | null;
  document_count: number;
}

export interface TagRow {
  name: string;
  document_count: number;
}

// --- System / health ---
export interface HealthPayload {
  status: string;
  service: string;
  version: string;
  neo4j_connected: boolean;
  document_count: number;
  page_count: number;
  gpu_available: boolean;
  config_loaded: boolean;
  details: {
    gpu_name?: string;
    vram_total_gb?: number;
    vram_free_gb?: number;
    models?: Array<{
      name: string;
      loaded: boolean;
      last_used_s_ago: number;
      est_vram_bytes: number;
    }>;
  };
}
