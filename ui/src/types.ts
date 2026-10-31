/** Shapes returned by the R3 `/ask` API and used by the R4 surfaces. */

export type Rect = [number, number, number, number];

export interface Citation {
  chunk_id: number;
  doc_id: string;
  page_no: number;
  char_start: number;
  char_end: number;
  page_text_sha256: string;
  quote: string;
  /** Present on committed fixtures; the R3 answer path resolves by quote first. */
  rects?: Rect[];
}

export interface Claim {
  text: string;
  citations: Citation[];
}

export interface Passage {
  chunk_id: number;
  doc_id: string;
  page_no: number;
  breadcrumb: string;
  text: string;
  score: number | null;
}

export interface ValidationReport {
  claims_returned: number;
  claims_kept: number;
  dropped_uncited: number;
  dropped_unknown_citation: number;
  dropped_quote_mismatch: number;
  dropped_bad_quote: number;
  schema_failures: number;
  repair_turns: number;
  claims_dropped: number;
}

export interface CalibrationInfo {
  fit_set_sha256: string | null;
  a: number | null;
  b: number | null;
  threshold: number | null;
  uncalibrated: boolean;
  provenance: string;
}

export interface TracePreview {
  chunk_id: number;
  doc_id: string;
  page_no: number;
  breadcrumb: string;
  preview: string;
}

export interface VectorHit extends TracePreview {
  rank: number;
  cosine_distance: number;
}

export interface LexicalHit extends TracePreview {
  rank: number;
  ts_rank_cd: number;
}

export interface FusionHit extends TracePreview {
  rrf_rank: number;
  rrf_score: number;
  vector_rank: number | null;
  lexical_rank: number | null;
  rerank_rank: number | null;
  rank_delta: number | null;
}

export interface RerankHit extends TracePreview {
  rerank_rank: number;
  rrf_rank: number;
  rrf_score: number;
  vector_rank: number | null;
  lexical_rank: number | null;
  rerank_score: number;
  kept: boolean;
  calibrated_probability: number | null;
}

export interface RefusalTrace {
  top_rerank_score: number | null;
  top_probability: number | null;
  threshold: number | null;
  decision: 'answered' | 'refused';
  reason: string | null;
  uncalibrated: boolean;
}

export interface RetrievalTrace {
  mode: 'vector' | 'hybrid' | 'hybrid_rerank';
  embedder: string;
  reranker: string;
  vector_hits: VectorHit[];
  lexical_hits: LexicalHit[];
  fusion_hits: FusionHit[];
  rerank_hits: RerankHit[];
  context_chunk_ids: number[];
  context_tokens: number;
  refusal: RefusalTrace;
}

export interface AskResponse {
  claims: Claim[];
  insufficient_evidence: boolean;
  reason: string | null;
  passages: Passage[];
  /** Null for a recorded demo answer whose cache predates the current model. */
  model: string | null;
  structured_output: 'tool' | 'json' | 'extractive';
  validation: ValidationReport;
  refusal_probability: number | null;
  calibration: CalibrationInfo;
  uncalibrated: boolean;
  model_reason: string | null;
  trace: RetrievalTrace | null;
  /** Client-measured misses are authoritative; the server slot is always 0. */
  citation_render_miss: number;
}

/** A visitor upload, as `GET /documents/{id}` reports it. */
export type UploadStatus =
  | 'queued'
  | 'parsing'
  | 'embedding'
  | 'ready'
  | 'failed';

export interface UploadDocument {
  id: string;
  status: UploadStatus;
  collection: string;
  filename: string;
  byte_size: number;
  pages_done: number;
  pages_total: number | null;
  error_code: string | null;
  error_detail: string | null;
  created_at: string;
  expires_at: string;
}

/** One row of `ui/src/fixtures/citations.json`. */
export interface FixtureCitation extends Citation {
  id: string;
  claim: string;
  rects: Rect[];
}

export interface FixturePage {
  page_no: number;
  /** Text rebuilt exactly as PDF.js `getTextContent({disableNormalization:true})`. */
  text: string;
}

export interface CitationFixture {
  doc_id: string;
  pdf: string;
  generated_with: string;
  pages: FixturePage[];
  citations: FixtureCitation[];
}
