import type { AskResponse, UploadDocument } from './types';

export const API_BASE = (import.meta.env.VITE_API_BASE ?? '/api').replace(/\/$/, '');
export const EVAL_REPORT_URL = `${API_BASE}/eval-report.json`;

export class ApiError extends Error {
  readonly status: number | null;

  constructor(message: string, status: number | null = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

/** A refused upload, carrying the limit the server says was hit. */
export class UploadError extends ApiError {
  readonly code: string | null;
  readonly limit: number | null;
  readonly retryAfterS: number | null;

  constructor(
    message: string,
    status: number | null,
    details: {
      code?: string | null;
      limit?: number | null;
      retryAfterS?: number | null;
    } = {},
  ) {
    super(message, status);
    this.name = 'UploadError';
    this.code = details.code ?? null;
    this.limit = details.limit ?? null;
    this.retryAfterS = details.retryAfterS ?? null;
  }
}

export interface SessionInfo {
  client_id: string;
  new: boolean;
}

// The free Render instance spins down after ~15 minutes of inactivity and takes
// up to ~60 s to start. Render's edge answers those requests with a 502 before
// the app can log anything; 503/504 are the same class of "not up yet".
const COLD_START_STATUSES = new Set([502, 503, 504]);
const WARMUP_TIMEOUT_MS = 90_000;

function isColdStart(error: unknown): boolean {
  return (
    error instanceof ApiError &&
    error.status !== null &&
    COLD_START_STATUSES.has(error.status)
  );
}

/**
 * GET /health with a bounded wait. Also serves as the wake-up call for a
 * sleeping free instance: Render holds the request until the app is listening.
 */
export async function warmApi(timeoutMs = WARMUP_TIMEOUT_MS): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/health`, {
      signal: AbortSignal.timeout(timeoutMs),
    });
    return response.ok;
  } catch {
    return false;
  }
}

async function readError(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (
      typeof body === 'object' &&
      body !== null &&
      'detail' in body &&
      typeof body.detail === 'string'
    ) {
      return body.detail;
    }
  } catch {
    // Fall through to the status text.
  }
  return response.statusText || `HTTP ${response.status}`;
}

/** POST /ask; the server has already validated every returned claim. */
export async function askQuestion(
  question: string,
  collection?: string,
): Promise<AskResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/ask`, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(
        collection ? { question, collection } : { question },
      ),
    });
  } catch (error) {
    throw new ApiError(
      error instanceof Error ? error.message : 'network request failed',
    );
  }
  if (!response.ok) {
    throw new ApiError(await readError(response), response.status);
  }
  return (await response.json()) as AskResponse;
}

/**
 * POST /ask, retrying once after a warm-up when the free instance was asleep.
 *
 * A sleeping instance answers the first request with the edge's 502 and never
 * reaches the app. The caller shows "starting the free instance" while the
 * health probe wakes it; only a failed retry is reported as an error.
 */
export async function askQuestionWithWarmup(
  question: string,
  onColdStart?: () => void,
  collection?: string,
): Promise<AskResponse> {
  try {
    return await askQuestion(question, collection);
  } catch (error) {
    if (!isColdStart(error)) {
      throw error;
    }
    onColdStart?.();
    await warmApi();
    return askQuestion(question, collection);
  }
}

function megabytes(bytes: number | null): string {
  return bytes === null ? 'the limit' : `${Math.round(bytes / (1024 * 1024))} MB`;
}

function uploadMessage(
  status: number,
  code: string | null,
  detail: string,
  limit: number | null,
  retryAfterS: number | null,
): string {
  if (status === 413) {
    return `This PDF is larger than ${megabytes(limit)}. Choose a smaller file.`;
  }
  if (status === 415) {
    return 'That file is not a PDF. Choose a PDF file.';
  }
  if (status === 409) {
    return 'This browser already has an upload in progress. Wait for it to finish.';
  }
  if (status === 429) {
    const minutes = Math.max(1, Math.ceil((retryAfterS ?? 60) / 60));
    return `${detail} Try again in ${minutes} minute${minutes === 1 ? '' : 's'}.`;
  }
  if (code === 'page_limit') {
    return detail;
  }
  if (code === 'encrypted') {
    return 'That PDF is password-protected, so its text cannot be read.';
  }
  if (code === 'not_parseable') {
    return 'That PDF could not be read. It may be damaged or incomplete.';
  }
  return detail;
}

async function uploadFailure(response: Response): Promise<UploadError> {
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    // Fall through to the status text.
  }
  const fields =
    typeof body === 'object' && body !== null
      ? (body as Record<string, unknown>)
      : {};
  const code = typeof fields.code === 'string' ? fields.code : null;
  const limit = typeof fields.limit === 'number' ? fields.limit : null;
  const retryAfterS =
    typeof fields.retry_after_s === 'number' ? fields.retry_after_s : null;
  const detail =
    typeof fields.detail === 'string'
      ? fields.detail
      : response.statusText || `HTTP ${response.status}`;
  return new UploadError(
    uploadMessage(response.status, code, detail, limit, retryAfterS),
    response.status,
    { code, limit, retryAfterS },
  );
}

/**
 * GET /session sets the signed HttpOnly cookie. It must run before the first
 * upload so the browser has a client id to send.
 */
export async function ensureSession(): Promise<SessionInfo> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/session`, { credentials: 'include' });
  } catch (error) {
    throw new ApiError(
      error instanceof Error ? error.message : 'network request failed',
    );
  }
  if (!response.ok) {
    throw new ApiError(await readError(response), response.status);
  }
  return (await response.json()) as SessionInfo;
}

/** POST /documents; 202 with the queued upload, or a limit/busy/quota error. */
export async function uploadDocument(file: File): Promise<UploadDocument> {
  const body = new FormData();
  body.append('file', file, file.name);
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/documents`, {
      method: 'POST',
      credentials: 'include',
      body,
    });
  } catch (error) {
    throw new ApiError(
      error instanceof Error ? error.message : 'network request failed',
    );
  }
  if (!response.ok) {
    throw await uploadFailure(response);
  }
  return (await response.json()) as UploadDocument;
}

/** GET /documents/{id} for the owning cookie's client only. */
export async function uploadStatus(id: string): Promise<UploadDocument> {
  let response: Response;
  try {
    response = await fetch(
      `${API_BASE}/documents/${encodeURIComponent(id)}`,
      { credentials: 'include' },
    );
  } catch (error) {
    throw new ApiError(
      error instanceof Error ? error.message : 'network request failed',
    );
  }
  if (!response.ok) {
    throw await uploadFailure(response);
  }
  return (await response.json()) as UploadDocument;
}

/** The original PDF, served to the owning cookie's client only. */
export function uploadedPdfUrl(id: string): string {
  return `${API_BASE}/documents/${encodeURIComponent(id)}/file`;
}

/** GET /eval/report; null means the R5 run has not produced a report yet. */
export async function loadEvalReport(): Promise<unknown | null> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/eval/report`);
  } catch (error) {
    throw new ApiError(
      error instanceof Error ? error.message : 'network request failed',
    );
  }
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new ApiError(await readError(response), response.status);
  }
  return (await response.json()) as unknown;
}
