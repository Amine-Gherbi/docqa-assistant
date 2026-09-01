import type { AskResponse } from './types';

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
export async function askQuestion(question: string): Promise<AskResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
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
): Promise<AskResponse> {
  try {
    return await askQuestion(question);
  } catch (error) {
    if (!isColdStart(error)) {
      throw error;
    }
    onColdStart?.();
    await warmApi();
    return askQuestion(question);
  }
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
