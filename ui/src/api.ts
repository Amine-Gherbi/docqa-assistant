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
