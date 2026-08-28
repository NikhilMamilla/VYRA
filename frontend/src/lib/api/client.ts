/**
 * The single place that knows how to talk to the VYRA API.
 *
 * Components never build URLs or read `import.meta.env`; they call the typed
 * functions in `endpoints.ts`, which go through `request` here.
 */

import { env } from '../../config/env';
import type { ApiErrorBody } from './types';

/** A failed API call, carrying the backend's error code for the UI to branch on. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details?: Record<string, unknown>;
  readonly requestId?: string;

  constructor(
    status: number,
    code: string,
    message: string,
    details?: Record<string, unknown>,
    requestId?: string,
  ) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.details = details;
    this.requestId = requestId;
  }

  /** True when the endpoint exists but the capability is not built yet. */
  get isNotImplemented(): boolean {
    return this.status === 501;
  }
}

/** Raised when the backend could not be reached at all. */
export class NetworkError extends Error {
  constructor(cause: unknown) {
    super('Could not reach the VYRA API.');
    this.name = 'NetworkError';
    this.cause = cause;
  }
}

export interface RequestOptions {
  method?: string;
  body?: BodyInit;
  signal?: AbortSignal;
  query?: Record<string, string | number | undefined>;
}

function buildUrl(path: string, query?: RequestOptions['query']): string {
  const url = `${env.apiBaseUrl}${path}`;
  if (!query) return url;

  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined) params.set(key, String(value));
  }
  const queryString = params.toString();
  return queryString ? `${url}?${queryString}` : url;
}

async function parseError(response: Response): Promise<ApiError> {
  let body: Partial<ApiErrorBody> = {};
  try {
    body = (await response.json()) as Partial<ApiErrorBody>;
  } catch {
    // A non-JSON error (a proxy timeout, say) still has to surface cleanly.
  }
  return new ApiError(
    response.status,
    body.error?.code ?? 'http_error',
    body.error?.message ?? response.statusText ?? 'Request failed.',
    body.error?.details,
    body.request_id ?? response.headers.get('X-Request-ID') ?? undefined,
  );
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(buildUrl(path, options.query), {
      method: options.method ?? 'GET',
      body: options.body,
      signal: options.signal,
      headers: { Accept: 'application/json' },
    });
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause;
    throw new NetworkError(cause);
  }

  if (!response.ok) throw await parseError(response);
  if (response.status === 204) return undefined as T;

  return (await response.json()) as T;
}
