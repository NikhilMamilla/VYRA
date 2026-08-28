/** Typed wrappers for every VYRA API route. */

import { request } from './client';
import type { Analysis, Health, Page } from './types';

const V1 = '/api/v1';

export function getHealth(signal?: AbortSignal): Promise<Health> {
  return request<Health>('/health', { signal });
}

export function listAnalyses(
  params: { limit?: number; offset?: number } = {},
  signal?: AbortSignal,
): Promise<Page<Analysis>> {
  return request<Page<Analysis>>(`${V1}/analyses`, { query: params, signal });
}

export function getAnalysis(id: string, signal?: AbortSignal): Promise<Analysis> {
  return request<Analysis>(`${V1}/analyses/${id}`, { signal });
}

/**
 * Uploads an image for analysis.
 *
 * The backend validates the file today but has no analysis model loaded, so
 * this currently rejects with an `ApiError` where `isNotImplemented` is true.
 */
export function createAnalysis(file: File, signal?: AbortSignal): Promise<Analysis> {
  const body = new FormData();
  body.append('file', file);
  return request<Analysis>(`${V1}/analyses`, { method: 'POST', body, signal });
}
