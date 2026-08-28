/**
 * Typed access to the build-time environment.
 *
 * Vite inlines `import.meta.env` at build time, so the API base URL is baked
 * into the bundle. In Docker the frontend is served by nginx, which proxies
 * `/api` to the backend, so the default empty base (same origin) is correct
 * there and only local development needs an override.
 */

const rawBaseUrl = import.meta.env.VITE_API_BASE_URL ?? '';

export const env = {
  /** API origin without a trailing slash; empty means "same origin". */
  apiBaseUrl: rawBaseUrl.replace(/\/+$/, ''),
  isDev: import.meta.env.DEV,
} as const;
