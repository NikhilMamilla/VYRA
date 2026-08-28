/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Origin of the VYRA API, e.g. `http://localhost:8000`. Empty = same origin. */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
