/**
 * Mirrors the backend response schemas (`backend/app/schemas`).
 *
 * Kept hand-written for now; if the contract grows, generate these from the
 * OpenAPI document the backend already publishes at `/openapi.json`.
 */

export type IssueType =
  | 'blur'
  | 'underexposure'
  | 'overexposure'
  | 'noise'
  | 'corruption'
  | 'defect';

export type Severity = 'low' | 'medium' | 'high';

/** How far an issue's detector has actually been validated. */
export type IssueValidation = 'real-world' | 'synthetic-only' | 'screening';

export type QualityLabel = 'GOOD' | 'ACCEPTABLE' | 'DEGRADED' | 'POOR';

export type AnalysisStatus = 'completed' | 'failed';

export interface Issue {
  type: IssueType;
  severity: Severity;
  confidence: number;
  validation: IssueValidation | null;
  detail: string | null;
}

export interface EvidenceItem {
  feature: string;
  value: number;
  direction: string;
}

export interface PotentialDefect {
  probability: number;
  flagged: boolean;
  region: [number, number, number, number] | null;
  evidence: { feature: string; z: number }[];
  note: string;
}

export interface Explanation {
  summary?: string;
  evidence?: EvidenceItem[];
  issue_probabilities?: Record<string, number>;
  potential_defect?: PotentialDefect;
  capabilities?: {
    real_world_validated?: string[];
    synthetic_validated_only?: string[];
    screening_only?: string[];
  };
  feature_version?: string;
  timings_ms?: Record<string, number>;
}

export interface ImageInfo {
  filename: string;
  content_type: string;
  size_bytes: number;
  width: number | null;
  height: number | null;
}

export interface Analysis {
  id: string;
  created_at: string;
  status: AnalysisStatus;
  image: ImageInfo;
  quality_score: number | null;
  quality_label: QualityLabel | null;
  model_version: string | null;
  issues: Issue[];
  metrics: Record<string, number>;
  explanation: Explanation;
  error_message: string | null;
}

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export type ComponentStatus = 'ok' | 'unavailable' | 'not_configured';

export interface ComponentHealth {
  status: ComponentStatus;
  detail: string | null;
  latency_ms: number | null;
}

export interface Health {
  status: 'ok' | 'degraded';
  version: string;
  environment: string;
  uptime_seconds: number;
  analyzer_model_version: string | null;
  components: Record<string, ComponentHealth>;
}

/** The envelope every non-2xx response uses. */
export interface ApiErrorBody {
  error: { code: string; message: string; details?: Record<string, unknown> };
  request_id?: string;
}
