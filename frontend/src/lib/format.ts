/** Presentational helpers shared across the analysis UI. */

import type { QualityLabel, Severity } from './api/types';

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatDateTime(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

export function relativeTime(iso: string): string {
  const d = new Date(iso).getTime();
  if (Number.isNaN(d)) return iso;
  const s = Math.round((Date.now() - d) / 1000);
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return new Date(iso).toLocaleDateString();
}

export const QUALITY_TONE: Record<QualityLabel, 'good' | 'ok' | 'degraded' | 'poor'> = {
  GOOD: 'good',
  ACCEPTABLE: 'ok',
  DEGRADED: 'degraded',
  POOR: 'poor',
};

/** CSS colour var for a 0–100 score. */
export function scoreVar(score: number): string {
  if (score >= 85) return 'rgb(var(--c-good))';
  if (score >= 68) return 'rgb(var(--c-ok))';
  if (score >= 45) return 'rgb(var(--c-degraded))';
  return 'rgb(var(--c-poor))';
}

export const SEVERITY_LABEL: Record<Severity, string> = {
  low: 'low impact',
  medium: 'medium impact',
  high: 'high impact',
};

export function severityTone(s: Severity): 'neutral' | 'ok' | 'poor' {
  return s === 'high' ? 'poor' : s === 'medium' ? 'ok' : 'neutral';
}

export const ISSUE_LABELS: Record<string, string> = {
  blur: 'Blur',
  underexposure: 'Underexposure',
  overexposure: 'Overexposure',
  noise: 'Noise',
  corruption: 'Compression / corruption',
  defect: 'Potential visual defect',
};

export const VALIDATION_TONE: Record<string, 'real' | 'synth' | 'screen'> = {
  'real-world': 'real',
  'synthetic-only': 'synth',
  screening: 'screen',
};

export const VALIDATION_LABEL: Record<string, string> = {
  'real-world': 'real-world validated',
  'synthetic-only': 'synthetic only',
  screening: 'screening',
};

export const VALIDATION_NOTE: Record<string, string> = {
  'real-world': 'Evaluated on real-world images (VizWiz-QualityIssues).',
  'synthetic-only': 'Validated on synthetic degradations only — no real-world evaluation exists.',
  screening: 'Screening signal only — not a confirmed defect.',
};

export const STATISTIC_LABELS: Record<string, string> = {
  sharpness: 'Sharpness',
  brightness: 'Brightness',
  contrast: 'Contrast',
  noise_sigma: 'Noise (σ)',
  saturation: 'Saturation',
  colourfulness: 'Colourfulness',
  blockiness: 'Blockiness',
  edge_density: 'Edge density',
  dark_clip_ratio: 'Crushed shadows',
  bright_clip_ratio: 'Blown highlights',
};

export function humaniseFeature(name: string): string {
  return name.replace(/_/g, ' ');
}
