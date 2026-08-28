/** Small presentational helpers shared across the analysis UI. */

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

export const QUALITY_LABEL_STYLES: Record<QualityLabel, string> = {
  GOOD: 'bg-emerald-100 text-emerald-800 ring-emerald-600/20',
  ACCEPTABLE: 'bg-lime-100 text-lime-800 ring-lime-600/20',
  DEGRADED: 'bg-amber-100 text-amber-900 ring-amber-600/20',
  POOR: 'bg-red-100 text-red-800 ring-red-600/20',
};

export function scoreColor(score: number): string {
  if (score >= 85) return 'text-emerald-600';
  if (score >= 68) return 'text-lime-600';
  if (score >= 45) return 'text-amber-600';
  return 'text-red-600';
}

export function scoreTrackColor(score: number): string {
  if (score >= 85) return 'stroke-emerald-500';
  if (score >= 68) return 'stroke-lime-500';
  if (score >= 45) return 'stroke-amber-500';
  return 'stroke-red-500';
}

export const SEVERITY_STYLES: Record<Severity, string> = {
  low: 'bg-slate-100 text-slate-700',
  medium: 'bg-amber-100 text-amber-800',
  high: 'bg-red-100 text-red-800',
};

export const ISSUE_LABELS: Record<string, string> = {
  blur: 'Blur',
  underexposure: 'Underexposure',
  overexposure: 'Overexposure',
  noise: 'Noise',
  corruption: 'Compression / corruption',
  defect: 'Potential visual defect',
};

export const VALIDATION_NOTE: Record<string, string> = {
  'real-world': 'Validated on real-world images (VizWiz).',
  'synthetic-only': 'Validated on synthetic degradations only — no real-world evaluation.',
  screening: 'Screening signal only — not a confirmed defect.',
};

export const STATISTIC_LABELS: Record<string, string> = {
  sharpness: 'Sharpness (Laplacian var.)',
  brightness: 'Brightness (mean luma)',
  contrast: 'Contrast (RMS)',
  noise_sigma: 'Noise estimate (σ)',
  saturation: 'Saturation',
  colourfulness: 'Colourfulness',
  blockiness: 'Blockiness',
  edge_density: 'Edge density',
  dark_clip_ratio: 'Crushed-shadow ratio',
  bright_clip_ratio: 'Blown-highlight ratio',
};

export function humaniseFeature(name: string): string {
  return name.replace(/_/g, ' ');
}
