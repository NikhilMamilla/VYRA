/**
 * Static facts about the shipped model, mirroring
 * ml/artifacts/vyra-quality-model-v1/bundle.json. Kept here so the explanatory
 * sections have one source; they are stable and documented.
 */

export const MODEL = {
  version: 'vyra-quality-model-v1',
  featureVersion: 'cvfeat-v2',
  family: 'RandomForest (one-vs-rest, 6 issues, 300 trees)',
  features: 42,
  training: 'BSDS500 clean photos + calibrated synthetic degradations · leakage-safe split',
  calibration: 'isotonic per-issue, fitted on a real VizWiz-train validation split',
  workLongEdge: 384,
} as const;

export const PIPELINE = [
  { icon: 'file', title: 'Image', text: 'JPEG / PNG / WebP / BMP / TIFF, up to 10 MB' },
  { icon: 'shield', title: 'Validation', text: 'magic-byte sniff, size, decodability' },
  { icon: 'scan', title: '42 CV features', text: 'sharpness · exposure · contrast · noise · texture · colour · blockiness' },
  { icon: 'cpu', title: 'RandomForest ×6', text: 'one calibrated classifier per issue' },
  { icon: 'activity', title: 'Calibration + thresholds', text: 'isotonic probabilities, real-validation decision points' },
  { icon: 'gauge', title: 'Quality score', text: 'operational 0–100, deterministic' },
  { icon: 'target', title: 'Defect scan', text: 'self-referential patch anomaly, region-localised' },
  { icon: 'code', title: 'Explanation', text: 'statistics + feature evidence + confidence' },
] as const;

export const TIERS = [
  {
    key: 'real',
    tone: 'real' as const,
    title: 'Real-world validated',
    issues: ['blur', 'underexposure', 'overexposure'],
    body: 'Evaluated on real images (VizWiz-QualityIssues, read once). blur F1 0.61, underexposure 0.49, overexposure 0.19.',
  },
  {
    key: 'synth',
    tone: 'synth' as const,
    title: 'Synthetic-validated only',
    issues: ['noise', 'corruption'],
    body: 'Strong on synthetic degradations (F1 0.84 / 0.97) but VizWiz has no matching labels, so there is no real-world number. Shown, clearly marked.',
  },
  {
    key: 'screen',
    tone: 'screen' as const,
    title: 'Screening only',
    issues: ['potential visual defect'],
    body: 'A region statistically unlike the rest of the image. Synthetic ROC-AUC 0.60, region hit-rate 0.32, not real-world validated. Advisory — never "confirmed defect".',
  },
] as const;

export const METRICS = {
  synthetic: [
    ['blur', '0.90'],
    ['underexposure', '0.84'],
    ['overexposure', '0.74'],
    ['noise', '0.84'],
    ['corruption', '0.97'],
  ],
  real: [
    ['blur', '0.61'],
    ['underexposure', '0.49'],
    ['overexposure', '0.19'],
    ['noise', '—'],
    ['corruption', '—'],
  ],
  headline: { synthetic: '0.80', real: '0.43' },
} as const;

export const DISCLAIMERS = [
  {
    title: 'Synthetic → real domain gap',
    body: 'Synthetic macro-F1 0.80, real-world 0.43. Only real labelled training data or domain adaptation would close it.',
  },
  {
    title: 'Operational, not perceptual',
    body: 'The quality score is a deterministic function of calibrated probabilities — not a human mean-opinion-score. No MOS data was available.',
  },
  {
    title: 'Defect is advisory',
    body: 'The Phase-2 global defect classifier was retired (real ROC-AUC 0.42). The patch detector that replaced it is a weak screening cue.',
  },
] as const;
