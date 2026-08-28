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
  training:
    'blur / underexposure / overexposure trained on real VizWiz photos (real crowd labels); noise / corruption on calibrated synthetic degradations · leakage-safe splits',
  calibration: 'isotonic per-issue, fitted on cross-validated out-of-fold predictions of a real VizWiz sample',
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
    title: 'Real-world trained & validated',
    issues: ['blur', 'underexposure', 'overexposure'],
    body: 'Trained on real VizWiz-QualityIssues photos and evaluated once on a held-out real sample. blur F1 0.63, underexposure 0.63, overexposure 0.34 (ROC-AUC 0.82 / 0.97 / 0.92). blur also keeps synthetic coverage so it stays robust to strong motion blur.',
  },
  {
    key: 'synth',
    tone: 'synth' as const,
    title: 'Synthetic-validated only',
    issues: ['noise', 'corruption'],
    body: 'Strong on synthetic degradations (F1 0.84 / 0.97) but VizWiz has no matching labels, so there is no real-world number and no way to train on real data. Shown, clearly marked.',
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
    ['blur', '0.63'],
    ['underexposure', '0.63'],
    ['overexposure', '0.34'],
    ['noise', '—'],
    ['corruption', '—'],
  ],
  headline: { synthetic: '0.80', real: '0.54' },
  previousReal: '0.43',
} as const;

export const DISCLAIMERS = [
  {
    title: 'Synthetic → real domain gap (mostly closed for the trained heads)',
    body: 'Training blur / underexposure / overexposure on real VizWiz labels lifted real primary macro-F1 from 0.43 to 0.54. noise and corruption stay synthetic-only — VizWiz has no such labels, so the gap there is unmeasured.',
  },
  {
    title: 'Operational, not perceptual',
    body: 'The quality score is a deterministic function of calibrated probabilities — not a human mean-opinion-score. No MOS data was available.',
  },
  {
    title: 'overexposure is still the weakest head',
    body: 'Real F1 0.34, low recall (0.23) — precise but conservative, and it under-fires on uniformly blown-out frames. Ranking signal is strong now (ROC-AUC 0.92, up from 0.65); only 49 real positives in the tuning sample, so the number is directional. A stress test (ml/scripts/phase3d_stress_test.py) tracks the gap.',
  },
  {
    title: 'Defect is advisory',
    body: 'The Phase-2 global defect classifier was retired (real ROC-AUC 0.42). The patch detector that replaced it is a weak screening cue.',
  },
] as const;
