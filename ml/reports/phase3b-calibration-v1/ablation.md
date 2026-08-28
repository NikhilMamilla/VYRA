# Phase 3B — ablation (real VizWiz `val` evaluation set, ≥3 votes)

Primary macro-F1 = mean F1 over blur / underexposure / overexposure. `defect` is reported separately and excluded (see below).

| step | primary macro-F1 | 4-label macro-F1 | blur F1 | underexp F1 | overexp F1 | defect F1 |
|---|---|---|---|---|---|---|
| Phase 3A baseline (v1 model, synthetic thresholds) | **0.3954** | 0.304 | 0.6108 | 0.4776 | 0.0978 | 0.03 |
| + real-val F1 thresholds | **0.4366** | 0.3361 | 0.6106 | 0.4928 | 0.2063 | 0.0347 |
| + isotonic probability calibration | **0.4309** | 0.332 | 0.6134 | 0.4853 | 0.194 | 0.0352 |
| + cvfeat-v2 blockiness fix (retrained) | **0.4305** | 0.3317 | 0.6114 | 0.4889 | 0.1912 | 0.0353 |
| + post-blur sensor noise (retrained) | **0.4033** | 0.3112 | 0.489 | 0.5037 | 0.2171 | 0.0349 |

Per-label precision / recall / ROC-AUC / PR-AUC and confusion counts are in `runs/phase3b-calibration-v1/final_evaluation.json`.

## defect

defect F1 moves from 0.03 to 0.0349; ROC-AUC stays around 0.5041. It remains below the level of a usable classifier and is excluded from the primary metric. See docs for the localisation recommendation.
