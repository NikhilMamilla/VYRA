# Phase 3D — real-trained issue heads

blur / underexposure / overexposure are now trained on real VizWiz features + real crowd labels instead of synthetic degradations. noise / corruption keep their synthetic heads (no VizWiz label exists). Evaluated once on the frozen VizWiz `val` sample (n=2496, ≥3/5 votes).

| issue | Phase 3C (synthetic-trained) | Phase 3D (real-trained) | CV OOF F1 |
|---|---|---|---|
| blur | 0.6114 | **0.6315** | 0.6839 |
| underexposure | 0.4889 | **0.6271** | 0.7705 |
| overexposure | 0.1912 | **0.3564** | 0.3579 |
| **primary macro-F1** | **0.4305** | **0.5383** | — |

Per-issue precision / recall / ROC-AUC / PR-AUC and confusion counts: `runs/phase3d-realtrain-v1/final_evaluation.json`.

## training data

* uniform real-val sample: 2489 images (threshold + CV estimate; natural prevalence)
* rare-enriched extra: 2500 images (training rows only)

| issue | uniform +ve | extra +ve | eval +ve |
|---|---|---|---|
| blur | 754 | 891 | 732 |
| underexposure | 69 | 325 | 69 |
| overexposure | 49 | 291 | 66 |

## thresholds

| issue | Phase 3C | Phase 3D (CV OOF, natural prevalence) |
|---|---|---|
| blur | 0.36 | 0.44 |
| underexposure | 0.50 | 0.56 |
| overexposure | 0.10 | 0.32 |
