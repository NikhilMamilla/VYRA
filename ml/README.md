# `ml/` — VYRA dataset, features and experiments

A **reproducible data + experimentation foundation** for VYRA's image-quality
AI, plus the shipped inference package. It builds a leakage-safe synthetic
dataset from clean real images, extracts interpretable CV features, trains a
classical multi-label baseline, validates it on real images, calibrates it, and
bundles the result for the backend.

**Phase 3C (shipped).** `vyra_ml.inference.VyraQualityModel` is the single
inference entry point; the backend `QualityAnalyzer` wraps it. Everything needed
for inference is in the self-describing bundle
[`artifacts/vyra-quality-model-v1/`](artifacts/vyra-quality-model-v1/) (`model.joblib`,
`calibrators.joblib`, `defect_detector.json`, `bundle.json`), rebuilt by
`python scripts/export_inference_bundle.py`. The `potential_visual_defect`
signal is a self-referential patch-anomaly detector
([`vyra_ml/defect/`](vyra_ml/defect/), see [`docs/defect.md`](docs/defect.md)),
calibrated by `python scripts/build_defect_detector.py`. Quality-score
definition: [`docs/quality-score.md`](docs/quality-score.md).

**Phase 3A (done)** validated the frozen baseline on real images
(VizWiz-QualityIssues). Transfer at frozen thresholds: macro-F1 **0.74 → 0.30**.
blur and underexposure keep real ranking signal (ROC-AUC 0.79 / 0.89) but are
mis-calibrated out of domain; overexposure and defect largely fail; `noise` and
`corruption` are not testable against VizWiz. See
[`docs/real-world-validation.md`](docs/real-world-validation.md) and
[`reports/phase3a-real-world-v1/`](reports/phase3a-real-world-v1/).

**Phase 3B (done)** — feature fix + synthetic-blur realism + real-validation
calibration, without fine-tuning on real data. `cvfeat-v2` bounds the
`compress_blockiness` blow-up; `configs/experiment_blurnoise.yaml` adds post-blur
sensor noise; thresholds and isotonic calibrators are fitted **only** on a real
*validation* split from VizWiz `train`, then measured once on the untouched
Phase 3A eval set. See
[`docs/phase3b-calibration.md`](docs/phase3b-calibration.md),
[`reports/phase3b-calibration-v1/`](reports/phase3b-calibration-v1/) and
`runs/phase3b-calibration-v1/`. Run: `python scripts/phase3b_run.py`.

## Quick start

```bash
cd ml
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

python scripts/pipeline.py all          # build → features → reports → baseline
#   or step by step:
python scripts/pipeline.py build            # ingest → split → degrade → manifest
python scripts/pipeline.py features         # CV feature extraction
python scripts/pipeline.py feature-report   # NaN / range / correlation / size-robustness
python scripts/pipeline.py stats            # dataset statistics + plots
python scripts/pipeline.py baseline         # train + evaluate the classical baseline

python scripts/export_inference_bundle.py  # assemble artifacts/vyra-quality-model-v1/
python scripts/make_demo_samples.py        # write demo/ images

pytest                                   # 118 tests
```

Everything is driven by [`configs/experiment.yaml`](configs/experiment.yaml).
Same source images + same config + same seed ⇒ same dataset and same metrics.

## What's here

| Path | Purpose |
|---|---|
| `vyra_ml/config.py` | typed loader for `experiment.yaml` |
| `vyra_ml/seeding.py` | name-scoped deterministic RNG derivation |
| `vyra_ml/ingest/` | source adapters (BSDS500 clean images; skimage offline fallback) |
| `vyra_ml/splitting.py` | **leakage-safe original-level split** + `assert_no_leakage` |
| `vyra_ml/degradations/` | one tested class per degradation (blur, exposure×2, noise, corruption, defect) |
| `vyra_ml/labels.py` | multi-label vector + provisional quality target |
| `vyra_ml/manifest.py` | manifest schema, Parquet + JSONL writers |
| `vyra_ml/dataset_build.py` | build orchestrator |
| `vyra_ml/features/` | `cvfeat-v2` — 42 interpretable features, grouped modules |
| `vyra_ml/feature_store.py` | feature extraction over a manifest → Parquet cache |
| `vyra_ml/feature_report.py` | feature-space sanity report |
| `vyra_ml/dataset_stats.py` | dataset statistics report + plots |
| `vyra_ml/experiment/baseline.py` | per-issue RF/HGB classifiers + quality regressor |
| `vyra_ml/evaluation/` | multi-label metrics + the 3-level evaluation protocol |
| `vyra_ml/realworld/` | **Phase 3A/3B** — VizWiz ingestion, label mapping, partial-zip fetch, feature tables, evaluation |
| `vyra_ml/calibration/` | **Phase 3B** — real-val threshold selection + isotonic probability calibration |
| `vyra_ml/experiment/phase3b.py` | **Phase 3B** orchestrator (resumable, status.json per step) |
| `vyra_ml/inference.py` | **Phase 3C** — `VyraQualityModel`: image bytes → structured quality analysis |
| `vyra_ml/defect/` | **Phase 3C** — self-referential patch-anomaly `potential_visual_defect` detector |
| `artifacts/vyra-quality-model-v1/` | **Phase 3C** — the shipped inference bundle (committed) |
| `docs/` | [`dataset.md`](docs/dataset.md), [`features.md`](docs/features.md), [`evaluation-protocol.md`](docs/evaluation-protocol.md), [`real-world-validation.md`](docs/real-world-validation.md), [`phase3b-calibration.md`](docs/phase3b-calibration.md) |
| `reports/<version>/` | committed: dataset + feature reports, plots |
| `runs/<version>_<ts>/` | committed: `experiment.json`, `metrics.json` (model `.joblib` git-ignored) |
| `data/` | git-ignored: raw datasets, generated images, feature cache, manifests |

## Baseline results (synthetic test split, `phase2-baseline-v1`)

400 originals → 3600 samples (2574 / 513 / 513). Per-issue one-vs-rest Random
Forest on 42 features. **Test macro-F1 0.79, micro-F1 0.79, subset accuracy
0.60.**

| Issue | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| corruption | 0.94 | 1.00 | 0.99 |
| blur | 0.90 | 0.98 | 0.96 |
| underexposure | 0.85 | 0.97 | 0.92 |
| noise | 0.83 | 0.97 | 0.91 |
| overexposure | 0.77 | 0.91 | 0.82 |
| **defect** | **0.46** | **0.75** | **0.52** |

`defect` (localised, heterogeneous) is the weak point — the Phase 2 global
classifier is **not shipped**; Phase 3C replaces it with a patch-anomaly
detector ([`docs/defect.md`](docs/defect.md)). Provisional quality regressor:
MAE 13.3 / R² 0.66 on a **provisional** target — not the shipped score
([`docs/quality-score.md`](docs/quality-score.md)).

## Real-world results (the shipped model, `vyra-quality-model-v1`)

VizWiz-QualityIssues `val` sample, ≥3/5 votes (Phase 3B row D). Primary macro-F1
**0.43** (blur / underexposure / overexposure). Full story:
[`docs/phase3b-calibration.md`](docs/phase3b-calibration.md).

| Issue | real F1 | tier |
|---|---|---|
| blur | 0.61 | real-world validated |
| underexposure | 0.49 | real-world validated |
| overexposure | 0.19 | real-world validated but weak |
| noise / corruption | — | synthetic-validated only (no VizWiz mapping) |
| potential visual defect | — | screening only (synthetic ROC-AUC 0.60) |

## Contract with `backend/`

`backend/app/analysis/contract.py` defines `QualityAnalyzer` (image **bytes** →
`AnalysisOutcome`). `backend/app/analysis/vyra_analyzer.py` implements it by
wrapping `vyra_ml.inference.VyraQualityModel.load(MODEL_PATH)`, loaded once at
startup. `MODEL_PATH` points at [`artifacts/vyra-quality-model-v1/`](artifacts/vyra-quality-model-v1/);
its `bundle.json` pins the model run, feature version, per-issue thresholds,
calibration, label definitions, quality-score formula and the real-world
evaluation numbers, so the backend hard-codes nothing about the model.
