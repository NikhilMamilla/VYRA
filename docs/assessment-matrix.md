# VYRA — assessment requirement matrix

Every requirement from the technical assessment → where it is implemented →
evidence. `✅` met, `⚠️` met with documented limitations, `➖` optional/bonus.

## 2. Required detection capabilities

| Requirement | Implementation | Location | Evidence |
|---|---|---|---|
| Blur / insufficient sharpness | RF **trained on real VizWiz photos**, 8 sharpness + texture features | `ml/vyra_ml/features/sharpness.py`, model `blur` head | ✅ synthetic F1 0.90, **real F1 0.63** (`ml/reports/phase3d-realtrain-v1/phase3d.md`); motion-blur stress recall 0.98 |
| Underexposure | RF **trained on real VizWiz photos**, exposure/contrast features | `features/exposure.py`, `underexposure` head | ✅ synthetic 0.84, **real 0.63** (ROC-AUC 0.97) |
| Overexposure | RF **trained on real VizWiz photos**, exposure/contrast features | `features/exposure.py`, `overexposure` head | ⚠️ synthetic 0.74, **real 0.34** (ROC-AUC 0.92; recall 0.23, 49 tuning positives — directional) |
| Image noise | RF over 4 no-reference noise estimators | `features/noise.py`, `noise` head | ⚠️ synthetic 0.84; **no real-world validation** (VizWiz has no noise label) |
| Corruption / severe degradation | RF over blockiness + resolution-loss features | `features/compression.py`, `corruption` head | ⚠️ synthetic 0.97; **no real-world validation** |
| Potential visual defect | self-referential patch-anomaly detector | `ml/vyra_ml/defect/patch_anomaly.py` | ⚠️ screening only, synthetic ROC-AUC 0.60, region hit-rate 0.32 (`ml/docs/defect.md`) |

## 3. AI / CV requirements

| Requirement | Implementation | Location | Evidence |
|---|---|---|---|
| Genuine learned decision component | 6× one-vs-rest RandomForest (blur/under/over trained on real VizWiz data) + isotonic calibration + patch-anomaly model | `ml/vyra_ml/experiment/{baseline,phase3d}.py`, `calibration/`, `defect/` | `ml/artifacts/vyra-quality-model-v1/model.joblib` (300-tree forests), `bundle.json` |
| Model selection explanation | classical over CNN; real-data training for the VizWiz-evaluable heads; anomaly detection for defect | README §4, `ml/docs/phase3d-realtrain.md`, `ml/docs/phase3b-calibration.md` §7 | ✅ |
| Data preparation | leakage-safe synthetic dataset from clean images; real VizWiz training set (uniform + rare-enriched, leakage-audited) | `ml/vyra_ml/dataset_build.py`, `splitting.py`, `degradations/`, `ml/scripts/phase3d_fetch_extra.py` | `ml/docs/dataset.md`, `ml/docs/phase3d-realtrain.md`; `ml/reports/phase2-baseline-v1/dataset_report.json` |
| Training methodology | per-issue RF, `class_weight="balanced"`, seeded; synthetic heads F1-tuned on `val`; real heads CV-tuned on a disjoint real sample (OOF threshold + isotonic) | `baseline.py`, `phase3d.py` | `ml/runs/phase2-baseline-v1_*/experiment.json`, `ml/runs/phase3d-realtrain-v1/status.json` |
| Evaluation | 3-level protocol: synthetic test, real-world (read once per model generation), failure analysis | `ml/vyra_ml/evaluation/`, `realworld/evaluate.py`, `experiment/phase3d.py` | `ml/reports/phase3a-real-world-v1/`, `ml/runs/{phase3b-calibration-v1,phase3d-realtrain-v1}/final_evaluation.json` |

## 4. Image analysis

| Characteristic | Feature(s) | Location |
|---|---|---|
| Sharpness | `sharp_laplacian_var`, `sharp_tenengrad`, `sharp_highfreq_ratio`, `sharp_edge_density` (+4) | `features/sharpness.py` |
| Brightness / exposure | `expo_luma_mean/median`, `expo_dark/bright_clip_ratio`, `expo_shadow/highlight_ratio` (+2) | `features/exposure.py` |
| Contrast | `contrast_std`, `contrast_dynamic_range`, `contrast_michelson`, `contrast_local_std_mean` (+3) | `features/exposure.py` |
| Noise | `noise_immerkaer_sigma`, `noise_highfreq_residual_std`, `noise_median_residual_mad`, `noise_flat_region_std` | `features/noise.py` |
| Texture | `texture_spectral_slope`, GLCM contrast/homogeneity/energy, LBP entropy | `features/texture.py` |
| Colour / saturation | `color_saturation_mean/std`, `color_colourfulness`, `color_cast`, `color_gray_pixel_ratio` (+2) | `features/color.py` |
| Compression | `compress_blockiness{,_h,_v}` | `features/compression.py` |

42 features total, surfaced in the API `metrics` block and `explanation.evidence`.
Full list: `ml/docs/features.md`.

## 5. Backend

| Requirement | Implementation | Location | Test |
|---|---|---|---|
| REST API for upload + analysis | `POST /api/v1/analyses` | `backend/app/api/v1/routes/analyses.py` | `test_analyzer.py::test_upload_analyze_persist_retrieve` |
| Validate files, handle invalid/unreadable gracefully | magic-byte sniff + size + declared-type check; analyzer raises `InvalidImageError` | `app/services/image_validation.py`, `app/analysis/vyra_analyzer.py` | `test_image_validation.py`, `test_analyses_api.py`, `test_analyzer.py::test_garbage_upload...` |
| Structured JSON result | `AnalysisRead` / `AnalysisOutcome` Pydantic models | `app/schemas/analysis.py` | ✅ |
| Persist results in a database | PostgreSQL via async SQLAlchemy; SQLite in tests | `app/db/`, `app/repositories/analysis_repository.py` | `test_analyzer.py`, `test_analyses_api.py` |
| Retrieve previous results | `GET /api/v1/analyses`, `GET /api/v1/analyses/{id}` | `routes/analyses.py` | `test_analyses_api.py`, `test_analyzer.py` |
| Error handling + HTTP status codes | one error envelope; 413/415/422/404/500/501 | `app/core/errors.py` | `test_analyses_api.py` (all codes), `test_analyzer.py::test_analyzer_failure...` |

## 6. Frontend

| Requirement | Implementation | Location |
|---|---|---|
| Usable upload interface | drag/drop + file picker + client pre-checks | `frontend/src/components/UploadDropzone.tsx` |
| Display image + assessment | preview with defect-region overlay | `components/ImageCanvas.tsx`, `AnalysisResult.tsx` |
| Quality score + detected issues | gauge + issue list | `components/QualityScore.tsx`, `IssueList.tsx` |
| Severity, confidence, statistics | per-issue badges + bar; statistics grid | `IssueList.tsx`, `StatisticsGrid.tsx` |
| History | list panel, click to re-view | `components/HistoryPanel.tsx`, `hooks/useHistory.ts` |
| Loading / success / error states | explicit state machine | `components/analyze/Workspace.tsx`, `hooks/useAnalyze.ts` |
| Responsive & polished | Tailwind, single-column mobile → sidebar desktop; light/dark themes (light default, persisted); glass/clay/neumorphic/skeuomorphic design system | `src/index.css`, `src/theme/`, `tailwind.config.js` |
| Explanatory content | hero + "How it works" (pipeline) + "Under the hood" (model card, capability tiers) + "Honest metrics" (synthetic vs real, disclaimers) | `components/layout/`, `components/marketing/` |
| Automated tests | 6 vitest tests (loading / result / error / history / API-down / theme toggle) | `src/App.test.tsx` |

## 7. Expected analysis result

`{ quality_score, quality_label, issues:[{type,severity,confidence,validation,detail}], metrics, explanation }`
— superset of the assessment's example shape. `docs/api.md`.

## 8. Dataset & training

| Requirement | Implementation | Evidence |
|---|---|---|
| Public dataset / synthetic degradation | BSDS500 clean images + controlled synthetic degradations (noise/corruption heads); VizWiz-QualityIssues real photos + crowd labels (blur/under/over heads) | `ml/docs/dataset.md`, `ml/docs/phase3d-realtrain.md` |
| Describe how train/eval data were generated | per-degradation classes, ranged severity params, seeded; real split = seeded uniform + rare-enriched VizWiz-train samples | `ml/docs/dataset.md`, `ml/docs/phase3d-realtrain.md`, `ml/vyra_ml/degradations/` |
| Evaluation on unseen images + generalization evidence | untouched synthetic `test` split + real VizWiz `val` (read once per model generation) | `ml/reports/phase3a-real-world-v1/`, `ml/reports/phase3d-realtrain-v1/`, `ml/docs/real-world-validation.md` |
| Leakage prevention | hash-based split on original `source_id` before degradation; `assert_no_leakage`; Phase 3D `data_audit` asserts 0 id/SHA-1 overlap of real train pools vs the frozen eval set | `ml/vyra_ml/splitting.py`, `ml/vyra_ml/experiment/phase3d.py`, `ml/tests/test_splitting.py` |

## 9. Evaluation

| Requirement | Implementation | Evidence |
|---|---|---|
| Appropriate metrics (F1, ROC-AUC, PR-AUC, confusion, regression error) | multi-label report + per-issue confusion + regression report | `ml/vyra_ml/evaluation/metrics.py`; `metrics.json` in every run |
| Failure cases | representative false-positive/negative images per issue | `ml/reports/phase3a-real-world-v1/failure_examples/`, `failure_analysis.json` |
| Limitations + uncertain-prediction discussion | dedicated sections | README §7, `ml/docs/*.md`, `ml/reports/phase3a-real-world-v1/synthetic_vs_real.md` |

## 10. Explainability

| Requirement | Implementation | Location |
|---|---|---|
| Interpretable image statistics | 10 named statistics in the API `metrics` block | `app/analysis/vyra_analyzer.py::_STATISTICS` |
| Feature importance / evidence | per-flagged-issue feature values + direction in `explanation.evidence` | `vyra_analyzer.py`; per-issue `feature_importances_top8` in `metrics.json` |
| Confidence | calibrated probability per issue + `issue_probabilities` for all | `bundle.json` calibrators |
| Region localization | flagged defect region (`explanation.potential_defect.region`, UI overlay) | `defect/patch_anomaly.py`, `components/ImageCanvas.tsx` |

## 11. Deployment

| Requirement | Implementation | Evidence |
|---|---|---|
| Runnable outside dev environment | Docker Compose | `docker-compose.yml`, `backend/Dockerfile`, `frontend/Dockerfile` |
| Clear setup instructions | README §13–15 | ✅ |
| Containerization (preferred) | 3 services (db, backend, frontend) | `docker compose up --build` verified |
| Frontend/backend communicate in deployment | nginx serves SPA + proxies `/api` same-origin | `frontend/nginx.conf` |
| Environment variables for config | 14 documented vars | `.env.example`, README §12 |
| Health/status endpoint | `GET /health` (service + db + storage + analyzer + model version) | `app/api/health.py`, `test_health.py` |
| Document model loading + inference after deployment | README §13 "Model loading", `docs/architecture.md` | ✅ |

## 12. Submission

| Item | Location |
|---|---|
| Frontend / backend / ML source | `frontend/`, `backend/`, `ml/` |
| Inference model | `ml/artifacts/vyra-quality-model-v1/` (committed, ~6 MB) |
| README (setup, model/training, API, deployment) | `README.md` |
| Database setup | README §11 |
| API documentation + example requests | `docs/api.md` |
| Evaluation results + technical explanation | README §5, `ml/reports/`, `ml/docs/` |
| Sample images (different quality conditions) | `demo/` (+ `demo/README.md`) |
| Docker configuration | `docker-compose.yml`, `*/Dockerfile`, `.dockerignore` |
| Deployed URL | not deployed online (local Docker Compose is acceptable per §11) |

## 13. Optional / bonus

| Bonus item | Status |
|---|---|
| Quality heatmaps / localization of problematic regions | ➖ done — patch-anomaly defect region + UI overlay |
| Confidence calibration / uncertainty estimation | ➖ done — isotonic per-issue calibration on a real validation split, Brier/ECE reported |
| Model versioning | ➖ done — `model_version` in every response and `/health`; self-describing `bundle.json` |
| Automated backend/frontend tests | ➖ done — 34 backend + 118 ML + 5 frontend |
| Batch analysis, CI/CD, monitoring, perf-opt for concurrency | not done (out of scope for the release) |
