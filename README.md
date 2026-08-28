# VYRA — AI-Powered Image Quality & Defect Detection

VYRA accepts an image, evaluates its visual quality with computer-vision features
plus a calibrated machine-learning model, and returns an operational quality
score (0–100), a quality label, the specific issues it found with per-issue
**confidence** and **severity**, a localised *potential visual defect* signal,
the image statistics behind the decision, and a short explanation. It runs
end-to-end with `docker compose up --build`.

---

## 1. Overview & features

| Capability | How | Tier |
|---|---|---|
| **Blur / insufficient sharpness** | RF over sharpness/texture features | real-world validated (VizWiz F1 **0.61**) |
| **Underexposure** | RF over exposure/contrast features | real-world validated (F1 **0.49**) |
| **Overexposure** | RF over exposure/contrast features | real-world validated but **weak** (F1 **0.19**) |
| **Image noise** | RF over 4 no-reference noise estimators | **synthetic-validated only** — no real-world evaluation exists (synthetic F1 0.84) |
| **Corruption / severe degradation** | RF over blockiness + resolution features | **synthetic-validated only** (synthetic F1 0.97) |
| **Potential visual defect** | self-referential patch-anomaly detector | **screening only** — ROC-AUC 0.60 on synthetic, not real-world validated |
| Overall **quality score + label** | operational formula over calibrated probabilities | deterministic, 0–100, `GOOD/ACCEPTABLE/DEGRADED/POOR` |
| **Explainability** | image statistics, per-issue feature evidence, confidence, defect region | on every response |
| **History** | `GET /api/v1/analyses` + UI panel | persisted in PostgreSQL |

> **Honest metrics.** Synthetic F1 is **not** real-world performance. blur /
> under / overexposure were evaluated on real images (VizWiz-QualityIssues);
> noise / corruption have **no** real-world evaluation (VizWiz doesn't annotate
> them); the defect signal is advisory. There is a real synthetic→real domain
> gap (macro-F1 0.79 synthetic → 0.43 real). See §5–§7.

Frontend: a single responsive page — a hero, the **analysis workspace**
(drag/drop upload with client pre-checks, image preview with defect-region
overlay, quality-score dial, issue list with severity + confidence +
validation-tier badge, image-statistics grid, explanation panel, clickable
history, and explicit loading / success / error states), then explanatory
sections — *How it works* (the pipeline), *Under the hood* (model card +
capability tiers) and *Honest metrics* (synthetic vs real-world + disclaimers).
**Light and dark themes** (light default, choice persisted); the design system
uses glass / clay / neumorphic / skeuomorphic surfaces each in a defined role.

## 2. Tech stack

| Layer | Choice |
|---|---|
| Frontend | React 19, TypeScript, Vite, Tailwind CSS |
| Backend | Python 3.11, FastAPI, Pydantic v2 |
| Database | PostgreSQL via SQLAlchemy 2 (async, `asyncpg`); Supabase-compatible; SQLite in tests |
| Storage | Pluggable `ObjectStorage` protocol; local filesystem / Docker volume by default |
| Inference | OpenCV, scikit-image, scikit-learn, NumPy, joblib (`vyra_ml` package) |
| Deployment | Docker + Docker Compose, nginx in front of the SPA |

No external AI or vision APIs. No API keys required.

## 3. Architecture

```
Browser ──▶ nginx (SPA + /api proxy) ──▶ FastAPI
                                          │
   POST /api/v1/analyses:                 │
     validate (magic bytes, size, type)   │  413 / 415 / 422
        └─▶ analyze  (vyra_ml.inference, on a worker thread)   422 on undecodable, else 500
              features → 6× RandomForest → isotonic calibration
              → real-val thresholds → quality score → patch-anomaly defect
        └─▶ store image  (ObjectStorage: <analysis_id>/original.<ext>)
        └─▶ persist row  (AnalysisRepository → PostgreSQL)
              on insert failure: delete the blob, re-raise
     201  ◀── full result JSON
```

Order is **analyze → store → persist**: an un-analyzable image is never stored;
a failed insert never leaves an orphan blob or a partial row. Three protocol
boundaries carry the design — `ObjectStorage`, `QualityAnalyzer`, and the
repository layer — so the CV/ML code knows nothing about HTTP, storage or the
database. Details: [`docs/architecture.md`](docs/architecture.md).

## 4. ML approach

**Classical ML over engineered features**, per the assessment's first listed
option. A per-issue one-vs-rest **RandomForest** (`class_weight="balanced"`,
300 trees) over **42 interpretable CV features** (`cvfeat-v2`), with **isotonic
probability calibration** and **decision thresholds fitted on a real-image
validation split**.

**Why not a CNN.** The evidence (Phase 3A/3B, §5) showed the only intervention
that improved *real-world* F1 was re-selecting thresholds on real data; a
synthetic-realism experiment made things worse. A CNN would need real labelled
training data we do not have, and would not be an honest, defensible result.
An anomaly-detection formulation *is* used for the one issue where global
features provably fail — the defect signal (§ Defect methodology).

**Model selection, in order:**
1. Phase 2: RF baseline over 42 features → synthetic test macro-F1 **0.79**.
2. Phase 3A: froze it, evaluated on VizWiz → macro-F1 **0.30** (domain gap).
3. Phase 3B: fixed a feature bug (`cvfeat-v2`), fitted thresholds + calibration
   on a real *validation* split → real primary macro-F1 **0.43**. Rejected a
   blur-noise augmentation (made blur worse).
4. Phase 3C: retired the untrustworthy global defect head, replaced it with a
   patch-anomaly detector, bundled everything, integrated.

The 42 features cover **sharpness** (Laplacian variance, Tenengrad, gradient
stats, FFT high-freq ratio, edge density), **exposure** (luma mean/median, clip
ratios, histogram entropy, skew), **contrast** (RMS, dynamic range, Michelson,
local std), **noise** (Immerkaer sigma, high-freq residual, median-residual MAD,
flat-region std), **colour** (saturation, Hasler-Suesstrunk colourfulness,
colour cast, grey-pixel ratio), **texture** (radial power-spectrum slope, GLCM
contrast/homogeneity/energy, LBP entropy), **compression** (8-px blockiness).
Full list + rationale: [`ml/docs/features.md`](ml/docs/features.md).

## 5. Dataset, training & evaluation

### Dataset — clean real images + controlled synthetic degradation

Base: **BSDS500** (400 clean natural photos, ~72 MB, Berkeley research licence).
Chosen over ingesting a large real quality dataset because ground truth is then
**known by construction** (we applied the degradation), coverage of all six
issues at all severities is guaranteed, and a 48-h assessment cannot responsibly
curate tens of GB. The cost — unproven generalisation to *real* degradation — is
measured, not hidden (§ Real-world validation).

### Synthetic degradation methodology

One tested class per degradation (`ml/vyra_ml/degradations/`). Each takes
`(image, severity 1–5, seeded RNG)` and returns the image plus the **exact
sampled parameters** (stored in the manifest). Parameters are drawn from a
**range per severity**, never a fixed value, so the model cannot memorise a
synthetic fingerprint. Realism choices: exposure applied in a light-linear
domain; dark frames get extra read noise so underexposure ≠ noise; highlights
roll off before clipping; JPEG is a real `cv2.imencode` round-trip; corruption
includes genuine resolution loss; contradictory pairs (under+overexposure)
excluded. Application order approximates a capture pipeline:
`exposure → blur → defect → noise → corruption`. Per original: 1 clean + 6
single-degradation + 2 multi-degradation samples → **3,600 samples**. Full
detail: [`ml/docs/dataset.md`](ml/docs/dataset.md).

### Train / validation / test strategy & leakage prevention

The split is decided on the **original `source_id`, before any degradation is
generated** (`ml/vyra_ml/splitting.py`): each original is hashed (blake2b,
seeded) to a unit interval and assigned train/val/test by ratio; every degraded
variant inherits that split. `assert_no_leakage()` runs on the finished manifest
and raises if any `source_id` appears under more than one split — asserted
end-to-end by a dedicated test. Because assignment is by hash, it is
order-independent and reproducible from the seed alone.

- **Synthetic:** 400 originals → 286 / 57 / 57 → **2,574 / 513 / 513** samples.
  `val` selects per-issue thresholds; `test` is untouched until final reporting.
- **Real validation** (Phase 3B): a seeded 2,489-image sample of VizWiz **train**
  — used *only* to fit thresholds and calibrators.
- **Real evaluation** (Phase 3A/3B): a seeded 2,496-image sample of VizWiz
  **val** — read **once**, at the end. `leakage_check` asserts 0 image-id and
  0 SHA-1 overlap between the two real splits and against the synthetic set
  (9 VizWiz images that appear in both its own train and val splits were dropped
  from the *validation* side).

### Evaluation results

| | metric | value |
|---|---|---|
| **Synthetic test** (`vyra-quality-model-v1`, unseen split) | macro-F1 | **0.80** |
| | blur / underexp / overexp / noise / corruption F1 | 0.90 / 0.84 / 0.74 / 0.84 / 0.97 |
| | defect F1 (retired global head) | 0.49 |
| **Real-world** (VizWiz-QualityIssues `val`, ≥3/5 votes, read once) | primary macro-F1 | **0.43** |
| | blur / underexposure / overexposure F1 | 0.61 / 0.49 / 0.19 |
| | blur / underexposure ROC-AUC | 0.79 / 0.89 |
| **Patch defect detector** (synthetic test) | ROC-AUC / precision / recall | 0.60 / 0.33 / 0.32 |
| | region localisation hit-rate | 0.32 |
| Provisional quality regressor (synthetic, not shipped) | MAE / R² | 13.3 / 0.66 |

Committed evidence: [`ml/reports/`](ml/reports/) (Phase 2 / 3A / 3B reports +
failure-example images), [`ml/runs/`](ml/runs/) (per-run `experiment.json` +
`metrics.json`), [`ml/docs/real-world-validation.md`](ml/docs/real-world-validation.md),
[`ml/docs/phase3b-calibration.md`](ml/docs/phase3b-calibration.md).

## 6. Failure analysis

From Phase 3A's failure examples (`ml/reports/phase3a-real-world-v1/failure_examples/`):

- **Blur false negatives:** slightly-soft phone photos where a textured
  background (counter speckle, fabric) keeps the sharpness features high.
- **Overexposure:** real "too bright" is often local glare / reflections, not
  global overexposure — global exposure features miss it (F1 0.19, ROC-AUC 0.63).
- **Underexposure false positives:** low-key but correctly-exposed scenes.
- **Defect:** the retired global head predicted near-randomly on real images
  (ROC-AUC 0.42); the patch detector still misses ~2/3 of synthetic defects and
  fires on legitimately high-contrast regions ~17% of the time.
- **cvfeat-v1 bug (fixed):** `compress_blockiness` diverged to ~590,000 on
  blurred+JPEG images; `cvfeat-v2` bounds it.

## 7. Limitations (honest)

- **Synthetic → real gap.** Synthetic macro-F1 0.79 vs real 0.43. Only real
  labelled training data or a domain-adaptation step would close it.
- **overexposure** is weak on real photos (F1 0.19).
- **noise and corruption have zero real-world validation** — VizWiz has no such
  categories. Their strong numbers are synthetic-only and are labelled
  `synthetic-only` in every API response.
- **potential visual defect** is a weak screening cue (ROC-AUC 0.60), advisory
  only, never "confirmed defect".
- **Quality score is operational, not perceptual / MOS** — no human-rated data
  was available.
- No authentication, no rate limiting, no DB migrations (tables auto-created),
  no Supabase Storage adapter, synchronous inference (~1.5 s/image).

## 8. Quality-score definition

**Operational**, not perceptual. For each issue with calibrated probability
`pᵢ`, threshold `tᵢ`, weight `wᵢ` and `severe = 0.9`:

```
impactᵢ = wᵢ · clip((pᵢ − tᵢ) / (severe − tᵢ), 0, 1)      # 0 below threshold
score   = 100 · Πᵢ (1 − impactᵢ)
```

Deterministic, bounded [0, 100], monotone (raising any `pᵢ` never raises the
score). Compounding models diminishing marginal damage. Weights (blur 0.55,
corruption 0.45, underexposure 0.45, overexposure 0.35, noise 0.30, defect 0.20)
are ordered by usability impact, with the unvalidated signals held down. Bands
(from the observed score distribution): **GOOD ≥ 85, ACCEPTABLE 68–84,
DEGRADED 45–67, POOR < 45**. Formula, weights and bands all live in
`bundle.json`. Full rationale: [`ml/docs/quality-score.md`](ml/docs/quality-score.md).

## 9. Defect methodology

The Phase 2 global `defect` classifier is **not exposed** — it did not localise
(max feature importance 0.035) and got real-world ROC-AUC 0.42 (worse than
chance). Replaced by an **anomaly-detection** approach: a **self-referential
patch-anomaly** detector (`ml/vyra_ml/defect/patch_anomaly.py`). The image is
tiled into overlapping 64-px patches; each gets 8 cheap local statistics; a
patch is scored by how far its statistics deviate (robust, clipped z-score) from
the **median patch of the same image**; the image score is the strongest patch,
mapped through a fixed-slope sigmoid, and the flagged region is that patch.

Self-referential → needs no training data, cannot memorise a synthetic
fingerprint, and the reference is the image itself. Calibrated once on the
synthetic defect bounding boxes. It is **weak and labelled as such everywhere**:
synthetic ROC-AUC 0.60, precision 0.33, region hit-rate 0.32, no real-world
validation. Exposed as `potential_visual_defect` with a permanent disclaimer;
quality-score weight only 0.20. Full write-up:
[`ml/docs/defect.md`](ml/docs/defect.md).

## 10. API

| Method | Path | Behaviour |
|---|---|---|
| `GET` | `/health` | service + DB + storage + analyzer status, `analyzer_model_version` |
| `POST` | `/api/v1/analyses` | upload → validate → analyze → store → persist → **201** with the full result |
| `GET` | `/api/v1/analyses?limit=&offset=` | paginated history, newest first |
| `GET` | `/api/v1/analyses/{id}` | one analysis, `404` if unknown |

Status codes: `413` too large, `415` unsupported type, `422` empty / not a
readable image, `500` analyzer failure (nothing stored or persisted), `501` no
model loaded. Every error: `{"error": {"code": "...", "message": "..."}, "request_id": "..."}`.
Full reference: [`docs/api.md`](docs/api.md). Interactive: `/docs`.

### Example `201` response (abridged)

```json
{
  "id": "a7f65b15-6b4b-4f31-94d2-b864fb3a8909",
  "status": "completed",
  "image": { "filename": "photo.jpg", "content_type": "image/jpeg", "size_bytes": 41720, "width": 481, "height": 321 },
  "quality_score": 68.2,
  "quality_label": "ACCEPTABLE",
  "model_version": "vyra-quality-model-v1",
  "issues": [
    { "type": "blur", "severity": "medium", "confidence": 0.6723, "validation": "real-world",
      "detail": "Validated on real images (VizWiz F1 0.6114)." }
  ],
  "metrics": { "sharpness": 56.42, "brightness": 0.365, "contrast": 0.151, "noise_sigma": 0.827,
               "saturation": 0.184, "colourfulness": 23.19, "blockiness": 0.0007, "edge_density": 0.0216 },
  "explanation": {
    "summary": "Quality score 68/100 (ACCEPTABLE). Flagged: blur.",
    "evidence": [ { "feature": "sharp_highfreq_ratio", "value": 0.2624, "direction": "lower_sharpness_supports_blur" } ],
    "issue_probabilities": { "blur": 0.67, "underexposure": 0.08, "overexposure": 0.01, "noise": 0.12, "corruption": 0.03 },
    "potential_defect": { "probability": 0.11, "flagged": false, "region": null },
    "capabilities": { "real_world_validated": ["blur","underexposure","overexposure"],
                      "synthetic_validated_only": ["noise","corruption"],
                      "screening_only": ["potential_visual_defect"] }
  }
}
```

`confidence` = calibrated P(issue present). `severity` (`low`/`medium`/`high`) =
bucketed estimate of impact — deliberately distinct from confidence.

## 11. Database & storage

**Database.** One `analyses` row per analysis. Queryable fields (`created_at`,
`status`, `quality_score`, `quality_label`, image metadata) are real columns; the
issue / metric / explanation payload is one `JSONB` column. `DATABASE_URL` is any
async SQLAlchemy URL. Under Docker Compose a Postgres container is started
automatically and the schema is created at startup (`DATABASE_AUTO_CREATE=true`).
**Supabase Postgres** works with no code change — take the connection string from
Supabase → Project Settings → Database and set the driver to
`postgresql+asyncpg://`. The Supabase client SDK is deliberately not a dependency.

**Storage.** Uploaded images go through the `ObjectStorage` protocol
(`backend/app/storage/`). The default `local` backend writes to
`STORAGE_LOCAL_DIR` (a Docker volume in Compose) with path-traversal protection
and atomic write-then-rename. Supabase Storage sits behind the same protocol as a
future adapter; `STORAGE_BACKEND=supabase` fails fast at startup until it exists.

## 12. Environment variables

[`.env.example`](.env.example) documents every value; each has a working default.

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://<user>:<password>@localhost:5432/vyra` | async SQLAlchemy URL |
| `DATABASE_AUTO_CREATE` | `true` | create tables at startup |
| `STORAGE_BACKEND` | `local` | `local` or `supabase` |
| `STORAGE_LOCAL_DIR` | `backend/data/uploads` | local image directory |
| `MAX_UPLOAD_BYTES` | `10485760` | upload size limit (10 MiB) |
| `MODEL_PATH` | `ml/artifacts/vyra-quality-model-v1` | inference bundle directory |
| `REQUIRE_ANALYZER` | `false` (local) / `true` (Docker) | fail startup if the model can't load |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | allowed browser origins |
| `ENVIRONMENT` / `LOG_LEVEL` / `LOG_JSON` | `development` / `INFO` / `false` | app config |
| `VITE_API_BASE_URL` | `""` (same origin) | API origin baked into the frontend build |
| `BACKEND_PORT` / `FRONTEND_PORT` | `8000` / `5173` | host ports for Compose |

No secrets are committed; `.env` is git-ignored; `.env.example` holds placeholders only.

## 13. Running with Docker (recommended)

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| Health | http://localhost:8000/health |

Starts PostgreSQL, the backend (with the model baked in), and nginx serving the
SPA and proxying `/api`. Ports 8000/5173 already in use? Override:
`BACKEND_PORT=8010 FRONTEND_PORT=5183 CORS_ORIGINS=http://localhost:5183 docker compose up --build`.
Stop and wipe volumes: `docker compose down -v`.

**Model loading.** `backend/Dockerfile` (built from the repo root) installs
`vyra_ml` (`pip install --no-deps ./ml`) and copies
`ml/artifacts/vyra-quality-model-v1` to `/app/model`. `app.main:create_app` calls
`load_analyzer` during lifespan startup, which builds `VyraAnalyzer` once from
`MODEL_PATH` and parks it on `app.state`; every request reuses it (inference runs
on a worker thread). `/health` reports `analyzer.status` and
`analyzer_model_version`. `REQUIRE_ANALYZER=true` makes a bad bundle abort
startup. The production image contains only application code, inference
dependencies and the ~6 MB bundle — **not** the datasets, runs or reports (see
[`.dockerignore`](.dockerignore)).

## 14. Running locally without Docker

Python 3.11+, Node 20+, a PostgreSQL instance.

```bash
# backend
cd backend
python -m venv .venv && .venv/Scripts/activate      # or: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install --no-deps ../ml                          # the vyra_ml inference package
export DATABASE_URL="postgresql+asyncpg://<user>:<password>@localhost:5432/vyra"
uvicorn app.main:app --reload --port 8000            # MODEL_PATH defaults to ../ml/artifacts/...

# frontend (separate terminal)
cd frontend
npm install
echo "VITE_API_BASE_URL=http://localhost:8000" > .env.local
npm run dev                                          # http://localhost:5173
```

If the bundle is missing: `cd ml && pip install -r requirements.txt && python scripts/export_inference_bundle.py`.

## 15. Reproducing the ML work

Everything in `ml/` is driven by `configs/experiment.yaml` + a fixed seed — same
inputs ⇒ same dataset ⇒ same metrics.

```bash
cd ml && pip install -r requirements.txt -r requirements-dev.txt
python scripts/pipeline.py all                # build → features → reports → baseline
python scripts/export_inference_bundle.py     # assemble artifacts/vyra-quality-model-v1/
python scripts/build_defect_detector.py       # (re)calibrate the patch defect detector
python scripts/make_demo_samples.py           # regenerate demo/ images
```

See [`ml/README.md`](ml/README.md) for the full pipeline and phase history.

## 16. Demo

Sample images spanning the quality conditions are in [`demo/`](demo/) with a
documented `README.md` (source degradation + observed model behaviour for each).
A 2–3 minute demonstration script is in
[`docs/demo-checklist.md`](docs/demo-checklist.md).

## 17. Tests

```bash
cd backend  && pytest && ruff check . && ruff format --check .      # 34 tests
cd ml       && pytest && ruff check . && ruff format --check .      # 118 tests
cd frontend && npm run lint && npm run typecheck && npm test && npm run build   # 5 tests
docker compose config
```

Requirement → implementation → evidence mapping:
[`docs/assessment-matrix.md`](docs/assessment-matrix.md).
