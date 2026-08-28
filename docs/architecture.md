# VYRA architecture

The Phase 1 foundation and the decisions behind it, updated for the Phase 3C
end-to-end integration.

## Request flow

```
Browser
  │  multipart/form-data
  ▼
FastAPI route            app/api/v1/routes/analyses.py
  │  bounded read (rejects oversized uploads without buffering them whole)
  ▼
AnalysisService.create_analysis          app/services/analysis_service.py
  │
  ├─▶ validate_upload    magic-byte sniff, size, media type    → 413 / 415 / 422
  ├─▶ QualityAnalyzer.analyze   CV features → RF → calibrate → threshold → score
  │        │  runs on a worker thread; InvalidImageError → 422, else 500
  │        ▼   (nothing stored yet — an un-analyzable image never touches storage)
  ├─▶ ObjectStorage.save    <analysis_id>/original.<ext>
  ├─▶ AnalysisRepository.add    one row, status="completed"
  │        └─ on failure: delete the blob just written, re-raise  → 500
  ▼
201  full AnalysisRead (score, label, issues, statistics, explanation)
```

Order matters: **analyze → store → persist**. An image the analyzer cannot read
is rejected before anything is written; if the database insert fails after the
blob is stored, the blob is removed. A failed request never leaves an orphan
file or a partial row.

## The analyzer

`app/analysis/vyra_analyzer.py::VyraAnalyzer` implements the `QualityAnalyzer`
protocol around `vyra_ml.inference.VyraQualityModel`. It is constructed once at
startup by `app/analysis/registry.py::load_analyzer` from `MODEL_PATH` (a
self-describing bundle directory) and parked on `app.state`. `REQUIRE_ANALYZER`
controls whether a load failure aborts startup (true in Docker) or degrades to a
no-analysis API (`/health` says so, POSTs get 501).

Inference per request (~1.5 s, synchronous, on a thread): resize to 384 px long
edge → 42 `cvfeat-v2` features → 6 one-vs-rest RandomForests (blur / under / over
exposure trained on real VizWiz photos, noise / corruption on synthetic data) →
isotonic calibration (blur / under / over exposure) → per-issue real-validation
thresholds → operational quality score → patch-anomaly defect pass. The bundle
(`bundle.json`) pins every threshold, the calibration, the feature version, the
score formula and the training/eval provenance, so nothing is hard-coded in the
backend and the configuration cannot drift.

## Layers

| Layer          | Package             | Responsibility                                            |
| -------------- | ------------------- | --------------------------------------------------------- |
| HTTP           | `app/api`           | Routing, dependency wiring, status codes. No business logic. |
| Orchestration  | `app/services`      | The pipeline, and upload validation                       |
| CV / ML        | `app/analysis`      | The analyzer contract, `VyraAnalyzer`, and how the bundle is loaded |
| Persistence    | `app/repositories`, `app/db` | Queries and ORM models                            |
| Storage        | `app/storage`       | Blob read/write behind a protocol                          |
| Contracts      | `app/schemas`       | Pydantic models shared by API and analyzer                 |
| Cross-cutting  | `app/core`          | Settings, errors, logging, middleware                      |

Dependencies point inwards: the analyzer knows nothing about HTTP, storage or
the database; storage knows nothing about analysis.

## Composition root

`app/main.py:create_app` builds the engine, session factory, storage backend and
analyzer during the lifespan startup and parks them on `app.state`. No module
performs I/O at import time, and `create_app(settings)` constructs an entire
independent application — which is exactly how the test suite runs it against
SQLite and a temporary directory.

## Key decisions

### Local PostgreSQL by default; Supabase supported, not required

The assessment requires the application to run outside the developer's machine
and states that API keys are not required. Making Supabase mandatory would force
a reviewer to create a cloud project before anything ran, and would tempt us into
committing credentials.

Instead the backend talks to a single `DATABASE_URL` through SQLAlchemy.
`docker compose up` starts a local Postgres container so the stack works with
zero configuration; pointing `DATABASE_URL` at Supabase Postgres is a one-line
change with no code impact, because Supabase Postgres *is* Postgres. The
Supabase client SDK is deliberately not a dependency — it would couple the
application to one vendor for no gain.

Supabase Storage is genuinely different from a local filesystem, so it sits
behind the `ObjectStorage` protocol as a future adapter. `STORAGE_BACKEND=supabase`
currently fails fast at startup rather than at the first upload.

### JSON column for the analysis result

`analyses` promotes to real columns only the fields we already know we will
list, sort or filter by (`created_at`, `status`, `quality_score`,
`quality_label`, image metadata). The detected issues, per-metric statistics and
explainability payload live in one JSON column (`JSONB` on Postgres).

Normalising the issue/metric/explanation payload would mean a migration every
time the model's output changes; a JSON column keeps that shape flexible while
the queryable fields stay typed and indexed.

### Analyzer resolved at startup, never faked

`app/analysis/registry.py::load_analyzer` builds the real `VyraAnalyzer` from the
bundle when `MODEL_PATH` is set, or returns `None` when it is not. That single
fact propagates honestly:

- model loaded → `/health` reports `analyzer: ok` with `analyzer_model_version`;
- `MODEL_PATH` unset → `analyzer: not_configured`, `200` health, `POST` gets
  `501` after validating the upload;
- `MODEL_PATH` set but broken → startup aborts if `REQUIRE_ANALYZER=true`
  (Docker default), otherwise logs and degrades.

No placeholder scores are generated anywhere.

### The model bundle is the source of truth

The backend imports `vyra_ml` but hard-codes nothing about the model. Thresholds,
calibration, the feature version, label definitions, the quality-score formula
and weights, and the training/evaluation provenance all live in
`ml/artifacts/vyra-quality-model-v1/bundle.json`. Swapping models is a bundle
swap. See [`ml/docs/quality-score.md`](../ml/docs/quality-score.md) and
[`ml/docs/defect.md`](../ml/docs/defect.md).

### One error envelope

Every failure — application error, request-validation error, unhandled exception
— is rendered by `app/core/errors.py` into `{"error": {"code", "message",
"details"}, "request_id"}`. The frontend's `ApiError` parses exactly that shape,
so the UI branches on a stable `code` rather than on message text. Unhandled
exceptions are logged with the request id and returned as a generic 500 so no
internals leak.

### Single origin in production

The frontend container is nginx: it serves the built SPA and proxies `/api` to
the backend. The browser therefore talks to one origin, CORS is not needed in
the deployed stack, and no backend URL is baked into the bundle. CORS is still
configured for local development, where Vite runs on port 5173 and the backend
on 8000.

## Testing strategy

`backend/tests` runs against SQLite (`aiosqlite`) and a temporary directory, so
`pytest` requires no Docker, no Postgres and no credentials. The ORM uses
portable types (`Uuid`, `JSON` with a `JSONB` variant) to make that possible
without diverging from production behaviour. Tests drive the ASGI app through
`httpx`, including the lifespan, so dependency wiring is covered too. The
analyzer-pipeline tests load the real model bundle and skip cleanly if it is
absent. 34 backend tests, 118 ML tests, 5 frontend tests.

## Docker: one image carries the API + inference + model

`backend/Dockerfile` is built from the **repo root** so it can `COPY ml/vyra_ml`
(installed `--no-deps`) and `COPY ml/artifacts/vyra-quality-model-v1` alongside
`backend/app`. `.dockerignore` keeps the datasets, runs and reports out — the
production image is application code + inference deps + the ~6 MB bundle. The
frontend image is unchanged (nginx serving the built SPA, proxying `/api`).

## Still not done

- **Alembic migrations** — tables are still auto-created at startup
  (`DATABASE_AUTO_CREATE`).
- **Supabase Storage adapter** — `STORAGE_BACKEND=supabase` is rejected at
  startup.
- **Auth / rate limiting** — the API is unauthenticated (out of scope for the
  assessment).
- **Async inference** — inference is synchronous (~1.5 s); a queue would only be
  warranted under real load.
