# VYRA API reference

Base URL: `http://localhost:8000` (Docker Compose maps this to a host port).
Interactive docs: `/docs`. OpenAPI document: `/openapi.json`. All routes are
unauthenticated.

## Error envelope

Every non-2xx response has this shape:

```json
{
  "error": {
    "code": "invalid_image",
    "message": "The uploaded image could not be decoded.",
    "details": { "supported": ["image/jpeg", "image/png", "..."] }
  },
  "request_id": "4f1c9e2a…"
}
```

`details` is present only when it adds information. `request_id` matches the
`X-Request-ID` response header and the server log line; send your own
`X-Request-ID` header to correlate a client-side trace.

| `code` | HTTP | Meaning |
|---|---|---|
| `validation_error` | 422 | Request parameters failed schema validation |
| `invalid_image` | 422 | File is empty, not a recognised image, or could not be decoded |
| `unsupported_media_type` | 415 | Declared content type is not supported |
| `payload_too_large` | 413 | File exceeds `MAX_UPLOAD_BYTES` |
| `not_found` | 404 | No such resource |
| `not_implemented` | 501 | Endpoint exists; no analysis model is loaded (`MODEL_PATH` unset) |
| `storage_error` | 500 | The image store is unavailable |
| `internal_error` | 500 | Unhandled failure (e.g. the analyzer) — nothing is stored or persisted; see the log for the id |

---

## `GET /health`

Service and dependency status. Unversioned. Returns `200` when the database and
storage are reachable, `503` when either is not. The analyzer is reported but
does not affect the overall status.

```json
{
  "status": "ok",
  "version": "0.1.0",
  "environment": "production",
  "uptime_seconds": 42.5,
  "analyzer_model_version": "vyra-quality-model-v1",
  "components": {
    "database": { "status": "ok", "detail": null, "latency_ms": 1.4 },
    "storage": { "status": "ok", "detail": null, "latency_ms": null },
    "analyzer": { "status": "ok", "detail": "model vyra-quality-model-v1", "latency_ms": null }
  }
}
```

`analyzer.status` is `ok` when a model is loaded, `not_configured` when
`MODEL_PATH` is unset. Component status values: `ok`, `unavailable`,
`not_configured`.

---

## `GET /metrics`

Process-level runtime metrics as JSON. Unversioned, no database dependency (it
answers during an outage). Counters are per worker process and reset on restart.

```json
{
  "service": "VYRA",
  "version": "0.1.0",
  "environment": "production",
  "uptime_seconds": 812.4,
  "requests_total": 143,
  "requests_in_flight": 1,
  "requests_by_status_class": { "2xx": 130, "4xx": 11, "5xx": 2 },
  "error_rate": 0.014,
  "latency_ms": { "window": 143, "p50": 34.1, "p95": 890.2, "p99": 1503.7, "max": 1902.0 }
}
```

`error_rate` is `5xx / requests_total`. `latency_ms` percentiles are over a
rolling window of the last 2048 requests.

---

## `POST /api/v1/analyses`

Uploads an image, analyzes it, stores it and persists the result.

Request: `multipart/form-data` with a single `file` part. Supported formats:
JPEG, PNG, WebP, BMP, TIFF (detected from magic bytes, not the declared type).
Maximum size: `MAX_UPLOAD_BYTES` (10 MiB default).

```bash
curl -i -X POST http://localhost:8000/api/v1/analyses -F "file=@sample.jpg"
```

Pipeline: **validate → analyze → store → persist**. If validation fails you get
`413`/`415`/`422` and nothing is written. If the analyzer cannot decode the image
you get `422`. Any other analyzer failure is `500` with no stored blob and no
row. On success: `201` with the `Analysis` object.

If no model is loaded (`MODEL_PATH` unset), a validated upload returns `501`.

### Example `201` response

```json
{
  "id": "a7f65b15-6b4b-4f31-94d2-b864fb3a8909",
  "created_at": "2026-08-28T11:51:46.844447Z",
  "status": "completed",
  "image": {
    "filename": "02_blur.jpg",
    "content_type": "image/jpeg",
    "size_bytes": 41720,
    "width": 481,
    "height": 321
  },
  "quality_score": 68.2,
  "quality_label": "ACCEPTABLE",
  "model_version": "vyra-quality-model-v1",
  "issues": [
    {
      "type": "blur",
      "severity": "medium",
      "confidence": 0.6723,
      "validation": "real-world",
      "detail": "Validated on real images (VizWiz F1 0.6314)."
    }
  ],
  "metrics": {
    "sharpness": 56.42, "brightness": 0.365, "contrast": 0.151,
    "noise_sigma": 0.827, "saturation": 0.184, "colourfulness": 23.19,
    "blockiness": 0.0007, "edge_density": 0.0216,
    "dark_clip_ratio": 0.0, "bright_clip_ratio": 0.0
  },
  "explanation": {
    "summary": "Quality score 68/100 (ACCEPTABLE). Flagged: blur.",
    "evidence": [
      { "feature": "sharp_highfreq_ratio", "value": 0.2624, "direction": "lower_sharpness_supports_blur" }
    ],
    "issue_probabilities": { "blur": 0.6723, "underexposure": 0.08, "overexposure": 0.01, "noise": 0.12, "corruption": 0.03 },
    "potential_defect": { "probability": 0.11, "flagged": false, "region": null, "evidence": [], "note": "Screening signal only: ..." },
    "capabilities": {
      "real_world_validated": ["blur", "underexposure", "overexposure"],
      "synthetic_validated_only": ["noise", "corruption"],
      "screening_only": ["potential_visual_defect"]
    },
    "feature_version": "cvfeat-v2",
    "timings_ms": { "features": 747, "issue_models": 570, "defect": 573, "total": 1890 }
  },
  "error_message": null
}
```

**`confidence` vs `severity`.** `confidence` is the calibrated probability the
issue is present. `severity` (`low`/`medium`/`high`) is a bucketed estimate of
*how far* past the decision threshold the probability sits — the estimated
impact, not the certainty.

**`validation`** on each issue: `real-world` (blur, under/overexposure —
evaluated on VizWiz), `synthetic-only` (noise, corruption — no real evaluation
exists), `screening` (the defect signal). Training/synthetic F1 is **not**
real-world performance.

---

## `POST /api/v1/analyses/batch`

Analyse several images in one request. `multipart/form-data` with a repeated
`files` part (up to `MAX_BATCH_SIZE`, default 10). Each image runs the same
**validate → analyze → store → persist** pipeline independently.

```bash
curl -s -X POST http://localhost:8000/api/v1/analyses/batch \
  -F "files=@a.jpg" -F "files=@b.png" -F "files=@broken.jpg"
```

Always returns `200` — a per-image failure is reported in `items`, not raised.
`413` (`payload_too_large`) if more than `MAX_BATCH_SIZE` files are sent; `501`
(`not_implemented`) if no model is loaded.

```json
{
  "total": 3,
  "succeeded": 2,
  "failed": 1,
  "items": [
    { "filename": "a.jpg", "ok": true, "analysis": { /* Analysis object */ }, "error": null },
    { "filename": "b.png", "ok": true, "analysis": { /* Analysis object */ }, "error": null },
    { "filename": "broken.jpg", "ok": false, "analysis": null,
      "error": { "code": "invalid_image", "message": "The uploaded file could not be decoded." } }
  ]
}
```

Successful items are persisted and appear in the history endpoints like any
single upload. `error.code` is the same vocabulary as the error envelope
(`invalid_image`, `unsupported_media_type`, `payload_too_large`, …).

---

## `GET /api/v1/analyses`

Previous analyses, newest first.

| Query | Type | Default | Range |
|---|---|---|---|
| `limit` | int | 20 | 1–100 |
| `offset` | int | 0 | ≥ 0 |

```json
{ "items": [ /* Analysis objects */ ], "total": 3, "limit": 20, "offset": 0 }
```

## `GET /api/v1/analyses/{id}`

One analysis by UUID. `404` (`not_found`) if unknown, `422` (`validation_error`)
if the id is not a UUID.

---

## Enumerations

| Field | Values |
|---|---|
| `status` | `completed`, `failed` |
| `quality_label` | `GOOD` (≥85), `ACCEPTABLE` (68–84), `DEGRADED` (45–67), `POOR` (<45) |
| `quality_score` | 0–100 (operational, not perceptual — see `ml/docs/quality-score.md`) |
| `issues[].type` | `blur`, `underexposure`, `overexposure`, `noise`, `corruption`, `defect` |
| `issues[].severity` | `low`, `medium`, `high` |
| `issues[].validation` | `real-world`, `synthetic-only`, `screening` |
| `issues[].confidence` | 0.0–1.0 |
