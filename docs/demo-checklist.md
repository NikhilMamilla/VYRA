# VYRA demo checklist (2–3 minutes)

Sample images: [`demo/`](../demo/) (`01_clean` … `08_defect_blotch`, each with a
documented expected behaviour in `demo/README.md`).

## Setup

```bash
docker compose up --build            # ~1 min after images are built
# ports 8000/5173 in use? add: BACKEND_PORT=8010 FRONTEND_PORT=5183 CORS_ORIGINS=http://localhost:5183
```

Wait for `curl -s localhost:8000/health` to return `"status": "ok"` with
`"analyzer": {"status": "ok"}`.

## Sequence

1. **Open** http://localhost:5173 — the header badge shows `model vyra-quality-model-v1` (analyzer loaded).
2. **Upload `demo/02_blur.jpg`** (drag onto the dropzone), click **Analyze image**.
3. **Quality score** — gauge shows ~68 / 100, label **ACCEPTABLE**.
4. **Detected issues** — `Blur`, `medium severity`, `~67% confidence`, badge **real-world**.
5. **Confidence vs severity** — point out these are different: confidence is the calibrated probability, severity is the estimated impact bucket.
6. **Image statistics** — sharpness / brightness / contrast / noise / saturation etc.
7. **Explanation** — "why flagged": `sharp_highfreq_ratio`, `sharp_laplacian_var`, `texture_spectral_slope` with values.
8. **Defect region** — upload `demo/08_defect_blotch.jpg`; if the patch detector fires, a red box highlights the region. Note it is *screening only* and misses ~2/3 of defects by design.
9. **Capability tiers** — upload `demo/05_noisy.jpg`; `noise` is flagged with a **synthetic-only** badge (no real-world validation), unlike `blur`.
10. **History** — the sidebar lists every analysis (score chip, filename, time, issue summary); click one to re-view its full result.
11. **Invalid uploads** — try a `.txt` renamed to `.jpg` → **422** with a friendly message; a `.pdf` → **415**; an empty file → **422**. Confirm none of them appear in history (invalid files are never persisted).

## Talking points

- Classical RF over 42 interpretable CV features — no CNN, no external APIs.
- Trained on synthetic degradations of clean BSDS500 photos; **leakage-safe**
  split on the original image.
- **Validated on real images** (VizWiz): blur / underexposure hold up
  (F1 0.61 / 0.49), overexposure is weak (0.19), noise / corruption have no real
  evaluation, defect is advisory.
- Thresholds + isotonic calibration fitted on a **separate real validation
  split**; final numbers measured once on an untouched real test set.
- Quality score is **operational, not perceptual** — deterministic function of
  calibrated probabilities.
