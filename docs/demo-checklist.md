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

1. **Open** http://localhost:5173 — hero loads; the nav status pill shows `vyra-quality-model-v1` (analyzer loaded). Flip the **theme toggle** (light ⇄ dark) once — it persists.
2. **Scroll to the Workspace** (or hit "Analyze an image"). **Upload `demo/02_blur.jpg`** (drag onto the drop-tray), click **Analyze image**.
3. **Quality score** — gauge shows ~53 / 100, label **DEGRADED**.
4. **Detected issues** — `Blur`, `high severity`, `~83% confidence`, badge **real-world** ("Validated on real images, VizWiz F1 0.63").
5. **Confidence vs severity** — point out these are different: confidence is the calibrated probability, severity is the estimated impact bucket.
6. **Image statistics** — sharpness / brightness / contrast / noise / saturation etc.
7. **Explanation** — "why flagged": `sharp_highfreq_ratio`, `sharp_laplacian_var`, `texture_spectral_slope` with values.
8. **Exposure** — upload `demo/03_underexposed.jpg` → `underexposure` flagged (real-world), and `demo/04_overexposed.jpg` → `overexposure` flagged. Both are trained on real "too dark" / "too bright" VizWiz photos.
9. **Defect region** — upload `demo/08_defect_blotch.jpg`; the patch detector fires and a red box highlights the region. Note it is *screening only* and misses ~2/3 of defects by design.
11. **Capability tiers** — upload `demo/05_noisy.jpg`; `noise` is flagged with a **synthetic-only** badge (no real-world validation), unlike `blur`.
12. **History** — the sidebar lists every analysis (score chip, filename, time, label); click one to re-view its full result.
13. **Invalid uploads** — try a `.txt` renamed to `.jpg` → **422** with a friendly message; a `.pdf` → **415**; an empty file → **422**. Confirm none of them appear in history (invalid files are never persisted).
14. **Scroll on** — *How it works* (the 8-step pipeline), *Under the hood* (model card + the three capability tiers), *Honest metrics* (synthetic vs real-world table + the domain-gap / operational-score / defect-advisory disclaimers).

## Talking points

- Classical RF over 42 interpretable CV features — no CNN, no external APIs.
- blur / underexposure / overexposure heads are **trained on real VizWiz photos**
  with real crowd labels (Phase 3D); noise / corruption on leakage-safe synthetic
  degradations of clean BSDS500 photos.
- **Evaluated once on a held-out real set** (VizWiz `val`): real primary macro-F1
  **0.43 → 0.54**. blur 0.63, underexposure 0.63, overexposure 0.36 (still weak,
  ROC-AUC 0.92); noise / corruption have no real evaluation; defect is advisory.
- Thresholds + isotonic calibration fitted on cross-validated out-of-fold
  predictions of a **disjoint real sample**; the eval set is never iterated against.
- Quality score is **operational, not perceptual** — deterministic function of
  calibrated probabilities.
