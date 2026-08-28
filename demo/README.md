# VYRA demo images

Clean BSDS500 photos with the project's synthetic degradation classes applied
(`ml/scripts/make_demo_samples.py`). blur / underexposure / overexposure are
**trained and validated on real VizWiz photos**; noise / corruption are
synthetic-validated only; the defect signal is a weak screening cue
(ROC-AUC 0.60, ~1 in 3 flags real).

Because the exposure heads now learn what real annotators call "too dark" /
"too bright" (genuinely extreme frames), `03` and `07` get an extra exposure
cut past the synthetic severity range so they land an unambiguous positive.

Observed behaviour of `vyra-quality-model-v1` on this exact set — an honest snapshot,
not a target. Overexposure still has lower recall on real photos (F1 0.36);
`06` shows the synthetic-only corruption head failing to transfer (the
re-encoded frame reads as mild blur instead); the patch-anomaly defect cue
also fires on the large black / blown regions in `03`, `04` and `07`.

| file | score / label | flagged |
|---|---|---|
| `01_clean.jpg` | 100 GOOD | — |
| `02_blur.jpg` | 45 DEGRADED | blur 0.96 |
| `03_underexposed.jpg` | 46 DEGRADED | underexposure 1.00, potential_defect 0.84 |
| `04_overexposed.jpg` | 56 DEGRADED | overexposure 0.81, potential_defect 1.00 |
| `05_noisy.jpg` | 68 ACCEPTABLE | blur 0.52, noise 0.82 |
| `06_compressed.jpg` | 80 ACCEPTABLE | blur 0.61 |
| `07_multi_blur_dark.jpg` | 45 POOR | blur 0.61, underexposure 0.80, potential_defect 0.86 |
| `08_defect_blotch.jpg` | 80 ACCEPTABLE | potential_defect 0.91 |

## Source degradations applied

| file | contents |
|---|---|
| `01_clean.jpg` | clean (GOOD — no issue flagged) |
| `02_blur.jpg` | blur sev4 (blur flagged (real-world validated)) |
| `03_underexposed.jpg` | exposure ×0.16 (underexposure flagged (real-world validated)) |
| `04_overexposed.jpg` | overexposure sev5 + overexposure sev5 (overexposure flagged (real-world validated)) |
| `05_noisy.jpg` | noise sev4 (noise flagged (synthetic-validated only)) |
| `06_compressed.jpg` | corruption sev5 (corruption flagged (synthetic-validated only)) |
| `07_multi_blur_dark.jpg` | blur sev3 + exposure ×0.22 (blur + underexposure, lower quality score) |
| `08_defect_blotch.jpg` | defect sev5 (potential visual defect may be flagged (screening only, ~33% precision)) |
