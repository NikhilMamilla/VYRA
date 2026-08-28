# VYRA demo images

Clean BSDS500 photos with the project's own synthetic degradations applied
(`ml/scripts/make_demo_samples.py`). blur / underexposure / overexposure are
real-world validated; noise / corruption are synthetic-validated only; the
defect signal is a weak screening cue (ROC-AUC 0.60, ~1 in 3 flags real).

Observed behaviour of `vyra-quality-model-v1` on this exact set — an honest
snapshot, not a target. The overexposure detector is weak (real F1 0.19),
corruption at this severity is not always caught, and the defect detector
misses ~2/3 of defects by design:

| file | score / label | flagged |
|---|---|---|
| `01_clean.jpg` | 100 GOOD | — |
| `02_blur.jpg` | 68 ACCEPTABLE | blur 0.67 |
| `03_underexposed.jpg` | 66 DEGRADED | underexposure 0.80, blur 0.37 |
| `04_overexposed.jpg` | 99 GOOD | overexposure 0.12 (just over threshold) |
| `05_noisy.jpg` | 84 ACCEPTABLE | noise 0.68, underexposure 0.50 |
| `06_compressed.jpg` | 88 GOOD | blur 0.48 (corruption not caught here) |
| `07_multi_blur_dark.jpg` | 45 DEGRADED | blur 0.94 |
| `08_defect_blotch.jpg` | 100 GOOD | — (defect missed — expected ~2/3 of the time) |

## Source degradations applied

| file | contents |
|---|---|
| `01_clean.jpg` | clean (GOOD — no issue flagged) |
| `02_blur.jpg` | blur sev4 (blur flagged (real-world validated)) |
| `03_underexposed.jpg` | underexposure sev4 (underexposure flagged) |
| `04_overexposed.jpg` | overexposure sev4 (overexposure flagged (weak detector)) |
| `05_noisy.jpg` | noise sev4 (noise flagged (synthetic-validated only)) |
| `06_compressed.jpg` | corruption sev4 (corruption flagged (synthetic-validated only)) |
| `07_multi_blur_dark.jpg` | underexposure sev3 + blur sev3 (blur + underexposure, lower quality score) |
| `08_defect_blotch.jpg` | defect sev4 (potential visual defect may be flagged (screening only, ~33% precision)) |
