# CV feature engine (`cvfeat-v2`)

42 interpretable scalar features per image. The authoritative list, with a
one-line definition each, is `FEATURE_DESCRIPTIONS` in
`vyra_ml/features/extract.py`; this document covers rationale and the
quality-check results.

**`cvfeat-v2` (Phase 3B).** Only `compress_blockiness{,_h,_v}` changed: from an
unbounded ratio `boundary/(interior+eps)` (which blew up to mean 800 / std
21,700 on synthetic blur samples — see `docs/phase3b-calibration.md`) to a
bounded normalised excess `(boundary-interior)/(boundary+interior+eps)` in
`[0,1)`. Monotone in the old ratio, so image ordering is preserved; the feature
set, count and ordering are otherwise identical to v1.

All features are computed on the image resized so its **longest edge is 384 px**
(`features.work_long_edge`). Fixing the working resolution is what makes
resolution-dependent measures (Laplacian variance, gradient stats) comparable
across source images of different sizes.

## Groups

| Group | Count | Features (prefix) | Why |
|---|---|---|---|
| Sharpness | 8 | `sharp_` | Laplacian variance (+ brightness-normalised), Tenengrad, gradient mean/p90, modified Laplacian, FFT high-freq ratio, Canny edge density. Blur suppresses all of these. |
| Exposure | 8 | `expo_` | Mean/median luma, dark/bright clip ratios, shadow/highlight ratios, histogram entropy, luma skew. Locate the luminance mass and measure clipping at both ends. |
| Contrast | 7 | `contrast_` | Std, dynamic range (p95-p5), Michelson (robust percentiles), coefficient of variation, mean local std, p5, p95. |
| Noise | 4 | `noise_` | Immerkaer sigma, high-freq residual std, median-residual MAD, flat-region std (only the smoothest 15% of tiles, to suppress texture). Four estimators because no single no-reference noise measure is robust to texture. |
| Colour | 7 | `color_` | Saturation mean/std, Hasler-Suesstrunk colourfulness, colour cast (abs + ratio), grey-pixel ratio, mean channel std. |
| Texture | 5 | `texture_` | Radial power-spectrum slope (blur steepens, noise flattens), GLCM contrast/homogeneity/energy, uniform-LBP entropy. |
| Compression | 3 | `compress_` | 8-pixel-grid blockiness (H, V, mean): normalised excess of gradient across JPEG block boundaries over the interior, in [0,1). |

## Feature quality check

Run: `python scripts/pipeline.py feature-report` -> `reports/<version>/feature_report.{json,csv}`.
Results on the 3600-sample build:

- **NaN / infinite values: 0 / 0.** All features finite on every sample,
  including the black / white / flat-grey / pure-noise edge cases
  (`tests/test_features.py::test_features_are_finite_on_edge_case_images`).
- **Near-constant features: none.**
- **Highly correlated pairs (|Spearman| >= 0.97):** 9 pairs, mostly within the
  sharpness/noise-residual family (e.g. `sharp_laplacian_var` vs
  `noise_highfreq_residual_std` at 0.985 -- both driven by high-frequency
  content). Kept for the baseline: Random Forest is unaffected by redundancy and
  the pairs still carry different information off the diagonal. Candidate for
  pruning / PCA if a linear or distance-based model is tried later.
- **Size robustness:** median relative change between extraction at 288 px and
  512 px is **5%**. A few absolute-scale features (`sharp_laplacian_var`,
  `noise_immerkaer_sigma`) drift up to ~60% across that range -- expected, since
  they scale with sampling density. Mitigated by fixing the working resolution
  everywhere; a future version could normalise them explicitly.
- **Cost:** ~70 ms/image single-threaded (dominated by two FFTs and the GLCM).

## Explainability

Per-issue Random Forest `feature_importances_` (top 8, in every run's
`metrics.json`) are directly interpretable and line up with the physics:

| Issue | Top features |
|---|---|
| blur | `sharp_highfreq_ratio`, `sharp_laplacian_var`, `texture_spectral_slope` |
| underexposure | `expo_highlight_ratio` (low), `contrast_p95`, `expo_luma_mean` |
| overexposure | `contrast_p95`, `expo_luma_mean`, `expo_bright_clip_ratio` |
| noise | `noise_immerkaer_sigma`, `noise_median_residual_mad` |
| corruption | `compress_blockiness{,_v,_h}` (0.29 + 0.25 + 0.22 importance) |
| defect | *no dominant feature* -- max importance 0.035 |

The `defect` row is the key negative result: global features do not localise a
small regional artefact, which is why that issue's F1 is ~0.46 while the others
are 0.77-0.94.
