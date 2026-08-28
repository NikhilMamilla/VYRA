# Phase 3B — feature fix, synthetic-blur realism, real-validation calibration

*Results below are filled verbatim from `runs/phase3b-calibration-v1/` and
`reports/phase3b-calibration-v1/` (orchestrator completed 2026-08-28).*

> Experimental improvement phase. **No backend integration, no frontend change,
> no real-data fine-tuning, no final defect model, no final quality score.**
> Every intervention is a separately versioned artifact; Phase 3A is untouched.

Question: **how much of the Phase 3A real-world gap (macro-F1 0.74 synthetic →
0.30 real) can be recovered by feature correctness + more realistic synthetic
data + calibration, without fine-tuning on real images?**

## Experiments

| # | ID | Parent | What |
|---|---|---|---|
| 0 | `phase3a-real-world-v1` | — | preserved unchanged (baseline) |
| 1 | `cvfeat-v2` | `cvfeat-v1` | fix `compress_blockiness` numerical blow-up |
| 2 | `phase3b-dataset-blurnoise-v1` | `phase2-baseline-v1` | add post-blur sensor noise to blur samples |
| 3 | `phase3b-realval-v1` | — | real **validation** split from VizWiz **train** (dev only) |
| 4 | `phase3b-thr-*` | per model | threshold selection on real-val (max F1) |
| 5 | `phase3b-cal-*` | per model | isotonic probability calibration on real-val |
| 6 | `phase3b-calibration-v1` | all | one final pass on the untouched VizWiz **val** eval set |

Models compared: `phase3a_v1` (frozen Phase 3A), `v2fix` (cvfeat-v2, same data,
retrained), `v2blur` (cvfeat-v2 + blur-noise dataset, retrained).

## 1. Feature fix — `compress_blockiness` (cvfeat-v1 → cvfeat-v2)

**What was wrong.** v1: `blockiness = boundary_grad_mean / (interior_grad_mean +
1e-8)`. A heavily blurred frame stored by the dataset builder as JPEG q97 has a
near-zero interior gradient but a faint stored-JPEG grid on the 8-px boundaries,
so the ratio diverged. Phase 3A measured mean **800**, std **21,700** on
synthetic blur samples — a pathological tail unlike anything in real data.

**Fix (no conceptual change to what it measures).** v2 uses the bounded
normalised-excess form

    blockiness = (boundary_mean - interior_mean) / (boundary_mean + interior_mean + 1e-6)

in `[0, 1)`. 0 = the block grid carries no more gradient than the interior
(clean / cleanly textured); →1 = the grid dominates what little structure the
image has. It is monotone in the old ratio, so image-to-image ordering is
preserved, but it can never diverge. Only these 3 of 42 features changed; the
feature set, ordering and count are identical, so it is a minor version bump.

**Regression tests** (`tests/test_compression_feature.py`): flat / near-flat /
black / white / textured / heavy-JPEG images all give finite values in `[0, 1)`;
heavy JPEG scores above light JPEG; the blurred-flat + JPEG case that broke v1
stays bounded.

**Before/after on the synthetic feature table** (3600 samples,
`reports/phase3b-calibration-v1/feature_fix.json`):

| stat | `compress_blockiness` v1 | v2 | `_h` v1 | v2 | `_v` v1 | v2 |
|---|---|---|---|---|---|---|
| min | 0.45 | -0.380 | 0.41 | -0.418 | 0.49 | -0.343 |
| max | **590,020** | **0.99983** | 574,937 | 0.99983 | 605,102 | 0.99983 |
| mean | 165.2 | 0.0611 | 161.0 | 0.0657 | 169.3 | 0.0565 |
| std | **9,834** | **0.148** | 9,582 | 0.156 | 10,085 | 0.145 |

The v1 tail (max ~590k, std ~9.8k) is gone. v2 is finite everywhere (0 NaN, 0
inf on all 3600 rows) and bounded. Note the realised v2 range is `(-1, 1)`, not
`[0, 1)` as first documented: when a near-flat blurred interior carries a hair
*more* residual gradient than the block boundary the excess goes slightly
negative (min observed -0.42). This is harmless — the property that matters is
*bounded and finite*, and image-to-image ordering is preserved. Corruption
feature importance is unchanged (`compress_blockiness{,_v,_h}` still the top 3
for that issue at 0.29 / 0.25 / 0.22).

## 2. Synthetic blur realism

**Why.** Phase 3A failure analysis: real blurry photos keep sensor noise and
surface texture (a blurry pill bottle on a granite counter was scored *sharp*
because the counter speckle kept the sharpness features high). Our synthetic
blur applies Gaussian/motion blur to a clean image and removes *all* high
frequency, because — unlike a real camera — no sensor noise is added afterward.
`domain_shift.json`: `sharp_laplacian_var` was 1674 for synthetic blur vs **162**
for real blur.

**Change.** New config flag `degradation.post_blur_sensor_noise` (default off, so
`phase2-baseline-v1` rebuilds identically). When on, after all degradations are
applied, any sample whose degradation list contains `blur` gets a light
zero-mean Gaussian read-noise pass: luma sigma sampled uniformly from
**1.5–4.0 / 255**, chroma sigma at half that, seeded by
`derive_rng(seed, "sensor_noise", sample_id)`. The sigma range sits *below* the
`noise` degradation's severity-1 range (3–7 / 255) on purpose — this is
image-formation realism, not a class cue, and blur must not become trivially
separable from clean or from `noise`.

**Scope.** Affects blur-containing samples only (single `blur` + multi
combinations that include blur); clean / exposure-only / noise-only / corruption
samples are untouched.

**Scale:** `phase3b-dataset-blurnoise-v1` = 3600 samples (same 400 originals,
same seed). **738** contain `blur` and received the sensor-noise pass; the other
2862 are byte-identical to `phase2-baseline-v1`. Retrained model `v2blur`
synthetic test macro-F1 **0.784** (vs `v2fix` 0.798) — the added noise slightly
hurt even in-domain.

## 3. Real validation vs real evaluation

| | source | purpose | may influence |
|---|---|---|---|
| **real validation** (`phase3b-realval-v1`) | seeded uniform sample of VizWiz **train** (`VizWiz_train_*.jpg`) | threshold selection, probability calibration | thresholds, calibrators |
| **real evaluation** (`phase3a-real-world-v1`) | seeded uniform sample of VizWiz **val** (`VizWiz_val_*.jpg`) | final measurement only | **nothing** — read once, at the end |

Disjoint by construction (train vs val filename namespaces). `leakage_check`
enforces: 0 image-id overlap and 0 SHA-1 overlap between the two real splits, and
0 SHA-1 overlap of either against the synthetic training set.

**One repair was needed.** VizWiz reuses **9** images across its own train and
val splits under different filenames (identical SHA-1, 0 id overlap). These 9
were dropped from the *validation* split (2498 → **2489**); the Phase 3A
evaluation set was left untouched. Synthetic-train leakage into either real split
remains a hard failure and did not occur (0 / 0).

**Label counts at ≥3 of 5 votes** (`vote_threshold_labels`):

| split | n | blur | underexposure | overexposure | defect (OBS proxy) |
|---|---|---|---|---|---|
| real validation (VizWiz train) | 2489 | 754 (30.3%) | 69 (2.8%) | 49 (2.0%) | 50 (2.0%) |
| real evaluation (VizWiz val) | 2496 | 732 (29.3%) | 69 (2.8%) | 66 (2.6%) | 44 (1.8%) |

Only blur has a comfortable positive count; the exposure and defect issues sit
near 2%, so their real-world F1 estimates carry wide error bars. `noise` and
`corruption` have no VizWiz mapping and are absent from both splits.

## 4. Thresholds — synthetic-val vs real-val

Criterion: **maximise F1** on the real validation split, tie-break toward higher
precision. Issues with < 20 real-val positives fall back to 0.5. Thresholds are
versioned JSON under `runs/phase3b-calibration-v1/thresholds/`.

| label | Phase 3A (synthetic-val) | real-val F1, raw prob (v2fix) | real-val F1, calibrated prob (v2fix) |
|---|---|---|---|
| blur | 0.45 | 0.42 | 0.36 |
| overexposure | 0.45 | **0.66** | **0.10** |
| underexposure | 0.55 | 0.52 | 0.50 |
| defect | 0.30 | 0.06 | 0.02 |

The big move is **overexposure**: the synthetic threshold (0.45) fired on far too
many real "bright" photos. `noise` and `corruption` keep their synthetic
thresholds (0.45 / 0.35) — VizWiz cannot tune them. The `defect` threshold
collapses toward 0 (predict-all) because real defect is barely separable; it is
**not used** in the shipped model (see Defect).

## 5. Probability calibration

Method: **isotonic regression**, one calibrator per issue, fitted on real-val
scores. Kept for an issue only if (a) real-val support ≥ 40 and (b) it does not
worsen the Brier score on the fit split — otherwise the transform is the
identity and that is recorded. Diagnostics per issue: Brier and expected
calibration error before/after, plus a reliability curve.

Brier / ECE on the real-validation fit split, `v2fix` (`calibration.json`):

| label | support | Brier before → after | ECE before → after |
|---|---|---|---|
| blur | 754 | 0.181 → 0.154 | 0.128 → 0.00 |
| overexposure | 49 | 0.106 → 0.019 | 0.248 → 0.00 |
| underexposure | 69 | 0.022 → 0.015 | 0.048 → 0.00 |
| defect | 50 | 0.104 → 0.020 | 0.260 → 0.00 |

Every label's Brier improved, so all four calibrators were kept. ECE → 0 is
expected — isotonic fits the fit-split reliability curve almost exactly, so this
number is optimistic; the honest test is the eval-set F1 in §6, where calibration
is **F1-neutral** (it rearranges probabilities monotonically, then thresholds are
re-picked, so the decision barely changes). The value of calibration is that the
probabilities are now usable as confidence values and as inputs to the quality
score — it is **not** included because it raises F1.

## 6. Final comparison — Phase 3A vs 3B (untouched VizWiz val, ≥3 votes)

Primary macro-F1 = mean F1 over blur / underexposure / overexposure. `defect` is
reported but excluded from the primary metric (see below).
`runs/phase3b-calibration-v1/final_evaluation.json` has full per-label
precision / recall / ROC-AUC / PR-AUC / confusion.

| step | primary macro-F1 | 4-label macro-F1 | blur F1 | underexp F1 | overexp F1 | defect F1 |
|---|---|---|---|---|---|---|
| **A** Phase 3A baseline (v1 model, synthetic thresholds) | 0.3954 | 0.3040 | 0.611 | 0.478 | 0.098 | 0.030 |
| **B** + real-val F1 thresholds (v1 model) | **0.4366** | 0.3361 | 0.611 | 0.493 | 0.206 | 0.035 |
| **C** + isotonic probability calibration (v1 model) | 0.4309 | 0.3320 | 0.613 | 0.485 | 0.194 | 0.035 |
| **D** + cvfeat-v2 blockiness fix (retrained `v2fix`) | 0.4305 | 0.3317 | 0.611 | 0.489 | 0.191 | 0.035 |
| **E** + post-blur sensor noise (retrained `v2blur`) | 0.4033 | 0.3112 | 0.489 | 0.504 | 0.217 | 0.035 |

Synthetic reference for the same model (`v2fix` test split): primary macro-F1
**0.80**, blur 0.90, underexposure 0.84, overexposure 0.74. The synthetic → real
gap is still large; Phase 3B narrows the *starting* real number from 0.40 to
0.43, no more.

## 7. Ablation — what actually helped

- **Threshold re-selection on real data (A→B): +0.041 primary macro-F1.** The
  only clear win, and almost entirely from overexposure (0.098 → 0.206, false
  positives 323 → 47). Blur is unchanged; its threshold was already fine.
- **Isotonic probability calibration (B→C): -0.006.** F1-neutral-to-slightly-
  negative. Retained only for probability quality (Brier, confidence display,
  quality-score input), not for accuracy.
- **`compress_blockiness` fix, cvfeat-v2 (C→D): -0.0004.** Real-world-neutral.
  Kept regardless: it removes a genuine numerical pathology (feature values to
  ~590,000) and is simply the correct implementation.
- **Post-blur sensor noise (D→E): -0.027.** Negative result. It made blur harder
  to detect on real photos (blur F1 0.61 → 0.49, ROC-AUC 0.79 → 0.62) — the
  added noise pushed synthetic blur toward the *noise* region of feature space
  rather than toward real blur. **Rejected.** `phase3b-dataset-blurnoise-v1` and
  `v2blur` are kept as artifacts but not used.

**Shipped configuration (Phase 3C): row D** — `v2fix` model, cvfeat-v2, isotonic
calibrators, real-val F1 thresholds on calibrated probabilities. Primary macro-F1
**0.43** on the untouched eval set. Row B is 0.006 higher but uses the buggy v1
feature and uncalibrated probabilities; the difference is ~4 images and within
noise.

## Defect

Per the evidence (Phase 3A real F1 0.03, ROC-AUC 0.42), `defect` is **not tuned
to inflate its metric**. Its threshold is selected by the same criterion and its
(poor) numbers are reported for completeness, but it is excluded from the
primary macro-F1 and must not be exposed as a trustworthy prediction. The
current global-feature formulation conflates 5 unrelated artefact types and has
no real analogue in VizWiz beyond the partial OBS mapping.

**Numbers across the ablation** (VizWiz val, ≥3 votes, OBS-proxy label): F1 stays
at 0.03–0.035 in every row; ROC-AUC is 0.42 (row A) rising only to 0.50–0.51
once calibrated — i.e. **no better than chance**. The real-val F1 threshold for
defect collapses to ~0.02, which means "label every image as defective" (recall
1.0, precision 0.018). This is a degenerate optimum and confirms the global
`defect` head must not be exposed. It is **excluded from the shipped model's
issue outputs**; Phase 3C ships a separate patch-based signal instead.

**Recommendation:** a separate Phase 3C experiment that (a) decomposes "defect"
into concrete sub-issues that map to real annotations, and (b) prototypes
patch-level or small-CNN localisation for the spatial ones. Do not tune the
current head further.

## Quality score — still unresolved

Not addressed in Phase 3B. The synthetic quality target remains **provisional**
(`vyra_ml/labels.py`); its regression MAE/R² are not a perceptual-quality result
and are not reported here as one. Resolving it needs human-rated (MOS) data —
Phase 3C+.

## What remains unresolved

- **The synthetic → real gap is still wide.** Primary macro-F1 0.80 synthetic vs
  0.43 real. Threshold work and feature correctness do not close a domain gap;
  only real training data or a domain-adaptation step would.
- **Overexposure is weak on real data** (F1 0.19, ROC-AUC 0.63). Real "too
  bright" is often local glare/reflection, which global exposure features miss.
- **`defect` has no usable real signal** and no real dataset that matches VYRA's
  five synthetic sub-types. Needs decomposition + localisation, not tuning.
- **`noise` / `corruption` have zero real-world validation** — VizWiz does not
  annotate them. Their strong synthetic F1 (0.84 / 0.97) is in-domain only.
- **Exposure/defect real-val positive counts are ~50–69**, so those F1 numbers
  have wide confidence intervals; treat the ranking (blur ≫ underexposure >
  overexposure ≫ defect) as the robust conclusion, not the exact values.
- **Quality score** is untouched — still the provisional synthetic formula.
  Phase 3C defines an *operational* score from calibrated probabilities; a
  perceptual (MOS) score still needs human-rated data.
