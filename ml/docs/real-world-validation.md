# Phase 3A — real-world validation (VizWiz-QualityIssues)

> An **experiment**, not integration. The Phase 2 baseline is **frozen**: loaded,
> never retrained, thresholds unchanged from the synthetic `val` split. All
> numbers are from `reports/phase3a-real-world-v1/`.

## 1. Dataset accessed

**VizWiz-QualityIssues** (Chiu et al., CVPR 2020,
<https://vizwiz.org/tasks-and-datasets/image-quality-issues/>).

| File | Contents | Size | Used |
|---|---|---|---|
| `image_quality/annotations.zip` → `{train,val}.json` | per-image flaw vote counts | 300 KB | **yes** (`val`, 7 750 labelled images) |
| `annotations/test.json` | image names only — **labels withheld for the challenge** | — | cannot be used for evaluation |
| `images/val.zip` | validation images | **3.5 GB** | **partial** — HTTP range requests (`remotezip`) pulled a seeded uniform sample of **2 500**; the full archive was never downloaded |

Evaluated: **2 496 / 2 500** images (4 dropped: min edge < 32 px; 0 unreadable,
0 grayscale). Image dimensions 50–2 592 px, median ≈ 968 × 1 296.

**Access limitation:** the official `test` split has no public labels, so
transfer is measured on `val`. `val` is used here purely as a held-out
evaluation set — no model development touched it, and a separate real
`val`/threshold-tuning split is reserved for Phase 3B.

## 2. Annotation structure

5 crowd workers per image. Per image: a vote count 0–5 for each of
`BLR BRT DRK OBS FRM ROT OTH NON`, plus an `unrecognizable` count 0–5.
**Vote-based, not binary.** Multi-issue (workers flag several per image; mean
sum of flaw votes ≈ 5). Genuine ambiguity exists (10 val images voted both BRT
and DRK ≥ 3).

Prevalence in the 2 496-image sample (≥ 3 votes): FRM 24 %, BLR 29 %, NON 30 %,
ROT 15 %, DRK 2.8 %, BRT 3.5 %, OBS 2.6 %, `unrecognizable` 9.5 %. **Framing and
rotation dominate real user complaints and are not VYRA concepts.**

## 3. Label mapping — full table in `label_mapping.md`

| VizWiz | VYRA | category | confidence | evaluated |
|---|---|---|---|---|
| BLR | blur | **A** directly mappable | high | yes |
| BRT | overexposure | **A** | medium (BRT also = glare/reflection) | yes |
| DRK | underexposure | **A** | medium (real dark frames also noisy) | yes |
| OBS | defect | **B** partial | low — OBS ≈ 1 of 5 synthetic defect sub-types | yes, heavily caveated |
| FRM, ROT | — | **C** not mappable | — | no |
| OTH, NON | — | **D** auxiliary | — | no |
| `unrecognizable` | — | **D** auxiliary (severity, not a cause) | — | no |
| — | **noise** | **C** | Not directly supported by this dataset | **no** |
| — | **corruption** | **C** | Not directly supported by this dataset | **no** |

`noise` and `corruption` have **no VizWiz category**. `unrecognizable` was
investigated as a `corruption` proxy and rejected: on `val` it correlates weakly
and diffusely with BLR (r = 0.32), DRK (0.35), BRT (0.28), OBS (0.30) — it is a
severity signal driven by many causes, not compression. No fake negatives were
assigned; those two labels are simply not scored.

Binarisation: a label is "present" at **≥ 3 / 5 votes** (majority). Metrics also
reported at ≥ 2 and ≥ 4 (`evaluation_metrics.json`).

## 4. Results — `synthetic_vs_real.md`

Frozen model `phase2-baseline-v1_20260828-123813`, feature version `cvfeat-v1`,
synthetic-`val` thresholds `{blur .45, underexposure .55, overexposure .45,
defect .30}`. **Leakage check: 0 sha1 overlap, 0 id overlap** vs 3 600 training
images.

| label | synthetic-test F1 | real F1 (≥3) | real P / R | real ROC-AUC | real PR-AUC | real support |
|---|---|---|---|---|---|---|
| blur | 0.90 | **0.61** | 0.51 / 0.77 | 0.79 | 0.57 | 732 |
| underexposure | 0.85 | **0.48** | 0.49 / 0.46 | **0.89** | 0.33 | 69 |
| overexposure | 0.77 | **0.10** | 0.06 / 0.30 | 0.63 | 0.16 | 66 |
| defect | 0.46 | **0.03** | 0.02 / 0.39 | **0.42** | 0.02 | 44 |

**Macro-F1 (4 evaluable labels): 0.744 → 0.304. Micro-F1 0.786 → 0.359.**
Hamming loss 0.085 → 0.226. Sensitivity: macro-F1 is 0.32 / 0.30 / 0.24 at vote
thresholds 2 / 3 / 4.

Reading:
- **blur — partial transfer.** ROC-AUC 0.79 says the score still ranks blurry
  above sharp; F1 collapses mostly because the synthetic threshold gives
  precision 0.51 (544 false positives). The signal survives; the operating
  point does not.
- **underexposure — ranking transfers, threshold does not.** ROC-AUC **0.89**
  is the best of any label, but at the frozen threshold recall is only 0.46.
- **overexposure — largely fails.** ROC-AUC 0.63; 323 false positives.
- **defect — total failure.** ROC-AUC **0.42 (worse than chance)**, 1 073 false
  positives. The output carries no real-world signal.

## 5. Failure analysis (`failure_analysis.json` + `failure_examples/`)

Grouped, observational causes — "likely contributing factor", not proven:

1. **Non-photographic content (biggest single factor).** VizWiz contains
   screenshots, meme/quote graphics and document photos; BSDS500 has **none**.
   - A phone screenshot of Google results (NON = 5) → `defect` score 0.71:
     axis-aligned UI structure reads as banding/blockiness.
   - A light-blue quote graphic (NON = 5) → `overexposure` 0.93: large uniform
     bright region with no photographic texture.
2. **Real blur ≠ synthetic blur.** Synthetic blur removes *all* high frequency
   uniformly. Real blurry photos keep sensor noise and textured surfaces:
   - a blurry pill bottle on a granite counter (BLR = 4) → predicted sharp; the
     counter's speckle keeps `sharp_highfreq_ratio` / Laplacian variance high.
   `domain_shift.json`: `sharp_laplacian_var` synthetic-blur mean 1 674 vs
   real-blur **162**.
3. **Low-texture real scenes → blur false positives.** Plain fabric, walls, dark
   plastic interiors have low high-frequency energy without being blurred
   (a soft-focus fabric shot annotated BLR = 1 → score 0.99).
4. **Threshold non-transfer.** The `defect` threshold 0.20 (set low in Phase 2
   because synthetic defect was hard) fires on ~1 090 / 2 496 real images.
5. **Soft labels / annotator disagreement.** Many "errors" at ≥ 3 votes are
   1–2-vote borderline images; the model's call is often defensible.
6. **Feature instability discovered.** `compress_blockiness` on synthetic
   degraded images has mean 800, std 21 700 — the interior-gradient denominator
   underflows on near-flat regions. Harmless within synthetic (RF is
   scale-robust) but it means the synthetic feature tail is pathological and
   unlike anything real. **Fix in Phase 3B.**

## 6. Defect — what the experiment revealed

- **VizWiz has no analogue for VYRA "defect".** OBS is the only candidate,
  covers 1 of 5 synthetic sub-types, and has 44 positives at ≥ 3 votes.
- Real defect-like cases (OBS) do **not** resemble the synthetic defects: OBS is
  a soft finger/shadow occlusion; synthetic defect is dominated by dead pixels,
  banding and block corruption.
- The classifier's defect output is **anti-correlated** with truth on real data
  (ROC-AUC 0.42). It is not merely weak — it is misleading.
- **Conclusion:** "potential visual defect" is not a coherent single concept and
  not a global-feature classification problem. It is (a) a label-definition
  problem — it needs decomposition into concrete sub-issues that map to real
  data, and (b) for the localised ones, a localisation/patch or small-CNN
  problem. It should be **removed from the shared multi-label head** until
  redefined.

## 7. Domain shift — synthetic vs real (`domain_shift.json`)

| feature | syn clean | real clean-ish (n=677) | syn blur | real blur (n=732) |
|---|---|---|---|---|
| `sharp_laplacian_var` | 1169 | 721 | 1674 | **162** |
| `sharp_highfreq_ratio` | 0.478 | 0.391 | 0.304 | 0.304 |
| `noise_immerkaer_sigma` | 3.07 | 2.15 | 2.90 | **1.26** |
| `contrast_p95` | 0.765 | 0.842 | 0.718 | 0.835 |
| `compress_blockiness` | 1.00 | 1.03 | **800** | 1.03 |

Even the "clean" distributions differ: real VizWiz photos are *lower* in absolute
sharpness and noise than degraded-BSDS "clean", and *higher* in contrast. The
synthetic pipeline's degraded tail is far outside anything real. Blur is the one
place high-frequency-ratio lines up — which is why blur is the one label that
partly transfers.

## 8. Limitations — what we cannot conclude

- **Nothing about `noise` or `corruption` transfer** — VizWiz cannot test them.
- **`overexposure` / `defect` supports are tiny** (66 / 44 at ≥ 3 votes). Their
  F1s are directional, not precise; CIs are wide.
- Only the VizWiz *distribution* was tested (blind-photographer images, heavy on
  screenshots/documents/close-ups). A different real corpus (e.g. curated
  photography) could shift every number.
- Transfer was measured at **one** operating point (frozen thresholds). ROC/PR
  curves suggest re-thresholding alone would recover a large part of blur and
  underexposure — that is a Phase 3B experiment, not a conclusion here.
- Single run, single seed for the subset. Re-sampling would move the rare-label
  numbers most.
- We did **not** retune, retrain, drop hard samples, or change features and
  re-report. This result stands as the honest first measurement.

## 9. What this says about the architecture

- The **hybrid CV-feature + classical-ML** approach has real signal for the
  global exposure/sharpness issues (blur ROC-AUC 0.79, underexposure 0.89) but
  is **badly calibrated** out of domain and **cannot** do localised defects.
- **Synthetic-only training does not generalise** at a fixed operating point.
  The synthetic degradations are too clean (no post-blur noise, pathological
  feature tails) and the source imagery is too narrow (no screenshots/docs/
  graphics).
- Per-issue independent classifiers were the right call — the failures are
  per-label and diagnosable, and `noise`/`corruption` were cleanly excluded
  rather than faked.
