# Potential visual defect (Phase 3C)

## Why the Phase 2 classifier was retired

The Phase 2 `defect` head was a global-feature RandomForest over one label that
lumps together five unrelated synthetic artefact types (dead-pixel clusters,
banding, block corruption, colour blotches, occlusion). It:

* did not localise (no region output);
* scored max feature importance 0.035 — no feature explained it;
* got synthetic-test ROC-AUC 0.75 but **real-world ROC-AUC 0.42** on VizWiz
  (Phase 3A) — worse than chance;
* produced a degenerate "flag everything" threshold when tuned on real data
  (Phase 3B).

It is **not exposed** by `vyra-quality-model-v1`.

## The replacement: self-referential patch anomaly

`vyra_ml.defect.patch_anomaly`. The image (resized to 512 px long edge) is tiled
into overlapping 64 px patches with 32 px stride. Each patch gets 8 cheap local
statistics: residual std / residual max (impulse & dead pixels), Laplacian
variance, 8-px blockiness, mean saturation, hue circular-std (colour blotches),
luma std, Canny edge density.

For each feature, a patch's **robust z-score** is `|x - median| / scale` where
`median` and `scale` (`1.4826·MAD` plus a relative floor) are taken over *the
other patches of the same image*, clipped at 25. A patch's anomaly is
`0.7·(top z) + 0.3·(2nd z)`. The image score is the strongest patch; a
fixed-slope sigmoid maps it to a probability, and the flagged region is that
patch.

**Why self-referential:** a defect is a local region unlike the rest of an
otherwise coherent image. Comparing a region to the same image needs no training
data, cannot memorise a synthetic fingerprint, and — because the reference is
the image itself — is expected to transfer to real images about as well as to
synthetic ones (though this is unverified; see below).

## Measured performance (synthetic test split, 513 images, 105 defect)

| metric | value |
|---|---|
| ROC-AUC | **0.60** |
| operating point | precision **0.33**, recall **0.32**, F1 0.33 |
| false-positive rate on clean images | ~17% |
| region localisation hit-rate | **0.32** (top patch centre inside the true bbox, among correct flags) |

The precision/recall curve is flat near precision 0.30–0.34 across the whole
probability range — the signal genuinely tops out there.

## Limitations — read before trusting a flag

* **Weak.** ROC-AUC 0.60. Roughly one flag in three is a real synthetic defect;
  it misses about two thirds of them.
* **Not real-world validated.** VizWiz has no defect annotation beyond the
  partial "obscured / finger over lens" category, so there is no real-world
  number for this signal at all.
* **False positives on busy images.** A legitimately high-contrast or
  saturated region (a bright window, a vivid sign) can look locally anomalous.
* **"Potential", not "confirmed".** The API field is `potential_visual_defect`
  and every response repeats: *not a confirmed physical defect. VYRA is an
  image-quality tool, not a diagnostic system.*
* Low weight (0.20) in the quality score, and only when flagged.

## If this were taken further (not done here)

Decompose `defect` into concrete sub-issues that map to real annotations
(occlusion, sensor dead-pixels, banding, …), collect or synthesise per-sub-issue
data with masks, and train a small patch classifier or lightweight segmentation
head per sub-issue. That is a project in itself and out of scope for Phase 3C.
