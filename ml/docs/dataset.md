# VYRA dataset (Phase 2)

## Approach: clean real images + controlled synthetic degradation

Phase 2 builds its dataset by applying calibrated, randomised degradations to
clean real photographs. This is a deliberate choice over ingesting a large
real-world quality dataset:

- **Label reliability.** Ground truth is known by construction -- we applied the
  degradation, so the multi-label vector and per-issue severity are exact. Human
  quality annotations are noisy and often single-label.
- **Leakage control.** With synthetic variants we can split on the *original*
  image and be certain no pixels leak between splits (see below).
- **Coverage.** We can guarantee balanced coverage of all six issue types at all
  five severities, and of multi-issue combinations, which no found dataset
  provides.
- **Practicality.** A 48h assessment cannot responsibly download and curate tens
  of GB. The clean-image base here is ~72 MB.

The cost -- generalisation to *real* degradation is unproven -- is real and is
addressed by the Level 2 / Level 3 evaluation plan below, not hidden.

## Source images

| Source | Role | Size | Licence | Status |
|---|---|---|---|---|
| **BSDS500** (BIDS GitHub mirror) | clean originals for synthetic degradation | ~72 MB, 500 images | Berkeley, research/educational use | **used** (400 originals) |
| VizWiz-QualityIssues | Level 2 real-world eval | ~few GB | VizWiz licence, research use | documented, not ingested |
| SPAQ | alt. real-world eval, MOS + attributes | ~30 GB | authors' form/agreement | documented, not ingested |

**Why not VizWiz / SPAQ now.** VizWiz-QualityIssues is downloadable but its
image payload is multi-GB and its annotations (blur / bright / dark / obscured /
framing / rotation / other) do not map cleanly onto our six issues -- it needs a
label-mapping study of its own. SPAQ requires accepting an agreement and hosts
its ~30 GB on drives that are slow/unreliable to script. Both serve the
*real-world evaluation* level, which is Phase 3. The ingestion interface
(`vyra_ml/ingest/base.py`) is where a `VizWizAdapter` slots in: it only has to
yield images plus a `labels` dict, and the rest of the pipeline (features,
metrics, protocol) already consumes that shape.

The Berkeley `www2.eecs.berkeley.edu` host for BSDS500 was unreachable during
the build; the adapter uses the `BIDS/BSDS500` GitHub mirror, which carries the
identical 500 images.

## Leakage prevention (non-negotiable)

The split is decided on the **original `source_id`, before any degradation is
generated**. Implementation in `vyra_ml/splitting.py`:

1. `split_originals()` hashes each `source_id` (blake2b, seeded) to a unit
   interval and assigns train / val / test by ratio.
2. Every degraded variant of an original is written with that same split.
3. `assert_no_leakage()` runs on the finished manifest and raises if any
   `source_id` appears under more than one split. The dataset build calls it
   before writing; a dedicated test
   (`tests/test_splitting.py::test_source_id_never_spans_train_and_test_over_the_whole_pipeline`
   and `test_manifest_and_build.py::test_build_output_has_no_leakage`) asserts it
   end-to-end.

Because assignment is by hash, not by shuffling a list, it is order-independent
(adding originals doesn't move existing ones) and reproducible from the seed
alone.

Current split: **400 originals -> 286 train / 57 val / 57 test**, expanding to
**2574 / 513 / 513 samples**.

## Synthetic degradation engine

Modular: one class per degradation in `vyra_ml/degradations/`, each independently
tested. Every degradation takes `(image, severity 1-5, seeded RNG)` and returns
the image plus the **exact parameters it sampled**, which are stored in the
manifest.

Parameters are sampled from a **range per severity level**, never a fixed value,
so the model cannot memorise a synthetic fingerprint. Severity 1 is near the
perceptual threshold; severity 5 is extreme.

| Degradation | Issue label | What varies (randomised) |
|---|---|---|
| `blur` | blur | kind (Gaussian 60% / motion 40%); Gaussian sigma 0.6-7.5 px; motion length 3-48 px + angle 0-180 |
| `underexposure` | underexposure | EV shift 0.3-3.8 stops (light-linear domain); added read-noise sigma scaled with darkness; small black lift |
| `overexposure` | overexposure | EV shift 0.3-3.6 stops; highlight knee (soft roll-off at low severity -> hard clip at high) |
| `noise` | noise | type (Gaussian / Poisson-shot / speckle / salt-pepper); per-type strength; luma + reduced chroma noise |
| `corruption` | corruption | JPEG quality 4-80; resolution loss 0.3-1.0x (downscale/upscale) at high severity; occasional double-JPEG at sev 5 |
| `defect` | defect | type (dead-pixel clusters / banding / block corruption / colour blotch / occlusion); region bbox; extent; intensity -- **localised**, random position |

**Realism choices** (also in code comments): exposure is applied in a
light-linear (sRGB-decoded) domain, not naive multiplication; dark frames get
extra read noise so underexposure is not trivially separable from `noise`;
highlights roll off before clipping; JPEG is a real `cv2.imencode` round-trip;
`corruption` includes genuine resolution loss so "severe degradation" is present,
not just blocking. Contradictory pairs (`underexposure` + `overexposure`) are
excluded from multi-issue samples.

**Application order** when combining degradations:
`exposure -> blur -> defect -> noise -> corruption` -- a pragmatic approximation
of a capture pipeline (documented in `degradations/__init__.py`).

**Phase 3B — post-blur sensor noise** (`degradation.post_blur_sensor_noise`,
off by default). Phase 3A found synthetic blur removes *all* high frequency,
whereas real blurred photos retain sensor noise (noise is added after the
optics). When enabled, blur-containing samples get a light zero-mean Gaussian
read-noise pass (luma sigma 1.5-4.0/255, below the `noise` class's severity-1
range) after all degradations. Used by `configs/experiment_blurnoise.yaml`; see
`docs/phase3b-calibration.md`.

Per original: 1 clean + 6 single-degradation + 2 multi-degradation (2-3
simultaneous issues) samples.

## Storage

- `data/raw/` -- downloaded source datasets, untouched (git-ignored)
- `data/processed/<split>/<sample_id>.jpg` -- generated images, **JPEG q=97**
  (near-lossless; keeps the full set to ~150 MB vs ~700 MB as PNG). BSDS
  originals are themselves JPEG, so this adds no meaningful artefact relative to
  the source. `corruption` samples bake their artefacts into pixels *before*
  this store step, so q=97 preserves them faithfully.
- `data/manifests/` -- `manifest_<version>.parquet` + `.jsonl` + build metadata
- All of `data/` is git-ignored and reproducible from `configs/experiment.yaml`
  + the seed.

## Manifest schema

One row per image. Flat label/severity columns for ML convenience, plus nested
JSON for full detail. Full column list in `vyra_ml/manifest.py`. Key fields:
`sample_id`, `source_id`, `source_dataset`, `split`, `image_path`, `is_clean`,
`degradations_json` (list of `{name, severity, params}`), `label_<issue>` x6,
`severity_<issue>` x6, `max_severity`, `quality_score` (provisional),
`width/height`, `orig_width/orig_height`, `file_bytes`, `sha1`.

## Quality score -- PROVISIONAL

We do **not** have human quality ratings for these synthetic images and do not
claim a ground-truth quality score. `quality_score` in the manifest is a
documented placeholder: `100 x product over active issues of a per-severity
factor` (factors `{1:.95, 2:.85, 3:.68, 4:.45, 5:.22}`, in `vyra_ml/labels.py`).
Compounding keeps it in range and reflects diminishing marginal damage. It
exists to support a regression baseline and will be replaced once MOS-rated real
data (SPAQ-style) is inspected in Phase 3.

## Known dataset issues

- **Class imbalance:** each issue is positive in ~19-21% of samples (negative
  :positive ~3.8:1). Handled with `class_weight="balanced"` and F1-tuned
  per-issue thresholds; reported per-class, not just averaged.
- **`defect` is heterogeneous:** five visually unrelated artefact types under one
  label, localised, so global image features capture it poorly (baseline F1
  ~0.46). This is the clearest signal that a patch/localisation or CNN approach
  is needed for that issue.
- **Synthetic-only:** no evidence yet of generalisation to real degradation.
  Levels 2-3 exist precisely to measure this and are not yet run.
- **Source content skew:** BSDS500 is natural outdoor/scene photography at
  ~481x321; no documents, faces-closeups, screenshots, or low-light night
  shots. Real deployment images will be broader.
- **BSDS originals are already lightly JPEG-compressed**, so the "clean" class
  carries a small baseline compression level. `corruption` severities are all
  well below it.
