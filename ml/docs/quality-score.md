# VYRA quality score (Phase 3C)

The API returns an overall **operational quality score** in `[0, 100]` and a
band label. This document defines it exactly and states what it is *not*.

## What it is not

It is **not** a perceptual / mean-opinion-score (MOS). VYRA has never been
trained or evaluated against human quality ratings — no MOS dataset was ingested
in any phase. Calling this number "perceptual quality" would be fabrication.

## Definition

Let `p_i` be the calibrated probability of issue *i* (isotonic-calibrated for
blur / underexposure / overexposure; raw model probability for noise /
corruption; the patch-anomaly probability for `potential_defect`). Let `t_i` be
that issue's decision threshold and `severe = 0.90`.

```
impact_i   = w_i * clip( (p_i - t_i) / (severe - t_i), 0, 1 )      # 0 below threshold
score      = 100 * Π_i (1 - impact_i)
```

* An issue below its threshold contributes nothing.
* Just past the threshold it contributes ~0; at `p_i >= 0.9` it removes its full
  weight `w_i` of the *remaining* score.
* Compounding (product, not sum) keeps the result in `[0, 100]` and models
  diminishing marginal damage — the second serious issue matters less than the
  first, which matches how a already-bad image cannot get much worse.
* Deterministic: same image → same score, no randomness.
* Monotone: raising any `p_i` never raises the score.

### Weights

| issue | `w_i` | rationale |
|---|---|---|
| blur | 0.55 | most destructive to usability; real-world validated |
| corruption | 0.45 | heavy compression / block corruption destroys detail; synthetic-validated |
| underexposure | 0.45 | real-world validated; recoverable but often severe |
| overexposure | 0.35 | real-world validated but weak detector (F1 0.19) — capped lower |
| noise | 0.30 | synthetic-validated only; usually less destructive than blur |
| potential_defect | 0.20 | screening signal only, ROC-AUC 0.60, not real-world validated — deliberately small |

Weights are **design choices, not fitted parameters** — there is no ground-truth
score to fit them to. They are ordered by how much each issue degrades
*usability* of a photo, and the two unvalidated signals (noise via synthetic
only, defect via screening) are held down so a shaky signal cannot dominate the
headline number.

### Bands

| score | label | meaning |
|---|---|---|
| 85–100 | GOOD | no issue past threshold, or only marginal ones |
| 68–84 | ACCEPTABLE | one moderate issue, image still usable |
| 45–67 | DEGRADED | a clear issue or several mild ones |
| 0–44 | POOR | a severe issue or a combination |

Band edges were chosen against the score distribution on the synthetic test
split and the real VizWiz evaluation split (see
`ml/artifacts/vyra-quality-model-v1/score_distribution.json`): clean synthetic
images cluster at 95–100, single severity-5 degradations land in the 30–55 range,
and the real-eval median sits in the ACCEPTABLE band. The four-band scheme is
kept simple on purpose — the evidence does not support finer distinctions.

## Reproducibility

The formula, `severe`, all weights and all band edges live in
`bundle.json → quality_score`, so the score is fully determined by the bundle
and `vyra_ml.inference` — nothing to remember, nothing hard-coded in the backend.
