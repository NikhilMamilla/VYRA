# Evaluation protocol

Canonical description lives in the module docstring of
`vyra_ml/evaluation/protocol.py`; this is the summary.

## Metrics (`vyra_ml/evaluation/metrics.py`)

Multi-label issue classification, reported on `val` (threshold tuning) and
`test` (held out):

- **Per class:** precision, recall, F1, support, and the full 2x2 confusion
  (tp/fp/fn/tn) -- because with ~4:1 imbalance the averaged numbers hide which
  issue fails.
- **Aggregate:** macro-F1, micro-F1, samples-F1, subset accuracy (exact match on
  all six labels), Hamming loss.
- **Ranking:** per-class ROC-AUC and PR-AUC from the classifier scores. PR-AUC is
  the honest one under imbalance.

Provisional quality regression: MAE, RMSE, R2, bias -- flagged as a provisional
target everywhere it appears.

## Three evaluation levels

| Level | Data | Question | Status |
|---|---|---|---|
| 1 · Synthetic test | held-out `test` split; originals unseen in training | detect controlled degradations on new content | **available**, in every run's `metrics.json` |
| 2 · Real-world | real photos with human quality annotations (VizWiz-QualityIssues) | generalisation beyond the synthetic degradation model | not ingested (Phase 3) |
| 3 · Challenge | ~50-100 hand-curated: pristine, ambiguous, 3+ simultaneous real issues, deliberate low/high-key, real high-ISO noise, re-compressed, severe real degradation | failure modes and calibration under shift | not built (Phase 3) |

`evaluate_feature_table(model, table)` in `protocol.py` scores any level with a
trained baseline, as long as the table has the feature columns (+ `label_<issue>`
columns for scoring). Wiring in Level 2 is: implement a source adapter that
yields real images + a label dict, run the existing feature extractor, call that
function.

## Planned once Levels 2-3 exist

- Per-issue ROC and PR curves (not just AUC scalars).
- Thresholds **re-tuned per level** -- the synthetic-`val` thresholds will not
  transfer to real data.
- Calibration: reliability diagrams and Brier score. Tree-ensemble probabilities
  are typically miscalibrated; a `CalibratedClassifierCV` wrapper or isotonic
  fit on real `val` is the intended fix.
- Explicit clean-image false-positive rate as a headline number (a quality
  gate's most user-visible failure is flagging a good image).

## Not done, on purpose

No results are reported for Levels 2 or 3. No accuracy or generalisation claim is
made for real-world images. The infrastructure is in place; the data is not.
