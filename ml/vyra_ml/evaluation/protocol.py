"""The three-level evaluation protocol.

Phase 2 has only Level 1 data (the synthetic test split). Levels 2 and 3 need
real-world images that are not ingested yet; this module defines the protocol
and provides the reusable evaluation function so wiring them in later is just
supplying a feature table with the same columns.

Level 1 - SYNTHETIC TEST SET
    The held-out `test` split of this pipeline. Originals never seen in
    training (leakage-safe). Measures: can the model detect controlled
    degradations it was trained on the *type* of, on new image content.
    Status: available. Reported in runs/<version>/metrics.json.

Level 2 - REAL-WORLD TEST SET
    Real photos with human quality-issue annotations (VizWiz-QualityIssues).
    Measures: does detection generalise past the synthetic degradation model.
    Status: DONE (Phase 3A) for blur / under- / overexposure / defect. See
    docs/real-world-validation.md and reports/phase3a-real-world-v1/. Result:
    macro-F1 0.744 (synthetic) -> 0.304 (real) at frozen thresholds. `noise` and
    `corruption` are not evaluable against VizWiz.

Level 3 - CHALLENGE SET
    A small, hand-curated set (~50-100 images): pristine images, genuinely
    ambiguous cases, images with 3+ simultaneous real issues, unusual-but-ok
    exposure (deliberate low-key / high-key), real high-ISO noise, heavily
    re-compressed images, and severe real degradation. Measures: failure modes
    and calibration under distribution shift. Curated, never fabricated.
    Status: not built (Phase 3).

Planned additions once Levels 2-3 exist: per-issue ROC / PR curves, threshold
sweeps re-tuned per evaluation level, and calibration (reliability diagrams,
Brier score) since thresholds tuned on synthetic `val` will not transfer.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from vyra_ml import ISSUE_LABELS
from vyra_ml.evaluation.metrics import multilabel_report

LEVELS = ("synthetic_test", "real_world", "challenge")


@dataclass
class LoadedModel:
    models: dict
    thresholds: dict
    feature_names: tuple[str, ...]
    feature_version: str


def load_model(path) -> LoadedModel:
    blob = joblib.load(path)
    return LoadedModel(
        models=blob["models"],
        thresholds=blob["thresholds"],
        feature_names=tuple(blob["feature_names"]),
        feature_version=blob["feature_version"],
    )


def evaluate_feature_table(model: LoadedModel, feature_table: pd.DataFrame) -> dict:
    """Score a feature table (any evaluation level) with a trained baseline model.

    ``feature_table`` must contain the model's feature columns and, for scoring,
    the ``label_<issue>`` columns.
    """
    x = feature_table[list(model.feature_names)].to_numpy(np.float64)
    scores = np.column_stack([model.models[issue].predict_proba(x)[:, 1] for issue in ISSUE_LABELS])
    thr = np.array([model.thresholds[i] for i in ISSUE_LABELS])
    pred = (scores >= thr).astype(int)

    label_cols = [f"label_{i}" for i in ISSUE_LABELS]
    if not set(label_cols).issubset(feature_table.columns):
        return {"predictions_only": pred.tolist()}
    y = feature_table[label_cols].to_numpy(int)
    return multilabel_report(y, pred, list(ISSUE_LABELS), scores)
