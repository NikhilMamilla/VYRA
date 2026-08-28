"""Classical ML baseline: CV features -> per-issue binary classifiers.

Multi-label is handled as one-vs-rest: an independent classifier per issue,
which keeps simultaneous issues representable and lets each issue use whichever
features matter for it. This is the baseline to beat, not a final architecture.

Model choice: tree ensembles (Random Forest by default, HistGradientBoosting as
an option). They handle the mixed scales and skewed distributions of these
features with no preprocessing, are cheap to train on a few thousand rows, and
expose feature importances -- which double as the explainability signal the
assessment asks for. A linear model was rejected: several feature/issue
relationships here are non-monotone (e.g. both very low and very high
`expo_luma_mean` are bad).

Splitting is read from the manifest (original-level, leakage-safe). `val` is
used for threshold selection; `test` is untouched until the final report.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)

from vyra_ml import ISSUE_LABELS, __version__
from vyra_ml.config import ExperimentConfig
from vyra_ml.evaluation.metrics import multilabel_report, regression_report
from vyra_ml.features import FEATURE_NAMES, FEATURE_VERSION


@dataclass
class BaselineArtifacts:
    run_dir: Path
    metrics: dict


def _make_classifier(cfg: ExperimentConfig, seed: int):
    if cfg.baseline.model == "random_forest":
        p = dict(cfg.baseline.random_forest)
        return RandomForestClassifier(random_state=seed, **p)
    if cfg.baseline.model == "hist_gradient_boosting":
        p = dict(cfg.baseline.hist_gradient_boosting)
        return HistGradientBoostingClassifier(random_state=seed, **p)
    raise ValueError(f"Unknown baseline model {cfg.baseline.model!r}")


def _select_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Threshold maximising F1 on the validation split for one issue."""
    if y_true.sum() == 0:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        pred = (y_score >= t).astype(int)
        tp = np.sum((y_true == 1) & (pred == 1))
        fp = np.sum((y_true == 0) & (pred == 1))
        fn = np.sum((y_true == 1) & (pred == 0))
        f1 = tp / (tp + 0.5 * (fp + fn) + 1e-9)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t


def run_baseline(cfg: ExperimentConfig, feature_table_path: str | Path) -> BaselineArtifacts:
    df = pd.read_parquet(feature_table_path)
    x_cols = list(FEATURE_NAMES)

    parts = {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
    x = {s: parts[s][x_cols].to_numpy(np.float64) for s in parts}
    y = {s: parts[s][[f"label_{n}" for n in ISSUE_LABELS]].to_numpy(int) for s in parts}

    started = time.perf_counter()
    models, thresholds, importances = {}, {}, {}
    val_scores = np.zeros_like(y["val"], dtype=float)
    test_scores = np.zeros_like(y["test"], dtype=float)

    for i, issue in enumerate(ISSUE_LABELS):
        clf = _make_classifier(cfg, cfg.seed + i)
        if y["train"][:, i].sum() == 0:
            raise RuntimeError(f"No positive '{issue}' samples in train split.")
        clf.fit(x["train"], y["train"][:, i])
        val_scores[:, i] = clf.predict_proba(x["val"])[:, 1]
        test_scores[:, i] = clf.predict_proba(x["test"])[:, 1]
        thresholds[issue] = _select_threshold(y["val"][:, i], val_scores[:, i])
        models[issue] = clf
        if hasattr(clf, "feature_importances_"):
            top = np.argsort(clf.feature_importances_)[::-1][:8]
            importances[issue] = {
                FEATURE_NAMES[j]: round(float(clf.feature_importances_[j]), 4) for j in top
            }

    thr = np.array([thresholds[n] for n in ISSUE_LABELS])
    val_pred = (val_scores >= thr).astype(int)
    test_pred = (test_scores >= thr).astype(int)

    metrics = {
        "issue_classification": {
            "val": multilabel_report(y["val"], val_pred, list(ISSUE_LABELS), val_scores),
            "test": multilabel_report(y["test"], test_pred, list(ISSUE_LABELS), test_scores),
        },
        "thresholds": {k: round(v, 3) for k, v in thresholds.items()},
        "feature_importances_top8": importances,
    }

    quality = None
    if cfg.baseline.fit_quality_regressor:
        reg = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=cfg.seed, n_jobs=-1
        )
        reg.fit(x["train"], parts["train"]["quality_score"].to_numpy(float))
        pred_test = reg.predict(x["test"])
        quality = {
            "note": "target is the PROVISIONAL synthetic quality score (see docs/dataset.md)",
            "test": regression_report(parts["test"]["quality_score"].to_numpy(float), pred_test),
        }
        metrics["quality_regression"] = quality

    elapsed = round(time.perf_counter() - started, 1)

    run_dir = cfg.data_dir("runs") / f"{cfg.version}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "thresholds": thresholds,
            "feature_names": FEATURE_NAMES,
            "feature_version": FEATURE_VERSION,
            "labels": ISSUE_LABELS,
        },
        run_dir / "model.joblib",
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    experiment_record = {
        "experiment": cfg.version,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": cfg.seed,
        "package_version": __version__,
        "python": platform.python_version(),
        "feature_version": FEATURE_VERSION,
        "n_features": len(FEATURE_NAMES),
        "source_dataset": cfg.dataset.source,
        "config_snapshot": {
            "dataset": vars(cfg.dataset) | {"splits": vars(cfg.dataset.splits)},
            "degradation": vars(cfg.degradation),
            "features": vars(cfg.features),
            "baseline": vars(cfg.baseline),
        },
        "split_counts": {s: int(len(parts[s])) for s in parts},
        "split_original_counts": {s: int(parts[s]["source_id"].nunique()) for s in parts},
        "model_type": cfg.baseline.model,
        "train_seconds": elapsed,
        "headline_metrics": {
            "test_macro_f1": metrics["issue_classification"]["test"]["macro_f1"],
            "test_micro_f1": metrics["issue_classification"]["test"]["micro_f1"],
            "test_subset_accuracy": metrics["issue_classification"]["test"]["subset_accuracy"],
            "quality_test_mae": quality["test"]["mae"] if quality else None,
        },
    }
    (run_dir / "experiment.json").write_text(
        json.dumps(experiment_record, indent=2, default=str), encoding="utf-8"
    )

    return BaselineArtifacts(run_dir=run_dir, metrics={**metrics, "_record": experiment_record})
