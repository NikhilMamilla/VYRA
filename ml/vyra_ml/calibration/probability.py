"""Probability calibration on the real validation split.

Tree-ensemble scores are rank-informative but poorly calibrated as
probabilities. We fit a per-issue calibrator (isotonic or sigmoid) on the real
validation split and report whether it actually improves reliability -- Brier
score and expected calibration error before/after. Calibration is only kept for
an issue where the validation support is large enough to fit it reliably.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        bins.append(
            {
                "bin": [float(edges[b]), float(edges[b + 1])],
                "count": count,
                "mean_predicted": float(y_prob[mask].mean()) if count else None,
                "empirical_frequency": float(y_true[mask].mean()) if count else None,
            }
        )
    return {"n_bins": n_bins, "bins": bins}


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    curve = reliability_curve(y_true, y_prob, n_bins)
    n = len(y_true)
    ece = 0.0
    for b in curve["bins"]:
        if b["count"]:
            ece += (b["count"] / n) * abs(b["empirical_frequency"] - b["mean_predicted"])
    return float(ece)


def _fit_one(y_true: np.ndarray, y_score: np.ndarray, method: str):
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(y_score, y_true)
        return model
    if method == "sigmoid":
        model = LogisticRegression(C=1e6, solver="lbfgs")
        model.fit(y_score.reshape(-1, 1), y_true)
        return model
    raise ValueError(f"Unknown calibration method {method!r}")


def _apply_one(model, y_score: np.ndarray) -> np.ndarray:
    if isinstance(model, IsotonicRegression):
        return np.clip(model.predict(y_score), 0.0, 1.0)
    return model.predict_proba(y_score.reshape(-1, 1))[:, 1]


@dataclass
class PerLabelCalibrator:
    """One probability calibrator per issue (or identity where unreliable)."""

    version: str
    parent: str
    method: str
    fitted_on: str
    min_support: int
    models: dict  # label -> fitted model or None (identity)
    diagnostics: dict  # label -> {before/after brier, ece, ...}

    def transform(self, label: str, y_score: np.ndarray) -> np.ndarray:
        model = self.models.get(label)
        if model is None:
            return np.asarray(y_score, dtype=float)
        return _apply_one(model, np.asarray(y_score, dtype=float))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> PerLabelCalibrator:
        return joblib.load(path)


def fit_calibrators(
    scores_by_label: dict[str, np.ndarray],
    truth_by_label: dict[str, np.ndarray],
    *,
    version: str,
    parent: str,
    method: str = "isotonic",
    fitted_on: str = "real validation split",
    min_support: int = 40,
) -> PerLabelCalibrator:
    models: dict = {}
    diagnostics: dict = {}
    for label, score in scores_by_label.items():
        y = np.asarray(truth_by_label[label], dtype=int)
        support = int(y.sum())
        before = {
            "brier": round(brier_score(y, score), 4),
            "ece": round(expected_calibration_error(y, score), 4),
        }
        if support < min_support or y.mean() in (0.0, 1.0):
            models[label] = None
            diagnostics[label] = {
                "support": support,
                "calibrated": False,
                "reason": f"support {support} < {min_support}"
                if support < min_support
                else "degenerate labels",
                "before": before,
            }
            continue
        model = _fit_one(y, np.asarray(score, dtype=float), method)
        after_prob = _apply_one(model, np.asarray(score, dtype=float))
        after = {
            "brier": round(brier_score(y, after_prob), 4),
            "ece": round(expected_calibration_error(y, after_prob), 4),
        }
        # Keep calibration only if it does not worsen Brier on the fit split.
        keep = after["brier"] <= before["brier"] + 1e-4
        models[label] = model if keep else None
        diagnostics[label] = {
            "support": support,
            "calibrated": keep,
            "reason": None if keep else "did not improve Brier on validation",
            "before": before,
            "after": after,
            "reliability_after": reliability_curve(y, after_prob),
        }
    return PerLabelCalibrator(
        version=version,
        parent=parent,
        method=method,
        fitted_on=fitted_on,
        min_support=min_support,
        models=models,
        diagnostics=diagnostics,
    )
