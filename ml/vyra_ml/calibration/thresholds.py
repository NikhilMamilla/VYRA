"""Decision-threshold selection on the real validation split.

The Phase 2 model outputs per-issue probabilities. Phase 3A used the thresholds
tuned on the *synthetic* val split and they did not transfer. Here we sweep
thresholds on the *real* validation split and pick one per issue.

Selection criterion: **maximise F1**, with a tie-break toward higher precision.
F1 is the right target for this application -- a quality gate should neither spam
false issues nor miss obvious ones, and both classes matter. The criterion is
recorded in the artifact so a later phase can revisit it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

_GRID = np.round(np.linspace(0.02, 0.98, 49), 4)


def sweep_thresholds(y_true: np.ndarray, y_score: np.ndarray) -> list[dict]:
    """Precision / recall / F1 at every grid threshold for one issue."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    rows = []
    for t in _GRID:
        pred = (y_score >= t).astype(int)
        tp = int(np.sum((y_true == 1) & (pred == 1)))
        fp = int(np.sum((y_true == 0) & (pred == 1)))
        fn = int(np.sum((y_true == 1) & (pred == 0)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "threshold": float(t),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    return rows


def select_threshold(
    y_true: np.ndarray, y_score: np.ndarray, *, criterion: str = "f1", min_support: int = 20
) -> tuple[float, dict]:
    """Return (threshold, chosen_row). Falls back to 0.5 below ``min_support``."""
    positives = int(np.sum(np.asarray(y_true) == 1))
    if positives < min_support:
        return 0.5, {"threshold": 0.5, "note": f"only {positives} positives (< {min_support})"}

    sweep = sweep_thresholds(y_true, y_score)
    if criterion != "f1":
        raise ValueError(f"Unsupported criterion {criterion!r}")
    # max F1, tie-break on precision then on a mid-range threshold.
    best = max(sweep, key=lambda r: (r["f1"], r["precision"], -abs(r["threshold"] - 0.5)))
    return best["threshold"], best


@dataclass
class ThresholdSet:
    """Versioned per-issue decision thresholds."""

    version: str
    parent: str
    model_artifact: str
    feature_version: str
    criterion: str
    fitted_on: str  # description of the real validation split
    seed: int
    thresholds: dict[str, float]
    selection_detail: dict[str, dict] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> ThresholdSet:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
