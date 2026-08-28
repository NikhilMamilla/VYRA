"""Multi-label classification metrics.

Reports per-class precision/recall/F1/support and both macro and micro averages,
plus subset accuracy and Hamming loss. Per-class 2x2 confusion counts are
included because, with class imbalance, the averaged numbers hide where the
model actually fails.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def multilabel_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    y_score: np.ndarray | None = None,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=range(len(label_names))
    )

    per_class = {}
    for i, name in enumerate(label_names):
        tp = int(np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1)))
        fp = int(np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1)))
        fn = int(np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0)))
        tn = int(np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 0)))
        entry = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        }
        if y_score is not None and 0 < support[i] < len(y_true):
            entry["roc_auc"] = round(float(roc_auc_score(y_true[:, i], y_score[:, i])), 4)
            entry["pr_auc"] = round(float(average_precision_score(y_true[:, i], y_score[:, i])), 4)
        per_class[name] = entry

    return {
        "per_class": per_class,
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "micro_f1": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "samples_f1": round(float(f1_score(y_true, y_pred, average="samples", zero_division=0)), 4),
        "subset_accuracy": round(float(np.mean(np.all(y_true == y_pred, axis=1))), 4),
        "hamming_loss": round(float(hamming_loss(y_true, y_pred)), 4),
    }


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-9
    return {
        "mae": round(float(np.mean(np.abs(err))), 3),
        "rmse": round(float(np.sqrt(np.mean(err**2))), 3),
        "r2": round(1.0 - ss_res / ss_tot, 4),
        "bias": round(float(np.mean(err)), 3),
    }
