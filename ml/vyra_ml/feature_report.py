"""Feature-space sanity report.

Answers the questions that must be settled before training: are there NaNs or
infinities, are ranges plausible, which features are numerically degenerate,
and which are near-duplicates of each other. Also checks that features are
size-robust (an image extracted at two resolutions should give similar values).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from vyra_ml.features import FEATURE_NAMES, extract_features

_HIGH_CORR = 0.97


def _size_robustness(sample_image_paths: list[Path]) -> dict:
    """Extract features at 288 and 512 long edge; report mean relative drift."""
    drifts = []
    for path in sample_image_paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        a = np.array(list(extract_features(bgr, 288).values()))
        b = np.array(list(extract_features(bgr, 512).values()))
        denom = np.maximum(np.abs(a), np.abs(b)) + 1e-6
        drifts.append(np.abs(a - b) / denom)
    if not drifts:
        return {}
    mean_drift = np.mean(drifts, axis=0)
    worst = np.argsort(mean_drift)[-5:][::-1]
    return {
        "median_relative_drift": float(np.median(mean_drift)),
        "worst_features": {FEATURE_NAMES[i]: round(float(mean_drift[i]), 3) for i in worst},
    }


def build_feature_report(
    feature_table_path: str | Path,
    out_dir: str | Path,
    *,
    sample_image_paths: list[Path] | None = None,
) -> Path:
    df = pd.read_parquet(feature_table_path)
    x = df[list(FEATURE_NAMES)]
    values = x.to_numpy(dtype=np.float64)

    per_feature = []
    for i, name in enumerate(FEATURE_NAMES):
        col = values[:, i]
        finite = col[np.isfinite(col)]
        per_feature.append(
            {
                "feature": name,
                "n_nan": int(np.isnan(col).sum()),
                "n_inf": int(np.isinf(col).sum()),
                "min": float(np.min(finite)) if finite.size else None,
                "max": float(np.max(finite)) if finite.size else None,
                "mean": float(np.mean(finite)) if finite.size else None,
                "std": float(np.std(finite)) if finite.size else None,
                "near_constant": bool(finite.size and np.std(finite) < 1e-9),
            }
        )

    corr = x.corr(method="spearman").to_numpy()
    redundant_pairs = [
        {"a": FEATURE_NAMES[i], "b": FEATURE_NAMES[j], "spearman": round(float(corr[i, j]), 3)}
        for i in range(len(FEATURE_NAMES))
        for j in range(i + 1, len(FEATURE_NAMES))
        if abs(corr[i, j]) >= _HIGH_CORR
    ]

    report = {
        "n_samples": int(len(df)),
        "n_features": len(FEATURE_NAMES),
        "total_nan": int(np.isnan(values).sum()),
        "total_inf": int(np.isinf(values).sum()),
        "near_constant_features": [f["feature"] for f in per_feature if f["near_constant"]],
        "highly_correlated_pairs": redundant_pairs,
        "per_feature": per_feature,
    }
    if sample_image_paths:
        report["size_robustness"] = _size_robustness(sample_image_paths[:12])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "feature_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    pd.DataFrame(per_feature).to_csv(out_dir / "feature_report.csv", index=False)
    return out_path
