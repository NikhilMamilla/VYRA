"""Compute CV features for every row of a manifest and cache them to Parquet.

The feature table is keyed by ``sample_id`` and carries the split, label columns
and provisional quality score alongside the features, so downstream code loads
one file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from vyra_ml import ISSUE_LABELS
from vyra_ml.config import ExperimentConfig
from vyra_ml.features import FEATURE_NAMES, FEATURE_VERSION, extract_features
from vyra_ml.manifest import read_manifest

_CARRY_COLUMNS = [
    "sample_id",
    "source_id",
    "split",
    "is_clean",
    "max_severity",
    "quality_score",
    *[f"label_{name}" for name in ISSUE_LABELS],
]


def _extract_one(sample_id: str, image_path: str, processed_dir: Path, work_long_edge: int):
    bgr = cv2.imread(str(processed_dir / image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return sample_id, None, f"unreadable: {image_path}"
    try:
        feats = extract_features(bgr, work_long_edge=work_long_edge)
    except Exception as exc:  # noqa: BLE001 - record and continue the batch
        return sample_id, None, f"{type(exc).__name__}: {exc}"
    return sample_id, feats, None


def build_feature_table(
    cfg: ExperimentConfig, manifest_path: str | Path | None = None, *, n_jobs: int = -1
) -> Path:
    manifest_path = manifest_path or (cfg.data_dir("manifests") / f"manifest_{cfg.version}.parquet")
    manifest = read_manifest(manifest_path)
    processed_dir = cfg.data_dir("processed")

    started = time.perf_counter()
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_extract_one)(
            row.sample_id, row.image_path, processed_dir, cfg.features.work_long_edge
        )
        for row in manifest.itertuples(index=False)
    )
    elapsed = time.perf_counter() - started

    feat_rows, failures = [], []
    for sample_id, feats, error in results:
        if error:
            failures.append({"sample_id": sample_id, "error": error})
            continue
        feat_rows.append({"sample_id": sample_id, **feats})

    features = pd.DataFrame(feat_rows, columns=["sample_id", *FEATURE_NAMES])
    merged = manifest[_CARRY_COLUMNS].merge(features, on="sample_id", how="inner")

    out_dir = cfg.data_dir("processed")
    out_path = out_dir / f"features_{cfg.version}_{FEATURE_VERSION}.parquet"
    merged.to_parquet(out_path, index=False)

    meta = {
        "feature_version": FEATURE_VERSION,
        "n_features": len(FEATURE_NAMES),
        "n_samples": len(merged),
        "n_failures": len(failures),
        "failures": failures[:50],
        "seconds": round(elapsed, 1),
        "ms_per_image": round(1000 * elapsed / max(1, len(results)), 2),
    }
    (out_dir / f"features_{cfg.version}_{FEATURE_VERSION}.meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    if failures:
        print(f"WARN: {len(failures)} feature-extraction failures (see meta json)", flush=True)
    return out_path


def load_feature_table(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Guard: features must be finite before training.
    feat = df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    if not np.isfinite(feat).all():
        bad = np.array(FEATURE_NAMES)[~np.isfinite(feat).all(axis=0)]
        raise ValueError(f"Non-finite values in cached features: {bad.tolist()}")
    return df
