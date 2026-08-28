"""Extract CV features for a set of VizWiz images into one Parquet table.

Shared by the real *validation* split (Phase 3B calibration) and the real
*evaluation* set (Phase 3A / 3B final eval). Every row carries:

* identity + load diagnostics (image_id, split, width, height, load_status, sha1)
* raw VizWiz vote counts (BLR..NON, unrecognizable)
* the current feature version's 42 features
* the three legacy cvfeat-v1 blockiness values (``v1_blockiness*``) so the
  Phase 3A model can be scored without a second full extraction pass

Nothing here reads labels for tuning -- it only records them.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from vyra_ml.features import FEATURE_NAMES, FEATURE_VERSION, extract_features
from vyra_ml.features.common import prepare
from vyra_ml.features.compression import legacy_ratio_blockiness
from vyra_ml.realworld.vizwiz import VIZWIZ_FLAW_CODES, VizWizAnnotation

_MIN_EDGE = 32
V1_BLOCKINESS_COLS = ("v1_blockiness", "v1_blockiness_h", "v1_blockiness_v")


def _one_row(
    ann: VizWizAnnotation, image_path: Path, split: str, work_long_edge: int
) -> dict | None:
    if not image_path.is_file():
        return {"image_id": ann.image, "split": split, "load_status": "missing"}
    data = image_path.read_bytes()
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    row: dict = {
        "image_id": ann.image,
        "split": split,
        "sha1": hashlib.sha1(data).hexdigest(),
        **{f"vote_{c}": ann.votes(c) for c in VIZWIZ_FLAW_CODES},
        "vote_unrecognizable": ann.unrecognizable_votes,
    }
    if bgr is None:
        return {**row, "load_status": "unreadable", "width": 0, "height": 0}
    h, w = bgr.shape[:2]
    row.update(width=w, height=h)
    if min(h, w) < _MIN_EDGE:
        return {**row, "load_status": "too_small"}
    try:
        feats = extract_features(bgr, work_long_edge=work_long_edge)
        legacy = legacy_ratio_blockiness(prepare(bgr, work_long_edge))
    except Exception as exc:  # noqa: BLE001
        return {**row, "load_status": f"feature_error:{type(exc).__name__}"}
    row["load_status"] = "ok"
    row.update(feats)
    row.update(legacy)
    return row


def build_vizwiz_feature_table(
    annotations: list[VizWizAnnotation],
    image_dir: Path,
    *,
    split: str,
    work_long_edge: int,
    out_path: Path,
    progress_every: int = 250,
) -> dict:
    rows: list[dict] = []
    started = time.perf_counter()
    for i, ann in enumerate(annotations):
        rows.append(_one_row(ann, image_dir / ann.image, split, work_long_edge))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  [{split}] {i + 1}/{len(annotations)}", flush=True)

    df = pd.DataFrame(rows)
    usable = df[df["load_status"] == "ok"].reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    usable.to_parquet(out_path, index=False)

    # finite check on the feature block
    feat = usable[list(FEATURE_NAMES) + list(V1_BLOCKINESS_COLS)].to_numpy(dtype=np.float64)
    stats = {
        "split": split,
        "feature_version": FEATURE_VERSION,
        "requested": len(annotations),
        "usable": int(len(usable)),
        "load_status_counts": df["load_status"].value_counts().to_dict(),
        "non_finite_cells": int((~np.isfinite(feat)).sum()),
        "seconds": round(time.perf_counter() - started, 1),
        "out_path": str(out_path),
    }
    return stats


def vote_threshold_labels(df: pd.DataFrame, vote_min: int) -> pd.DataFrame:
    """Add label_<vyra> columns for the evaluable labels at a vote threshold."""
    from vyra_ml.realworld.label_map import evaluable_labels, vyra_to_vizwiz_code

    code_by_label = vyra_to_vizwiz_code()
    out = df.copy()
    for lbl in evaluable_labels():
        out[f"label_{lbl}"] = (out[f"vote_{code_by_label[lbl]}"] >= vote_min).astype(int)
    return out


def feature_matrix(df: pd.DataFrame, feature_version: str) -> np.ndarray:
    """Return the 42-column feature matrix for a model.

    ``cvfeat-v1`` swaps in the legacy blockiness columns; ``cvfeat-v2`` uses the
    table's own (fixed) blockiness columns.
    """
    cols = list(FEATURE_NAMES)
    x = df[cols].to_numpy(dtype=np.float64).copy()
    if feature_version == "cvfeat-v1":
        for name in ("compress_blockiness", "compress_blockiness_h", "compress_blockiness_v"):
            x[:, cols.index(name)] = df[f"v1_{name.split('_', 1)[1]}"].to_numpy(np.float64)
    elif feature_version != FEATURE_VERSION:
        raise ValueError(f"Unsupported feature_version {feature_version!r}")
    return x
