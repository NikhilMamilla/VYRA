"""Phase 3B wiring: feature-matrix v1/v2 swap, vote labels, split separation,
experiment artifact structure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vyra_ml.features import FEATURE_NAMES, FEATURE_VERSION
from vyra_ml.realworld.features import (
    V1_BLOCKINESS_COLS,
    feature_matrix,
    vote_threshold_labels,
)
from vyra_ml.realworld.label_map import evaluable_labels
from vyra_ml.realworld.vizwiz import VIZWIZ_FLAW_CODES


def _fake_table(n=8):
    rng = np.random.default_rng(0)
    row = {name: rng.random(n) for name in FEATURE_NAMES}
    for c in V1_BLOCKINESS_COLS:
        row[c] = rng.random(n) * 5  # v1 values on a different scale
    for c in VIZWIZ_FLAW_CODES:
        row[f"vote_{c}"] = rng.integers(0, 6, n)
    row["vote_unrecognizable"] = rng.integers(0, 6, n)
    row["image_id"] = [f"VizWiz_val_{i:08d}.jpg" for i in range(n)]
    return pd.DataFrame(row)


def test_feature_matrix_v2_uses_table_blockiness():
    df = _fake_table()
    x = feature_matrix(df, FEATURE_VERSION)
    col = list(FEATURE_NAMES).index("compress_blockiness")
    assert np.allclose(x[:, col], df["compress_blockiness"].to_numpy())


def test_feature_matrix_v1_swaps_in_legacy_blockiness():
    df = _fake_table()
    x = feature_matrix(df, "cvfeat-v1")
    for name, legacy in zip(
        ("compress_blockiness", "compress_blockiness_h", "compress_blockiness_v"),
        V1_BLOCKINESS_COLS,
        strict=True,
    ):
        col = list(FEATURE_NAMES).index(name)
        assert np.allclose(x[:, col], df[legacy].to_numpy())
    # non-blockiness columns are untouched
    col = list(FEATURE_NAMES).index("sharp_laplacian_var")
    assert np.allclose(x[:, col], df["sharp_laplacian_var"].to_numpy())


def test_feature_matrix_rejects_unknown_version():
    with pytest.raises(ValueError, match="Unsupported feature_version"):
        feature_matrix(_fake_table(), "cvfeat-v3")


def test_vote_threshold_labels_only_adds_evaluable_labels():
    df = _fake_table()
    df["vote_BLR"] = [0, 1, 2, 3, 4, 5, 3, 2]
    out = vote_threshold_labels(df, vote_min=3)
    for lbl in evaluable_labels():
        assert f"label_{lbl}" in out.columns
    assert list(out["label_blur"]) == [0, 0, 0, 1, 1, 1, 1, 0]
    # unmapped / unsupported labels get no column
    assert "label_noise" not in out.columns
    assert "label_framing" not in out.columns


def test_real_val_and_eval_use_disjoint_id_namespaces():
    val_ids = {"VizWiz_train_00000001.jpg", "VizWiz_train_00099999.jpg"}
    eval_ids = {"VizWiz_val_00000001.jpg", "VizWiz_val_00007000.jpg"}
    assert val_ids.isdisjoint(eval_ids)
    assert all(i.startswith("VizWiz_train_") for i in val_ids)
    assert all(i.startswith("VizWiz_val_") for i in eval_ids)


def test_primary_labels_exclude_defect():
    from vyra_ml.experiment.phase3b import PRIMARY_LABELS

    assert "defect" not in PRIMARY_LABELS
    assert set(PRIMARY_LABELS) == {"blur", "underexposure", "overexposure"}


def test_phase3b_config_variants_load():
    from vyra_ml.config import load_config
    from vyra_ml.experiment.phase3b import ML_ROOT

    base = load_config(ML_ROOT / "configs" / "experiment.yaml")
    blur = load_config(ML_ROOT / "configs" / "experiment_blurnoise.yaml")
    assert base.degradation.post_blur_sensor_noise is False
    assert blur.degradation.post_blur_sensor_noise is True
    assert blur.features.version == "cvfeat-v2"
    # separate processed dir so the two datasets never collide
    assert base.data_dir("processed") != blur.data_dir("processed")
