"""Phase 3B calibration: threshold sweep/selection and probability calibration.

All fitting here uses synthetic arrays -- the point is the mechanics, and that
the evaluation set is never touched (enforced structurally in the orchestrator).
"""

from __future__ import annotations

import numpy as np
import pytest

from vyra_ml.calibration import (
    PerLabelCalibrator,
    ThresholdSet,
    brier_score,
    expected_calibration_error,
    fit_calibrators,
    reliability_curve,
    select_threshold,
    sweep_thresholds,
)


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=400)
    # score correlated with truth but poorly calibrated (compressed toward 0.5)
    score = np.clip(0.5 + 0.15 * (y * 2 - 1) + rng.normal(0, 0.1, size=400), 0, 1)
    return y, score


def test_sweep_is_monotone_in_coverage(scored):
    y, s = scored
    sweep = sweep_thresholds(y, s)
    tps = [r["tp"] for r in sweep]
    assert tps == sorted(tps, reverse=True)  # higher threshold -> fewer positives
    assert all(0 <= r["f1"] <= 1 for r in sweep)


def test_select_threshold_maximises_f1_and_is_deterministic(scored):
    y, s = scored
    t1, detail1 = select_threshold(y, s)
    t2, _ = select_threshold(y, s)
    assert t1 == t2
    # the chosen row's F1 is the max over the sweep
    best = max(r["f1"] for r in sweep_thresholds(y, s))
    assert abs(detail1["f1"] - best) < 1e-9


def test_select_threshold_falls_back_below_min_support():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    t, detail = select_threshold(y, np.linspace(0, 1, 10), min_support=20)
    assert t == 0.5
    assert "positives" in detail["note"]


def test_threshold_set_roundtrip(tmp_path):
    ts = ThresholdSet(
        version="v",
        parent="m",
        model_artifact="x.joblib",
        feature_version="cvfeat-v2",
        criterion="f1",
        fitted_on="real val",
        seed=1,
        thresholds={"blur": 0.3, "noise": 0.5},
    )
    p = ts.save(tmp_path / "t.json")
    back = ThresholdSet.load(p)
    assert back.thresholds == {"blur": 0.3, "noise": 0.5}
    assert back.feature_version == "cvfeat-v2"


def test_brier_and_ece_and_reliability(scored):
    y, s = scored
    assert 0 <= brier_score(y, s) <= 1
    assert 0 <= expected_calibration_error(y, s) <= 1
    curve = reliability_curve(y, s, n_bins=5)
    assert len(curve["bins"]) == 5
    assert sum(b["count"] for b in curve["bins"]) == len(y)


def test_isotonic_calibration_improves_or_is_dropped(scored):
    y, s = scored
    cal = fit_calibrators({"blur": s}, {"blur": y}, version="v", parent="m", min_support=10)
    diag = cal.diagnostics["blur"]
    if diag["calibrated"]:
        assert diag["after"]["brier"] <= diag["before"]["brier"] + 1e-4
        p = cal.transform("blur", s)
        assert p.min() >= 0 and p.max() <= 1
    else:
        # dropped => transform is identity
        assert np.allclose(cal.transform("blur", s), s)


def test_calibrator_not_fitted_on_tiny_support():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    cal = fit_calibrators(
        {"defect": np.linspace(0, 1, 8)}, {"defect": y}, version="v", parent="m", min_support=40
    )
    assert cal.diagnostics["defect"]["calibrated"] is False
    assert cal.models["defect"] is None


def test_calibrator_roundtrip(tmp_path, scored):
    y, s = scored
    cal = fit_calibrators({"blur": s}, {"blur": y}, version="v", parent="m", min_support=10)
    p = cal.save(tmp_path / "c.joblib")
    back = PerLabelCalibrator.load(p)
    assert np.allclose(back.transform("blur", s), cal.transform("blur", s))
