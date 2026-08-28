"""The bundled inference entry point (vyra_ml.inference)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vyra_ml.inference import VyraQualityModel

BUNDLE = Path(__file__).resolve().parents[1] / "artifacts" / "vyra-quality-model-v1"
pytestmark = pytest.mark.skipif(not (BUNDLE / "bundle.json").is_file(), reason="bundle not built")


def _img(kind: str) -> np.ndarray:
    g = np.tile(np.linspace(20, 235, 512).astype(np.uint8), (384, 1))
    bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(bgr, (120, 90), (360, 280), (255, 255, 255), 2)
    if kind == "blurry":
        bgr = cv2.GaussianBlur(bgr, (0, 0), 6.0)
    elif kind == "dark":
        bgr = (bgr * 0.15).astype(np.uint8)
    return bgr


@pytest.fixture(scope="module")
def model() -> VyraQualityModel:
    return VyraQualityModel.load(BUNDLE)


def test_bundle_metadata(model: VyraQualityModel) -> None:
    assert model.model_version == "vyra-quality-model-v1"
    assert model.feature_version == "cvfeat-v2"
    d = model.describe()
    assert set(d["capabilities"]["real_world_validated"]) == {
        "blur",
        "underexposure",
        "overexposure",
    }


def test_analyze_shape_and_bounds(model: VyraQualityModel) -> None:
    r = model.analyze_bgr(_img("clean"))
    assert 0.0 <= r.quality_score <= 100.0
    assert r.quality_label in {"GOOD", "ACCEPTABLE", "DEGRADED", "POOR"}
    assert len(r.features) == 42
    assert {i.issue for i in r.issues} == set(model.describe()["issues"])
    for i in r.issues:
        assert 0.0 <= i.probability <= 1.0
        assert (i.severity is None) == (not i.flagged)


def test_deterministic(model: VyraQualityModel) -> None:
    a = model.analyze_bgr(_img("blurry"))
    b = model.analyze_bgr(_img("blurry"))
    assert a.quality_score == b.quality_score
    assert [i.probability for i in a.issues] == [i.probability for i in b.issues]
    assert a.potential_defect.probability == b.potential_defect.probability


def test_blur_and_dark_are_detected(model: VyraQualityModel) -> None:
    blurry = model.analyze_bgr(_img("blurry"))
    assert next(i for i in blurry.issues if i.issue == "blur").flagged
    dark = model.analyze_bgr(_img("dark"))
    assert next(i for i in dark.issues if i.issue == "underexposure").flagged
    assert dark.quality_score < 100.0


def test_undecodable_bytes_raise(model: VyraQualityModel) -> None:
    with pytest.raises(ValueError):
        model.analyze_bytes(b"not an image at all")


def test_noise_corruption_marked_synthetic_only(model: VyraQualityModel) -> None:
    r = model.analyze_bgr(_img("clean"))
    by_issue = {i.issue: i for i in r.issues}
    assert by_issue["noise"].validation == "synthetic-only"
    assert by_issue["corruption"].validation == "synthetic-only"
    assert by_issue["noise"].calibrated is False
