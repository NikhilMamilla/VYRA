"""The self-referential patch-anomaly defect detector."""

from __future__ import annotations

import cv2
import numpy as np

from vyra_ml.defect.patch_anomaly import DefectDetector, patch_anomaly_map


def _flat(colour: int = 128) -> np.ndarray:
    return np.full((400, 600, 3), colour, np.uint8)


def _textured() -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.integers(60, 200, (400, 600, 3))).astype(np.uint8)


def test_raw_score_is_bounded_and_finite() -> None:
    for img in (_flat(0), _flat(255), _flat(128), _textured()):
        m = patch_anomaly_map(img)
        assert 0.0 <= m["raw_score"] <= 25.0
        assert np.isfinite(m["raw_score"])


def test_flat_image_scores_zero() -> None:
    assert patch_anomaly_map(_flat())["raw_score"] == 0.0


def test_local_blotch_is_flagged_and_localised() -> None:
    img = _flat(130)
    img[120:170, 300:380] = (20, 20, 220)  # a saturated red patch
    m = patch_anomaly_map(img)
    assert m["raw_score"] > 5.0
    x, y, w, h = m["region_norm"]
    # region centre should land on the blotch (~[0.55, 0.36])
    assert 0.40 <= x + w / 2 <= 0.72
    assert 0.20 <= y + h / 2 <= 0.52


def test_tiny_image_degrades_gracefully() -> None:
    m = patch_anomaly_map(np.full((20, 20, 3), 100, np.uint8))
    assert m["raw_score"] == 0.0
    assert m["region_norm"] is None


def test_detector_roundtrip(tmp_path) -> None:
    d = DefectDetector(a=0.5, b=3.0, threshold=0.5, version="test", calibration={"x": 1})
    path = d.save(tmp_path / "d.json")
    loaded = DefectDetector.load(path)
    assert (loaded.a, loaded.b, loaded.threshold) == (0.5, 3.0, 0.5)


def test_detector_probability_is_monotone() -> None:
    d = DefectDetector(a=0.5, b=5.0, threshold=0.5, version="t", calibration={})
    ps = [d.probability_from_raw(r) for r in range(0, 20, 2)]
    assert ps == sorted(ps)
    assert 0.0 < ps[0] < 0.5 < ps[-1] < 1.0


def test_detector_analyze_coherent_image_not_flagged() -> None:
    d = DefectDetector(
        a=0.4715, b=17.85, threshold=0.5, version="phase3c-defect-v1", calibration={}
    )
    # A smooth gradient with light grain -- a locally coherent "photo".
    g = np.tile(np.linspace(30, 210, 600).astype(np.float32), (400, 1))
    rng = np.random.default_rng(0)
    coherent = np.clip(cv2.merge([g, g, g]) + rng.normal(0, 3, (400, 600, 3)), 0, 255).astype(
        np.uint8
    )
    result = d.analyze(coherent)
    assert not result.flagged
    assert result.region_norm is None
