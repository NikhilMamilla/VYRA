from __future__ import annotations

import cv2
import numpy as np

from vyra_ml.degradations import get_degradation
from vyra_ml.features import FEATURE_DESCRIPTIONS, FEATURE_NAMES, extract_features
from vyra_ml.seeding import derive_rng


def test_feature_set_is_stable_and_documented():
    feats = extract_features(np.full((200, 200, 3), 127, np.uint8))
    assert tuple(feats) == FEATURE_NAMES
    assert set(FEATURE_DESCRIPTIONS) == set(FEATURE_NAMES)


def test_features_are_finite_on_edge_case_images():
    cases = [
        np.zeros((128, 160, 3), np.uint8),  # pure black
        np.full((128, 160, 3), 255, np.uint8),  # pure white
        np.full((128, 160, 3), 127, np.uint8),  # flat grey
        (np.random.default_rng(0).random((128, 160, 3)) * 255).astype(np.uint8),  # pure noise
    ]
    for img in cases:
        values = np.array(list(extract_features(img).values()))
        assert np.isfinite(values).all()


def test_size_robustness(synthetic_image):
    a = np.array(list(extract_features(synthetic_image, 288).values()))
    b = np.array(list(extract_features(synthetic_image, 448).values()))
    rel = np.abs(a - b) / (np.maximum(np.abs(a), np.abs(b)) + 1e-6)
    assert np.median(rel) < 0.15


def test_blur_lowers_sharpness_features(synthetic_image):
    clean = extract_features(synthetic_image)
    blurred = extract_features(
        get_degradation("blur").apply(synthetic_image, 5, derive_rng(0, "b")).image
    )
    assert blurred["sharp_laplacian_var"] < clean["sharp_laplacian_var"]
    assert blurred["sharp_edge_density"] < clean["sharp_edge_density"]


def test_noise_raises_noise_features(synthetic_image):
    clean = extract_features(synthetic_image)
    noisy = extract_features(
        get_degradation("noise").apply(synthetic_image, 5, derive_rng(0, "n")).image
    )
    assert noisy["noise_immerkaer_sigma"] > clean["noise_immerkaer_sigma"]


def test_underexposure_lowers_luma_feature(synthetic_image):
    clean = extract_features(synthetic_image)
    dark = extract_features(
        get_degradation("underexposure").apply(synthetic_image, 4, derive_rng(0, "u")).image
    )
    assert dark["expo_luma_mean"] < clean["expo_luma_mean"]
    assert dark["expo_shadow_ratio"] > clean["expo_shadow_ratio"]


def test_grayscale_input_is_accepted():
    gray = cv2.cvtColor(np.full((100, 120, 3), 90, np.uint8), cv2.COLOR_BGR2GRAY)
    feats = extract_features(gray)
    assert np.isfinite(np.array(list(feats.values()))).all()
