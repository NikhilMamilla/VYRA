from __future__ import annotations

import cv2
import numpy as np
import pytest

from vyra_ml.degradations import DEGRADATIONS, SEVERITIES, get_degradation
from vyra_ml.degradations.noise import estimate_noise_sigma
from vyra_ml.seeding import derive_rng


def _sharpness(img: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def _brightness(img: np.ndarray) -> float:
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())


@pytest.mark.parametrize("name", list(DEGRADATIONS))
@pytest.mark.parametrize("severity", SEVERITIES)
def test_output_is_valid_image(synthetic_image, name, severity):
    rng = derive_rng(0, name, severity)
    result = get_degradation(name).apply(synthetic_image, severity, rng)

    assert result.image.shape == synthetic_image.shape
    assert result.image.dtype == np.uint8
    assert np.isfinite(result.image).all()
    assert result.params  # every degradation records what it sampled


@pytest.mark.parametrize("name", list(DEGRADATIONS))
def test_deterministic_given_seed(synthetic_image, name):
    a = get_degradation(name).apply(synthetic_image, 3, derive_rng(1, name))
    b = get_degradation(name).apply(synthetic_image, 3, derive_rng(1, name))
    assert np.array_equal(a.image, b.image)
    assert a.params == b.params


def test_blur_reduces_sharpness_monotonically(synthetic_image):
    sharp = [
        _sharpness(get_degradation("blur").apply(synthetic_image, s, derive_rng(s, "b")).image)
        for s in SEVERITIES
    ]
    assert _sharpness(synthetic_image) > sharp[0]
    # Non-increasing across severities (allow tiny noise).
    assert all(sharp[i] >= sharp[i + 1] - 5 for i in range(len(sharp) - 1))


def test_underexposure_darkens_and_overexposure_brightens(synthetic_image):
    base = _brightness(synthetic_image)
    dark = _brightness(
        get_degradation("underexposure").apply(synthetic_image, 4, derive_rng(0, "u")).image
    )
    bright = _brightness(
        get_degradation("overexposure").apply(synthetic_image, 4, derive_rng(0, "o")).image
    )
    assert dark < base < bright


def test_noise_increases_noise_estimate(synthetic_image):
    base = estimate_noise_sigma(cv2.cvtColor(synthetic_image, cv2.COLOR_BGR2GRAY))
    noisy = get_degradation("noise").apply(synthetic_image, 5, derive_rng(0, "n")).image
    assert estimate_noise_sigma(cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)) > base * 1.5


def test_severity_scales_parameters(synthetic_image):
    # Sampled parameters differ between mild and extreme severity.
    mild = get_degradation("noise").apply(synthetic_image, 1, derive_rng(3, "n")).params
    extreme = get_degradation("noise").apply(synthetic_image, 5, derive_rng(3, "n")).params
    assert mild != extreme


def test_corruption_shrinks_file_size(synthetic_image):
    result = get_degradation("corruption").apply(synthetic_image, 5, derive_rng(0, "c"))
    ok_clean, clean_buf = cv2.imencode(".png", synthetic_image)
    ok_deg, deg_buf = cv2.imencode(".jpg", result.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok_clean and ok_deg
    assert result.params["jpeg_quality"] <= 15
