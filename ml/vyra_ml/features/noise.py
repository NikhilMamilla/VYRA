"""Noise features.

The hard part of no-reference noise estimation is separating noise from texture.
We use three complementary estimators, plus a flat-region estimate that only
looks at the smoothest tiles (where residual variance is most likely to be
noise rather than content).
"""

from __future__ import annotations

import cv2
import numpy as np

from vyra_ml.degradations.noise import estimate_noise_sigma
from vyra_ml.features.common import PreparedImage


def _flat_region_noise(gray_f: np.ndarray, tile: int = 24) -> float:
    h, w = gray_f.shape
    grad = np.abs(cv2.Laplacian(gray_f, cv2.CV_32F))
    stds, activities = [], []
    for y in range(0, h - tile, tile):
        for x in range(0, w - tile, tile):
            patch = gray_f[y : y + tile, x : x + tile]
            stds.append(float(patch.std()))
            activities.append(float(grad[y : y + tile, x : x + tile].mean()))
    if not stds:
        return float(gray_f.std())
    stds_arr = np.array(stds)
    activities_arr = np.array(activities)
    # Average std over the flattest 15% of tiles.
    k = max(1, int(0.15 * len(stds_arr)))
    flattest = np.argsort(activities_arr)[:k]
    return float(stds_arr[flattest].mean())


def compute(img: PreparedImage) -> dict[str, float]:
    gray_f = img.gray_f

    blurred = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=1.0)
    hf_residual = gray_f - blurred

    median = cv2.medianBlur(img.gray, 3).astype(np.float32) / 255.0
    mad = float(np.median(np.abs(gray_f - median)))

    return {
        "noise_immerkaer_sigma": estimate_noise_sigma(img.gray),
        "noise_highfreq_residual_std": float(hf_residual.std()),
        "noise_median_residual_mad": mad,
        "noise_flat_region_std": _flat_region_noise(gray_f),
    }
