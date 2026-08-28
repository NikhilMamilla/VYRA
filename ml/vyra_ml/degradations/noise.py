"""Image noise.

Randomly Gaussian, Poisson (shot), speckle or salt-and-pepper, applied mostly to
luminance with a little chroma noise so it looks sensor-like rather than a flat
additive layer.
"""

from __future__ import annotations

import cv2
import numpy as np

from vyra_ml.degradations.base import (
    Degradation,
    DegradationResult,
    lerp_range,
    to_float,
    to_uint8,
)

# Gaussian sigma on a 0-255 scale.
_GAUSS_SIGMA = {1: (3.0, 7.0), 2: (7.0, 13.0), 3: (13.0, 22.0), 4: (22.0, 38.0), 5: (38.0, 60.0)}
# Poisson: simulated full-well photon count. Fewer photons -> more shot noise.
_POISSON_PHOTONS = {1: (180, 320), 2: (90, 180), 3: (45, 90), 4: (22, 45), 5: (8, 22)}
# Salt-and-pepper fraction of pixels.
_SP_AMOUNT = {
    1: (0.001, 0.004),
    2: (0.004, 0.01),
    3: (0.01, 0.025),
    4: (0.025, 0.055),
    5: (0.055, 0.12),
}
_SPECKLE_VAR = {
    1: (0.002, 0.006),
    2: (0.006, 0.015),
    3: (0.015, 0.035),
    4: (0.035, 0.07),
    5: (0.07, 0.13),
}


class NoiseDegradation(Degradation):
    name = "noise"
    issue_label = "noise"

    def apply(
        self, image: np.ndarray, severity: int, rng: np.random.Generator
    ) -> DegradationResult:
        kind = rng.choice(
            ["gaussian", "poisson", "speckle", "salt_pepper"], p=[0.45, 0.25, 0.2, 0.1]
        )
        f = to_float(image)
        params: dict = {"kind": kind}

        if kind == "gaussian":
            sigma = lerp_range(rng, _GAUSS_SIGMA, severity) / 255.0
            luma = rng.normal(0, sigma, size=f.shape[:2])[..., None]
            chroma = rng.normal(0, sigma * 0.4, size=f.shape)
            out = f + luma + chroma
            params["sigma_255"] = round(sigma * 255, 2)
        elif kind == "poisson":
            photons = lerp_range(rng, _POISSON_PHOTONS, severity)
            out = rng.poisson(np.clip(f, 0, 1) * photons) / photons
            params["photons"] = round(photons, 1)
        elif kind == "speckle":
            var = lerp_range(rng, _SPECKLE_VAR, severity)
            out = f + f * rng.normal(0, np.sqrt(var), size=f.shape)
            params["variance"] = round(var, 4)
        else:  # salt_pepper
            amount = lerp_range(rng, _SP_AMOUNT, severity)
            out = f.copy()
            mask = rng.random(size=f.shape[:2])
            out[mask < amount / 2] = 0.0
            out[mask > 1 - amount / 2] = 1.0
            params["amount"] = round(amount, 4)

        return DegradationResult(image=to_uint8(np.clip(out, 0, 1)), params=params)


def estimate_noise_sigma(gray_uint8: np.ndarray) -> float:
    """Fast robust noise estimate (Immerkaer 1996): MAD-free Laplacian response.

    Shared by the feature extractor; kept here next to the noise model.
    """
    laplacian_mask = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    conv = cv2.filter2D(gray_uint8.astype(np.float32), -1, laplacian_mask)
    h, w = gray_uint8.shape[:2]
    return float(np.sqrt(np.pi / 2) * np.sum(np.abs(conv)) / (6 * max(1, (h - 2) * (w - 2))))
