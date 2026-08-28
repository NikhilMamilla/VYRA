"""Blur / insufficient sharpness.

Randomly either an out-of-focus (Gaussian) blur or a motion blur, so the model
cannot key on one kernel shape. Parameters are sampled from severity-dependent
ranges.
"""

from __future__ import annotations

import cv2
import numpy as np

from vyra_ml.degradations.base import Degradation, DegradationResult, lerp_range

# Gaussian blur sigma (px) by severity. Sev 1 is near the perceptual threshold
# for a ~384px image; sev 5 is heavy defocus.
_SIGMA_RANGES = {
    1: (0.6, 1.1),
    2: (1.1, 1.9),
    3: (1.9, 3.0),
    4: (3.0, 4.8),
    5: (4.8, 7.5),
}
# Motion blur path length (px) by severity.
_MOTION_RANGES = {
    1: (3.0, 6.0),
    2: (6.0, 11.0),
    3: (11.0, 19.0),
    4: (19.0, 31.0),
    5: (31.0, 48.0),
}


def _motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    length = max(3, length | 1)  # odd, >= 3
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    total = kernel.sum()
    return kernel / total if total > 0 else kernel


class BlurDegradation(Degradation):
    name = "blur"
    issue_label = "blur"

    def apply(
        self, image: np.ndarray, severity: int, rng: np.random.Generator
    ) -> DegradationResult:
        kind = rng.choice(["gaussian", "motion"], p=[0.6, 0.4])

        if kind == "gaussian":
            sigma = lerp_range(rng, _SIGMA_RANGES, severity)
            ksize = int(2 * np.ceil(3 * sigma) + 1)
            out = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
            params = {"kind": "gaussian", "sigma": round(sigma, 3), "kernel_size": ksize}
        else:
            length = int(round(lerp_range(rng, _MOTION_RANGES, severity)))
            angle = float(rng.uniform(0, 180))
            kernel = _motion_kernel(length, angle)
            out = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT101)
            params = {
                "kind": "motion",
                "length": length,
                "angle_deg": round(angle, 1),
                "kernel_size": kernel.shape[0],
            }

        return DegradationResult(image=out, params=params)
