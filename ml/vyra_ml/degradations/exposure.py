"""Under- and over-exposure.

Modelled in a light-linear domain as an exposure shift in stops (EV), plus a
small gamma nudge and, for underexposure, elevated read noise -- dark frames are
genuinely noisier, and omitting that would make underexposure separable from
noise too cleanly.
"""

from __future__ import annotations

import numpy as np

from vyra_ml.degradations.base import (
    Degradation,
    DegradationResult,
    lerp_range,
    to_float,
    to_uint8,
)

_UNDER_STOPS = {
    1: (0.3, 0.7),
    2: (0.7, 1.2),
    3: (1.2, 1.9),
    4: (1.9, 2.7),
    5: (2.7, 3.8),
}
_OVER_STOPS = {
    1: (0.3, 0.7),
    2: (0.7, 1.2),
    3: (1.2, 1.8),
    4: (1.8, 2.6),
    5: (2.6, 3.6),
}


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1 / 2.4)) - 0.055)


class UnderexposureDegradation(Degradation):
    name = "underexposure"
    issue_label = "underexposure"

    def apply(
        self, image: np.ndarray, severity: int, rng: np.random.Generator
    ) -> DegradationResult:
        stops = lerp_range(rng, _UNDER_STOPS, severity)
        linear = _srgb_to_linear(to_float(image)) * (2.0**-stops)

        read_noise = (0.004 + 0.006 * severity) * float(rng.uniform(0.6, 1.4))
        linear = linear + rng.normal(0.0, read_noise, size=linear.shape)

        black_lift = float(rng.uniform(0.0, 0.02 * severity))
        out = _linear_to_srgb(np.clip(linear, 0.0, 1.0)) * (1 - black_lift) + black_lift
        return DegradationResult(
            image=to_uint8(out),
            params={
                "stops": round(stops, 3),
                "read_noise_sigma": round(read_noise, 4),
                "black_lift": round(black_lift, 4),
            },
        )


class OverexposureDegradation(Degradation):
    name = "overexposure"
    issue_label = "overexposure"

    def apply(
        self, image: np.ndarray, severity: int, rng: np.random.Generator
    ) -> DegradationResult:
        stops = lerp_range(rng, _OVER_STOPS, severity)
        linear = _srgb_to_linear(to_float(image)) * (2.0**stops)

        # Soft shoulder at low severity, hard clip at high -- mimics how cameras
        # roll off highlights before they blow out entirely.
        knee = float(np.interp(severity, [1, 5], [0.9, 0.999]))
        over = linear > knee
        linear[over] = knee + (1 - knee) * np.tanh((linear[over] - knee) / (1 - knee))

        out = _linear_to_srgb(np.clip(linear, 0.0, 1.0))
        clipped_ratio = float(np.mean(to_float(image) * 0 + (linear >= 0.999)))
        return DegradationResult(
            image=to_uint8(out),
            params={
                "stops": round(stops, 3),
                "knee": round(knee, 4),
                "clipped_ratio": round(clipped_ratio, 4),
            },
        )
