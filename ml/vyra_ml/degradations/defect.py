"""Potential visual defect -- localised, structural artefacts.

Distinct from the global degradations: a defect affects a *region* of the frame.
One of several defect types is chosen at random (dead-pixel clusters, sensor
banding, block/decode corruption, colour blotch, lens occlusion), with location
and extent sampled per sample so there is no fixed spatial fingerprint.
"""

from __future__ import annotations

import cv2
import numpy as np

from vyra_ml.degradations.base import Degradation, DegradationResult, to_float, to_uint8

# Fraction of image area the defect covers, by severity.
_AREA_FRAC = {1: (0.01, 0.03), 2: (0.03, 0.07), 3: (0.07, 0.14), 4: (0.14, 0.28), 5: (0.28, 0.5)}


def _region(shape, area_frac: float, rng: np.random.Generator) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    aspect = float(rng.uniform(0.5, 2.0))
    rh = int(np.clip(np.sqrt(area_frac * h * w / aspect), 4, h))
    rw = int(np.clip(rh * aspect, 4, w))
    y0 = int(rng.integers(0, max(1, h - rh)))
    x0 = int(rng.integers(0, max(1, w - rw)))
    return x0, y0, rw, rh


class DefectDegradation(Degradation):
    name = "defect"
    issue_label = "defect"

    def apply(
        self, image: np.ndarray, severity: int, rng: np.random.Generator
    ) -> DegradationResult:
        kind = rng.choice(
            ["dead_pixels", "banding", "block_corruption", "color_blotch", "occlusion"],
            p=[0.22, 0.2, 0.22, 0.18, 0.18],
        )
        area = float(rng.uniform(*_AREA_FRAC[severity]))
        x0, y0, rw, rh = _region(image.shape, area, rng)
        out = image.copy()
        roi = out[y0 : y0 + rh, x0 : x0 + rw]
        params: dict = {"kind": kind, "bbox": [x0, y0, rw, rh], "area_frac": round(area, 4)}

        if kind == "dead_pixels":
            density = 0.02 + 0.12 * severity / 5
            mask = rng.random(roi.shape[:2])
            roi[mask < density / 2] = 0
            roi[mask > 1 - density / 2] = 255
            params["density"] = round(density, 4)
        elif kind == "banding":
            freq = float(rng.uniform(0.05, 0.3))
            amp = (0.05 + 0.25 * severity / 5) * 255
            rows = np.arange(rh)[:, None, None]
            stripes = (amp * np.sin(2 * np.pi * freq * rows)).astype(np.float32)
            roi[:] = np.clip(roi.astype(np.float32) + stripes, 0, 255).astype(np.uint8)
            params.update(freq=round(freq, 4), amplitude_255=round(amp, 2))
        elif kind == "block_corruption":
            block = max(4, int(min(rw, rh) / rng.integers(3, 8)))
            small = cv2.resize(
                roi, (max(1, rw // block), max(1, rh // block)), interpolation=cv2.INTER_NEAREST
            )
            shifted = np.roll(small, int(rng.integers(1, 4)), axis=rng.integers(0, 2))
            roi[:] = cv2.resize(shifted, (rw, rh), interpolation=cv2.INTER_NEAREST)
            params["block_px"] = block
        elif kind == "color_blotch":
            colour = rng.uniform(0, 1, size=3)
            strength = 0.25 + 0.5 * severity / 5
            yy, xx = np.mgrid[0:rh, 0:rw]
            g = np.exp(-(((yy - rh / 2) / (rh / 3)) ** 2 + ((xx - rw / 2) / (rw / 3)) ** 2))
            blend = (strength * g)[..., None]
            roi[:] = to_uint8(to_float(roi) * (1 - blend) + colour * blend)
            params.update(
                color_bgr=[round(float(c), 3) for c in colour], strength=round(strength, 3)
            )
        else:  # occlusion
            shade = float(rng.uniform(0.0, 0.25))
            feather = cv2.GaussianBlur(
                np.ones((rh, rw), np.float32), (0, 0), sigmaX=max(rw, rh) / 8
            )[..., None]
            roi[:] = to_uint8(to_float(roi) * (1 - feather) + shade * feather)
            params["shade"] = round(shade, 3)

        out[y0 : y0 + rh, x0 : x0 + rw] = roi
        return DegradationResult(image=out, params=params)
