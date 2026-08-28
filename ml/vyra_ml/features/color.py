"""Colour / saturation features."""

from __future__ import annotations

import numpy as np

from vyra_ml.features.common import EPS, PreparedImage, safe_ratio


def compute(img: PreparedImage) -> dict[str, float]:
    bgr = img.bgr.astype(np.float32)
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    sat = img.hsv[..., 1].astype(np.float32) / 255.0

    # Hasler-Suesstrunk colourfulness.
    rg = r - g
    yb = 0.5 * (r + g) - b
    colourfulness = float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )

    # Colour cast: how far the average pixel is from neutral.
    mean_bgr = bgr.reshape(-1, 3).mean(axis=0)
    cast = float(np.max(mean_bgr) - np.min(mean_bgr))

    near_gray = (np.max(bgr, axis=2) - np.min(bgr, axis=2)) < 12
    return {
        "color_saturation_mean": float(sat.mean()),
        "color_saturation_std": float(sat.std()),
        "color_colourfulness": colourfulness,
        "color_cast": cast,
        "color_cast_ratio": safe_ratio(cast, float(mean_bgr.mean())),
        "color_gray_pixel_ratio": float(near_gray.mean()),
        "color_channel_std_mean": float(np.mean([b.std(), g.std(), r.std()]) / (255.0 + EPS)),
    }
