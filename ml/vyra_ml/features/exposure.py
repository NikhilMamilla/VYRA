"""Exposure and contrast features.

Exposure: where the luminance mass sits and how much is clipped at the ends.
Contrast: how spread out the luminance is. Kept in one module because both read
the same luminance histogram.
"""

from __future__ import annotations

import cv2
import numpy as np

from vyra_ml.features.common import EPS, PreparedImage, safe_ratio


def compute(img: PreparedImage) -> dict[str, float]:
    y = img.gray_f
    flat = y.ravel()

    hist = cv2.calcHist([img.gray], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + EPS)
    entropy = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))

    p1, p5, p50, p95, p99 = (float(np.percentile(flat, q)) for q in (1, 5, 50, 95, 99))
    mean = float(flat.mean())
    std = float(flat.std())

    # Local contrast: mean of the per-window standard deviation.
    mu = cv2.blur(y, (15, 15))
    mu2 = cv2.blur(y * y, (15, 15))
    local_std = np.sqrt(np.clip(mu2 - mu * mu, 0, None))

    return {
        # --- exposure ---
        "expo_luma_mean": mean,
        "expo_luma_median": p50,
        "expo_dark_clip_ratio": float(np.mean(flat < 5 / 255)),
        "expo_bright_clip_ratio": float(np.mean(flat > 250 / 255)),
        "expo_shadow_ratio": float(np.mean(flat < 0.25)),
        "expo_highlight_ratio": float(np.mean(flat > 0.75)),
        "expo_hist_entropy": entropy,
        "expo_skew": float(((flat - mean) ** 3).mean() / (std**3 + EPS)),
        # --- contrast ---
        "contrast_std": std,
        "contrast_dynamic_range": p95 - p5,
        "contrast_michelson": safe_ratio(p99 - p1, p99 + p1),
        "contrast_rms_coeff_var": safe_ratio(std, mean),
        "contrast_local_std_mean": float(local_std.mean()),
        "contrast_p95": p95,
        "contrast_p5": p5,
    }
