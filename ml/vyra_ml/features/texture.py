"""Texture and frequency-structure features.

The radially-averaged power-spectrum slope is the most useful single texture
feature for this task: blur steepens it (energy concentrates at low frequency),
noise flattens it (energy added broadband). GLCM and LBP statistics add local
structure information.
"""

from __future__ import annotations

import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from vyra_ml.features.common import EPS, PreparedImage


def _spectral_slope(gray_f: np.ndarray) -> float:
    f = np.fft.fftshift(np.fft.fft2(gray_f - gray_f.mean()))
    power = np.abs(f) ** 2
    h, w = gray_f.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    r_max = min(cy, cx)
    radial = np.bincount(r.ravel(), power.ravel()) / (np.bincount(r.ravel()) + EPS)
    radial = radial[1:r_max]
    freqs = np.arange(1, len(radial) + 1)
    # Linear fit of log power vs log frequency.
    slope = np.polyfit(np.log(freqs), np.log(radial + EPS), 1)[0]
    return float(slope)


def compute(img: PreparedImage) -> dict[str, float]:
    gray = img.gray
    q = (gray // 32).astype(np.uint8)  # 8 grey levels
    glcm = graycomatrix(
        q, distances=[1], angles=[0, np.pi / 2], levels=8, symmetric=True, normed=True
    )

    lbp = local_binary_pattern(gray, P=8, R=1.0, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    lbp_entropy = float(-np.sum(lbp_hist[lbp_hist > 0] * np.log2(lbp_hist[lbp_hist > 0])))

    return {
        "texture_spectral_slope": _spectral_slope(img.gray_f),
        "texture_glcm_contrast": float(graycoprops(glcm, "contrast").mean()),
        "texture_glcm_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "texture_glcm_energy": float(graycoprops(glcm, "energy").mean()),
        "texture_lbp_entropy": lbp_entropy,
    }
