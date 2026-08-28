"""Sharpness / blur features.

All computed on the fixed-resolution working greyscale image so they are
comparable across source resolutions. Blur reduces high-frequency energy and
edge strength, so these features fall as blur severity rises.
"""

from __future__ import annotations

import cv2
import numpy as np

from vyra_ml.features.common import EPS, PreparedImage, safe_ratio


def compute(img: PreparedImage) -> dict[str, float]:
    gray = img.gray
    gray_f = img.gray_f

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())

    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)

    # Modified Laplacian (Nayar): sum of abs second derivatives along each axis.
    kx = np.array([[-1, 2, -1]], np.float32)
    mlv = np.abs(cv2.filter2D(gray_f, cv2.CV_32F, kx)) + np.abs(
        cv2.filter2D(gray_f, cv2.CV_32F, kx.T)
    )

    # High-frequency energy ratio via FFT.
    f = np.fft.fftshift(np.fft.fft2(gray_f))
    mag = np.abs(f)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    high = radius > 0.25 * min(h, w)
    hf_ratio = safe_ratio(float(mag[high].sum()), float(mag.sum()))

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.count_nonzero(edges) / edges.size)

    brightness = float(gray_f.mean())
    return {
        "sharp_laplacian_var": lap_var,
        # Brightness-normalised: a dark image has low Laplacian variance without
        # being blurred, so we divide it out.
        "sharp_laplacian_var_norm": lap_var / (brightness**2 * 255.0**2 + EPS),
        "sharp_tenengrad": float(np.mean(grad_mag**2)),
        "sharp_grad_mean": float(grad_mag.mean()),
        "sharp_grad_p90": float(np.percentile(grad_mag, 90)),
        "sharp_modified_laplacian": float(mlv.mean()),
        "sharp_highfreq_ratio": hf_ratio,
        "sharp_edge_density": edge_density,
    }
