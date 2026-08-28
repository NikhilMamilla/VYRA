"""Feature extraction entry point.

``extract_features(bgr)`` returns an ordered dict of ~45 interpretable scalar
features. ``FEATURE_NAMES`` is the canonical ordering; ``FEATURE_DESCRIPTIONS``
documents every one. Groups live in sibling modules so each stays small and
independently testable.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from vyra_ml.features import (
    color,
    compression,
    exposure,
    noise,
    sharpness,
    texture,
)
from vyra_ml.features.common import prepare

# cvfeat-v2: compress_blockiness* changed from an unbounded ratio to a bounded
# normalised-excess form (see vyra_ml/features/compression.py). No other feature
# changed. Feature set, ordering and count are identical to v1.
FEATURE_VERSION = "cvfeat-v2"

_GROUPS = (sharpness, exposure, noise, color, texture, compression)

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "sharp_laplacian_var": "Variance of the Laplacian; classic focus measure, falls with blur.",
    "sharp_laplacian_var_norm": "Laplacian variance divided by mean-luminance^2; blur measure robust to darkness.",
    "sharp_tenengrad": "Mean squared Sobel gradient magnitude (Tenengrad focus measure).",
    "sharp_grad_mean": "Mean gradient magnitude.",
    "sharp_grad_p90": "90th-percentile gradient magnitude; strong-edge sharpness.",
    "sharp_modified_laplacian": "Mean modified Laplacian (Nayar); sum of abs 2nd derivatives.",
    "sharp_highfreq_ratio": "Fraction of FFT magnitude beyond 0.25*min(H,W) radius; high-freq energy.",
    "sharp_edge_density": "Fraction of pixels flagged as edges by Canny.",
    "expo_luma_mean": "Mean luminance in [0,1].",
    "expo_luma_median": "Median luminance in [0,1].",
    "expo_dark_clip_ratio": "Fraction of pixels below 5/255 (crushed blacks).",
    "expo_bright_clip_ratio": "Fraction of pixels above 250/255 (blown highlights).",
    "expo_shadow_ratio": "Fraction of pixels below 0.25.",
    "expo_highlight_ratio": "Fraction of pixels above 0.75.",
    "expo_hist_entropy": "Shannon entropy of the 256-bin luminance histogram.",
    "expo_skew": "Skewness of the luminance distribution (sign of exposure bias).",
    "contrast_std": "Standard deviation of luminance (RMS contrast).",
    "contrast_dynamic_range": "p95 - p5 of luminance.",
    "contrast_michelson": "(p99-p1)/(p99+p1) Michelson contrast, robust percentiles.",
    "contrast_rms_coeff_var": "Luminance std / mean (coefficient of variation).",
    "contrast_local_std_mean": "Mean of local (15x15) standard deviation; local contrast.",
    "contrast_p95": "95th percentile of luminance.",
    "contrast_p5": "5th percentile of luminance.",
    "noise_immerkaer_sigma": "Immerkaer (1996) fast noise sigma estimate via Laplacian mask.",
    "noise_highfreq_residual_std": "Std of (image - Gaussian(sigma=1)); broadband high-freq residual.",
    "noise_median_residual_mad": "Median abs deviation of (image - 3x3 median); impulse/grain noise.",
    "noise_flat_region_std": "Mean std over the flattest 15% of 24px tiles; content-suppressed noise.",
    "color_saturation_mean": "Mean HSV saturation in [0,1].",
    "color_saturation_std": "Std of HSV saturation.",
    "color_colourfulness": "Hasler-Suesstrunk colourfulness metric.",
    "color_cast": "max-min of the per-channel mean (absolute colour cast).",
    "color_cast_ratio": "Colour cast normalised by overall brightness.",
    "color_gray_pixel_ratio": "Fraction of near-neutral pixels (channel spread < 12).",
    "color_channel_std_mean": "Mean per-channel std, normalised to [0,1].",
    "texture_spectral_slope": "Slope of log radial power spectrum vs log frequency.",
    "texture_glcm_contrast": "GLCM contrast (8 levels, d=1, 0 and 90 deg).",
    "texture_glcm_homogeneity": "GLCM homogeneity.",
    "texture_glcm_energy": "GLCM energy (angular second moment).",
    "texture_lbp_entropy": "Entropy of the uniform LBP(8,1) histogram.",
    "compress_blockiness": "Mean of the H and V normalised 8px-grid blocking excess, in [0,1).",
    "compress_blockiness_h": "(boundary-interior)/(boundary+interior) gradient across vertical 8px lines.",
    "compress_blockiness_v": "(boundary-interior)/(boundary+interior) gradient across horizontal 8px lines.",
}

FEATURE_NAMES: tuple[str, ...] = tuple(FEATURE_DESCRIPTIONS.keys())


def extract_features(bgr: np.ndarray, work_long_edge: int = 384) -> OrderedDict[str, float]:
    prepared = prepare(bgr, work_long_edge)
    out: OrderedDict[str, float] = OrderedDict()
    for group in _GROUPS:
        out.update(group.compute(prepared))

    # Guarantee the documented ordering and set — fail loudly on drift.
    if tuple(out.keys()) != FEATURE_NAMES:
        missing = set(FEATURE_NAMES) - set(out)
        extra = set(out) - set(FEATURE_NAMES)
        raise RuntimeError(f"Feature set drift. missing={missing} extra={extra}")
    return out


def features_to_vector(feats: dict[str, float]) -> np.ndarray:
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float64)
