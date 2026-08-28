"""Self-referential patch-anomaly detector for 'potential visual defect'.

Idea
----
An image is tiled into overlapping patches. Each patch gets a short vector of
cheap local statistics. A patch is *anomalous* to the degree its statistics
deviate -- as a robust z-score -- from the **median patch of the same image**.
The image-level defect score is the strongest patch anomaly, passed through a
logistic that was calibrated once on the synthetic defect set; the flagged
region is that patch.

Why this and not a classifier
-----------------------------
The five synthetic defect kinds (dead-pixel clusters, banding, block corruption,
colour blotches, occlusion) are all *local* departures from an otherwise
coherent image, so "unlike the rest of this image" is a reasonable proxy that
(a) needs no training data, (b) cannot memorise a synthetic fingerprint, and
(c) transfers as well to real images as to synthetic ones because the reference
is the image itself. Its ceiling is modest and it is a *screening* signal, not a
diagnosis -- see ``docs/defect.md`` for measured performance and failure modes.

The output is deliberately named "potential visual defect", never "defect".
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Working resolution for the patch grid. Independent of the CV feature engine's
# work_long_edge -- this is its own, coarser analysis.
_LONG_EDGE = 512
_PATCH = 64
_STRIDE = 32
_EPS = 1e-6
_Z_CLIP = 25.0  # a patch 25 robust-MADs out is "very anomalous"; beyond is noise
# Feature names in the per-patch vector, in order. Documented for explainability.
PATCH_FEATURES: tuple[str, ...] = (
    "resid_std",  # std of (patch - Gaussian blur): impulse / block noise
    "resid_max",  # max abs residual: dead / hot pixels, salt-and-pepper
    "lap_var",  # variance of Laplacian: local sharpness spikes and dead zones
    "blockiness",  # 8-px grid gradient excess: block corruption
    "sat_mean",  # mean HSV saturation: colour blotches
    "hue_std",  # circular std of hue on saturated pixels: colour blotches
    "luma_std",  # local luminance spread: flat occlusion (low) / noisy block (high)
    "edge_density",  # Canny edge fraction: foreign edges at an occlusion boundary
)


def _prepare(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    h, w = bgr.shape[:2]
    scale = _LONG_EDGE / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(
            bgr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA
        )
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, gray


def _patch_vector(bgr_patch: np.ndarray, gray_patch: np.ndarray) -> np.ndarray:
    g = gray_patch.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(g, (0, 0), 1.0)
    resid = g - blur
    resid_std = float(np.std(resid))
    resid_max = float(np.max(np.abs(resid)))

    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    lap_var = float(np.var(lap))

    # 8-px block-grid gradient excess (bounded normalised form, as in cvfeat-v2).
    dv = np.abs(np.diff(g, axis=1))
    dh = np.abs(np.diff(g, axis=0))
    bcols = np.arange(7, dv.shape[1], 8)
    brows = np.arange(7, dh.shape[0], 8)
    if len(bcols) and len(brows):
        b_v = dv[:, bcols].mean()
        i_v = np.delete(dv, bcols, axis=1).mean()
        b_h = dh[brows, :].mean()
        i_h = np.delete(dh, brows, axis=0).mean()
        block = 0.5 * ((b_v - i_v) / (b_v + i_v + _EPS) + (b_h - i_h) / (b_h + i_h + _EPS))
    else:
        block = 0.0
    blockiness = float(max(0.0, block))

    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    sat_mean = float(sat.mean())
    # Circular std of hue (0-180 in OpenCV) over reasonably saturated pixels.
    hue = hsv[:, :, 0].astype(np.float32) * (np.pi / 90.0)
    mask = sat > 0.15
    if mask.sum() >= 16:
        cbar = np.cos(hue[mask]).mean()
        sbar = np.sin(hue[mask]).mean()
        r = min(1.0, float(np.hypot(cbar, sbar)))
        hue_std = float(np.sqrt(max(0.0, -2.0 * np.log(max(r, 1e-6)))))
    else:
        hue_std = 0.0
    luma_std = float(np.std(g))

    edges = cv2.Canny(gray_patch, 60, 160)
    edge_density = float((edges > 0).mean())

    return np.array(
        [resid_std, resid_max, lap_var, blockiness, sat_mean, hue_std, luma_std, edge_density],
        dtype=np.float64,
    )


def patch_anomaly_map(bgr: np.ndarray) -> dict:
    """Compute the raw patch-anomaly map for one image.

    Returns a dict with the per-patch anomaly scores, the grid geometry and the
    strongest patch. All coordinates are in the internal ``_LONG_EDGE`` working
    resolution; ``region_norm`` gives the same box in [0, 1] image fractions.
    """
    bgr, gray = _prepare(bgr)
    h, w = gray.shape[:2]

    coords: list[tuple[int, int]] = []
    vectors: list[np.ndarray] = []
    for y in range(0, max(1, h - _PATCH + 1), _STRIDE):
        for x in range(0, max(1, w - _PATCH + 1), _STRIDE):
            gp = gray[y : y + _PATCH, x : x + _PATCH]
            bp = bgr[y : y + _PATCH, x : x + _PATCH]
            if gp.shape[0] < 16 or gp.shape[1] < 16:
                continue
            coords.append((x, y))
            vectors.append(_patch_vector(bp, gp))

    if len(vectors) < 4:
        return {
            "raw_score": 0.0,
            "n_patches": len(vectors),
            "grid": {"long_edge": _LONG_EDGE, "patch": _PATCH, "stride": _STRIDE},
            "region": None,
            "region_norm": None,
            "top_features": [],
        }

    mat = np.vstack(vectors)  # (n_patches, n_features)
    median = np.median(mat, axis=0)
    mad = np.median(np.abs(mat - median), axis=0)
    # Robust scale with a relative floor so a near-constant feature (mad ~= 0)
    # cannot manufacture an unbounded z-score from a rounding-level wobble.
    spread = np.percentile(mat, 90, axis=0) - np.percentile(mat, 10, axis=0)
    scale = 1.4826 * mad + 0.05 * np.abs(median) + 0.10 * spread + _EPS
    z = np.clip(np.abs(mat - median) / scale, 0.0, _Z_CLIP)  # (n_patches, n_features)

    # A patch's anomaly weights its single largest feature z-score (dead pixels
    # or a colour blotch spike one statistic hard) but requires some corroboration
    # from the next, so an isolated single-feature blip on a legitimate bright
    # window is damped.
    zs = np.sort(z, axis=1)
    patch_scores = 0.7 * zs[:, -1] + 0.3 * zs[:, -2]

    best = int(np.argmax(patch_scores))
    bx, by = coords[best]
    raw_score = float(patch_scores[best])
    feat_order = np.argsort(z[best])[::-1]
    top_features = [
        {"feature": PATCH_FEATURES[i], "z": round(float(z[best, i]), 2)}
        for i in feat_order[:3]
        if z[best, i] > 1.0
    ]

    return {
        "raw_score": raw_score,
        "n_patches": len(vectors),
        "grid": {"long_edge": _LONG_EDGE, "patch": _PATCH, "stride": _STRIDE},
        "region": [bx, by, _PATCH, _PATCH],
        "region_norm": [
            round(bx / w, 4),
            round(by / h, 4),
            round(_PATCH / w, 4),
            round(_PATCH / h, 4),
        ],
        "top_features": top_features,
    }


@dataclass(frozen=True)
class DefectResult:
    probability: float  # calibrated, [0, 1]
    flagged: bool
    raw_score: float
    region_norm: list[float] | None  # [x, y, w, h] in image fractions
    top_features: list[dict]
    note: str


@dataclass(frozen=True)
class DefectDetector:
    """Calibrated wrapper around :func:`patch_anomaly_map`.

    ``a`` / ``b`` are the logistic parameters ``P = sigmoid(a * (raw - b))`` and
    ``threshold`` the decision point, all fitted once by
    ``scripts/build_defect_detector.py`` on the synthetic defect set.
    """

    a: float
    b: float
    threshold: float
    version: str
    calibration: dict  # fit provenance + measured performance, for the bundle

    _NOTE = (
        "Screening signal only: a localised region that is statistically unlike "
        "the rest of the image. Not a confirmed physical defect."
    )

    @classmethod
    def load(cls, path: str | Path) -> DefectDetector:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            a=float(raw["a"]),
            b=float(raw["b"]),
            threshold=float(raw["threshold"]),
            version=raw.get("version", "unknown"),
            calibration=raw.get("calibration", {}),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "version": self.version,
                    "a": self.a,
                    "b": self.b,
                    "threshold": self.threshold,
                    "patch_features": list(PATCH_FEATURES),
                    "calibration": self.calibration,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def probability_from_raw(self, raw_score: float) -> float:
        return 1.0 / (1.0 + math.exp(-self.a * (raw_score - self.b)))

    def analyze(self, bgr: np.ndarray) -> DefectResult:
        amap = patch_anomaly_map(bgr)
        prob = self.probability_from_raw(amap["raw_score"])
        return DefectResult(
            probability=round(prob, 4),
            flagged=prob >= self.threshold,
            raw_score=round(amap["raw_score"], 4),
            region_norm=amap["region_norm"] if prob >= self.threshold else None,
            top_features=amap["top_features"],
            note=self._NOTE,
        )
