"""Out-of-distribution stress test for the shipped bundle's real-trained heads.

Real VizWiz training data is narrow: blur is almost all mild defocus, "too dark"
is near-black, "too bright" is heavy highlight-clipping. A detector that only
works on that slice is not usable. This script applies **strong, unambiguous**
degradations that a person would obviously flag -- directional motion blur, zoom
blur, extreme under/overexposure -- to held-out BSDS500 photos and reports what
fraction the shipped model catches.

It is not a headline metric (synthetic images, known-degraded by construction);
it is a regression guard. Phase 3D's first cut (real-only training) scored 0.88
recall on the motion-blur set and let obvious blur through; mixing weighted
synthetic rows back into training took it to ~0.99.

Run: ``python ml/scripts/phase3d_stress_test.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from vyra_ml.inference import VyraQualityModel  # noqa: E402

SRC = ML_ROOT / "data" / "raw" / "bsds500" / "images"
BUNDLE = ML_ROOT / "artifacts" / "vyra-quality-model-v1"
N_SOURCES = 50


def _motion_kernel(size: int, angle: float) -> np.ndarray:
    k = np.zeros((size, size), np.float32)
    k[size // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, rot, (size, size))
    return k / k.sum()


def _zoom_blur(img: np.ndarray, strength: float = 0.04, steps: int = 12) -> np.ndarray:
    h, w = img.shape[:2]
    acc = np.zeros_like(img, np.float32)
    for i in range(steps):
        s = 1.0 + strength * (i / steps)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s)
        acc += cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32)
    return (acc / steps).astype(np.uint8)


def _gain(img: np.ndarray, g: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)


def main() -> None:
    if not SRC.is_dir():
        raise SystemExit(f"BSDS images not found at {SRC}; run scripts/pipeline.py build")
    model = VyraQualityModel.load(BUNDLE)

    srcs = sorted(list((SRC / "test").glob("*.jpg")) + list((SRC / "train").glob("*.jpg")))
    srcs = [p for p in srcs if cv2.imread(str(p)) is not None][:N_SOURCES]

    mblur, zblur, dark, bright = [], [], [], []
    for p in srcs:
        im = cv2.imread(str(p))
        for size, ang in [(21, 0.0), (31, 45.0), (41, 90.0), (27, 135.0)]:
            mblur.append(cv2.filter2D(im, -1, _motion_kernel(size, ang)))
        zblur.append(_zoom_blur(im, 0.05))
        dark.append(_gain(im, 0.14))
        for g in (3.0, 4.0):
            bright.append(_gain(im, g))  # >= ~40-80% of pixels clipped to white

    # (name, target issue, images, gate). gate=None -> reported, not enforced.
    groups: list[tuple[str, str, list, float | None]] = [
        ("motion_blur", "blur", mblur, 0.90),
        ("severe_underexposure", "underexposure", dark, 0.90),
        # Gross overexposure (40%+ of the frame pure white) must be caught by the
        # deterministic bright-clip floor (phase3d.ISSUE_FLOORS) even though the
        # learned head is weak. Milder / partial blowout is NOT gated -- there the
        # real head is conservative and VizWiz itself is inconsistent.
        ("gross_overexposure", "overexposure", bright, 0.90),
        # radial/zoom blur keeps a sharp centre, is not in the synthetic blur
        # set -- a known residual gap, reported not gated.
        ("zoom_blur", "blur", zblur, None),
    ]

    print(f"stress test -- shipped bundle {model.model_version}, {len(srcs)} source images\n")
    failures = []
    for name, want, imgs, gate in groups:
        hits = sum(
            any(i.issue == want and i.flagged for i in model.analyze_bgr(b).issues) for b in imgs
        )
        rate = hits / len(imgs)
        tag = "" if gate is None else ("  OK" if rate >= gate else f"  FAIL (<{gate:.2f})")
        print(f"  {name:22s} -> {want:14s} recall {rate:.2f}  ({hits}/{len(imgs)}){tag}")
        if gate is not None and rate < gate:
            failures.append(name)

    if failures:
        raise SystemExit(f"FAIL: {', '.join(failures)} below gate")
    print("\nPASS")


if __name__ == "__main__":
    main()
