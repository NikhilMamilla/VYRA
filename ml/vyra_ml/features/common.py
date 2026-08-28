"""Shared image preparation for feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

EPS = 1e-8


@dataclass(frozen=True)
class PreparedImage:
    """An image resized to a common working resolution, with cached views."""

    bgr: np.ndarray  # uint8, HxWx3
    gray: np.ndarray  # uint8, HxW  (BT.601 luma)
    gray_f: np.ndarray  # float32 in [0,1]
    hsv: np.ndarray  # uint8, HxWx3
    lab: np.ndarray  # uint8, HxWx3

    @property
    def shape(self) -> tuple[int, int]:
        return self.gray.shape[:2]


def prepare(bgr: np.ndarray, work_long_edge: int) -> PreparedImage:
    """Resize so the longest edge is ``work_long_edge`` and precompute colour views.

    A fixed working resolution makes every feature comparable across source
    images of different sizes (see the size-robustness check in the feature
    report).
    """
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    h, w = bgr.shape[:2]
    scale = work_long_edge / max(h, w)
    if scale < 1.0:
        interp = cv2.INTER_AREA
        bgr = cv2.resize(
            bgr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=interp
        )
    elif scale > 1.0:
        bgr = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return PreparedImage(
        bgr=bgr,
        gray=gray,
        gray_f=gray.astype(np.float32) / 255.0,
        hsv=cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV),
        lab=cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB),
    )


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / (denominator + EPS))
