"""Compression artefacts and severe degradation.

Raises the ``corruption`` issue label ("image corruption or severe
degradation"). Implemented as aggressive JPEG re-encoding, with a
downscale/upscale resolution loss added at the top severities so "severe
degradation" is genuinely present and not just blocking.
"""

from __future__ import annotations

import cv2
import numpy as np

from vyra_ml.degradations.base import Degradation, DegradationResult, lerp_range

_JPEG_QUALITY = {
    1: (62, 80),
    2: (44, 62),
    3: (28, 44),
    4: (15, 28),
    5: (4, 15),
}
# Resolution-loss factor (downscale then upscale back). 1.0 = no loss.
_RESCALE = {
    1: (1.0, 1.0),
    2: (1.0, 1.0),
    3: (0.85, 1.0),
    4: (0.55, 0.8),
    5: (0.3, 0.55),
}


class CorruptionDegradation(Degradation):
    name = "corruption"
    issue_label = "corruption"

    def apply(
        self, image: np.ndarray, severity: int, rng: np.random.Generator
    ) -> DegradationResult:
        h, w = image.shape[:2]
        out = image
        params: dict = {}

        factor = lerp_range(rng, _RESCALE, severity)
        if factor < 0.999:
            small = cv2.resize(
                out,
                (max(1, int(w * factor)), max(1, int(h * factor))),
                interpolation=cv2.INTER_AREA,
            )
            out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            params["rescale_factor"] = round(factor, 3)

        quality = int(round(lerp_range(rng, _JPEG_QUALITY, severity)))
        # A double JPEG pass at the highest severity compounds the artefacts,
        # as happens to images repeatedly shared through messaging apps.
        passes = 2 if severity == 5 and rng.random() < 0.5 else 1
        for _ in range(passes):
            ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:  # pragma: no cover - imencode failure is not expected
                raise RuntimeError("JPEG encode failed during corruption degradation")
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        params["jpeg_quality"] = quality
        params["jpeg_passes"] = passes

        return DegradationResult(image=out, params=params)
