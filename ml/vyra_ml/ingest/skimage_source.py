"""Offline fallback clean-image source: scikit-image's bundled photographs.

Only ~10 usable natural images, so this is not adequate for a serious
experiment (too few originals for a meaningful split). It exists so the pipeline
still runs end-to-end with no network access -- e.g. in CI or when BSDS500 is
unreachable.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from skimage import data

from vyra_ml.ingest.base import SourceAdapter, SourceImage

# Colour photographs only; line drawings / microscopy / test patterns excluded.
_IMAGE_LOADERS = {
    "astronaut": data.astronaut,
    "coffee": data.coffee,
    "chelsea": data.chelsea,
    "rocket": data.rocket,
    "cat": data.cat,
}


class SkimageAdapter(SourceAdapter):
    name = "skimage"

    def prepare(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        for key, loader in _IMAGE_LOADERS.items():
            dest = self.raw_dir / f"{key}.png"
            if dest.is_file():
                continue
            rgb = np.asarray(loader())
            cv2.imwrite(str(dest), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def iter_originals(self, limit: int | None = None) -> Iterator[SourceImage]:
        keys = sorted(_IMAGE_LOADERS)
        if limit is not None:
            keys = keys[:limit]
        for key in keys:
            yield SourceImage(
                source_id=f"skimage/{key}",
                source_dataset="skimage",
                path=Path(self.raw_dir / f"{key}.png"),
            )
