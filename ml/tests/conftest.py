from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def synthetic_image(rng: np.random.Generator) -> np.ndarray:
    """A textured colour image with structure at several scales (uint8 BGR)."""
    h, w = 256, 320
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 0.5 + 0.25 * np.sin(xx / 12.0) + 0.15 * np.cos(yy / 7.0) + 0.1 * np.sin((xx + yy) / 23.0)
    base = np.clip(base, 0, 1)
    img = np.stack([base, np.roll(base, 5, axis=1), np.roll(base, 9, axis=0)], axis=-1)
    img = img + rng.normal(0, 0.02, img.shape)
    return np.clip(img * 255, 0, 255).astype(np.uint8)
