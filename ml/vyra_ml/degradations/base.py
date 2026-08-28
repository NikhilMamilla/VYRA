"""Degradation interface and shared helpers.

Each degradation is a small, independently testable object. It receives a uint8
BGR image, a severity in 1..5 and a seeded RNG, and returns the degraded image
plus the exact parameters it sampled (which are written to the manifest, so any
sample can be reproduced or audited).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SEVERITIES = (1, 2, 3, 4, 5)


@dataclass
class DegradationResult:
    image: np.ndarray  # uint8 BGR, same shape as input
    params: dict[str, Any] = field(default_factory=dict)


def to_float(image: np.ndarray) -> np.ndarray:
    """uint8 BGR -> float32 in [0, 1]."""
    return image.astype(np.float32) / 255.0


def to_uint8(image: np.ndarray) -> np.ndarray:
    """float image in [0, 1] -> uint8, with clipping."""
    return np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)


def lerp_range(rng: np.random.Generator, ranges: dict[int, tuple[float, float]], severity: int):
    """Sample uniformly from the (low, high) parameter range for ``severity``.

    Using a *range* per level, not a fixed value, is deliberate: it stops the
    model from memorising a synthetic fingerprint tied to one exact parameter.
    """
    low, high = ranges[severity]
    return float(rng.uniform(low, high))


class Degradation(abc.ABC):
    """Base class for a single, self-contained image-quality degradation."""

    #: Stable identifier used in configs and manifests.
    name: str
    #: Which of the six issue labels this degradation raises when applied.
    issue_label: str

    @abc.abstractmethod
    def apply(
        self, image: np.ndarray, severity: int, rng: np.random.Generator
    ) -> DegradationResult:
        """Return a degraded copy of ``image`` at the given severity."""

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<Degradation {self.name!r} -> label {self.issue_label!r}>"
