"""Degradation registry.

Order of application when several degradations are combined in one sample:

    exposure -> blur -> defect -> noise -> corruption

This is a pragmatic approximation of a capture pipeline (scene exposure, then
lens blur, then sensor/optical defects, then read noise, then compression). It
is not physically exact but avoids obviously wrong orderings such as compressing
before adding noise.
"""

from __future__ import annotations

from vyra_ml.degradations.base import SEVERITIES, Degradation, DegradationResult
from vyra_ml.degradations.blur import BlurDegradation
from vyra_ml.degradations.corruption import CorruptionDegradation
from vyra_ml.degradations.defect import DefectDegradation
from vyra_ml.degradations.exposure import OverexposureDegradation, UnderexposureDegradation
from vyra_ml.degradations.noise import NoiseDegradation

_ALL: tuple[Degradation, ...] = (
    UnderexposureDegradation(),
    OverexposureDegradation(),
    BlurDegradation(),
    DefectDegradation(),
    NoiseDegradation(),
    CorruptionDegradation(),
)

#: Deterministic application order (see module docstring).
APPLICATION_ORDER: tuple[str, ...] = tuple(d.name for d in _ALL)

DEGRADATIONS: dict[str, Degradation] = {d.name: d for d in _ALL}

ISSUE_LABEL_BY_DEGRADATION: dict[str, str] = {d.name: d.issue_label for d in _ALL}


def get_degradation(name: str) -> Degradation:
    try:
        return DEGRADATIONS[name]
    except KeyError:
        raise KeyError(f"Unknown degradation {name!r}. Known: {sorted(DEGRADATIONS)}") from None


def order_key(name: str) -> int:
    return APPLICATION_ORDER.index(name)


__all__ = [
    "APPLICATION_ORDER",
    "DEGRADATIONS",
    "ISSUE_LABEL_BY_DEGRADATION",
    "SEVERITIES",
    "Degradation",
    "DegradationResult",
    "get_degradation",
    "order_key",
]
