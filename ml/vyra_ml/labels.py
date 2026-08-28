"""Multi-label issue representation and the provisional quality target.

Issues are treated as independent and simultaneously possible: a sample may be
blurred *and* noisy *and* underexposed. The label for an issue is 1 whenever a
degradation raising that issue was applied, at any severity -- ground truth is
known by construction. Severity 1 is subtle by design; see ``docs/dataset.md``
for the label-noise discussion around the near-threshold tier.

The quality score is explicitly *provisional*. We do not have human quality
ratings for these synthetic samples, so we do not claim one. The formula below
is a documented placeholder for a regression baseline; it will be replaced once
real MOS-rated data is inspected (Phase 3).
"""

from __future__ import annotations

from dataclasses import dataclass

from vyra_ml import ISSUE_LABELS
from vyra_ml.degradations import ISSUE_LABEL_BY_DEGRADATION

# How much a single issue at severity s (1..5) multiplies remaining quality by.
# Chosen so one extreme issue lands near 0.2 and a barely-noticeable one near
# 0.97. Provisional -- see module docstring.
_SEVERITY_QUALITY_FACTOR = {1: 0.95, 2: 0.85, 3: 0.68, 4: 0.45, 5: 0.22}


@dataclass(frozen=True)
class AppliedDegradation:
    name: str
    severity: int
    params: dict


def label_vector(applied: list[AppliedDegradation]) -> dict[str, int]:
    """Binary label per issue in canonical order."""
    active = {ISSUE_LABEL_BY_DEGRADATION[a.name] for a in applied}
    return {label: int(label in active) for label in ISSUE_LABELS}


def severity_by_issue(applied: list[AppliedDegradation]) -> dict[str, int]:
    """Max applied severity per issue (0 if the issue is absent)."""
    out = dict.fromkeys(ISSUE_LABELS, 0)
    for a in applied:
        issue = ISSUE_LABEL_BY_DEGRADATION[a.name]
        out[issue] = max(out[issue], a.severity)
    return out


def provisional_quality_score(applied: list[AppliedDegradation]) -> float:
    """A 0-100 placeholder target: product of per-issue quality factors.

    Compounding (rather than summing penalties) keeps the score in range and
    reflects that a second severe issue matters less once the image is already
    poor. NOT ground truth.
    """
    quality = 1.0
    for sev in severity_by_issue(applied).values():
        if sev > 0:
            quality *= _SEVERITY_QUALITY_FACTOR[sev]
    return round(100.0 * quality, 2)


def quality_label(score: float) -> str:
    """Coarse band used for reporting only."""
    if score >= 80:
        return "GOOD"
    if score >= 60:
        return "ACCEPTABLE"
    if score >= 35:
        return "DEGRADED"
    return "DEFECTIVE"
