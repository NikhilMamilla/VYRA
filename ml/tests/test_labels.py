from __future__ import annotations

from vyra_ml import ISSUE_LABELS
from vyra_ml.labels import (
    AppliedDegradation,
    label_vector,
    provisional_quality_score,
    quality_label,
    severity_by_issue,
)


def _ad(name, severity):
    return AppliedDegradation(name=name, severity=severity, params={})


def test_clean_sample_has_all_zero_labels():
    labels = label_vector([])
    assert set(labels) == set(ISSUE_LABELS)
    assert sum(labels.values()) == 0
    assert provisional_quality_score([]) == 100.0


def test_multi_label_is_simultaneous():
    applied = [_ad("blur", 3), _ad("noise", 2), _ad("underexposure", 4)]
    labels = label_vector(applied)
    assert labels["blur"] == labels["noise"] == labels["underexposure"] == 1
    assert labels["overexposure"] == labels["corruption"] == labels["defect"] == 0


def test_corruption_degradation_maps_to_corruption_label():
    assert label_vector([_ad("corruption", 2)])["corruption"] == 1


def test_severity_by_issue_takes_the_max():
    applied = [_ad("blur", 2), _ad("blur", 5)]
    assert severity_by_issue(applied)["blur"] == 5


def test_quality_score_decreases_with_severity_and_count():
    one_mild = provisional_quality_score([_ad("blur", 1)])
    one_severe = provisional_quality_score([_ad("blur", 5)])
    two_severe = provisional_quality_score([_ad("blur", 5), _ad("noise", 5)])
    assert 100 > one_mild > one_severe > two_severe
    assert quality_label(one_mild) == "GOOD"
    assert quality_label(two_severe) == "DEFECTIVE"
