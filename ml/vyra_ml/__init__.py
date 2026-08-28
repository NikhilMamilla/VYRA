"""VYRA machine-learning package: dataset preparation, CV features, experiments.

This package is deliberately independent of ``backend/``. The only artefact that
crosses into the service is a serialised model plus its feature specification.
"""

__version__ = "0.2.0"

# The six issue labels the assessment requires, in canonical order. Every label
# vector, manifest column and metric report uses this ordering.
ISSUE_LABELS: tuple[str, ...] = (
    "blur",
    "underexposure",
    "overexposure",
    "noise",
    "corruption",
    "defect",
)
