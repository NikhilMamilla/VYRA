"""Phase 3B: real-validation threshold selection and probability calibration.

Nothing here ever sees the Phase 3A evaluation set. All fitting is done on the
real *validation* split built from VizWiz train (``vyra_ml.realworld.val_split``).
"""

from vyra_ml.calibration.probability import (
    PerLabelCalibrator,
    brier_score,
    expected_calibration_error,
    fit_calibrators,
    reliability_curve,
)
from vyra_ml.calibration.thresholds import (
    ThresholdSet,
    select_threshold,
    sweep_thresholds,
)

__all__ = [
    "PerLabelCalibrator",
    "ThresholdSet",
    "brier_score",
    "expected_calibration_error",
    "fit_calibrators",
    "reliability_curve",
    "select_threshold",
    "sweep_thresholds",
]
