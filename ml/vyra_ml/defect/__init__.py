"""Localised 'potential visual defect' detection (Phase 3C).

The Phase 2 global-feature ``defect`` classifier does not localise and does not
transfer to real images (Phase 3A real ROC-AUC 0.42; Phase 3B threshold tuning
was degenerate). This package replaces it with a deliberately simple,
training-free, self-referential patch-anomaly score.
"""

from vyra_ml.defect.patch_anomaly import (
    DefectDetector,
    DefectResult,
    patch_anomaly_map,
)

__all__ = ["DefectDetector", "DefectResult", "patch_anomaly_map"]
