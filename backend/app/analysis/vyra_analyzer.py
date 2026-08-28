"""The production :class:`~app.analysis.contract.QualityAnalyzer`.

Wraps :class:`vyra_ml.inference.VyraQualityModel` -- the self-describing bundle
produced by ``ml/scripts/export_inference_bundle.py`` -- and maps its output onto
the API's :class:`~app.schemas.analysis.AnalysisOutcome`. The heavy model is
loaded once (:meth:`from_path`) and every request runs the CV pipeline on a
worker thread so the event loop is never blocked.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import anyio.to_thread

from app.core.errors import InvalidImageError
from app.schemas.analysis import AnalysisOutcome, Issue

logger = logging.getLogger(__name__)

# Feature -> the human-facing statistic name shown in the API and UI.
_STATISTICS: dict[str, str] = {
    "sharpness": "sharp_laplacian_var",
    "brightness": "expo_luma_mean",
    "contrast": "contrast_std",
    "noise_sigma": "noise_immerkaer_sigma",
    "saturation": "color_saturation_mean",
    "colourfulness": "color_colourfulness",
    "blockiness": "compress_blockiness",
    "edge_density": "sharp_edge_density",
    "dark_clip_ratio": "expo_dark_clip_ratio",
    "bright_clip_ratio": "expo_bright_clip_ratio",
}

_DIRECTION = {
    "blur": "lower_sharpness_supports_blur",
    "underexposure": "supports_underexposure",
    "overexposure": "supports_overexposure",
    "noise": "higher_supports_noise",
    "corruption": "higher_blockiness_supports_corruption",
}


class VyraAnalyzer:
    def __init__(self, model: Any) -> None:
        self._model = model
        self._description = model.describe()

    @classmethod
    def from_path(cls, bundle_dir: str | Path) -> VyraAnalyzer:
        """Load the model bundle, or raise so startup fails loudly."""
        from vyra_ml.inference import VyraQualityModel

        bundle_dir = Path(bundle_dir)
        if not (bundle_dir / "bundle.json").is_file():
            raise FileNotFoundError(
                f"No inference bundle at {bundle_dir} (expected bundle.json). "
                "Run ml/scripts/export_inference_bundle.py or set MODEL_PATH."
            )
        model = VyraQualityModel.load(bundle_dir)
        logger.info(
            "VYRA analyzer ready: %s (%s), bundle=%s",
            model.model_version,
            model.feature_version,
            bundle_dir,
        )
        return cls(model)

    @property
    def model_version(self) -> str:
        return self._model.model_version

    @property
    def description(self) -> dict:
        return self._description

    async def analyze(self, image_bytes: bytes, *, content_type: str) -> AnalysisOutcome:
        try:
            analysis = await anyio.to_thread.run_sync(self._model.analyze_bytes, image_bytes)
        except ValueError as exc:
            raise InvalidImageError("The uploaded image could not be decoded.") from exc
        return self._to_outcome(analysis)

    # -- mapping ------------------------------------------------------------
    def _to_outcome(self, a: Any) -> AnalysisOutcome:
        feats: dict[str, float] = a.features
        issues: list[Issue] = []
        evidence: list[dict] = []

        for pred in a.issues:
            if not pred.flagged:
                continue
            issues.append(
                Issue(
                    type=pred.issue,
                    severity=pred.severity or "low",
                    confidence=round(pred.probability, 4),
                    validation=pred.validation,
                    detail=self._issue_detail(pred),
                )
            )
            for fname in self._model.evidence_features(pred.issue):
                if fname in feats:
                    evidence.append(
                        {
                            "feature": fname,
                            "value": round(float(feats[fname]), 4),
                            "direction": _DIRECTION.get(pred.issue, f"supports_{pred.issue}"),
                        }
                    )

        defect = a.potential_defect
        if defect.flagged:
            issues.append(
                Issue(
                    type="defect",
                    severity=defect.severity or "low",
                    confidence=round(defect.probability, 4),
                    validation="screening",
                    detail=defect.note,
                )
            )

        statistics = {
            name: round(float(feats[src]), 4) for name, src in _STATISTICS.items() if src in feats
        }

        explanation: dict[str, Any] = {
            "summary": _summary(a.quality_score, a.quality_label, issues),
            "evidence": evidence,
            "issue_probabilities": {p.issue: round(p.probability, 4) for p in a.issues},
            "potential_defect": {
                "probability": defect.probability,
                "flagged": defect.flagged,
                "region": defect.region,
                "evidence": defect.evidence,
                "note": defect.note,
            },
            "capabilities": self._description.get("capabilities", {}),
            "feature_version": a.feature_version,
            "timings_ms": a.timings_ms,
        }

        return AnalysisOutcome(
            quality_score=a.quality_score,
            quality_label=a.quality_label,
            issues=issues,
            metrics=statistics,
            explanation=explanation,
            model_version=a.model_version,
            image_width=getattr(a, "image_width", None),
            image_height=getattr(a, "image_height", None),
        )

    def _issue_detail(self, pred: Any) -> str:
        if pred.validation == "real-world":
            return f"Validated on real images (VizWiz F1 {pred.real_world_f1})."
        if pred.validation == "synthetic-only":
            return (
                "Detected by the model but only validated on synthetic degradations "
                "-- no real-world evaluation exists for this issue."
            )
        return "Screening signal."


def _summary(score: float, label: str, issues: list[Issue]) -> str:
    if not issues:
        return f"No quality issues detected. Operational quality score {score:.0f}/100 ({label})."
    names = ", ".join(sorted({i.type for i in issues}))
    return (
        f"Quality score {score:.0f}/100 ({label}). "
        f"Flagged: {names}. Confidence and severity are per issue."
    )
