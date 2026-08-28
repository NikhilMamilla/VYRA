"""The single VYRA inference entry point.

Everything the service needs to turn image bytes into a structured quality
analysis lives behind :class:`VyraQualityModel`. It loads *one* self-describing
bundle directory (see ``scripts/export_inference_bundle.py``) that pins the
model, the probability calibrators, the defect detector, the per-issue decision
thresholds, the feature version and the quality-score definition -- so inference
never depends on remembering which experiment produced what.

    model = VyraQualityModel.load("artifacts/vyra-quality-model-v1")
    result = model.analyze_bytes(open("photo.jpg", "rb").read())

The module deliberately has no knowledge of HTTP, storage or the database.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import joblib
import numpy as np

from vyra_ml.calibration.probability import PerLabelCalibrator  # noqa: F401  (unpickle)
from vyra_ml.defect.patch_anomaly import DefectDetector
from vyra_ml.features import FEATURE_NAMES, extract_features

BUNDLE_FILE = "bundle.json"
_SEVERITY_BANDS = ((0.34, "low"), (0.67, "medium"), (1.01, "high"))


@dataclass(frozen=True)
class IssuePrediction:
    issue: str
    probability: float  # calibrated where a calibrator was fitted
    flagged: bool
    severity: str | None  # low | medium | high, only when flagged
    threshold: float
    calibrated: bool
    validation: str  # "real-world" | "synthetic-only"
    real_world_f1: float | None
    synthetic_f1: float | None


@dataclass(frozen=True)
class DefectPrediction:
    probability: float
    flagged: bool
    severity: str | None
    region: list[float] | None  # [x, y, w, h] as image fractions
    evidence: list[dict]
    method: str
    validation: str
    note: str


@dataclass(frozen=True)
class QualityAnalysis:
    quality_score: float
    quality_label: str
    issues: list[IssuePrediction]
    potential_defect: DefectPrediction
    features: dict[str, float]
    model_version: str
    feature_version: str
    image_width: int | None = None
    image_height: int | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _severity(prob: float, threshold: float) -> str:
    frac = 0.0 if prob <= threshold else (prob - threshold) / max(1e-6, 1.0 - threshold)
    for hi, name in _SEVERITY_BANDS:
        if frac < hi:
            return name
    return "high"


class VyraQualityModel:
    """Loaded once at application startup; ``analyze_*`` is called per request."""

    def __init__(
        self,
        *,
        bundle: dict,
        estimators: dict,
        calibrators: PerLabelCalibrator | None,
        defect: DefectDetector,
        bundle_dir: Path,
    ) -> None:
        self._bundle = bundle
        self._estimators = estimators
        self._calibrators = calibrators
        self._defect = defect
        self._bundle_dir = bundle_dir
        self.model_version: str = bundle["model_version"]
        self.feature_version: str = bundle["feature_version"]
        self._work_long_edge: int = int(bundle.get("work_long_edge", 384))
        self._issue_cfg: dict = bundle["issues"]
        self._score_cfg: dict = bundle["quality_score"]

    # -- loading ----------------------------------------------------------------
    @classmethod
    def load(cls, bundle_dir: str | Path) -> VyraQualityModel:
        bundle_dir = Path(bundle_dir)
        bundle_path = bundle_dir / BUNDLE_FILE
        if not bundle_path.is_file():
            raise FileNotFoundError(f"No {BUNDLE_FILE} in {bundle_dir}")
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

        model_blob = joblib.load(bundle_dir / bundle["artifacts"]["model"])
        estimators = model_blob["models"]
        # Trained with n_jobs=-1; for single-image inference the thread-pool
        # spin-up per call dwarfs the work, so force serial prediction.
        for est in estimators.values():
            if hasattr(est, "n_jobs"):
                est.n_jobs = 1
        if model_blob.get("feature_version") != bundle["feature_version"]:
            raise ValueError(
                f"feature_version mismatch: model {model_blob.get('feature_version')!r} "
                f"vs bundle {bundle['feature_version']!r}"
            )
        if tuple(model_blob["feature_names"]) != tuple(FEATURE_NAMES):
            raise ValueError("installed vyra_ml feature set does not match the model bundle")

        calibrators = None
        cal_rel = bundle["artifacts"].get("calibrators")
        if cal_rel:
            calibrators = PerLabelCalibrator.load(bundle_dir / cal_rel)

        defect = DefectDetector.load(bundle_dir / bundle["artifacts"]["defect_detector"])

        missing = set(bundle["issues"]) - set(estimators)
        if missing:
            raise ValueError(f"bundle issues without an estimator: {sorted(missing)}")
        return cls(
            bundle=bundle,
            estimators=estimators,
            calibrators=calibrators,
            defect=defect,
            bundle_dir=bundle_dir,
        )

    # -- inference ------------------------------------------------------------
    def analyze_bytes(self, data: bytes) -> QualityAnalysis:
        arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None or bgr.size == 0:
            raise ValueError("image bytes could not be decoded")
        return self.analyze_bgr(bgr)

    def analyze_bgr(self, bgr: np.ndarray) -> QualityAnalysis:
        import time

        t0 = time.perf_counter()
        feats = extract_features(bgr, work_long_edge=self._work_long_edge)
        x = np.array([[feats[name] for name in FEATURE_NAMES]], dtype=np.float64)
        t_feat = time.perf_counter()

        issues: list[IssuePrediction] = []
        probs: dict[str, float] = {}
        for issue, cfg in self._issue_cfg.items():
            raw = float(self._estimators[issue].predict_proba(x)[0, 1])
            prob = raw
            if self._calibrators is not None:
                prob = float(self._calibrators.transform(issue, np.array([raw]))[0])
            threshold = float(cfg["threshold"])
            flagged = prob >= threshold
            probs[issue] = prob
            issues.append(
                IssuePrediction(
                    issue=issue,
                    probability=round(prob, 4),
                    flagged=flagged,
                    severity=_severity(prob, threshold) if flagged else None,
                    threshold=threshold,
                    calibrated=bool(cfg.get("calibrated", False)),
                    validation=cfg["validation"],
                    real_world_f1=cfg.get("real_world_f1"),
                    synthetic_f1=cfg.get("synthetic_f1"),
                )
            )
        t_issue = time.perf_counter()

        d = self._defect.analyze(bgr)
        defect = DefectPrediction(
            probability=d.probability,
            flagged=d.flagged,
            severity=_severity(d.probability, self._defect.threshold) if d.flagged else None,
            region=d.region_norm,
            evidence=d.top_features,
            method=self._score_cfg.get("defect_method", "patch-anomaly"),
            validation="synthetic-only (screening)",
            note=d.note,
        )
        t_defect = time.perf_counter()

        score, label = self._quality_score(probs, defect.probability)
        return QualityAnalysis(
            quality_score=score,
            quality_label=label,
            issues=issues,
            potential_defect=defect,
            features={k: round(float(v), 6) for k, v in feats.items()},
            model_version=self.model_version,
            feature_version=self.feature_version,
            image_width=int(bgr.shape[1]),
            image_height=int(bgr.shape[0]),
            timings_ms={
                "features": round((t_feat - t0) * 1000, 1),
                "issue_models": round((t_issue - t_feat) * 1000, 1),
                "defect": round((t_defect - t_issue) * 1000, 1),
                "total": round((time.perf_counter() - t0) * 1000, 1),
            },
        )

    # -- quality score ------------------------------------------------------
    def _quality_score(self, probs: dict[str, float], defect_prob: float) -> tuple[float, str]:
        """Operational 0-100 score: 100 x product of per-issue retention factors.

        Compounding (rather than summing penalties) keeps the score bounded and
        reflects diminishing marginal damage when several issues co-occur. It is
        NOT a perceptual/MOS score -- see docs/quality-score.md.
        """
        cfg = self._score_cfg
        weights: dict[str, float] = cfg["weights"]
        severe_at: float = float(cfg.get("severe_probability", 0.9))
        retention = 1.0
        for issue, w in weights.items():
            if issue == "potential_defect":
                p = defect_prob
                t = float(self._defect.threshold)
            else:
                p = probs.get(issue, 0.0)
                t = float(self._issue_cfg[issue]["threshold"])
            impact = 0.0 if p <= t else w * min(1.0, (p - t) / max(1e-6, severe_at - t))
            retention *= 1.0 - impact
        score = round(100.0 * retention, 1)

        label = cfg["bands"][-1]["label"]
        for band in cfg["bands"]:
            if score >= band["min"]:
                label = band["label"]
                break
        return score, label

    def describe(self) -> dict:
        return {
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "issues": self._issue_cfg,
            "defect": self._bundle.get("defect", {}),
            "quality_score": self._score_cfg,
            "training": self._bundle.get("training", {}),
            "calibration": self._bundle.get("calibration", {}),
            "capabilities": self._bundle.get("capabilities", {}),
            "real_world_evaluation": self._bundle.get("real_world_evaluation", {}),
        }

    def evidence_features(self, issue: str) -> list[str]:
        return list(self._issue_cfg.get(issue, {}).get("evidence_features", []))
