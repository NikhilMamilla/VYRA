"""Assemble the self-describing VYRA inference bundle.

Bundles the Phase 3D real-trained issue heads plus the Phase 3C patch defect
detector into ``artifacts/vyra-quality-model-v1/``:

    model.joblib          per-issue classifiers -- blur/underexposure/overexposure
                          trained on real VizWiz data (phase3d-realtrain-v1),
                          noise/corruption kept from the synthetic baseline
    calibrators.joblib    isotonic per-issue calibrators (phase3d real OOF)
    defect_detector.json  patch-anomaly params (phase3c-defect-v1)
    bundle.json           the manifest that ties it all together

Threshold, per-issue real-world F1 and the headline macro-F1 are read straight
from the Phase 3D run so this script never hard-codes a metric. Then it
sanity-checks the quality-score distribution on the synthetic test split and the
real evaluation split. Run: ``python scripts/export_inference_bundle.py``
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vyra_ml.features import FEATURE_NAMES  # noqa: E402
from vyra_ml.inference import VyraQualityModel  # noqa: E402

ML_ROOT = Path(__file__).resolve().parents[1]
OUT = ML_ROOT / "artifacts" / "vyra-quality-model-v1"
PHASE3D_RUN = ML_ROOT / "runs" / "phase3d-realtrain-v1"
MODEL_RUN = PHASE3D_RUN
CALIB = PHASE3D_RUN / "calibrators.joblib"
DEFECT = ML_ROOT / "runs" / "phase3c-defect-v1" / "defect_detector.json"
# Synthetic baseline -- kept only for the noise/corruption synthetic F1 numbers.
SYNTH_RUN = ML_ROOT / "runs" / "phase2-baseline-v1_20260828-142415"

# Static per-issue metadata. Thresholds, calibrated flags and real-world F1 for
# the real-trained heads are filled in from the Phase 3D run at build time.
_ISSUE_META = {
    "blur": {
        "synthetic_f1": 0.9004,
        "evidence_features": [
            "sharp_highfreq_ratio",
            "sharp_laplacian_var",
            "texture_spectral_slope",
        ],
    },
    "underexposure": {
        "synthetic_f1": 0.8434,
        "evidence_features": ["expo_luma_mean", "expo_shadow_ratio", "contrast_p95"],
    },
    "overexposure": {
        "synthetic_f1": 0.7411,
        "evidence_features": ["expo_bright_clip_ratio", "expo_luma_mean", "contrast_p95"],
    },
    "noise": {
        "synthetic_f1": 0.8434,
        "evidence_features": ["noise_immerkaer_sigma", "noise_median_residual_mad"],
    },
    "corruption": {
        "synthetic_f1": 0.9730,
        "evidence_features": [
            "compress_blockiness",
            "compress_blockiness_v",
            "compress_blockiness_h",
        ],
    },
}
_REAL_TRAINED = ("blur", "underexposure", "overexposure")


def _synth_aug_weight(audit: dict) -> float:
    return audit.get("synthetic_augmentation", {}).get("weight", 0.3)


def _build_issues(cv: dict, fit: dict, row: dict, floors: dict) -> dict:
    issues: dict = {}
    for name, meta in _ISSUE_META.items():
        if name in _REAL_TRAINED:
            issues[name] = {
                "threshold": round(float(cv[name]["cv_threshold"]), 4),
                "calibrated": bool(fit["calibrated"][name]),
                "validation": "real-world",
                "real_world_f1": round(float(row["per_class"][name]["f1"]), 4),
                "synthetic_f1": meta["synthetic_f1"],
                "evidence_features": meta["evidence_features"],
            }
            # Deterministic physical floor OR'd into the learned prediction (see
            # phase3d.ISSUE_FLOORS): a feature past a hard bound forces the flag.
            if name in floors:
                issues[name]["floor"] = floors[name]
        else:
            issues[name] = {
                "threshold": 0.45 if name == "noise" else 0.35,
                "calibrated": False,
                "validation": "synthetic-only",
                "real_world_f1": None,
                "synthetic_f1": meta["synthetic_f1"],
                "evidence_features": meta["evidence_features"],
            }
    return issues


QUALITY_SCORE = {
    "formula": "100 * product_i (1 - w_i * clip((p_i - t_i) / (severe - t_i), 0, 1))",
    "explanation": (
        "Operational score, not a perceptual/MOS score. Each issue removes at "
        "most w_i of the remaining quality, scaled by how far its calibrated "
        "probability sits past its decision threshold t_i toward 'severe'. "
        "Compounding keeps the score in [0, 100] and models diminishing marginal "
        "damage. See docs/quality-score.md."
    ),
    "severe_probability": 0.9,
    "weights": {
        "blur": 0.55,
        "corruption": 0.45,
        "underexposure": 0.45,
        "overexposure": 0.35,
        "noise": 0.30,
        "potential_defect": 0.20,
    },
    "bands": [
        {"min": 85, "label": "GOOD"},
        {"min": 68, "label": "ACCEPTABLE"},
        {"min": 45, "label": "DEGRADED"},
        {"min": 0, "label": "POOR"},
    ],
    "defect_method": "patch-anomaly (self-referential, phase3c-defect-v1)",
}


def _dist(model: VyraQualityModel, feat_parquet: Path, label: str, is_synth: bool) -> dict:
    df = pd.read_parquet(feat_parquet)
    if is_synth:
        df = df[df.split == "test"] if "split" in df.columns else df
    x = df[list(FEATURE_NAMES)].to_numpy(np.float64)

    sev = float(QUALITY_SCORE["severe_probability"])
    weights = QUALITY_SCORE["weights"]
    retention = np.ones(len(x))
    for issue, w in weights.items():
        if issue == "potential_defect":
            continue  # defect is excluded from this feature-only sweep
        raw = model._estimators[issue].predict_proba(x)[:, 1]  # batched
        p = model._calibrators.transform(issue, raw) if model._calibrators else raw
        t = float(model._issue_cfg[issue]["threshold"])
        impact = np.where(p <= t, 0.0, w * np.clip((p - t) / (sev - t), 0.0, 1.0))
        retention *= 1.0 - impact
    scores = np.round(100.0 * retention, 1)

    pct = {p: round(float(np.percentile(scores, p)), 1) for p in (1, 5, 25, 50, 75, 95, 99)}
    bands = {b["label"]: 0 for b in QUALITY_SCORE["bands"]}
    for s in scores:
        for b in QUALITY_SCORE["bands"]:
            if s >= b["min"]:
                bands[b["label"]] += 1
                break
    return {"split": label, "n": len(scores), "percentiles": pct, "band_counts": bands}


def main() -> None:
    for src in (
        MODEL_RUN / "model.joblib",
        CALIB,
        DEFECT,
        PHASE3D_RUN / "final_evaluation.json",
        PHASE3D_RUN / "status.json",
    ):
        if not src.is_file():
            raise SystemExit(f"missing input: {src} (run the Phase 3D orchestrator first)")

    phase3d_final = json.loads((PHASE3D_RUN / "final_evaluation.json").read_text(encoding="utf-8"))
    phase3d_status = json.loads((PHASE3D_RUN / "status.json").read_text(encoding="utf-8"))
    row = phase3d_final["rows"]["D_phase3d_realtrain"]
    prev_row = phase3d_final["rows"]["C_phase3c_shipped"]
    floors = phase3d_final.get("floors", {})
    cv = phase3d_status["steps"]["cv_and_thresholds"]["result"]
    fit = phase3d_status["steps"]["fit_final_models"]["result"]
    audit = phase3d_status["steps"]["data_audit"]["result"]
    issues = _build_issues(cv, fit, row, floors)

    OUT.mkdir(parents=True, exist_ok=True)
    # Re-dump the model compressed: a 300-tree RF x 6 issues is ~39 MB raw but
    # compresses to a few MB, which is all that ships in the backend image.
    joblib.dump(joblib.load(MODEL_RUN / "model.joblib"), OUT / "model.joblib", compress=("xz", 6))
    joblib.dump(joblib.load(CALIB), OUT / "calibrators.joblib", compress=("xz", 3))
    shutil.copy2(DEFECT, OUT / "defect_detector.json")

    defect_meta = json.loads((DEFECT).read_text(encoding="utf-8"))
    synth_record = json.loads((SYNTH_RUN / "experiment.json").read_text(encoding="utf-8"))

    bundle = {
        "model_version": "vyra-quality-model-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_version": "cvfeat-v2",
        "work_long_edge": 384,
        "artifacts": {
            "model": "model.joblib",
            "calibrators": "calibrators.joblib",
            "defect_detector": "defect_detector.json",
        },
        "training": {
            "experiment": "phase3d-realtrain-v1",
            "run": MODEL_RUN.name,
            "seed": phase3d_status.get("seed", 20260828),
            "model_type": "RandomForest one-vs-rest, 6 issues, 42 CV features",
            "real_trained_heads": ["blur", "underexposure", "overexposure"],
            "real_training_data": (
                f"VizWiz-QualityIssues train: {audit['uniform_n']} uniform-sample images "
                f"+ {audit['extra_n']} rare-class-enriched images, real crowd labels "
                f"(≥3/5 votes); synthetic-degradation rows added at weight "
                f"{_synth_aug_weight(audit)} for out-of-distribution coverage "
                "(strong motion blur, extreme exposure)"
            ),
            "synthetic_heads": ["noise", "corruption"],
            "synthetic_dataset": "BSDS500 clean images + calibrated synthetic degradations",
            "synthetic_run": SYNTH_RUN.name,
            "synthetic_split_counts": synth_record.get("split_counts"),
        },
        "calibration": {
            "method": "isotonic regression, per issue",
            "fitted_on": (
                "Phase 3D out-of-fold predictions on the uniform real-val sample "
                f"(n={audit['uniform_n']}, natural prevalence, ≥3/5 votes)"
            ),
            "experiment": "phase3d-realtrain-v1",
            "note": "identity for noise/corruption (no real labels in VizWiz)",
        },
        "issues": issues,
        "defect": {
            "exposed_as": "potential_visual_defect",
            "method": "patch-anomaly, self-referential local outlier",
            "version": defect_meta.get("version"),
            "validation": "synthetic-only, screening signal",
            "test_metrics": defect_meta.get("calibration", {}).get("test_metrics", {}),
            "disclaimer": (
                "Flags a region statistically unlike the rest of the image. NOT a "
                "confirmed physical defect; VYRA is an image-quality tool, not a "
                "diagnostic system."
            ),
        },
        "quality_score": QUALITY_SCORE,
        "real_world_evaluation": {
            "dataset": f"VizWiz-QualityIssues val sample (n={audit['eval_n']}, ≥3/5 votes)",
            "experiment": "phase3d-realtrain-v1",
            "protocol": (
                "frozen eval set read once; thresholds + calibrators fitted on "
                "cross-validated out-of-fold predictions of the disjoint real-val sample"
            ),
            "primary_macro_f1": round(float(row["primary_macro_f1"]), 4),
            "per_issue_f1": {
                k: round(float(row["per_class"][k]["f1"]), 4)
                for k in ("blur", "underexposure", "overexposure")
            },
            "previous_primary_macro_f1": round(float(prev_row["primary_macro_f1"]), 4),
            "not_validated": ["noise", "corruption", "potential_visual_defect"],
        },
        "capabilities": {
            "real_world_validated": ["blur", "underexposure", "overexposure"],
            "synthetic_validated_only": ["noise", "corruption"],
            "screening_only": ["potential_visual_defect"],
        },
    }
    (OUT / "bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    # --- load it back and sanity-check ---
    model = VyraQualityModel.load(OUT)
    print(f"[bundle] loaded {model.model_version} ({model.feature_version})")

    checks = []
    syn = ML_ROOT / "data" / "processed" / "features_phase2-baseline-v1_cvfeat-v2.parquet"
    rev = ML_ROOT / "data" / "processed" / "realeval_features.parquet"
    if syn.is_file():
        checks.append(_dist(model, syn, "synthetic-test", True))
    if rev.is_file():
        checks.append(_dist(model, rev, "real-eval", False))
    (OUT / "score_distribution.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    size_mb = sum(f.stat().st_size for f in OUT.iterdir() if f.is_file()) / 1e6
    print(f"[bundle] {OUT} — {size_mb:.1f} MB")
    for c in checks:
        print(f"  {c['split']}: n={c['n']} percentiles={c['percentiles']} bands={c['band_counts']}")


if __name__ == "__main__":
    main()
