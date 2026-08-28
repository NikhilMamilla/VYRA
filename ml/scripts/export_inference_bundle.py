"""Assemble the self-describing VYRA inference bundle.

Bundles the Phase 3B decision (row D) plus the Phase 3C patch defect detector
into ``artifacts/vyra-quality-model-v1/``:

    model.joblib          per-issue RandomForest (cvfeat-v2, phase2-baseline-v1)
    calibrators.joblib    isotonic per-issue calibrators (phase3b real-val)
    defect_detector.json  patch-anomaly params (phase3c-defect-v1)
    bundle.json           the manifest that ties it all together

Then sanity-checks the quality-score distribution on the synthetic test split
and the real evaluation split. Run: ``python scripts/export_inference_bundle.py``
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
MODEL_RUN = ML_ROOT / "runs" / "phase2-baseline-v1_20260828-142415"
CALIB = ML_ROOT / "runs" / "phase3b-calibration-v1" / "calibrators" / "v2fix.joblib"
DEFECT = ML_ROOT / "runs" / "phase3c-defect-v1" / "defect_detector.json"

# Phase 3B row D: thresholds on calibrated probability (cal_v2fix.json) for the
# VizWiz-evaluable issues; v2fix synthetic-val thresholds for the rest.
ISSUES = {
    "blur": {
        "threshold": 0.36,
        "calibrated": True,
        "validation": "real-world",
        "real_world_f1": 0.6114,
        "synthetic_f1": 0.9004,
        "evidence_features": [
            "sharp_highfreq_ratio",
            "sharp_laplacian_var",
            "texture_spectral_slope",
        ],
    },
    "underexposure": {
        "threshold": 0.50,
        "calibrated": True,
        "validation": "real-world",
        "real_world_f1": 0.4889,
        "synthetic_f1": 0.8434,
        "evidence_features": ["expo_luma_mean", "expo_shadow_ratio", "contrast_p95"],
    },
    "overexposure": {
        "threshold": 0.10,
        "calibrated": True,
        "validation": "real-world",
        "real_world_f1": 0.1912,
        "synthetic_f1": 0.7411,
        "evidence_features": ["expo_bright_clip_ratio", "expo_luma_mean", "contrast_p95"],
    },
    "noise": {
        "threshold": 0.45,
        "calibrated": False,
        "validation": "synthetic-only",
        "real_world_f1": None,
        "synthetic_f1": 0.8434,
        "evidence_features": ["noise_immerkaer_sigma", "noise_median_residual_mad"],
    },
    "corruption": {
        "threshold": 0.35,
        "calibrated": False,
        "validation": "synthetic-only",
        "real_world_f1": None,
        "synthetic_f1": 0.9730,
        "evidence_features": [
            "compress_blockiness",
            "compress_blockiness_v",
            "compress_blockiness_h",
        ],
    },
}

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
    for src in (MODEL_RUN / "model.joblib", CALIB, DEFECT):
        if not src.is_file():
            raise SystemExit(f"missing input: {src}")

    OUT.mkdir(parents=True, exist_ok=True)
    # Re-dump the model compressed: a 300-tree RF x 6 issues is ~39 MB raw but
    # compresses to a few MB, which is all that ships in the backend image.
    joblib.dump(joblib.load(MODEL_RUN / "model.joblib"), OUT / "model.joblib", compress=("xz", 3))
    joblib.dump(joblib.load(CALIB), OUT / "calibrators.joblib", compress=("xz", 3))
    shutil.copy2(DEFECT, OUT / "defect_detector.json")

    defect_meta = json.loads((DEFECT).read_text(encoding="utf-8"))
    model_record = json.loads((MODEL_RUN / "experiment.json").read_text(encoding="utf-8"))

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
            "experiment": "phase2-baseline-v1",
            "run": MODEL_RUN.name,
            "seed": model_record.get("seed"),
            "model_type": "RandomForest one-vs-rest, 6 issues, 42 CV features",
            "dataset": "BSDS500 clean images + calibrated synthetic degradations",
            "split_counts": model_record.get("split_counts"),
        },
        "calibration": {
            "method": "isotonic regression, per issue",
            "fitted_on": "VizWiz train real-validation sample (n=2489, ≥3/5 votes)",
            "experiment": "phase3b-calibration-v1",
            "note": "identity for noise/corruption (no real labels in VizWiz)",
        },
        "issues": ISSUES,
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
            "dataset": "VizWiz-QualityIssues val sample (n=2496, ≥3/5 votes)",
            "experiment": "phase3b-calibration-v1 row D",
            "primary_macro_f1": 0.4305,
            "per_issue_f1": {"blur": 0.6114, "underexposure": 0.4889, "overexposure": 0.1912},
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
