"""Phase 3B orchestrator: feature fix + blur realism + real-validation calibration.

Runs as one background job. Writes a machine-readable ``status.json`` after every
step and never overwrites Phase 3A. Re-running skips steps whose outputs already
exist (delete the run dir to force a full rebuild).

Discipline enforced here:
  * the Phase 3A evaluation set (VizWiz val) is read exactly once, at the end;
  * every threshold / calibrator is fitted only on the real validation split
    (VizWiz train sample);
  * the Phase 3A baseline artifact and reports are never modified.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from vyra_ml import ISSUE_LABELS
from vyra_ml.calibration import PerLabelCalibrator, ThresholdSet, select_threshold, sweep_thresholds
from vyra_ml.calibration.probability import fit_calibrators
from vyra_ml.config import load_config
from vyra_ml.evaluation.metrics import multilabel_report
from vyra_ml.features import FEATURE_VERSION
from vyra_ml.realworld.config import load_real_world_config
from vyra_ml.realworld.features import (
    build_vizwiz_feature_table,
    feature_matrix,
    vote_threshold_labels,
)
from vyra_ml.realworld.label_map import evaluable_labels
from vyra_ml.realworld.vizwiz import parse_annotations

ML_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ML_ROOT / "runs" / "phase3b-calibration-v1"
REPORT_DIR = ML_ROOT / "reports" / "phase3b-calibration-v1"

PRIMARY_LABELS = ["blur", "underexposure", "overexposure"]  # defect excluded (see docs)
VOTE_MIN = 3


# --------------------------------------------------------------------------- #
# status helpers
# --------------------------------------------------------------------------- #
def _status_path() -> Path:
    return RUN_DIR / "status.json"


def _load_status() -> dict:
    p = _status_path()
    return json.loads(p.read_text()) if p.exists() else {"steps": {}, "started": None}


def _save_status(st: dict) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    _status_path().write_text(json.dumps(st, indent=2, default=str), encoding="utf-8")


def _step(st: dict, name: str, fn):
    if st["steps"].get(name, {}).get("state") == "done":
        print(f"[phase3b] SKIP {name} (already done)", flush=True)
        return st["steps"][name].get("result", {})
    print(f"[phase3b] START {name}", flush=True)
    st["steps"][name] = {"state": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
    _save_status(st)
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001
        st["steps"][name] = {
            "state": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        _save_status(st)
        print(f"[phase3b] FAILED {name}: {exc}", flush=True)
        raise
    st["steps"][name] = {
        "state": "done",
        "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "result": result,
    }
    _save_status(st)
    print(f"[phase3b] DONE {name}", flush=True)
    return result


# --------------------------------------------------------------------------- #
# model handling
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    key: str
    bundle_path: Path
    feature_version: str
    description: str


def _predict_scores(bundle: dict, x: np.ndarray) -> dict[str, np.ndarray]:
    return {lbl: bundle["models"][lbl].predict_proba(x)[:, 1] for lbl in ISSUE_LABELS}


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #
def step_synthetic_features_v2() -> dict:
    from vyra_ml.feature_store import build_feature_table

    cfg = load_config(ML_ROOT / "configs" / "experiment.yaml")
    out = build_feature_table(cfg, n_jobs=-1)
    return {"feature_table": str(out), "feature_version": FEATURE_VERSION}


def step_blurnoise_dataset() -> dict:
    from vyra_ml.dataset_build import build_dataset
    from vyra_ml.feature_store import build_feature_table

    cfg = load_config(ML_ROOT / "configs" / "experiment_blurnoise.yaml")
    manifest = cfg.data_dir("manifests") / f"manifest_{cfg.version}.parquet"
    if not manifest.exists():
        res = build_dataset(cfg, progress=True)
        n_samples = res.n_samples
    else:
        n_samples = len(pd.read_parquet(manifest))
    ft = build_feature_table(cfg, n_jobs=-1)
    # how many samples actually got the sensor-noise pass
    mf = pd.read_parquet(manifest)
    with_blur = int(mf["degradations_json"].str.contains('"name": "blur"').sum())
    return {
        "manifest": str(manifest),
        "feature_table": str(ft),
        "n_samples": n_samples,
        "blur_samples_modified": with_blur,
    }


def step_retrain_baselines(st: dict) -> dict:
    from vyra_ml.experiment import run_baseline

    out = {}
    for key, cfg_name in (
        ("v2fix", "experiment.yaml"),
        ("v2blur", "experiment_blurnoise.yaml"),
    ):
        cfg = load_config(ML_ROOT / "configs" / cfg_name)
        ft = cfg.data_dir("processed") / f"features_{cfg.version}_{FEATURE_VERSION}.parquet"
        art = run_baseline(cfg, ft)
        bundle_path = art.run_dir / "model.joblib"
        out[key] = {
            "run_dir": str(art.run_dir),
            "bundle": str(bundle_path),
            "synthetic_test": art.metrics["_record"]["headline_metrics"],
        }
    return out


def step_feature_report_v2() -> dict:
    from vyra_ml.feature_report import build_feature_report

    cfg = load_config(ML_ROOT / "configs" / "experiment.yaml")
    ft = cfg.data_dir("processed") / f"features_{cfg.version}_{FEATURE_VERSION}.parquet"
    out = build_feature_report(ft, REPORT_DIR / "feature_report_v2")
    rep = json.loads(Path(out).read_text())
    blk = {
        f["feature"]: {"min": f["min"], "max": f["max"], "mean": f["mean"], "std": f["std"]}
        for f in rep["per_feature"]
        if f["feature"].startswith("compress_blockiness")
    }
    v1_meta = cfg.data_dir("processed") / f"features_{cfg.version}_cvfeat-v1.parquet"
    v1_blk = {}
    if v1_meta.exists():
        v1 = pd.read_parquet(v1_meta)
        for c in ("compress_blockiness", "compress_blockiness_h", "compress_blockiness_v"):
            v1_blk[c] = {
                "min": float(v1[c].min()),
                "max": float(v1[c].max()),
                "mean": float(v1[c].mean()),
                "std": float(v1[c].std()),
            }
    return {
        "report": str(out),
        "v2_blockiness": blk,
        "v1_blockiness": v1_blk,
        "total_nan": rep["total_nan"],
        "total_inf": rep["total_inf"],
    }


def _build_real_table(cfg_name: str, split_label: str, out_name: str) -> dict:
    cfg = load_real_world_config(ML_ROOT / "configs" / cfg_name)
    # partial download of the needed images (skips cached)
    from vyra_ml.realworld.adapter import select_subset
    from vyra_ml.realworld.fetch import fetch_from_remote_zip

    anns = parse_annotations(cfg.annotation_file)
    subset = select_subset(anns, cfg.subset_size, cfg.seed)
    members = [f"{cfg.eval_split}/{a.image}" for a in subset]
    fetch_from_remote_zip(
        cfg.images_zip_url,
        members,
        cfg.image_cache_dir,
        workers=cfg.download_workers,
        on_progress=lambda d, t: print(f"  [{split_label}] fetched {d}/{t}", flush=True),
    )
    out_path = ML_ROOT / "data" / "processed" / out_name
    stats = build_vizwiz_feature_table(
        subset,
        cfg.image_cache_dir,
        split=split_label,
        work_long_edge=384,
        out_path=out_path,
    )
    return stats


def step_real_val_features() -> dict:
    return _build_real_table("real_world_val.yaml", "real_val", "realval_features.parquet")


def step_real_eval_features() -> dict:
    return _build_real_table("real_world_eval.yaml", "real_eval", "realeval_features.parquet")


def step_leakage_check() -> dict:
    rv_path = ML_ROOT / "data" / "processed" / "realval_features.parquet"
    rv = pd.read_parquet(rv_path)
    re = pd.read_parquet(ML_ROOT / "data" / "processed" / "realeval_features.parquet")
    syn = pd.read_parquet(
        next((ML_ROOT / "data" / "manifests").glob("manifest_phase2-baseline-v1.parquet"))
    )
    syn_sha = set(syn["sha1"])
    eval_ids, eval_sha = set(re["image_id"]), set(re["sha1"])

    # The Phase 3A evaluation set (VizWiz val sample) is frozen and never edited.
    # VizWiz reuses a small number of images across its train and val splits under
    # different ids, so a few can land in both real splits. Any such row is
    # dropped from the *validation* split (the new, disposable dev set) — never
    # from eval.
    dup_mask = rv["image_id"].isin(eval_ids) | rv["sha1"].isin(eval_sha)
    n_dropped = int(dup_mask.sum())
    if n_dropped:
        rv = rv.loc[~dup_mask].reset_index(drop=True)
        rv.to_parquet(rv_path, index=False)

    result = {
        "real_val_images": int(len(rv)),
        "real_eval_images": int(len(re)),
        "validation_rows_dropped_as_cross_split_duplicates": n_dropped,
        "val_eval_id_overlap": len(set(rv["image_id"]) & eval_ids),
        "val_eval_sha_overlap": len(set(rv["sha1"]) & eval_sha),
        "val_vs_synthetic_sha_overlap": len(set(rv["sha1"]) & syn_sha),
        "eval_vs_synthetic_sha_overlap": len(set(re["sha1"]) & syn_sha),
    }
    # Synthetic training data leaking into either real split is a hard failure.
    if result["val_vs_synthetic_sha_overlap"] or result["eval_vs_synthetic_sha_overlap"]:
        raise AssertionError(f"synthetic-train leakage into a real split: {result}")
    # The de-duplication above must have fully separated val from eval.
    if result["val_eval_id_overlap"] or result["val_eval_sha_overlap"]:
        raise AssertionError(f"val/eval still overlap after de-duplication: {result}")
    return result


_MODELS: dict[str, ModelConfig] = {}


def _model_configs(st: dict) -> dict[str, ModelConfig]:
    if _MODELS:
        return _MODELS
    retrain = st["steps"]["retrain_baselines"]["result"]
    _MODELS["phase3a_v1"] = ModelConfig(
        "phase3a_v1",
        ML_ROOT / "runs" / "phase2-baseline-v1_20260828-123813" / "model.joblib",
        "cvfeat-v1",
        "Phase 3A baseline: synthetic-trained, cvfeat-v1 (buggy blockiness)",
    )
    _MODELS["v2fix"] = ModelConfig(
        "v2fix",
        Path(retrain["v2fix"]["bundle"]),
        "cvfeat-v2",
        "Same synthetic data, cvfeat-v2 (fixed blockiness), retrained",
    )
    _MODELS["v2blur"] = ModelConfig(
        "v2blur",
        Path(retrain["v2blur"]["bundle"]),
        "cvfeat-v2",
        "cvfeat-v2 + post-blur sensor noise synthetic dataset, retrained",
    )
    return _MODELS


def step_threshold_and_calibration(st: dict) -> dict:
    rv = pd.read_parquet(ML_ROOT / "data" / "processed" / "realval_features.parquet")
    rv = vote_threshold_labels(rv, VOTE_MIN)
    eval_labels = evaluable_labels()
    out: dict = {}

    for mkey, mc in _model_configs(st).items():
        bundle = joblib.load(mc.bundle_path)
        x = feature_matrix(rv, mc.feature_version)
        scores = _predict_scores(bundle, x)

        truth = {lbl: rv[f"label_{lbl}"].to_numpy(int) for lbl in eval_labels}

        # --- raw threshold selection (uncalibrated) ---
        raw_thr, raw_detail, raw_sweeps = {}, {}, {}
        for lbl in eval_labels:
            t, detail = select_threshold(truth[lbl], scores[lbl])
            raw_thr[lbl] = t
            raw_detail[lbl] = detail
            raw_sweeps[lbl] = sweep_thresholds(truth[lbl], scores[lbl])
        ThresholdSet(
            version=f"phase3b-thr-raw-{mkey}",
            parent=mc.key,
            model_artifact=str(mc.bundle_path),
            feature_version=mc.feature_version,
            criterion="f1",
            fitted_on="VizWiz train sample (real validation)",
            seed=20260828,
            thresholds=raw_thr,
            selection_detail=raw_detail,
        ).save(RUN_DIR / "thresholds" / f"raw_{mkey}.json")

        # --- probability calibration (isotonic) ---
        cal = fit_calibrators(
            {lbl: scores[lbl] for lbl in eval_labels},
            truth,
            version=f"phase3b-cal-{mkey}",
            parent=mc.key,
            method="isotonic",
            fitted_on="VizWiz train sample (real validation)",
        )
        cal.save(RUN_DIR / "calibrators" / f"{mkey}.joblib")

        # --- threshold selection on calibrated probabilities ---
        cal_thr, cal_detail = {}, {}
        for lbl in eval_labels:
            p = cal.transform(lbl, scores[lbl])
            t, detail = select_threshold(truth[lbl], p)
            cal_thr[lbl] = t
            cal_detail[lbl] = detail
        ThresholdSet(
            version=f"phase3b-thr-cal-{mkey}",
            parent=mc.key,
            model_artifact=str(mc.bundle_path),
            feature_version=mc.feature_version,
            criterion="f1 (on calibrated prob)",
            fitted_on="VizWiz train sample",
            seed=20260828,
            thresholds=cal_thr,
            selection_detail=cal_detail,
        ).save(RUN_DIR / "thresholds" / f"cal_{mkey}.json")

        out[mkey] = {
            "raw_thresholds": raw_thr,
            "calibrated_thresholds": cal_thr,
            "calibration_diagnostics": {
                lbl: {k: v for k, v in cal.diagnostics[lbl].items() if k != "reliability_after"}
                for lbl in eval_labels
            },
        }
        (RUN_DIR / "threshold_sweeps").mkdir(parents=True, exist_ok=True)
        (RUN_DIR / "threshold_sweeps" / f"{mkey}.json").write_text(
            json.dumps(raw_sweeps, indent=2), encoding="utf-8"
        )
    return out


def _evaluate_row(bundle, feat_version, cal, thr, re_df, eval_labels) -> dict:
    x = feature_matrix(re_df, feat_version)
    scores = _predict_scores(bundle, x)
    y_true = np.column_stack([re_df[f"label_{lbl}"].to_numpy(int) for lbl in eval_labels])
    prob = np.column_stack(
        [(cal.transform(lbl, scores[lbl]) if cal else scores[lbl]) for lbl in eval_labels]
    )
    pred = np.column_stack(
        [(prob[:, i] >= thr[lbl]).astype(int) for i, lbl in enumerate(eval_labels)]
    )
    rep = multilabel_report(y_true, pred, eval_labels, prob)
    rep["primary_macro_f1"] = round(
        float(np.mean([rep["per_class"][x]["f1"] for x in PRIMARY_LABELS])), 4
    )
    rep["primary_labels"] = PRIMARY_LABELS
    return rep


def step_final_evaluation(st: dict) -> dict:
    """Read the Phase 3A evaluation set exactly once."""
    re_df = pd.read_parquet(ML_ROOT / "data" / "processed" / "realeval_features.parquet")
    re_df = vote_threshold_labels(re_df, VOTE_MIN)
    eval_labels = evaluable_labels()
    mcs = _model_configs(st)

    a = joblib.load(mcs["phase3a_v1"].bundle_path)
    syn_thr = {lbl: a["thresholds"][lbl] for lbl in eval_labels}

    rows: dict = {}
    # A: Phase 3A exact (v1 model, v1 features, synthetic thresholds, no calibration)
    rows["A_phase3a_baseline"] = _evaluate_row(a, "cvfeat-v1", None, syn_thr, re_df, eval_labels)
    # B: + real-val F1 thresholds (still v1 model/features, no calibration)
    b_thr = ThresholdSet.load(RUN_DIR / "thresholds" / "raw_phase3a_v1.json").thresholds
    rows["B_plus_threshold_calibration"] = _evaluate_row(
        a, "cvfeat-v1", None, b_thr, re_df, eval_labels
    )
    # C: + probability calibration (isotonic) and thresholds re-picked on calibrated prob
    c_cal = PerLabelCalibrator.load(RUN_DIR / "calibrators" / "phase3a_v1.joblib")
    c_thr = ThresholdSet.load(RUN_DIR / "thresholds" / "cal_phase3a_v1.json").thresholds
    rows["C_plus_probability_calibration"] = _evaluate_row(
        a, "cvfeat-v1", c_cal, c_thr, re_df, eval_labels
    )
    # D: + feature fix (v2 model/features), calibrated
    d_b = joblib.load(mcs["v2fix"].bundle_path)
    d_cal = PerLabelCalibrator.load(RUN_DIR / "calibrators" / "v2fix.joblib")
    d_thr = ThresholdSet.load(RUN_DIR / "thresholds" / "cal_v2fix.json").thresholds
    rows["D_plus_feature_fix"] = _evaluate_row(d_b, "cvfeat-v2", d_cal, d_thr, re_df, eval_labels)
    # E: + blur realism (v2blur model), calibrated
    e_b = joblib.load(mcs["v2blur"].bundle_path)
    e_cal = PerLabelCalibrator.load(RUN_DIR / "calibrators" / "v2blur.joblib")
    e_thr = ThresholdSet.load(RUN_DIR / "thresholds" / "cal_v2blur.json").thresholds
    rows["E_plus_blur_realism"] = _evaluate_row(e_b, "cvfeat-v2", e_cal, e_thr, re_df, eval_labels)

    (RUN_DIR / "final_evaluation.json").write_text(
        json.dumps({"vote_min": VOTE_MIN, "rows": rows}, indent=2), encoding="utf-8"
    )
    return {
        k: {"primary_macro_f1": v["primary_macro_f1"], "macro_f1": v["macro_f1"]}
        for k, v in rows.items()
    }


def step_write_reports(st: dict) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    final = json.loads((RUN_DIR / "final_evaluation.json").read_text())["rows"]
    eval_labels = evaluable_labels()

    # ablation table
    order = [
        ("A_phase3a_baseline", "Phase 3A baseline (v1 model, synthetic thresholds)"),
        ("B_plus_threshold_calibration", "+ real-val F1 thresholds"),
        ("C_plus_probability_calibration", "+ isotonic probability calibration"),
        ("D_plus_feature_fix", "+ cvfeat-v2 blockiness fix (retrained)"),
        ("E_plus_blur_realism", "+ post-blur sensor noise (retrained)"),
    ]
    lines = [
        "# Phase 3B — ablation (real VizWiz `val` evaluation set, ≥3 votes)",
        "",
        "Primary macro-F1 = mean F1 over blur / underexposure / overexposure. "
        "`defect` is reported separately and excluded (see below).",
        "",
        "| step | primary macro-F1 | 4-label macro-F1 | blur F1 | underexp F1 | overexp F1 | defect F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, label in order:
        r = final[key]
        pc = r["per_class"]
        lines.append(
            f"| {label} | **{r['primary_macro_f1']}** | {r['macro_f1']} | "
            f"{pc['blur']['f1']} | {pc['underexposure']['f1']} | "
            f"{pc['overexposure']['f1']} | {pc['defect']['f1']} |"
        )
    lines += [
        "",
        "Per-label precision / recall / ROC-AUC / PR-AUC and confusion counts are in "
        "`runs/phase3b-calibration-v1/final_evaluation.json`.",
        "",
        "## defect",
        "",
        f"defect F1 moves from {final['A_phase3a_baseline']['per_class']['defect']['f1']} to "
        f"{final['E_plus_blur_realism']['per_class']['defect']['f1']}; ROC-AUC stays around "
        f"{final['E_plus_blur_realism']['per_class']['defect'].get('roc_auc')}. It remains "
        "below the level of a usable classifier and is excluded from the primary metric. "
        "See docs for the localisation recommendation.",
    ]
    (REPORT_DIR / "ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # thresholds before/after
    thr_rows = [
        "# Thresholds: synthetic-val vs real-val",
        "",
        "| label | Phase 3A (synthetic) | real-val F1 (v1) | real-val F1 on calibrated prob (v2fix) |",
        "|---|---|---|---|",
    ]
    a = joblib.load(_model_configs(st)["phase3a_v1"].bundle_path)
    raw_v1 = ThresholdSet.load(RUN_DIR / "thresholds" / "raw_phase3a_v1.json").thresholds
    cal_v2 = ThresholdSet.load(RUN_DIR / "thresholds" / "cal_v2fix.json").thresholds
    for lbl in eval_labels:
        thr_rows.append(f"| {lbl} | {a['thresholds'][lbl]} | {raw_v1[lbl]} | {cal_v2[lbl]} |")
    (REPORT_DIR / "thresholds.md").write_text("\n".join(thr_rows) + "\n", encoding="utf-8")

    # calibration diagnostics
    cal_summary = {}
    for mkey in ("phase3a_v1", "v2fix", "v2blur"):
        c = PerLabelCalibrator.load(RUN_DIR / "calibrators" / f"{mkey}.joblib")
        cal_summary[mkey] = {
            lbl: {
                k: c.diagnostics[lbl].get(k) for k in ("support", "calibrated", "before", "after")
            }
            for lbl in eval_labels
        }
    (REPORT_DIR / "calibration.json").write_text(
        json.dumps(cal_summary, indent=2), encoding="utf-8"
    )

    # feature fix comparison
    fr = st["steps"]["feature_report_v2"]["result"]
    (REPORT_DIR / "feature_fix.json").write_text(
        json.dumps(
            {
                "v1_blockiness_stats": fr.get("v1_blockiness"),
                "v2_blockiness_stats": fr.get("v2_blockiness"),
                "v2_total_nan": fr.get("total_nan"),
                "v2_total_inf": fr.get("total_inf"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"reports": sorted(p.name for p in REPORT_DIR.iterdir())}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_all() -> None:
    st = _load_status()
    st.setdefault("started", time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    st["feature_version"] = FEATURE_VERSION
    _save_status(st)

    _step(st, "synthetic_features_v2", step_synthetic_features_v2)
    _step(st, "blurnoise_dataset", step_blurnoise_dataset)
    _step(st, "retrain_baselines", lambda: step_retrain_baselines(st))
    _step(st, "feature_report_v2", step_feature_report_v2)
    _step(st, "real_val_features", step_real_val_features)
    _step(st, "real_eval_features", step_real_eval_features)
    _step(st, "leakage_check", step_leakage_check)
    _step(st, "threshold_and_calibration", lambda: step_threshold_and_calibration(st))
    _step(st, "final_evaluation", lambda: step_final_evaluation(st))
    _step(st, "write_reports", lambda: step_write_reports(st))

    st["state"] = "complete"
    st["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    _save_status(st)
    print("[phase3b] COMPLETE", flush=True)
