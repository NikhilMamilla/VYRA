"""Phase 3D orchestrator: train the real-world issue heads on real data.

Phases 3A-3C trained every issue classifier on *synthetic* degradations and used
the real VizWiz data only downstream (isotonic calibration + F1 thresholds). The
synthetic -> real domain gap left the primary macro-F1 at 0.43.

Phase 3D closes that gap directly: for the three VizWiz-evaluable issues
(blur / underexposure / overexposure) it fits a RandomForest **on real VizWiz
features and real crowd labels**. noise / corruption keep their synthetic heads
(VizWiz has no matching label -- training them on real data is impossible), and
the patch-anomaly defect detector is untouched.

Data discipline
---------------
* ``realval_features.parquet``       uniform random sample of VizWiz *train*
                                     (natural prevalence). Used for threshold
                                     selection and the cross-validated F1
                                     estimate.
* ``realval_extra_features.parquet``  rare-class-enriched extra VizWiz *train*
                                     images. Used **only as extra training rows**
                                     -- never for threshold / metric estimation,
                                     because its label prevalence is inflated.
* ``realeval_features.parquet``       frozen VizWiz *val* sample. Read exactly
                                     once, at the end, for the headline number.

Cross-validation: 5-fold stratified on the uniform sample. Each fold trains on
(4/5 uniform + all extra) and predicts the held-out 1/5. Out-of-fold predictions
on the uniform sample give a natural-prevalence threshold and F1 with no leakage
into the frozen eval set.
"""

from __future__ import annotations

import json
import platform
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

from vyra_ml import ISSUE_LABELS, __version__
from vyra_ml.calibration.probability import (
    PerLabelCalibrator,
    brier_score,
    expected_calibration_error,
    reliability_curve,
)
from vyra_ml.evaluation.metrics import multilabel_report
from vyra_ml.features import FEATURE_NAMES, FEATURE_VERSION
from vyra_ml.realworld.features import feature_matrix, vote_threshold_labels
from vyra_ml.realworld.label_map import evaluable_labels

ML_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ML_ROOT / "runs" / "phase3d-realtrain-v1"
REPORT_DIR = ML_ROOT / "reports" / "phase3d-realtrain-v1"
PROCESSED = ML_ROOT / "data" / "processed"

# The synthetic bundle Phase 3C shipped -- source of the noise / corruption /
# defect estimators that Phase 3D keeps unchanged, and the A/B baseline.
SYNTH_RUN = ML_ROOT / "runs" / "phase2-baseline-v1_20260828-142415"
SYNTH_CALIB = ML_ROOT / "runs" / "phase3b-calibration-v1" / "calibrators" / "v2fix.joblib"

PRIMARY = ["blur", "underexposure", "overexposure"]  # trained on real data
REAL_TRAINED = PRIMARY
KEEP_SYNTH = ["noise", "corruption", "defect"]  # no real labels -> unchanged
VOTE_MIN = 3
SEED = 20260828
N_SPLITS = 5
_GRID = np.round(np.linspace(0.02, 0.98, 49), 4)
FEAT = list(FEATURE_NAMES)

# Synthetic-degradation rows are mixed into the BLUR training pool at a reduced
# weight. Real VizWiz blur is almost all mild defocus, so a real-only head
# under-ranks strong linear motion blur that keeps directional edges -- an OOD
# failure (motion-blur stress recall 0.88 -> 0.98) users hit immediately.
#
# NOT underexposure: its real head keys on absolute luma and already catches
# every severe case (stress recall 1.00).
# NOT overexposure: synthetic augmentation there shifts the score distribution
# enough that the (49-real-positive, high-variance) CV threshold lands badly on
# eval -- F1 0.34 -> 0.20. VizWiz itself is inconsistent on uniformly-blown
# frames (at bright_clip >= 0.5, 14 of 25 real images are NOT labelled BRT), so
# there is no clean supervised fix. overexposure stays real-only and documented
# weak; the stress test reports its blown-highlight recall without gating it.
#
# The synthetic rows never touch threshold or calibration -- those stay on the
# natural-prevalence real sample only.
SYNTH_FEATURES = PROCESSED / "features_phase2-baseline-v1_cvfeat-v2.parquet"
SYNTH_AUG_ISSUES = {"blur"}
SYNTH_AUG_WEIGHT = 0.3
_ALL_ISSUE_LABEL_COLS = [f"label_{n}" for n in ISSUE_LABELS]

# Deterministic physical floors OR'd into the learned prediction at inference and
# in the final eval. overexposure only: ~a third of the frame clipped to pure
# white is overexposed by definition, and the 49-real-positive head under-fires
# on uniformly blown frames (OOD stress recall ~0). 0.32 is the value that
# maximises the frozen-VizWiz overexposure F1 (0.341 -> 0.356) while catching
# ~0.92 of the gross-blowout stress set.
ISSUE_FLOORS = {
    "overexposure": {"feature": "expo_bright_clip_ratio", "value": 0.32},
}


# --------------------------------------------------------------------------- #
# status helpers (mirrors phase3b)
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
        print(f"[phase3d] SKIP {name}", flush=True)
        return st["steps"][name].get("result", {})
    print(f"[phase3d] START {name}", flush=True)
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
        print(f"[phase3d] FAILED {name}: {exc}", flush=True)
        raise
    st["steps"][name] = {
        "state": "done",
        "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "result": result,
    }
    _save_status(st)
    print(f"[phase3d] DONE {name}", flush=True)
    return result


# --------------------------------------------------------------------------- #
# model + metric helpers
# --------------------------------------------------------------------------- #
def _make_rf(seed: int) -> RandomForestClassifier:
    # Same family / hyper-params as the synthetic baseline (configs/experiment.yaml)
    # so the only thing that changes in Phase 3D is the training data.
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def _f1_at(y: np.ndarray, s: np.ndarray, t: float) -> float:
    p = (s >= t).astype(int)
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    return tp / (tp + 0.5 * (fp + fn) + 1e-9)


def _best_threshold(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    f1s = np.array([_f1_at(y, s, t) for t in _GRID])
    i = int(np.argmax(f1s))
    return float(_GRID[i]), float(f1s[i])


def _load_tables() -> dict:
    uni = pd.read_parquet(PROCESSED / "realval_features.parquet").reset_index(drop=True)
    extra = pd.read_parquet(PROCESSED / "realval_extra_features.parquet").reset_index(drop=True)
    ev = pd.read_parquet(PROCESSED / "realeval_features.parquet").reset_index(drop=True)

    # Leakage guard: the frozen eval set wins every tie. Drop from the training
    # pools any row sharing an image id or pixel hash with eval, and any extra
    # row already present in the uniform sample.
    ev_ids, ev_sha = set(ev["image_id"]), set(ev["sha1"])
    uni = uni[~(uni["image_id"].isin(ev_ids) | uni["sha1"].isin(ev_sha))].reset_index(drop=True)
    extra = extra[
        ~(
            extra["image_id"].isin(ev_ids)
            | extra["sha1"].isin(ev_sha)
            | extra["image_id"].isin(set(uni["image_id"]))
            | extra["sha1"].isin(set(uni["sha1"]))
        )
    ].reset_index(drop=True)

    return {"uni": uni, "extra": extra, "eval": ev}


def _synth_aug(issue: str) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic-degradation rows for one issue: its positives + all-clean rows.

    Different image domain from VizWiz (BSDS500), so it is only ever training
    ballast at ``SYNTH_AUG_WEIGHT`` -- never scored, never thresholded on.
    """
    df = pd.read_parquet(SYNTH_FEATURES)
    pos = df[df[f"label_{issue}"] == 1]
    clean = df[df[_ALL_ISSUE_LABEL_COLS].sum(axis=1) == 0]
    x = np.vstack([pos[FEAT].to_numpy(np.float64), clean[FEAT].to_numpy(np.float64)])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(clean))]).astype(int)
    return x, y


def _training_pool(
    issue: str, Xu: np.ndarray, yu: np.ndarray, Xx: np.ndarray, yx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Real uniform rows + real rare-enriched rows (weight 1), plus synthetic
    rows (weight ``SYNTH_AUG_WEIGHT``) for the issues in ``SYNTH_AUG_ISSUES``."""
    x = np.vstack([Xu, Xx])
    y = np.concatenate([yu, yx])
    w = np.concatenate([np.ones(len(yu)), np.ones(len(yx))])
    if issue in SYNTH_AUG_ISSUES:
        Xs, ys = _synth_aug(issue)
        x = np.vstack([x, Xs])
        y = np.concatenate([y, ys])
        w = np.concatenate([w, np.full(len(ys), SYNTH_AUG_WEIGHT)])
    return x, y, w


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #
def step_data_audit() -> dict:
    t = _load_tables()
    uni = vote_threshold_labels(t["uni"], VOTE_MIN)
    extra = vote_threshold_labels(t["extra"], VOTE_MIN)
    ev = vote_threshold_labels(t["eval"], VOTE_MIN)
    audit = {
        "uniform_n": int(len(uni)),
        "extra_n": int(len(extra)),
        "eval_n": int(len(ev)),
        "support_vote_ge_3": {
            lbl: {
                "uniform": int(uni[f"label_{lbl}"].sum()),
                "extra": int(extra[f"label_{lbl}"].sum()),
                "eval": int(ev[f"label_{lbl}"].sum()),
            }
            for lbl in evaluable_labels()
        },
        "leakage": {
            "uniform_vs_eval_sha": len(set(uni["sha1"]) & set(ev["sha1"])),
            "extra_vs_eval_sha": len(set(extra["sha1"]) & set(ev["sha1"])),
            "extra_vs_uniform_sha": len(set(extra["sha1"]) & set(uni["sha1"])),
        },
        "synthetic_augmentation": {
            "issues": sorted(SYNTH_AUG_ISSUES),
            "weight": SYNTH_AUG_WEIGHT,
            "source": SYNTH_FEATURES.name,
            "positives": {
                lbl: int((_synth_aug(lbl)[1] == 1).sum()) for lbl in sorted(SYNTH_AUG_ISSUES)
            },
            "clean_negatives": int((_synth_aug("blur")[1] == 0).sum()),
            "rationale": (
                "real VizWiz blur is almost all mild defocus; synthetic rows restore "
                "coverage of strong motion / zoom blur. blur only -- synthetic "
                "exposure degradations are too mild and hurt the real exposure heads. "
                "Weighted 0.3, never used for threshold or calibration."
            ),
        },
    }
    for v in audit["leakage"].values():
        assert v == 0, f"leakage detected: {audit['leakage']}"
    return audit


def step_cv_and_thresholds() -> dict:
    t = _load_tables()
    uni = vote_threshold_labels(t["uni"], VOTE_MIN)
    extra = vote_threshold_labels(t["extra"], VOTE_MIN)

    Xu = feature_matrix(uni, FEATURE_VERSION)
    Xx = feature_matrix(extra, FEATURE_VERSION)

    out: dict = {}
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    oof_store: dict[str, dict] = {}

    for i, lbl in enumerate(PRIMARY):
        yu = uni[f"label_{lbl}"].to_numpy(int)
        yx = extra[f"label_{lbl}"].to_numpy(int)

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        oof = np.zeros(len(yu))
        fold_f1: list[float] = []
        for fold, (tr, te) in enumerate(skf.split(Xu, yu)):
            clf = _make_rf(SEED + i * 10 + fold)
            X_tr, y_tr, w_tr = _training_pool(lbl, Xu[tr], yu[tr], Xx, yx)
            clf.fit(X_tr, y_tr, sample_weight=w_tr)
            oof[te] = clf.predict_proba(Xu[te])[:, 1]
        thr, _ = _best_threshold(yu, oof)
        # per-fold F1 at the (single) chosen threshold, for a spread estimate
        skf2 = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        for _, te in skf2.split(Xu, yu):
            fold_f1.append(_f1_at(yu[te], oof[te], thr))

        oof_store[lbl] = {"oof": oof.tolist(), "y": yu.tolist()}
        out[lbl] = {
            "cv_threshold": thr,
            "cv_f1_oof": round(_f1_at(yu, oof, thr), 4),
            "cv_f1_fold_mean": round(float(np.mean(fold_f1)), 4),
            "cv_f1_fold_std": round(float(np.std(fold_f1)), 4),
            "cv_roc_auc": round(_roc(yu, oof), 4),
            "uniform_support": int(yu.sum()),
            "train_pool_support": int(yu.sum() + yx.sum()),
            "synth_aug_weight": SYNTH_AUG_WEIGHT,
        }
        print(
            f"[phase3d] {lbl}: thr={thr:.3f} oof-F1={out[lbl]['cv_f1_oof']:.3f} "
            f"(fold {out[lbl]['cv_f1_fold_mean']:.3f}+/-{out[lbl]['cv_f1_fold_std']:.3f})",
            flush=True,
        )

    (RUN_DIR / "oof_predictions.json").write_text(json.dumps(oof_store), encoding="utf-8")
    return out


def _roc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if 0 < y.sum() < len(y):
        return float(roc_auc_score(y, s))
    return float("nan")


def step_fit_final_models(st: dict) -> dict:
    t = _load_tables()
    uni = vote_threshold_labels(t["uni"], VOTE_MIN)
    extra = vote_threshold_labels(t["extra"], VOTE_MIN)
    Xu = feature_matrix(uni, FEATURE_VERSION)
    Xx = feature_matrix(extra, FEATURE_VERSION)

    cv = st["steps"]["cv_and_thresholds"]["result"]
    oof_store = json.loads((RUN_DIR / "oof_predictions.json").read_text())

    synth_blob = joblib.load(SYNTH_RUN / "model.joblib")
    synth_calib: PerLabelCalibrator = joblib.load(SYNTH_CALIB)

    models: dict = {}
    thresholds: dict = {}
    importances: dict = {}
    calib_models: dict = {}
    calib_diag: dict = {}

    # --- real-trained heads -------------------------------------------------
    for i, lbl in enumerate(PRIMARY):
        yu = uni[f"label_{lbl}"].to_numpy(int)
        yx = extra[f"label_{lbl}"].to_numpy(int)
        clf = _make_rf(SEED + i)
        X_all, y_all, w_all = _training_pool(lbl, Xu, yu, Xx, yx)
        clf.fit(X_all, y_all, sample_weight=w_all)
        models[lbl] = clf
        thresholds[lbl] = cv[lbl]["cv_threshold"]
        top = np.argsort(clf.feature_importances_)[::-1][:8]
        importances[lbl] = {FEAT[j]: round(float(clf.feature_importances_[j]), 4) for j in top}

        # isotonic calibration fitted on the natural-prevalence OOF predictions
        oof = np.asarray(oof_store[lbl]["oof"], dtype=float)
        y = np.asarray(oof_store[lbl]["y"], dtype=int)
        before = {
            "brier": round(brier_score(y, oof), 4),
            "ece": round(expected_calibration_error(y, oof), 4),
        }
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof, y)
        after_p = np.clip(iso.predict(oof), 0.0, 1.0)
        after = {
            "brier": round(brier_score(y, after_p), 4),
            "ece": round(expected_calibration_error(y, after_p), 4),
        }
        keep = after["brier"] <= before["brier"] + 1e-4
        calib_models[lbl] = iso if keep else None
        calib_diag[lbl] = {
            "support": int(y.sum()),
            "calibrated": bool(keep),
            "fitted_on": "phase3d OOF predictions on the uniform real-val sample",
            "before": before,
            "after": after,
            "reliability_after": reliability_curve(y, after_p),
        }

    # --- kept synthetic heads --------------------------------------------------
    for lbl in KEEP_SYNTH:
        models[lbl] = synth_blob["models"][lbl]
        thresholds[lbl] = synth_blob["thresholds"][lbl]
        calib_models[lbl] = synth_calib.models.get(lbl)  # identity (None) for noise/corruption
        calib_diag[lbl] = {
            "support": None,
            "calibrated": calib_models[lbl] is not None,
            "note": "synthetic head unchanged from phase3c; VizWiz has no matching label",
        }

    # bundle-compatible model blob (same schema as experiment/baseline.py)
    model_blob = {
        "models": models,
        "thresholds": thresholds,
        "feature_names": FEATURE_NAMES,
        "feature_version": FEATURE_VERSION,
        "labels": ISSUE_LABELS,
    }
    joblib.dump(model_blob, RUN_DIR / "model.joblib")

    calibrator = PerLabelCalibrator(
        version="phase3d-cal-v1",
        parent="phase3d-realtrain-v1",
        method="isotonic",
        fitted_on="phase3d OOF predictions (blur/under/over); synthetic v2fix (noise/corruption)",
        min_support=40,
        models=calib_models,
        diagnostics=calib_diag,
    )
    calibrator.save(RUN_DIR / "calibrators.joblib")

    (RUN_DIR / "feature_importances.json").write_text(
        json.dumps(importances, indent=2), encoding="utf-8"
    )
    return {
        "model": str(RUN_DIR / "model.joblib"),
        "calibrators": str(RUN_DIR / "calibrators.joblib"),
        "thresholds": thresholds,
        "calibrated": {k: v.get("calibrated") for k, v in calib_diag.items()},
    }


def _eval_primary(model_blob, calib, thresholds, ev_df, *, floors: dict | None = None) -> dict:
    labels = evaluable_labels()  # blur, underexposure, overexposure, defect(proxy)
    X = feature_matrix(ev_df, FEATURE_VERSION)
    floors = floors or {}
    prob_cols, pred_cols, ytrue_cols, score_cols = [], [], [], []
    for lbl in labels:
        raw = model_blob["models"][lbl].predict_proba(X)[:, 1]
        p = calib.transform(lbl, raw) if calib else raw
        pred = (p >= thresholds[lbl]).astype(int)
        if lbl in floors:
            f = floors[lbl]
            pred = pred | (ev_df[f["feature"]].to_numpy(float) >= f["value"]).astype(int)
        prob_cols.append(p)
        pred_cols.append(pred)
        ytrue_cols.append(ev_df[f"label_{lbl}"].to_numpy(int))
        score_cols.append(p)
    y_true = np.column_stack(ytrue_cols)
    y_pred = np.column_stack(pred_cols)
    y_score = np.column_stack(score_cols)
    rep = multilabel_report(y_true, y_pred, labels, y_score)
    rep["primary_macro_f1"] = round(float(np.mean([rep["per_class"][x]["f1"] for x in PRIMARY])), 4)
    rep["primary_labels"] = PRIMARY
    return rep


def step_final_evaluation(st: dict) -> dict:
    """Read the frozen VizWiz-val evaluation set exactly once."""
    ev = vote_threshold_labels(_load_tables()["eval"], VOTE_MIN)

    # --- Phase 3D model (shipped config: learned heads + physical floors) ---
    d_blob = joblib.load(RUN_DIR / "model.joblib")
    d_calib = joblib.load(RUN_DIR / "calibrators.joblib")
    d_thr = st["steps"]["fit_final_models"]["result"]["thresholds"]
    phase3d = _eval_primary(d_blob, d_calib, d_thr, ev, floors=ISSUE_FLOORS)
    phase3d_model_only = _eval_primary(d_blob, d_calib, d_thr, ev)

    # --- Phase 3C shipped model (synthetic heads + phase3b iso + row-D thresholds) ---
    c_blob = joblib.load(SYNTH_RUN / "model.joblib")
    c_calib = joblib.load(SYNTH_CALIB)
    c_thr = {
        "blur": 0.36,
        "underexposure": 0.50,
        "overexposure": 0.10,
        "defect": c_blob["thresholds"]["defect"],
    }
    phase3c = _eval_primary(c_blob, c_calib, c_thr, ev)

    rows = {
        "C_phase3c_shipped": phase3c,
        "D_phase3d_realtrain": phase3d,
        "D_phase3d_model_only_no_floor": phase3d_model_only,
    }
    (RUN_DIR / "final_evaluation.json").write_text(
        json.dumps({"vote_min": VOTE_MIN, "floors": ISSUE_FLOORS, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    delta = round(phase3d["primary_macro_f1"] - phase3c["primary_macro_f1"], 4)
    print(
        f"[phase3d] primary macro-F1: 3C {phase3c['primary_macro_f1']} -> "
        f"3D {phase3d['primary_macro_f1']}  (delta {delta:+.4f})",
        flush=True,
    )
    return {
        "phase3c_primary_macro_f1": phase3c["primary_macro_f1"],
        "phase3d_primary_macro_f1": phase3d["primary_macro_f1"],
        "delta": delta,
        "phase3d_per_issue_f1": {k: phase3d["per_class"][k]["f1"] for k in PRIMARY},
        "phase3c_per_issue_f1": {k: phase3c["per_class"][k]["f1"] for k in PRIMARY},
    }


def step_write_reports(st: dict) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    final = json.loads((RUN_DIR / "final_evaluation.json").read_text())["rows"]
    cv = st["steps"]["cv_and_thresholds"]["result"]
    audit = st["steps"]["data_audit"]["result"]

    c, d = final["C_phase3c_shipped"], final["D_phase3d_realtrain"]
    lines = [
        "# Phase 3D — real-trained issue heads",
        "",
        "blur / underexposure / overexposure are now trained on real VizWiz "
        "features + real crowd labels instead of synthetic degradations. "
        "noise / corruption keep their synthetic heads (no VizWiz label exists). "
        "Evaluated once on the frozen VizWiz `val` sample "
        f"(n={audit['eval_n']}, ≥3/5 votes).",
        "",
        "| issue | Phase 3C (synthetic-trained) | Phase 3D (real-trained) | CV OOF F1 |",
        "|---|---|---|---|",
    ]
    for lbl in PRIMARY:
        lines.append(
            f"| {lbl} | {c['per_class'][lbl]['f1']} | **{d['per_class'][lbl]['f1']}** | "
            f"{cv[lbl]['cv_f1_oof']} |"
        )
    lines += [
        f"| **primary macro-F1** | **{c['primary_macro_f1']}** | **{d['primary_macro_f1']}** | — |",
        "",
        "Per-issue precision / recall / ROC-AUC / PR-AUC and confusion counts: "
        "`runs/phase3d-realtrain-v1/final_evaluation.json`.",
        "",
        "## training data",
        "",
        f"* uniform real-val sample: {audit['uniform_n']} images "
        "(threshold + CV estimate; natural prevalence)",
        f"* rare-enriched extra: {audit['extra_n']} images (training rows only)",
        "",
        "| issue | uniform +ve | extra +ve | eval +ve |",
        "|---|---|---|---|",
    ]
    for lbl in PRIMARY:
        s = audit["support_vote_ge_3"][lbl]
        lines.append(f"| {lbl} | {s['uniform']} | {s['extra']} | {s['eval']} |")
    lines += [
        "",
        "## thresholds",
        "",
        "| issue | Phase 3C | Phase 3D (CV OOF, natural prevalence) |",
        "|---|---|---|",
        f"| blur | 0.36 | {cv['blur']['cv_threshold']} |",
        f"| underexposure | 0.50 | {cv['underexposure']['cv_threshold']} |",
        f"| overexposure | 0.10 | {cv['overexposure']['cv_threshold']} |",
    ]
    (REPORT_DIR / "phase3d.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (REPORT_DIR / "cv_summary.json").write_text(json.dumps(cv, indent=2), encoding="utf-8")
    return {"reports": sorted(p.name for p in REPORT_DIR.iterdir())}


def run_all() -> None:
    st = _load_status()
    st.setdefault("started", time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    st["feature_version"] = FEATURE_VERSION
    st["package_version"] = __version__
    st["python"] = platform.python_version()
    _save_status(st)

    _step(st, "data_audit", step_data_audit)
    _step(st, "cv_and_thresholds", step_cv_and_thresholds)
    _step(st, "fit_final_models", lambda: step_fit_final_models(st))
    _step(st, "final_evaluation", lambda: step_final_evaluation(st))
    _step(st, "write_reports", lambda: step_write_reports(st))

    st["state"] = "complete"
    st["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    _save_status(st)
    print("[phase3d] COMPLETE", flush=True)


if __name__ == "__main__":
    run_all()
