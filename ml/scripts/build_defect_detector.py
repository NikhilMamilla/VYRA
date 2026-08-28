"""Calibrate the patch-anomaly defect detector on the synthetic defect set.

Fits ``P = sigmoid(a * (raw - b))`` and a decision threshold on the
``phase2-baseline-v1`` train+val splits, then measures image-level detection and
region localisation on the untouched **test** split. Writes
``runs/phase3c-defect-v1/defect_detector.json`` and a metrics file.

Run: ``python scripts/build_defect_detector.py``
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vyra_ml.defect.patch_anomaly import DefectDetector, patch_anomaly_map  # noqa: E402

ML_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ML_ROOT / "runs" / "phase3c-defect-v1"
MANIFEST = ML_ROOT / "data" / "manifests" / "manifest_phase2-baseline-v1.parquet"
PROCESSED = ML_ROOT / "data" / "processed"
SEED = 20260828
DATASET_LONG_EDGE = 384  # target_long_edge of phase2-baseline-v1 (bbox coord space)
DETECTOR_LONG_EDGE = 512  # patch_anomaly _LONG_EDGE


def _raw_score_and_region(path: Path) -> tuple[float, list[float] | None]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return 0.0, None
    amap = patch_anomaly_map(bgr)
    return amap["raw_score"], amap["region_norm"]


def _defect_bbox_norm(row: pd.Series) -> list[float] | None:
    for deg in json.loads(row["degradations_json"]):
        if deg["name"] == "defect" and deg["params"].get("bbox"):
            x, y, w, h = deg["params"]["bbox"]
            # bbox is in DATASET_LONG_EDGE pixel space; normalise to fractions.
            return [
                x / row["width"],
                y / row["height"],
                w / row["width"],
                h / row["height"],
            ]
    return None


def _centre_in_bbox(region_norm: list[float] | None, bbox_norm: list[float] | None) -> bool:
    if region_norm is None or bbox_norm is None:
        return False
    cx = region_norm[0] + region_norm[2] / 2
    cy = region_norm[1] + region_norm[3] / 2
    bx, by, bw, bh = bbox_norm
    return bx <= cx <= bx + bw and by <= cy <= by + bh


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    mf = pd.read_parquet(MANIFEST)
    rng = np.random.default_rng(SEED)

    fit = mf[mf.split.isin(["train", "val"])].reset_index(drop=True)
    # Balance the fit set: all defect samples + an equal random sample of the rest.
    pos = fit[fit.label_defect == 1]
    neg_all = fit[fit.label_defect == 0]
    n_neg = min(len(pos) * 2, len(neg_all))
    neg = neg_all.iloc[rng.choice(len(neg_all), size=n_neg, replace=False)]
    fit_rows = pd.concat([pos, neg]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    print(f"[defect] scoring {len(fit_rows)} fit images...", flush=True)
    t0 = time.perf_counter()
    fit_paths = [PROCESSED / r["split"] / f"{r['sample_id']}.jpg" for _, r in fit_rows.iterrows()]
    fit_scored = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_raw_score_and_region)(p) for p in fit_paths
    )
    raw = np.array([s for s, _ in fit_scored])
    y = fit_rows["label_defect"].to_numpy(int)
    np.savez(RUN_DIR / "fit_scores.npz", raw=raw, y=y)

    # Calibrate P = sigmoid(a * (raw - b)). A single-feature LogisticRegression
    # collapses here (classes overlap heavily, ROC-AUC ~0.6), so we fix the
    # operating point directly. defect is a weak *screening* signal: F1-optimal
    # selection degenerates to "flag everything", so instead we take the raw
    # threshold with the highest fit-set precision among those keeping recall
    # >= 0.25. b is that threshold (P(b) = 0.5); a is the slope putting the 90th
    # percentile of positive scores at P ~= 0.9.
    cand = np.unique(np.round(np.quantile(raw, np.linspace(0.02, 0.98, 193)), 4))
    sweep = []
    for c in cand:
        pred = (raw >= c).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        sweep.append((float(c), prec, rec, f1_score(y, pred, zero_division=0)))
    usable = [s for s in sweep if s[2] >= 0.25]
    b = max(usable, key=lambda s: (s[1], s[2]))[0] if usable else max(sweep, key=lambda s: s[3])[0]
    hi = float(np.percentile(raw[y == 1], 90))
    a = math.log(9.0) / max(1e-3, hi - b)
    best_t = 0.5  # decision on probability; equivalent to raw >= b
    fit_op = next(s for s in sweep if s[0] == b)
    print(f"[defect] fit op p={fit_op[1]:.3f} r={fit_op[2]:.3f} b={b:.3f}", flush=True)

    # --- untouched test-split evaluation ---
    test_rows = mf[mf.split == "test"].reset_index(drop=True)
    print(f"[defect] evaluating {len(test_rows)} test images...", flush=True)
    test_paths = [PROCESSED / r["split"] / f"{r['sample_id']}.jpg" for _, r in test_rows.iterrows()]
    test_scored = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_raw_score_and_region)(p) for p in test_paths
    )
    t_raw = np.array([s for s, _ in test_scored])
    t_y = test_rows["label_defect"].to_numpy(int)
    t_prob = 1.0 / (1.0 + np.exp(-a * (t_raw - b)))
    t_pred = (t_prob >= best_t).astype(int)

    hits, n_pos_regions = 0, 0
    for (_s, region), (_, r) in zip(test_scored, test_rows.iterrows(), strict=True):
        if r["label_defect"] == 1 and 1.0 / (1.0 + math.exp(-a * (_s - b))) >= best_t:
            n_pos_regions += 1
            if _centre_in_bbox(region, _defect_bbox_norm(r)):
                hits += 1

    tp = int(((t_pred == 1) & (t_y == 1)).sum())
    fp = int(((t_pred == 1) & (t_y == 0)).sum())
    fn = int(((t_pred == 0) & (t_y == 1)).sum())
    # Precision/recall curve on the test split (probability grid).
    pr_curve = []
    for pt in np.round(np.linspace(0.3, 0.9, 7), 2):
        pp = (t_prob >= pt).astype(int)
        c_tp = int(((pp == 1) & (t_y == 1)).sum())
        c_fp = int(((pp == 1) & (t_y == 0)).sum())
        c_fn = int(((pp == 0) & (t_y == 1)).sum())
        pr_curve.append(
            {
                "prob_threshold": float(pt),
                "precision": round(c_tp / (c_tp + c_fp), 4) if c_tp + c_fp else 0.0,
                "recall": round(c_tp / (c_tp + c_fn), 4) if c_tp + c_fn else 0.0,
            }
        )
    metrics = {
        "fit_images": len(fit_rows),
        "test_images": len(test_rows),
        "test_defect_support": int(t_y.sum()),
        "roc_auc": round(float(roc_auc_score(t_y, t_prob)), 4),
        "operating_point": {
            "prob_threshold": best_t,
            "f1": round(float(f1_score(t_y, t_pred, zero_division=0)), 4),
            "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
            "confusion": {"tp": tp, "fp": fp, "fn": fn},
        },
        "precision_recall_curve": pr_curve,
        "localisation_hit_rate": round(hits / n_pos_regions, 4) if n_pos_regions else None,
        "localisation_note": (
            "fraction of correctly-flagged defect images whose top patch centre "
            "falls inside the true defect bounding box"
        ),
    }

    detector = DefectDetector(
        a=round(a, 5),
        b=round(b, 5),
        threshold=best_t,
        version="phase3c-defect-v1",
        calibration={
            "method": "patch-anomaly raw score -> fixed-slope sigmoid; operating "
            "point at highest fit precision with recall >= 0.25",
            "fitted_on": "phase2-baseline-v1 train+val (synthetic defect set)",
            "seed": SEED,
            "detector_long_edge": DETECTOR_LONG_EDGE,
            "dataset_long_edge": DATASET_LONG_EDGE,
            "fit_operating_point": {
                "precision": round(fit_op[1], 4),
                "recall": round(fit_op[2], 4),
            },
            "test_metrics": metrics,
        },
    )
    detector.save(RUN_DIR / "defect_detector.json")
    (RUN_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[defect] a={a:.4f} b={b:.4f} threshold={best_t}", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"[defect] done in {time.perf_counter() - t0:.0f}s -> {RUN_DIR}", flush=True)


if __name__ == "__main__":
    main()
