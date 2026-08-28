"""Run the frozen Phase 2 baseline on real VizWiz images and measure transfer.

Scientific rules enforced here:
  * the model is loaded, never fitted;
  * thresholds come from the Phase 2 synthetic ``val`` split, unchanged;
  * the identical feature pipeline (same version, same working resolution) is used;
  * VYRA labels VizWiz does not support are never scored (no fake negatives);
  * a leakage check asserts no eval image is in Phase 2 training.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

from vyra_ml import ISSUE_LABELS, __version__
from vyra_ml.evaluation.metrics import multilabel_report
from vyra_ml.features import FEATURE_NAMES, extract_features
from vyra_ml.realworld.adapter import RealSample, load_image_record, prepare_samples
from vyra_ml.realworld.config import RealWorldEvalConfig
from vyra_ml.realworld.label_map import (
    POSITIVE_VOTE_MIN,
    binarize,
    evaluable_labels,
    mapping_table,
    vyra_to_vizwiz_code,
)
from vyra_ml.realworld.vizwiz import VIZWIZ_CODE_MEANING, VIZWIZ_FLAW_CODES


def _load_model_bundle(cfg: RealWorldEvalConfig) -> dict:
    bundle = joblib.load(cfg.model_path)
    if bundle["feature_version"] != cfg.expected_feature_version:
        raise RuntimeError(
            f"Feature version mismatch: model={bundle['feature_version']} "
            f"config expects {cfg.expected_feature_version}"
        )
    if tuple(bundle["feature_names"]) != FEATURE_NAMES:
        raise RuntimeError("Model feature names differ from the current extractor.")
    return bundle


def _phase2_context(cfg: RealWorldEvalConfig) -> dict:
    record = json.loads((cfg.model_run_dir / "experiment.json").read_text())
    metrics = json.loads((cfg.model_run_dir / "metrics.json").read_text())
    work_long_edge = record["config_snapshot"]["features"]["work_long_edge"]
    return {"record": record, "metrics": metrics, "work_long_edge": int(work_long_edge)}


def _assert_no_leakage(cfg: RealWorldEvalConfig, eval_sha1: set[str], eval_ids: set[str]) -> dict:
    """No VizWiz eval image may appear in the Phase 2 training manifest."""
    manifest_path = None
    for cand in (cfg.model_run_dir.parent.parent / "data" / "manifests").glob("manifest_*.parquet"):
        manifest_path = cand
        break
    if manifest_path is None:
        return {"checked": False, "reason": "phase 2 manifest not found"}

    train = pd.read_parquet(manifest_path)
    train_sha1 = set(train["sha1"])
    train_src = set(train["source_id"])
    overlap_sha1 = eval_sha1 & train_sha1
    # source_id overlap is structurally impossible (bsds500/* vs VizWiz_*) but checked.
    overlap_ids = {i for i in eval_ids if f"vizwiz/{i}" in train_src or i in train_src}
    result = {
        "checked": True,
        "phase2_manifest": manifest_path.name,
        "train_images": len(train),
        "sha1_overlap": len(overlap_sha1),
        "id_overlap": len(overlap_ids),
    }
    if overlap_sha1 or overlap_ids:
        raise AssertionError(f"LEAKAGE: eval images found in training: {result}")
    return result


def _extract(sample: RealSample, work_long_edge: int) -> np.ndarray | None:
    if sample.bgr is None:
        return None
    try:
        feats = extract_features(sample.bgr, work_long_edge=work_long_edge)
    except Exception:  # noqa: BLE001
        sample.notes.append("feature extraction raised")
        sample.load_status = "feature_error"
        return None
    return np.array([feats[n] for n in FEATURE_NAMES], dtype=np.float64)


def run_real_world_eval(cfg: RealWorldEvalConfig, *, verbose: bool = True) -> Path:
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    bundle = _load_model_bundle(cfg)
    ctx = _phase2_context(cfg)
    work_long_edge = ctx["work_long_edge"]

    if verbose:
        print("[real-eval] preparing samples (partial download)...")
    samples = prepare_samples(cfg, verbose=verbose)

    # Stream: load one image, extract its features, then drop the pixel buffer.
    # Holding 2500 decoded VizWiz frames at once exhausts memory.
    feats, kept = [], []
    for i, s in enumerate(samples):
        load_image_record(s)
        if s.bgr is not None:
            vec = _extract(s, work_long_edge)
            if vec is not None:
                feats.append(vec)
                kept.append(s)
        s.bgr = None  # release pixels; failure examples are re-read from disk
        if verbose and (i + 1) % 250 == 0:
            print(f"[real-eval] processed {i + 1}/{len(samples)}", flush=True)

    if verbose:
        print(f"[real-eval] {len(kept)}/{len(samples)} images usable for evaluation")
    x = np.vstack(feats)

    # --- predict with the FROZEN model + FROZEN thresholds ---------------------
    scores = np.column_stack(
        [bundle["models"][issue].predict_proba(x)[:, 1] for issue in ISSUE_LABELS]
    )
    thr = np.array([bundle["thresholds"][issue] for issue in ISSUE_LABELS])
    preds = (scores >= thr).astype(int)
    score_by_label = dict(zip(ISSUE_LABELS, scores.T, strict=True))
    pred_by_label = dict(zip(ISSUE_LABELS, preds.T, strict=True))

    eval_labels = evaluable_labels()
    eval_idx = [ISSUE_LABELS.index(lbl) for lbl in eval_labels]

    # --- leakage check -------------------------------------------------------
    leakage = _assert_no_leakage(cfg, {s.sha1 for s in kept if s.sha1}, {s.image_id for s in kept})

    # --- metrics at each vote threshold ------------------------------------
    metrics_by_threshold: dict[str, dict] = {}
    for vote_min in cfg.sensitivity_vote_thresholds:
        y_true = np.array(
            [
                [binarize(s.annotation.flaw_votes, vote_min)[lbl] for lbl in eval_labels]
                for s in kept
            ]
        )
        y_pred = preds[:, eval_idx]
        y_score = scores[:, eval_idx]
        metrics_by_threshold[str(vote_min)] = multilabel_report(
            y_true, y_pred, eval_labels, y_score
        )

    primary = metrics_by_threshold[str(cfg.positive_vote_min)]

    # --- assemble reports --------------------------------------------------
    _write_label_mapping(cfg)
    _write_dataset_report(cfg, samples, kept)
    synthetic_vs_real = _write_synthetic_vs_real(cfg, ctx, primary)
    _write_failure_analysis(cfg, kept, pred_by_label, score_by_label, x, eval_labels)
    _write_domain_shift(cfg, ctx, kept, x, eval_labels)

    experiment = {
        "version": cfg.version,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": cfg.seed,
        "package_version": __version__,
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "model": {
            "run_dir": str(cfg.model_run_dir.name),
            "feature_version": bundle["feature_version"],
            "n_features": len(FEATURE_NAMES),
            "threshold_strategy": cfg.threshold_strategy,
            "thresholds_used": {k: round(float(v), 3) for k, v in bundle["thresholds"].items()},
            "trained_on": "synthetic (BSDS500 + degradations), phase2-baseline-v1",
            "retrained_on_real_data": False,
            "thresholds_tuned_on_real_data": False,
        },
        "dataset": {
            "name": "VizWiz-QualityIssues",
            "split": cfg.eval_split,
            "requested_subset": cfg.subset_size,
            "images_evaluated": len(kept),
            "annotation_file": cfg.annotation_file.name,
        },
        "leakage_check": leakage,
        "evaluable_labels": eval_labels,
        "unsupported_labels": [lbl for lbl in ISSUE_LABELS if lbl not in eval_labels],
        "primary_vote_threshold": cfg.positive_vote_min,
        "headline": {
            "real_macro_f1": primary["macro_f1"],
            "real_micro_f1": primary["micro_f1"],
            "synthetic_macro_f1_evaluable": synthetic_vs_real[
                "synthetic_macro_f1_evaluable_labels"
            ],
        },
        "artifacts": {
            "label_mapping": "label_mapping.json / label_mapping.md",
            "dataset_report": "dataset_report.json",
            "evaluation_metrics": "evaluation_metrics.json",
            "synthetic_vs_real": "synthetic_vs_real.json / synthetic_vs_real.md",
            "failure_analysis": "failure_analysis.json",
            "domain_shift": "domain_shift.json",
        },
    }
    (cfg.reports_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=2, default=str), encoding="utf-8"
    )
    (cfg.reports_dir / "evaluation_metrics.json").write_text(
        json.dumps(
            {
                "primary_vote_threshold": cfg.positive_vote_min,
                "by_vote_threshold": metrics_by_threshold,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if verbose:
        print(
            f"[real-eval] real macro-F1 (>= {cfg.positive_vote_min} votes) = {primary['macro_f1']}"
        )
        print(f"[real-eval] reports in {cfg.reports_dir}")
    return cfg.reports_dir / "experiment.json"


# --------------------------------------------------------------------------- #
# report writers
# --------------------------------------------------------------------------- #
def _write_label_mapping(cfg: RealWorldEvalConfig) -> None:
    table = mapping_table()
    (cfg.reports_dir / "label_mapping.json").write_text(
        json.dumps(
            {
                "vizwiz_codes": VIZWIZ_CODE_MEANING,
                "mappings": table,
                "evaluable_vyra_labels": evaluable_labels(),
                "primary_vote_threshold": POSITIVE_VOTE_MIN,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = [
        "# VizWiz -> VYRA label mapping",
        "",
        "Categories: **A** directly mappable · **B** partially · **C** not mappable · "
        "**D** auxiliary only.",
        "",
        "| VizWiz | meaning | VYRA label | category | confidence | reasoning |",
        "|---|---|---|---|---|---|",
    ]
    for m in table:
        code = m["vizwiz_code"] or "—"
        meaning = VIZWIZ_CODE_MEANING.get(m["vizwiz_code"], "—")
        lines.append(
            f"| {code} | {meaning} | {m['vyra_label'] or '—'} | {m['category']} | "
            f"{m['confidence']} | {m['reasoning']} |"
        )
    lines += [
        "",
        f"**Evaluable VYRA labels:** {', '.join(evaluable_labels())}.",
        "",
        "**Not supported by VizWiz** (never scored, no fake negatives): `noise`, `corruption`.",
    ]
    (cfg.reports_dir / "label_mapping.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_dataset_report(
    cfg: RealWorldEvalConfig, samples: list[RealSample], kept: list[RealSample]
) -> dict:
    status_counts: dict[str, int] = {}
    for s in samples:
        status_counts[s.load_status] = status_counts.get(s.load_status, 0) + 1

    widths = np.array([s.width for s in kept])
    heights = np.array([s.height for s in kept])

    vote_hist = {code: {str(v): 0 for v in range(6)} for code in VIZWIZ_FLAW_CODES}
    unrec_hist = {str(v): 0 for v in range(6)}
    for s in kept:
        for code in VIZWIZ_FLAW_CODES:
            vote_hist[code][str(s.annotation.votes(code))] += 1
        unrec_hist[str(s.annotation.unrecognizable_votes)] += 1

    code_by_label = vyra_to_vizwiz_code()
    positives = {}
    for lbl, code in code_by_label.items():
        positives[lbl] = {
            f">= {t} votes": int(sum(s.annotation.votes(code) >= t for s in kept))
            for t in (1, 2, 3, 4, 5)
        }

    report = {
        "requested_subset": cfg.subset_size,
        "sampled": len(samples),
        "usable_for_eval": len(kept),
        "load_status_counts": status_counts,
        "image_dims": {
            "width": {
                "min": int(widths.min()),
                "max": int(widths.max()),
                "median": int(np.median(widths)),
            },
            "height": {
                "min": int(heights.min()),
                "max": int(heights.max()),
                "median": int(np.median(heights)),
            },
            "grayscale_content": int(sum(s.load_status == "grayscale_ok" for s in kept)),
        },
        "vizwiz_vote_histograms": vote_hist,
        "unrecognizable_vote_histogram": unrec_hist,
        "mapped_label_positive_counts": positives,
        "mapped_label_co_occurrence_at_3": _cooccurrence_at(kept, 3),
        "notes_sample": [
            {"image": s.image_id, "status": s.load_status, "notes": s.notes}
            for s in samples
            if s.notes
        ][:25],
    }
    (cfg.reports_dir / "dataset_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _cooccurrence_at(kept: list[RealSample], vote_min: int) -> dict:
    labels = evaluable_labels()
    code_by_label = vyra_to_vizwiz_code()
    mat = {a: {b: 0 for b in labels} for a in labels}
    for s in kept:
        active = [lbl for lbl in labels if s.annotation.votes(code_by_label[lbl]) >= vote_min]
        for a in active:
            for b in active:
                mat[a][b] += 1
    return mat


def _write_synthetic_vs_real(cfg: RealWorldEvalConfig, ctx: dict, real_primary: dict) -> dict:
    syn = ctx["metrics"]["issue_classification"]["test"]["per_class"]
    eval_labels = evaluable_labels()

    rows = []
    for lbl in eval_labels:
        s = syn.get(lbl, {})
        r = real_primary["per_class"].get(lbl, {})
        rows.append(
            {
                "label": lbl,
                "synthetic_f1": s.get("f1"),
                "synthetic_precision": s.get("precision"),
                "synthetic_recall": s.get("recall"),
                "synthetic_pr_auc": s.get("pr_auc"),
                "real_f1": r.get("f1"),
                "real_precision": r.get("precision"),
                "real_recall": r.get("recall"),
                "real_pr_auc": r.get("pr_auc"),
                "real_support": r.get("support"),
                "f1_drop": round((s.get("f1", 0) or 0) - (r.get("f1", 0) or 0), 3),
            }
        )

    syn_macro_evaluable = round(
        float(np.mean([syn[lbl]["f1"] for lbl in eval_labels if lbl in syn])), 4
    )
    out = {
        "note": (
            "Synthetic = Phase 2 held-out synthetic test split. Real = VizWiz "
            f"{cfg.eval_split} subset, primary vote threshold {cfg.positive_vote_min}. "
            "Macro-F1 comparison restricted to the 4 evaluable labels."
        ),
        "synthetic_macro_f1_all6": ctx["metrics"]["issue_classification"]["test"]["macro_f1"],
        "synthetic_macro_f1_evaluable_labels": syn_macro_evaluable,
        "real_macro_f1_evaluable_labels": real_primary["macro_f1"],
        "per_label": rows,
    }
    (cfg.reports_dir / "synthetic_vs_real.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    md = [
        "# Synthetic vs real-world transfer",
        "",
        out["note"],
        "",
        "| label | syn F1 | real F1 | Δ | syn P/R | real P/R | real support |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['label']} | {r['synthetic_f1']} | {r['real_f1']} | {r['f1_drop']} | "
            f"{r['synthetic_precision']}/{r['synthetic_recall']} | "
            f"{r['real_precision']}/{r['real_recall']} | {r['real_support']} |"
        )
    md += [
        "",
        f"Macro-F1 (4 evaluable labels): synthetic **{syn_macro_evaluable}** -> "
        f"real **{real_primary['macro_f1']}**.",
    ]
    (cfg.reports_dir / "synthetic_vs_real.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return out


def _write_failure_analysis(
    cfg: RealWorldEvalConfig,
    kept: list[RealSample],
    pred_by_label: dict,
    score_by_label: dict,
    x: np.ndarray,
    eval_labels: list[str],
) -> dict:
    cfg.failure_examples_dir.mkdir(parents=True, exist_ok=True)
    code_by_label = vyra_to_vizwiz_code()
    vote_min = cfg.positive_vote_min
    out: dict = {}

    for lbl in eval_labels:
        code = code_by_label[lbl]
        preds = pred_by_label[lbl]
        scores = score_by_label[lbl]
        groups = {"false_positive": [], "false_negative": []}
        for i, s in enumerate(kept):
            truth = int(s.annotation.votes(code) >= vote_min)
            pred = int(preds[i])
            if pred == truth:
                continue
            group = "false_positive" if pred == 1 and truth == 0 else "false_negative"
            groups[group].append(
                {
                    "image": s.image_id,
                    "score": round(float(scores[i]), 3),
                    "vizwiz_votes": dict(s.annotation.flaw_votes),
                    "unrecognizable_votes": s.annotation.unrecognizable_votes,
                    "dims": [s.width, s.height],
                }
            )
        # sort by "confidence of the mistake"
        groups["false_positive"].sort(key=lambda d: -d["score"])
        groups["false_negative"].sort(key=lambda d: d["score"])

        # save a few representative images per group (re-read from the cache;
        # pixel buffers were released after feature extraction)
        by_id = {s.image_id: s for s in kept}
        saved = {}
        for group, items in groups.items():
            saved[group] = []
            for item in items[: cfg.max_failure_examples_per_group]:
                src = by_id[item["image"]]
                img = cv2.imread(str(src.local_path), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                # downscale big frames so committed examples stay small
                h, w = img.shape[:2]
                if max(h, w) > 640:
                    sc = 640 / max(h, w)
                    img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
                dest = cfg.failure_examples_dir / f"{lbl}__{group}__{item['image']}"
                cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
                saved[group].append(dest.name)

        out[lbl] = {
            "counts": {k: len(v) for k, v in groups.items()},
            "examples": {k: v[:15] for k, v in groups.items()},
            "saved_images": saved,
        }

    (cfg.reports_dir / "failure_analysis.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    return out


def _write_domain_shift(
    cfg: RealWorldEvalConfig, ctx: dict, kept: list[RealSample], x: np.ndarray, eval_labels: list
) -> dict:
    """Compare feature distributions: synthetic vs real, per condition."""
    # Phase 2 feature table (has features + labels + split + is_clean).
    ft_dir = cfg.model_run_dir.parents[1] / "data" / "processed"
    ft = next(ft_dir.glob("features_phase2-baseline-v1_*.parquet"), None)
    syn_stats = {}
    if ft is not None:
        syn = pd.read_parquet(ft)
        conditions = {
            "synthetic_clean": syn[syn["is_clean"]],
            "synthetic_blur": syn[syn["label_blur"] == 1],
            "synthetic_overexposure": syn[syn["label_overexposure"] == 1],
            "synthetic_underexposure": syn[syn["label_underexposure"] == 1],
        }
        for name, sub in conditions.items():
            syn_stats[name] = {
                f: [round(float(sub[f].mean()), 4), round(float(sub[f].std()), 4)]
                for f in FEATURE_NAMES
            }

    real_clean_mask = np.array(
        [
            all(s.annotation.votes(c) <= 1 for c in ("BLR", "BRT", "DRK", "OBS"))
            and s.annotation.votes("NON") >= 3
            for s in kept
        ]
    )
    real_conditions = {
        "real_clean_ish": real_clean_mask,
        "real_blur": np.array([s.annotation.votes("BLR") >= 3 for s in kept]),
        "real_overexposure": np.array([s.annotation.votes("BRT") >= 3 for s in kept]),
        "real_underexposure": np.array([s.annotation.votes("DRK") >= 3 for s in kept]),
    }
    real_stats = {}
    for name, mask in real_conditions.items():
        if mask.sum() == 0:
            continue
        sub = x[mask]
        real_stats[name] = {
            "n": int(mask.sum()),
            "features": {
                f: [round(float(sub[:, i].mean()), 4), round(float(sub[:, i].std()), 4)]
                for i, f in enumerate(FEATURE_NAMES)
            },
        }

    out = {
        "note": (
            "Per-feature [mean, std]. Compare a synthetic condition with its real "
            "counterpart to see distribution shift. Large gaps in the features a "
            "label depends on explain transfer failure."
        ),
        "synthetic": syn_stats,
        "real": real_stats,
    }
    (cfg.reports_dir / "domain_shift.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
