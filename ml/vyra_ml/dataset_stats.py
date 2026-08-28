"""Dataset statistics report + a few diagnostic plots.

Purpose is understanding, not decoration: split sizes, issue and severity
distributions, co-occurrence, source spread, image dimensions, class imbalance.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from vyra_ml import ISSUE_LABELS  # noqa: E402
from vyra_ml.manifest import read_manifest  # noqa: E402

_LABEL_COLS = [f"label_{n}" for n in ISSUE_LABELS]


def _imbalance(frame: pd.DataFrame) -> dict:
    out = {}
    n = len(frame)
    for name in ISSUE_LABELS:
        pos = int(frame[f"label_{name}"].sum())
        out[name] = {
            "positives": pos,
            "negatives": n - pos,
            "positive_rate": round(pos / n, 4),
            "imbalance_ratio": round((n - pos) / max(1, pos), 2),
        }
    return out


def _cooccurrence(frame: pd.DataFrame) -> list[list[int]]:
    m = frame[_LABEL_COLS].to_numpy()
    return (m.T @ m).astype(int).tolist()


def build_dataset_report(manifest_path: str | Path, out_dir: str | Path) -> Path:
    frame = read_manifest(manifest_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_issues_per_sample = frame[_LABEL_COLS].sum(axis=1)
    severity_dist = {
        name: frame.loc[frame[f"label_{name}"] == 1, f"severity_{name}"]
        .value_counts()
        .sort_index()
        .to_dict()
        for name in ISSUE_LABELS
    }

    report = {
        "n_samples": int(len(frame)),
        "n_originals": int(frame["source_id"].nunique()),
        "clean_samples": int(frame["is_clean"].sum()),
        "split_counts_samples": frame["split"].value_counts().to_dict(),
        "split_counts_originals": frame.groupby("split")["source_id"].nunique().to_dict(),
        "source_dataset_counts": frame["source_dataset"].value_counts().to_dict(),
        "issues_per_sample": n_issues_per_sample.value_counts().sort_index().to_dict(),
        "class_imbalance": _imbalance(frame),
        "severity_distribution": severity_dist,
        "label_cooccurrence": {
            "labels": list(ISSUE_LABELS),
            "matrix": _cooccurrence(frame),
        },
        "image_dims": {
            "width": {
                "min": int(frame.width.min()),
                "max": int(frame.width.max()),
                "median": int(frame.width.median()),
            },
            "height": {
                "min": int(frame.height.min()),
                "max": int(frame.height.max()),
                "median": int(frame.height.median()),
            },
        },
        "orig_dims": {
            "width_median": int(frame.orig_width.median()),
            "height_median": int(frame.orig_height.median()),
        },
        "quality_score_provisional": {
            "min": float(frame.quality_score.min()),
            "max": float(frame.quality_score.max()),
            "mean": round(float(frame.quality_score.mean()), 2),
            "by_label_band": frame["quality_label"].value_counts().to_dict(),
        },
        "invalid_or_missing": int(frame["file_bytes"].le(0).sum()),
        "per_split_positive_rate": {
            split: {name: round(float(g[f"label_{name}"].mean()), 3) for name in ISSUE_LABELS}
            for split, g in frame.groupby("split")
        },
    }
    out_path = out_dir / "dataset_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    _plots(frame, out_dir)
    return out_path


def _plots(frame: pd.DataFrame, out_dir: Path) -> None:
    pos_rates = [frame[f"label_{n}"].mean() for n in ISSUE_LABELS]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(ISSUE_LABELS, pos_rates, color="#3b6ea5")
    ax.set_title("Issue positive rate")
    ax.set_ylabel("fraction of samples")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "issue_positive_rate.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    counts = frame[_LABEL_COLS].sum(axis=1).value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.to_numpy(), color="#6a8caf")
    ax.set_title("Number of simultaneous issues per sample")
    ax.set_xlabel("issues")
    ax.set_ylabel("samples")
    fig.tight_layout()
    fig.savefig(out_dir / "issues_per_sample.png", dpi=110)
    plt.close(fig)

    m = frame[_LABEL_COLS].to_numpy()
    co = (m.T @ m).astype(float)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(co, cmap="Blues")
    ax.set_xticks(range(len(ISSUE_LABELS)), ISSUE_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(ISSUE_LABELS)), ISSUE_LABELS)
    for i in range(len(ISSUE_LABELS)):
        for j in range(len(ISSUE_LABELS)):
            ax.text(j, i, int(co[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title("Label co-occurrence")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "label_cooccurrence.png", dpi=110)
    plt.close(fig)
