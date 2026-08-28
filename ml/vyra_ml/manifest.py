"""The dataset manifest: one row per generated/ingested image.

Written as both Parquet (primary; typed, compact, fast to load for training) and
JSONL (human-diffable, easy to inspect with plain tools). The two are always
written together and carry identical rows.

Schema (flat columns, plus nested detail in ``degradations_json`` / ``labels_json``):

    sample_id          globally unique id for this image
    source_id          id of the ORIGINAL image it derives from
    source_dataset     where the original came from (e.g. "bsds500")
    split              train | val | test  (decided at original level)
    image_path         path relative to the processed data dir
    is_clean           bool: no degradation applied
    degradations_json  JSON list of {name, severity, params}
    n_degradations     int
    label_<issue>      0/1 for each of the six canonical issues
    labels_json        JSON dict mirror of the label_ columns
    severity_<issue>   max applied severity for that issue (0 if absent)
    max_severity       max severity across all applied degradations (0 if clean)
    quality_score      PROVISIONAL 0-100 target (see vyra_ml/labels.py)
    quality_label      coarse band of quality_score
    width, height      stored image dimensions
    orig_width, orig_height   dimensions of the source original
    file_bytes         size on disk
    sha1               hash of the stored image bytes (integrity / dedup)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from vyra_ml import ISSUE_LABELS

MANIFEST_SCHEMA_VERSION = "manifest-v1"

_LABEL_COLS = [f"label_{name}" for name in ISSUE_LABELS]
_SEVERITY_COLS = [f"severity_{name}" for name in ISSUE_LABELS]

COLUMNS = [
    "sample_id",
    "source_id",
    "source_dataset",
    "split",
    "image_path",
    "is_clean",
    "degradations_json",
    "n_degradations",
    *_LABEL_COLS,
    "labels_json",
    *_SEVERITY_COLS,
    "max_severity",
    "quality_score",
    "quality_label",
    "width",
    "height",
    "orig_width",
    "orig_height",
    "file_bytes",
    "sha1",
]


@dataclass
class ManifestRow:
    sample_id: str
    source_id: str
    source_dataset: str
    split: str
    image_path: str
    is_clean: bool
    degradations: list[dict[str, Any]]
    labels: dict[str, int]
    severity_by_issue: dict[str, int]
    max_severity: int
    quality_score: float
    quality_label: str
    width: int
    height: int
    orig_width: int
    orig_height: int
    file_bytes: int
    sha1: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "sample_id": self.sample_id,
            "source_id": self.source_id,
            "source_dataset": self.source_dataset,
            "split": self.split,
            "image_path": self.image_path,
            "is_clean": bool(self.is_clean),
            "degradations_json": json.dumps(self.degradations, sort_keys=True),
            "n_degradations": len(self.degradations),
            "labels_json": json.dumps(self.labels, sort_keys=True),
            "max_severity": int(self.max_severity),
            "quality_score": float(self.quality_score),
            "quality_label": self.quality_label,
            "width": int(self.width),
            "height": int(self.height),
            "orig_width": int(self.orig_width),
            "orig_height": int(self.orig_height),
            "file_bytes": int(self.file_bytes),
            "sha1": self.sha1,
        }
        for name in ISSUE_LABELS:
            rec[f"label_{name}"] = int(self.labels[name])
            rec[f"severity_{name}"] = int(self.severity_by_issue[name])
        return rec


def rows_to_frame(rows: list[ManifestRow]) -> pd.DataFrame:
    frame = pd.DataFrame([r.to_record() for r in rows], columns=COLUMNS)
    return frame


def write_manifest(frame: pd.DataFrame, out_dir: Path, stem: str = "manifest") -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"{stem}.parquet"
    jsonl_path = out_dir / f"{stem}.jsonl"
    frame.to_parquet(parquet_path, index=False)
    frame.to_json(jsonl_path, orient="records", lines=True)
    return {"parquet": parquet_path, "jsonl": jsonl_path}


def read_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".jsonl", ".json"}:
        return pd.read_json(path, orient="records", lines=True)
    raise ValueError(f"Unsupported manifest format: {path.suffix}")


def label_columns() -> list[str]:
    return list(_LABEL_COLS)
