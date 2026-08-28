"""Typed loader for ``configs/real_world_eval.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ML_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ML_ROOT / "configs" / "real_world_eval.yaml"


@dataclass(frozen=True)
class RealWorldEvalConfig:
    version: str
    seed: int
    model_run_dir: Path
    expected_feature_version: str
    threshold_strategy: str
    annotations_dir: Path
    eval_split: str
    images_zip_url: str
    image_cache_dir: Path
    subset_size: int | None
    download_workers: int
    positive_vote_min: int
    sensitivity_vote_thresholds: tuple[int, ...]
    reports_dir: Path
    failure_examples_dir: Path
    max_failure_examples_per_group: int
    source_path: Path

    @property
    def model_path(self) -> Path:
        return self.model_run_dir / "model.joblib"

    @property
    def annotation_file(self) -> Path:
        return self.annotations_dir / f"{self.eval_split}.json"


def _abs(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (ML_ROOT / p).resolve()


def load_real_world_config(path: str | Path | None = None) -> RealWorldEvalConfig:
    cfg_path = Path(path) if path else DEFAULT_PATH
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    allowed = {"version", "seed", "model", "vizwiz", "evaluation", "paths"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in real_world_eval config: {sorted(unknown)}")

    return RealWorldEvalConfig(
        version=raw["version"],
        seed=int(raw["seed"]),
        model_run_dir=_abs(raw["model"]["run_dir"]),
        expected_feature_version=raw["model"]["expected_feature_version"],
        threshold_strategy=raw["model"]["threshold_strategy"],
        annotations_dir=_abs(raw["vizwiz"]["annotations_dir"]),
        eval_split=raw["vizwiz"]["eval_split"],
        images_zip_url=raw["vizwiz"]["images_zip_url"],
        image_cache_dir=_abs(raw["vizwiz"]["image_cache_dir"]),
        subset_size=raw["vizwiz"]["subset_size"],
        download_workers=int(raw["vizwiz"]["download_workers"]),
        positive_vote_min=int(raw["evaluation"]["positive_vote_min"]),
        sensitivity_vote_thresholds=tuple(raw["evaluation"]["sensitivity_vote_thresholds"]),
        reports_dir=_abs(raw["paths"]["reports_subdir"]),
        failure_examples_dir=_abs(raw["paths"]["failure_examples_subdir"]),
        max_failure_examples_per_group=int(raw["paths"]["max_failure_examples_per_group"]),
        source_path=cfg_path,
    )
