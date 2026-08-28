"""Typed access to ``configs/experiment.yaml``.

The config is loaded once and passed explicitly down the pipeline; nothing reads
it from a global. Unknown keys are rejected so a typo cannot silently disable a
step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ML_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ML_ROOT / "configs" / "experiment.yaml"


@dataclass(frozen=True)
class Paths:
    data_root: Path
    raw_subdir: str
    processed_subdir: str
    manifests_subdir: str
    reports_subdir: str
    runs_subdir: str

    def _resolve(self, root: Path, sub: str) -> Path:
        return (root / self.data_root / sub).resolve()

    def raw(self, root: Path) -> Path:
        return self._resolve(root, self.raw_subdir)

    def processed(self, root: Path) -> Path:
        return self._resolve(root, self.processed_subdir)

    def manifests(self, root: Path) -> Path:
        return self._resolve(root, self.manifests_subdir)

    def reports(self, root: Path) -> Path:
        return self._resolve(root, self.reports_subdir)

    def runs(self, root: Path) -> Path:
        return self._resolve(root, self.runs_subdir)


@dataclass(frozen=True)
class SplitRatios:
    train: float
    val: float
    test: float

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass(frozen=True)
class DatasetConfig:
    source: str
    max_originals: int | None
    target_long_edge: int
    splits: SplitRatios


@dataclass(frozen=True)
class DegradationConfig:
    clean_per_original: int
    single_per_original: int
    multi_per_original: int
    multi_min: int
    multi_max: int
    severity_levels: list[int]
    enabled: list[str]
    # Phase 3B: add light sensor read-noise AFTER blur, on blur-containing samples
    # only, to model real image formation (lens blur then sensor noise). Off by
    # default so phase2-baseline-v1 rebuilds identically.
    post_blur_sensor_noise: bool = False


@dataclass(frozen=True)
class FeatureConfig:
    version: str
    work_long_edge: int


@dataclass(frozen=True)
class BaselineConfig:
    model: str
    random_forest: dict[str, Any]
    hist_gradient_boosting: dict[str, Any]
    fit_quality_regressor: bool


@dataclass(frozen=True)
class ExperimentConfig:
    version: str
    seed: int
    paths: Paths
    dataset: DatasetConfig
    degradation: DegradationConfig
    features: FeatureConfig
    baseline: BaselineConfig
    source_path: Path = field(default=DEFAULT_CONFIG_PATH)

    @property
    def repo_root(self) -> Path:
        # ml/ sits one level under the repository root.
        return REPO_ML_ROOT.parent

    def data_dir(self, which: str) -> Path:
        return getattr(self.paths, which)(REPO_ML_ROOT)


def _require_keys(name: str, data: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in '{name}' config section: {sorted(unknown)}")


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    _require_keys(
        "root",
        raw,
        {"version", "seed", "paths", "dataset", "degradation", "features", "baseline"},
    )

    paths = Paths(
        data_root=Path(raw["paths"]["data_root"]),
        raw_subdir=raw["paths"]["raw_subdir"],
        processed_subdir=raw["paths"]["processed_subdir"],
        manifests_subdir=raw["paths"]["manifests_subdir"],
        reports_subdir=raw["paths"]["reports_subdir"],
        runs_subdir=raw["paths"]["runs_subdir"],
    )
    dataset = DatasetConfig(
        source=raw["dataset"]["source"],
        max_originals=raw["dataset"]["max_originals"],
        target_long_edge=int(raw["dataset"]["target_long_edge"]),
        splits=SplitRatios(**raw["dataset"]["splits"]),
    )
    degradation = DegradationConfig(**raw["degradation"])
    features = FeatureConfig(**raw["features"])
    baseline = BaselineConfig(
        model=raw["baseline"]["model"],
        random_forest=raw["baseline"]["random_forest"],
        hist_gradient_boosting=raw["baseline"]["hist_gradient_boosting"],
        fit_quality_regressor=bool(raw["baseline"]["fit_quality_regressor"]),
    )

    return ExperimentConfig(
        version=raw["version"],
        seed=int(raw["seed"]),
        paths=paths,
        dataset=dataset,
        degradation=degradation,
        features=features,
        baseline=baseline,
        source_path=config_path,
    )
