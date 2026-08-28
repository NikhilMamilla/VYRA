"""Dataset build orchestrator: ingest -> split -> degrade -> manifest.

Reproducibility contract: given the same source images, the same
``configs/experiment.yaml`` and the same seed, this produces the same images
(stored as JPEG q=97 -- see ``docs/dataset.md``) and the same manifest.

Leakage safety: the split is decided on original ``source_id`` *before* any
degradation is generated (:mod:`vyra_ml.splitting`), and every variant inherits
its original's split. ``assert_no_leakage`` runs on the finished manifest.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from vyra_ml import ISSUE_LABELS, __version__
from vyra_ml.config import ExperimentConfig
from vyra_ml.degradations import APPLICATION_ORDER, get_degradation
from vyra_ml.features import FEATURE_VERSION
from vyra_ml.ingest import get_adapter
from vyra_ml.labels import (
    AppliedDegradation,
    label_vector,
    provisional_quality_score,
    quality_label,
    severity_by_issue,
)
from vyra_ml.manifest import (
    MANIFEST_SCHEMA_VERSION,
    ManifestRow,
    rows_to_frame,
    write_manifest,
)
from vyra_ml.seeding import derive_rng
from vyra_ml.splitting import assert_no_leakage, split_counts, split_originals

_STORAGE_JPEG_QUALITY = 97  # near-lossless; keeps ~3-4k samples to ~150 MB
_CONTRADICTORY = frozenset({frozenset({"underexposure", "overexposure"})})


@dataclass
class SamplePlan:
    sample_id: str
    kind: str  # clean | single | multi
    degradations: list[tuple[str, int]]  # (degradation_name, severity)


@dataclass
class BuildResult:
    manifest_paths: dict[str, Path]
    metadata_path: Path
    n_originals: int
    n_samples: int
    split_counts: dict[str, int]


def _resize_long_edge(image: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = long_edge / max(h, w)
    if abs(scale - 1.0) < 1e-3:
        return image
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(
        image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=interp
    )


def _plan_samples(source_stem: str, cfg: ExperimentConfig) -> list[SamplePlan]:
    d = cfg.degradation
    rng = derive_rng(cfg.seed, "plan", source_stem)
    plans: list[SamplePlan] = []

    for i in range(d.clean_per_original):
        plans.append(SamplePlan(f"{source_stem}__clean{i}", "clean", []))

    for i in range(d.single_per_original):
        name = str(rng.choice(d.enabled))
        severity = int(rng.choice(d.severity_levels))
        plans.append(SamplePlan(f"{source_stem}__single{i}", "single", [(name, severity)]))

    for i in range(d.multi_per_original):
        k = int(rng.integers(d.multi_min, d.multi_max + 1))
        k = min(k, len(d.enabled))
        chosen: list[str] = []
        for name in map(str, rng.permutation(d.enabled)):
            if len(chosen) >= k:
                break
            if any(frozenset({name, c}) in _CONTRADICTORY for c in chosen):
                continue
            chosen.append(name)
        degs = [(name, int(rng.choice(d.severity_levels))) for name in chosen]
        plans.append(SamplePlan(f"{source_stem}__multi{i}", "multi", degs))

    return plans


# Sensor read-noise sigma (0-255 scale) added after blur. Deliberately below the
# `noise` degradation's severity-1 range (3-7) so blur does not become trivially
# separable from clean or from the `noise` class -- this is realism, not a cue.
_SENSOR_NOISE_SIGMA = (1.5, 4.0)


def _add_sensor_noise(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    f = image.astype(np.float32) / 255.0
    sigma = float(rng.uniform(*_SENSOR_NOISE_SIGMA)) / 255.0
    luma = rng.normal(0.0, sigma, size=f.shape[:2])[..., None]
    chroma = rng.normal(0.0, sigma * 0.5, size=f.shape)
    out = np.clip(f + luma + chroma, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8), {"sigma_255": round(sigma * 255, 3)}


def _apply_plan(
    clean_bgr: np.ndarray, plan: SamplePlan, cfg: ExperimentConfig
) -> tuple[np.ndarray, list[AppliedDegradation]]:
    image = clean_bgr
    ordered = sorted(plan.degradations, key=lambda ns: APPLICATION_ORDER.index(ns[0]))
    applied: list[AppliedDegradation] = []
    for name, severity in ordered:
        rng = derive_rng(cfg.seed, "apply", plan.sample_id, name)
        result = get_degradation(name).apply(image, severity, rng)
        image = result.image
        applied.append(AppliedDegradation(name=name, severity=severity, params=result.params))

    if cfg.degradation.post_blur_sensor_noise and any(a.name == "blur" for a in applied):
        rng = derive_rng(cfg.seed, "sensor_noise", plan.sample_id)
        image, sensor_params = _add_sensor_noise(image, rng)
        # Recorded on the blur entry's params; it does NOT raise the `noise` label.
        applied = [
            replace(a, params={**a.params, "post_blur_sensor_noise": sensor_params})
            if a.name == "blur"
            else a
            for a in applied
        ]

    return image, applied


def build_dataset(cfg: ExperimentConfig, *, progress: bool = True) -> BuildResult:
    raw_dir = cfg.data_dir("raw")
    processed_dir = cfg.data_dir("processed")
    raw_dir.mkdir(parents=True, exist_ok=True)

    adapter = get_adapter(cfg.dataset.source, raw_dir)
    adapter.prepare()
    originals = list(adapter.iter_originals(cfg.dataset.max_originals))
    if len(originals) < 3:
        raise RuntimeError(
            f"Only {len(originals)} originals from source {cfg.dataset.source!r}; "
            "the adapter looks broken. A real experiment needs a few hundred "
            "(see docs/dataset.md)."
        )

    split_map = split_originals((o.source_id for o in originals), cfg.dataset.splits, cfg.seed)

    rows: list[ManifestRow] = []
    for idx, original in enumerate(originals):
        if progress and idx % 25 == 0:
            print(f"  [{idx}/{len(originals)}] {original.source_id}", flush=True)

        raw = cv2.imread(str(original.path), cv2.IMREAD_COLOR)
        if raw is None:
            print(f"  WARN: unreadable original skipped: {original.path}", flush=True)
            continue
        clean = _resize_long_edge(raw, cfg.dataset.target_long_edge)
        oh, ow = raw.shape[:2]
        split = split_map[original.source_id]
        source_stem = original.source_id.replace("/", "_")
        out_dir = processed_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)

        for plan in _plan_samples(source_stem, cfg):
            image, applied = _apply_plan(clean, plan, cfg)
            rel_path = f"{split}/{plan.sample_id}.jpg"
            abs_path = processed_dir / rel_path
            cv2.imwrite(str(abs_path), image, [cv2.IMWRITE_JPEG_QUALITY, _STORAGE_JPEG_QUALITY])
            data = abs_path.read_bytes()

            labels = label_vector(applied)
            sev_by_issue = severity_by_issue(applied)
            score = provisional_quality_score(applied)
            rows.append(
                ManifestRow(
                    sample_id=plan.sample_id,
                    source_id=original.source_id,
                    source_dataset=original.source_dataset,
                    split=split,
                    image_path=rel_path,
                    is_clean=(plan.kind == "clean"),
                    degradations=[
                        {"name": a.name, "severity": a.severity, "params": a.params}
                        for a in applied
                    ],
                    labels=labels,
                    severity_by_issue=sev_by_issue,
                    max_severity=max((a.severity for a in applied), default=0),
                    quality_score=score,
                    quality_label=quality_label(score),
                    width=image.shape[1],
                    height=image.shape[0],
                    orig_width=ow,
                    orig_height=oh,
                    file_bytes=len(data),
                    sha1=hashlib.sha1(data).hexdigest(),
                )
            )

    frame = rows_to_frame(rows)
    assert_no_leakage(frame.to_dict("records"))

    manifests_dir = cfg.data_dir("manifests")
    manifest_paths = write_manifest(frame, manifests_dir, stem=f"manifest_{cfg.version}")

    issue_counts = {name: int(frame[f"label_{name}"].sum()) for name in ISSUE_LABELS}
    metadata = {
        "manifest_schema": MANIFEST_SCHEMA_VERSION,
        "package_version": __version__,
        "feature_version": FEATURE_VERSION,
        "config_version": cfg.version,
        "seed": cfg.seed,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "source_dataset": cfg.dataset.source,
        "storage_format": f"jpeg q{_STORAGE_JPEG_QUALITY}",
        "n_originals": len(originals),
        "n_samples": len(rows),
        "split_counts_originals": split_counts(split_map),
        "split_counts_samples": frame["split"].value_counts().to_dict(),
        "issue_positive_counts": issue_counts,
        "clean_samples": int(frame["is_clean"].sum()),
        "degradation_config": vars(cfg.degradation),
    }
    metadata_path = manifests_dir / f"build_metadata_{cfg.version}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    return BuildResult(
        manifest_paths=manifest_paths,
        metadata_path=metadata_path,
        n_originals=len(originals),
        n_samples=len(rows),
        split_counts=split_counts(split_map),
    )
