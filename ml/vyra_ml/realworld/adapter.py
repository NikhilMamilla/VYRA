"""Turns VizWiz annotations + the remote image zip into loadable eval samples.

Deterministic: the subset is a seeded uniform sample of the official split, so
re-running reproduces the same evaluation set.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from vyra_ml.realworld.config import RealWorldEvalConfig
from vyra_ml.realworld.fetch import fetch_from_remote_zip
from vyra_ml.realworld.vizwiz import VizWizAnnotation, parse_annotations


@dataclass
class RealSample:
    image_id: str
    local_path: Path
    annotation: VizWizAnnotation
    # Populated by load_image_record:
    bgr: np.ndarray | None = None
    load_status: str = "pending"  # "ok" | "unreadable" | "too_small" | "grayscale_ok"
    width: int = 0
    height: int = 0
    channels: int = 0
    sha1: str = ""
    notes: list[str] = field(default_factory=list)


_MIN_EDGE = 32


def select_subset(
    annotations: list[VizWizAnnotation], subset_size: int | None, seed: int
) -> list[VizWizAnnotation]:
    ordered = sorted(annotations, key=lambda a: a.image)
    if subset_size is None or subset_size >= len(ordered):
        return ordered
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ordered), size=subset_size, replace=False)
    return [ordered[i] for i in sorted(idx)]


def prepare_samples(cfg: RealWorldEvalConfig, *, verbose: bool = True) -> list[RealSample]:
    annotations = parse_annotations(cfg.annotation_file)
    if not annotations:
        raise RuntimeError(
            f"No labelled annotations in {cfg.annotation_file}. The 'test' split "
            "ships no labels - use eval_split: val."
        )
    subset = select_subset(annotations, cfg.subset_size, cfg.seed)

    members = [f"{cfg.eval_split}/{a.image}" for a in subset]

    def _progress(done: int, total: int) -> None:
        if verbose:
            print(f"  fetched {done}/{total}", flush=True)

    fetched = fetch_from_remote_zip(
        cfg.images_zip_url,
        members,
        cfg.image_cache_dir,
        workers=cfg.download_workers,
        on_progress=_progress,
    )

    samples: list[RealSample] = []
    for ann in subset:
        member = f"{cfg.eval_split}/{ann.image}"
        local = fetched.get(member)
        if local is None:
            samples.append(
                RealSample(
                    image_id=ann.image,
                    local_path=cfg.image_cache_dir / ann.image,
                    annotation=ann,
                    load_status="download_failed",
                )
            )
            continue
        samples.append(RealSample(image_id=ann.image, local_path=local, annotation=ann))
    return samples


def load_image_record(sample: RealSample) -> RealSample:
    """Read the image, recording any anomaly. Same reader the synthetic pipeline uses."""
    if sample.load_status == "download_failed":
        return sample

    data = sample.local_path.read_bytes()
    sample.sha1 = hashlib.sha1(data).hexdigest()
    bgr = cv2.imread(str(sample.local_path), cv2.IMREAD_COLOR)
    if bgr is None:
        sample.load_status = "unreadable"
        sample.notes.append("cv2.imread returned None")
        return sample

    h, w = bgr.shape[:2]
    sample.height, sample.width, sample.channels = h, w, bgr.shape[2] if bgr.ndim == 3 else 1
    if min(h, w) < _MIN_EDGE:
        sample.load_status = "too_small"
        sample.notes.append(f"min edge {min(h, w)}px < {_MIN_EDGE}px")
        return sample

    # Detect originally-grayscale content (all channels equal) but still usable.
    gray_like = (
        bgr.ndim == 3
        and np.array_equal(bgr[..., 0], bgr[..., 1])
        and np.array_equal(bgr[..., 1], bgr[..., 2])
    )
    sample.bgr = bgr
    sample.load_status = "grayscale_ok" if gray_like else "ok"
    return sample
