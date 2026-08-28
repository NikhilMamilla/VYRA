"""Compression-artefact features: 8x8 blockiness.

JPEG quantises in 8x8 blocks, so heavy compression leaves discontinuities on the
8-pixel grid. We compare the mean absolute gradient *across* block boundaries
with the mean gradient *inside* blocks.

**cvfeat-v2 change.** v1 used a plain ratio ``boundary_mean / (interior_mean +
1e-8)``. On a near-flat image (e.g. a heavily blurred frame that is then stored
as JPEG q97 by the dataset builder) the interior gradient underflows toward zero
while the faint stored-JPEG grid keeps the boundary gradient non-zero, so the
ratio exploded -- Phase 3A observed mean 800, std 21700 on synthetic blur
samples. v2 uses the bounded normalised-excess form

    blockiness = (boundary_mean - interior_mean) / (boundary_mean + interior_mean + eps)

which lies in [0, 1): 0 when the grid carries no more gradient than the interior
(clean or cleanly textured), approaching 1 when the block grid dominates what
little structure the image has. It is monotone in the old boundary/interior
ratio, so ordering between images is preserved, but it can never diverge.
"""

from __future__ import annotations

import numpy as np

from vyra_ml.features.common import PreparedImage

_EPS = 1e-6


def _blockiness(diff: np.ndarray, boundary_idx: np.ndarray, interior_idx: np.ndarray, axis: int):
    take = (lambda a: a[:, boundary_idx]) if axis == 1 else (lambda a: a[boundary_idx, :])
    take_in = (lambda a: a[:, interior_idx]) if axis == 1 else (lambda a: a[interior_idx, :])
    boundary = float(take(diff).mean())
    interior = float(take_in(diff).mean())
    return (boundary - interior) / (boundary + interior + _EPS)


def compute(img: PreparedImage) -> dict[str, float]:
    y = img.gray_f
    dh = np.abs(np.diff(y, axis=1))
    dv = np.abs(np.diff(y, axis=0))

    boundary_cols = np.arange(7, dh.shape[1], 8)
    boundary_rows = np.arange(7, dv.shape[0], 8)
    if len(boundary_cols) == 0 or len(boundary_rows) == 0:
        return {
            "compress_blockiness": 0.0,
            "compress_blockiness_h": 0.0,
            "compress_blockiness_v": 0.0,
        }

    interior_cols = np.setdiff1d(np.arange(dh.shape[1]), boundary_cols)
    interior_rows = np.setdiff1d(np.arange(dv.shape[0]), boundary_rows)

    block_h = _blockiness(dh, boundary_cols, interior_cols, axis=1)
    block_v = _blockiness(dv, boundary_rows, interior_rows, axis=0)

    return {
        "compress_blockiness": 0.5 * (block_h + block_v),
        "compress_blockiness_h": block_h,
        "compress_blockiness_v": block_v,
    }


def legacy_ratio_blockiness(img: PreparedImage) -> dict[str, float]:
    """The cvfeat-v1 (unbounded ratio) blockiness, kept only for the Phase 3B
    ablation that isolates the effect of the feature fix. Not part of any feature
    version; do not use in new models."""
    y = img.gray_f
    dh = np.abs(np.diff(y, axis=1))
    dv = np.abs(np.diff(y, axis=0))
    bc = np.arange(7, dh.shape[1], 8)
    br = np.arange(7, dv.shape[0], 8)
    if len(bc) == 0 or len(br) == 0:
        return {"v1_blockiness": 0.0, "v1_blockiness_h": 0.0, "v1_blockiness_v": 0.0}
    ic = np.setdiff1d(np.arange(dh.shape[1]), bc)
    ir = np.setdiff1d(np.arange(dv.shape[0]), br)
    h = float(dh[:, bc].mean()) / (float(dh[:, ic].mean()) + 1e-8)
    v = float(dv[br, :].mean()) / (float(dv[ir, :].mean()) + 1e-8)
    return {"v1_blockiness": 0.5 * (h + v), "v1_blockiness_h": h, "v1_blockiness_v": v}
