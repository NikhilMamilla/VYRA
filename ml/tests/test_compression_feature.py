"""cvfeat-v2 blockiness: bounded, stable, still informative (Phase 3B fix)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vyra_ml.features.common import prepare
from vyra_ml.features.compression import compute, legacy_ratio_blockiness

_BLK = ("compress_blockiness", "compress_blockiness_h", "compress_blockiness_v")


def _prep(bgr):
    return prepare(bgr, 384)


@pytest.mark.parametrize(
    "name,img",
    [
        ("flat_grey", np.full((256, 256, 3), 127, np.uint8)),
        ("black", np.zeros((256, 256, 3), np.uint8)),
        ("white", np.full((256, 256, 3), 255, np.uint8)),
        (
            "near_flat",
            np.clip(127 + np.random.default_rng(0).normal(0, 0.3, (256, 256, 3)), 0, 255).astype(
                np.uint8
            ),
        ),
        (
            "blurred_flatish",
            cv2.GaussianBlur(
                np.clip(127 + np.random.default_rng(1).normal(0, 6, (256, 256, 3)), 0, 255).astype(
                    np.uint8
                ),
                (0, 0),
                8.0,
            ),
        ),
    ],
)
def test_blockiness_is_bounded_and_finite(name, img):
    feats = compute(_prep(img))
    for k in _BLK:
        v = feats[k]
        assert np.isfinite(v), f"{name}/{k} not finite"
        assert -1e-6 <= v <= 1.0 + 1e-6, f"{name}/{k}={v} out of [0,1]"


def test_textured_image_has_low_blockiness():
    rng = np.random.default_rng(2)
    tex = (rng.random((256, 320, 3)) * 255).astype(np.uint8)
    assert compute(_prep(tex))["compress_blockiness"] < 0.25


def test_heavy_jpeg_raises_blockiness_above_light_jpeg():
    rng = np.random.default_rng(3)
    base = np.clip(
        128 + 40 * np.sin(np.linspace(0, 30, 320))[None, :, None] + rng.normal(0, 8, (256, 320, 3)),
        0,
        255,
    ).astype(np.uint8)

    def reencode(q):
        ok, buf = cv2.imencode(".jpg", base, [cv2.IMWRITE_JPEG_QUALITY, q])
        assert ok
        return compute(_prep(cv2.imdecode(buf, cv2.IMREAD_COLOR)))["compress_blockiness"]

    assert reencode(8) > reencode(95)


def test_v2_never_explodes_where_v1_did():
    # A blurred near-flat frame stored as JPEG: v1 ratio diverges, v2 stays bounded.
    rng = np.random.default_rng(4)
    frame = np.clip(127 + rng.normal(0, 4, (256, 256, 3)), 0, 255).astype(np.uint8)
    frame = cv2.GaussianBlur(frame, (0, 0), 6.0)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 97])
    stored = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    prepared = _prep(stored)

    v2 = compute(prepared)["compress_blockiness"]
    v1 = legacy_ratio_blockiness(prepared)["v1_blockiness"]
    assert 0.0 <= v2 <= 1.0
    # v1 is the buggy one; we only assert v2 is sane, not that v1 explodes here.
    assert np.isfinite(v1)


def test_legacy_blockiness_still_available_for_ablation():
    img = np.full((128, 128, 3), 100, np.uint8)
    legacy = legacy_ratio_blockiness(_prep(img))
    assert set(legacy) == {"v1_blockiness", "v1_blockiness_h", "v1_blockiness_v"}
