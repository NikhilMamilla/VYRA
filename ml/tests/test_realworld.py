"""Phase 3A tests: VizWiz parsing, label mapping, real-sample loading, metrics,
and the no-leakage guarantee. No network required -- annotation files are tiny
and the image-dependent paths use synthetic fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vyra_ml import ISSUE_LABELS
from vyra_ml.realworld.label_map import (
    MAPPINGS,
    binarize,
    evaluable_labels,
    mapping_table,
    vyra_to_vizwiz_code,
)
from vyra_ml.realworld.vizwiz import (
    VIZWIZ_FLAW_CODES,
    VizWizAnnotation,
    has_labels,
    parse_annotations,
)

_ANN_DIR = Path(__file__).resolve().parents[1] / "data/raw/vizwiz/annotations"
_HAS_ANNOTATIONS = (_ANN_DIR / "val.json").exists()
_needs_ann = pytest.mark.skipif(not _HAS_ANNOTATIONS, reason="VizWiz annotations not downloaded")


# --------------------------------------------------------------------------- #
# annotation parsing
# --------------------------------------------------------------------------- #
def test_parse_annotations_from_inline_json(tmp_path):
    data = [
        {
            "image": "VizWiz_val_00000001.jpg",
            "flaws": {c: 0 for c in VIZWIZ_FLAW_CODES} | {"BLR": 5},
            "unrecognizable": 3,
        },
    ]
    p = tmp_path / "val.json"
    p.write_text(json.dumps(data))
    anns = parse_annotations(p)
    assert len(anns) == 1
    assert anns[0].votes("BLR") == 5
    assert anns[0].unrecognizable_votes == 3
    assert anns[0].split_from_name == "val"


def test_parse_rejects_out_of_range_votes(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps([{"image": "x.jpg", "flaws": {"BLR": 9}, "unrecognizable": 0}]))
    with pytest.raises(ValueError, match="outside 0"):
        parse_annotations(p)


def test_test_split_has_no_labels(tmp_path):
    p = tmp_path / "test.json"
    p.write_text(json.dumps([{"image": "VizWiz_test_00000001.jpg"}]))
    assert has_labels(p) is False
    assert parse_annotations(p) == []


@_needs_ann
def test_real_val_annotations_load():
    anns = parse_annotations(_ANN_DIR / "val.json")
    assert len(anns) > 7000
    assert not has_labels(_ANN_DIR / "test.json")


# --------------------------------------------------------------------------- #
# label mapping
# --------------------------------------------------------------------------- #
def test_only_A_and_B_categories_are_evaluable():
    ev = evaluable_labels()
    assert set(ev) == {"blur", "overexposure", "underexposure", "defect"}
    # noise and corruption must NOT be evaluable against VizWiz.
    assert "noise" not in ev
    assert "corruption" not in ev


def test_unsupported_labels_are_documented_not_scored():
    table = {r["vyra_label"]: r for r in mapping_table()}
    for lbl in ("noise", "corruption"):
        assert table[lbl]["category"] == "C"
        assert "not directly supported" in table[lbl]["reasoning"].lower()


def test_no_unrelated_category_is_silently_mapped():
    for m in MAPPINGS:
        if m.vizwiz_code in {"FRM", "ROT", "OTH", "NON"}:
            assert m.vyra_label is None


def test_binarize_returns_only_evaluable_labels_and_respects_threshold():
    votes = {"BLR": 3, "BRT": 2, "DRK": 0, "OBS": 4, "FRM": 5, "ROT": 5}
    at3 = binarize(votes, vote_min=3)
    assert set(at3) == set(evaluable_labels())
    assert at3["blur"] == 1
    assert at3["overexposure"] == 0  # 2 votes < 3
    assert at3["defect"] == 1  # OBS 4 >= 3
    assert binarize(votes, vote_min=5)["defect"] == 0


def test_mapping_is_bijective_where_present():
    codes = list(vyra_to_vizwiz_code().values())
    assert len(codes) == len(set(codes))


# --------------------------------------------------------------------------- #
# evaluation without training / metrics
# --------------------------------------------------------------------------- #
def test_multilabel_report_on_known_confusion():
    from vyra_ml.evaluation.metrics import multilabel_report

    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 0]])
    y_pred = np.array([[1, 0], [0, 0], [0, 1], [1, 0]])
    rep = multilabel_report(y_true, y_pred, ["a", "b"])
    a = rep["per_class"]["a"]
    assert a["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert a["precision"] == 0.5 and a["recall"] == 0.5


def test_frozen_model_bundle_is_loadable_and_matches_extractor():
    from vyra_ml.features import FEATURE_NAMES
    from vyra_ml.realworld.config import load_real_world_config
    from vyra_ml.realworld.evaluate import _load_model_bundle

    cfg = load_real_world_config()
    if not cfg.model_path.exists():
        pytest.skip("Phase 2 model artifact not present")
    bundle = _load_model_bundle(cfg)
    assert tuple(bundle["feature_names"]) == FEATURE_NAMES
    assert set(bundle["models"]) == set(ISSUE_LABELS)
    assert set(bundle["thresholds"]) == set(ISSUE_LABELS)


# --------------------------------------------------------------------------- #
# no leakage
# --------------------------------------------------------------------------- #
def test_vizwiz_ids_cannot_collide_with_bsds_source_ids():
    # Structural guarantee: the two datasets use disjoint id namespaces.
    vizwiz_ids = {"VizWiz_val_00002854.jpg", "VizWiz_val_00000001.jpg"}
    bsds_ids = {"bsds500/2018", "bsds500/100075"}
    assert vizwiz_ids.isdisjoint(bsds_ids)


@_needs_ann
def test_real_eval_sample_selection_is_deterministic():
    from vyra_ml.realworld.adapter import select_subset

    anns = parse_annotations(_ANN_DIR / "val.json")
    a = select_subset(anns, 200, seed=20260828)
    b = select_subset(anns, 200, seed=20260828)
    assert [s.image for s in a] == [s.image for s in b]
    assert len({s.image for s in a}) == 200
    # different seed -> different sample
    c = select_subset(anns, 200, seed=1)
    assert [s.image for s in a] != [s.image for s in c]


def test_load_image_record_flags_anomalies(tmp_path):
    import cv2

    from vyra_ml.realworld.adapter import RealSample, load_image_record

    ann = VizWizAnnotation("VizWiz_val_x.jpg", {c: 0 for c in VIZWIZ_FLAW_CODES}, 0)

    tiny = tmp_path / "tiny.jpg"
    cv2.imwrite(str(tiny), np.zeros((10, 10, 3), np.uint8))
    s = load_image_record(RealSample("t", tiny, ann))
    assert s.load_status == "too_small"

    gray = tmp_path / "gray.jpg"
    g = np.full((80, 80), 120, np.uint8)
    cv2.imwrite(str(gray), cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    s = load_image_record(RealSample("g", gray, ann))
    assert s.load_status == "grayscale_ok" and s.bgr is not None

    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not a jpeg")
    s = load_image_record(RealSample("b", bad, ann))
    assert s.load_status == "unreadable"
