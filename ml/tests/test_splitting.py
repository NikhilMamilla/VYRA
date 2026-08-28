from __future__ import annotations

import pytest

from vyra_ml.config import SplitRatios
from vyra_ml.splitting import (
    assert_no_leakage,
    split_counts,
    split_originals,
)

RATIOS = SplitRatios(train=0.7, val=0.15, test=0.15)


def test_assignment_is_deterministic():
    ids = [f"src/{i}" for i in range(200)]
    a = split_originals(ids, RATIOS, seed=123)
    b = split_originals(ids, RATIOS, seed=123)
    assert a == b


def test_assignment_is_order_independent():
    ids = [f"src/{i}" for i in range(200)]
    forward = split_originals(ids, RATIOS, seed=1)
    backward = split_originals(list(reversed(ids)), RATIOS, seed=1)
    assert forward == backward


def test_ratios_approximately_respected():
    ids = [f"src/{i}" for i in range(4000)]
    counts = split_counts(split_originals(ids, RATIOS, seed=7))
    assert abs(counts["train"] / 4000 - 0.7) < 0.03
    assert abs(counts["val"] / 4000 - 0.15) < 0.03
    assert abs(counts["test"] / 4000 - 0.15) < 0.03


def test_seed_changes_partition():
    ids = [f"src/{i}" for i in range(500)]
    assert split_originals(ids, RATIOS, seed=1) != split_originals(ids, RATIOS, seed=2)


def test_no_leakage_passes_for_variant_rows_sharing_split():
    assignment = split_originals([f"o{i}" for i in range(50)], RATIOS, seed=3)
    # Simulate 5 degraded variants per original, each inheriting the split.
    rows = [
        {"source_id": sid, "split": split} for sid, split in assignment.items() for _ in range(5)
    ]
    assert_no_leakage(rows)


def test_no_leakage_detects_a_planted_violation():
    rows = [
        {"source_id": "o1", "split": "train"},
        {"source_id": "o1", "split": "test"},  # same original in two splits
    ]
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage(rows)


def test_source_id_never_spans_train_and_test_over_the_whole_pipeline():
    """The core guarantee: build the split, expand to variants, verify integrity."""
    originals = [f"bsds500/img_{i:04d}" for i in range(300)]
    assignment = split_originals(originals, RATIOS, seed=20260828)

    manifest_rows = []
    for sid in originals:
        split = assignment[sid]
        for kind in ("clean", "blur", "noise", "multi"):
            manifest_rows.append({"source_id": sid, "split": split, "sample_id": f"{sid}__{kind}"})

    by_source = {}
    for row in manifest_rows:
        by_source.setdefault(row["source_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_source.values())
    assert_no_leakage(manifest_rows)
