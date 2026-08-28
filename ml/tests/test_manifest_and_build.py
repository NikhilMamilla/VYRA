"""Manifest schema + a full end-to-end build on the offline skimage source.

The skimage source has only 5 originals -- too few for a real experiment, but
enough to exercise ingest -> split -> degrade -> manifest and prove the leakage
guard on the real pipeline output.
"""

from __future__ import annotations

import json

import pytest

from vyra_ml import ISSUE_LABELS
from vyra_ml.config import load_config
from vyra_ml.manifest import COLUMNS, read_manifest
from vyra_ml.splitting import assert_no_leakage


@pytest.fixture
def tiny_cfg(tmp_path):
    base = load_config()
    # Rewrite paths under tmp_path and shrink to the offline source.
    raw = json.loads(
        json.dumps(
            {
                "version": "test-build",
                "seed": 20260828,
                "paths": {
                    "data_root": str(tmp_path / "data"),
                    "raw_subdir": "raw",
                    "processed_subdir": "processed",
                    "manifests_subdir": "manifests",
                    "reports_subdir": "reports",
                    "runs_subdir": "runs",
                },
                "dataset": {
                    "source": "skimage",
                    "max_originals": 5,
                    "target_long_edge": 128,
                    "splits": {"train": 0.6, "val": 0.2, "test": 0.2},
                },
                "degradation": {
                    "clean_per_original": 1,
                    "single_per_original": 4,
                    "multi_per_original": 2,
                    "multi_min": 2,
                    "multi_max": 3,
                    "severity_levels": [1, 2, 3, 4, 5],
                    "enabled": list(base.degradation.enabled),
                },
                "features": {"version": "cvfeat-v1", "work_long_edge": 128},
                "baseline": base.baseline.__dict__ | {"model": "random_forest"},
            }
        )
    )
    cfg_path = tmp_path / "experiment.yaml"
    import yaml

    cfg_path.write_text(yaml.safe_dump(raw))
    return load_config(cfg_path)


def test_end_to_end_build(tiny_cfg):
    from vyra_ml.dataset_build import build_dataset

    result = build_dataset(tiny_cfg, progress=False)
    assert result.n_originals == 5
    assert result.n_samples == 5 * (1 + 4 + 2)

    frame = read_manifest(result.manifest_paths["parquet"])
    assert list(frame.columns) == COLUMNS
    assert frame["sample_id"].is_unique

    # Every stored image exists and is non-empty.
    for rel in frame["image_path"]:
        assert (tiny_cfg.data_dir("processed") / rel).stat().st_size > 0

    # Clean rows carry all-zero labels; degraded rows carry at least one.
    clean = frame[frame.is_clean]
    assert (clean[[f"label_{n}" for n in ISSUE_LABELS]].sum(axis=1) == 0).all()
    degraded = frame[~frame.is_clean]
    assert (degraded[[f"label_{n}" for n in ISSUE_LABELS]].sum(axis=1) >= 1).all()


def test_build_output_has_no_leakage(tiny_cfg):
    from vyra_ml.dataset_build import build_dataset

    result = build_dataset(tiny_cfg, progress=False)
    frame = read_manifest(result.manifest_paths["parquet"])
    assert_no_leakage(frame.to_dict("records"))

    # Explicit: intersection of source_ids across splits is empty.
    by_split = {s: set(g["source_id"]) for s, g in frame.groupby("split")}
    splits = list(by_split)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            assert by_split[splits[i]].isdisjoint(by_split[splits[j]])


def test_build_is_reproducible(tiny_cfg):
    from vyra_ml.dataset_build import build_dataset

    r1 = build_dataset(tiny_cfg, progress=False)
    f1 = read_manifest(r1.manifest_paths["parquet"]).sort_values("sample_id").reset_index(drop=True)
    r2 = build_dataset(tiny_cfg, progress=False)
    f2 = read_manifest(r2.manifest_paths["parquet"]).sort_values("sample_id").reset_index(drop=True)

    # Same sample ids, same labels, same content hashes.
    assert list(f1["sample_id"]) == list(f2["sample_id"])
    assert list(f1["sha1"]) == list(f2["sha1"])
    assert f1["degradations_json"].tolist() == f2["degradations_json"].tolist()
