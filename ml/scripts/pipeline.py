"""VYRA Phase 2 pipeline CLI.

    python scripts/pipeline.py build       # ingest -> split -> degrade -> manifest
    python scripts/pipeline.py features    # extract CV features for the manifest
    python scripts/pipeline.py feature-report
    python scripts/pipeline.py stats       # dataset statistics + plots
    python scripts/pipeline.py baseline    # train + evaluate the classical baseline
    python scripts/pipeline.py all         # everything, in order

All steps read configs/experiment.yaml (override with --config).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vyra_ml.config import load_config  # noqa: E402


def _cfg(args):
    return load_config(args.config)


def cmd_build(args) -> None:
    from vyra_ml.dataset_build import build_dataset

    cfg = _cfg(args)
    print(f"[build] source={cfg.dataset.source} seed={cfg.seed}")
    result = build_dataset(cfg)
    print(f"[build] {result.n_originals} originals -> {result.n_samples} samples")
    print(f"[build] originals per split: {result.split_counts}")
    print(f"[build] manifest: {result.manifest_paths['parquet']}")


def cmd_features(args) -> None:
    from vyra_ml.feature_store import build_feature_table

    cfg = _cfg(args)
    print("[features] extracting...")
    path = build_feature_table(cfg, n_jobs=args.jobs)
    print(f"[features] wrote {path}")


def cmd_feature_report(args) -> None:
    from vyra_ml.feature_report import build_feature_report
    from vyra_ml.features import FEATURE_VERSION
    from vyra_ml.manifest import read_manifest

    cfg = _cfg(args)
    ft = cfg.data_dir("processed") / f"features_{cfg.version}_{FEATURE_VERSION}.parquet"
    manifest = read_manifest(cfg.data_dir("manifests") / f"manifest_{cfg.version}.parquet")
    sample_paths = [
        cfg.data_dir("processed") / p
        for p in manifest["image_path"].sample(min(12, len(manifest)), random_state=cfg.seed)
    ]
    out = build_feature_report(
        ft, cfg.data_dir("reports") / cfg.version, sample_image_paths=sample_paths
    )
    print(f"[feature-report] wrote {out}")


def cmd_stats(args) -> None:
    from vyra_ml.dataset_stats import build_dataset_report

    cfg = _cfg(args)
    out = build_dataset_report(
        cfg.data_dir("manifests") / f"manifest_{cfg.version}.parquet",
        cfg.data_dir("reports") / cfg.version,
    )
    print(f"[stats] wrote {out}")


def cmd_baseline(args) -> None:
    from vyra_ml.experiment import run_baseline
    from vyra_ml.features import FEATURE_VERSION

    cfg = _cfg(args)
    ft = cfg.data_dir("processed") / f"features_{cfg.version}_{FEATURE_VERSION}.parquet"
    artifacts = run_baseline(cfg, ft)
    head = artifacts.metrics["_record"]["headline_metrics"]
    print(f"[baseline] run dir: {artifacts.run_dir}")
    print(f"[baseline] test macro-F1 = {head['test_macro_f1']}  micro-F1 = {head['test_micro_f1']}")
    print(f"[baseline] test subset accuracy = {head['test_subset_accuracy']}")


def cmd_all(args) -> None:
    cmd_build(args)
    cmd_features(args)
    cmd_feature_report(args)
    cmd_stats(args)
    cmd_baseline(args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None, help="Path to experiment.yaml")
    parser.add_argument(
        "--jobs", type=int, default=-1, help="Parallel workers for feature extraction"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in {
        "build": cmd_build,
        "features": cmd_features,
        "feature-report": cmd_feature_report,
        "stats": cmd_stats,
        "baseline": cmd_baseline,
        "all": cmd_all,
    }.items():
        sub.add_parser(name).set_defaults(func=fn)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
