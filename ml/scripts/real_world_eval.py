"""Phase 3A CLI: real-world validation on VizWiz-QualityIssues.

    python scripts/real_world_eval.py annotations   # download the small annotation files
    python scripts/real_world_eval.py mapping       # write the label-mapping report only
    python scripts/real_world_eval.py run           # full experiment (partial download + eval)

Reads configs/real_world_eval.yaml (override with --config).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vyra_ml.realworld.config import load_real_world_config  # noqa: E402

_ANNOTATION_ZIP = "https://vizwiz.cs.colorado.edu/VizWiz_final/image_quality/annotations.zip"


def cmd_annotations(args) -> None:
    cfg = load_real_world_config(args.config)
    cfg.annotations_dir.mkdir(parents=True, exist_ok=True)
    dest = cfg.annotations_dir.parent / "annotations.zip"
    if not all((cfg.annotations_dir / f"{s}.json").exists() for s in ("train", "val", "test")):
        print(f"[annotations] downloading {_ANNOTATION_ZIP}")
        urllib.request.urlretrieve(_ANNOTATION_ZIP, dest)  # noqa: S310 - known host
        with zipfile.ZipFile(dest) as z:
            z.extractall(cfg.annotations_dir)
    print(f"[annotations] ready in {cfg.annotations_dir}")


def cmd_mapping(args) -> None:
    from vyra_ml.realworld.evaluate import _write_label_mapping

    cfg = load_real_world_config(args.config)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    _write_label_mapping(cfg)
    print(f"[mapping] wrote {cfg.reports_dir / 'label_mapping.md'}")


def cmd_run(args) -> None:
    from vyra_ml.realworld.evaluate import run_real_world_eval

    cfg = load_real_world_config(args.config)
    if not cfg.annotation_file.exists():
        raise SystemExit(
            f"{cfg.annotation_file} missing - run: python scripts/real_world_eval.py annotations"
        )
    if not cfg.model_path.exists():
        raise SystemExit(f"Model artifact missing: {cfg.model_path}")
    out = run_real_world_eval(cfg)
    print(f"[run] experiment record: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in {
        "annotations": cmd_annotations,
        "mapping": cmd_mapping,
        "run": cmd_run,
    }.items():
        sub.add_parser(name).set_defaults(func=fn)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
