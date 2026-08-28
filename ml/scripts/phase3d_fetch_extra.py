"""Phase 3D: fetch + feature-extract EXTRA real VizWiz-train images.

The current real validation split (``realval_features.parquet``, n~2489, a uniform
random sample of VizWiz train) has very thin support for the rare issues
(underexposure ~69, overexposure ~49 at >=3 votes). This starves a real-trained
classifier for those labels.

Here we download every *additional* VizWiz-train image that at least two workers
flagged as too dark / too bright / obstructed, and extract cvfeat-v2 features for
it. These rows are used **only as extra training data** for Phase 3D. Threshold
selection and the cross-validated F1 estimate stay on the original uniform sample
(natural prevalence); the frozen evaluation set (VizWiz val) is untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

import numpy as np  # noqa: E402

from vyra_ml.realworld.adapter import select_subset  # noqa: E402
from vyra_ml.realworld.features import build_vizwiz_feature_table  # noqa: E402
from vyra_ml.realworld.fetch import fetch_from_remote_zip  # noqa: E402
from vyra_ml.realworld.vizwiz import parse_annotations  # noqa: E402

SEED = 20260828
ZIP_URL = "https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip"
IMAGE_DIR = ML_ROOT / "data" / "raw" / "vizwiz" / "images_train"
OUT_PATH = ML_ROOT / "data" / "processed" / "realval_extra_features.parquet"
MAX_EXTRA = 4200


def main() -> None:
    anns = parse_annotations(ML_ROOT / "data" / "raw" / "vizwiz" / "annotations" / "train.json")
    already = {a.image for a in select_subset(anns, 2500, SEED)}

    # rare-class enrichment: >=2 votes on a rare issue, not already sampled
    rare = [
        a
        for a in anns
        if a.image not in already
        and (a.votes("DRK") >= 2 or a.votes("BRT") >= 2 or a.votes("OBS") >= 2)
    ]
    rare_names = {a.image for a in rare}

    # a little extra uniform random (clean negatives / blur) so the training
    # pool is not *only* rare-positive images
    rest = [a for a in anns if a.image not in already and a.image not in rare_names]
    rng = np.random.default_rng(SEED + 1)
    n_uniform = min(len(rest), max(0, MAX_EXTRA - len(rare)))
    uniform_extra = [rest[i] for i in sorted(rng.choice(len(rest), size=n_uniform, replace=False))]

    subset = sorted(rare + uniform_extra, key=lambda a: a.image)
    print(
        f"extra subset: {len(subset)}  (rare-enriched {len(rare)}, uniform {len(uniform_extra)})",
        flush=True,
    )
    for code in ("BLR", "DRK", "BRT", "OBS"):
        print(f"  {code}: >=2 {sum(a.votes(code) >= 2 for a in subset)}  "
              f">=3 {sum(a.votes(code) >= 3 for a in subset)}", flush=True)

    members = [f"train/{a.image}" for a in subset]
    fetch_from_remote_zip(
        ZIP_URL,
        members,
        IMAGE_DIR,
        workers=16,
        on_progress=lambda d, t: print(f"  fetched {d}/{t}", flush=True),
    )

    stats = build_vizwiz_feature_table(
        subset,
        IMAGE_DIR,
        split="real_val_extra",
        work_long_edge=384,
        out_path=OUT_PATH,
    )
    print("FEATURE TABLE:", stats, flush=True)


if __name__ == "__main__":
    main()
