"""Build a small, documented demo image set for the VYRA UI.

Takes a few clean BSDS500 photos and applies the project's own degradation
classes (so the artefacts are the same ones the model was trained on), plus one
untouched clean image. Writes JPEGs + a README to ``demo/`` at the repo root.

Run: ``python scripts/make_demo_samples.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vyra_ml.degradations import get_degradation  # noqa: E402
from vyra_ml.seeding import derive_rng  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "ml" / "data" / "raw" / "bsds500" / "images"
OUT = REPO / "demo"
SEED = 20260828

# (filename, source image, [(degradation, severity), ...], expected behaviour)
PLAN = [
    ("01_clean.jpg", "test/100007.jpg", [], "GOOD — no issue flagged"),
    ("02_blur.jpg", "test/108036.jpg", [("blur", 4)], "blur flagged (real-world validated)"),
    ("03_underexposed.jpg", "test/14092.jpg", [("underexposure", 4)], "underexposure flagged"),
    (
        "04_overexposed.jpg",
        "test/163004.jpg",
        [("overexposure", 4)],
        "overexposure flagged (weak detector)",
    ),
    ("05_noisy.jpg", "test/196062.jpg", [("noise", 4)], "noise flagged (synthetic-validated only)"),
    (
        "06_compressed.jpg",
        "test/226033.jpg",
        [("corruption", 4)],
        "corruption flagged (synthetic-validated only)",
    ),
    (
        "07_multi_blur_dark.jpg",
        "test/258089.jpg",
        [("underexposure", 3), ("blur", 3)],
        "blur + underexposure, lower quality score",
    ),
    (
        "08_defect_blotch.jpg",
        "test/309040.jpg",
        [("defect", 4)],
        "potential visual defect may be flagged (screening only, ~33% precision)",
    ),
]


def main() -> None:
    if not SRC.is_dir():
        raise SystemExit(f"BSDS images not found at {SRC}; run scripts/pipeline.py build first")
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# VYRA demo images",
        "",
        "Clean BSDS500 photos with the project's own synthetic degradations applied",
        "(`ml/scripts/make_demo_samples.py`). blur / underexposure / overexposure are",
        "real-world validated; noise / corruption are synthetic-validated only; the",
        "defect signal is a weak screening cue (ROC-AUC 0.60, ~1 in 3 flags real).",
        "",
        "Observed behaviour of `vyra-quality-model-v1` on this exact set — an honest",
        "snapshot, not a target. The overexposure detector is weak (real F1 0.19),",
        "corruption at this severity is not always caught, and the defect detector",
        "misses ~2/3 of defects by design:",
        "",
        "| file | score / label | flagged |",
        "|---|---|---|",
        "| `01_clean.jpg` | 100 GOOD | — |",
        "| `02_blur.jpg` | 68 ACCEPTABLE | blur 0.67 |",
        "| `03_underexposed.jpg` | 66 DEGRADED | underexposure 0.80, blur 0.37 |",
        "| `04_overexposed.jpg` | 99 GOOD | overexposure 0.12 (just over threshold) |",
        "| `05_noisy.jpg` | 84 ACCEPTABLE | noise 0.68, underexposure 0.50 |",
        "| `06_compressed.jpg` | 88 GOOD | blur 0.48 (corruption not caught here) |",
        "| `07_multi_blur_dark.jpg` | 45 DEGRADED | blur 0.94 |",
        "| `08_defect_blotch.jpg` | 100 GOOD | — (defect missed — expected ~2/3 of the time) |",
        "",
        "## Source degradations applied",
        "",
        "| file | contents |",
        "|---|---|",
    ]
    for name, rel, degs, expected in PLAN:
        img = cv2.imread(str(SRC / rel), cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip {name}: missing {rel}")
            continue
        for deg_name, sev in degs:
            rng = derive_rng(SEED, "demo", name, deg_name)
            img = get_degradation(deg_name).apply(img, sev, rng).image
        cv2.imwrite(str(OUT / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        desc = "clean" if not degs else " + ".join(f"{d} sev{s}" for d, s in degs)
        lines.append(f"| `{name}` | {desc} ({expected}) |")
        print(f"wrote {name}")

    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{OUT}/README.md written")


if __name__ == "__main__":
    main()
