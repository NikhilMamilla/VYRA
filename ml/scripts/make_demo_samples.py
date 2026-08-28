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

import numpy as np  # noqa: E402

from vyra_ml.degradations import get_degradation  # noqa: E402
from vyra_ml.seeding import derive_rng  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "ml" / "data" / "raw" / "bsds500" / "images"
OUT = REPO / "demo"
SEED = 20260828

# blur / underexposure / overexposure are now trained on real VizWiz photos, where
# "too dark" / "too bright" mean genuinely extreme frames (median luma of a real
# ≥4/5-vote dark image is ~0.04, and bright images carry large blown-out regions,
# not just a high mean). The synthetic exposure degradation is deliberately mild
# and clamps its shadow floor, so a couple of demo images get an extra
# linear-light exposure push ("gain") past the training range to land an
# unambiguous positive -- the kind of frame a person would also call too dark.
#
# (filename, source image, [(degradation, severity), ...], gain, expected behaviour)
PLAN = [
    ("01_clean.jpg", "test/100007.jpg", [], 1.0, "GOOD — no issue flagged"),
    ("02_blur.jpg", "test/108036.jpg", [("blur", 4)], 1.0, "blur flagged (real-world validated)"),
    (
        "03_underexposed.jpg",
        "test/141048.jpg",
        [],
        0.16,
        "underexposure flagged (real-world validated)",
    ),
    (
        "04_overexposed.jpg",
        "test/157087.jpg",
        [("overexposure", 5), ("overexposure", 5)],
        1.0,
        "overexposure flagged (real-world validated)",
    ),
    (
        "05_noisy.jpg",
        "test/196062.jpg",
        [("noise", 4)],
        1.0,
        "noise flagged (synthetic-validated only)",
    ),
    (
        "06_compressed.jpg",
        "test/226033.jpg",
        [("corruption", 5)],
        1.0,
        "corruption flagged (synthetic-validated only)",
    ),
    (
        "07_multi_blur_dark.jpg",
        "test/258089.jpg",
        [("blur", 3)],
        0.22,
        "blur + underexposure, lower quality score",
    ),
    (
        "08_defect_blotch.jpg",
        "test/309040.jpg",
        [("defect", 5)],
        1.0,
        "potential visual defect may be flagged (screening only, ~33% precision)",
    ),
]


def _apply_gain(img: np.ndarray, gain: float) -> np.ndarray:
    """Extra tone-scale on the encoded pixels (gain<1 darkens, >1 brightens).

    Applied in the display domain, not linear light: a real "too dark" phone
    photo has an encoded mean luma near 0.04, which a linear-light cut of the
    same factor never reaches.
    """
    if gain == 1.0:
        return img
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def main() -> None:
    if not SRC.is_dir():
        raise SystemExit(f"BSDS images not found at {SRC}; run scripts/pipeline.py build first")
    OUT.mkdir(parents=True, exist_ok=True)

    source_rows: list[str] = []
    generated: list[tuple[str, str]] = []
    for name, rel, degs, gain, expected in PLAN:
        img = cv2.imread(str(SRC / rel), cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip {name}: missing {rel}")
            continue
        for i, (deg_name, sev) in enumerate(degs):
            rng = derive_rng(SEED, "demo", name, f"{deg_name}{i}")
            img = get_degradation(deg_name).apply(img, sev, rng).image
        img = _apply_gain(img, gain)
        cv2.imwrite(str(OUT / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        parts = [f"{d} sev{s}" for d, s in degs]
        if gain != 1.0:
            parts.append(f"exposure ×{gain}")
        desc = "clean" if not parts else " + ".join(parts)
        source_rows.append(f"| `{name}` | {desc} ({expected}) |")
        generated.append((name, expected))
        print(f"wrote {name}")

    # Observed behaviour: run the shipped bundle on the exact files we just wrote.
    obs_rows: list[str] = []
    try:
        from vyra_ml.inference import VyraQualityModel

        model = VyraQualityModel.load(REPO / "ml" / "artifacts" / "vyra-quality-model-v1")
        for name, _ in generated:
            a = model.analyze_bytes((OUT / name).read_bytes())
            flagged = [f"{i.issue} {i.probability:.2f}" for i in a.issues if i.flagged]
            if a.potential_defect.flagged:
                flagged.append(f"potential_defect {a.potential_defect.probability:.2f}")
            obs_rows.append(
                f"| `{name}` | {a.quality_score:.0f} {a.quality_label} | "
                f"{', '.join(flagged) if flagged else '—'} |"
            )
        obs_version = model.model_version
    except Exception:  # noqa: BLE001 - README still useful without the bundle
        obs_rows = ["| (model bundle not available — run scripts/export_inference_bundle.py) | | |"]
        obs_version = "vyra-quality-model-v1"

    lines = [
        "# VYRA demo images",
        "",
        "Clean BSDS500 photos with the project's synthetic degradation classes applied",
        "(`ml/scripts/make_demo_samples.py`). blur / underexposure / overexposure are",
        "**trained and validated on real VizWiz photos**; noise / corruption are",
        "synthetic-validated only; the defect signal is a weak screening cue",
        "(ROC-AUC 0.60, ~1 in 3 flags real).",
        "",
        "Because the exposure heads now learn what real annotators call \"too dark\" /",
        "\"too bright\" (genuinely extreme frames), `03` and `07` get an extra exposure",
        "cut past the synthetic severity range so they land an unambiguous positive.",
        "",
        f"Observed behaviour of `{obs_version}` on this exact set — an honest snapshot,",
        "not a target. Overexposure still has lower recall on real photos (F1 0.36);",
        "`06` shows the synthetic-only corruption head failing to transfer (the",
        "re-encoded frame reads as mild blur instead); the patch-anomaly defect cue",
        "also fires on the large black / blown regions in `03`, `04` and `07`.",
        "",
        "| file | score / label | flagged |",
        "|---|---|---|",
        *obs_rows,
        "",
        "## Source degradations applied",
        "",
        "| file | contents |",
        "|---|---|",
        *source_rows,
    ]

    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{OUT}/README.md written")


if __name__ == "__main__":
    main()
