"""VizWiz-QualityIssues annotation parsing.

Dataset: Chiu et al., "Assessing Image Quality Issues for Real-World Problems"
(CVPR 2020). https://vizwiz.org/tasks-and-datasets/image-quality-issues/

Annotation format (``annotations/{train,val}.json``): a JSON list, one object per
image::

    {"image": "VizWiz_val_00002854.jpg",
     "flaws": {"BLR": 5, "BRT": 0, "DRK": 0, "OBS": 0,
               "FRM": 1, "ROT": 0, "OTH": 0, "NON": 0},
     "unrecognizable": 4}

* Five crowd workers per image.
* ``flaws[X]`` = number of workers (0-5) who flagged issue X. **Vote-based, not
  binary.** We keep the raw counts and binarise only at report time.
* ``unrecognizable`` = number of workers (0-5) who judged the image too degraded
  to recognise its content. A severity signal, not an issue category.
* Multiple issues co-occur freely (workers flag several per image).
* ``annotations/test.json`` contains image names only -- labels are withheld for
  the challenge, so **test cannot be used for evaluation**; we use ``val``.

VizWiz issue codes:
    BLR blur   BRT too bright   DRK too dark   OBS obstruction/obscured
    FRM framing/incomplete   ROT wrong rotation   OTH other   NON no issue
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

VIZWIZ_FLAW_CODES: tuple[str, ...] = ("BLR", "BRT", "DRK", "OBS", "FRM", "ROT", "OTH", "NON")
N_WORKERS = 5

VIZWIZ_CODE_MEANING = {
    "BLR": "blurry / out of focus / motion blur",
    "BRT": "too bright / overexposed",
    "DRK": "too dark / underexposed",
    "OBS": "obstruction - finger or object partly covering the lens",
    "FRM": "framing - subject cut off / incomplete / bad composition",
    "ROT": "wrong rotation / orientation",
    "OTH": "other quality issue (free-text)",
    "NON": "no quality issue",
}


@dataclass(frozen=True)
class VizWizAnnotation:
    image: str
    flaw_votes: dict[str, int]  # code -> 0..5
    unrecognizable_votes: int  # 0..5

    @property
    def split_from_name(self) -> str:
        # "VizWiz_val_00002854.jpg" -> "val"
        return self.image.split("_")[1]

    def votes(self, code: str) -> int:
        return self.flaw_votes.get(code, 0)

    def any_flaw_votes(self) -> int:
        return sum(v for c, v in self.flaw_votes.items() if c != "NON")


def parse_annotations(path: str | Path) -> list[VizWizAnnotation]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[VizWizAnnotation] = []
    for rec in raw:
        if "flaws" not in rec:
            # test.json: images only, no labels.
            continue
        flaws = {code: int(rec["flaws"].get(code, 0)) for code in VIZWIZ_FLAW_CODES}
        for code, v in flaws.items():
            if not 0 <= v <= N_WORKERS:
                raise ValueError(f"{rec['image']}: {code} vote {v} outside 0..{N_WORKERS}")
        out.append(
            VizWizAnnotation(
                image=rec["image"],
                flaw_votes=flaws,
                unrecognizable_votes=int(rec.get("unrecognizable", 0)),
            )
        )
    return out


def has_labels(path: str | Path) -> bool:
    """Whether an annotation file carries flaw votes (train/val) or not (test)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return bool(raw) and "flaws" in raw[0]
