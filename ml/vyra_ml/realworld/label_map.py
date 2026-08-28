"""Explicit VizWiz -> VYRA label mapping.

Every VizWiz category is classified A/B/C/D with reasoning. Categories that
cannot support a VYRA label are left unmapped -- we never invent ground truth.

    A  directly mappable          B  partially mappable
    C  not mappable               D  useful only as auxiliary information

Binarisation of vote counts is a separate, explicit choice (``POSITIVE_VOTE_MIN``):
a VYRA label is "present" for evaluation when the mapped VizWiz code has >= 3 of
5 worker votes (simple majority). Metrics are also reported at thresholds 2 and 4
as a sensitivity check. Raw vote counts are always preserved in the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass

from vyra_ml import ISSUE_LABELS

POSITIVE_VOTE_MIN = 3
SENSITIVITY_THRESHOLDS = (2, 3, 4)


@dataclass(frozen=True)
class Mapping:
    vizwiz_code: str
    vyra_label: str | None
    category: str  # "A" | "B" | "C" | "D"
    confidence: str  # "high" | "medium" | "low" | "n/a"
    reasoning: str


MAPPINGS: tuple[Mapping, ...] = (
    Mapping(
        "BLR",
        "blur",
        "A",
        "high",
        "VizWiz BLR (blurry / out of focus / motion) is the same concept as VYRA "
        "'blur'. Both defocus and motion blur are covered on each side.",
    ),
    Mapping(
        "BRT",
        "overexposure",
        "A",
        "medium",
        "'Too bright' corresponds to VYRA 'overexposure'. Slight gap: BRT can also "
        "be flagged for glare/reflections that are not global overexposure.",
    ),
    Mapping(
        "DRK",
        "underexposure",
        "A",
        "medium",
        "'Too dark' corresponds to VYRA 'underexposure'. Real dark images also "
        "carry heavy sensor noise, which VYRA models as a separate label.",
    ),
    Mapping(
        "OBS",
        "defect",
        "B",
        "low",
        "OBS (finger/object over the lens) overlaps ONLY the 'occlusion' sub-type "
        "of VYRA's synthetic 'defect' (1 of 5 synthetic defect types). VYRA defect "
        "also covers dead pixels, banding, block corruption, colour blotches -- "
        "none of which VizWiz labels. Treated as a weak proxy, reported with heavy "
        "caveats.",
    ),
    Mapping(
        "FRM",
        None,
        "C",
        "n/a",
        "Framing / subject cut off. VYRA has no framing concept and it is not an "
        "image-quality degradation in VYRA's taxonomy. Unmapped.",
    ),
    Mapping(
        "ROT",
        None,
        "C",
        "n/a",
        "Wrong orientation. Not a VYRA concept and not a degradation. Unmapped.",
    ),
    Mapping(
        "OTH",
        None,
        "D",
        "n/a",
        "Free-text 'other'. Too heterogeneous to map; kept only as context.",
    ),
    Mapping(
        "NON",
        None,
        "D",
        "n/a",
        "'No issue' vote count. Auxiliary: high NON is corroborating evidence for "
        "a clean image (used in analysis, not as a label).",
    ),
)

# VizWiz's separate unrecognizability question.
UNRECOGNIZABLE_NOTE = (
    "'unrecognizable' (content too degraded to recognise) is a SEVERITY signal, "
    "not a cause. Cross-tabulation on val shows it correlates weakly and diffusely "
    "with BLR (r=0.32), DRK (0.35), BRT (0.28) and OBS (0.30) -- i.e. it is driven "
    "by many degradations, not by compression. It is therefore NOT mapped to VYRA "
    "'corruption'. Kept as auxiliary severity information."
)

# VYRA labels with no VizWiz support at all.
UNSUPPORTED_VYRA_LABELS = {
    "noise": (
        "VizWiz has no noise category. Real sensor noise is present in many dark "
        "images but is never annotated. Not directly supported by this dataset."
    ),
    "corruption": (
        "VizWiz has no compression / digital-corruption category. 'unrecognizable' "
        "is severity, not corruption (see note). Not directly supported by this "
        "dataset."
    ),
}


def evaluable_labels() -> list[str]:
    """VYRA labels that have a defensible VizWiz mapping (categories A and B)."""
    return [m.vyra_label for m in MAPPINGS if m.category in {"A", "B"} and m.vyra_label]


def vyra_to_vizwiz_code() -> dict[str, str]:
    return {m.vyra_label: m.vizwiz_code for m in MAPPINGS if m.vyra_label}


def binarize(annotation_flaw_votes: dict[str, int], vote_min: int = POSITIVE_VOTE_MIN) -> dict:
    """Project one annotation onto the evaluable VYRA labels at a vote threshold.

    Returns ``{label: 0/1}`` for evaluable labels only. Labels VizWiz does not
    support are absent from the dict -- callers must not assume 0.
    """
    code_by_label = vyra_to_vizwiz_code()
    return {
        label: int(annotation_flaw_votes.get(code_by_label[label], 0) >= vote_min)
        for label in evaluable_labels()
    }


def mapping_table() -> list[dict]:
    rows = [
        {
            "vizwiz_code": m.vizwiz_code,
            "vyra_label": m.vyra_label,
            "category": m.category,
            "confidence": m.confidence,
            "reasoning": m.reasoning,
        }
        for m in MAPPINGS
    ]
    for label in ISSUE_LABELS:
        if label in UNSUPPORTED_VYRA_LABELS:
            rows.append(
                {
                    "vizwiz_code": None,
                    "vyra_label": label,
                    "category": "C",
                    "confidence": "n/a",
                    "reasoning": UNSUPPORTED_VYRA_LABELS[label],
                }
            )
    return rows
