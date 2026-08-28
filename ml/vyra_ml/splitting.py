"""Leakage-safe, original-level train/val/test splitting.

THE non-negotiable rule of this project: the split is decided on *original source
images*, before any degradation is generated. Every synthetic variant inherits
its original's split. The same ``source_id`` can therefore never contribute
pixels to more than one split.

Assignment is by hashing the ``source_id`` (not by shuffling a list), so:

* it is order-independent -- adding or removing originals does not move the
  others between splits;
* it is reproducible from the seed alone, with no stored split file required;
* it is trivially auditable: ``assign_split(source_id, ...)`` recomputes it.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable

from vyra_ml.config import SplitRatios

_SPLIT_NAMES = ("train", "val", "test")
_RESOLUTION = 1_000_000  # hash buckets


def _unit_hash(source_id: str, seed: int) -> float:
    payload = f"{seed}:{source_id}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return (int.from_bytes(digest, "big") % _RESOLUTION) / _RESOLUTION


def assign_split(source_id: str, ratios: SplitRatios, seed: int) -> str:
    """Return 'train' | 'val' | 'test' for one original image."""
    h = _unit_hash(source_id, seed)
    if h < ratios.train:
        return "train"
    if h < ratios.train + ratios.val:
        return "val"
    return "test"


def split_originals(source_ids: Iterable[str], ratios: SplitRatios, seed: int) -> dict[str, str]:
    """Map every source id to its split. Deduplicates and sorts for determinism."""
    unique = sorted(set(source_ids))
    return {sid: assign_split(sid, ratios, seed) for sid in unique}


def split_counts(assignment: dict[str, str]) -> dict[str, int]:
    counts = Counter(assignment.values())
    return {name: counts.get(name, 0) for name in _SPLIT_NAMES}


def assert_no_leakage(manifest_rows: Iterable[dict]) -> None:
    """Raise if any ``source_id`` appears under more than one split.

    Call this on the final manifest as a build-time guard.
    """
    seen: dict[str, str] = {}
    offenders: dict[str, set[str]] = {}
    for row in manifest_rows:
        sid, split = row["source_id"], row["split"]
        if sid in seen and seen[sid] != split:
            offenders.setdefault(sid, {seen[sid]}).add(split)
        seen.setdefault(sid, split)
    if offenders:
        detail = ", ".join(f"{sid}: {sorted(splits)}" for sid, splits in offenders.items())
        raise AssertionError(f"Data leakage - source_id in multiple splits: {detail}")
