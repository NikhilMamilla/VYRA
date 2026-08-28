"""Deterministic randomness.

Every stochastic step in the dataset build derives its RNG from the single
master seed in the experiment config via :func:`derive_rng`. The derivation is
name-based, so adding a new degradation does not shift the parameters sampled
for existing ones, and rebuilding with the same seed reproduces the dataset
byte-for-byte (subject to identical library versions).
"""

from __future__ import annotations

import hashlib

import numpy as np


def _seed_from(master_seed: int, *parts: str | int) -> int:
    """Stable 64-bit seed from the master seed and a tuple of scope identifiers."""
    payload = "::".join(str(p) for p in (master_seed, *parts)).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big")


def derive_rng(master_seed: int, *parts: str | int) -> np.random.Generator:
    """A NumPy ``Generator`` scoped to ``parts``.

    Example::

        rng = derive_rng(seed, "degrade", sample_id, "blur")

    Two calls with the same arguments return generators that produce identical
    streams; different scopes are independent.
    """
    return np.random.default_rng(_seed_from(master_seed, *parts))
