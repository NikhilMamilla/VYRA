"""Source-dataset ingestion interface.

A source adapter is responsible for making a set of *clean original images*
available locally and yielding one :class:`SourceImage` per original. It does
not split, degrade or extract features -- it only produces originals with a
stable ``source_id``.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceImage:
    #: Stable, globally unique id for this original (adapter name + local key).
    source_id: str
    #: Name of the originating dataset, recorded in the manifest.
    source_dataset: str
    #: Absolute path to the clean image on disk.
    path: Path


class SourceAdapter(abc.ABC):
    name: str

    def __init__(self, raw_dir: Path) -> None:
        # Each adapter owns a subdirectory of data/raw/.
        self.raw_dir = raw_dir / self.name

    @abc.abstractmethod
    def prepare(self) -> None:
        """Download / unpack the dataset into ``self.raw_dir`` if not already present."""

    @abc.abstractmethod
    def iter_originals(self, limit: int | None = None) -> Iterator[SourceImage]:
        """Yield clean originals in a deterministic order."""
