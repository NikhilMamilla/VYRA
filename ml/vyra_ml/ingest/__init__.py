"""Source adapters registry.

Phase 2 uses synthetic degradation of clean originals, so the adapters here
supply *clean* images. Real-world quality datasets (VizWiz-QualityIssues, SPAQ)
are documented in ``docs/dataset.md`` with an ingestion protocol; they are not
wired in yet because their scale makes them impractical to ingest inside this
assessment window, and the real-world evaluation level they serve is a Phase 3
task.
"""

from __future__ import annotations

from pathlib import Path

from vyra_ml.ingest.base import SourceAdapter, SourceImage
from vyra_ml.ingest.bsds500 import BSDS500Adapter
from vyra_ml.ingest.skimage_source import SkimageAdapter

_ADAPTERS: dict[str, type[SourceAdapter]] = {
    BSDS500Adapter.name: BSDS500Adapter,
    SkimageAdapter.name: SkimageAdapter,
}


def get_adapter(name: str, raw_dir: Path) -> SourceAdapter:
    try:
        return _ADAPTERS[name](raw_dir)
    except KeyError:
        raise KeyError(f"Unknown source {name!r}. Known: {sorted(_ADAPTERS)}") from None


__all__ = ["SourceAdapter", "SourceImage", "get_adapter"]
