"""The boundary between the web application and the CV/ML pipeline.

This is the only thing the API layer knows about image analysis. An analyzer
receives raw image bytes and returns a structured outcome -- it has no knowledge
of HTTP, storage or the database, so the Phase 2 pipeline (feature extraction +
learned decision component) can be developed and tested entirely on its own.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.schemas.analysis import AnalysisOutcome


@runtime_checkable
class QualityAnalyzer(Protocol):
    """Decides the quality of a single image."""

    @property
    def model_version(self) -> str:
        """Identifier of the loaded model artifact, recorded with every analysis."""

    async def analyze(self, image_bytes: bytes, *, content_type: str) -> AnalysisOutcome:
        """Analyze one image.

        Must raise :class:`app.core.errors.InvalidImageError` if the bytes cannot
        be decoded, so unreadable uploads surface as a 422 rather than a 500.
        """
