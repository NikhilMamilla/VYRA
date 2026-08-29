"""Orchestration for image-quality analyses.

The request pipeline is::

    upload -> validate -> analyze (CV/ML) -> store image -> persist result

Analysis runs *before* anything is written: an image we cannot analyze is never
stored, and if persistence fails the just-stored blob is removed, so a failed
request never leaves an orphan file or a half-written row.
"""

from __future__ import annotations

import logging
import uuid

from app.analysis.contract import QualityAnalyzer
from app.core.config import Settings
from app.core.errors import FeatureNotAvailableError, NotFoundError
from app.db.models import Analysis
from app.repositories.analysis_repository import AnalysisRepository
from app.schemas.analysis import AnalysisOutcome, AnalysisRead, ImageInfo, Issue
from app.services.image_validation import ValidatedImage, validate_upload
from app.storage.base import ObjectStorage

logger = logging.getLogger(__name__)


def to_read_model(analysis: Analysis) -> AnalysisRead:
    """Project a stored row onto the public API shape."""
    result = analysis.result or {}
    return AnalysisRead(
        id=analysis.id,
        created_at=analysis.created_at,
        status=analysis.status,  # type: ignore[arg-type]
        image=ImageInfo(
            filename=analysis.image_filename,
            content_type=analysis.image_content_type,
            size_bytes=analysis.image_size_bytes,
            width=analysis.image_width,
            height=analysis.image_height,
        ),
        quality_score=analysis.quality_score,
        quality_label=analysis.quality_label,  # type: ignore[arg-type]
        model_version=analysis.model_version,
        issues=[Issue.model_validate(issue) for issue in result.get("issues", [])],
        metrics=result.get("metrics", {}),
        explanation=result.get("explanation", {}),
        error_message=analysis.error_message,
    )


class AnalysisService:
    def __init__(
        self,
        *,
        repository: AnalysisRepository,
        storage: ObjectStorage,
        settings: Settings,
        analyzer: QualityAnalyzer | None,
    ) -> None:
        self._repository = repository
        self._storage = storage
        self._settings = settings
        self._analyzer = analyzer

    @property
    def analyzer_available(self) -> bool:
        return self._analyzer is not None

    def validate(self, data: bytes, *, declared_media_type: str | None) -> ValidatedImage:
        return validate_upload(
            data,
            declared_media_type=declared_media_type,
            max_bytes=self._settings.max_upload_bytes,
        )

    async def create_analysis(
        self, data: bytes, *, filename: str, declared_media_type: str | None
    ) -> AnalysisRead:
        """Validate, analyze, store and persist one upload."""
        validated = self.validate(data, declared_media_type=declared_media_type)

        if self._analyzer is None:
            raise FeatureNotAvailableError(
                "Image analysis is not available in this build. The upload was "
                "validated successfully but no analysis model is loaded.",
                details={"filename": filename},
            )

        # Analyze first: InvalidImageError -> 422, any other failure -> 500, and
        # in both cases nothing has been written yet.
        outcome: AnalysisOutcome = await self._analyzer.analyze(
            validated.data, content_type=validated.media_type
        )

        analysis_id = uuid.uuid4()
        image_key = f"{analysis_id}/original.{validated.extension}"
        await self._storage.save(image_key, validated.data, content_type=validated.media_type)

        try:
            row = await self._repository.add(
                Analysis(
                    id=analysis_id,
                    image_key=image_key,
                    image_filename=filename[:255],
                    image_content_type=validated.media_type,
                    image_size_bytes=validated.size_bytes,
                    image_width=outcome.image_width,
                    image_height=outcome.image_height,
                    status="completed",
                    quality_score=outcome.quality_score,
                    quality_label=outcome.quality_label,
                    model_version=outcome.model_version,
                    result={
                        "issues": [issue.model_dump() for issue in outcome.issues],
                        "metrics": outcome.metrics,
                        "explanation": outcome.explanation,
                    },
                )
            )
        except Exception:
            # Roll back the blob we just wrote so a failed insert leaves nothing.
            await self._safe_delete(image_key)
            raise

        return to_read_model(row)

    async def _safe_delete(self, key: str) -> None:
        try:
            await self._storage.delete(key)
        except Exception:  # pragma: no cover - best-effort cleanup
            logger.warning("Could not remove orphaned blob %s after a failed analysis", key)

    async def get_analysis(self, analysis_id: uuid.UUID) -> AnalysisRead:
        analysis = await self._repository.get(analysis_id)
        if analysis is None:
            raise NotFoundError(f"No analysis with id {analysis_id}.")
        return to_read_model(analysis)

    async def list_analyses(self, *, limit: int, offset: int) -> tuple[list[AnalysisRead], int]:
        rows = await self._repository.list(limit=limit, offset=offset)
        total = await self._repository.count()
        return [to_read_model(row) for row in rows], total
