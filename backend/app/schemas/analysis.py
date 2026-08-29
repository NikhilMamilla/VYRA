"""API contract for image-quality analyses.

The issue vocabulary comes straight from the assessment brief. Everything the
model is still free to shape -- per-metric statistics, explanation payloads --
lives in open ``dict`` fields so Phase 2 can fill it in without breaking clients.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

IssueType = Literal[
    "blur",
    "underexposure",
    "overexposure",
    "noise",
    "corruption",
    "defect",
]
Severity = Literal["low", "medium", "high"]
# How far a given issue's detector has actually been validated. Kept on every
# issue so the UI never presents a synthetic-only or screening signal as if it
# carried the same evidence as a real-world-validated one.
IssueValidation = Literal["real-world", "synthetic-only", "screening"]
QualityLabel = Literal["GOOD", "ACCEPTABLE", "DEGRADED", "POOR"]
AnalysisStatus = Literal["completed", "failed"]


class Issue(BaseModel):
    type: IssueType
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0, description="Calibrated P(issue present).")
    validation: IssueValidation | None = None
    detail: str | None = None


class AnalysisOutcome(BaseModel):
    """What an analyzer returns for one image. Produced by the CV/ML layer."""

    quality_score: float = Field(ge=0.0, le=100.0)
    quality_label: QualityLabel
    issues: list[Issue] = Field(default_factory=list)
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Interpretable image statistics used by the decision component.",
    )
    explanation: dict[str, Any] = Field(
        default_factory=dict,
        description="Model-dependent explainability payload (evidence, defect region, etc.).",
    )
    model_version: str
    image_width: int | None = None
    image_height: int | None = None


class BatchItemError(BaseModel):
    """Why one image in a batch could not be analysed."""

    code: str
    message: str


class ImageInfo(BaseModel):
    filename: str
    content_type: str
    size_bytes: int
    width: int | None = None
    height: int | None = None


class AnalysisRead(BaseModel):
    """A stored analysis as returned by the history endpoints.

    Built from the ORM row by ``app.services.analysis_service.to_read_model``;
    the flat row is deliberately not a 1:1 match for this nested shape.
    """

    id: uuid.UUID
    created_at: datetime
    status: AnalysisStatus
    image: ImageInfo
    quality_score: float | None = None
    quality_label: QualityLabel | None = None
    model_version: str | None = None
    issues: list[Issue] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    explanation: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None


class BatchAnalysisItem(BaseModel):
    """One image's outcome within a batch request."""

    filename: str
    ok: bool
    analysis: AnalysisRead | None = None
    error: BatchItemError | None = None


class BatchAnalysisResponse(BaseModel):
    """Result of ``POST /analyses/batch``.

    Always ``200``: per-image failures are reported in ``items`` rather than
    aborting the request. Successful images are persisted like a single upload.
    """

    total: int
    succeeded: int
    failed: int
    items: list[BatchAnalysisItem]
