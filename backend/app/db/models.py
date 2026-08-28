"""Database models.

Deliberately minimal for Phase 1. The per-issue analysis payload is stored in a
single JSON column rather than normalised tables: its shape is still being
designed (Phase 2), and JSON lets it evolve without a migration. The columns that
*are* promoted to real columns are the ones we already know we will filter, sort
or list on.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

# JSONB on Postgres (indexable, typed); plain JSON elsewhere (e.g. SQLite in tests).
JsonColumn = JSON().with_variant(JSONB(), "postgresql")


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    # --- Source image --------------------------------------------------------
    image_key: Mapped[str] = mapped_column(String(512), nullable=False)
    image_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    image_content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    image_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    image_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_height: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # --- Outcome -------------------------------------------------------------
    # "completed" | "failed" -- recorded per analysis so a failed run is still
    # retrievable through the history endpoints rather than silently dropped.
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_label: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Full analyzer output: detected issues, per-metric statistics, explanation.
    result: Mapped[dict[str, Any] | None] = mapped_column(JsonColumn, nullable=True)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Analysis id={self.id} status={self.status} score={self.quality_score}>"
