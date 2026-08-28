"""Data access for :class:`~app.db.models.Analysis`.

Keeping queries here means the service layer never writes SQL and the API layer
never sees the ORM.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Analysis


class AnalysisRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, analysis: Analysis) -> Analysis:
        self._session.add(analysis)
        await self._session.flush()
        await self._session.refresh(analysis)
        return analysis

    async def get(self, analysis_id: uuid.UUID) -> Analysis | None:
        return await self._session.get(Analysis, analysis_id)

    async def list(self, *, limit: int, offset: int) -> list[Analysis]:
        statement = (
            select(Analysis).order_by(Analysis.created_at.desc()).limit(limit).offset(offset)
        )
        result = await self._session.execute(statement)
        return list(result.scalars().all())

    async def count(self) -> int:
        result = await self._session.execute(select(func.count()).select_from(Analysis))
        return int(result.scalar_one())
