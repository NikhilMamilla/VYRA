"""FastAPI dependencies.

Long-lived objects (engine, session factory, storage, analyzer) are built once at
startup and parked on ``app.state``; these helpers hand them to request handlers
so nothing reaches for a module-level global.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.contract import QualityAnalyzer
from app.core.config import Settings
from app.db.session import session_scope
from app.repositories.analysis_repository import AnalysisRepository
from app.services.analysis_service import AnalysisService
from app.storage.base import ObjectStorage


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_storage(request: Request) -> ObjectStorage:
    return request.app.state.storage


def get_analyzer(request: Request) -> QualityAnalyzer | None:
    return request.app.state.analyzer


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    async for session in session_scope(request.app.state.session_factory):
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
StorageDep = Annotated[ObjectStorage, Depends(get_storage)]
AnalyzerDep = Annotated["QualityAnalyzer | None", Depends(get_analyzer)]


def get_analysis_service(
    session: SessionDep,
    settings: SettingsDep,
    storage: StorageDep,
    analyzer: AnalyzerDep,
) -> AnalysisService:
    return AnalysisService(
        repository=AnalysisRepository(session),
        storage=storage,
        settings=settings,
        analyzer=analyzer,
    )


AnalysisServiceDep = Annotated[AnalysisService, Depends(get_analysis_service)]
