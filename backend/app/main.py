"""Application entrypoint and composition root.

Everything the app depends on is constructed here and stored on ``app.state``,
which keeps the rest of the codebase free of import-time side effects and makes
the whole application constructible in a test with different settings.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.analysis.registry import load_analyzer
from app.api.health import router as health_router
from app.api.v1.router import api_router
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import RequestContextMiddleware
from app.db.session import create_all, create_engine, create_session_factory
from app.storage.factory import create_storage

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_engine(settings)
        app.state.settings = settings
        app.state.engine = engine
        app.state.session_factory = create_session_factory(engine)
        app.state.storage = create_storage(settings)
        app.state.analyzer = load_analyzer(settings)

        if settings.database_auto_create:
            try:
                await create_all(engine)
            except Exception:
                # Starting without a database is survivable: /health reports the
                # failure instead of the container crash-looping on a slow db.
                logger.exception("Could not create database tables at startup.")

        logger.info(
            "%s %s started in %s mode",
            settings.project_name,
            settings.version,
            settings.environment,
        )
        try:
            yield
        finally:
            await engine.dispose()

    app = FastAPI(
        title=settings.project_name,
        version=settings.version,
        description="AI-powered image quality and defect detection.",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    register_exception_handlers(app)

    app.include_router(health_router)
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    return app


app = create_app()
