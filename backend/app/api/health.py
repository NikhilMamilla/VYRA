"""Service health.

Unversioned and mounted at ``/health`` because container orchestrators and
uptime checks should not have to track the API version. It reports each
dependency separately so a failure is diagnosable from the response alone.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from app.api.deps import AnalyzerDep, SessionDep, SettingsDep, StorageDep
from app.core.metrics import METRICS
from app.schemas.health import ComponentHealth, HealthResponse

router = APIRouter(tags=["health"])

_STARTED_AT = time.monotonic()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health and dependency status",
    responses={503: {"description": "A critical dependency is unavailable"}},
)
async def health(
    response: Response,
    session: SessionDep,
    storage: StorageDep,
    settings: SettingsDep,
    analyzer: AnalyzerDep,
) -> HealthResponse:
    components: dict[str, ComponentHealth] = {}

    started = time.perf_counter()
    try:
        await session.execute(text("SELECT 1"))
    except Exception as exc:
        components["database"] = ComponentHealth(status="unavailable", detail=str(exc))
    else:
        components["database"] = ComponentHealth(
            status="ok", latency_ms=round((time.perf_counter() - started) * 1000, 2)
        )

    try:
        await storage.health_check()
    except Exception as exc:
        components["storage"] = ComponentHealth(status="unavailable", detail=str(exc))
    else:
        components["storage"] = ComponentHealth(status="ok")

    # The analyzer can legitimately be absent (no MODEL_PATH), so its absence is
    # reported without failing the check -- the API still serves history.
    model_version = getattr(analyzer, "model_version", None)
    components["analyzer"] = (
        ComponentHealth(status="ok", detail=f"model {model_version}")
        if analyzer is not None
        else ComponentHealth(status="not_configured", detail="No analysis model is loaded.")
    )

    critical_ok = all(components[name].status == "ok" for name in ("database", "storage"))
    if not critical_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if critical_ok else "degraded",
        version=settings.version,
        environment=settings.environment,
        uptime_seconds=round(time.monotonic() - _STARTED_AT, 2),
        analyzer_model_version=model_version,
        components=components,
    )


@router.get(
    "/metrics",
    summary="Process-level runtime metrics",
    description=(
        "Request counts, error rate and latency percentiles for this worker "
        "process, as JSON. Dependency-free: it answers even when the database "
        "is down. Counters reset on restart."
    ),
    tags=["health"],
)
async def metrics(settings: SettingsDep) -> dict[str, object]:
    return {
        "service": settings.project_name,
        "version": settings.version,
        "environment": settings.environment,
        **METRICS.snapshot(),
    }
