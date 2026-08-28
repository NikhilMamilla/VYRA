"""Health/status response shapes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ComponentStatus = Literal["ok", "unavailable", "not_configured"]
ServiceStatus = Literal["ok", "degraded"]


class ComponentHealth(BaseModel):
    status: ComponentStatus
    detail: str | None = None
    latency_ms: float | None = None


class HealthResponse(BaseModel):
    status: ServiceStatus
    version: str
    environment: str
    uptime_seconds: float
    analyzer_model_version: str | None = None
    components: dict[str, ComponentHealth]
