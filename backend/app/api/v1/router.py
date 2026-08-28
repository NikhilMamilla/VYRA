"""Aggregates every v1 route module into one router."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.routes import analyses

api_router = APIRouter()
api_router.include_router(analyses.router)
