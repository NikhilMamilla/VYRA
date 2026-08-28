"""Shared response shapes."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int = Field(description="Total rows matching the query, ignoring pagination.")
    limit: int
    offset: int


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict | None = None


class ErrorResponse(BaseModel):
    """The envelope every non-2xx response uses."""

    error: ErrorDetail
    request_id: str | None = None
