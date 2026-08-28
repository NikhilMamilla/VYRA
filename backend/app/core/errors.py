"""Application error types and the handlers that render them.

Every error leaves the API in the same JSON envelope so the frontend has exactly
one shape to parse:

    {"error": {"code": "not_found", "message": "...", "details": {...}}}
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class VyraError(Exception):
    """Base class for errors that map onto a deliberate HTTP response."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"
    message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None, *, details: dict[str, Any] | None = None):
        super().__init__(message or self.message)
        self.message = message or self.message
        self.details = details or {}


class NotFoundError(VyraError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"
    message = "The requested resource was not found."


class InvalidImageError(VyraError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "invalid_image"
    message = "The uploaded file could not be read as an image."


class UnsupportedMediaTypeError(VyraError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    code = "unsupported_media_type"
    message = "The uploaded file type is not supported."


class PayloadTooLargeError(VyraError):
    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    code = "payload_too_large"
    message = "The uploaded file exceeds the maximum allowed size."


class StorageError(VyraError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "storage_error"
    message = "The image could not be stored."


class FeatureNotAvailableError(VyraError):
    """Raised by a seam that exists but has no implementation wired in yet."""

    status_code = status.HTTP_501_NOT_IMPLEMENTED
    code = "not_implemented"
    message = "This capability is not available in the current build."


def _envelope(
    code: str, message: str, details: dict[str, Any] | None, request: Request
) -> dict[str, Any]:
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        body["request_id"] = request_id
    return body


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(VyraError)
    async def _handle_vyra_error(request: Request, exc: VyraError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.exception("Unhandled application error: %s", exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.code, exc.message, exc.details, request),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=_envelope(
                "validation_error",
                "Request validation failed.",
                {"errors": exc.errors()},
                request,
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope("http_error", str(exc.detail), None, request),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # Never leak internals to the client; the request id ties this to the log line.
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_envelope("internal_error", "An unexpected error occurred.", None, request),
        )
