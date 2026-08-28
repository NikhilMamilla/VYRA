"""Upload validation.

Runs before anything is stored or analyzed. The declared ``Content-Type`` of an
upload is attacker-controlled, so the format is confirmed from the file's own
magic bytes; the declared type is only used to reject obvious mismatches early.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.errors import InvalidImageError, PayloadTooLargeError, UnsupportedMediaTypeError

# Formats the eventual CV pipeline can decode.
SUPPORTED_MEDIA_TYPES: frozenset[str] = frozenset(
    {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
)

_EXTENSIONS = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
}


@dataclass(frozen=True)
class ValidatedImage:
    """An upload that is safe to hand to storage and to the analyzer."""

    data: bytes
    media_type: str
    """Media type detected from the file contents, not from the request header."""
    size_bytes: int
    extension: str


def _sniff_media_type(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    return None


def validate_upload(
    data: bytes,
    *,
    declared_media_type: str | None,
    max_bytes: int,
) -> ValidatedImage:
    """Validate an uploaded file, raising the matching :class:`VyraError` on failure."""
    if not data:
        raise InvalidImageError("The uploaded file is empty.")

    if len(data) > max_bytes:
        raise PayloadTooLargeError(
            f"The uploaded file is {len(data)} bytes; the maximum is {max_bytes} bytes.",
            details={"size_bytes": len(data), "max_bytes": max_bytes},
        )

    if declared_media_type and declared_media_type.split(";")[0].strip() not in (
        SUPPORTED_MEDIA_TYPES
    ):
        raise UnsupportedMediaTypeError(
            f"Content type {declared_media_type!r} is not supported.",
            details={"supported": sorted(SUPPORTED_MEDIA_TYPES)},
        )

    media_type = _sniff_media_type(data)
    if media_type is None:
        raise InvalidImageError(
            "The uploaded file is not a recognised image format.",
            details={"supported": sorted(SUPPORTED_MEDIA_TYPES)},
        )

    return ValidatedImage(
        data=data,
        media_type=media_type,
        size_bytes=len(data),
        extension=_EXTENSIONS[media_type],
    )
