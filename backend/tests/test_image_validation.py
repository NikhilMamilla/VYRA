from __future__ import annotations

import pytest

from app.core.errors import InvalidImageError, PayloadTooLargeError, UnsupportedMediaTypeError
from app.services.image_validation import validate_upload
from tests.conftest import PNG_1X1

MAX = 1024 * 1024


def test_detects_format_from_content() -> None:
    result = validate_upload(PNG_1X1, declared_media_type="image/png", max_bytes=MAX)

    assert result.media_type == "image/png"
    assert result.extension == "png"
    assert result.size_bytes == len(PNG_1X1)


def test_content_wins_over_a_lying_header() -> None:
    """A JPEG announced as a PNG is still recognised as a JPEG."""
    jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 32

    result = validate_upload(jpeg, declared_media_type="image/png", max_bytes=MAX)

    assert result.media_type == "image/jpeg"


def test_charset_suffix_on_content_type_is_tolerated() -> None:
    result = validate_upload(
        PNG_1X1, declared_media_type="image/png; charset=binary", max_bytes=MAX
    )

    assert result.media_type == "image/png"


def test_missing_content_type_falls_back_to_sniffing() -> None:
    result = validate_upload(PNG_1X1, declared_media_type=None, max_bytes=MAX)

    assert result.media_type == "image/png"


def test_empty_upload_is_invalid() -> None:
    with pytest.raises(InvalidImageError):
        validate_upload(b"", declared_media_type="image/png", max_bytes=MAX)


def test_unknown_signature_is_invalid() -> None:
    with pytest.raises(InvalidImageError):
        validate_upload(b"GIF89a" + b"\x00" * 16, declared_media_type=None, max_bytes=MAX)


def test_unsupported_declared_type_is_rejected() -> None:
    with pytest.raises(UnsupportedMediaTypeError):
        validate_upload(PNG_1X1, declared_media_type="application/pdf", max_bytes=MAX)


def test_oversized_upload_is_rejected() -> None:
    with pytest.raises(PayloadTooLargeError):
        validate_upload(PNG_1X1, declared_media_type="image/png", max_bytes=len(PNG_1X1) - 1)
