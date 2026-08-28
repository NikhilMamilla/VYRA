"""Selects the storage backend named by configuration."""

from __future__ import annotations

from app.core.config import Settings
from app.core.errors import FeatureNotAvailableError
from app.storage.base import ObjectStorage
from app.storage.local import LocalObjectStorage


def create_storage(settings: Settings) -> ObjectStorage:
    if settings.storage_backend == "local":
        return LocalObjectStorage(settings.storage_local_dir)

    # A Supabase Storage adapter implements the same protocol; it is not written
    # yet, and pretending otherwise would fail only once an image was uploaded.
    raise FeatureNotAvailableError(
        f"Storage backend {settings.storage_backend!r} is not implemented yet; "
        "set STORAGE_BACKEND=local."
    )
