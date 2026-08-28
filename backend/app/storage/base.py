"""Object storage abstraction.

The rest of the application only ever sees this protocol, so swapping the local
filesystem for Supabase Storage (or S3) is a configuration change, not a code
change. Crucially, the CV/ML layer never touches storage at all -- it is handed
image bytes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ObjectStorage(Protocol):
    """A content-addressable blob store keyed by an opaque string."""

    async def save(self, key: str, data: bytes, *, content_type: str) -> None:
        """Persist ``data`` under ``key``, overwriting any existing object."""

    async def load(self, key: str) -> bytes:
        """Return the bytes stored under ``key``. Raises ``NotFoundError`` if absent."""

    async def delete(self, key: str) -> None:
        """Remove ``key``. Deleting a missing key is not an error."""

    async def exists(self, key: str) -> bool:
        """Whether ``key`` currently holds an object."""

    async def health_check(self) -> None:
        """Raise ``StorageError`` if the backend is not usable."""
