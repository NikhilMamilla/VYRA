"""Filesystem-backed :class:`~app.storage.base.ObjectStorage`.

Suitable for local development and single-container deployments backed by a
Docker volume. Blocking file I/O is pushed to a worker thread so it never stalls
the event loop.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import anyio.to_thread

from app.core.errors import NotFoundError, StorageError


class LocalObjectStorage:
    def __init__(self, root: Path) -> None:
        self._root = root

    def _resolve(self, key: str) -> Path:
        # Reject traversal: keys are relative POSIX paths and must stay under root.
        pure = PurePosixPath(key)
        if pure.is_absolute() or ".." in pure.parts:
            raise StorageError(f"Invalid storage key: {key!r}")
        return self._root / Path(*pure.parts)

    async def save(self, key: str, data: bytes, *, content_type: str) -> None:
        path = self._resolve(key)

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename so a reader never observes a partial object.
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_bytes(data)
            temp.replace(path)

        try:
            await anyio.to_thread.run_sync(_write)
        except OSError as exc:
            raise StorageError(f"Could not write object {key!r}: {exc}") from exc

    async def load(self, key: str) -> bytes:
        path = self._resolve(key)
        try:
            return await anyio.to_thread.run_sync(path.read_bytes)
        except FileNotFoundError as exc:
            raise NotFoundError(f"No stored object for key {key!r}") from exc
        except OSError as exc:
            raise StorageError(f"Could not read object {key!r}: {exc}") from exc

    async def delete(self, key: str) -> None:
        path = self._resolve(key)
        try:
            await anyio.to_thread.run_sync(lambda: path.unlink(missing_ok=True))
        except OSError as exc:
            raise StorageError(f"Could not delete object {key!r}: {exc}") from exc

    async def exists(self, key: str) -> bool:
        return await anyio.to_thread.run_sync(self._resolve(key).is_file)

    async def health_check(self) -> None:
        def _probe() -> None:
            self._root.mkdir(parents=True, exist_ok=True)
            probe = self._root / ".write-probe"
            probe.write_bytes(b"")
            probe.unlink(missing_ok=True)

        try:
            await anyio.to_thread.run_sync(_probe)
        except OSError as exc:
            raise StorageError(f"Storage directory {self._root} is not writable: {exc}") from exc
