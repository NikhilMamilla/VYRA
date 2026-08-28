from __future__ import annotations

from pathlib import Path

import pytest

from app.core.errors import NotFoundError, StorageError
from app.storage.local import LocalObjectStorage


async def test_roundtrip(tmp_path: Path) -> None:
    storage = LocalObjectStorage(tmp_path)

    await storage.save("2026/01/01/abc.png", b"payload", content_type="image/png")

    assert await storage.exists("2026/01/01/abc.png")
    assert await storage.load("2026/01/01/abc.png") == b"payload"

    await storage.delete("2026/01/01/abc.png")
    assert not await storage.exists("2026/01/01/abc.png")


async def test_deleting_a_missing_key_is_not_an_error(tmp_path: Path) -> None:
    await LocalObjectStorage(tmp_path).delete("nope.png")


async def test_loading_a_missing_key_raises_not_found(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError):
        await LocalObjectStorage(tmp_path).load("nope.png")


@pytest.mark.parametrize("key", ["../escape.png", "/etc/passwd", "a/../../b.png"])
async def test_path_traversal_is_refused(tmp_path: Path, key: str) -> None:
    with pytest.raises(StorageError):
        await LocalObjectStorage(tmp_path).save(key, b"x", content_type="image/png")


async def test_health_check_creates_the_root(tmp_path: Path) -> None:
    root = tmp_path / "does-not-exist-yet"

    await LocalObjectStorage(root).health_check()

    assert root.is_dir()
