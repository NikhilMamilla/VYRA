"""Test fixtures.

The suite runs against a temporary SQLite database and a temporary storage
directory, so `pytest` needs no Docker, no Postgres and no credentials.
"""

from __future__ import annotations

import base64
import io
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.main import create_app

# A real, decodable 1x1 PNG -- the smallest input that is genuinely an image.
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGA"
    "hKmMIQAAAABJRU5ErkJggg=="
)

# Repo model bundle -- present after ml/scripts/export_inference_bundle.py has run.
BUNDLE_DIR = Path(__file__).resolve().parents[2] / "ml" / "artifacts" / "vyra-quality-model-v1"


def make_test_jpeg(kind: str = "clean", size: tuple[int, int] = (480, 640)) -> bytes:
    """Synthesise a small JPEG with a known dominant quality issue.

    Used by the analyzer/API tests so they need no fixture image files.
    """
    import cv2
    import numpy as np

    h, w = size
    rng = np.random.default_rng(0)
    base = np.tile(np.linspace(30, 220, w, dtype=np.float32), (h, 1))
    base = cv2.merge([base, np.roll(base, 40, axis=1), np.roll(base, -40, axis=1)])
    # A bit of structure so "clean" genuinely looks sharp.
    cv2.rectangle(base, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (255, 255, 255), 3)
    for i in range(0, w, 24):
        cv2.line(base, (i, 0), (i, h), (0, 0, 0), 1)
    img = base + rng.normal(0, 2.0, base.shape).astype(np.float32)

    if kind == "blurry":
        img = cv2.GaussianBlur(img, (0, 0), 6.0)
    elif kind == "dark":
        img = img * 0.18
    elif kind == "bright":
        img = np.clip(img * 2.6, 0, 255)
    elif kind == "noisy":
        img = img + rng.normal(0, 32.0, img.shape).astype(np.float32)

    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return buf.tobytes()


def _upload(kind: str = "clean") -> dict:
    return {"file": (f"{kind}.jpg", io.BytesIO(make_test_jpeg(kind)), "image/jpeg")}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # model_path is pinned to None here so the base API tests stay hermetic (no
    # ML dependencies, no model file). The analyzer-pipeline tests build their
    # own Settings pointing at the real bundle -- see tests/test_analyzer.py.
    return Settings(
        environment="test",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        storage_local_dir=tmp_path / "uploads",
        log_level="WARNING",
        model_path=None,
    )


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[AsyncClient]:
    app = create_app(settings)
    # httpx does not run ASGI lifespan events; drive them explicitly so the
    # engine, storage and analyzer are wired up exactly as in production.
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as async_client:
            yield async_client


@pytest.fixture
def analyzer_settings(tmp_path: Path) -> Settings:
    """Settings with the real model bundle wired in."""
    if not (BUNDLE_DIR / "bundle.json").is_file():
        pytest.skip(f"model bundle not built ({BUNDLE_DIR}); run export_inference_bundle.py")
    return Settings(
        environment="test",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        storage_local_dir=tmp_path / "uploads",
        log_level="WARNING",
        model_path=BUNDLE_DIR,
        require_analyzer=True,
    )


@pytest.fixture
async def analyzer_client(analyzer_settings: Settings) -> AsyncIterator[AsyncClient]:
    app = create_app(analyzer_settings)
    async with app.router.lifespan_context(app):
        # raise_app_exceptions=False so a deliberately-triggered 500 comes back as
        # a response to assert on, rather than propagating into the test.
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as async_client:
            yield async_client
