from __future__ import annotations

import uuid

from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.main import create_app
from tests.conftest import PNG_1X1


async def test_history_is_empty_but_functional(client: AsyncClient) -> None:
    response = await client.get("/api/v1/analyses")

    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 0, "limit": 20, "offset": 0}


async def test_unknown_analysis_returns_404_envelope(client: AsyncClient) -> None:
    response = await client.get(f"/api/v1/analyses/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_malformed_id_is_a_validation_error(client: AsyncClient) -> None:
    response = await client.get("/api/v1/analyses/not-a-uuid")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_valid_image_is_accepted_but_analysis_is_unavailable(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/analyses", files={"file": ("tiny.png", PNG_1X1, "image/png")}
    )

    assert response.status_code == 501
    assert response.json()["error"]["code"] == "not_implemented"


async def test_non_image_upload_is_rejected(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/analyses", files={"file": ("notes.png", b"this is not a png", "image/png")}
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_image"


async def test_unsupported_content_type_is_rejected(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/analyses", files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")}
    )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


async def test_oversized_upload_is_rejected(settings: Settings) -> None:
    small = settings.model_copy(update={"max_upload_bytes": 1024})
    app = create_app(small)

    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post(
            "/api/v1/analyses",
            files={"file": ("big.png", b"\x89PNG\r\n\x1a\n" + b"0" * 4096, "image/png")},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
