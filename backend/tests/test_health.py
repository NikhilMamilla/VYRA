from __future__ import annotations

from httpx import AsyncClient


async def test_health_reports_ok_with_component_detail(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["environment"] == "test"
    assert body["components"]["database"]["status"] == "ok"
    assert body["components"]["storage"]["status"] == "ok"
    # Phase 1 ships no model; health must say so rather than claim readiness.
    assert body["components"]["analyzer"]["status"] == "not_configured"


async def test_health_echoes_request_id(client: AsyncClient) -> None:
    response = await client.get("/health", headers={"X-Request-ID": "abc123"})

    assert response.headers["X-Request-ID"] == "abc123"
