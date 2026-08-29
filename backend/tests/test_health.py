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


async def test_metrics_reports_request_counters(client: AsyncClient) -> None:
    from app.core.metrics import METRICS

    METRICS.reset()
    await client.get("/health")

    response = await client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "VYRA"
    # The /health call above is counted; this /metrics call is still in flight.
    assert body["requests_total"] >= 1
    assert body["requests_by_status_class"].get("2xx", 0) >= 1
    assert set(body["latency_ms"]) == {"window", "p50", "p95", "p99", "max"}
    assert body["error_rate"] == 0.0
