"""The real VYRA analyzer and the full analysis pipeline.

These tests need the model bundle (ml/artifacts/vyra-quality-model-v1) and the
`vyra_ml` package; they skip cleanly if the bundle has not been built.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.analysis.vyra_analyzer import VyraAnalyzer
from tests.conftest import BUNDLE_DIR, _upload, make_test_jpeg

pytestmark = pytest.mark.skipif(
    not (BUNDLE_DIR / "bundle.json").is_file(), reason="model bundle not built"
)


def test_analyzer_loads_and_reports_version() -> None:
    analyzer = VyraAnalyzer.from_path(BUNDLE_DIR)
    assert analyzer.model_version == "vyra-quality-model-v1"
    assert analyzer.description["feature_version"] == "cvfeat-v2"


def test_missing_bundle_fails_loudly(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        VyraAnalyzer.from_path(tmp_path)


async def test_analyze_is_deterministic() -> None:
    analyzer = VyraAnalyzer.from_path(BUNDLE_DIR)
    data = make_test_jpeg("blurry")
    a = (await analyzer.analyze(data, content_type="image/jpeg")).model_dump()
    b = (await analyzer.analyze(data, content_type="image/jpeg")).model_dump()
    # Wall-clock timings are the only thing allowed to vary run to run.
    for d in (a, b):
        d["explanation"].pop("timings_ms", None)
    assert a == b


async def test_analyze_rejects_undecodable_bytes() -> None:
    from app.core.errors import InvalidImageError

    analyzer = VyraAnalyzer.from_path(BUNDLE_DIR)
    with pytest.raises(InvalidImageError):
        await analyzer.analyze(b"\xff\xd8\xff not really a jpeg", content_type="image/jpeg")


async def test_blurry_image_flags_blur() -> None:
    analyzer = VyraAnalyzer.from_path(BUNDLE_DIR)
    outcome = await analyzer.analyze(make_test_jpeg("blurry"), content_type="image/jpeg")
    assert 0.0 <= outcome.quality_score <= 100.0
    assert "blur" in {i.type for i in outcome.issues}
    blur = next(i for i in outcome.issues if i.type == "blur")
    assert blur.validation == "real-world"
    assert 0.0 <= blur.confidence <= 1.0
    assert outcome.explanation["evidence"]


async def test_health_reports_the_loaded_model(analyzer_client: AsyncClient) -> None:
    body = (await analyzer_client.get("/health")).json()
    assert body["components"]["analyzer"]["status"] == "ok"
    assert body["analyzer_model_version"] == "vyra-quality-model-v1"


async def test_upload_analyze_persist_retrieve(analyzer_client: AsyncClient) -> None:
    created = await analyzer_client.post("/api/v1/analyses", files=_upload("dark"))
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["status"] == "completed"
    assert body["model_version"] == "vyra-quality-model-v1"
    assert body["quality_label"] in {"GOOD", "ACCEPTABLE", "DEGRADED", "POOR"}
    assert body["image"]["width"] and body["image"]["height"]
    analysis_id = body["id"]

    fetched = await analyzer_client.get(f"/api/v1/analyses/{analysis_id}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == analysis_id

    listed = await analyzer_client.get("/api/v1/analyses")
    assert listed.json()["total"] == 1
    assert listed.json()["items"][0]["id"] == analysis_id


async def test_garbage_upload_is_rejected_and_not_persisted(analyzer_client: AsyncClient) -> None:
    resp = await analyzer_client.post(
        "/api/v1/analyses", files={"file": ("x.jpg", b"\xff\xd8\xffnope", "image/jpeg")}
    )
    assert resp.status_code == 422
    assert (await analyzer_client.get("/api/v1/analyses")).json()["total"] == 0


async def test_analyzer_failure_leaves_no_row_or_blob(
    analyzer_client: AsyncClient, monkeypatch
) -> None:
    from app.analysis.vyra_analyzer import VyraAnalyzer as _VA

    async def boom(self, image_bytes, *, content_type):  # noqa: ANN001
        raise RuntimeError("analyzer exploded")

    monkeypatch.setattr(_VA, "analyze", boom)
    resp = await analyzer_client.post("/api/v1/analyses", files=_upload("clean"))
    assert resp.status_code == 500
    assert (await analyzer_client.get("/api/v1/analyses")).json()["total"] == 0


async def test_failed_persist_removes_the_stored_blob(
    analyzer_settings, monkeypatch, tmp_path
) -> None:
    """If the DB insert fails after the blob is written, the blob is cleaned up."""
    from app.main import create_app
    from app.repositories.analysis_repository import AnalysisRepository

    async def boom_add(self, analysis):  # noqa: ANN001
        raise RuntimeError("insert failed")

    monkeypatch.setattr(AnalysisRepository, "add", boom_add)
    app = create_app(analyzer_settings)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
        ) as c,
    ):
        resp = await c.post("/api/v1/analyses", files=_upload("clean"))
        assert resp.status_code == 500
    uploads = analyzer_settings.storage_local_dir
    stored = list(uploads.rglob("*")) if uploads.exists() else []
    assert [p for p in stored if p.is_file()] == []
