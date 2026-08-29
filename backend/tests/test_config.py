from __future__ import annotations

import pytest

from app.core.config import Settings


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"environment": "test", "model_path": None}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_database_url_assembled_from_parts_when_unset() -> None:
    s = _settings(db_host="db", db_user="u", db_password="p", db_name="vyra", db_port=5432)
    assert s.database_url == "postgresql+asyncpg://u:p@db:5432/vyra"


@pytest.mark.parametrize(
    "given",
    [
        "postgres://u:p@host:5432/vyra",
        "postgresql://u:p@host:5432/vyra",
    ],
)
def test_platform_postgres_url_is_normalised_to_asyncpg(given: str) -> None:
    s = _settings(database_url=given)
    assert s.database_url == "postgresql+asyncpg://u:p@host:5432/vyra"


def test_explicit_asyncpg_url_is_left_alone() -> None:
    url = "postgresql+asyncpg://u:p@host:5432/vyra"
    assert _settings(database_url=url).database_url == url


def test_sqlite_test_url_is_left_alone() -> None:
    url = "sqlite+aiosqlite:///./test.db"
    assert _settings(database_url=url).database_url == url
