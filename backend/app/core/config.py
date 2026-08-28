"""Application configuration.

All settings are sourced from environment variables (or a local ``.env`` file).
Nothing here may contain a real credential -- see ``.env.example`` at the repo root.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app import __version__

Environment = Literal["development", "test", "production"]
StorageBackendName = Literal["local", "supabase"]

# Repository-root-relative default so the app behaves the same however it is launched.
BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(BACKEND_ROOT / ".env", BACKEND_ROOT.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application ---------------------------------------------------------
    project_name: str = "VYRA"
    version: str = __version__
    environment: Environment = "development"
    debug: bool = False
    log_level: str = "INFO"
    log_json: bool = False
    api_v1_prefix: str = "/api/v1"

    # Comma-separated. Kept as a plain string because pydantic-settings expects
    # JSON for list-typed env vars, which is awkward to write in a .env file.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- Database ------------------------------------------------------------
    # Set DATABASE_URL to any async SQLAlchemy URL to use it verbatim (Supabase
    # Postgres works unchanged, e.g.
    # postgresql+asyncpg://postgres:<pw>@db.<ref>.supabase.co:5432/postgres).
    # When it is unset the URL is assembled from the DB_* parts below, whose
    # defaults target the local docker-compose Postgres container (which has no
    # published port -- it is only reachable inside the compose network).
    database_url: str | None = None
    db_user: str = "vyra"
    # Local placeholder for the unexposed docker-compose Postgres; override in .env.
    db_password: str = "change-me-local-only"  # noqa: S105
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "vyra"
    database_echo: bool = False
    # Phase 1 convenience: create tables at startup. Replaced by migrations once
    # the analysis result schema stabilises (see docs/architecture.md).
    database_auto_create: bool = True

    # --- Storage -------------------------------------------------------------
    storage_backend: StorageBackendName = "local"
    storage_local_dir: Path = BACKEND_ROOT / "data" / "uploads"
    supabase_url: str | None = None
    supabase_key: str | None = None
    supabase_bucket: str = "vyra-images"

    # --- Uploads -------------------------------------------------------------
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)

    # --- ML ------------------------------------------------------------------
    # Directory of the VYRA inference bundle (model.joblib + calibrators.joblib +
    # defect_detector.json + bundle.json), produced by
    # ml/scripts/export_inference_bundle.py. When unset the API reports the
    # analyzer as not configured and POST /analyses returns 501.
    model_path: Path | None = Field(
        default=BACKEND_ROOT.parent / "ml" / "artifacts" / "vyra-quality-model-v1"
    )
    # If true, a MODEL_PATH that fails to load aborts startup instead of running
    # the API without analysis. Recommended in production/Docker.
    require_analyzer: bool = False

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @model_validator(mode="after")
    def _assemble_database_url(self) -> Settings:
        """Build DATABASE_URL from the DB_* parts when it is not given explicitly."""
        if not self.database_url:
            self.database_url = (
                f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
                f"@{self.db_host}:{self.db_port}/{self.db_name}"
            )
        return self

    @model_validator(mode="after")
    def _check_backend_credentials(self) -> Settings:
        if self.storage_backend == "supabase" and not (self.supabase_url and self.supabase_key):
            raise ValueError(
                "STORAGE_BACKEND=supabase requires SUPABASE_URL and SUPABASE_KEY to be set"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance. Clear the cache in tests via ``get_settings.cache_clear()``."""
    return Settings()
