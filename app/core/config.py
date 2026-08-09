"""Application configuration.

Every tunable knob lives here and is sourced from the environment (12-factor).
Secrets are *never* defaulted to usable values in production: :func:`Settings`
validates itself on construction and refuses to boot a production process with
a placeholder secret, a wildcard host list, or SSRF protection disabled.
"""

from __future__ import annotations

import secrets
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# A placeholder that must never survive into a production deployment.
PLACEHOLDER_SECRET = "CHANGE_ME_generate_a_64_byte_random_secret"  # noqa: S105
MIN_SECRET_LENGTH = 32


class Environment(StrEnum):
    """Deployment environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"

    @property
    def is_production(self) -> bool:
        return self is Environment.PRODUCTION


def _split_csv(value: Any) -> Any:
    """Accept ``a,b,c`` strings for list-typed settings (env vars are strings)."""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):  # already JSON
            return value
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    """Typed, validated application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    # ---------------------------------------------------------------- runtime
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    app_name: str = "News Intelligence Platform"
    api_v1_prefix: str = "/api/v1"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["console", "json"] = "console"

    # --------------------------------------------------------------- database
    database_url: str = "sqlite+aiosqlite:///./data/news.db"
    db_pool_size: Annotated[int, Field(ge=1, le=100)] = 10
    db_max_overflow: Annotated[int, Field(ge=0, le=200)] = 20
    db_pool_timeout: Annotated[int, Field(ge=1, le=300)] = 30
    db_echo: bool = False

    # ------------------------------------------------------------------ redis
    redis_url: str | None = None
    cache_enabled: bool = True
    cache_ttl_seconds: Annotated[int, Field(ge=1, le=86_400)] = 60

    # --------------------------------------------------------------- security
    jwt_secret_key: SecretStr = SecretStr(PLACEHOLDER_SECRET)
    jwt_algorithm: Literal["HS256", "HS384", "HS512"] = "HS256"
    access_token_expire_minutes: Annotated[int, Field(ge=1, le=1440)] = 30
    refresh_token_expire_days: Annotated[int, Field(ge=1, le=90)] = 7
    password_min_length: Annotated[int, Field(ge=8, le=128)] = 12
    max_failed_logins: Annotated[int, Field(ge=1, le=100)] = 5
    lockout_minutes: Annotated[int, Field(ge=1, le=1440)] = 15

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8000"])
    trusted_hosts: list[str] = Field(default_factory=lambda: ["*"])
    max_request_bytes: Annotated[int, Field(ge=1024, le=104_857_600)] = 1_048_576
    secure_headers_enabled: bool = True
    hsts_enabled: bool = False

    rate_limit_enabled: bool = True
    rate_limit_requests: Annotated[int, Field(ge=1, le=100_000)] = 120
    rate_limit_window_seconds: Annotated[int, Field(ge=1, le=3600)] = 60
    auth_rate_limit_requests: Annotated[int, Field(ge=1, le=1000)] = 10
    auth_rate_limit_window_seconds: Annotated[int, Field(ge=1, le=3600)] = 60

    bootstrap_admin_email: str = "admin@example.com"
    bootstrap_admin_password: SecretStr | None = None

    # ------------------------------------------------------------------- http
    http_timeout_seconds: Annotated[float, Field(gt=0, le=300)] = 15.0
    http_max_retries: Annotated[int, Field(ge=0, le=10)] = 3
    http_backoff_base: Annotated[float, Field(ge=0, le=30)] = 0.5
    http_max_redirects: Annotated[int, Field(ge=0, le=10)] = 3
    http_max_response_bytes: Annotated[int, Field(ge=1024, le=104_857_600)] = 5_242_880
    http_user_agent: str = "NewsIntelligenceBot/1.0 (+https://example.com/bot)"
    http_max_concurrency: Annotated[int, Field(ge=1, le=64)] = 8
    respect_robots_txt: bool = True
    default_request_delay_seconds: Annotated[float, Field(ge=0, le=60)] = 1.0

    ssrf_protection_enabled: bool = True
    allowed_url_schemes: list[str] = Field(default_factory=lambda: ["http", "https"])
    source_host_allowlist: list[str] = Field(default_factory=list)

    # ---------------------------------------------------------------- sources
    newsapi_key: SecretStr | None = None
    gdelt_enabled: bool = True
    sources_config_path: str = "configs/sources.yaml"

    # ----------------------------------------------------------- intelligence
    ingest_batch_size: Annotated[int, Field(ge=1, le=10_000)] = 100
    dedup_title_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.85
    dedup_content_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.80
    dedup_lookback_hours: Annotated[int, Field(ge=1, le=8760)] = 72
    trend_window_hours: Annotated[int, Field(ge=1, le=720)] = 24
    trend_min_articles: Annotated[int, Field(ge=1, le=1000)] = 3
    event_similarity_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.55
    event_window_hours: Annotated[int, Field(ge=1, le=720)] = 48
    retention_days: Annotated[int, Field(ge=1, le=3650)] = 90

    # ---------------------------------------------------------------- workers
    worker_concurrency: Annotated[int, Field(ge=1, le=64)] = 4
    schedule_ingest_seconds: Annotated[int, Field(ge=10)] = 300
    schedule_process_seconds: Annotated[int, Field(ge=10)] = 900
    schedule_trends_seconds: Annotated[int, Field(ge=10)] = 1800
    schedule_events_seconds: Annotated[int, Field(ge=10)] = 3600
    schedule_cleanup_seconds: Annotated[int, Field(ge=60)] = 86_400

    # ----------------------------------------------------------------- alerts
    alerts_enabled: bool = True
    alert_webhook_timeout_seconds: Annotated[float, Field(gt=0, le=120)] = 10.0
    smtp_host: str | None = None
    smtp_port: Annotated[int, Field(ge=1, le=65_535)] = 587
    smtp_username: str | None = None
    smtp_password: SecretStr | None = None
    smtp_from: str = "news-intel@example.com"
    smtp_use_tls: bool = True

    # ------------------------------------------------------------- validators
    _split_lists = field_validator(
        "cors_origins",
        "trusted_hosts",
        "allowed_url_schemes",
        "source_host_allowlist",
        mode="before",
    )(_split_csv)

    @field_validator("allowed_url_schemes")
    @classmethod
    def _validate_schemes(cls, value: list[str]) -> list[str]:
        allowed = {"http", "https"}
        normalised = [scheme.lower().strip() for scheme in value]
        invalid = set(normalised) - allowed
        if invalid:
            raise ValueError(f"unsupported URL scheme(s): {sorted(invalid)}; only http/https")
        return normalised or ["https"]

    @field_validator("database_url")
    @classmethod
    def _validate_database_url(cls, value: str) -> str:
        supported = ("sqlite+aiosqlite://", "postgresql+asyncpg://", "postgresql+psycopg://")
        if not value.startswith(supported):
            raise ValueError(
                "DATABASE_URL must use an async driver: "
                "sqlite+aiosqlite://, postgresql+asyncpg:// or postgresql+psycopg://"
            )
        return value

    @model_validator(mode="after")
    def _enforce_production_hardening(self) -> Settings:
        """Fail fast rather than run a production process with unsafe defaults."""
        if not self.environment.is_production:
            return self

        secret = self.jwt_secret_key.get_secret_value()
        problems: list[str] = []
        if secret == PLACEHOLDER_SECRET or len(secret) < MIN_SECRET_LENGTH:
            problems.append(
                f"JWT_SECRET_KEY must be a random value of at least {MIN_SECRET_LENGTH} characters"
            )
        if self.debug:
            problems.append("DEBUG must be false in production")
        if "*" in self.cors_origins:
            problems.append("CORS_ORIGINS must not contain '*' in production")
        if "*" in self.trusted_hosts:
            problems.append("TRUSTED_HOSTS must list explicit hostnames in production")
        if not self.ssrf_protection_enabled:
            problems.append("SSRF_PROTECTION_ENABLED must stay true in production")
        if not self.rate_limit_enabled:
            problems.append("RATE_LIMIT_ENABLED must stay true in production")
        if self.database_url.startswith("sqlite"):
            problems.append("SQLite is not supported in production; use PostgreSQL")
        if problems:
            raise ValueError("insecure production configuration:\n  - " + "\n  - ".join(problems))
        return self

    # ------------------------------------------------------------- helpers
    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def testing(self) -> bool:
        return self.environment is Environment.TEST

    def sqlite_path(self) -> Path | None:
        """Filesystem path backing a SQLite URL, if any (``:memory:`` -> ``None``)."""
        if not self.is_sqlite:
            return None
        raw = self.database_url.split("///", 1)[-1]
        if not raw or raw.startswith(":memory:"):
            return None
        path = Path(raw)
        return path if path.is_absolute() else (PROJECT_ROOT / path)

    def generate_secret(self) -> str:  # pragma: no cover - convenience helper
        """Return a fresh random secret suitable for ``JWT_SECRET_KEY``."""
        return secrets.token_urlsafe(64)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (cached; call ``.cache_clear()`` in tests)."""
    return Settings()


settings = get_settings()
