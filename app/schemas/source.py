"""Source management schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import Field, field_validator

from app.core.url_safety import is_safe_url
from app.database.models.source import SourceKind, SourceStatus
from app.schemas.common import ORMModel, StrictModel

SLUG_PATTERN = r"^[a-z0-9][a-z0-9_-]{1,62}[a-z0-9]$"
ENV_VAR_PATTERN = r"^[A-Z][A-Z0-9_]{2,63}$"


class SourceBase(StrictModel):
    """Fields shared by create/update payloads."""

    name: Annotated[str, Field(min_length=2, max_length=160)]
    kind: SourceKind = SourceKind.RSS
    url: Annotated[str, Field(min_length=8, max_length=2048)]
    homepage: Annotated[str | None, Field(max_length=2048)] = None
    description: Annotated[str | None, Field(max_length=2000)] = None
    language: Annotated[str | None, Field(max_length=8, pattern=r"^[a-z]{2}(-[A-Za-z]{2})?$")] = (
        None
    )
    country: Annotated[str | None, Field(max_length=8, pattern=r"^[A-Za-z]{2}$")] = None
    category: Annotated[str | None, Field(max_length=48)] = None
    #: Name of the env var holding the API key - never the key itself.
    api_key_env: Annotated[str | None, Field(max_length=64, pattern=ENV_VAR_PATTERN)] = None
    config: dict[str, Any] = Field(default_factory=dict)
    weight: Annotated[float, Field(ge=0.0, le=2.0)] = 1.0
    request_delay_seconds: Annotated[float, Field(ge=0.0, le=60.0)] = 1.0
    max_articles_per_run: Annotated[int, Field(ge=1, le=1000)] = 100
    respect_robots: bool = True
    enabled: bool = True

    @field_validator("url", "homepage")
    @classmethod
    def _url_must_pass_policy(cls, value: str | None) -> str | None:
        """Reject SSRF-unsafe endpoints at the API boundary, not at fetch time."""
        if value is None:
            return None
        if not is_safe_url(value):
            raise ValueError("URL is rejected by the URL safety policy")
        return value

    @field_validator("config")
    @classmethod
    def _config_must_be_shallow(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Bound the connector config so a payload cannot blow up the row."""
        if len(value) > 50:
            raise ValueError("config may contain at most 50 keys")
        serialized = str(value)
        if len(serialized) > 20_000:
            raise ValueError("config is too large")
        for key in value:
            if not isinstance(key, str) or len(key) > 64:
                raise ValueError("config keys must be strings of at most 64 characters")
            if key.lower() in {"api_key", "apikey", "token", "password", "secret"}:
                raise ValueError(
                    f"'{key}' must not be stored in config; reference an env var via api_key_env"
                )
        return value


class SourceCreate(SourceBase):
    slug: Annotated[str, Field(min_length=3, max_length=64, pattern=SLUG_PATTERN)]


class SourceUpdate(StrictModel):
    """Partial update - every field optional."""

    name: Annotated[str | None, Field(min_length=2, max_length=160)] = None
    url: Annotated[str | None, Field(min_length=8, max_length=2048)] = None
    homepage: Annotated[str | None, Field(max_length=2048)] = None
    description: Annotated[str | None, Field(max_length=2000)] = None
    language: Annotated[str | None, Field(max_length=8)] = None
    country: Annotated[str | None, Field(max_length=8)] = None
    category: Annotated[str | None, Field(max_length=48)] = None
    api_key_env: Annotated[str | None, Field(max_length=64, pattern=ENV_VAR_PATTERN)] = None
    config: dict[str, Any] | None = None
    weight: Annotated[float | None, Field(ge=0.0, le=2.0)] = None
    request_delay_seconds: Annotated[float | None, Field(ge=0.0, le=60.0)] = None
    max_articles_per_run: Annotated[int | None, Field(ge=1, le=1000)] = None
    respect_robots: bool | None = None
    enabled: bool | None = None
    status: SourceStatus | None = None

    @field_validator("url", "homepage")
    @classmethod
    def _url_policy(cls, value: str | None) -> str | None:
        if value is not None and not is_safe_url(value):
            raise ValueError("URL is rejected by the URL safety policy")
        return value


class SourceRead(ORMModel):
    """Public view of a source. Credentials are never included."""

    id: int
    slug: str
    name: str
    kind: SourceKind
    status: SourceStatus
    url: str
    homepage: str | None = None
    description: str | None = None
    language: str | None = None
    country: str | None = None
    category: str | None = None
    weight: float
    reliability_score: float
    enabled: bool
    respect_robots: bool
    request_delay_seconds: float
    max_articles_per_run: int
    last_fetched_at: datetime | None = None
    last_success_at: datetime | None = None
    consecutive_failures: int = 0
    total_articles: int = 0
    created_at: datetime


class SourceHealthRead(ORMModel):
    checked_at: datetime
    success: bool
    status_code: int | None = None
    latency_ms: float | None = None
    articles_fetched: int = 0
    articles_valid: int = 0
    duplicates: int = 0
    error_type: str | None = None


class SourceStats(ORMModel):
    """Per-source aggregate used by the dashboard's source comparison."""

    slug: str
    name: str
    total_articles: int = 0
    articles_24h: int = 0
    avg_sentiment: float = 0.0
    avg_relevance: float = 0.0
    duplicate_rate: float = 0.0
    reliability_score: float = 0.0
    success_rate: float = 0.0


class IngestionResult(ORMModel):
    """Outcome of one ingestion run for one source."""

    source: str
    success: bool
    fetched: int = 0
    valid: int = 0
    rejected: int = 0
    duplicates: int = 0
    stored: int = 0
    duration_ms: float = 0.0
    error: str | None = None


__all__ = [
    "IngestionResult",
    "SourceCreate",
    "SourceHealthRead",
    "SourceRead",
    "SourceStats",
    "SourceUpdate",
]
