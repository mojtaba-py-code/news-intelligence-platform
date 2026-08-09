"""Schemas for topics, entities, trends, events, analytics and alerts."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import Field, field_validator

from app.core.url_safety import is_safe_url
from app.database.models.job import AlertChannel
from app.database.models.taxonomy import EntityType, TrendDirection, TrendSubject
from app.schemas.common import ORMModel, StrictModel


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
class TopicRead(ORMModel):
    id: int
    slug: str
    name: str
    description: str | None = None
    keywords: list[str] = Field(default_factory=list)
    article_count: int = 0
    is_active: bool = True


class TopicCreate(StrictModel):
    slug: Annotated[str, Field(min_length=2, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$")]
    name: Annotated[str, Field(min_length=2, max_length=120)]
    description: Annotated[str | None, Field(max_length=1000)] = None
    keywords: Annotated[list[str], Field(max_length=200)] = Field(default_factory=list)
    parent_slug: Annotated[str | None, Field(max_length=64)] = None

    @field_validator("keywords")
    @classmethod
    def _clean(cls, value: list[str]) -> list[str]:
        return [term.strip().lower() for term in value if term.strip()][:200]


class EntityRead(ORMModel):
    id: int
    name: str
    entity_type: EntityType
    mention_count: int = 0
    article_count: int = 0
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None


class EntityGraphEdge(StrictModel):
    subject: str
    subject_type: str
    predicate: str
    object: str
    object_type: str
    weight: float = 1.0


class EntityGraph(StrictModel):
    """Knowledge-graph slice returned by ``/entities/{id}/graph``."""

    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[EntityGraphEdge] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Trends & events
# --------------------------------------------------------------------------- #
class TrendRead(ORMModel):
    subject_type: TrendSubject
    subject_key: str
    subject_label: str
    window_start: datetime
    window_end: datetime
    current_count: int
    previous_count: int
    growth_percent: float
    trend_score: float
    direction: TrendDirection
    confidence: float
    avg_sentiment: float
    sentiment_delta: float
    source_count: int


class EventRead(ORMModel):
    id: int
    event_key: str
    title: str
    summary: str | None = None
    first_seen_at: datetime
    last_updated_at: datetime
    article_count: int
    source_count: int
    sources: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    entities: list[dict[str, Any]] = Field(default_factory=list)
    avg_sentiment: float = 0.0
    importance: float = 0.0


# --------------------------------------------------------------------------- #
# Analytics
# --------------------------------------------------------------------------- #
class SentimentBreakdown(StrictModel):
    very_negative: int = 0
    negative: int = 0
    neutral: int = 0
    positive: int = 0
    very_positive: int = 0
    average_score: float = 0.0


class CountBucket(StrictModel):
    key: str
    label: str
    count: int
    percentage: float = 0.0


class TimeSeriesPoint(StrictModel):
    timestamp: datetime
    count: int
    avg_sentiment: float = 0.0


class AnalyticsOverview(StrictModel):
    total_articles: int = 0
    articles_today: int = 0
    articles_24h: int = 0
    active_sources: int = 0
    total_sources: int = 0
    detected_events: int = 0
    duplicate_rate: float = 0.0
    avg_relevance: float = 0.0
    trending_topics: list[CountBucket] = Field(default_factory=list)
    sentiment: SentimentBreakdown = Field(default_factory=SentimentBreakdown)
    generated_at: datetime


class DataQualityReport(StrictModel):
    """Counters emitted by the validation stage."""

    total_processed: int = 0
    valid_articles: int = 0
    invalid_articles: int = 0
    duplicate_articles: int = 0
    missing_title: int = 0
    invalid_url: int = 0
    invalid_date: int = 0
    empty_content: int = 0
    unsupported_language: int = 0
    malformed_response: int = 0

    @property
    def duplicate_rate(self) -> float:
        return self.duplicate_articles / self.total_processed if self.total_processed else 0.0

    @property
    def failure_rate(self) -> float:
        return self.invalid_articles / self.total_processed if self.total_processed else 0.0


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #
class AlertCreate(StrictModel):
    """Structured alert rule. No free-form expression is ever accepted."""

    name: Annotated[str, Field(min_length=2, max_length=120)]
    keywords: Annotated[list[str], Field(max_length=30)] = Field(default_factory=list)
    topics: Annotated[list[str], Field(max_length=20)] = Field(default_factory=list)
    entities: Annotated[list[str], Field(max_length=30)] = Field(default_factory=list)
    sources: Annotated[list[str], Field(max_length=30)] = Field(default_factory=list)
    languages: Annotated[list[str], Field(max_length=10)] = Field(default_factory=list)
    min_articles: Annotated[int, Field(ge=1, le=1000)] = 1
    window_minutes: Annotated[int, Field(ge=5, le=10_080)] = 60
    min_relevance: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    sentiment_filter: Annotated[str | None, Field(pattern=r"^(positive|neutral|negative)$")] = None
    channel: AlertChannel = AlertChannel.IN_APP
    destination: Annotated[str | None, Field(max_length=2048)] = None
    cooldown_minutes: Annotated[int, Field(ge=5, le=10_080)] = 60

    @field_validator("keywords", "topics", "entities", "sources", "languages")
    @classmethod
    def _clean(cls, value: list[str]) -> list[str]:
        cleaned = [term.strip() for term in value if term.strip()]
        if any(len(term) > 100 for term in cleaned):
            raise ValueError("each entry must be at most 100 characters")
        return cleaned

    @field_validator("destination")
    @classmethod
    def _safe_destination(cls, value: str | None) -> str | None:
        """A webhook URL is an outbound request we make on the user's behalf."""
        if value is None:
            return None
        if value.startswith(("http://", "https://")) and not is_safe_url(value):
            raise ValueError("destination URL is rejected by the URL safety policy")
        return value


class AlertRead(ORMModel):
    id: int
    name: str
    keywords: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    min_articles: int
    window_minutes: int
    min_relevance: float
    channel: AlertChannel
    destination: str | None = None
    is_active: bool
    last_triggered_at: datetime | None = None
    trigger_count: int = 0
    created_at: datetime


class AlertTriggerRead(ORMModel):
    id: int
    alert_id: int
    triggered_at: datetime
    article_count: int
    article_ids: list[str] = Field(default_factory=list)
    message: str
    delivered: bool = False
    acknowledged: bool = False


__all__ = [
    "AlertCreate",
    "AlertRead",
    "AlertTriggerRead",
    "AnalyticsOverview",
    "CountBucket",
    "DataQualityReport",
    "EntityGraph",
    "EntityGraphEdge",
    "EntityRead",
    "EventRead",
    "SentimentBreakdown",
    "TimeSeriesPoint",
    "TopicCreate",
    "TopicRead",
    "TrendRead",
]
