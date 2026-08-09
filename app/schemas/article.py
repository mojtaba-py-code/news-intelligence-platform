"""Article schemas: the connector contract, the normalised form and API views."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.utils import ensure_utc, utcnow
from app.database.models.article import ProcessingStatus, SentimentLabel
from app.schemas.common import ORMModel, SortOrder

# Bounds sized for real news payloads. They also cap memory: a hostile source
# could otherwise stream a 100 MB "title" straight into the pipeline.
MAX_TITLE = 512
MAX_DESCRIPTION = 5_000
MAX_CONTENT = 200_000
MAX_URL = 2_048
MAX_KEYWORDS = 30


class RawArticle(BaseModel):
    """A single item as produced by a source connector, before normalisation.

    This is the *only* shape the pipeline accepts from connectors. Anything a
    source-specific parser wants to pass along has to fit here, which is what
    keeps source quirks out of the core.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    source_slug: Annotated[str, Field(min_length=1, max_length=64)]
    source_name: Annotated[str, Field(min_length=1, max_length=160)]
    title: Annotated[str, Field(min_length=1, max_length=MAX_TITLE * 4)]
    url: Annotated[str, Field(min_length=5, max_length=MAX_URL)]

    external_id: Annotated[str | None, Field(max_length=255)] = None
    description: Annotated[str | None, Field(max_length=MAX_DESCRIPTION * 4)] = None
    content: Annotated[str | None, Field(max_length=MAX_CONTENT)] = None
    author: Annotated[str | None, Field(max_length=400)] = None
    image_url: Annotated[str | None, Field(max_length=MAX_URL)] = None
    published_at: datetime | None = None
    updated_at: datetime | None = None
    language: Annotated[str | None, Field(max_length=16)] = None
    country: Annotated[str | None, Field(max_length=16)] = None
    category: Annotated[str | None, Field(max_length=64)] = None
    #: Anything else the source returned; retained for debugging, never trusted.
    raw: dict[str, Any] = Field(default_factory=dict)

    @field_validator("published_at", "updated_at", mode="before")
    @classmethod
    def _parse_dates(cls, value: Any) -> Any:
        """Accept RFC 822, ISO 8601, epochs and "3 hours ago".

        Connectors hand through whatever the source produced; normalising here
        keeps every one of them free of date-parsing code.
        """
        if value is None or isinstance(value, datetime):
            return ensure_utc(value)
        from app.processing.normalization.dates import parse_datetime

        return parse_datetime(value)

    @field_validator("published_at", "updated_at")
    @classmethod
    def _tz_aware(cls, value: datetime | None) -> datetime | None:
        return ensure_utc(value)

    @field_validator("title", "description", "content", "author", mode="before")
    @classmethod
    def _drop_control_chars(cls, value: Any) -> Any:
        """Strip NUL and other C0 controls that break PostgreSQL text columns."""
        if isinstance(value, str):
            return "".join(ch for ch in value if ch == "\n" or ch == "\t" or ord(ch) >= 32)
        return value


class NormalizedArticle(BaseModel):
    """A cleaned, canonicalised, fingerprinted article ready for persistence."""

    model_config = ConfigDict(extra="forbid")

    source_slug: str
    source_name: str
    external_id: str | None = None

    title: Annotated[str, Field(min_length=1, max_length=MAX_TITLE)]
    description: Annotated[str | None, Field(max_length=MAX_DESCRIPTION)] = None
    content: Annotated[str | None, Field(max_length=MAX_CONTENT)] = None
    author_name: Annotated[str | None, Field(max_length=200)] = None

    url: Annotated[str, Field(max_length=MAX_URL)]
    canonical_url: Annotated[str, Field(max_length=MAX_URL)]
    image_url: Annotated[str | None, Field(max_length=MAX_URL)] = None

    published_at: datetime
    source_updated_at: datetime | None = None

    language: Annotated[str | None, Field(max_length=8)] = None
    country: Annotated[str | None, Field(max_length=8)] = None
    category: Annotated[str | None, Field(max_length=48)] = None

    content_hash: Annotated[str, Field(min_length=64, max_length=64)]
    title_hash: Annotated[str, Field(min_length=64, max_length=64)]
    simhash: str | None = None
    word_count: int = 0
    quality_score: float = 0.0

    @model_validator(mode="after")
    def _published_not_in_far_future(self) -> NormalizedArticle:
        """Clamp implausible timestamps.

        Feeds regularly emit dates from a broken CMS clock; letting them through
        would park an article permanently at the top of every recency ranking.
        """
        now = utcnow()
        if self.published_at > now:
            object.__setattr__(self, "published_at", now)
        return self


# --------------------------------------------------------------------------- #
# API read models
# --------------------------------------------------------------------------- #
class TopicRef(ORMModel):
    slug: str
    name: str
    score: float = 0.0


class EntityRef(ORMModel):
    name: str
    entity_type: str
    salience: float = 0.0
    mentions: int = 1


class ArticleRead(ORMModel):
    """Listing representation - deliberately excludes full content."""

    id: int
    source_id: int
    source_name: str
    title: str
    description: str | None = None
    summary: str | None = None
    url: str
    canonical_url: str
    image_url: str | None = None
    author_name: str | None = None
    published_at: datetime
    language: str | None = None
    country: str | None = None
    category: str | None = None
    sentiment_score: float
    sentiment_label: SentimentLabel
    relevance_score: float
    quality_score: float
    keywords: list[str] = Field(default_factory=list)
    word_count: int = 0
    is_duplicate: bool = False
    status: ProcessingStatus
    created_at: datetime


class ArticleDetail(ArticleRead):
    """Single-article representation, including content and enrichment."""

    content: str | None = None
    content_hash: str
    title_hash: str
    sentiment_confidence: float = 0.0
    readability_score: float | None = None
    enrichment: dict[str, Any] = Field(default_factory=dict)
    topics: list[TopicRef] = Field(default_factory=list)
    entities: list[EntityRef] = Field(default_factory=list)
    duplicate_of_id: int | None = None
    processed_at: datetime | None = None


class ArticleSortField(StrEnum):
    """Sortable columns, as an allowlist.

    ``ORDER BY`` cannot be parameterised, so the only safe way to expose
    client-chosen sorting is to accept an enum and map it to a column object.
    """

    PUBLISHED_AT = "published_at"
    RELEVANCE = "relevance_score"
    SENTIMENT = "sentiment_score"
    QUALITY = "quality_score"
    CREATED_AT = "created_at"
    WORD_COUNT = "word_count"


class ArticleSearchQuery(BaseModel):
    """Validated search/filter parameters.

    ``sort_by`` is an enum, never a raw column name: it is interpolated into an
    ``ORDER BY`` clause, so an allowlist is the only safe design.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    q: Annotated[str | None, Field(max_length=200, description="Free-text query")] = None
    phrase: Annotated[str | None, Field(max_length=200, description="Exact phrase")] = None
    source: Annotated[str | None, Field(max_length=64)] = None
    source_id: int | None = None
    category: Annotated[str | None, Field(max_length=48)] = None
    language: Annotated[str | None, Field(max_length=8)] = None
    country: Annotated[str | None, Field(max_length=8)] = None
    topic: Annotated[str | None, Field(max_length=64)] = None
    entity: Annotated[str | None, Field(max_length=200)] = None
    author: Annotated[str | None, Field(max_length=200)] = None
    sentiment: SentimentLabel | None = None
    min_sentiment: Annotated[float | None, Field(ge=-1.0, le=1.0)] = None
    max_sentiment: Annotated[float | None, Field(ge=-1.0, le=1.0)] = None
    min_relevance: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
    published_after: datetime | None = None
    published_before: datetime | None = None
    include_duplicates: bool = False
    sort_by: ArticleSortField = ArticleSortField.PUBLISHED_AT
    order: SortOrder = SortOrder.DESC

    @field_validator("q", "phrase", "entity", "author")
    @classmethod
    def _reject_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("published_after", "published_before")
    @classmethod
    def _tz(cls, value: datetime | None) -> datetime | None:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _coherent_ranges(self) -> ArticleSearchQuery:
        if (
            self.published_after
            and self.published_before
            and self.published_after > self.published_before
        ):
            raise ValueError("published_after must be earlier than published_before")
        if (
            self.min_sentiment is not None
            and self.max_sentiment is not None
            and self.min_sentiment > self.max_sentiment
        ):
            raise ValueError("min_sentiment must be <= max_sentiment")
        return self


#: Convenience view of the sort allowlist for the repository layer.
ALLOWED_SORT_FIELDS: frozenset[str] = frozenset(field.value for field in ArticleSortField)


class ArticleStats(BaseModel):
    """Aggregate counters used by the analytics endpoints."""

    total: int = 0
    today: int = 0
    last_24h: int = 0
    last_7d: int = 0
    duplicates: int = 0
    pending: int = 0
    failed: int = 0
    avg_relevance: float = 0.0
    avg_sentiment: float = 0.0


__all__ = [
    "ALLOWED_SORT_FIELDS",
    "MAX_CONTENT",
    "MAX_DESCRIPTION",
    "MAX_TITLE",
    "MAX_URL",
    "ArticleDetail",
    "ArticleRead",
    "ArticleSearchQuery",
    "ArticleSortField",
    "ArticleStats",
    "EntityRef",
    "NormalizedArticle",
    "RawArticle",
    "TopicRef",
]
