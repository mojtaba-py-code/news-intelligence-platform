"""The connector contract.

Every source - REST API, RSS feed, scraped site, custom - implements
:class:`NewsSource` and returns :class:`RawArticle` objects. The base class
owns everything generic (fetching, credential resolution, per-item validation,
error accounting) so a concrete connector only implements ``_collect``.

That split is what keeps source-specific logic out of the core pipeline: the
pipeline never learns that "this one paginates with a cursor" or "that one puts
the body in ``fields.bodyHtml``".
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from app.core.config import Settings, get_settings
from app.core.errors import ConfigurationError, SourceError
from app.core.logging import get_logger
from app.core.metrics import articles_fetched_total, articles_rejected_total, source_errors_total
from app.database.models.source import SourceKind
from app.ingestion.fetchers.http import FetchResponse, SecureHTTPFetcher, get_fetcher
from app.schemas.article import RawArticle

logger = get_logger(__name__)


@dataclass(slots=True)
class SourceContext:
    """Everything a connector needs, decoupled from the ORM.

    Built from a ``Source`` row (or a YAML entry) so connectors can be unit
    tested without a database.
    """

    slug: str
    name: str
    kind: SourceKind
    url: str
    config: dict[str, Any] = field(default_factory=dict)
    api_key_env: str | None = None
    language: str | None = None
    country: str | None = None
    category: str | None = None
    max_articles: int = 100
    request_delay: float = 1.0
    respect_robots: bool = True
    source_id: int | None = None

    @classmethod
    def from_model(cls, source: Any) -> SourceContext:
        """Build a context from a ``Source`` ORM row."""
        return cls(
            slug=source.slug,
            name=source.name,
            kind=SourceKind(source.kind),
            url=source.url,
            config=dict(source.config or {}),
            api_key_env=source.api_key_env,
            language=source.language,
            country=source.country,
            category=source.category,
            max_articles=source.max_articles_per_run,
            request_delay=source.request_delay_seconds,
            respect_robots=source.respect_robots,
            source_id=source.id,
        )

    def option(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)


@dataclass(slots=True)
class FetchOutcome:
    """Result of one connector run."""

    source: str
    articles: list[RawArticle] = field(default_factory=list)
    fetched: int = 0
    rejected: int = 0
    duration_ms: float = 0.0
    status_code: int | None = None
    error: str | None = None
    error_type: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


class NewsSource(ABC):
    """Base class for every connector."""

    #: Identifier used in ``configs/sources.yaml`` and the ``sources.kind`` column.
    kind: SourceKind = SourceKind.CUSTOM

    def __init__(
        self,
        context: SourceContext,
        *,
        fetcher: SecureHTTPFetcher | None = None,
        config: Settings | None = None,
    ) -> None:
        self.context = context
        self.settings = config or get_settings()
        self._fetcher = fetcher
        self._last_status: int | None = None

    # --------------------------------------------------------------- plumbing
    @property
    def slug(self) -> str:
        return self.context.slug

    @property
    def fetcher(self) -> SecureHTTPFetcher:
        return self._fetcher if self._fetcher is not None else get_fetcher(self.settings)

    def api_key(self) -> str | None:
        """Resolve this source's credential from the environment.

        The key never touches the database or a config file - only the *name*
        of the variable does.
        """
        if not self.context.api_key_env:
            return None
        value = os.environ.get(self.context.api_key_env)
        if not value:
            raise ConfigurationError(
                f"Source '{self.slug}' requires environment variable "
                f"'{self.context.api_key_env}', which is not set."
            )
        return value

    async def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        accept: str | None = None,
    ) -> FetchResponse:
        """Fetch through the shared secure fetcher."""
        response = await self.fetcher.fetch(
            url,
            source=self.slug,
            params=params,
            headers=headers,
            accept=accept,
            respect_robots=self.context.respect_robots,
            delay=self.context.request_delay,
        )
        self._last_status = response.status_code
        return response

    # ------------------------------------------------------------- public API
    async def fetch(self) -> FetchOutcome:
        """Run the connector, isolating any failure to this source."""
        started = time.perf_counter()
        outcome = FetchOutcome(source=self.slug)

        try:
            raw_items = await self._collect()
        except SourceError as exc:
            source_errors_total.inc(labels={"source": self.slug, "kind": exc.code})
            outcome.error = exc.message
            outcome.error_type = exc.__class__.__name__
            outcome.status_code = exc.details.get("status") or self._last_status
            outcome.duration_ms = (time.perf_counter() - started) * 1000
            logger.warning(
                "source_failed",
                extra={"source": self.slug, "error_type": outcome.error_type},
            )
            return outcome
        except Exception as exc:
            source_errors_total.inc(labels={"source": self.slug, "kind": "unexpected"})
            outcome.error = f"{exc.__class__.__name__}: {exc}"[:500]
            outcome.error_type = exc.__class__.__name__
            outcome.duration_ms = (time.perf_counter() - started) * 1000
            logger.exception("source_crashed", extra={"source": self.slug})
            return outcome

        articles, rejected = self._validate_items(raw_items)
        outcome.articles = articles[: self.context.max_articles]
        outcome.fetched = len(raw_items)
        outcome.rejected = rejected + max(0, len(articles) - len(outcome.articles))
        outcome.status_code = self._last_status
        outcome.duration_ms = (time.perf_counter() - started) * 1000

        articles_fetched_total.inc(len(outcome.articles), labels={"source": self.slug})
        if outcome.rejected:
            articles_rejected_total.inc(outcome.rejected, labels={"source": self.slug})
        logger.info(
            "source_fetched",
            extra={
                "source": self.slug,
                "articles": len(outcome.articles),
                "rejected": outcome.rejected,
                "duration_ms": round(outcome.duration_ms, 1),
            },
        )
        return outcome

    @abstractmethod
    async def _collect(self) -> list[dict[str, Any]]:
        """Fetch and parse, returning ``RawArticle``-shaped dictionaries."""

    # ------------------------------------------------------------------ utils
    def _validate_items(self, items: list[dict[str, Any]]) -> tuple[list[RawArticle], int]:
        """Coerce raw dictionaries into validated models, counting rejects.

        This is the first validation boundary: whatever a source returns, only
        well-formed :class:`RawArticle` objects continue into the pipeline.
        """
        articles: list[RawArticle] = []
        rejected = 0
        for item in items:
            payload = self._with_defaults(item)
            try:
                articles.append(RawArticle.model_validate(payload))
            except PydanticValidationError as exc:
                rejected += 1
                logger.debug(
                    "raw_article_rejected",
                    extra={"source": self.slug, "errors": exc.error_count()},
                )
        return articles, rejected

    def _with_defaults(self, item: dict[str, Any]) -> dict[str, Any]:
        """Fill in source identity and inherited metadata."""
        payload = dict(item)
        payload.setdefault("source_slug", self.slug)
        payload.setdefault("source_name", self.context.name)
        if not payload.get("language") and self.context.language:
            payload["language"] = self.context.language
        if not payload.get("country") and self.context.country:
            payload["country"] = self.context.country
        if not payload.get("category") and self.context.category:
            payload["category"] = self.context.category
        return payload


__all__ = ["FetchOutcome", "NewsSource", "SourceContext"]
