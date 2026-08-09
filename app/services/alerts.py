"""Alert evaluation and delivery.

Rules are structured data, never expressions, so evaluation is a set of
comparisons rather than an interpreter - there is no code-injection surface.

Delivery is best-effort and isolated: a webhook that times out marks the
trigger undelivered but never fails the evaluation run. Webhook URLs go through
the same SSRF policy as every other outbound request, so an alert cannot be
used to probe the internal network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.errors import UnsafeURLError
from app.core.logging import get_logger
from app.core.url_safety import validate_url
from app.core.utils import truncate, utcnow
from app.database.models.article import Article
from app.database.models.job import Alert, AlertChannel
from app.database.repositories.article import ArticleRepository
from app.database.repositories.job import AlertRepository
from app.schemas.article import ArticleSearchQuery

logger = get_logger(__name__)

MAX_MATCHES_PER_ALERT = 100


@dataclass(slots=True)
class AlertEvaluation:
    """Outcome of evaluating one rule."""

    alert_id: int
    name: str
    matched: int = 0
    triggered: bool = False
    delivered: bool = False
    error: str | None = None
    article_ids: list[int] = field(default_factory=list)


class AlertService:
    """Evaluates alert rules against recently ingested articles."""

    def __init__(self, session: AsyncSession, *, config: Settings | None = None) -> None:
        self.session = session
        self.config = config or get_settings()
        self.alerts = AlertRepository(session)
        self.articles = ArticleRepository(session)

    async def evaluate_all(self) -> list[AlertEvaluation]:
        """Evaluate every rule whose cooldown has elapsed."""
        if not self.config.alerts_enabled:
            return []
        due = await self.alerts.due()
        results = [await self.evaluate(alert) for alert in due]
        fired = sum(1 for result in results if result.triggered)
        if results:
            logger.info("alerts_evaluated", extra={"rules": len(results), "triggered": fired})
        return results

    async def evaluate(self, alert: Alert) -> AlertEvaluation:
        """Evaluate one rule and fire it when the threshold is met."""
        evaluation = AlertEvaluation(alert_id=alert.id, name=alert.name)
        window_start = utcnow() - timedelta(minutes=alert.window_minutes)

        query = ArticleSearchQuery(
            published_after=window_start,
            min_relevance=alert.min_relevance or None,
            language=(alert.languages[0][:8] if alert.languages else None),
        )
        candidates, _ = await self.articles.search(query, limit=MAX_MATCHES_PER_ALERT, offset=0)

        matches = [article for article in candidates if self._matches(alert, article)]
        evaluation.matched = len(matches)
        if len(matches) < alert.min_articles:
            return evaluation

        evaluation.triggered = True
        evaluation.article_ids = [article.id for article in matches[:50]]
        message = self._message(alert, matches)

        delivered, error = await self._deliver(alert, message, matches)
        evaluation.delivered = delivered
        evaluation.error = error

        await self.alerts.record_trigger(
            alert,
            article_ids=evaluation.article_ids,
            message=message,
            delivered=delivered,
            delivery_error=error,
        )
        return evaluation

    # ------------------------------------------------------------- matching
    @staticmethod
    def _matches(alert: Alert, article: Article) -> bool:
        """All configured conditions must hold (AND semantics)."""
        haystack = " ".join(
            filter(
                None,
                [
                    article.title,
                    article.description or "",
                    " ".join(article.keywords or []),
                ],
            )
        ).casefold()

        if alert.keywords and not any(keyword.casefold() in haystack for keyword in alert.keywords):
            return False
        if alert.topics and (article.category or "").casefold() not in {
            topic.casefold() for topic in alert.topics
        }:
            return False
        if alert.sources and (article.source.slug if article.source else "") not in {
            source.lower() for source in alert.sources
        }:
            return False
        if alert.entities and not any(entity.casefold() in haystack for entity in alert.entities):
            return False
        if alert.sentiment_filter:
            from app.services.feed import FeedService

            if not FeedService.matches_sentiment(article, alert.sentiment_filter):
                return False
        return True

    @staticmethod
    def _message(alert: Alert, matches: list[Article]) -> str:
        headline = truncate(matches[0].title, 120) if matches else ""
        return (
            f"Alert '{alert.name}': {len(matches)} matching article(s) "
            f"in the last {alert.window_minutes} minutes. Latest: {headline}"
        )

    # ------------------------------------------------------------- delivery
    async def _deliver(
        self, alert: Alert, message: str, matches: list[Article]
    ) -> tuple[bool, str | None]:
        """Send the notification. Never raises."""
        if alert.channel == AlertChannel.IN_APP:
            return True, None  # stored as a trigger row; the UI reads it
        if alert.channel == AlertChannel.WEBHOOK:
            return await self._deliver_webhook(alert, message, matches)
        if alert.channel == AlertChannel.EMAIL:
            # SMTP delivery is intentionally out of scope for the default build;
            # the trigger is recorded so nothing is lost.
            if not self.config.smtp_host:
                return False, "SMTP is not configured"
            return False, "e-mail delivery is not enabled in this deployment"
        return False, f"unknown channel '{alert.channel}'"

    async def _deliver_webhook(
        self, alert: Alert, message: str, matches: list[Article]
    ) -> tuple[bool, str | None]:
        if not alert.destination:
            return False, "no destination configured"
        try:
            validate_url(alert.destination, config=self.config)
        except UnsafeURLError as exc:
            logger.warning("alert_webhook_rejected", extra={"alert_id": alert.id})
            return False, exc.message

        payload = {
            "alert": alert.name,
            "message": message,
            "triggered_at": utcnow().isoformat(),
            "article_count": len(matches),
            "articles": [
                {
                    "id": article.id,
                    "title": article.title,
                    "url": article.canonical_url,
                    "source": article.source_name,
                    "published_at": article.published_at.isoformat(),
                    "sentiment": article.sentiment_score,
                }
                for article in matches[:20]
            ],
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.config.alert_webhook_timeout_seconds,
                follow_redirects=False,
                trust_env=False,
            ) as client:
                response = await client.post(
                    alert.destination,
                    json=payload,
                    headers={"User-Agent": self.config.http_user_agent},
                )
            if response.status_code >= 400:
                return False, f"webhook returned {response.status_code}"
            return True, None
        except httpx.HTTPError as exc:
            logger.warning(
                "alert_webhook_failed",
                extra={"alert_id": alert.id, "error_type": exc.__class__.__name__},
            )
            return False, f"delivery failed: {exc.__class__.__name__}"


__all__ = ["AlertEvaluation", "AlertService"]
