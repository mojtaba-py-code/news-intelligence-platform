"""Personalised intelligence feed.

Filtering happens in SQL (cheap, indexed); re-ranking happens in Python over the
already-filtered slice. The scoring reuses :class:`RelevanceScorer` with the
user's preferences as the interest profile, so personalised and global ranking
never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.database.models.article import Article, SentimentLabel
from app.database.models.user import User
from app.database.repositories.article import ArticleRepository
from app.database.repositories.user import UserRepository
from app.intelligence.ranking import InterestProfile, RelevanceScorer
from app.schemas.article import ArticleSearchQuery

logger = get_logger(__name__)

#: How many rows to pull before re-ranking, so personalisation has room to work.
OVERFETCH_FACTOR = 4
MAX_CANDIDATES = 300


@dataclass(frozen=True, slots=True)
class ScoredArticle:
    """An article with its personalised score attached."""

    article: Article
    score: float


class FeedService:
    """Builds a personalised feed for one user."""

    def __init__(self, session: AsyncSession, *, config: Settings | None = None) -> None:
        self.session = session
        self.config = config or get_settings()
        self.articles = ArticleRepository(session)
        self.users = UserRepository(session)
        self.scorer = RelevanceScorer()

    async def personalized(
        self, user: User, *, limit: int = 20, offset: int = 0, hours: int = 72
    ) -> tuple[list[Article], int]:
        """Return the user's feed, re-ranked by their preferences."""
        preference = user.preference or await self.users.get_preference(user.id)
        profile = InterestProfile.from_preference(preference)

        query = self._build_query(preference)
        candidates, total = await self.articles.search(
            query, limit=min(MAX_CANDIDATES, (limit + offset) * OVERFETCH_FACTOR), offset=0
        )
        if not candidates:
            return [], 0

        if profile.is_empty:
            # No stated preferences: recency-ordered results are already right.
            window = list(candidates)[offset : offset + limit]
            return window, total

        scored = [
            ScoredArticle(
                article=article,
                score=self.scorer.score(
                    published_at=article.published_at,
                    source_weight=article.source.weight if article.source else 1.0,
                    source_reliability=article.source.reliability_score if article.source else 0.5,
                    source_slug=article.source.slug if article.source else "",
                    quality_score=article.quality_score,
                    topics=[article.category] if article.category else [],
                    keywords=list(article.keywords or []),
                    entities=[],
                    profile=profile,
                ).score,
            )
            for article in candidates
        ]
        scored = [item for item in scored if item.score > 0.0]
        scored.sort(key=lambda item: (item.score, item.article.published_at), reverse=True)

        page = scored[offset : offset + limit]
        _ = hours
        return [item.article for item in page], len(scored)

    def _build_query(self, preference: object | None) -> ArticleSearchQuery:
        """Translate stored preferences into a validated search query."""
        query = ArticleSearchQuery()
        if preference is None:
            return query

        languages = list(getattr(preference, "languages", None) or [])
        countries = list(getattr(preference, "countries", None) or [])
        sentiment = str(getattr(preference, "sentiment_preference", "any") or "any")
        min_relevance = float(getattr(preference, "min_relevance", 0.0) or 0.0)

        updates: dict[str, object] = {}
        if len(languages) == 1:
            updates["language"] = languages[0][:8]
        if len(countries) == 1:
            updates["country"] = countries[0][:8]
        if min_relevance > 0:
            updates["min_relevance"] = min(1.0, min_relevance)
        if sentiment == "positive":
            updates["min_sentiment"] = 0.1
        elif sentiment == "negative":
            updates["max_sentiment"] = -0.1
        elif sentiment == "neutral":
            updates["min_sentiment"] = -0.1
            updates["max_sentiment"] = 0.1

        return query.model_copy(update=updates) if updates else query

    async def recommendations(self, article: Article, *, limit: int = 5) -> list[Article]:
        """Related articles for the "you might also read" panel."""
        return list(await self.articles.similar_to(article, limit=limit))

    @staticmethod
    def matches_sentiment(article: Article, preference: str) -> bool:
        """Sentiment filter used by alert evaluation."""
        if preference == "positive":
            return article.sentiment_label in (
                str(SentimentLabel.POSITIVE),
                str(SentimentLabel.VERY_POSITIVE),
            )
        if preference == "negative":
            return article.sentiment_label in (
                str(SentimentLabel.NEGATIVE),
                str(SentimentLabel.VERY_NEGATIVE),
            )
        if preference == "neutral":
            return article.sentiment_label == str(SentimentLabel.NEUTRAL)
        return True


__all__ = ["FeedService", "ScoredArticle"]
