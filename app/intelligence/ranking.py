"""Relevance scoring.

The score is a **weighted sum of independent, individually explainable
signals**, each normalised to ``[0, 1]`` before weighting:

===================  =========================================================
signal               meaning
===================  =========================================================
source               editorial weight and measured reliability of the outlet
recency              exponential decay with a configurable half-life
topic                overlap with the reader's (or deployment's) topics
keyword              overlap with tracked keywords
entity               overlap with tracked entities
quality              completeness of the article record
corroboration        how many independent sources cover the same story
engagement           optional external signal (defaults to neutral)
===================  =========================================================

Weights live in :class:`RelevanceWeights`, so a deployment can re-tune the
ranking without touching the algorithm, and personalised scoring reuses exactly
the same code path with the user's preferences supplied as the profile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.core.utils import clamp, ensure_utc, utcnow


@dataclass(frozen=True, slots=True)
class RelevanceWeights:
    """Relative importance of each signal. Normalised at scoring time."""

    source: float = 0.20
    recency: float = 0.25
    topic: float = 0.18
    keyword: float = 0.12
    entity: float = 0.08
    quality: float = 0.10
    corroboration: float = 0.05
    engagement: float = 0.02

    def total(self) -> float:
        return (
            self.source
            + self.recency
            + self.topic
            + self.keyword
            + self.entity
            + self.quality
            + self.corroboration
            + self.engagement
        )


@dataclass(frozen=True, slots=True)
class InterestProfile:
    """What the scorer should consider interesting.

    Empty collections mean "no preference": the corresponding signal returns a
    neutral 0.5 rather than zero, so an unpersonalised feed is not flattened.
    """

    topics: frozenset[str] = frozenset()
    keywords: frozenset[str] = frozenset()
    entities: frozenset[str] = frozenset()
    preferred_sources: frozenset[str] = frozenset()
    excluded_sources: frozenset[str] = frozenset()
    languages: frozenset[str] = frozenset()
    countries: frozenset[str] = frozenset()

    @classmethod
    def from_preference(cls, preference: Any | None) -> InterestProfile:
        """Build a profile from a ``UserPreference`` row (or ``None``)."""
        if preference is None:
            return cls()
        lower = lambda values: frozenset(str(v).casefold() for v in (values or []))  # noqa: E731
        return cls(
            topics=lower(getattr(preference, "topics", None)),
            keywords=lower(getattr(preference, "keywords", None)),
            entities=lower(getattr(preference, "entities", None)),
            preferred_sources=lower(getattr(preference, "sources", None)),
            excluded_sources=lower(getattr(preference, "excluded_sources", None)),
            languages=lower(getattr(preference, "languages", None)),
            countries=lower(getattr(preference, "countries", None)),
        )

    @property
    def is_empty(self) -> bool:
        return not (self.topics or self.keywords or self.entities or self.preferred_sources)


@dataclass(frozen=True, slots=True)
class RelevanceBreakdown:
    """The final score plus every component that produced it.

    Explainability is a feature: an analyst who cannot see *why* an article
    ranked first will not trust the ranking.
    """

    score: float
    components: dict[str, float] = field(default_factory=dict)

    def explain(self) -> str:
        parts = ", ".join(f"{name}={value:.2f}" for name, value in sorted(self.components.items()))
        return f"score={self.score:.3f} ({parts})"


#: Hours after which the recency signal has decayed to 0.5.
DEFAULT_HALF_LIFE_HOURS: float = 18.0


def recency_score(
    published_at: datetime | None,
    *,
    half_life_hours: float = DEFAULT_HALF_LIFE_HOURS,
    now: datetime | None = None,
) -> float:
    """Exponential decay: ``0.5 ** (age / half_life)``."""
    if published_at is None:
        return 0.0
    aware = ensure_utc(published_at)
    if aware is None:
        return 0.0
    age_hours = max(0.0, ((now or utcnow()) - aware).total_seconds() / 3600.0)
    return clamp(math.pow(0.5, age_hours / max(0.5, half_life_hours)))


def _overlap(values: frozenset[str], candidates: frozenset[str]) -> float:
    """Share of the profile's terms that the article satisfies."""
    if not values:
        return 0.5  # neutral: the reader expressed no preference
    if not candidates:
        return 0.0
    hits = sum(1 for value in values if any(value in candidate for candidate in candidates))
    return clamp(hits / len(values))


class RelevanceScorer:
    """Computes relevance for an article, optionally personalised."""

    def __init__(
        self,
        weights: RelevanceWeights | None = None,
        *,
        half_life_hours: float = DEFAULT_HALF_LIFE_HOURS,
    ) -> None:
        self.weights = weights or RelevanceWeights()
        self.half_life_hours = half_life_hours

    def score(
        self,
        *,
        published_at: datetime | None,
        source_weight: float = 1.0,
        source_reliability: float = 0.5,
        source_slug: str = "",
        quality_score: float = 0.5,
        topics: list[str] | None = None,
        keywords: list[str] | None = None,
        entities: list[str] | None = None,
        corroborating_sources: int = 1,
        engagement: float | None = None,
        profile: InterestProfile | None = None,
        now: datetime | None = None,
    ) -> RelevanceBreakdown:
        """Score one article and return the breakdown."""
        profile = profile or InterestProfile()
        slug = source_slug.casefold()

        # An excluded source is a hard zero - the reader asked not to see it.
        if slug and slug in profile.excluded_sources:
            return RelevanceBreakdown(score=0.0, components={"excluded": 1.0})

        article_topics = frozenset(str(t).casefold() for t in (topics or []))
        article_keywords = frozenset(str(k).casefold() for k in (keywords or []))
        article_entities = frozenset(str(e).casefold() for e in (entities or []))

        components: dict[str, float] = {
            "source": clamp(0.5 * clamp(source_weight / 2.0) + 0.5 * clamp(source_reliability)),
            "recency": recency_score(published_at, half_life_hours=self.half_life_hours, now=now),
            "topic": _overlap(profile.topics, article_topics),
            "keyword": _overlap(profile.keywords, article_keywords),
            "entity": _overlap(profile.entities, article_entities),
            "quality": clamp(quality_score),
            # log2 so the 2nd corroborating source matters far more than the 9th.
            "corroboration": clamp(math.log2(1 + max(0, corroborating_sources)) / 4.0),
            "engagement": clamp(engagement) if engagement is not None else 0.5,
        }

        if slug and slug in profile.preferred_sources:
            components["source"] = clamp(components["source"] + 0.3)

        weights = self.weights
        weighted = (
            weights.source * components["source"]
            + weights.recency * components["recency"]
            + weights.topic * components["topic"]
            + weights.keyword * components["keyword"]
            + weights.entity * components["entity"]
            + weights.quality * components["quality"]
            + weights.corroboration * components["corroboration"]
            + weights.engagement * components["engagement"]
        )
        total_weight = weights.total() or 1.0
        return RelevanceBreakdown(
            score=round(clamp(weighted / total_weight), 4),
            components={name: round(value, 4) for name, value in components.items()},
        )

    def score_article(
        self,
        article: Any,
        *,
        profile: InterestProfile | None = None,
        corroborating_sources: int = 1,
        now: datetime | None = None,
    ) -> RelevanceBreakdown:
        """Convenience wrapper for an ORM ``Article`` row."""
        source = getattr(article, "source", None)
        return self.score(
            published_at=getattr(article, "published_at", None),
            source_weight=getattr(source, "weight", 1.0) if source else 1.0,
            source_reliability=getattr(source, "reliability_score", 0.5) if source else 0.5,
            source_slug=getattr(source, "slug", "") if source else "",
            quality_score=getattr(article, "quality_score", 0.5),
            topics=[link.topic.slug for link in getattr(article, "topics", []) if link.topic],
            keywords=list(getattr(article, "keywords", []) or []),
            entities=[link.entity.name for link in getattr(article, "entities", []) if link.entity],
            corroborating_sources=corroborating_sources,
            profile=profile,
            now=now,
        )


_default_scorer = RelevanceScorer()


def score_relevance(**kwargs: Any) -> RelevanceBreakdown:
    """Score with the process-wide scorer."""
    return _default_scorer.score(**kwargs)


__all__ = [
    "DEFAULT_HALF_LIFE_HOURS",
    "InterestProfile",
    "RelevanceBreakdown",
    "RelevanceScorer",
    "RelevanceWeights",
    "recency_score",
    "score_relevance",
]
