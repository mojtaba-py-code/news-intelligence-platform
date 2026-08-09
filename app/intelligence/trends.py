"""Emerging-trend detection.

Compares a *current* window against the *previous* window of equal length and
scores each subject (topic, keyword, entity, source) on three things:

* **growth** - relative change in article volume;
* **volume** - absolute size, so a 1 -> 4 jump does not outrank 120 -> 390;
* **breadth** - how many distinct sources are covering it, which separates a
  real story from one outlet publishing ten follow-ups.

Confidence is reported separately from the score: a spike measured on three
articles is a weak signal even when the percentage looks dramatic.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.core.utils import clamp, percentage_change, utcnow
from app.database.models.taxonomy import TrendDirection, TrendSubject

#: A subject must clear this many articles in the current window to be scored.
DEFAULT_MIN_ARTICLES: int = 3
#: Growth above this percentage counts as a full-strength spike.
SPIKE_CEILING_PERCENT: float = 300.0


@dataclass(frozen=True, slots=True)
class SubjectObservation:
    """One article's contribution to one subject."""

    subject_type: TrendSubject
    key: str
    label: str
    source_slug: str
    sentiment: float = 0.0
    article_id: int | None = None


@dataclass(frozen=True, slots=True)
class TrendResult:
    """A scored trend for one subject over one window."""

    subject_type: TrendSubject
    subject_key: str
    subject_label: str
    current_count: int
    previous_count: int
    growth_percent: float
    trend_score: float
    direction: TrendDirection
    confidence: float
    avg_sentiment: float
    sentiment_delta: float
    source_count: int
    sample_article_ids: tuple[int, ...] = ()

    @property
    def is_breaking(self) -> bool:
        """Sudden, broadly corroborated, high-volume coverage."""
        return (
            self.direction in (TrendDirection.RISING, TrendDirection.NEW)
            and self.growth_percent >= 150.0
            and self.source_count >= 3
            and self.confidence >= 0.5
        )


@dataclass
class _Bucket:
    count: int = 0
    sentiment_total: float = 0.0
    sources: set[str] = field(default_factory=set)
    article_ids: list[int] = field(default_factory=list)
    label: str = ""

    @property
    def avg_sentiment(self) -> float:
        return self.sentiment_total / self.count if self.count else 0.0


def _aggregate(observations: Iterable[SubjectObservation]) -> dict[tuple[str, str], _Bucket]:
    buckets: dict[tuple[str, str], _Bucket] = defaultdict(_Bucket)
    for observation in observations:
        bucket = buckets[(str(observation.subject_type), observation.key)]
        bucket.count += 1
        bucket.sentiment_total += observation.sentiment
        if observation.source_slug:
            bucket.sources.add(observation.source_slug)
        if observation.article_id is not None and len(bucket.article_ids) < 10:
            bucket.article_ids.append(observation.article_id)
        if not bucket.label:
            bucket.label = observation.label
    return buckets


def detect_trends(
    current: Iterable[SubjectObservation],
    previous: Iterable[SubjectObservation],
    *,
    min_articles: int = DEFAULT_MIN_ARTICLES,
    limit: int = 50,
) -> list[TrendResult]:
    """Score every subject present in the current window."""
    current_buckets = _aggregate(current)
    previous_buckets = _aggregate(previous)
    if not current_buckets:
        return []

    peak_volume = max(bucket.count for bucket in current_buckets.values())
    results: list[TrendResult] = []

    for (subject_type, key), bucket in current_buckets.items():
        if bucket.count < min_articles:
            continue
        before = previous_buckets.get((subject_type, key))
        previous_count = before.count if before else 0
        growth = percentage_change(previous_count, bucket.count)

        # Component scores, each in [0, 1].
        growth_score = clamp(growth / SPIKE_CEILING_PERCENT) if growth > 0 else 0.0
        volume_score = clamp(math.log1p(bucket.count) / math.log1p(max(peak_volume, 2)))
        breadth_score = clamp(math.log1p(len(bucket.sources)) / math.log1p(8))

        trend_score = round(
            clamp(0.45 * growth_score + 0.30 * volume_score + 0.25 * breadth_score), 4
        )

        # Confidence is about evidence, not excitement.
        confidence = round(
            clamp(
                0.5 * clamp(bucket.count / 10.0)
                + 0.3 * clamp(len(bucket.sources) / 4.0)
                + 0.2 * (1.0 if previous_count else 0.4)
            ),
            4,
        )

        results.append(
            TrendResult(
                subject_type=TrendSubject(subject_type),
                subject_key=key,
                subject_label=bucket.label or key,
                current_count=bucket.count,
                previous_count=previous_count,
                growth_percent=round(growth, 2),
                trend_score=trend_score,
                direction=_direction(previous_count, bucket.count, growth),
                confidence=confidence,
                avg_sentiment=round(bucket.avg_sentiment, 4),
                sentiment_delta=round(
                    bucket.avg_sentiment - (before.avg_sentiment if before else 0.0), 4
                ),
                source_count=len(bucket.sources),
                sample_article_ids=tuple(bucket.article_ids),
            )
        )

    results.sort(key=lambda result: (result.trend_score, result.current_count), reverse=True)
    return results[:limit]


def _direction(previous_count: int, current_count: int, growth: float) -> TrendDirection:
    if previous_count == 0:
        return TrendDirection.NEW
    if growth >= 25.0:
        return TrendDirection.RISING
    if growth <= -25.0:
        return TrendDirection.FALLING
    return TrendDirection.STABLE


def window_bounds(
    *, hours: int, now: datetime | None = None
) -> tuple[tuple[datetime, datetime], tuple[datetime, datetime]]:
    """``((current_start, current_end), (previous_start, previous_end))``."""
    end = now or utcnow()
    span = timedelta(hours=hours)
    current_start = end - span
    return (current_start, end), (current_start - span, current_start)


def top_by_type(
    results: list[TrendResult], subject_type: TrendSubject, limit: int = 10
) -> list[TrendResult]:
    """Filter results down to one subject type."""
    return [result for result in results if result.subject_type is subject_type][:limit]


def summarise_counts(observations: Iterable[SubjectObservation]) -> Counter[str]:
    """Plain frequency table, used by the dashboard's topic distribution."""
    counter: Counter[str] = Counter()
    for observation in observations:
        counter[observation.label or observation.key] += 1
    return counter


__all__ = [
    "DEFAULT_MIN_ARTICLES",
    "SubjectObservation",
    "TrendResult",
    "detect_trends",
    "summarise_counts",
    "top_by_type",
    "window_bounds",
]
