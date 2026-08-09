"""Event detection: grouping articles that cover the same real-world story.

Deduplication asks "is this the *same article*?"; event clustering asks "is this
the *same story*?". The threshold is therefore much lower, and cross-source
agreement is the point rather than something to suppress.

Algorithm: single-pass incremental clustering over a TF-IDF space, ordered by
publication time. Each article joins the nearest existing cluster above the
similarity threshold or starts a new one. This is O(n·k) rather than O(n²) over
the whole corpus, and the time ordering means the earliest article naturally
becomes the cluster's anchor.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from app.core.utils import clamp, sha256_text, utcnow
from app.processing.deduplication.similarity import TfidfIndex, centroid, cosine_similarity

#: Below this cosine similarity two articles are treated as different stories.
DEFAULT_THRESHOLD: float = 0.55
#: A cluster needs at least this many articles to be worth calling an event.
MIN_CLUSTER_SIZE: int = 2


class ClusterableArticle(Protocol):
    """Structural type of the rows the clusterer consumes."""

    id: int
    title: str
    description: str | None
    content: str | None
    source_name: str
    published_at: datetime
    sentiment_score: float
    keywords: list[str]


@dataclass
class EventCluster:
    """A group of articles about one story."""

    article_ids: list[int] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)
    keywords: list[str] = field(default_factory=list)
    similarities: list[float] = field(default_factory=list)
    sentiment_total: float = 0.0
    first_seen: datetime | None = None
    last_updated: datetime | None = None
    vectors: list[dict[str, float]] = field(default_factory=list)
    _centroid: dict[str, float] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.article_ids)

    @property
    def source_count(self) -> int:
        return len(self.sources)

    @property
    def avg_sentiment(self) -> float:
        return self.sentiment_total / self.size if self.size else 0.0

    @property
    def title(self) -> str:
        """Representative headline: the shortest one, which is usually the
        least editorialised phrasing of the underlying fact."""
        return min(self.titles, key=len) if self.titles else ""

    @property
    def key(self) -> str:
        """Stable identifier derived from the anchor article and headline."""
        anchor = self.article_ids[0] if self.article_ids else 0
        return sha256_text(f"{anchor}|{self.title.casefold()}")[:40]

    @property
    def importance(self) -> float:
        """Corroboration-weighted significance in ``[0, 1]``.

        Independent sources dominate: eight outlets reporting once each is a
        bigger story than one outlet publishing eight follow-ups.
        """
        breadth = clamp(math.log1p(self.source_count) / math.log1p(10))
        volume = clamp(math.log1p(self.size) / math.log1p(20))
        cohesion = (
            clamp(sum(self.similarities) / len(self.similarities)) if self.similarities else 0.5
        )
        intensity = clamp(abs(self.avg_sentiment))
        return round(clamp(0.45 * breadth + 0.25 * volume + 0.2 * cohesion + 0.1 * intensity), 4)

    def add(
        self,
        *,
        article_id: int,
        title: str,
        source: str,
        published_at: datetime,
        sentiment: float,
        keywords: Sequence[str],
        vector: dict[str, float],
        similarity: float,
    ) -> None:
        self.article_ids.append(article_id)
        self.titles.append(title)
        self.sources.add(source)
        self.sentiment_total += sentiment
        self.vectors.append(vector)
        self.similarities.append(similarity)
        for keyword in keywords:
            if keyword not in self.keywords and len(self.keywords) < 20:
                self.keywords.append(keyword)
        if self.first_seen is None or published_at < self.first_seen:
            self.first_seen = published_at
        if self.last_updated is None or published_at > self.last_updated:
            self.last_updated = published_at
        self._centroid = {}

    def centroid_vector(self) -> dict[str, float]:
        if not self._centroid:
            self._centroid = centroid(self.vectors)
        return self._centroid

    def similarity_to(self, vector: dict[str, float]) -> float:
        return cosine_similarity(vector, self.centroid_vector())


def cluster_articles(
    articles: Sequence[ClusterableArticle],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    require_distinct_sources: bool = True,
) -> list[EventCluster]:
    """Group ``articles`` into event clusters.

    ``require_distinct_sources`` keeps a single outlet's serial coverage of one
    topic from being promoted to an "event" on its own.
    """
    if len(articles) < min_cluster_size:
        return []

    documents = [_document(article) for article in articles]
    index = TfidfIndex().fit(documents)
    vectors = [index.transform(document) for document in documents]

    order = sorted(range(len(articles)), key=lambda i: articles[i].published_at)
    clusters: list[EventCluster] = []

    for position in order:
        article = articles[position]
        vector = vectors[position]
        if not vector:
            continue

        best_cluster: EventCluster | None = None
        best_similarity = 0.0
        for cluster in clusters:
            similarity = cluster.similarity_to(vector)
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster

        if best_cluster is not None and best_similarity >= threshold:
            target, similarity = best_cluster, best_similarity
        else:
            target, similarity = EventCluster(), 1.0
            clusters.append(target)

        target.add(
            article_id=article.id,
            title=article.title,
            source=article.source_name,
            published_at=article.published_at,
            sentiment=article.sentiment_score,
            keywords=list(article.keywords or []),
            vector=vector,
            similarity=similarity,
        )

    result = [cluster for cluster in clusters if cluster.size >= min_cluster_size]
    if require_distinct_sources:
        result = [cluster for cluster in result if cluster.source_count >= 2]
    result.sort(key=lambda cluster: cluster.importance, reverse=True)
    return result


def _document(article: ClusterableArticle) -> str:
    """Text used for clustering: headline (doubled) plus the opening body."""
    body = (article.content or article.description or "")[:1500]
    keywords = " ".join(article.keywords or [])
    return f"{article.title} {article.title} {keywords} {body}"


def detect_breaking(
    clusters: Sequence[EventCluster], *, window_hours: int = 6, now: datetime | None = None
) -> list[EventCluster]:
    """Clusters that formed recently across several sources - i.e. breaking news."""
    reference = now or utcnow()
    breaking: list[EventCluster] = []
    for cluster in clusters:
        if cluster.first_seen is None:
            continue
        age_hours = (reference - cluster.first_seen).total_seconds() / 3600
        if age_hours <= window_hours and cluster.source_count >= 3:
            breaking.append(cluster)
    return breaking


__all__ = [
    "DEFAULT_THRESHOLD",
    "MIN_CLUSTER_SIZE",
    "ClusterableArticle",
    "EventCluster",
    "cluster_articles",
    "detect_breaking",
]
