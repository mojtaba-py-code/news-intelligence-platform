"""Article listing, search and detail endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Path, Query

from app.api.dependencies import (
    ArticleRepoDep,
    CurrentUser,
    FeedServiceDep,
    PaginationDep,
)
from app.core.errors import ErrorResponse, NotFoundError
from app.database.models.article import SentimentLabel
from app.schemas.article import (
    ArticleDetail,
    ArticleRead,
    ArticleSearchQuery,
    ArticleSortField,
    ArticleStats,
    EntityRef,
    TopicRef,
)
from app.schemas.common import Page, SortOrder

router = APIRouter(prefix="/articles", tags=["articles"])


def _to_detail(article: object) -> ArticleDetail:
    """Build the detail view, flattening the topic/entity link rows."""
    detail = ArticleDetail.model_validate(article)
    detail.topics = [
        TopicRef(slug=link.topic.slug, name=link.topic.name, score=link.score)
        for link in getattr(article, "topics", [])
        if link.topic is not None
    ]
    detail.entities = [
        EntityRef(
            name=link.entity.name,
            entity_type=link.entity.entity_type,
            salience=link.salience,
            mentions=link.mentions,
        )
        for link in getattr(article, "entities", [])
        if link.entity is not None
    ]
    return detail


@router.get("", response_model=Page[ArticleRead], summary="List articles")
async def list_articles(
    repository: ArticleRepoDep,
    pagination: PaginationDep,
    source: Annotated[str | None, Query(max_length=64)] = None,
    category: Annotated[str | None, Query(max_length=48)] = None,
    language: Annotated[str | None, Query(max_length=8)] = None,
    country: Annotated[str | None, Query(max_length=8)] = None,
    sentiment: SentimentLabel | None = None,
    min_relevance: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    published_after: datetime | None = None,
    published_before: datetime | None = None,
    include_duplicates: bool = False,
    sort_by: ArticleSortField = ArticleSortField.PUBLISHED_AT,
    order: SortOrder = SortOrder.DESC,
) -> Page[ArticleRead]:
    """Paginated article feed with filtering and sorting.

    ``sort_by`` is an enum mapped to a column allowlist in the repository -
    arbitrary column names are not accepted.
    """
    query = ArticleSearchQuery(
        source=source,
        category=category,
        language=language,
        country=country,
        sentiment=sentiment,
        min_relevance=min_relevance,
        published_after=published_after,
        published_before=published_before,
        include_duplicates=include_duplicates,
        sort_by=sort_by,
        order=order,
    )
    rows, total = await repository.search(query, limit=pagination.limit, offset=pagination.offset)
    return Page.build([ArticleRead.model_validate(row) for row in rows], total, pagination)


@router.get("/search", response_model=Page[ArticleRead], summary="Search articles")
async def search_articles(
    repository: ArticleRepoDep,
    pagination: PaginationDep,
    q: Annotated[str | None, Query(max_length=200, description="Free-text query")] = None,
    phrase: Annotated[str | None, Query(max_length=200, description="Exact phrase")] = None,
    source: Annotated[str | None, Query(max_length=64)] = None,
    category: Annotated[str | None, Query(max_length=48)] = None,
    language: Annotated[str | None, Query(max_length=8)] = None,
    country: Annotated[str | None, Query(max_length=8)] = None,
    topic: Annotated[str | None, Query(max_length=64)] = None,
    entity: Annotated[str | None, Query(max_length=200)] = None,
    author: Annotated[str | None, Query(max_length=200)] = None,
    sentiment: SentimentLabel | None = None,
    min_sentiment: Annotated[float | None, Query(ge=-1.0, le=1.0)] = None,
    max_sentiment: Annotated[float | None, Query(ge=-1.0, le=1.0)] = None,
    min_relevance: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    published_after: datetime | None = None,
    published_before: datetime | None = None,
    include_duplicates: bool = False,
    sort_by: ArticleSortField = ArticleSortField.PUBLISHED_AT,
    order: SortOrder = SortOrder.DESC,
) -> Page[ArticleRead]:
    """Full search surface: keyword, phrase, facets, ranges and sorting."""
    query = ArticleSearchQuery(
        q=q,
        phrase=phrase,
        source=source,
        category=category,
        language=language,
        country=country,
        topic=topic,
        entity=entity,
        author=author,
        sentiment=sentiment,
        min_sentiment=min_sentiment,
        max_sentiment=max_sentiment,
        min_relevance=min_relevance,
        published_after=published_after,
        published_before=published_before,
        include_duplicates=include_duplicates,
        sort_by=sort_by,
        order=order,
    )
    rows, total = await repository.search(query, limit=pagination.limit, offset=pagination.offset)
    return Page.build([ArticleRead.model_validate(row) for row in rows], total, pagination)


@router.get("/stats", response_model=ArticleStats, summary="Article counters")
async def article_stats(repository: ArticleRepoDep) -> ArticleStats:
    return await repository.stats()


@router.get(
    "/feed",
    response_model=Page[ArticleRead],
    summary="Personalised intelligence feed",
)
async def personalized_feed(
    user: CurrentUser, service: FeedServiceDep, pagination: PaginationDep
) -> Page[ArticleRead]:
    """Articles re-ranked against the caller's stored preferences."""
    rows, total = await service.personalized(user, limit=pagination.limit, offset=pagination.offset)
    return Page.build([ArticleRead.model_validate(row) for row in rows], total, pagination)


@router.get(
    "/{article_id}",
    response_model=ArticleDetail,
    summary="Article detail",
    responses={404: {"model": ErrorResponse, "description": "Article not found"}},
)
async def get_article(
    repository: ArticleRepoDep, article_id: Annotated[int, Path(ge=1)]
) -> ArticleDetail:
    article = await repository.get_detail(article_id)
    if article is None:
        raise NotFoundError(f"Article {article_id} does not exist.")
    return _to_detail(article)


@router.get(
    "/{article_id}/similar",
    response_model=list[ArticleRead],
    summary="Related articles",
)
async def similar_articles(
    repository: ArticleRepoDep,
    service: FeedServiceDep,
    article_id: Annotated[int, Path(ge=1)],
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
) -> list[ArticleRead]:
    """Articles covering related ground, by shared topics and recency."""
    article = await repository.get_detail(article_id)
    if article is None:
        raise NotFoundError(f"Article {article_id} does not exist.")
    related = await service.recommendations(article, limit=limit)
    return [ArticleRead.model_validate(row) for row in related]


__all__ = ["router"]
