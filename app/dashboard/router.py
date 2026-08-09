"""Dashboard routes.

The page is rendered server-side with Jinja2 (autoescaping on) and refreshed
client-side from the public analytics endpoints. All assets are local, so the
strict Content-Security-Policy in :mod:`app.api.middleware` holds without
``unsafe-inline`` anywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app import __version__
from app.api.dependencies import AnalyticsServiceDep, ArticleRepoDep, EventRepoDep, TrendRepoDep
from app.core.config import get_settings
from app.schemas.article import ArticleSearchQuery

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
# Explicit: Jinja2 autoescape defaults to off for non-.html loaders, and this
# page renders article titles that came from third-party feeds.
templates.env.autoescape = True

router = APIRouter(tags=["dashboard"], include_in_schema=False)


@router.get("/", response_class=HTMLResponse, summary="Analytics dashboard")
async def dashboard(
    request: Request,
    analytics: AnalyticsServiceDep,
    articles: ArticleRepoDep,
    trends: TrendRepoDep,
    events: EventRepoDep,
    hours: Annotated[int, Query(ge=1, le=168)] = 24,
) -> HTMLResponse:
    """Render the dashboard with a server-side first paint."""
    config = get_settings()
    overview = await analytics.overview()
    sentiment = await analytics.sentiment(hours=hours)
    topics = await analytics.topics(hours=hours, limit=10)
    source_stats = await analytics.source_stats(hours=hours)
    timeseries = await analytics.timeseries(hours=hours)
    latest, _ = await articles.search(ArticleSearchQuery(), limit=12, offset=0)
    trending = await trends.latest(limit=10, min_score=0.0)
    recent_events, _ = await events.recent(limit=6, offset=0)

    peak = max((point.count for point in timeseries), default=1) or 1
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "version": __version__,
            "environment": str(config.environment),
            "hours": hours,
            "overview": overview,
            "sentiment": sentiment,
            "topics": topics,
            "sources": source_stats[:10],
            "timeseries": [
                {
                    "label": point.timestamp.strftime("%H:%M"),
                    "count": point.count,
                    "height": round(100 * point.count / peak),
                    "sentiment": point.avg_sentiment,
                }
                for point in timeseries[-24:]
            ],
            "articles": latest,
            "trends": trending,
            "events": recent_events,
        },
    )


__all__ = ["STATIC_DIR", "router"]
