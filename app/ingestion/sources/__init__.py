"""Source connectors. Every connector implements :class:`NewsSource`."""

from app.ingestion.sources.api_source import APINewsSource
from app.ingestion.sources.base import FetchOutcome, NewsSource, SourceContext
from app.ingestion.sources.json_source import JSONFeedSource
from app.ingestion.sources.registry import (
    SourceRegistry,
    build_source,
    register_source_type,
    registry,
)
from app.ingestion.sources.rss_source import RSSNewsSource
from app.ingestion.sources.scraper_source import WebScraperSource

__all__ = [
    "APINewsSource",
    "FetchOutcome",
    "JSONFeedSource",
    "NewsSource",
    "RSSNewsSource",
    "SourceContext",
    "SourceRegistry",
    "WebScraperSource",
    "build_source",
    "register_source_type",
    "registry",
]
