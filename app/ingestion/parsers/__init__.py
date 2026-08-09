"""Format parsers: RSS/Atom, JSON feeds and HTML pages."""

from app.ingestion.parsers.html import ExtractionRules, HTMLArticleParser
from app.ingestion.parsers.json_feed import JSONFieldMapping, parse_json_items
from app.ingestion.parsers.rss import FeedItem, parse_feed

__all__ = [
    "ExtractionRules",
    "FeedItem",
    "HTMLArticleParser",
    "JSONFieldMapping",
    "parse_feed",
    "parse_json_items",
]
