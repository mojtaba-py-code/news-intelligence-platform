"""JSON Feed (jsonfeed.org) connector.

A thin specialisation of the API connector: JSON Feed has a fixed schema, so
the field mapping is known in advance and no authentication is involved.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import ParseError
from app.database.models.source import SourceKind
from app.ingestion.parsers.json_feed import JSONFieldMapping, parse_json_items
from app.ingestion.sources.base import NewsSource

JSON_FEED_MAPPING = JSONFieldMapping(
    items_path="items",
    title=("title",),
    url=("url", "external_url", "id"),
    description=("summary",),
    content=("content_html", "content_text"),
    author=("author.name", "authors.0.name"),
    published_at=("date_published",),
    updated_at=("date_modified",),
    image_url=("image", "banner_image"),
    external_id=("id",),
    language=("language",),
    category=("tags.0",),
    source_name=(),
)


class JSONFeedSource(NewsSource):
    """Reads a JSON Feed document."""

    kind = SourceKind.JSON_FEED

    async def _collect(self) -> list[dict[str, Any]]:
        response = await self.get(
            self.context.url, accept="application/feed+json, application/json"
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise ParseError(self.slug, "Response body is not valid JSON") from exc

        if not isinstance(payload, dict) or "items" not in payload:
            raise ParseError(self.slug, "Document does not look like a JSON Feed")

        feed_language = payload.get("language")
        items = parse_json_items(
            payload, JSON_FEED_MAPPING, source_slug=self.slug, source_name=self.context.name
        )
        for item in items:
            if not item.get("language"):
                item["language"] = feed_language
        return items[: self.context.max_articles]


__all__ = ["JSON_FEED_MAPPING", "JSONFeedSource"]
