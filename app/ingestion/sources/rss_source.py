"""RSS / Atom / RDF connector."""

from __future__ import annotations

from typing import Any

from app.core.errors import ParseError
from app.core.logging import get_logger
from app.database.models.source import SourceKind
from app.ingestion.parsers.rss import FeedItem, parse_feed
from app.ingestion.sources.base import NewsSource

logger = get_logger(__name__)

FEED_ACCEPT = (
    "application/rss+xml, application/atom+xml, application/xml;q=0.9, text/xml;q=0.9, */*;q=0.5"
)


class RSSNewsSource(NewsSource):
    """Reads one or more feed URLs and maps entries to raw articles.

    Config keys
    -----------
    ``feeds``
        Extra feed URLs beyond ``url`` (a publisher often exposes one per
        section). Each is fetched independently so one broken feed does not
        cost the others.
    ``prefer_content``
        When ``true`` (default) ``content:encoded`` is used as the body;
        otherwise only the summary is kept.
    """

    kind = SourceKind.RSS

    async def _collect(self) -> list[dict[str, Any]]:
        feeds: list[str] = [self.context.url]
        extra = self.context.option("feeds", [])
        if isinstance(extra, list):
            feeds.extend(str(url) for url in extra[:20])

        prefer_content = bool(self.context.option("prefer_content", True))
        items: list[dict[str, Any]] = []
        failures: list[str] = []

        for feed_url in feeds:
            if len(items) >= self.context.max_articles:
                break
            try:
                response = await self.get(feed_url, accept=FEED_ACCEPT)
                parsed = parse_feed(response.content, source=self.slug)
            except ParseError as exc:
                failures.append(f"{feed_url}: {exc.message}")
                logger.warning("feed_parse_failed", extra={"source": self.slug})
                continue

            for entry in parsed.items:
                items.append(
                    self._map(entry, feed_language=parsed.language, prefer_content=prefer_content)
                )
                if len(items) >= self.context.max_articles:
                    break

        if not items and failures:
            # Every feed failed - report it rather than silently returning zero.
            raise ParseError(self.slug, "; ".join(failures)[:400])
        return items

    def _map(self, entry: FeedItem, *, feed_language: str, prefer_content: bool) -> dict[str, Any]:
        body = entry.content if (prefer_content and entry.content) else ""
        return {
            "title": entry.title,
            "url": entry.link,
            "external_id": entry.guid or None,
            "description": entry.description or None,
            "content": body or entry.description or None,
            "author": entry.author or None,
            "image_url": entry.image or None,
            "published_at": entry.published or entry.updated or None,
            "updated_at": entry.updated or None,
            "language": self.context.language or feed_language or None,
            "category": (entry.categories[0] if entry.categories else self.context.category),
            "raw": {"categories": entry.categories[:5]} if entry.categories else {},
        }


__all__ = ["RSSNewsSource"]
