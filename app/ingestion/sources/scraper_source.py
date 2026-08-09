"""Web-scraper connector: crawl an index page, then extract each article.

Deliberately conservative. It honours ``robots.txt`` (enforced in the fetcher),
spaces requests using the source's configured delay, follows a bounded number
of links per run, and never leaves the source's own host. Scraping is the last
resort - a feed or an API is always preferred when the publisher offers one.
"""

from __future__ import annotations

import asyncio
from typing import Any, Final

from app.core.errors import FetchError, ParseError, SourceError, UnsafeURLError
from app.core.logging import get_logger
from app.core.url_safety import extract_domain
from app.database.models.source import SourceKind
from app.ingestion.parsers.html import ExtractionRules, HTMLArticleParser
from app.ingestion.sources.base import NewsSource

logger = get_logger(__name__)

HTML_ACCEPT: Final[str] = "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5"
MAX_ARTICLE_PAGES: Final[int] = 40


class WebScraperSource(NewsSource):
    """Two-phase scraper: link discovery, then per-article extraction.

    Config keys
    -----------
    ``selectors``
        A :class:`ExtractionRules` block (``title``, ``content``,
        ``article_links``, ``link_must_contain``, …).
    ``max_pages``
        How many article pages to fetch per run (hard-capped at 40).
    ``same_domain_only``
        Default ``true``. Prevents a compromised or spammy index page from
        redirecting the crawler to unrelated hosts.
    ``concurrency``
        Parallel article fetches (default 2, capped at 4 - politeness first).
    """

    kind = SourceKind.SCRAPER

    async def _collect(self) -> list[dict[str, Any]]:
        rules = ExtractionRules.from_config(self.context.config)
        parser = HTMLArticleParser(rules)

        index = await self.get(self.context.url, accept=HTML_ACCEPT)
        links = parser.extract_links(index.text, base_url=index.final_url or self.context.url)

        if bool(self.context.option("same_domain_only", True)):
            domain = extract_domain(self.context.url)
            links = [link for link in links if extract_domain(link) == domain]

        limit = min(
            int(self.context.option("max_pages", 20) or 20),
            MAX_ARTICLE_PAGES,
            self.context.max_articles,
        )
        links = links[:limit]
        if not links:
            raise ParseError(self.slug, "No article links found on the index page")

        concurrency = max(1, min(int(self.context.option("concurrency", 2) or 2), 4))
        semaphore = asyncio.Semaphore(concurrency)

        async def scrape(link: str) -> dict[str, Any] | None:
            async with semaphore:
                return await self._scrape_article(link, parser)

        results = await asyncio.gather(*(scrape(link) for link in links), return_exceptions=True)

        items: list[dict[str, Any]] = []
        failures = 0
        for result in results:
            if isinstance(result, BaseException):
                failures += 1
                continue
            if result:
                items.append(result)

        if not items:
            raise ParseError(
                self.slug, f"All {len(links)} article pages failed ({failures} errors)"
            )
        if failures:
            logger.info(
                "scraper_partial_failure",
                extra={"source": self.slug, "ok": len(items), "failed": failures},
            )
        return items

    async def _scrape_article(self, url: str, parser: HTMLArticleParser) -> dict[str, Any] | None:
        """Fetch and extract one article page; failures are per-page, not fatal."""
        try:
            response = await self.get(url, accept=HTML_ACCEPT)
        except (SourceError, FetchError, UnsafeURLError) as exc:
            logger.debug(
                "article_fetch_failed",
                extra={"source": self.slug, "error_type": exc.__class__.__name__},
            )
            return None

        extracted = parser.parse(response.text, url=response.final_url or url)
        if not extracted.get("title"):
            return None

        return {
            "title": extracted["title"],
            "url": extracted.get("canonical_url") or response.final_url or url,
            "description": extracted.get("description"),
            "content": extracted.get("content"),
            "author": extracted.get("author"),
            "image_url": extracted.get("image_url"),
            "published_at": extracted.get("published_at"),
            "updated_at": extracted.get("updated_at"),
            "language": extracted.get("language") or self.context.language,
            "category": extracted.get("category") or self.context.category,
        }


__all__ = ["WebScraperSource"]
