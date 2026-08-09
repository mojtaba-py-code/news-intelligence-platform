"""The secure fetcher and the source connectors, driven by mocked HTTP."""

from __future__ import annotations

import httpx
import pytest
import respx

from app.core.config import get_settings
from app.core.errors import (
    CircuitOpenError,
    FetchError,
    RateLimitedUpstreamError,
    UnsafeURLError,
)
from app.core.resilience import breakers
from app.database.models.source import SourceKind
from app.ingestion.fetchers.http import SecureHTTPFetcher
from app.ingestion.sources.api_source import APINewsSource
from app.ingestion.sources.base import SourceContext
from app.ingestion.sources.json_source import JSONFeedSource
from app.ingestion.sources.registry import (
    SourceDefinition,
    build_source,
    load_source_definitions,
    registry,
)
from app.ingestion.sources.rss_source import RSSNewsSource
from app.ingestion.sources.scraper_source import WebScraperSource

pytestmark = pytest.mark.integration

FEED = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>Example</title><link>https://feed.example.com</link><description>d</description>
  <item>
    <title>First real headline about the market</title>
    <link>https://feed.example.com/a</link>
    <description>Summary of the first story for readers.</description>
    <pubDate>Mon, 05 Jan 2026 10:00:00 GMT</pubDate>
  </item>
  <item>
    <title>Second real headline about technology</title>
    <link>https://feed.example.com/b</link>
    <description>Summary of the second story for readers.</description>
    <pubDate>Mon, 05 Jan 2026 11:00:00 GMT</pubDate>
  </item>
</channel></rss>
"""


@pytest.fixture
def config():
    """Settings with SSRF checks relaxed so ``*.example.com`` can be mocked."""
    return get_settings().model_copy(
        update={
            "ssrf_protection_enabled": False,
            "respect_robots_txt": False,
            "http_max_retries": 1,
            "http_backoff_base": 0.0,
            "default_request_delay_seconds": 0.0,
        }
    )


@pytest.fixture
async def fetcher(config):
    instance = SecureHTTPFetcher(config=config)
    try:
        yield instance
    finally:
        await instance.aclose()


def context(
    kind: SourceKind = SourceKind.RSS, url: str = "https://feed.example.com/rss", **options
):
    return SourceContext(
        slug="example",
        name="Example",
        kind=kind,
        url=url,
        config=options,
        max_articles=50,
        request_delay=0.0,
        respect_robots=False,
    )


class TestSecureFetcher:
    @respx.mock
    async def test_successful_fetch(self, fetcher: SecureHTTPFetcher) -> None:
        respx.get("https://feed.example.com/rss").mock(
            return_value=httpx.Response(200, text=FEED, headers={"content-type": "text/xml"})
        )
        response = await fetcher.fetch("https://feed.example.com/rss", source="example")
        assert response.ok
        assert "First real headline" in response.text
        assert response.content_type == "text/xml"

    @respx.mock
    async def test_transient_error_is_retried(self, fetcher: SecureHTTPFetcher) -> None:
        route = respx.get("https://feed.example.com/rss").mock(
            side_effect=[
                httpx.Response(503),
                httpx.Response(200, text="ok", headers={"content-type": "text/plain"}),
            ]
        )
        response = await fetcher.fetch("https://feed.example.com/rss", source="example")
        assert response.ok
        assert route.call_count == 2

    @respx.mock
    async def test_client_error_is_not_retried(self, fetcher: SecureHTTPFetcher) -> None:
        route = respx.get("https://feed.example.com/missing").mock(return_value=httpx.Response(404))
        with pytest.raises(FetchError):
            await fetcher.fetch("https://feed.example.com/missing", source="example")
        assert route.call_count == 1

    @respx.mock
    async def test_rate_limited_upstream(self, fetcher: SecureHTTPFetcher) -> None:
        route = respx.get("https://feed.example.com/rss").mock(
            return_value=httpx.Response(429, headers={"retry-after": "30"})
        )
        with pytest.raises(RateLimitedUpstreamError):
            await fetcher.fetch("https://feed.example.com/rss", source="example")
        # 429 is transient, so it is retried within the attempt budget.
        assert route.call_count > 1

    @respx.mock
    async def test_redirects_are_followed_and_revalidated(self, fetcher: SecureHTTPFetcher) -> None:
        respx.get("https://feed.example.com/start").mock(
            return_value=httpx.Response(302, headers={"location": "https://feed.example.com/end"})
        )
        respx.get("https://feed.example.com/end").mock(
            return_value=httpx.Response(200, text="done", headers={"content-type": "text/plain"})
        )
        response = await fetcher.fetch("https://feed.example.com/start", source="example")
        assert response.text == "done"
        assert response.final_url.endswith("/end")

    @respx.mock
    async def test_redirect_to_a_private_address_is_blocked(self, config) -> None:
        """The SSRF guard must run on every hop, not just the first."""
        strict = config.model_copy(update={"ssrf_protection_enabled": True})
        instance = SecureHTTPFetcher(config=strict)
        respx.get("https://93.184.216.34/start").mock(
            return_value=httpx.Response(302, headers={"location": "http://169.254.169.254/creds"})
        )
        try:
            with pytest.raises(UnsafeURLError):
                await instance.fetch("https://93.184.216.34/start", source="example")
        finally:
            await instance.aclose()

    @respx.mock
    async def test_authorization_is_dropped_across_origins(
        self, fetcher: SecureHTTPFetcher
    ) -> None:
        respx.get("https://a.example.com/start").mock(
            return_value=httpx.Response(302, headers={"location": "https://b.example.com/end"})
        )
        final = respx.get("https://b.example.com/end").mock(
            return_value=httpx.Response(200, text="ok", headers={"content-type": "text/plain"})
        )
        await fetcher.fetch(
            "https://a.example.com/start",
            source="example",
            headers={"Authorization": "Bearer secret-token"},
        )
        assert "authorization" not in {key.lower() for key in final.calls[0].request.headers}

    @respx.mock
    async def test_oversized_response_is_refused(self, config) -> None:
        small = config.model_copy(update={"http_max_response_bytes": 1024})
        instance = SecureHTTPFetcher(config=small)
        respx.get("https://feed.example.com/big").mock(
            return_value=httpx.Response(
                200, content=b"x" * 5000, headers={"content-type": "text/plain"}
            )
        )
        try:
            with pytest.raises(FetchError, match="maximum allowed size"):
                await instance.fetch("https://feed.example.com/big", source="example")
        finally:
            await instance.aclose()

    @respx.mock
    async def test_unsupported_content_type_is_refused(self, fetcher: SecureHTTPFetcher) -> None:
        respx.get("https://feed.example.com/binary").mock(
            return_value=httpx.Response(
                200, content=b"\x00\x01", headers={"content-type": "application/octet-stream"}
            )
        )
        with pytest.raises(FetchError, match="content type"):
            await fetcher.fetch("https://feed.example.com/binary", source="example")

    @respx.mock
    async def test_redirect_loop_is_bounded(self, fetcher: SecureHTTPFetcher) -> None:
        respx.get("https://feed.example.com/loop").mock(
            return_value=httpx.Response(302, headers={"location": "https://feed.example.com/loop"})
        )
        with pytest.raises(FetchError, match="redirects"):
            await fetcher.fetch("https://feed.example.com/loop", source="example")

    @respx.mock
    async def test_circuit_opens_after_repeated_failures(self, fetcher: SecureHTTPFetcher) -> None:
        breakers.clear()
        route = respx.get("https://feed.example.com/down").mock(return_value=httpx.Response(500))
        for _ in range(5):
            with pytest.raises((FetchError, CircuitOpenError)):
                await fetcher.fetch("https://feed.example.com/down", source="flaky")

        assert breakers.get("flaky").is_open
        calls_before = route.call_count
        with pytest.raises(CircuitOpenError):
            await fetcher.fetch("https://feed.example.com/down", source="flaky")
        # The open circuit short-circuits, so no further request is made.
        assert route.call_count == calls_before

    async def test_ssrf_guard_rejects_before_any_request(self, config) -> None:
        strict = config.model_copy(update={"ssrf_protection_enabled": True})
        instance = SecureHTTPFetcher(config=strict)
        try:
            with pytest.raises(UnsafeURLError):
                await instance.fetch("http://127.0.0.1:6379/", source="example")
        finally:
            await instance.aclose()

    @respx.mock
    async def test_robots_disallow_is_honoured(self, config) -> None:
        polite = config.model_copy(update={"respect_robots_txt": True})
        instance = SecureHTTPFetcher(config=polite)
        respx.get("https://feed.example.com/robots.txt").mock(
            return_value=httpx.Response(
                200,
                text="User-agent: *\nDisallow: /private\n",
                headers={"content-type": "text/plain"},
            )
        )
        try:
            with pytest.raises(FetchError, match="robots"):
                await instance.fetch("https://feed.example.com/private/page", source="example")
        finally:
            await instance.aclose()


class TestConnectors:
    @respx.mock
    async def test_rss_connector(self, fetcher: SecureHTTPFetcher, config) -> None:
        respx.get("https://feed.example.com/rss").mock(
            return_value=httpx.Response(
                200, text=FEED, headers={"content-type": "application/rss+xml"}
            )
        )
        source = RSSNewsSource(context(), fetcher=fetcher, config=config)
        outcome = await source.fetch()
        assert outcome.success
        assert len(outcome.articles) == 2
        assert outcome.articles[0].source_slug == "example"
        assert outcome.articles[0].url == "https://feed.example.com/a"

    @respx.mock
    async def test_rss_connector_reports_parse_failure(
        self, fetcher: SecureHTTPFetcher, config
    ) -> None:
        respx.get("https://feed.example.com/rss").mock(
            return_value=httpx.Response(
                200, text="<not-a-feed/>", headers={"content-type": "text/xml"}
            )
        )
        outcome = await RSSNewsSource(context(), fetcher=fetcher, config=config).fetch()
        assert outcome.success is False
        assert outcome.error_type == "ParseError"
        assert outcome.articles == []

    @respx.mock
    async def test_api_connector_with_header_auth(
        self, fetcher: SecureHTTPFetcher, config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_NEWS_KEY", "secret-key-value")
        route = respx.get("https://api.example.com/v2/news").mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "ok",
                    "articles": [
                        {
                            "title": "API story about the economy",
                            "url": "https://api.example.com/story-1",
                            "publishedAt": "2026-01-05T10:00:00Z",
                            "description": "Summary",
                            "source": {"name": "API Source"},
                        }
                    ],
                },
                headers={"content-type": "application/json"},
            )
        )
        ctx = context(
            SourceKind.API,
            "https://api.example.com/v2/news",
            auth="header",
            auth_header="X-Api-Key",
            mapping={"items_path": "articles"},
        )
        ctx.api_key_env = "TEST_NEWS_KEY"

        outcome = await APINewsSource(ctx, fetcher=fetcher, config=config).fetch()
        assert outcome.success
        assert outcome.articles[0].title == "API story about the economy"
        assert route.calls[0].request.headers["X-Api-Key"] == "secret-key-value"

    @respx.mock
    async def test_api_connector_surfaces_error_bodies(
        self, fetcher: SecureHTTPFetcher, config
    ) -> None:
        respx.get("https://api.example.com/v2/news").mock(
            return_value=httpx.Response(
                200,
                json={"status": "error", "message": "invalid key"},
                headers={"content-type": "application/json"},
            )
        )
        ctx = context(SourceKind.API, "https://api.example.com/v2/news", auth="none")
        outcome = await APINewsSource(ctx, fetcher=fetcher, config=config).fetch()
        assert outcome.success is False
        assert "invalid key" in (outcome.error or "")

    async def test_api_connector_requires_its_credential(self, fetcher, config) -> None:
        ctx = context(SourceKind.API, "https://api.example.com/v2/news", auth="header")
        ctx.api_key_env = "DEFINITELY_NOT_SET_12345"
        outcome = await APINewsSource(ctx, fetcher=fetcher, config=config).fetch()
        assert outcome.success is False
        assert "DEFINITELY_NOT_SET_12345" in (outcome.error or "")

    @respx.mock
    async def test_json_feed_connector(self, fetcher: SecureHTTPFetcher, config) -> None:
        respx.get("https://json.example.com/feed").mock(
            return_value=httpx.Response(
                200,
                json={
                    "version": "https://jsonfeed.org/version/1.1",
                    "language": "en",
                    "items": [
                        {
                            "id": "1",
                            "title": "JSON feed headline about science",
                            "url": "https://json.example.com/1",
                            "content_html": "<p>Body</p>",
                            "date_published": "2026-01-05T10:00:00Z",
                        }
                    ],
                },
                headers={"content-type": "application/feed+json"},
            )
        )
        ctx = context(SourceKind.JSON_FEED, "https://json.example.com/feed")
        outcome = await JSONFeedSource(ctx, fetcher=fetcher, config=config).fetch()
        assert outcome.success
        assert outcome.articles[0].language == "en"

    @respx.mock
    async def test_scraper_connector(self, fetcher: SecureHTTPFetcher, config) -> None:
        index = """
        <html><body>
          <a class="story" href="/news/one">One</a>
          <a class="story" href="/news/two">Two</a>
        </body></html>
        """
        article = """
        <html><head><meta property="og:title" content="Scraped headline about policy"></head>
        <body><article><div class="article-body">
          <p>The council approved the plan after a debate lasting several hours yesterday.</p>
        </div></article></body></html>
        """
        respx.get("https://site.example.com/news").mock(
            return_value=httpx.Response(200, text=index, headers={"content-type": "text/html"})
        )
        respx.get("https://site.example.com/news/one").mock(
            return_value=httpx.Response(200, text=article, headers={"content-type": "text/html"})
        )
        respx.get("https://site.example.com/news/two").mock(
            return_value=httpx.Response(200, text=article, headers={"content-type": "text/html"})
        )

        ctx = context(
            SourceKind.SCRAPER,
            "https://site.example.com/news",
            selectors={"article_links": "a.story", "content": "div.article-body"},
            same_domain_only=True,
            concurrency=1,
        )
        outcome = await WebScraperSource(ctx, fetcher=fetcher, config=config).fetch()
        assert outcome.success
        assert outcome.articles
        assert outcome.articles[0].title == "Scraped headline about policy"

    @respx.mock
    async def test_scraper_tolerates_individual_page_failures(
        self, fetcher: SecureHTTPFetcher, config
    ) -> None:
        index = '<a class="story" href="/news/ok">ok</a><a class="story" href="/news/bad">bad</a>'
        page = (
            '<html><head><meta property="og:title" content="Working page headline here">'
            "</head><body><article><p>"
            "Enough body text for the extractor to consider this a real paragraph."
            "</p></article></body></html>"
        )
        respx.get("https://site.example.com/news").mock(
            return_value=httpx.Response(200, text=index, headers={"content-type": "text/html"})
        )
        respx.get("https://site.example.com/news/ok").mock(
            return_value=httpx.Response(200, text=page, headers={"content-type": "text/html"})
        )
        respx.get("https://site.example.com/news/bad").mock(return_value=httpx.Response(500))

        ctx = context(
            SourceKind.SCRAPER,
            "https://site.example.com/news",
            selectors={"article_links": "a.story"},
            concurrency=1,
        )
        outcome = await WebScraperSource(ctx, fetcher=fetcher, config=config).fetch()
        assert outcome.success
        assert len(outcome.articles) == 1


class TestRegistry:
    def test_every_kind_has_a_connector(self) -> None:
        assert set(registry.kinds) >= {"rss", "api", "scraper", "json_feed"}

    def test_build_returns_the_right_class(self) -> None:
        assert isinstance(build_source(context(SourceKind.RSS)), RSSNewsSource)
        assert isinstance(
            build_source(context(SourceKind.API, "https://api.example.com")), APINewsSource
        )

    def test_unknown_kind_raises(self) -> None:
        from app.core.errors import ConfigurationError

        with pytest.raises(ConfigurationError):
            registry.get("telepathy")

    def test_yaml_catalogue_loads_and_validates(self, tmp_path) -> None:
        path = tmp_path / "sources.yaml"
        path.write_text(
            """
sources:
  - slug: good-source
    name: Good Source
    kind: rss
    url: https://93.184.216.34/feed.xml
  - slug: local-source
    name: Local
    kind: rss
    url: http://127.0.0.1/feed.xml
  - slug: inline-secret
    name: Bad
    kind: api
    url: https://93.184.216.34/api
    config:
      api_key: leaked-value
  - name: no slug
    url: https://93.184.216.34/x
""",
            encoding="utf-8",
        )
        definitions = load_source_definitions(path)
        slugs = {definition.slug for definition in definitions}
        assert slugs == {"good-source"}  # the other three are rejected

    def test_definition_converts_to_row_and_context(self) -> None:
        definition = SourceDefinition(
            slug="s", name="S", kind=SourceKind.RSS, url="https://93.184.216.34/feed"
        )
        row = definition.to_row()
        assert row["kind"] == "rss"
        assert definition.to_context().slug == "s"
