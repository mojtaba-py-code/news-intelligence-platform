"""End-to-end: feed → ingestion → NLP → storage → analytics → API → dashboard.

This is the test that would catch a wiring mistake no unit test can see: it
runs the real connectors against mocked HTTP, the real pipeline against a real
database, and then reads the results back through the public HTTP surface.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
import respx
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.utils import utcnow
from app.database.models.article import Article
from app.database.models.source import Source, SourceKind
from app.database.models.user import User
from app.database.repositories.taxonomy import TopicRepository
from app.ingestion.pipeline import IngestionPipeline
from app.intelligence.topics import default_topic_seed
from app.services.analytics import AnalyticsService
from app.services.events import EventService
from app.services.trends import TrendService

pytestmark = pytest.mark.e2e


def feed(publisher: str, stories: list[tuple[str, str, str]]) -> str:
    """Build an RSS document from ``(slug, headline, body)`` triples."""
    items = "".join(
        f"""
        <item>
          <title>{headline}</title>
          <link>https://{publisher}.example.com/{slug}</link>
          <description>{body[:140]}</description>
          <content:encoded><![CDATA[<p>{body}</p>]]></content:encoded>
          <dc:creator>Staff Reporter</dc:creator>
          <pubDate>{utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")}</pubDate>
          <guid>{publisher}-{slug}</guid>
        </item>
        """
        for slug, headline, body in stories
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/"
     xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <title>{publisher.title()}</title>
    <link>https://{publisher}.example.com</link>
    <description>{publisher} feed</description>
    <language>en</language>
    {items}
  </channel>
</rss>
"""


# The same story covered by three publishers, plus unrelated coverage.
QUAKE = (
    "A powerful earthquake struck the coastal region early on Tuesday morning, collapsing "
    "residential buildings and prompting a large rescue operation. Emergency services said "
    "hundreds of workers had been deployed and that the search would continue overnight."
)
AI_STORY = (
    "OpenAI released a new language model that the company said outperforms every previous "
    "system on reasoning benchmarks. Microsoft, an investor, will offer it through its cloud."
)
MARKET_STORY = (
    "The central bank raised interest rates by half a percentage point, citing persistent "
    "inflation. Analysts at Goldman Sachs expect one further increase before the year ends."
)
CYBER_STORY = (
    "A ransomware group exploited a zero-day vulnerability in a widely used VPN appliance, "
    "leaking credentials belonging to several government contractors according to researchers."
)

FEEDS = {
    "https://alpha.example.com/rss": feed(
        "alpha",
        [
            ("quake", "Powerful earthquake strikes the coastal region", QUAKE),
            ("ai", "OpenAI releases a new reasoning model", AI_STORY),
        ],
    ),
    "https://beta.example.com/rss": feed(
        "beta",
        [
            ("quake", "Earthquake hits coast, rescue operation under way", QUAKE),
            ("markets", "Central bank raises interest rates again", MARKET_STORY),
        ],
    ),
    "https://gamma.example.com/rss": feed(
        "gamma",
        [
            ("quake", "Rescue teams search after coastal earthquake", QUAKE),
            ("cyber", "Ransomware group exploits VPN zero-day", CYBER_STORY),
        ],
    ),
}


@pytest.fixture
def config():
    return get_settings().model_copy(
        update={
            "ssrf_protection_enabled": False,
            "respect_robots_txt": False,
            "http_max_retries": 0,
            "default_request_delay_seconds": 0.0,
            "trend_min_articles": 1,
            "event_similarity_threshold": 0.4,
        }
    )


@pytest.fixture
async def publishers(session: AsyncSession, config) -> list[Source]:
    rows = [
        Source(
            slug=name,
            name=name.title(),
            kind=str(SourceKind.RSS),
            url=f"https://{name}.example.com/rss",
            language="en",
            country="US",
            weight=1.2,
            reliability_score=0.7,
            request_delay_seconds=0.0,
            respect_robots=False,
        )
        for name in ("alpha", "beta", "gamma")
    ]
    session.add_all(rows)
    await TopicRepository(session).seed(default_topic_seed())
    await session.commit()
    for row in rows:
        await session.refresh(row)
    return rows


@respx.mock
async def test_full_platform_flow(
    session: AsyncSession,
    client: AsyncClient,
    publishers: list[Source],
    user: User,
    auth_headers: Callable[[User], dict[str, str]],
    config,
) -> None:
    for url, body in FEEDS.items():
        respx.get(url).mock(
            return_value=httpx.Response(
                200, text=body, headers={"content-type": "application/rss+xml"}
            )
        )

    # ---------------------------------------------------------------- ingest
    pipeline = IngestionPipeline(session=session, config=config)
    await pipeline.prepare()
    results = [await pipeline.ingest_source(source) for source in publishers]
    await session.commit()

    assert all(result.success for result in results), [r.error for r in results]
    stored = sum(result.stored for result in results)
    duplicates = sum(result.duplicates for result in results)

    # Three publishers ran the same earthquake copy: two of them are duplicates.
    assert duplicates >= 2
    assert stored == 6 - duplicates

    rows = (await session.execute(select(Article))).scalars().all()
    assert len(rows) == stored
    assert all(row.content_hash for row in rows)
    assert all(row.status == "processed" for row in rows)
    assert any(row.keywords for row in rows)

    # ------------------------------------------------------------ enrichment
    by_url = {row.canonical_url: row for row in rows}
    cyber = by_url.get("https://gamma.example.com/cyber")
    assert cyber is not None
    assert cyber.category in {"cybersecurity", "technology"}
    assert cyber.sentiment_score < 0  # a breach is bad news
    assert cyber.language == "en"
    assert cyber.relevance_score > 0

    # ------------------------------------------------------------ analytics
    trends = await TrendService(session, config=config).compute(hours=24)
    await EventService(session, config=config).detect(hours=48)
    await session.commit()
    assert trends

    overview = await AnalyticsService(session, config=config).overview(use_cache=False)
    assert overview.total_articles == stored
    assert overview.active_sources == 3

    # ------------------------------------------------------------------- API
    listing = await client.get("/api/v1/articles?page_size=20")
    assert listing.status_code == 200
    assert listing.json()["meta"]["total"] == stored

    search = await client.get("/api/v1/articles/search?q=earthquake")
    assert search.json()["meta"]["total"] >= 1

    detail = await client.get(f"/api/v1/articles/{rows[0].id}")
    assert detail.status_code == 200
    assert detail.json()["topics"] is not None

    trend_response = await client.get("/api/v1/trends?limit=10")
    assert trend_response.status_code == 200
    assert trend_response.json()

    analytics = await client.get("/api/v1/analytics/overview?refresh=true")
    assert analytics.json()["total_articles"] == stored

    sources = await client.get("/api/v1/sources")
    assert sources.json()["meta"]["total"] == 3

    # -------------------------------------------------------- personalisation
    headers = auth_headers(user)
    preferences = await client.put(
        "/api/v1/users/me/preferences",
        json={"topics": ["cybersecurity"], "keywords": ["ransomware"]},
        headers=headers,
    )
    assert preferences.status_code == 200

    personal = await client.get("/api/v1/articles/feed", headers=headers)
    assert personal.status_code == 200
    assert personal.json()["items"]

    # ------------------------------------------------------------- dashboard
    dashboard = await client.get("/")
    assert dashboard.status_code == 200
    assert "Total articles" in dashboard.text
    assert "earthquake" in dashboard.text.lower()

    # ---------------------------------------------------------- idempotency
    repeat = IngestionPipeline(session=session, config=config)
    await repeat.prepare()
    second = [await repeat.ingest_source(source) for source in publishers]
    await session.commit()

    assert sum(result.stored for result in second) == 0
    total = (await session.execute(select(func.count(Article.id)))).scalar_one()
    assert total == stored


@respx.mock
async def test_platform_survives_a_failing_source(
    session: AsyncSession, publishers: list[Source], config
) -> None:
    """One dead source must not stop the others - the core resilience promise."""
    respx.get("https://alpha.example.com/rss").mock(
        return_value=httpx.Response(
            200,
            text=FEEDS["https://alpha.example.com/rss"],
            headers={"content-type": "application/rss+xml"},
        )
    )
    respx.get("https://beta.example.com/rss").mock(return_value=httpx.Response(503))
    respx.get("https://gamma.example.com/rss").mock(
        side_effect=httpx.ConnectTimeout("connection timed out")
    )

    pipeline = IngestionPipeline(session=session, config=config)
    await pipeline.prepare()
    results = [await pipeline.ingest_source(source) for source in publishers]
    await session.commit()

    by_source = {result.source: result for result in results}
    assert by_source["alpha"].success is True
    assert by_source["alpha"].stored == 2
    assert by_source["beta"].success is False
    assert by_source["gamma"].success is False

    # Failures are recorded against the sources, and the good data is stored.
    total = (await session.execute(select(func.count(Article.id)))).scalar_one()
    assert total == 2
    for name in ("beta", "gamma"):
        source = next(row for row in publishers if row.slug == name)
        await session.refresh(source)
        assert source.consecutive_failures == 1
        assert source.last_error
