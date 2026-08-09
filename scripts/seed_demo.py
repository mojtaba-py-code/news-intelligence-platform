"""Seed a demonstration dataset so the dashboard is not empty offline.

Runs the *real* pipeline over a canned set of articles: normalisation,
validation, deduplication, NLP enrichment and relevance scoring all execute
exactly as they would for live sources - only the network is skipped.

    python scripts/seed_demo.py [--articles 60] [--reset]
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.core.utils import utcnow  # noqa: E402
from app.database.models.source import Source, SourceKind  # noqa: E402
from app.database.repositories.source import SourceRepository  # noqa: E402
from app.database.repositories.taxonomy import TopicRepository  # noqa: E402
from app.database.session import create_all, dispose_engine, session_scope  # noqa: E402
from app.ingestion.pipeline import IngestionPipeline  # noqa: E402
from app.ingestion.sources.base import NewsSource, SourceContext  # noqa: E402
from app.intelligence.topics import default_topic_seed  # noqa: E402
from app.services.events import EventService  # noqa: E402
from app.services.trends import TrendService  # noqa: E402

PUBLISHERS = [
    ("demo-wire", "Demo Wire", 1.6),
    ("demo-tech", "Demo Tech Daily", 1.3),
    ("demo-markets", "Demo Markets", 1.2),
    ("demo-security", "Demo Security Report", 1.4),
]

STORIES: list[tuple[str, str, str]] = [
    (
        "ai",
        "Research lab unveils a language model with stronger reasoning",
        "A research laboratory published a language model that improves markedly on "
        "reasoning benchmarks. The team said the gains come from a new training method "
        "rather than from additional parameters. Independent researchers welcomed the "
        "results but cautioned that they must be reproduced before the claims are accepted.",
    ),
    (
        "ai",
        "Chipmaker announces hardware built for model training",
        "A semiconductor company announced an accelerator designed specifically for "
        "training large machine learning models. Analysts expect strong demand from cloud "
        "providers, though supply constraints may delay shipments into the next quarter.",
    ),
    (
        "cybersecurity",
        "Ransomware group exploits a zero-day in a VPN appliance",
        "Attackers exploited a previously unknown vulnerability in a widely deployed VPN "
        "appliance, stealing credentials from several organisations. The vendor released an "
        "emergency patch and urged customers to rotate every credential immediately.",
    ),
    (
        "cybersecurity",
        "Investigators trace a phishing campaign against government accounts",
        "Security investigators linked a phishing campaign to a group targeting government "
        "email accounts. The attackers used convincing login pages and stolen session "
        "cookies to bypass multi-factor authentication on several occasions.",
    ),
    (
        "finance",
        "Central bank raises interest rates to counter inflation",
        "The central bank raised interest rates by half a percentage point, citing "
        "persistent inflation in the services sector. Markets had expected the decision, "
        "and officials signalled that a further increase remains possible this year.",
    ),
    (
        "finance",
        "Markets rally after better than expected earnings",
        "Equity markets rallied following a run of stronger than expected corporate "
        "earnings. Investors pointed to resilient consumer demand, while analysts warned "
        "that margins may compress if input costs continue rising.",
    ),
    (
        "world",
        "Rescue operation continues after a powerful earthquake",
        "Rescue teams continued searching collapsed buildings after a powerful earthquake "
        "struck the coastal region. Aid agencies warned that thousands remain without "
        "shelter, and neighbouring countries have sent emergency supplies.",
    ),
    (
        "energy",
        "Grid operator reports record renewable generation",
        "The national grid operator reported a record share of electricity generated from "
        "renewable sources last month. Officials credited new wind capacity and improved "
        "battery storage, while warning that transmission remains a bottleneck.",
    ),
    (
        "health",
        "Clinical trial reports encouraging results for a new treatment",
        "Researchers reported encouraging results from a clinical trial of a new treatment. "
        "The study was small, and the authors stressed that a larger trial is required "
        "before regulators could consider approval.",
    ),
    (
        "science",
        "Telescope observations refine estimates of galaxy formation",
        "Astronomers published observations that refine current estimates of how early "
        "galaxies formed. The data came from a space telescope and challenges parts of the "
        "prevailing model, though the team described the findings as preliminary.",
    ),
]


class DemoSource(NewsSource):
    """A connector that yields canned items instead of fetching."""

    kind = SourceKind.CUSTOM

    def __init__(self, context: SourceContext, items: list[dict[str, Any]]) -> None:
        super().__init__(context, config=get_settings())
        self._items = items

    async def _collect(self) -> list[dict[str, Any]]:
        return self._items


#: Publisher-specific framing. Each outlet covers the same events in its own
#: words, which is what real syndication looks like: similar enough to cluster
#: into one event, different enough not to be dropped as a duplicate.
ANGLES: dict[str, tuple[str, str]] = {
    "demo-wire": (
        "Reporting from the newsroom, our correspondents confirmed the following account.",
        "Officials declined to comment further when contacted late on Thursday evening.",
    ),
    "demo-tech": (
        "Engineers briefed on the work described the technical detail behind the change.",
        "Practitioners said tooling and documentation will decide how quickly this spreads.",
    ),
    "demo-markets": (
        "Investors reacted quickly, and trading desks reported unusually heavy volume.",
        "Portfolio managers said positioning would depend on guidance issued next quarter.",
    ),
    "demo-security": (
        "Defenders were advised to review logs and rotate any credential that may be exposed.",
        "Incident responders published indicators of compromise for detection engineering teams.",
    ),
}


def build_items(publisher: str, count: int, rng: random.Random) -> list[dict[str, Any]]:
    """Generate items, with several publishers covering the same events."""
    opening, closing = ANGLES.get(publisher, ("", ""))
    items: list[dict[str, Any]] = []

    # Each publisher covers a *slice* of the story set with a small overlap.
    # If they all covered everything, deduplication would (correctly) discard
    # almost the whole dataset and the demo would look broken.
    position = [slug for slug, _, _ in PUBLISHERS].index(publisher)
    beat = [(position * 3 + step) % len(STORIES) for step in range(4)]

    for index in range(count):
        story_index = beat[index % len(beat)]
        category, headline, body = STORIES[story_index]
        variant = index // len(beat)
        suffix = "" if variant == 0 else f" Follow-up report {variant}."
        content = " ".join(part for part in (opening, body, closing, suffix) if part)
        items.append(
            {
                "title": f"{headline}{suffix}",
                "url": f"https://{publisher}.example.com/{category}/{story_index}-{variant}",
                "description": body[:180],
                "content": content,
                "author": rng.choice(["Jane Doe", "John Roe", "Staff Reporter"]),
                "published_at": (
                    utcnow() - timedelta(hours=rng.uniform(0, 47), minutes=rng.uniform(0, 59))
                ).isoformat(),
                "language": "en",
                "category": category,
            }
        )
    return items


async def seed(article_count: int, reset: bool) -> None:
    configure_logging(get_settings())
    await create_all()

    rng = random.Random(20260101)  # noqa: S311 - reproducible demo data, not crypto

    async with session_scope() as session:
        await TopicRepository(session).seed(default_topic_seed())
        sources = SourceRepository(session)
        for slug, name, weight in PUBLISHERS:
            await sources.upsert_definition(
                {
                    "slug": slug,
                    "name": name,
                    "kind": str(SourceKind.CUSTOM),
                    "url": f"https://{slug}.example.com/feed",
                    "language": "en",
                    "country": "US",
                    "weight": weight,
                    "enabled": True,
                }
            )

    per_publisher = max(1, article_count // len(PUBLISHERS))
    stored_total = 0

    async with session_scope() as session:
        pipeline = IngestionPipeline(session=session, config=get_settings())
        await pipeline.prepare()
        repository = SourceRepository(session)

        for slug, _, _ in PUBLISHERS:
            source: Source | None = await repository.get_by_slug(slug)
            if source is None:
                continue
            connector = DemoSource(
                SourceContext.from_model(source), build_items(slug, per_publisher, rng)
            )
            result = await pipeline.ingest_source(source, connector=connector)
            stored_total += result.stored
            print(
                f"  {slug:<16} stored={result.stored:<4} "
                f"duplicates={result.duplicates:<4} rejected={result.rejected}"
            )

    async with session_scope() as session:
        trends = await TrendService(session).compute(hours=24)
        clusters = await EventService(session).detect(hours=48)

    print(f"\nStored {stored_total} articles, {len(trends)} trends, {len(clusters)} events.")
    print("Start the dashboard with:  news-platform serve")
    _ = reset


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed demonstration data.")
    parser.add_argument("--articles", type=int, default=60, help="approximate article count")
    parser.add_argument("--reset", action="store_true", help="reserved for future use")
    args = parser.parse_args()

    try:
        asyncio.run(_run(args.articles, args.reset))
    except KeyboardInterrupt:  # pragma: no cover
        print("Interrupted.")


async def _run(article_count: int, reset: bool) -> None:
    try:
        await seed(article_count, reset)
    finally:
        await dispose_engine()


if __name__ == "__main__":
    main()
