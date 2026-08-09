"""Configurable topic classification.

A weighted-lexicon classifier: each topic owns a set of seed terms, a match in
the title counts more than a match in the body, and scores are normalised so a
long article does not automatically outrank a short one. Categories are
**data**, not code - they ship as defaults here, live in the ``topics`` table at
runtime, and can be edited through the API without a deployment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final

from app.core.utils import clamp

MAX_TEXT_CHARS: Final[int] = 30_000
#: Below this score a topic assignment is noise rather than a classification.
MIN_TOPIC_SCORE: Final[float] = 0.12
TITLE_WEIGHT: Final[float] = 3.0


@dataclass(frozen=True, slots=True)
class TopicDefinition:
    """A classification category and the terms that identify it."""

    slug: str
    name: str
    keywords: frozenset[str]
    #: Terms that strongly indicate the topic on their own.
    strong_keywords: frozenset[str] = frozenset()
    parent: str | None = None


@dataclass(frozen=True, slots=True)
class TopicMatch:
    """A topic assignment with its confidence."""

    slug: str
    name: str
    score: float
    matched_terms: tuple[str, ...] = ()


def _terms(*words: str) -> frozenset[str]:
    return frozenset(word.casefold() for word in words)


DEFAULT_TOPICS: Final[tuple[TopicDefinition, ...]] = (
    TopicDefinition(
        slug="technology",
        name="Technology",
        keywords=_terms(
            "technology",
            "tech",
            "software",
            "hardware",
            "computer",
            "computing",
            "device",
            "gadget",
            "smartphone",
            "app",
            "platform",
            "startup",
            "silicon",
            "semiconductor",
            "chip",
            "processor",
            "cloud",
            "saas",
            "developer",
            "engineering",
            "open source",
            "algorithm",
            "digital",
            "internet",
            "network",
            "5g",
            "quantum",
            "robotics",
        ),
        strong_keywords=_terms(
            "semiconductor", "silicon valley", "open source", "quantum computing"
        ),
    ),
    TopicDefinition(
        slug="ai",
        name="Artificial Intelligence",
        keywords=_terms(
            "ai",
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "neural",
            "llm",
            "large language model",
            "chatbot",
            "generative",
            "openai",
            "anthropic",
            "chatgpt",
            "claude",
            "gemini",
            "copilot",
            "transformer",
            "training data",
            "inference",
            "model",
            "agi",
            "computer vision",
            "nlp",
            "diffusion",
        ),
        strong_keywords=_terms(
            "artificial intelligence", "machine learning", "large language model", "generative ai"
        ),
        parent="technology",
    ),
    TopicDefinition(
        slug="cybersecurity",
        name="Cybersecurity",
        keywords=_terms(
            "cybersecurity",
            "security",
            "hacker",
            "hacking",
            "breach",
            "ransomware",
            "malware",
            "phishing",
            "vulnerability",
            "exploit",
            "zero-day",
            "ddos",
            "encryption",
            "firewall",
            "cyberattack",
            "data leak",
            "credential",
            "botnet",
            "spyware",
            "threat actor",
            "patch",
            "cve",
            "infosec",
            "penetration testing",
        ),
        strong_keywords=_terms("ransomware", "zero-day", "data breach", "cyberattack"),
        parent="technology",
    ),
    TopicDefinition(
        slug="business",
        name="Business",
        keywords=_terms(
            "business",
            "company",
            "corporate",
            "ceo",
            "merger",
            "acquisition",
            "revenue",
            "earnings",
            "profit",
            "quarterly",
            "layoffs",
            "hiring",
            "ipo",
            "shareholder",
            "enterprise",
            "industry",
            "supply chain",
            "manufacturing",
            "retail",
            "logistics",
        ),
        strong_keywords=_terms("merger", "acquisition", "ipo", "quarterly earnings"),
    ),
    TopicDefinition(
        slug="finance",
        name="Finance",
        keywords=_terms(
            "finance",
            "financial",
            "market",
            "stock",
            "shares",
            "investor",
            "investment",
            "trading",
            "bond",
            "yield",
            "interest rate",
            "inflation",
            "recession",
            "gdp",
            "central bank",
            "federal reserve",
            "economy",
            "economic",
            "fund",
            "hedge fund",
            "portfolio",
            "dividend",
            "nasdaq",
            "dow jones",
            "s&p",
            "currency",
            "forex",
        ),
        strong_keywords=_terms("federal reserve", "interest rate", "stock market", "inflation"),
    ),
    TopicDefinition(
        slug="cryptocurrency",
        name="Cryptocurrency",
        keywords=_terms(
            "crypto",
            "cryptocurrency",
            "bitcoin",
            "ethereum",
            "blockchain",
            "token",
            "defi",
            "nft",
            "stablecoin",
            "mining",
            "wallet",
            "exchange",
            "binance",
            "coinbase",
            "altcoin",
            "solana",
            "web3",
            "smart contract",
            "halving",
        ),
        strong_keywords=_terms("bitcoin", "cryptocurrency", "blockchain", "stablecoin"),
        parent="finance",
    ),
    TopicDefinition(
        slug="politics",
        name="Politics",
        keywords=_terms(
            "politics",
            "political",
            "election",
            "vote",
            "voter",
            "campaign",
            "candidate",
            "parliament",
            "congress",
            "senate",
            "president",
            "prime minister",
            "government",
            "policy",
            "legislation",
            "bill",
            "law",
            "regulation",
            "party",
            "democrat",
            "republican",
            "referendum",
            "coalition",
            "minister",
            "diplomacy",
            "sanctions",
        ),
        strong_keywords=_terms("election", "parliament", "prime minister", "legislation"),
    ),
    TopicDefinition(
        slug="world",
        name="World",
        keywords=_terms(
            "world",
            "international",
            "global",
            "foreign",
            "war",
            "conflict",
            "military",
            "troops",
            "ceasefire",
            "refugee",
            "united nations",
            "nato",
            "treaty",
            "border",
            "humanitarian",
            "peace talks",
            "embassy",
            "geopolitical",
        ),
        strong_keywords=_terms("united nations", "ceasefire", "humanitarian", "geopolitical"),
    ),
    TopicDefinition(
        slug="science",
        name="Science",
        keywords=_terms(
            "science",
            "scientific",
            "research",
            "researcher",
            "study",
            "experiment",
            "physics",
            "chemistry",
            "biology",
            "genome",
            "space",
            "nasa",
            "telescope",
            "satellite",
            "mars",
            "astronomy",
            "particle",
            "discovery",
            "peer-reviewed",
            "laboratory",
            "hypothesis",
        ),
        strong_keywords=_terms("peer-reviewed", "astronomy", "genome"),
    ),
    TopicDefinition(
        slug="health",
        name="Health",
        keywords=_terms(
            "health",
            "healthcare",
            "medical",
            "medicine",
            "hospital",
            "patient",
            "doctor",
            "disease",
            "virus",
            "vaccine",
            "pandemic",
            "outbreak",
            "treatment",
            "therapy",
            "clinical trial",
            "drug",
            "fda",
            "who",
            "mental health",
            "cancer",
            "diabetes",
            "surgery",
            "diagnosis",
        ),
        strong_keywords=_terms("clinical trial", "vaccine", "outbreak", "mental health"),
    ),
    TopicDefinition(
        slug="sports",
        name="Sports",
        keywords=_terms(
            "sports",
            "football",
            "soccer",
            "basketball",
            "baseball",
            "tennis",
            "cricket",
            "olympics",
            "championship",
            "tournament",
            "league",
            "match",
            "player",
            "coach",
            "team",
            "goal",
            "score",
            "season",
            "playoff",
            "fifa",
            "nba",
            "nfl",
            "formula 1",
        ),
        strong_keywords=_terms("championship", "olympics", "playoff", "world cup"),
    ),
    TopicDefinition(
        slug="energy",
        name="Energy & Climate",
        keywords=_terms(
            "energy",
            "oil",
            "gas",
            "petroleum",
            "opec",
            "renewable",
            "solar",
            "wind power",
            "nuclear",
            "electricity",
            "grid",
            "battery",
            "climate",
            "carbon",
            "emissions",
            "sustainability",
            "fossil fuel",
            "hydrogen",
            "ev",
            "electric vehicle",
        ),
        strong_keywords=_terms(
            "renewable energy", "carbon emissions", "fossil fuel", "climate change"
        ),
    ),
    TopicDefinition(
        slug="entertainment",
        name="Entertainment",
        keywords=_terms(
            "entertainment",
            "movie",
            "film",
            "cinema",
            "streaming",
            "netflix",
            "series",
            "album",
            "music",
            "artist",
            "concert",
            "celebrity",
            "actor",
            "actress",
            "box office",
            "award",
            "oscar",
            "grammy",
            "festival",
            "game",
            "gaming",
            "console",
        ),
        strong_keywords=_terms("box office", "streaming service", "grammy", "oscar"),
    ),
)


@dataclass
class TopicClassifier:
    """Scores text against a set of :class:`TopicDefinition` objects."""

    topics: tuple[TopicDefinition, ...] = DEFAULT_TOPICS
    min_score: float = MIN_TOPIC_SCORE
    _patterns: dict[str, list[tuple[str, re.Pattern[str], float]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._compile()

    def _compile(self) -> None:
        """Pre-compile word-boundary patterns once; classification is hot."""
        self._patterns = {}
        for topic in self.topics:
            entries: list[tuple[str, re.Pattern[str], float]] = []
            for term in topic.keywords:
                weight = 2.0 if term in topic.strong_keywords else 1.0
                entries.append((term, re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE), weight))
            self._patterns[topic.slug] = entries

    @classmethod
    def from_definitions(cls, definitions: list[TopicDefinition]) -> TopicClassifier:
        return cls(topics=tuple(definitions))

    def classify(
        self, text: str | None, *, title: str | None = None, limit: int = 3
    ) -> list[TopicMatch]:
        """Return the best-matching topics, most confident first."""
        body = (text or "")[:MAX_TEXT_CHARS]
        headline = title or ""
        if not body and not headline:
            return []

        raw_scores: dict[str, tuple[float, list[str]]] = {}
        for topic in self.topics:
            score = 0.0
            matched: list[str] = []
            for term, pattern, weight in self._patterns[topic.slug]:
                title_hits = len(pattern.findall(headline)) if headline else 0
                body_hits = len(pattern.findall(body)) if body else 0
                if not title_hits and not body_hits:
                    continue
                matched.append(term)
                # Diminishing returns: the tenth mention adds little evidence.
                score += weight * (TITLE_WEIGHT * min(title_hits, 2) + min(body_hits, 5) ** 0.7)
            if score > 0:
                raw_scores[topic.slug] = (score, matched)

        if not raw_scores:
            return []

        # Normalise against the strongest signal so scores stay comparable
        # across documents of very different lengths.
        best = max(score for score, _ in raw_scores.values())
        names = {topic.slug: topic.name for topic in self.topics}

        matches = [
            TopicMatch(
                slug=slug,
                name=names[slug],
                score=round(clamp(score / best), 4),
                matched_terms=tuple(sorted(matched)[:8]),
            )
            for slug, (score, matched) in raw_scores.items()
        ]
        matches.sort(key=lambda match: match.score, reverse=True)
        return [match for match in matches if match.score >= self.min_score][:limit]

    def primary_category(self, text: str | None, *, title: str | None = None) -> str | None:
        """Single best category slug, or ``None`` when nothing matches."""
        matches = self.classify(text, title=title, limit=1)
        return matches[0].slug if matches else None


_default_classifier = TopicClassifier()


def classify_topics(
    text: str | None,
    *,
    title: str | None = None,
    limit: int = 3,
    classifier: TopicClassifier | None = None,
) -> list[TopicMatch]:
    """Classify with the process-wide classifier unless one is supplied."""
    return (classifier or _default_classifier).classify(text, title=title, limit=limit)


def set_default_classifier(classifier: TopicClassifier) -> None:
    """Install a classifier built from the database's topic definitions."""
    global _default_classifier
    _default_classifier = classifier


def default_topic_seed() -> list[dict[str, object]]:
    """Rows used to seed the ``topics`` table on first run."""
    return [
        {
            "slug": topic.slug,
            "name": topic.name,
            "keywords": sorted(topic.keywords),
            "parent_slug": topic.parent,
        }
        for topic in DEFAULT_TOPICS
    ]


__all__ = [
    "DEFAULT_TOPICS",
    "MIN_TOPIC_SCORE",
    "TopicClassifier",
    "TopicDefinition",
    "TopicMatch",
    "classify_topics",
    "default_topic_seed",
    "set_default_classifier",
]
