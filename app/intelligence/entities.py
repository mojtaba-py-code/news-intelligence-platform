"""Named entity recognition without a statistical model.

Approach: a **gazetteer + orthographic rules** recogniser.

* Curated gazetteers settle the entities that matter most in news (countries,
  major organisations, well-known products) with high precision.
* Capitalisation patterns find the rest: sequences of capitalised tokens that
  are not sentence-initial artefacts, joined across ``of``/``de``/``&``.
* Type inference uses suffix and context cues (``Inc``, ``Ltd``, ``University``,
  ``said``, ``CEO of``, ``in``/``at`` for locations).

This is a pragmatic trade-off: a spaCy/transformer NER would be more accurate,
but it costs hundreds of megabytes and seconds of import time. The
:class:`EntityRecognizer` interface makes the swap a one-line change.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Final, Protocol

from app.database.models.taxonomy import EntityType

MAX_TEXT_CHARS: Final[int] = 30_000
MAX_ENTITIES: Final[int] = 40

COUNTRIES: Final[frozenset[str]] = frozenset(
    [
        "Afghanistan",
        "Albania",
        "Algeria",
        "Argentina",
        "Armenia",
        "Australia",
        "Austria",
        "Azerbaijan",
        "Bahrain",
        "Bangladesh",
        "Belarus",
        "Belgium",
        "Bolivia",
        "Brazil",
        "Bulgaria",
        "Cambodia",
        "Cameroon",
        "Canada",
        "Chile",
        "China",
        "Colombia",
        "Croatia",
        "Cuba",
        "Cyprus",
        "Czechia",
        "Denmark",
        "Ecuador",
        "Egypt",
        "Estonia",
        "Ethiopia",
        "Finland",
        "France",
        "Georgia",
        "Germany",
        "Ghana",
        "Greece",
        "Hungary",
        "Iceland",
        "India",
        "Indonesia",
        "Iran",
        "Iraq",
        "Ireland",
        "Israel",
        "Italy",
        "Japan",
        "Jordan",
        "Kazakhstan",
        "Kenya",
        "Kuwait",
        "Kyrgyzstan",
        "Laos",
        "Latvia",
        "Lebanon",
        "Libya",
        "Lithuania",
        "Luxembourg",
        "Malaysia",
        "Mexico",
        "Moldova",
        "Mongolia",
        "Morocco",
        "Myanmar",
        "Nepal",
        "Netherlands",
        "Nigeria",
        "Norway",
        "Oman",
        "Pakistan",
        "Palestine",
        "Panama",
        "Peru",
        "Philippines",
        "Poland",
        "Portugal",
        "Qatar",
        "Romania",
        "Russia",
        "Rwanda",
        "Saudi",
        "Senegal",
        "Serbia",
        "Singapore",
        "Slovakia",
        "Slovenia",
        "Somalia",
        "Spain",
        "Sudan",
        "Sweden",
        "Switzerland",
        "Syria",
        "Taiwan",
        "Tajikistan",
        "Tanzania",
        "Thailand",
        "Tunisia",
        "Turkey",
        "Turkmenistan",
        "Uganda",
        "Ukraine",
        "Uruguay",
        "Uzbekistan",
        "Venezuela",
        "Vietnam",
        "Yemen",
        "Zambia",
        "Zimbabwe",
    ]
) | {
    "United States",
    "United Kingdom",
    "South Korea",
    "North Korea",
    "South Africa",
    "New Zealand",
    "Saudi Arabia",
    "United Arab Emirates",
    "Sri Lanka",
    "Costa Rica",
    "Czech Republic",
    "Hong Kong",
}

ORGANIZATIONS: Final[frozenset[str]] = frozenset(
    {
        "Apple",
        "Google",
        "Alphabet",
        "Microsoft",
        "Amazon",
        "Meta",
        "Facebook",
        "Tesla",
        "Nvidia",
        "Intel",
        "AMD",
        "IBM",
        "Oracle",
        "Samsung",
        "Sony",
        "Netflix",
        "Uber",
        "OpenAI",
        "Anthropic",
        "DeepMind",
        "Hugging Face",
        "Stability AI",
        "Mistral",
        "Twitter",
        "TikTok",
        "ByteDance",
        "Alibaba",
        "Tencent",
        "Baidu",
        "Huawei",
        "Xiaomi",
        "Qualcomm",
        "Broadcom",
        "Cisco",
        "Dell",
        "HP",
        "Salesforce",
        "Adobe",
        "SAP",
        "Siemens",
        "Boeing",
        "Airbus",
        "SpaceX",
        "Blue Origin",
        "NASA",
        "ESA",
        "Toyota",
        "Volkswagen",
        "Ford",
        "General Motors",
        "BMW",
        "Mercedes-Benz",
        "Stellantis",
        "Rivian",
        "Goldman Sachs",
        "JPMorgan",
        "Morgan Stanley",
        "BlackRock",
        "Citigroup",
        "HSBC",
        "Visa",
        "Mastercard",
        "PayPal",
        "Stripe",
        "Coinbase",
        "Binance",
        "Robinhood",
        "Pfizer",
        "Moderna",
        "AstraZeneca",
        "Johnson & Johnson",
        "Novartis",
        "Roche",
        "Reuters",
        "Associated Press",
        "Bloomberg",
        "BBC",
        "CNN",
        "Fox News",
        "Al Jazeera",
        "The New York Times",
        "The Guardian",
        "Financial Times",
        "Wall Street Journal",
        "TechCrunch",
        "The Verge",
        "Wired",
        "Ars Technica",
        "Politico",
        "Axios",
        "United Nations",
        "NATO",
        "European Union",
        "World Bank",
        "IMF",
        "WHO",
        "WTO",
        "OPEC",
        "Federal Reserve",
        "European Central Bank",
        "SEC",
        "FBI",
        "CIA",
        "NSA",
        "Pentagon",
        "White House",
        "Congress",
        "Parliament",
        "Supreme Court",
        "Interpol",
        "Europol",
    }
)

PRODUCTS: Final[frozenset[str]] = frozenset(
    {
        "iPhone",
        "iPad",
        "MacBook",
        "Android",
        "Windows",
        "Linux",
        "ChatGPT",
        "GPT-4",
        "GPT-5",
        "Claude",
        "Gemini",
        "Copilot",
        "Llama",
        "Bitcoin",
        "Ethereum",
        "Solana",
        "Dogecoin",
        "PlayStation",
        "Xbox",
        "Nintendo Switch",
        "Model 3",
        "Model Y",
        "Cybertruck",
        "Falcon 9",
        "Starship",
        "Kubernetes",
        "Docker",
        "TensorFlow",
        "PyTorch",
    }
)

CITIES: Final[frozenset[str]] = frozenset(
    {
        "London",
        "Paris",
        "Berlin",
        "Madrid",
        "Rome",
        "Moscow",
        "Kyiv",
        "Warsaw",
        "Vienna",
        "Amsterdam",
        "Brussels",
        "Stockholm",
        "Oslo",
        "Copenhagen",
        "Helsinki",
        "Dublin",
        "Lisbon",
        "Athens",
        "Istanbul",
        "Ankara",
        "Tehran",
        "Baghdad",
        "Riyadh",
        "Dubai",
        "Doha",
        "Cairo",
        "Jerusalem",
        "Tel Aviv",
        "Beirut",
        "Damascus",
        "Kabul",
        "Islamabad",
        "New Delhi",
        "Mumbai",
        "Bangalore",
        "Beijing",
        "Shanghai",
        "Shenzhen",
        "Tokyo",
        "Osaka",
        "Seoul",
        "Taipei",
        "Singapore",
        "Jakarta",
        "Bangkok",
        "Hanoi",
        "Manila",
        "Sydney",
        "Melbourne",
        "Auckland",
        "Toronto",
        "Vancouver",
        "Montreal",
        "New York",
        "Washington",
        "Los Angeles",
        "San Francisco",
        "Silicon Valley",
        "Chicago",
        "Boston",
        "Seattle",
        "Austin",
        "Miami",
        "Houston",
        "Mexico City",
        "Bogota",
        "Lima",
        "Santiago",
        "Buenos Aires",
        "Sao Paulo",
        "Rio de Janeiro",
        "Lagos",
        "Nairobi",
        "Johannesburg",
        "Cape Town",
        "Casablanca",
        "Geneva",
        "Zurich",
        "Munich",
        "Frankfurt",
    }
)

_ORG_SUFFIXES: Final[tuple[str, ...]] = (
    "Inc",
    "Inc.",
    "Corp",
    "Corp.",
    "Ltd",
    "Ltd.",
    "LLC",
    "PLC",
    "plc",
    "GmbH",
    "AG",
    "SA",
    "NV",
    "Group",
    "Holdings",
    "Company",
    "Co",
    "Co.",
    "Bank",
    "University",
    "Institute",
    "Foundation",
    "Association",
    "Agency",
    "Ministry",
    "Department",
    "Commission",
    "Council",
    "Committee",
    "Party",
    "Union",
    "Federation",
    "Labs",
    "Laboratories",
    "Technologies",
    "Systems",
    "Solutions",
    "Partners",
    "Capital",
    "Ventures",
    "Media",
    "Network",
    "News",
)

_PERSON_TITLES: Final[frozenset[str]] = frozenset(
    {
        "Mr",
        "Mrs",
        "Ms",
        "Dr",
        "Prof",
        "President",
        "Vice",
        "Senator",
        "Governor",
        "Mayor",
        "Minister",
        "Chancellor",
        "Prime",
        "King",
        "Queen",
        "Prince",
        "Princess",
        "Pope",
        "General",
        "Colonel",
        "Captain",
        "Judge",
        "Justice",
        "Sir",
        "Lady",
        "CEO",
        "CFO",
        "CTO",
        "Chairman",
        "Chairwoman",
        "Director",
        "Founder",
        "Secretary",
        "Ambassador",
    }
)

#: Verbs/phrases that almost always follow a person's name in reportage.
_PERSON_CONTEXT: Final[frozenset[str]] = frozenset(
    {"said", "says", "told", "added", "wrote", "announced", "argued", "warned", "denied", "noted"}
)

_LOCATION_PREPOSITIONS: Final[frozenset[str]] = frozenset({"in", "at", "from", "near", "across"})

#: Sentence-initial words that produce false positives when capitalised.
_SENTENCE_STARTERS: Final[frozenset[str]] = frozenset(
    [
        "The",
        "A",
        "An",
        "This",
        "That",
        "These",
        "Those",
        "It",
        "He",
        "She",
        "They",
        "We",
        "You",
        "I",
        "But",
        "And",
        "Or",
        "So",
        "If",
        "When",
        "While",
        "After",
        "Before",
        "Since",
        "Because",
        "However",
        "Although",
        "Meanwhile",
        "Meanwhile",
        "Now",
        "Then",
        "There",
        "Here",
        "What",
        "Who",
        "How",
        "New",
        "More",
        "Most",
        "Some",
        "Many",
        "Several",
        "One",
        "Two",
        "Three",
        "Its",
        "His",
        "Her",
        "Their",
        "Our",
        "My",
        "Your",
        "Also",
        "On",
        "In",
        "At",
        "For",
        "To",
        "From",
        "By",
        "With",
        "Of",
        "As",
        "Over",
        "Under",
        "About",
        "According",
        "Following",
        "During",
        "Despite",
        "Under",
    ]
)

# Capitalised runs, optionally joined by a lowercase connector ("Bank of America",
# "Johnson & Johnson"). ``and``/``the`` are deliberately excluded as connectors:
# they merge unrelated neighbours ("London and Iran") far more often than they help.
_CAPITALISED_RE = re.compile(
    r"\b([A-Z][\w&.'-]*(?:\s+(?:of|de|del|van|der|&)\s+[A-Z][\w&.'-]*"
    r"|\s+[A-Z][\w&.'-]*){0,4})\b"
)
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,6})\b")


@dataclass(frozen=True, slots=True)
class ExtractedEntity:
    """One recognised entity with its mention count and salience."""

    name: str
    entity_type: EntityType
    mentions: int = 1
    salience: float = 0.0

    @property
    def key(self) -> str:
        return f"{normalize_entity_name(self.name)}|{self.entity_type}"


#: Abbreviations whose trailing period does not end a sentence.
_ABBREVIATIONS: Final[frozenset[str]] = frozenset(
    {
        "Inc",
        "Ltd",
        "Corp",
        "Co",
        "Bros",
        "Mt",
        "St",
        "Dr",
        "Mr",
        "Mrs",
        "Ms",
        "Prof",
        "Jr",
        "Sr",
        "Gov",
        "Sen",
        "Rep",
        "Gen",
        "Col",
        "Capt",
        "Lt",
        "Sgt",
        "Ave",
        "Blvd",
        "U.S",
        "U.K",
        "U.N",
        "D.C",
    }
)


def _split_at_sentence_break(candidate: str) -> list[str]:
    """Split ``"Monday. Microsoft"`` into two candidates, keeping ``"Inc. Group"`` whole."""
    words = candidate.split()
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        if word.endswith(".") and word.rstrip(".") not in _ABBREVIATIONS:
            pieces.append(" ".join(current))
            current = []
    if current:
        pieces.append(" ".join(current))
    return [piece for piece in pieces if piece]


#: Determiners that a sentence start glues onto an entity ("The United States").
_LEADING_ARTICLES: Final[frozenset[str]] = frozenset({"The", "A", "An"})


def _strip_leading_article(name: str) -> str:
    """Drop a leading determiner when the remainder is a known entity.

    ``"The United States"`` must resolve to the country, while
    ``"The New York Times"`` must stay intact - hence the gazetteer check.
    """
    words = name.split()
    if len(words) < 2 or words[0] not in _LEADING_ARTICLES:
        return name
    if name in ORGANIZATIONS or name in PRODUCTS:
        return name
    remainder = " ".join(words[1:])
    if remainder in COUNTRIES or remainder in CITIES or remainder in ORGANIZATIONS:
        return remainder
    return name


def normalize_entity_name(name: str) -> str:
    """Case/punctuation-insensitive key for entity deduplication."""
    cleaned = re.sub(r"[^\w\s&-]", " ", name.casefold())
    cleaned = re.sub(r"\b(inc|corp|ltd|llc|plc|gmbh|co)\b", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


class EntityRecognizer(Protocol):
    """Swap-in point for a statistical NER model."""

    name: str

    def extract(self, text: str, *, title: str | None = None) -> list[ExtractedEntity]: ...


class RuleBasedEntityRecognizer:
    """Gazetteer + orthographic-rule recogniser (the default implementation)."""

    name = "rules-v1"

    def __init__(self, *, max_entities: int = MAX_ENTITIES) -> None:
        self._max_entities = max_entities

    def extract(self, text: str, *, title: str | None = None) -> list[ExtractedEntity]:
        """Find entities in ``text``; ``title`` mentions count double."""
        if not text and not title:
            return []

        body = (text or "")[:MAX_TEXT_CHARS]
        corpus = f"{title}. {body}" if title else body
        title_text = title or ""

        counts: Counter[str] = Counter()
        types: dict[str, EntityType] = {}
        surface: dict[str, str] = {}

        for candidate, context_before, context_after, sentence_initial in self._candidates(corpus):
            name = _strip_leading_article(candidate.strip(" .,'-&"))
            if not self._is_plausible(name, sentence_initial=sentence_initial):
                continue
            entity_type = self._classify(name, context_before, context_after)
            key = f"{normalize_entity_name(name)}|{entity_type}"
            if not key.split("|")[0]:
                continue
            weight = 2 if name in title_text else 1
            counts[key] += weight
            types.setdefault(key, entity_type)
            # Keep the longest surface form seen ("Tim Cook" over "Cook").
            if key not in surface or len(name) > len(surface[key]):
                surface[key] = name

        if not counts:
            return []

        total = sum(counts.values())
        entities = [
            ExtractedEntity(
                name=surface[key],
                entity_type=types[key],
                mentions=count,
                salience=round(count / total, 4),
            )
            for key, count in counts.most_common(self._max_entities)
        ]
        return entities

    # ------------------------------------------------------------------ inner
    def _candidates(self, text: str) -> list[tuple[str, str, str, bool]]:
        """Yield ``(candidate, preceding_word, following_word, sentence_initial)``."""
        found: list[tuple[str, str, str, bool]] = []
        for match in _CAPITALISED_RE.finditer(text):
            prefix = text[max(0, match.start() - 40) : match.start()]
            before = prefix.split()
            after = text[match.end() : match.end() + 40].split()
            initial = not prefix.strip() or prefix.rstrip()[-1:] in {".", "!", "?", "\n", '"'}
            # A capitalised run can straddle a sentence boundary
            # ("…on Monday. Microsoft said…"); split it back apart.
            pieces = _split_at_sentence_break(match.group(1))
            for index, piece in enumerate(pieces):
                found.append(
                    (
                        piece,
                        before[-1] if (index == 0 and before) else "",
                        after[0] if (index == len(pieces) - 1 and after) else "",
                        initial if index == 0 else True,
                    )
                )
        for match in _ACRONYM_RE.finditer(text):
            acronym = match.group(1)
            if acronym in ORGANIZATIONS or len(acronym) >= 3:
                found.append((acronym, "", "", False))
        return found

    @staticmethod
    def _is_plausible(name: str, *, sentence_initial: bool = False) -> bool:
        if len(name) < 2 or len(name) > 80:
            return False
        if name.isdigit():
            return False
        words = name.split()
        if len(words) == 1:
            single = words[0]
            if single in _SENTENCE_STARTERS or single in _PERSON_TITLES:
                return False
            if single.isupper() and len(single) <= 2:
                return False
            # A lone capitalised word that merely opens a sentence ("Shares",
            # "Researchers") is ordinary prose unless a gazetteer knows it.
            if sentence_initial and not (
                single in COUNTRIES
                or single in CITIES
                or single in ORGANIZATIONS
                or single in PRODUCTS
                or single.isupper()
            ):
                return False
            if single.lower() in {
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
                "january",
                "february",
                "march",
                "april",
                "may",
                "june",
                "july",
                "august",
                "september",
                "october",
                "november",
                "december",
            }:
                return False
        elif words[0] in _SENTENCE_STARTERS and len(words) == 2:
            # "The Guardian" is real, "The company" is not - require the second
            # word to look like a proper noun too.
            return words[1][:1].isupper() and words[1] not in _SENTENCE_STARTERS
        return any(char.isalpha() for char in name)

    @staticmethod
    def _classify(name: str, before: str, after: str) -> EntityType:
        if name in COUNTRIES:
            return EntityType.COUNTRY
        if name in CITIES:
            return EntityType.LOCATION
        if name in ORGANIZATIONS:
            return EntityType.ORGANIZATION
        if name in PRODUCTS:
            return EntityType.PRODUCT

        words = name.split()
        if any(word.rstrip(".") in _ORG_SUFFIXES for word in words):
            return EntityType.ORGANIZATION
        if before.rstrip(".,") in _PERSON_TITLES:
            return EntityType.PERSON
        if after.rstrip(".,") in _PERSON_CONTEXT:
            return EntityType.PERSON
        if before.lower().rstrip(".,") in _LOCATION_PREPOSITIONS and len(words) <= 2:
            return EntityType.LOCATION
        if len(words) == 2 and all(word[:1].isupper() and word.isalpha() for word in words):
            # Two capitalised plain words with no other signal: most often a
            # person's first + last name in news copy.
            return EntityType.PERSON
        if name.isupper():
            return EntityType.ORGANIZATION
        return EntityType.OTHER


_default_recognizer = RuleBasedEntityRecognizer()


def extract_entities(
    text: str | None,
    *,
    title: str | None = None,
    recognizer: EntityRecognizer | None = None,
    limit: int = 25,
) -> list[ExtractedEntity]:
    """Extract entities using the configured recogniser."""
    engine = recognizer or _default_recognizer
    return engine.extract(text or "", title=title)[:limit]


def set_default_recognizer(recognizer: EntityRecognizer) -> None:
    """Replace the process-wide recogniser."""
    global _default_recognizer
    _default_recognizer = recognizer  # type: ignore[assignment]


__all__ = [
    "CITIES",
    "COUNTRIES",
    "ORGANIZATIONS",
    "PRODUCTS",
    "EntityRecognizer",
    "ExtractedEntity",
    "RuleBasedEntityRecognizer",
    "extract_entities",
    "normalize_entity_name",
    "set_default_recognizer",
]
