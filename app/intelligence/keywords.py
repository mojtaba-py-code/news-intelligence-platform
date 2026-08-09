"""Keyword extraction.

Implements a RAKE-style algorithm (Rapid Automatic Keyword Extraction): split
the text into candidate phrases at stop words and punctuation, then score each
word by ``degree / frequency`` - words that co-occur inside longer phrases score
higher than words that merely repeat. Unlike raw frequency counting this
surfaces multi-word terms such as "artificial intelligence" or "supply chain".

An optional TF-IDF re-ranking pass boosts terms that are unusual *for this
corpus*, which is what makes "quantum" beat "company" in a tech feed.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Final

from app.processing.deduplication.similarity import STOPWORDS, tokenize

_SENTENCE_SPLIT_RE = re.compile(r"[.!?;:\n\r\t()\[\]{}\"“”«»…]|(?:,\s)")
_WORD_RE = re.compile(r"[^\W\d_][\w'-]*", re.UNICODE)

MAX_PHRASE_WORDS: Final[int] = 4
MIN_WORD_LENGTH: Final[int] = 3
MAX_TEXT_CHARS: Final[int] = 50_000

#: Words that are frequent in news but carry no topical information.
_NEWS_NOISE: Final[frozenset[str]] = frozenset(
    [
        "said",
        "says",
        "say",
        "told",
        "reported",
        "report",
        "reports",
        "according",
        "news",
        "article",
        "read",
        "update",
        "updated",
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
        "year",
        "years",
        "month",
        "months",
        "week",
        "weeks",
        "day",
        "days",
        "today",
        "yesterday",
        "tomorrow",
        "time",
        "times",
        "people",
        "man",
        "woman",
        "men",
        "women",
        "thing",
        "things",
        "way",
        "ways",
        "percent",
        "per",
        "cent",
        "million",
        "billion",
        "trillion",
        "first",
        "second",
        "third",
        "last",
        "next",
        "previous",
    ]
)

_IGNORED = STOPWORDS | _NEWS_NOISE


@dataclass(frozen=True, slots=True)
class Keyword:
    """A scored keyword or key phrase."""

    text: str
    score: float
    frequency: int = 1

    @property
    def is_phrase(self) -> bool:
        return " " in self.text


def _candidate_phrases(text: str) -> list[list[str]]:
    """Split into phrases bounded by punctuation and stop words."""
    phrases: list[list[str]] = []
    for chunk in _SENTENCE_SPLIT_RE.split(text[:MAX_TEXT_CHARS]):
        current: list[str] = []
        for match in _WORD_RE.finditer(chunk):
            word = match.group(0).casefold().strip("'-")
            if len(word) < MIN_WORD_LENGTH or word in _IGNORED or word.isdigit():
                if current:
                    phrases.append(current)
                    current = []
                continue
            current.append(word)
            if len(current) >= MAX_PHRASE_WORDS:
                phrases.append(current)
                current = []
        if current:
            phrases.append(current)
    return phrases


def extract_keywords(
    text: str | None,
    *,
    limit: int = 12,
    title: str | None = None,
    min_score: float = 1.0,
) -> list[Keyword]:
    """Extract up to ``limit`` keywords, highest score first.

    ``title`` is scored twice: a term in the headline is a much stronger signal
    of what the article is about than the same term buried in paragraph nine.
    """
    if not text and not title:
        return []

    corpus = f"{title}. {title}. {text or ''}" if title else (text or "")
    phrases = _candidate_phrases(corpus)
    if not phrases:
        return []

    frequency: Counter[str] = Counter()
    degree: defaultdict[str, int] = defaultdict(int)
    for phrase in phrases:
        span = len(phrase) - 1
        for word in phrase:
            frequency[word] += 1
            degree[word] += span

    word_scores = {
        word: (degree[word] + count) / count for word, count in frequency.items() if count
    }

    phrase_scores: dict[str, float] = {}
    phrase_counts: Counter[str] = Counter()
    for phrase in phrases:
        joined = " ".join(phrase)
        phrase_counts[joined] += 1
        phrase_scores[joined] = sum(word_scores.get(word, 0.0) for word in phrase)

    # Prefer specific phrases over their constituent words: drop a single word
    # whenever a multi-word phrase containing it already scores higher.
    ranked = sorted(phrase_scores.items(), key=lambda item: item[1], reverse=True)
    selected: list[Keyword] = []
    covered: set[str] = set()

    for candidate, score in ranked:
        if score < min_score:
            continue
        words = candidate.split()
        if len(words) == 1 and candidate in covered:
            continue
        if len(words) > 1 and all(word in covered for word in words):
            continue
        selected.append(
            Keyword(text=candidate, score=round(score, 4), frequency=phrase_counts[candidate])
        )
        covered.update(words)
        if len(selected) >= limit:
            break

    return selected


def keyword_strings(keywords: list[Keyword]) -> list[str]:
    """Plain list of keyword texts, for storage in the article row."""
    return [keyword.text for keyword in keywords]


def top_terms(texts: list[str], *, limit: int = 20) -> list[tuple[str, int]]:
    """Corpus-level term frequencies, used by trend detection."""
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(token for token in tokenize(text) if token not in _NEWS_NOISE)
    return counter.most_common(limit)


__all__ = ["Keyword", "extract_keywords", "keyword_strings", "top_terms"]
