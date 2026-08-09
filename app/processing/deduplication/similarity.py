"""Similarity measures and a small in-memory TF-IDF index.

scikit-learn would do this too, but it (plus SciPy) is ~80 MB of wheels for
functionality that is ~150 lines of NumPy here. Keeping it in-house also means
the vectoriser's vocabulary can be pinned to a candidate window, which is what
the deduplication and event-clustering stages actually need.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final

from app.core.utils import clamp

_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

#: Extremely common words carry no discriminative signal for similarity.
STOPWORDS: Final[frozenset[str]] = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "there",
        "here",
        "of",
        "in",
        "on",
        "at",
        "to",
        "from",
        "by",
        "for",
        "with",
        "without",
        "within",
        "about",
        "into",
        "over",
        "under",
        "again",
        "further",
        "once",
        "is",
        "am",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "as",
        "not",
        "no",
        "nor",
        "so",
        "too",
        "very",
        "s",
        "t",
        "don",
        "now",
        "also",
        "said",
        "says",
        "say",
        "new",
        "news",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "which",
        "who",
        "whom",
        "what",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "many",
        "just",
        "because",
        "while",
        "during",
        "before",
        "after",
        "above",
        "below",
        "up",
        "down",
        "out",
        "off",
        "between",
        "against",
        "among",
        "per",
        "via",
        "amid",
        "amp",
    ]
)

MIN_TOKEN_LENGTH: Final[int] = 2


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    """Word tokens, lower-cased, optionally stop-word filtered."""
    if not text:
        return []
    tokens = [
        token for token in _TOKEN_RE.findall(text.casefold()) if len(token) >= MIN_TOKEN_LENGTH
    ]
    if remove_stopwords:
        return [token for token in tokens if token not in STOPWORDS]
    return tokens


def jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    """Intersection over union of two token sets."""
    left_set, right_set = set(left), set(right)
    if not left_set or not right_set:
        return 0.0
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    return intersection / union if union else 0.0


def token_set_ratio(left: str, right: str) -> float:
    """Order-insensitive title similarity.

    Headlines are short and often reordered across outlets ("Apple unveils X" /
    "X unveiled by Apple"), so a set-based measure beats edit distance here.
    """
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return jaccard_similarity(left_tokens, right_tokens)


def sequence_ratio(left: str, right: str) -> float:
    """Character-level similarity via difflib, for order-sensitive comparison."""
    from difflib import SequenceMatcher

    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left.casefold(), right.casefold()).ratio()


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    """Cosine similarity of two sparse term-weight vectors."""
    if not left or not right:
        return 0.0
    # Iterate over the smaller vector: the dot product only needs shared terms.
    small, large = (left, right) if len(left) <= len(right) else (right, left)
    dot = sum(weight * large.get(term, 0.0) for term, weight in small.items())
    if dot == 0.0:
        return 0.0
    left_norm = math.sqrt(sum(weight * weight for weight in left.values()))
    right_norm = math.sqrt(sum(weight * weight for weight in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return clamp(dot / (left_norm * right_norm))


@dataclass
class TfidfIndex:
    """A tiny TF-IDF vectoriser over a fixed document set.

    Built per candidate window (e.g. "articles from the last 72 hours"), which
    keeps the vocabulary small and the IDF statistics meaningful for the
    comparison actually being made.
    """

    max_features: int = 4_000
    sublinear_tf: bool = True
    _document_frequency: Counter[str] = field(default_factory=Counter, init=False)
    _document_count: int = field(default=0, init=False)
    _vocabulary: set[str] = field(default_factory=set, init=False)

    def fit(self, documents: Sequence[str]) -> TfidfIndex:
        """Learn document frequencies from ``documents``."""
        self._document_frequency.clear()
        self._document_count = 0
        for document in documents:
            tokens = set(tokenize(document))
            if not tokens:
                continue
            self._document_count += 1
            self._document_frequency.update(tokens)

        if len(self._document_frequency) > self.max_features:
            most_common = self._document_frequency.most_common(self.max_features)
            self._vocabulary = {term for term, _ in most_common}
        else:
            self._vocabulary = set(self._document_frequency)
        return self

    def transform(self, document: str) -> dict[str, float]:
        """Vectorise ``document`` against the fitted vocabulary."""
        tokens = tokenize(document)
        if not tokens:
            return {}
        counts = Counter(
            token for token in tokens if not self._vocabulary or token in self._vocabulary
        )
        if not counts:
            return {}

        total = sum(counts.values())
        vector: dict[str, float] = {}
        for term, count in counts.items():
            tf = 1.0 + math.log(count) if self.sublinear_tf else count / total
            df = self._document_frequency.get(term, 0)
            # Smoothed IDF; +1 in the numerator keeps unseen terms informative.
            idf = math.log((1 + self._document_count) / (1 + df)) + 1.0
            vector[term] = tf * idf
        return vector

    def fit_transform(self, documents: Sequence[str]) -> list[dict[str, float]]:
        self.fit(documents)
        return [self.transform(document) for document in documents]

    def similarity(self, left: str, right: str) -> float:
        return cosine_similarity(self.transform(left), self.transform(right))

    @property
    def vocabulary_size(self) -> int:
        return len(self._vocabulary)

    @property
    def document_count(self) -> int:
        return self._document_count


def centroid(vectors: Sequence[dict[str, float]], *, max_terms: int = 200) -> dict[str, float]:
    """Mean vector of a cluster, truncated to its heaviest terms."""
    if not vectors:
        return {}
    totals: dict[str, float] = defaultdict(float)
    for vector in vectors:
        for term, weight in vector.items():
            totals[term] += weight
    count = len(vectors)
    top = sorted(totals.items(), key=lambda item: item[1], reverse=True)[:max_terms]
    return {term: round(weight / count, 6) for term, weight in top}


__all__ = [
    "STOPWORDS",
    "TfidfIndex",
    "centroid",
    "cosine_similarity",
    "jaccard_similarity",
    "sequence_ratio",
    "token_set_ratio",
    "tokenize",
]
