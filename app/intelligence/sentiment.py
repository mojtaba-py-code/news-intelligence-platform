"""Sentiment analysis with a pluggable analyser interface.

The default implementation is a **valence-shifted lexicon** model in the spirit
of VADER: it is not "keyword counting". It handles

* negation scope (``not good`` flips polarity within a three-token window),
* intensifiers and downtoners (``very``/``slightly`` scale magnitude),
* contrastive conjunctions (``but`` down-weights the earlier clause),
* punctuation and capitalisation emphasis,
* per-sentence aggregation with length normalisation.

:class:`SentimentAnalyzer` is a Protocol, so a transformer model can be dropped
in later without touching the pipeline - which is the point of the abstraction.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

from app.core.utils import clamp
from app.database.models.article import SentimentLabel

_WORD_RE = re.compile(r"[A-Za-z']+|[!?]+", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")

#: Polarity lexicon in [-1, 1]. Curated for news register rather than reviews.
LEXICON: Final[dict[str, float]] = {
    # strongly negative
    "killed": -0.85,
    "kill": -0.8,
    "death": -0.75,
    "died": -0.75,
    "dead": -0.8,
    "murder": -0.9,
    "massacre": -0.95,
    "genocide": -0.95,
    "terror": -0.85,
    "terrorist": -0.85,
    "attack": -0.7,
    "attacked": -0.7,
    "bombing": -0.85,
    "war": -0.7,
    "conflict": -0.55,
    "invasion": -0.75,
    "crisis": -0.7,
    "disaster": -0.8,
    "catastrophe": -0.9,
    "collapse": -0.75,
    "crash": -0.7,
    "plunge": -0.65,
    "plummet": -0.7,
    "slump": -0.6,
    "recession": -0.7,
    "bankruptcy": -0.75,
    "fraud": -0.8,
    "scandal": -0.7,
    "corruption": -0.75,
    "breach": -0.65,
    "hacked": -0.7,
    "ransomware": -0.75,
    "malware": -0.65,
    "vulnerability": -0.45,
    "exploit": -0.5,
    "outage": -0.55,
    "failure": -0.6,
    "lawsuit": -0.5,
    "sued": -0.5,
    "fined": -0.5,
    "penalty": -0.45,
    "layoffs": -0.7,
    "layoff": -0.7,
    "fired": -0.6,
    "resign": -0.4,
    "protest": -0.4,
    "riot": -0.7,
    "violence": -0.8,
    "victim": -0.65,
    "injured": -0.6,
    "wounded": -0.6,
    "damage": -0.55,
    "loss": -0.5,
    "losses": -0.55,
    "decline": -0.45,
    "fell": -0.4,
    "drop": -0.4,
    "warning": -0.4,
    "warned": -0.4,
    "threat": -0.6,
    "risk": -0.35,
    "concern": -0.3,
    "concerns": -0.3,
    "fear": -0.55,
    "fears": -0.55,
    "criticism": -0.45,
    "criticized": -0.5,
    "condemn": -0.6,
    "condemned": -0.6,
    "ban": -0.4,
    "banned": -0.45,
    "sanctions": -0.5,
    "delay": -0.3,
    "delayed": -0.3,
    "shortage": -0.5,
    "inflation": -0.35,
    "unemployment": -0.5,
    "corrupt": -0.75,
    "illegal": -0.6,
    "arrest": -0.5,
    "arrested": -0.5,
    "guilty": -0.6,
    "convicted": -0.6,
    "controversial": -0.35,
    "dispute": -0.4,
    "struggle": -0.4,
    "struggling": -0.45,
    "weak": -0.35,
    "weaker": -0.35,
    "bad": -0.5,
    "worse": -0.6,
    "worst": -0.75,
    "poor": -0.45,
    "negative": -0.5,
    "fail": -0.6,
    "failed": -0.65,
    "fails": -0.6,
    "failing": -0.6,
    "catastrophic": -0.9,
    "unsuccessful": -0.6,
    "lack": -0.45,
    "lacking": -0.45,
    "denied": -0.45,
    "denies": -0.45,
    "refused": -0.5,
    "halted": -0.5,
    "suspended": -0.45,
    "shutdown": -0.6,
    "recall": -0.5,
    "defect": -0.55,
    "flaw": -0.5,
    "boycott": -0.55,
    "downgrade": -0.55,
    "problem": -0.4,
    "problems": -0.4,
    "issue": -0.25,
    "difficult": -0.35,
    # strongly positive
    "breakthrough": 0.85,
    "record": 0.5,
    "surge": 0.6,
    "soar": 0.7,
    "soared": 0.7,
    "rally": 0.55,
    "gain": 0.5,
    "gains": 0.5,
    "rise": 0.4,
    "rose": 0.4,
    "growth": 0.55,
    "grew": 0.5,
    "profit": 0.6,
    "profits": 0.6,
    "revenue": 0.3,
    "success": 0.75,
    "successful": 0.7,
    "win": 0.65,
    "wins": 0.65,
    "won": 0.65,
    "victory": 0.75,
    "achievement": 0.7,
    "milestone": 0.6,
    "innovation": 0.6,
    "innovative": 0.6,
    "launch": 0.35,
    "launched": 0.35,
    "unveiled": 0.35,
    "improve": 0.55,
    "improved": 0.55,
    "improvement": 0.55,
    "boost": 0.6,
    "recovery": 0.6,
    "recovered": 0.55,
    "rebound": 0.55,
    "upgrade": 0.5,
    "approve": 0.45,
    "approved": 0.5,
    "agreement": 0.45,
    "deal": 0.35,
    "partnership": 0.45,
    "investment": 0.4,
    "funding": 0.4,
    "expansion": 0.45,
    "hiring": 0.45,
    "opportunity": 0.5,
    "promising": 0.6,
    "optimistic": 0.65,
    "confidence": 0.5,
    "strong": 0.5,
    "stronger": 0.55,
    "robust": 0.55,
    "excellent": 0.85,
    "outstanding": 0.8,
    "remarkable": 0.7,
    "impressive": 0.7,
    "good": 0.5,
    "better": 0.55,
    "best": 0.75,
    "positive": 0.6,
    "benefit": 0.5,
    "benefits": 0.5,
    "support": 0.4,
    "supported": 0.4,
    "praise": 0.65,
    "praised": 0.65,
    "celebrate": 0.7,
    "celebrated": 0.7,
    "peace": 0.75,
    "rescue": 0.6,
    "rescued": 0.6,
    "saved": 0.6,
    "cure": 0.8,
    "healthy": 0.6,
    "safe": 0.5,
    "safety": 0.4,
    "secure": 0.45,
    "resolved": 0.55,
    "solution": 0.5,
}

#: Multipliers applied to the *next* sentiment-bearing word.
INTENSIFIERS: Final[dict[str, float]] = {
    "very": 1.4,
    "extremely": 1.7,
    "highly": 1.4,
    "deeply": 1.5,
    "severely": 1.6,
    "massively": 1.6,
    "hugely": 1.5,
    "significantly": 1.4,
    "substantially": 1.4,
    "dramatically": 1.5,
    "sharply": 1.4,
    "strongly": 1.4,
    "absolutely": 1.6,
    "completely": 1.5,
    "totally": 1.5,
    "utterly": 1.6,
    "incredibly": 1.6,
    "particularly": 1.3,
    "especially": 1.3,
    "really": 1.3,
    "so": 1.2,
    "too": 1.2,
    "slightly": 0.6,
    "somewhat": 0.7,
    "marginally": 0.6,
    "barely": 0.5,
    "hardly": 0.5,
    "partially": 0.7,
    "relatively": 0.8,
    "fairly": 0.85,
    "mildly": 0.6,
    "moderately": 0.8,
}

#: Only *grammatical* negators belong here. Verbs such as "failed" or "denied"
#: are negative sentiment in their own right; treating them as negators flipped
#: the polarity of whatever followed ("failed catastrophically" scored positive).
NEGATIONS: Final[frozenset[str]] = frozenset(
    [
        "no",
        "not",
        "never",
        "none",
        "nobody",
        "nothing",
        "neither",
        "nowhere",
        "without",
        "cannot",
        "cant",
        "can't",
        "don't",
        "dont",
        "doesn't",
        "doesnt",
        "didn't",
        "didnt",
        "won't",
        "wont",
        "wouldn't",
        "shouldn't",
        "isn't",
        "isnt",
        "aren't",
        "arent",
        "wasn't",
        "weren't",
        "rarely",
        "seldom",
        "hardly",
        "barely",
    ]
)

CONTRAST_MARKERS: Final[frozenset[str]] = frozenset(
    {"but", "however", "although", "though", "yet", "nevertheless", "nonetheless", "despite"}
)

#: Window (in tokens) over which a negation flips polarity.
#: ``(suffix, stem replacement)`` pairs, longest first.
_SUFFIX_RULES: Final[tuple[tuple[str, str], ...]] = (
    ("ically", "e"),
    ("ingly", ""),
    ("ically", ""),
    ("edly", ""),
    ("ing", ""),
    ("ing", "e"),
    ("ies", "y"),
    ("ied", "y"),
    ("ed", ""),
    ("ed", "e"),
    ("es", ""),
    ("ly", ""),
    ("s", ""),
)

NEGATION_WINDOW: Final[int] = 3
NEGATION_FACTOR: Final[float] = -0.75


@dataclass(frozen=True, slots=True)
class SentimentResult:
    """Polarity in ``[-1, 1]``, a discrete label and a confidence."""

    score: float
    label: SentimentLabel
    confidence: float
    positive_hits: int = 0
    negative_hits: int = 0

    @property
    def is_neutral(self) -> bool:
        return self.label is SentimentLabel.NEUTRAL


@runtime_checkable
class SentimentAnalyzer(Protocol):
    """Swap-in point for a different sentiment model."""

    name: str

    def analyze(self, text: str) -> SentimentResult: ...


def score_to_label(score: float) -> SentimentLabel:
    """Map a continuous polarity onto the five-level scale."""
    if score <= -0.5:
        return SentimentLabel.VERY_NEGATIVE
    if score <= -0.1:
        return SentimentLabel.NEGATIVE
    if score < 0.1:
        return SentimentLabel.NEUTRAL
    if score < 0.5:
        return SentimentLabel.POSITIVE
    return SentimentLabel.VERY_POSITIVE


class LexiconSentimentAnalyzer:
    """Default analyser: valence lexicon with negation and intensity handling."""

    name = "lexicon-v1"

    def __init__(
        self,
        lexicon: dict[str, float] | None = None,
        *,
        max_chars: int = 20_000,
    ) -> None:
        self._lexicon = lexicon if lexicon is not None else LEXICON
        self._max_chars = max_chars

    def analyze(self, text: str | None) -> SentimentResult:
        """Score ``text``; empty or signal-free input yields a neutral verdict."""
        if not text or not text.strip():
            return SentimentResult(0.0, SentimentLabel.NEUTRAL, 0.0)

        sentences = [s for s in _SENTENCE_RE.split(text[: self._max_chars]) if s.strip()]
        if not sentences:
            return SentimentResult(0.0, SentimentLabel.NEUTRAL, 0.0)

        sentence_scores: list[float] = []
        positive_hits = 0
        negative_hits = 0
        total_hits = 0
        total_tokens = 0

        for sentence in sentences:
            score, pos, neg, hits, tokens = self._score_sentence(sentence)
            if tokens:
                sentence_scores.append(score)
                positive_hits += pos
                negative_hits += neg
                total_hits += hits
                total_tokens += tokens

        if not sentence_scores or not total_hits:
            return SentimentResult(0.0, SentimentLabel.NEUTRAL, 0.0)

        # Mean of sentence scores, then squashed. The tanh keeps a long article
        # full of mild signals from saturating at ±1.
        mean = sum(sentence_scores) / len(sentence_scores)
        score = clamp(math.tanh(mean * 1.6), -1.0, 1.0)

        # Confidence grows with evidence density and the magnitude of the score.
        density = min(1.0, total_hits / max(8.0, total_tokens * 0.08))
        confidence = round(clamp(0.35 * density + 0.65 * abs(score)), 4)

        return SentimentResult(
            score=round(score, 4),
            label=score_to_label(score),
            confidence=confidence,
            positive_hits=positive_hits,
            negative_hits=negative_hits,
        )

    # ------------------------------------------------------------------ inner
    def _lookup(self, token: str) -> float | None:
        """Lexicon lookup with light suffix stripping.

        News copy inflects heavily ("fail"/"failed"/"failing"); listing every
        form would triple the lexicon and still miss some, so unknown tokens
        fall back to a small set of English suffix rules.
        """
        direct = self._lexicon.get(token)
        if direct is not None:
            return direct
        for suffix, replacement in _SUFFIX_RULES:
            if len(token) > len(suffix) + 2 and token.endswith(suffix):
                stem = token[: -len(suffix)] + replacement
                value = self._lexicon.get(stem)
                if value is not None:
                    # Derived forms are slightly damped: the match is weaker
                    # evidence than an exact one.
                    return value * 0.9
        return None

    def _score_sentence(self, sentence: str) -> tuple[float, int, int, int, int]:
        raw_tokens = _WORD_RE.findall(sentence)
        if not raw_tokens:
            return 0.0, 0, 0, 0, 0

        tokens = [token.lower() for token in raw_tokens]
        emphasis = 1.0 + 0.25 * sentence.count("!") + (0.1 if "?" in sentence else 0.0)
        emphasis = min(emphasis, 1.75)

        contributions: list[float] = []
        positive_hits = 0
        negative_hits = 0
        contrast_index = next(
            (index for index, token in enumerate(tokens) if token in CONTRAST_MARKERS), None
        )

        for index, token in enumerate(tokens):
            valence = self._lookup(token)
            if valence is None:
                continue

            # Intensifier immediately before the sentiment word.
            if index > 0 and tokens[index - 1] in INTENSIFIERS:
                valence *= INTENSIFIERS[tokens[index - 1]]

            # ALL-CAPS emphasis on the original token.
            if raw_tokens[index].isupper() and len(raw_tokens[index]) > 2:
                valence *= 1.3

            # Negation anywhere in the preceding window flips and dampens.
            window = tokens[max(0, index - NEGATION_WINDOW) : index]
            if any(word in NEGATIONS for word in window):
                valence *= NEGATION_FACTOR

            # Text after "but" carries the writer's actual stance.
            if contrast_index is not None:
                valence *= 0.5 if index < contrast_index else 1.5

            valence = clamp(valence * emphasis, -2.0, 2.0)
            contributions.append(valence)
            if valence > 0:
                positive_hits += 1
            elif valence < 0:
                negative_hits += 1

        if not contributions:
            return 0.0, 0, 0, 0, len(tokens)

        # Length normalisation: divide by sqrt(n) so a long sentence with many
        # weak signals does not outweigh a short, emphatic one.
        total = sum(contributions)
        normalised = total / math.sqrt(len(contributions) + 3)
        return normalised, positive_hits, negative_hits, len(contributions), len(tokens)


_default_analyzer = LexiconSentimentAnalyzer()


def analyze_sentiment(
    text: str | None, *, analyzer: SentimentAnalyzer | None = None
) -> SentimentResult:
    """Analyse ``text`` with the configured analyser."""
    engine = analyzer or _default_analyzer
    return engine.analyze(text or "")


def set_default_analyzer(analyzer: SentimentAnalyzer) -> None:
    """Replace the process-wide analyser (used to plug in a different model)."""
    global _default_analyzer
    _default_analyzer = analyzer  # type: ignore[assignment]


__all__ = [
    "CONTRAST_MARKERS",
    "INTENSIFIERS",
    "LEXICON",
    "NEGATIONS",
    "LexiconSentimentAnalyzer",
    "SentimentAnalyzer",
    "SentimentResult",
    "analyze_sentiment",
    "score_to_label",
    "set_default_analyzer",
]
