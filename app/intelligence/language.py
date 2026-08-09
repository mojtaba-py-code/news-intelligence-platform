"""Language detection without a model download.

Two complementary signals:

1. **Script detection** - Unicode ranges settle Arabic, Persian, Cyrillic, CJK,
   Greek and Hebrew almost immediately.
2. **Function-word profiles** - for Latin-script languages, the ratio of
   language-specific stop words is a strong, cheap discriminator on news text.

A confidence value accompanies every verdict so callers can refuse to act on a
weak guess (``ArticleNormalizer`` requires >= 0.35).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final

_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

#: Minimum characters before detection is attempted at all.
MIN_TEXT_LENGTH: Final[int] = 20

#: High-frequency function words per language. Deliberately small: these are
#: the words whose *presence* is nearly deterministic for the language.
_PROFILES: Final[dict[str, frozenset[str]]] = {
    "en": frozenset(
        [
            "the",
            "of",
            "and",
            "to",
            "in",
            "that",
            "is",
            "was",
            "for",
            "it",
            "with",
            "as",
            "his",
            "on",
            "be",
            "at",
            "by",
            "this",
            "had",
            "not",
            "are",
            "but",
            "from",
            "they",
            "have",
            "has",
            "were",
            "said",
            "will",
            "one",
            "all",
            "would",
            "there",
            "their",
            "what",
            "about",
            "which",
            "when",
        ]
    ),
    "es": frozenset(
        [
            "el",
            "la",
            "de",
            "que",
            "y",
            "en",
            "los",
            "del",
            "se",
            "las",
            "por",
            "un",
            "para",
            "con",
            "no",
            "una",
            "su",
            "al",
            "lo",
            "como",
            "más",
            "pero",
            "sus",
            "le",
            "ya",
            "este",
            "sí",
            "porque",
            "esta",
            "entre",
            "cuando",
            "muy",
            "sobre",
            "también",
        ]
    ),
    "fr": frozenset(
        [
            "le",
            "de",
            "un",
            "et",
            "être",
            "avoir",
            "que",
            "pour",
            "dans",
            "ce",
            "il",
            "qui",
            "ne",
            "sur",
            "se",
            "pas",
            "plus",
            "par",
            "je",
            "avec",
            "tout",
            "faire",
            "son",
            "mais",
            "nous",
            "comme",
            "les",
            "des",
            "du",
            "au",
            "aux",
            "cette",
            "ces",
        ]
    ),
    "de": frozenset(
        [
            "der",
            "die",
            "und",
            "in",
            "den",
            "von",
            "zu",
            "das",
            "mit",
            "sich",
            "des",
            "auf",
            "für",
            "ist",
            "im",
            "dem",
            "nicht",
            "ein",
            "eine",
            "als",
            "auch",
            "es",
            "an",
            "werden",
            "aus",
            "er",
            "hat",
            "dass",
            "sie",
            "nach",
            "wird",
            "bei",
        ]
    ),
    "pt": frozenset(
        [
            "de",
            "que",
            "não",
            "um",
            "para",
            "com",
            "uma",
            "os",
            "no",
            "se",
            "na",
            "por",
            "mais",
            "das",
            "dos",
            "como",
            "mas",
            "ao",
            "ele",
            "das",
            "à",
            "seu",
            "sua",
            "ou",
            "quando",
            "muito",
            "nos",
            "já",
            "eu",
            "também",
            "só",
            "pelo",
        ]
    ),
    "it": frozenset(
        [
            "di",
            "che",
            "il",
            "la",
            "per",
            "una",
            "con",
            "non",
            "del",
            "sono",
            "le",
            "si",
            "da",
            "in",
            "al",
            "lo",
            "come",
            "più",
            "anche",
            "dei",
            "della",
            "nel",
            "alla",
            "questo",
            "ma",
            "ha",
            "sono",
            "loro",
            "quando",
            "essere",
        ]
    ),
    "nl": frozenset(
        [
            "de",
            "van",
            "het",
            "een",
            "en",
            "dat",
            "is",
            "op",
            "te",
            "met",
            "voor",
            "zijn",
            "aan",
            "er",
            "maar",
            "om",
            "door",
            "over",
            "ze",
            "uit",
            "bij",
            "nog",
            "kan",
            "worden",
            "wordt",
            "naar",
            "heeft",
            "ook",
            "dan",
        ]
    ),
    "tr": frozenset(
        [
            "bir",
            "ve",
            "bu",
            "için",
            "ile",
            "de",
            "da",
            "olarak",
            "çok",
            "daha",
            "en",
            "olan",
            "sonra",
            "kadar",
            "ancak",
            "gibi",
            "ise",
            "her",
            "ya",
            "ama",
            "olduğunu",
            "göre",
            "var",
            "yok",
        ]
    ),
}

#: Unicode block prefixes that identify a script outright.
_SCRIPT_HINTS: Final[tuple[tuple[str, str], ...]] = (
    ("ARABIC", "ar"),
    ("HEBREW", "he"),
    ("CYRILLIC", "ru"),
    ("GREEK", "el"),
    ("HIRAGANA", "ja"),
    ("KATAKANA", "ja"),
    ("HANGUL", "ko"),
    ("CJK", "zh"),
    ("THAI", "th"),
    ("DEVANAGARI", "hi"),
    ("BENGALI", "bn"),
    ("ARMENIAN", "hy"),
    ("GEORGIAN", "ka"),
)

#: Characters exclusive to Persian/Urdu inside the Arabic script.
_PERSIAN_CHARS: Final[frozenset[str]] = frozenset("پچژگک")
#: Words that appear in exactly one profile. A hit on one of these is far more
#: informative than a hit on a word several Romance languages share ("de",
#: "la", "que"), which is what makes short-text detection reliable.
_DISCRIMINATIVE: Final[dict[str, frozenset[str]]] = {
    language: frozenset(
        word
        for word in words
        if not any(word in other for name, other in _PROFILES.items() if name != language)
    )
    for language, words in _PROFILES.items()
}

_PERSIAN_WORDS: Final[frozenset[str]] = frozenset(
    [
        "و",
        "در",
        "به",
        "از",
        "که",
        "این",
        "را",
        "با",
        "است",
        "برای",
        "های",
        "می",
        "یک",
        "آن",
        "هم",
        "تا",
        "کرد",
        "شد",
        "بود",
        "خود",
        "اما",
        "اگر",
        "یا",
        "نیز",
    ]
)


@dataclass(frozen=True, slots=True)
class LanguageDetection:
    """Detected language and how much to trust it."""

    language: str | None
    confidence: float
    script: str | None = None

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.5


def _dominant_script(text: str) -> tuple[str | None, float]:
    """Most common Unicode script in ``text`` and its share of the letters."""
    counts: dict[str, int] = {}
    letters = 0
    for char in text[:2000]:
        if not char.isalpha():
            continue
        letters += 1
        try:
            name = unicodedata.name(char)
        except ValueError:
            continue
        block = name.split(" ")[0]
        counts[block] = counts.get(block, 0) + 1

    if not letters or not counts:
        return None, 0.0
    block, count = max(counts.items(), key=lambda item: item[1])
    return block, count / letters


def detect_language(text: str | None) -> LanguageDetection:
    """Identify the language of ``text``."""
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return LanguageDetection(language=None, confidence=0.0)

    sample = text[:4000]
    block, share = _dominant_script(sample)

    if block is not None and share >= 0.5:
        for prefix, language in _SCRIPT_HINTS:
            if block.startswith(prefix):
                if language == "ar":
                    return _disambiguate_arabic(sample, share)
                return LanguageDetection(
                    language=language, confidence=round(share, 3), script=block
                )

    tokens = [token for token in _TOKEN_RE.findall(sample.casefold()) if len(token) > 1]
    if len(tokens) < 8:
        return LanguageDetection(language=None, confidence=0.0, script=block)

    total = len(tokens)
    scores: dict[str, float] = {}
    for language, profile in _PROFILES.items():
        shared_hits = sum(1 for token in tokens if token in profile)
        if not shared_hits:
            continue
        unique = _DISCRIMINATIVE[language]
        unique_hits = sum(1 for token in tokens if token in unique)
        # Unique function words count triple: "des"/"une" identify French far
        # more strongly than "de"/"la", which Spanish and Italian share.
        scores[language] = (shared_hits + 2 * unique_hits) / total

    if not scores:
        return LanguageDetection(language=None, confidence=0.0, script=block)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_language, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

    # Confidence combines absolute stop-word density with the margin over the
    # runner-up: "many hits, and clearly more than the alternative".
    density = min(1.0, best_score / 0.25)
    margin = (best_score - runner_up) / best_score if best_score else 0.0
    confidence = round(min(0.99, 0.5 * density + 0.5 * margin), 3)

    if confidence < 0.15:
        return LanguageDetection(language=None, confidence=confidence, script=block)
    return LanguageDetection(language=best_language, confidence=confidence, script=block)


def _disambiguate_arabic(text: str, share: float) -> LanguageDetection:
    """Persian and Arabic share a script; letters and function words separate them."""
    if any(char in _PERSIAN_CHARS for char in text):
        return LanguageDetection(
            language="fa", confidence=round(min(0.95, share), 3), script="ARABIC"
        )
    tokens = set(_TOKEN_RE.findall(text))
    if len(tokens & _PERSIAN_WORDS) >= 3:
        return LanguageDetection(
            language="fa", confidence=round(min(0.9, share), 3), script="ARABIC"
        )
    return LanguageDetection(language="ar", confidence=round(share, 3), script="ARABIC")


def is_supported(language: str | None, supported: frozenset[str]) -> bool:
    """True when ``language`` is unknown (benefit of the doubt) or supported."""
    return language is None or language in supported


__all__ = ["LanguageDetection", "detect_language", "is_supported"]
