"""Extractive summarisation.

A TextRank-flavoured sentence ranker: sentences are scored on term salience
(TF-IDF weight of the words they contain), position (news is written
inverted-pyramid, so early sentences matter), and overlap with the headline.
The top sentences are then emitted **in their original order**, which keeps the
summary readable rather than a jumble of high-scoring fragments.

Extractive rather than abstractive on purpose: it cannot hallucinate facts, and
it needs no model weights.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Final

from app.core.utils import truncate
from app.processing.deduplication.similarity import tokenize

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])|\n{2,}")
MIN_SENTENCE_CHARS: Final[int] = 40
MAX_SENTENCE_CHARS: Final[int] = 400
MAX_INPUT_CHARS: Final[int] = 40_000


@dataclass(frozen=True, slots=True)
class Summary:
    """Extractive summary plus the sentences it was built from."""

    text: str
    sentences: tuple[str, ...] = ()
    compression: float = 0.0


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, keeping only usable ones."""
    if not text:
        return []
    parts = _SENTENCE_RE.split(text[:MAX_INPUT_CHARS].replace("\n", " "))
    return [
        sentence.strip()
        for sentence in parts
        if MIN_SENTENCE_CHARS <= len(sentence.strip()) <= MAX_SENTENCE_CHARS
    ]


def summarize(
    text: str | None,
    *,
    title: str | None = None,
    max_sentences: int = 3,
    max_chars: int = 600,
) -> Summary:
    """Produce an extractive summary of ``text``."""
    if not text or not text.strip():
        return Summary(text="", sentences=(), compression=0.0)

    sentences = split_sentences(text)
    if not sentences:
        return Summary(text=truncate(text, max_chars), sentences=(), compression=1.0)
    if len(sentences) <= max_sentences:
        joined = " ".join(sentences)
        return Summary(
            text=truncate(joined, max_chars), sentences=tuple(sentences), compression=1.0
        )

    # Term weights: sub-linear TF so a repeated word does not dominate.
    document_tokens = tokenize(text)
    frequencies = Counter(document_tokens)
    if not frequencies:
        return Summary(text=truncate(sentences[0], max_chars), sentences=(sentences[0],))
    peak = max(frequencies.values())
    weights = {
        term: (1 + math.log(count)) / (1 + math.log(peak)) for term, count in frequencies.items()
    }

    title_tokens = set(tokenize(title or ""))
    total = len(sentences)
    scored: list[tuple[int, float]] = []

    for index, sentence in enumerate(sentences):
        tokens = tokenize(sentence)
        if not tokens:
            continue
        salience = sum(weights.get(token, 0.0) for token in tokens) / math.sqrt(len(tokens))
        # Inverted pyramid: the lede carries the most information.
        position = 1.0 - (index / total) * 0.5
        title_overlap = len(set(tokens) & title_tokens) / len(title_tokens) if title_tokens else 0.0
        scored.append((index, salience * position * (1.0 + 0.6 * title_overlap)))

    if not scored:
        return Summary(text=truncate(sentences[0], max_chars), sentences=(sentences[0],))

    scored.sort(key=lambda item: item[1], reverse=True)
    chosen = sorted(index for index, _ in scored[:max_sentences])
    selected = tuple(sentences[index] for index in chosen)
    summary_text = truncate(" ".join(selected), max_chars)

    return Summary(
        text=summary_text,
        sentences=selected,
        compression=round(len(summary_text) / max(1, len(text)), 4),
    )


def lede(text: str | None, *, max_chars: int = 300) -> str:
    """First substantial sentence - a cheap fallback when scoring is overkill."""
    for sentence in split_sentences(text or ""):
        return truncate(sentence, max_chars)
    return truncate(text or "", max_chars)


__all__ = ["Summary", "lede", "split_sentences", "summarize"]
