"""Split a risk section into passages of about 170 words.

The embedding model reads the first 256 tokens (roughly 190 words), so a
passage should fit in that window.
"""
from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.;])\s+(?=[A-Z])")
_ONLY_SYMBOLS = re.compile(r"[\d\s|\-–—.•]+")


def _is_noise(line: str) -> bool:
    """Headings, page numbers and page footers carry no risk description."""
    return len(line.split()) < 8 or bool(_ONLY_SYMBOLS.fullmatch(line))


def split_into_chunks(text: str, target_words: int = 150, min_words: int = 50, max_words: int = 300) -> list[str]:
    """Passages that follow paragraph and sentence borders.

    Lines are added to a passage until it reaches `target_words`, so a passage
    never stops in the middle of a sentence. A short tail is dropped. A single
    "sentence" longer than `max_words` (long bullet lists) is cut by length.
    """
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln and not _is_noise(ln)]

    units: list[str] = []
    for ln in lines:
        if len(ln.split()) > target_words:   # very long paragraph: cut at sentence ends
            for sentence in _SENTENCE_END.split(ln):
                words = sentence.split()
                units.extend(" ".join(words[i: i + max_words]) for i in range(0, len(words), max_words))
        else:
            units.append(ln)

    chunks, current = [], []
    for unit in units:
        current.extend(unit.split())
        if len(current) >= target_words:
            chunks.append(" ".join(current))
            current = []
    if len(current) >= min_words:
        chunks.append(" ".join(current))
    return chunks
