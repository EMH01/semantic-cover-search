"""Pairwise lyric comparison for the songs available in the session catalog.

The scores are descriptive similarities, not probabilities of a genuine cover.
"""
import re
import unicodedata
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .models import Song


@dataclass(frozen=True, slots=True)
class LyricComparison:
    first: Song
    second: Song
    phrase_overlap: float
    semantic_similarity: float
    exact_text: bool


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())


def lyric_phrase_overlap(first: str, second: str) -> float:
    """Jaccard similarity of 3-word phrases (single words for short inputs).

    This favors genuinely shared word sequences over vague thematic similarity.
    Repeated choruses count once; score is between 0 and 1.
    """
    a, b = _tokens(first), _tokens(second)
    if not a or not b:
        return 0.0
    n = 3 if min(len(a), len(b)) >= 3 else 1
    left = {tuple(a[i:i + n]) for i in range(len(a) - n + 1)}
    right = {tuple(b[i:i + n]) for i in range(len(b) - n + 1)}
    return len(left & right) / len(left | right)


def compare_all_lyrics(
    songs: list[Song], vectors: list[np.ndarray]
) -> list[LyricComparison]:
    """Compare each unordered pair exactly once using already-computed vectors."""
    if len(songs) != len(vectors):
        raise ValueError("Song and vector counts must match.")
    results: list[LyricComparison] = []
    for i, j in combinations(range(len(songs)), 2):
        first, second = songs[i], songs[j]
        a, b = _tokens(first.transcript), _tokens(second.transcript)
        # Embeddings are stored with L2 normalization, so dot product is cosine.
        cosine = float(np.dot(vectors[i], vectors[j]))
        results.append(
            LyricComparison(
                first=first,
                second=second,
                phrase_overlap=lyric_phrase_overlap(
                    first.transcript, second.transcript
                ),
                semantic_similarity=max(-1.0, min(1.0, cosine)),
                exact_text=bool(a and a == b),
            )
        )
    results.sort(
        key=lambda result: (
            -result.phrase_overlap,
            -result.semantic_similarity,
            result.first.label.casefold(),
            result.second.label.casefold(),
        )
    )
    return results
