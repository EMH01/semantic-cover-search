"""Pairwise comparisons of lyrics and available audio from the session catalog.

Text and melody metrics are independent; none imply cover-identification confidence.
"""
import re
import unicodedata
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .melody import MelodyError, melody_match
from .models import Song


@dataclass(frozen=True, slots=True)
class LyricComparison:
    first: Song
    second: Song
    phrase_overlap: float | None
    semantic_similarity: float | None
    exact_text: bool
    melody_similarity: float | None = None


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())


def lyric_phrase_overlap(first: str, second: str) -> float:
    """Jaccard overlap of consecutive three-word sequences (or single words)."""
    a, b = _tokens(first), _tokens(second)
    if not a or not b:
        return 0.0
    n = 3 if min(len(a), len(b)) >= 3 else 1
    left = {tuple(a[i:i + n]) for i in range(len(a) - n + 1)}
    right = {tuple(b[i:i + n]) for i in range(len(b) - n + 1)}
    return len(left & right) / len(left | right)


def _compare_audio(first: np.ndarray, second: np.ndarray) -> float | None:
    """Compare short reference excerpts by chroma alignment.

    This is an exploratory approximation: accompaniment may dominate pitch classes.
    Avoid comparing long full-track matrices in both directions on every pair.
    """
    shorter, longer = sorted((first, second), key=lambda value: value.shape[0])
    # Five frames per second, representative first 25 seconds of the shorter file.
    segment = shorter[:125]
    try:
        score, _offset, _shift = melody_match(segment, longer)
    except MelodyError:
        return None
    return score


def compare_all_lyrics(
    songs: list[Song],
    vectors: list[np.ndarray],
    melodies: dict[Song, np.ndarray] | None = None,
) -> list[LyricComparison]:
    """Compare each pair sharing at least one measurable text/audio modality.

    No external API or model is invoked: embeddings and chroma are cached in memory.
    """
    if len(songs) != len(vectors):
        raise ValueError("Song and vector counts must match.")
    vector_by_song = dict(zip(songs, vectors, strict=True))
    melodies = melodies or {}
    available = list(dict.fromkeys([*songs, *melodies.keys()]))
    results: list[LyricComparison] = []
    for first, second in combinations(available, 2):
        left_vec = vector_by_song.get(first)
        right_vec = vector_by_song.get(second)
        lexical = semantic = None
        identical = False
        if left_vec is not None and right_vec is not None:
            lexical = lyric_phrase_overlap(first.transcript, second.transcript)
            semantic = float(
                np.clip(np.dot(left_vec, right_vec), -1.0, 1.0)
            )
            identical = bool(
                _tokens(first.transcript)
                and _tokens(first.transcript) == _tokens(second.transcript)
            )
        audio_score = None
        if first in melodies and second in melodies:
            audio_score = _compare_audio(melodies[first], melodies[second])
        if lexical is None and audio_score is None:
            continue
        results.append(
            LyricComparison(
                first=first,
                second=second,
                phrase_overlap=lexical,
                semantic_similarity=semantic,
                exact_text=identical,
                melody_similarity=audio_score,
            )
        )
    results.sort(
        key=lambda item: (
            -(item.phrase_overlap if item.phrase_overlap is not None else -1),
            -(item.melody_similarity if item.melody_similarity is not None else -1),
            -(item.semantic_similarity if item.semantic_similarity is not None else -1),
            item.first.label.casefold(),
            item.second.label.casefold(),
        )
    )
    return results
