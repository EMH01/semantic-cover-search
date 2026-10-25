"""Pairwise catalog comparison regressions, without external models or APIs."""
import numpy as np
import pytest

from cover_search.comparison import lyric_phrase_overlap
from cover_search.index import SongIndex
from cover_search.models import Song


def test_pairwise_comparison_ranks_shared_lyrics_first():
    catalog = SongIndex()
    catalog.add(
        Song("Original", "Artist A", "hello world this is the song"),
        np.array([1.0, 0.0], dtype=np.float32),
    )
    catalog.add(
        Song("Cover", "Artist B", "hello world this is the song"),
        np.array([0.95, 0.05], dtype=np.float32),
    )
    catalog.add(
        Song("Different", "Artist C", "completely unrelated musical words"),
        np.array([0.0, 1.0], dtype=np.float32),
    )
    pairs = catalog.compare_all()

    assert len(pairs) == 3  # n choose 2, never compare a song with itself
    assert {pairs[0].first.title, pairs[0].second.title} == {"Original", "Cover"}
    assert pairs[0].exact_text is True
    assert pairs[0].phrase_overlap == 1.0
    assert pairs[0].semantic_similarity == pytest.approx(0.9986, abs=0.0005)
    assert pairs[-1].phrase_overlap == 0.0


def test_metadata_only_entries_are_excluded_and_count_is_preserved():
    catalog = SongIndex()
    catalog.add(Song("No text", "Artist", "", text_origin="none"))
    catalog.add(Song("A", "X", "a lyric phrase"), np.array([1, 0]))
    catalog.add(Song("B", "Y", "another phrase"), np.array([0, 1]))
    assert len(catalog.songs) == 3
    assert catalog.indexed_count == 2
    assert len(catalog.compare_all()) == 1


def test_compare_catalog_does_not_trigger_inference_or_change_search():
    catalog = SongIndex()
    catalog.add(Song("One", "X", "words here"), np.array([1.0, 0.0]))
    catalog.add(Song("Two", "Y", "words there"), np.array([0.0, 1.0]))
    first = catalog.compare_all()
    second = catalog.compare_all()
    assert first == second
    assert catalog.search(np.array([1.0, 0.0]))[0][0].title == "One"


def test_not_enough_indexed_songs_yields_no_pairs():
    catalog = SongIndex()
    assert catalog.compare_all() == []
    catalog.add(Song("Solo", "X", "one lyric"), np.array([1, 0]))
    assert catalog.compare_all() == []


@pytest.mark.parametrize(
    ("one", "two", "expected"),
    [
        ("One, TWO! three", "one two three", 1.0),
        ("one two three four", "one two three five", 1 / 3),
        ("one two three", "four five six", 0.0),
        ("", "some lyrics", 0.0),
    ],
)
def test_shared_phrase_overlap(one, two, expected):
    assert lyric_phrase_overlap(one, two) == pytest.approx(expected)


def test_semantic_similarity_is_not_interpreted_as_shared_lyrics():
    catalog = SongIndex()
    catalog.add(Song("First", "A", "the moon is shining"), np.array([1.0, 0.0]))
    catalog.add(Song("Second", "B", "my day is lovely"), np.array([1.0, 0.0]))
    match = catalog.compare_all()[0]
    assert match.phrase_overlap == 0.0
    assert match.semantic_similarity == pytest.approx(1.0)
    assert match.exact_text is False
