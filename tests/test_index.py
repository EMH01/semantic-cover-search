import numpy as np
import pytest

from cover_search.index import SongIndex
from cover_search.models import Song


def test_search_orders_by_cosine_similarity():
    index = SongIndex()
    index.add(Song("Original", "Artist", "lyrics"), np.array([1.0, 0.0]))
    index.add(Song("Different", "Other", "lyrics"), np.array([0.0, 1.0]))

    results = index.search(np.array([0.9, 0.1]), top_k=2)

    assert results[0][0].title == "Original"
    assert results[0][1] > results[1][1]


def test_duplicate_song_is_rejected():
    index = SongIndex()
    song = Song("Same", "Artist", "lyrics")
    index.add(song, np.array([1.0, 0.0]))

    with pytest.raises(ValueError, match="already in catalog"):
        index.add(song, np.array([1.0, 0.0]))


def test_empty_index_returns_no_results():
    assert SongIndex().search(np.array([1.0, 0.0])) == []


@pytest.mark.parametrize("vector", [
    np.array([0.0, 0.0]),
    np.array([float("nan"), 1.0]),
    np.array([float("inf"), 1.0]),
    np.array([]),
])
def test_invalid_embeddings_are_rejected(vector):
    index = SongIndex()
    with pytest.raises(ValueError, match="Embedding"):
        index.add(Song("Track", "Artist", "words"), vector)


def test_duplicates_ignore_case():
    index = SongIndex()
    index.add(Song("Song", "Artist", "words"), np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="already in catalog"):
        index.add(Song("SONG", "artist", "words"), np.array([0.0, 1.0]))


def test_metadata_only_entry_is_not_searched():
    index = SongIndex()
    index.add(Song("Unknown", "Unknown performer", "", source="upload"))
    index.add(Song("Known", "Artist", "lyrics", source="youtube"), np.array([1.0, 0.0]))
    assert len(index.songs) == 2
    assert index.indexed_count == 1
    assert [song.title for song, _ in index.search(np.array([1.0, 0.0]))] == ["Known"]
