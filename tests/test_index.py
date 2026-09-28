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

    with pytest.raises(ValueError, match="already indexed"):
        index.add(song, np.array([1.0, 0.0]))


def test_empty_index_returns_no_results():
    assert SongIndex().search(np.array([1.0, 0.0])) == []
