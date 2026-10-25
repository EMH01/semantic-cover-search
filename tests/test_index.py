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



def test_retry_metadata_upgrades_lyrics_without_duplicate_rows():
    index = SongIndex()
    old = Song("Angels Like You", "Miley Cyrus", "", text_origin="none")
    index.add(old)
    upgraded = Song(
        "Angels Like You", "Miley Cyrus", "Reference lyrics",
        source="youtube", text_origin="lrclib",
    )
    refreshed = index.retry_metadata(upgraded, np.array([1.0, 0.0]))
    assert refreshed == upgraded
    assert index.songs == (upgraded,)
    assert index.indexed_count == 1
    assert index.search(np.array([1.0, 0.0]))[0][0] == upgraded
    with pytest.raises(ValueError, match="already has indexed lyrics"):
        index.retry_metadata(upgraded, np.array([1.0, 0.0]))


def test_retry_metadata_without_new_data_keeps_existing_melody():
    index = SongIndex()
    old = Song("Song", "Singer", "", source="upload", text_origin="none")
    index.add(old)
    chroma = np.ones((8, 12), dtype=np.float32)
    index.index_melody(old, chroma)
    reattempt = Song("Song", "Singer", "", source="upload", text_origin="none")
    assert index.retry_metadata(reattempt) is old
    assert index.songs == (old,)
    assert index.melody_count == 1
    upgraded = Song("Song", "Singer", "Recognized words", text_origin="asr")
    result = index.retry_metadata(upgraded, np.array([0.0, 1.0]))
    assert index.has_melody(result)
    assert index.melody_count == 1
    assert index.indexed_count == 1


def test_retry_metadata_cannot_change_index_dimension():
    index = SongIndex()
    index.add(Song("Known", "A", "words"), np.array([1.0, 0.0]))
    index.add(Song("Pending", "B", "", text_origin="none"))
    with pytest.raises(ValueError, match="dimension"):
        index.retry_metadata(
            Song("Pending", "B", "new words", text_origin="asr"),
            np.array([1.0, 0.0, 0.0]),
        )
    assert index.indexed_count == 1
    assert len(index.songs) == 2
