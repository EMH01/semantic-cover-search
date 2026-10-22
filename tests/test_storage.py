import sqlite3

import numpy as np
import pytest

from cover_search.models import Song
from cover_search.storage import load_catalog, store_song


def test_catalog_survives_reload(tmp_path):
    database = tmp_path / "catalog.sqlite3"
    original = Song("Hallelujah", "Jeff Buckley", "recognized words", "youtube")
    store_song(original, np.array([3, 4], dtype=np.float32), database)
    restored = load_catalog(database)

    assert restored.songs == (original,)
    matches = restored.search(np.array([0.6, 0.8]))
    assert matches[0][0] == original
    assert matches[0][1] == pytest.approx(1.0)


def test_duplicate_is_not_inserted(tmp_path):
    database = tmp_path / "catalog.sqlite3"
    store_song(Song("Song", "Singer", "words"), np.array([1.0, 0.0]), database)
    with pytest.raises(ValueError, match="already indexed"):
        store_song(Song("SONG", "singer", "words"), np.array([1.0, 0.0]), database)
    assert len(load_catalog(database).songs) == 1


def test_invalid_embedding_does_not_create_row(tmp_path):
    database = tmp_path / "catalog.sqlite3"
    with pytest.raises(ValueError, match="Embedding"):
        store_song(Song("Song", "Artist", "words"), np.zeros(2), database)
    assert len(load_catalog(database).songs) == 0


def test_corrupt_vector_is_detected(tmp_path):
    database = tmp_path / "catalog.sqlite3"
    store_song(Song("Song", "Artist", "words"), np.array([1.0, 0.0]), database)
    with sqlite3.connect(database) as conn:
        conn.execute("UPDATE songs SET dimensions = 3")
    with pytest.raises(ValueError, match="invalid embedding"):
        load_catalog(database)
