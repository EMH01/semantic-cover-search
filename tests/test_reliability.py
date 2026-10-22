import numpy as np
import pytest

from cover_search.index import SongIndex
from cover_search.models import Song
from cover_search.youtube import download_audio


def test_youtube_rejects_unrelated_urls_without_network():
    with pytest.raises(ValueError, match="YouTube"):
        download_audio("https://example.com/large-audio")


def test_independent_catalogs_do_not_share_songs():
    first, second = SongIndex(), SongIndex()
    first.add(Song("Original", "Artist", "words"), np.array([1.0, 0.0]))
    assert len(first.songs) == 1
    assert second.songs == ()
