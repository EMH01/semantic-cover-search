import numpy as np
import pytest

from cover_search.index import SongIndex
from cover_search.models import Song
from cover_search.youtube import download_audio, validate_youtube_url


def test_youtube_rejects_unrelated_urls_without_network():
    with pytest.raises(ValueError, match="YouTube"):
        download_audio("https://example.com/large-audio")


def test_independent_catalogs_do_not_share_songs():
    first, second = SongIndex(), SongIndex()
    first.add(Song("Original", "Artist", "words"), np.array([1.0, 0.0]))
    assert len(first.songs) == 1
    assert second.songs == ()


@pytest.mark.parametrize("url", [
    "https://music.youtube.com/watch?v=IZ3XM0dOdKM&list=abc",
    "https://www.youtube.com/watch?v=IZ3XM0dOdKM",
    "https://youtu.be/IZ3XM0dOdKM",
])
def test_youtube_video_variants_are_accepted(url):
    assert validate_youtube_url(url) == url


@pytest.mark.parametrize("url", [
    "https://music.youtube.com.evil.example/watch?v=123",
    "https://music.youtube.com/playlist?list=123",
    "http://www.youtube.com/watch?v=123",
])
def test_youtube_invalid_urls_are_rejected(url):
    with pytest.raises(ValueError):
        validate_youtube_url(url)
