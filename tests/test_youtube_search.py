from unittest.mock import patch

import pytest

from cover_search.youtube import search_youtube_videos


def test_search_youtube_videos_uses_video_metadata_without_download():
    data = {
        "entries": [
            {
                "id": "ABC123xyz",
                "title": "Hallelujah - acoustic cover",
                "channel": "Test Performer",
            },
            {"id": "XYZ456abc", "title": "Hallelujah", "uploader": "Other Uploader"},
        ]
    }

    class FakeDL:
        def __init__(self, options):
            assert options["skip_download"] is True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def extract_info(self, query, download):
            assert query == "ytsearch12:Hallelujah cover"
            assert download is False
            return data

    with patch("yt_dlp.YoutubeDL", FakeDL):
        videos = search_youtube_videos("Hallelujah")
    assert videos[0]["channel"] == "Test Performer"
    assert videos[0]["url"] == "https://www.youtube.com/watch?v=ABC123xyz"
    assert videos[1]["channel"] == "Other Uploader"


def test_youtube_video_search_rejects_empty_query():
    with pytest.raises(ValueError, match="song title"):
        search_youtube_videos(" ")
