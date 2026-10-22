from unittest.mock import patch

from cover_search.lyrics import find_lyrics, lyric_title_for_result, title_from_video


def test_video_title_cleanup():
    assert title_from_video("Miley Cyrus - Used To Be Young (IMY2 Cover)") == "Used To Be Young"
    assert title_from_video("Hallelujah (acoustic cover)") == "Hallelujah"


def test_lyrics_must_match_track_title():
    records = [
        {
            "id": 10,
            "trackName": "Another Hallelujah",
            "artistName": "Artist",
            "plainLyrics": "unrelated",
        },
        {
            "id": 11,
            "trackName": "Hallelujah",
            "artistName": "Leonard Cohen",
            "plainLyrics": "Original sample words",
        },
    ]

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return records

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs["headers"]["User-Agent"].startswith("SemanticCoverSearch/")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url, params):
            assert url == "https://lrclib.net/api/search"
            assert params["track_name"] == "Hallelujah"
            return FakeResponse()

    with patch("cover_search.lyrics.httpx.Client", FakeClient):
        match = find_lyrics("Hallelujah")
    assert match.text == "Original sample words"
    assert match.reference_artist == "Leonard Cohen"
    assert match.url == "https://lrclib.net/api/get/11"


def test_unrelated_lyrics_return_none():
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"trackName": "Other song", "plainLyrics": "words", "id": 7}]

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, *args, **kwargs):
            return FakeResponse()

    with patch("cover_search.lyrics.httpx.Client", FakeClient):
        assert find_lyrics("Hallelujah") is None


def test_selected_youtube_result_uses_known_song_title_for_lyrics():
    assert (
        lyric_title_for_result(
            "Hallelujah - Lucy Thomas - (Official Music Video)",
            "Hallelujah",
        )
        == "Hallelujah"
    )


def test_lyric_search_title_must_appear_in_video_title():
    assert (
        lyric_title_for_result("Hallelujah (Acoustic cover)", "Used To Be Young")
        == "Hallelujah"
    )
