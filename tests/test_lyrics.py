from unittest.mock import patch

import httpx
import pytest

from cover_search.lyrics import (
    LyricsLookupError,
    LyricsMatch,
    find_lyrics,
    lyric_title_for_result,
    render_lyrics,
    title_from_video,
)


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


def test_search_phrase_including_performer_uses_actual_track_title():
    assert (
        lyric_title_for_result(
            "Leonard Cohen - Hallelujah (Official Live in London 2008)",
            "Leonard Cohen Hallelujah",
        )
        == "Hallelujah"
    )


def test_lrclib_broad_query_fallback(monkeypatch):
    seen = []
    payloads = iter([
        [],
        [{"id": 15, "trackName": "Hallelujah", "artistName": "Leonard Cohen",
          "plainLyrics": "Some sample words"}],
    ])

    class Response:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, params):
            seen.append(params)
            return Response(next(payloads))

    monkeypatch.setattr("cover_search.lyrics.httpx.Client", Client)
    monkeypatch.setattr("cover_search.lyrics.time.sleep", lambda *_: None)
    match = find_lyrics("Hallelujah")
    assert match is not None
    assert match.reference_artist == "Leonard Cohen"
    assert seen == [{"track_name": "Hallelujah"}, {"q": "Hallelujah"}]


def test_lrclib_error_is_not_misreported_as_lyrics_missing(monkeypatch):
    request = httpx.Request("GET", "https://lrclib.net/api/search")
    response = httpx.Response(429, request=request)

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, *args, **kwargs):
            return response

    monkeypatch.setattr("cover_search.lyrics.httpx.Client", Client)
    with pytest.raises(LyricsLookupError, match="rate-limiting"):
        find_lyrics("Hallelujah")


def test_synced_lyrics_without_plain_are_supported(monkeypatch):
    data = [
        {"id": 23, "trackName": "Song", "artistName": "Singer",
         "plainLyrics": None,
         "syncedLyrics": "[00:01.20] First line\n[00:03.40] Second line"}
    ]

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return data

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr("cover_search.lyrics.httpx.Client", Client)
    match = find_lyrics("Song", "Singer")
    assert match is not None
    assert match.text == "First line\nSecond line"


def test_lyric_view_is_readable_and_escapes_provider_content():
    match = LyricsMatch(
        "Song <test>", "Singer", "Verse <one>\nVerse two\n\nNext verse",
        "https://lrclib.net/api/get/23",
    )
    markup = render_lyrics(match)
    assert "Verse &lt;one&gt;" in markup
    assert "max-height:450px" in markup
    assert "View source" in markup
    assert "Next verse" in markup
