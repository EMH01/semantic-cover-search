from unittest.mock import patch

import pytest

from cover_search.discovery import discover_versions, find_works, recordings_for_work


def test_find_works_parses_search_results():
    payload = {"works": [{"id": "abc", "title": "Hallelujah", "score": 98}]}
    with patch("cover_search.discovery._fetch", return_value=payload) as fetch:
        assert find_works("Hallelujah") == [
            {"id": "abc", "title": "Hallelujah", "score": 98}
        ]
        assert fetch.call_args.args[0] == "work"


def test_find_works_rejects_blank_title():
    with pytest.raises(ValueError, match="song title"):
        find_works(" ")


def test_recordings_for_work_parses_artist_credit():
    work_id = "d73c9c72-fd4a-4e9d-9e96-8c4a89c7ccfa"
    payload = {
        "recordings": [
            {
                "id": "abc",
                "title": "Hallelujah",
                "artist-credit": [{"name": "Jeff Buckley"}],
            }
        ]
    }
    with patch("cover_search.discovery._fetch", return_value=payload):
        results = recordings_for_work(work_id)
    assert results[0]["artist"] == "Jeff Buckley"
    assert results[0]["url"] == "https://musicbrainz.org/recording/abc"


def test_discover_versions_returns_empty_when_no_work_found():
    with patch("cover_search.discovery.find_works", return_value=[]):
        assert discover_versions("unknown") == ([], [])
