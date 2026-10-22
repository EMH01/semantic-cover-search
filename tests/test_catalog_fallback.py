"""Integration tests for the lyrics -> ASR -> metadata-only fallback sequence."""

import numpy as np

from cover_search.index import SongIndex
from cover_search.lyrics import LyricsLookupError, LyricsMatch
from cover_search.youtube import YouTubeAudio


def test_reference_lyrics_do_not_trigger_youtube_download(monkeypatch):
    import app

    cat = SongIndex()
    monkeypatch.setattr(
        app,
        "find_lyrics",
        lambda *args: LyricsMatch(
            "Hallelujah", "Leonard Cohen", "Reference lyric words", "https://lrclib.net/api/get/1"
        ),
    )

    def unexpectedly_download(*args):
        raise AssertionError("No download should happen when reference lyrics are found")

    monkeypatch.setattr(app, "download_audio", unexpectedly_download)
    monkeypatch.setattr(app, "embed", lambda text: np.array([1.0, 0.0]))

    status, rows, _, view = app.add_known_song(
        cat, "Hallelujah (Cover)", "Uploader: Singer", "youtube",
        lyric_title="Hallelujah", fallback_youtube_url="https://www.youtube.com/watch?v=123",
    )
    assert cat.songs[0].text_origin == "lrclib"
    assert cat.indexed_count == 1
    assert "LRCLIB" in status
    assert "Reference lyrics" in rows[0][-1]
    assert "View source" in view


def test_audio_is_transcribed_when_lyrics_are_missing(monkeypatch):
    import app

    cat = SongIndex()
    monkeypatch.setattr(app, "find_lyrics", lambda *args: None)
    monkeypatch.setattr(
        app,
        "download_audio",
        lambda url: YouTubeAudio("Title", "Uploader", "video.webm", b"sample audio"),
    )
    monkeypatch.setattr(app, "transcribe", lambda data, name: "Recognized example words")
    monkeypatch.setattr(app, "embed", lambda text: np.array([1.0, 0.0]))

    status, rows, _, view = app.add_known_song(
        cat, "Title", "Uploader: Singer", "youtube",
        lyric_title="Title", fallback_youtube_url="https://www.youtube.com/watch?v=123",
    )
    assert cat.songs[0].text_origin == "asr"
    assert cat.songs[0].transcript == "Recognized example words"
    assert cat.indexed_count == 1
    assert "transcribed automatically" in status
    assert rows[0][-1] == "Automatic transcription"
    assert "Recognized example words" in view


def test_lrclib_service_error_still_tries_audio(monkeypatch):
    import app

    def unavailable(*args):
        raise LyricsLookupError("LRCLIB could not be reached.")

    cat = SongIndex()
    monkeypatch.setattr(app, "find_lyrics", unavailable)
    monkeypatch.setattr(
        app,
        "download_audio",
        lambda url: YouTubeAudio("Title", "Singer", "video.webm", b"audio"),
    )
    monkeypatch.setattr(app, "transcribe", lambda data, name: "Sung words")
    monkeypatch.setattr(app, "embed", lambda text: np.array([1.0, 0.0]))

    status, _, _, _ = app.add_known_song(
        cat, "Title", "Uploader: Singer", "youtube", fallback_youtube_url="https://x.test",
    )
    assert "LRCLIB issue" in status
    assert cat.songs[0].text_origin == "asr"


def test_empty_asr_keeps_metadata_without_crashing(monkeypatch):
    import app

    cat = SongIndex()
    monkeypatch.setattr(app, "find_lyrics", lambda *args: None)
    monkeypatch.setattr(
        app,
        "download_audio",
        lambda url: YouTubeAudio("Title", "Singer", "video.webm", b"audio"),
    )
    monkeypatch.setattr(
        app, "transcribe", lambda data, name: (_ for _ in ()).throw(ValueError("empty"))
    )
    status, rows, _, view = app.add_known_song(
        cat, "Title", "Uploader: Singer", "youtube", fallback_youtube_url="https://x.test",
    )
    assert len(cat.songs) == 1
    assert cat.indexed_count == 0
    assert rows[0][-1] == "Metadata only"
    assert "speech recognition failed" in status
    assert "metadata-only" in view
