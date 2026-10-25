"""Integration tests for the lyrics -> ASR -> metadata-only fallback sequence."""

import numpy as np

from cover_search.index import SongIndex
from cover_search.models import Song
from cover_search.lyrics import LyricsLookupError, LyricsMatch
from cover_search.youtube import YouTubeAudio


def test_reference_lyrics_downloads_once_for_melody_without_asr(monkeypatch):
    import app

    cat = SongIndex()
    monkeypatch.setattr(
        app,
        "find_lyrics",
        lambda *args: LyricsMatch(
            "Hallelujah", "Leonard Cohen", "Reference lyric words",
            "https://lrclib.net/api/get/1",
        ),
    )
    downloads = []

    def get_audio(url):
        downloads.append(url)
        return YouTubeAudio("Hallelujah", "Artist", "video.webm", b"audio")

    monkeypatch.setattr(app, "download_audio", get_audio)
    monkeypatch.setattr(
        app, "transcribe",
        lambda *args: (_ for _ in ()).throw(AssertionError("ASR must not run")),
    )
    monkeypatch.setattr(
        app, "extract_chroma",
        lambda data: np.ones((12, 12), dtype=np.float32),
    )
    monkeypatch.setattr(app, "embed", lambda text: np.array([1.0, 0.0]))

    status, rows, _, view = app.add_known_song(
        cat, "Hallelujah (Cover)", "Uploader: Singer", "youtube",
        lyric_title="Hallelujah",
        fallback_youtube_url="https://www.youtube.com/watch?v=123",
    )
    assert len(downloads) == 1
    assert cat.songs[0].text_origin == "lrclib"
    assert cat.indexed_count == 1
    assert cat.melody_count == 1
    assert rows[0][3] == "Ready"
    assert "Melody features available" in status
    assert "View source" in view


def test_blocked_youtube_audio_keeps_reference_lyrics(monkeypatch):
    import app

    cat = SongIndex()
    monkeypatch.setattr(
        app, "find_lyrics",
        lambda *args: LyricsMatch(
            "Hallelujah", "Leonard Cohen", "Reference lyric words",
            "https://lrclib.net/api/get/1",
        ),
    )
    monkeypatch.setattr(
        app, "download_audio",
        lambda *args: (_ for _ in ()).throw(RuntimeError("Unavailable format")),
    )
    monkeypatch.setattr(app, "embed", lambda _: np.array([1.0, 0.0]))
    monkeypatch.setattr(
        app, "transcribe",
        lambda *args: (_ for _ in ()).throw(AssertionError("No ASR required")),
    )

    status, rows, _, view = app.add_known_song(
        cat, "Hallelujah", "Uploader: Singer", "youtube",
        fallback_youtube_url="https://www.youtube.com/watch?v=123",
    )
    assert cat.indexed_count == 1
    assert cat.melody_count == 0
    assert rows[0][3] == "Not analyzed"
    assert rows[0][-1] == "Reference lyrics (LRCLIB)"
    assert "Melody unavailable" in status
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
    monkeypatch.setattr(
        app, "extract_chroma", lambda data: np.ones((12, 12), dtype=np.float32)
    )
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
    assert "Lyrics service issue" in status
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
    assert "Audio transcription failed" in status
    assert "metadata-only" in view


def test_retry_existing_metadata_only_song_after_services_recover(monkeypatch):
    import app

    cat = SongIndex()
    cat.add(Song("Angels Like You", "Uploader: Miley Cyrus - Topic", "",
                 source="youtube", text_origin="none"))
    monkeypatch.setattr(
        app, "find_lyrics",
        lambda *args: LyricsMatch(
            "Angels Like You", "Miley Cyrus", "Some reference lyrics",
            "https://lrclib.net/api/get/15",
        ),
    )
    monkeypatch.setattr(app, "embed", lambda _: np.array([1.0, 0.0]))
    monkeypatch.setattr(
        app, "download_audio",
        lambda url: YouTubeAudio("Angels Like You", "Miley Cyrus", "song.webm", b"audio"),
    )
    monkeypatch.setattr(
        app, "extract_chroma", lambda data: np.ones((10, 12), dtype=np.float32)
    )
    status, rows, _, _ = app.add_known_song(
        cat, "Angels Like You", "Uploader: Miley Cyrus - Topic", "youtube",
        lyric_title="Angels Like You",
        fallback_youtube_url="https://www.youtube.com/watch?v=123",
    )
    assert "Retried in local catalog" in status
    assert len(cat.songs) == 1
    assert cat.indexed_count == 1
    assert cat.melody_count == 1
    assert rows[0][3] == "Ready"
    assert rows[0][4] == "Reference lyrics (LRCLIB)"
