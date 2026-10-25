"""Regression tests for automatic melody comparison without extra UI controls."""

from pathlib import Path

import numpy as np

from cover_search.index import SongIndex
from cover_search.lyrics import LyricsMatch


def _notes():
    pitches = [0, 2, 4, 5, 7, 9, 7, 5, 4, 2, 0, 11]
    frames = np.zeros((len(pitches), 12), dtype=np.float32)
    frames[np.arange(len(pitches)), pitches] = 1
    return frames


def test_audio_upload_indexes_melody_automatically(tmp_path, monkeypatch):
    import app

    audio = tmp_path / "song.wav"
    audio.write_bytes(b"placeholder audio")
    monkeypatch.setattr(app, "audio_metadata", lambda _: ("Song", "Singer"))
    monkeypatch.setattr(app, "read_audio", lambda _: b"decoded audio")
    monkeypatch.setattr(app, "extract_chroma", lambda *args, **kwargs: _notes())
    monkeypatch.setattr(
        app, "find_lyrics",
        lambda *args: LyricsMatch(
            "Song", "Singer", "A short example lyric", "https://lrclib.net/api/get/1"
        ),
    )
    monkeypatch.setattr(app, "embed", lambda text: np.array([1.0, 0.0]))
    status, rows, catalog, _view = app.add_upload(
        str(audio), "", "", SongIndex(),
        progress=lambda *args, **kwargs: None,
    )
    assert "Melody features available" in status
    assert catalog.melody_count == 1
    assert catalog.indexed_count == 1
    assert rows[0][3] == "Ready"
    assert rows[0][4] == "Reference lyrics (LRCLIB)"


def test_youtube_link_automatically_indexes_melody(monkeypatch):
    import app

    monkeypatch.setattr(
        app, "video_metadata",
        lambda _: {
            "title": "Hallelujah", "channel": "Singer",
            "url": "https://www.youtube.com/watch?v=abcd",
        },
    )
    monkeypatch.setattr(
        app, "find_lyrics",
        lambda *args: LyricsMatch(
            "Hallelujah", "Singer", "Example lyrics",
            "https://lrclib.net/api/get/1",
        ),
    )
    monkeypatch.setattr(app, "embed", lambda _: np.array([1.0, 0.0]))
    download_urls = []

    def get_audio(url):
        download_urls.append(url)
        return type("YouTubeAudio", (), {
            "data": b"audio", "filename": "song.webm",
        })()

    monkeypatch.setattr(app, "download_audio", get_audio)
    monkeypatch.setattr(app, "extract_chroma", lambda data: _notes())
    monkeypatch.setattr(
        app, "transcribe",
        lambda *args: (_ for _ in ()).throw(AssertionError("ASR must not run")),
    )
    catalog = SongIndex()
    _status, rows, _, _ = app.add_youtube(
        "https://www.youtube.com/watch?v=abcd",
        catalog, progress=lambda *args, **kwargs: None,
    )
    assert len(download_urls) == 1
    assert catalog.melody_count == 1
    assert catalog.indexed_count == 1
    assert rows[0][3] == "Ready"


def test_youtube_song_uses_single_download_for_asr_and_melody(monkeypatch):
    import app

    monkeypatch.setattr(
        app, "video_metadata",
        lambda _: {
            "title": "Unknown song", "channel": "Singer",
            "url": "https://www.youtube.com/watch?v=abcd",
        },
    )
    monkeypatch.setattr(app, "find_lyrics", lambda *args: None)
    downloads = []
    monkeypatch.setattr(
        app, "download_audio",
        lambda url: (downloads.append(url) or type("Audio", (), {
            "data": b"audio", "filename": "song.webm",
        })()),
    )
    monkeypatch.setattr(app, "transcribe", lambda data, filename: "Sung words")
    monkeypatch.setattr(app, "extract_chroma", lambda data: _notes())
    monkeypatch.setattr(app, "embed", lambda _: np.array([1.0, 0.0]))
    catalog = SongIndex()
    _status, rows, _, _ = app.add_youtube(
        "https://www.youtube.com/watch?v=abcd",
        catalog, progress=lambda *args, **kwargs: None,
    )
    assert len(downloads) == 1
    assert rows[0][3] == "Ready"
    assert rows[0][-1] == "Automatic transcription"


def test_youtube_melody_failure_does_not_break_lyrics(monkeypatch):
    import app

    monkeypatch.setattr(
        app, "video_metadata",
        lambda _: {
            "title": "Hallelujah", "channel": "Singer",
            "url": "https://www.youtube.com/watch?v=abcd",
        },
    )
    monkeypatch.setattr(
        app, "find_lyrics",
        lambda *args: LyricsMatch(
            "Hallelujah", "Singer", "Example lyrics",
            "https://lrclib.net/api/get/1",
        ),
    )
    monkeypatch.setattr(app, "embed", lambda _: np.array([1.0, 0.0]))
    monkeypatch.setattr(
        app, "download_audio",
        lambda url: type("Audio", (), {
            "data": b"audio", "filename": "song.webm",
        })(),
    )
    monkeypatch.setattr(
        app, "extract_chroma",
        lambda data: (_ for _ in ()).throw(app.MelodyError("No melody")),
    )
    catalog = SongIndex()
    status, rows, _, _ = app.add_youtube(
        "https://www.youtube.com/watch?v=abcd",
        catalog, progress=lambda *args, **kwargs: None,
    )
    assert catalog.indexed_count == 1
    assert catalog.melody_count == 0
    assert rows[0][3] == "Not analyzed"
    assert "Melody extraction was unavailable" in status


def test_confirmed_recording_result_uses_same_url_ingestion_as_paste(monkeypatch):
    import app

    catalog = SongIndex()
    seen = []
    sentinel = ("saved", [], catalog, "<p>lyrics</p>")

    def save_link(url, target, progress):
        seen.append((url, target))
        return sentinel

    monkeypatch.setattr(app, "add_youtube", save_link)
    candidates = [
        {
            "title": "Hallelujah - Lucy Thomas",
            "channel": "Lucy Thomas Music",
            "url": "https://www.youtube.com/watch?v=ABC123xyz12",
        },
    ]
    result = app.add_confirmed_recording_result(
        "0", candidates, catalog, progress=lambda *args, **kwargs: None
    )
    assert result is sentinel
    assert seen == [(candidates[0]["url"], catalog)]


def test_recording_search_does_not_index_without_confirmation(monkeypatch, tmp_path):
    import app

    audio = tmp_path / "fragment.wav"
    audio.write_bytes(b"fake recording")
    monkeypatch.setattr(app, "read_audio", lambda *_: b"fake recording")
    monkeypatch.setattr(
        app, "transcribe", lambda *args: "singing some recognizable lyrics"
    )
    monkeypatch.setattr(
        app, "search_youtube_lyrics_fragment",
        lambda words: [
            {
                "title": "Candidate recording",
                "channel": "Artist",
                "url": "https://www.youtube.com/watch?v=ABC123xyz12",
            },
        ],
    )
    monkeypatch.setattr(
        app, "add_youtube",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Confirmation required before adding to catalog")
        ),
    )
    result = app.search_recording_on_youtube(
        str(audio), progress=lambda *args, **kwargs: None
    )
    assert len(result[2]) == 1
    assert result[-1] == "singing some recognizable lyrics"
    assert "unverified" in result[0]


def test_recording_add_without_confirmation_does_not_import():
    import gradio as gr
    import pytest

    from app import add_confirmed_recording_result

    with pytest.raises(gr.Error, match="Choose and confirm"):
        add_confirmed_recording_result(
            None, [], SongIndex(),
            progress=lambda *args, **kwargs: None,
        )


def test_melody_is_third_metric_not_standalone_link_action():
    import app

    source = Path(app.__file__).read_text(encoding="utf-8")
    assert "Melody similarity · experimental" in source
    assert "Analyze this YouTube melody" not in source
    assert "Analyze this audio as a melody reference" not in source
    assert "Find matching melodies" not in source
    assert "Find a song by singing · experimental" in source
    assert "Confirm and add to catalog" in source
    assert "Compare recording with catalog (lyrics + melody)" not in source
    assert "Local top results" not in source
    assert 'sources=["microphone"]' in source


def test_audio_melody_failure_does_not_break_lyric_index(tmp_path, monkeypatch):
    import app

    audio = tmp_path / "song.wav"
    audio.write_bytes(b"placeholder")
    monkeypatch.setattr(app, "audio_metadata", lambda _: ("Song", "Singer"))
    monkeypatch.setattr(app, "read_audio", lambda _: b"audio")
    monkeypatch.setattr(
        app, "extract_chroma",
        lambda *_: (_ for _ in ()).throw(app.MelodyError("No clear pitch")),
    )
    monkeypatch.setattr(
        app, "find_lyrics",
        lambda *args: LyricsMatch(
            "Song", "Singer", "A short example lyric", "https://lrclib.net/api/get/1"
        ),
    )
    monkeypatch.setattr(app, "embed", lambda text: np.array([1.0, 0.0]))
    status, rows, catalog, _view = app.add_upload(
        str(audio), "", "", SongIndex(), progress=lambda *args, **kwargs: None
    )
    assert "Melody extraction was unavailable" in status
    assert catalog.indexed_count == 1
    assert catalog.melody_count == 0
    assert rows[0][-1] == "Reference lyrics (LRCLIB)"
