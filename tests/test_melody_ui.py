"""Regression tests for automatic melody comparison without extra UI controls."""

from pathlib import Path

import numpy as np

from cover_search.index import SongIndex
from cover_search.lyrics import LyricsMatch
from cover_search.models import Song


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


def test_youtube_link_never_adds_melody_without_audio(monkeypatch):
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
        lambda *_: (_ for _ in ()).throw(AssertionError("No audio download")),
    )
    catalog = SongIndex()
    _status, rows, _, _ = app.add_youtube(
        "https://www.youtube.com/watch?v=abcd",
        catalog, progress=lambda *args, **kwargs: None,
    )
    assert catalog.melody_count == 0
    assert rows[0][3] == "Not analyzed"


def test_mic_local_comparison_uses_melody_when_asr_fails(monkeypatch):
    import app

    catalog = SongIndex()
    song = Song("Hallelujah", "Singer", "", text_origin="none")
    catalog.add(song)
    catalog.index_melody(song, _notes())
    monkeypatch.setattr(app, "read_audio", lambda path: b"microphone")
    monkeypatch.setattr(app, "extract_chroma", lambda *args, **kwargs: _notes())
    monkeypatch.setattr(
        app, "transcribe",
        lambda *args: (_ for _ in ()).throw(AssertionError("Should not run ASR")),
    )
    text_rows, melody_rows, note = app.search_upload("sample.wav", 5, catalog)
    assert text_rows == []
    assert melody_rows[0][1] == "Hallelujah"
    assert melody_rows[0][3] == "100.0"
    assert "experimental" in note


def test_melody_is_third_metric_not_standalone_link_action():
    import app

    source = Path(app.__file__).read_text(encoding="utf-8")
    assert "Melody similarity · experimental" in source
    assert "Analyze this YouTube melody" not in source
    assert "Analyze this audio as a melody reference" not in source
    assert "Find matching melodies" not in source
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
