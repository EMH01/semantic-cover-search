"""Gradio melody flow tests without invoking an API or decoding an MP3."""

from pathlib import Path

import numpy as np

from cover_search.index import SongIndex


def _notes():
    notes = [0, 2, 4, 5, 7, 9, 7, 5, 4, 2, 0, 11]
    features = np.zeros((len(notes), 12), dtype=np.float32)
    features[np.arange(len(notes)), notes] = 1
    return features


def test_youtube_melody_index_does_not_require_lyrics_or_asr(monkeypatch):
    import app

    catalog = SongIndex()
    monkeypatch.setattr(
        app, "video_metadata",
        lambda url: {
            "title": "Hallelujah", "channel": "Test artist",
            "url": "https://www.youtube.com/watch?v=abcd",
        },
    )
    monkeypatch.setattr(
        app, "download_audio",
        lambda url: type("Audio", (), {"data": b"audio"})(),
    )
    monkeypatch.setattr(app, "extract_chroma", lambda data: _notes())
    monkeypatch.setattr(
        app, "find_lyrics",
        lambda *args: (_ for _ in ()).throw(AssertionError("Do not query lyrics")),
    )

    status, rows, saved = app.index_youtube_melody(
        "https://www.youtube.com/watch?v=abcd",
        catalog, progress=lambda *args, **kwargs: None,
    )
    assert "Melody ready" in status
    assert len(rows) == 1
    assert saved.melody_count == 1
    assert saved.indexed_count == 0


def test_recorded_melody_returns_matches_without_transcription(monkeypatch):
    import app

    catalog = SongIndex()
    song = app.Song("Hallelujah", "Singer", "", text_origin="none")
    catalog.add(song)
    catalog.index_melody(song, _notes())
    monkeypatch.setattr(app, "read_audio", lambda path: b"recording")
    monkeypatch.setattr(app, "extract_chroma", lambda *args, **kwargs: _notes())
    monkeypatch.setattr(
        app, "transcribe",
        lambda *args: (_ for _ in ()).throw(AssertionError("Do not run ASR")),
    )
    note, rows = app.search_recorded_melody(
        "sample.wav", 5, catalog, progress=lambda *args, **kwargs: None
    )
    assert "not a probability" in note
    assert rows[0][1] == "Hallelujah"
    assert rows[0][3] == "100.0"


def test_recording_controls_are_located_in_local_catalog_only():
    import app

    source = Path(app.__file__).read_text(encoding="utf-8")
    first_tab = source.index('with gr.Tab("Discover covers on YouTube"):')
    second_tab = source.index('with gr.Tab("Local catalog"):')
    recording = source.index("Search by singing or humming · experimental")
    assert first_tab < second_tab < recording
    assert 'sources=["microphone"]' in source[second_tab:]
    assert 'with gr.Tab("Experimental singing search"):' not in source
