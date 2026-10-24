"""Gradio comparison view tests that do not invoke ASR, embeddings, or network."""
import numpy as np

from cover_search.index import SongIndex
from cover_search.models import Song


def test_comparison_view_contains_every_pair_count_and_source_labels():
    from app import compare_catalog_lyrics

    catalog = SongIndex()
    catalog.add(
        Song("Hallelujah", "Leonard Cohen", "first second third", text_origin="lrclib"),
        np.array([1.0, 0.0]),
    )
    catalog.add(
        Song("Hallelujah Cover", "Artist", "first second third", text_origin="asr"),
        np.array([1.0, 0.0]),
    )
    catalog.add(Song("No lyrics", "Nobody", "", text_origin="none"))
    status, html = compare_catalog_lyrics(catalog, 10)

    assert "1 pair compared" in status
    assert "1 metadata-only song excluded" in status
    assert "Hallelujah" in html
    assert "Text: LRCLIB" in html
    assert "Text: ASR" in html
    assert "100.0%" in html
    assert "scs-match-grid" in html
    assert "scs-meter" in html
    assert "not proof" in html or "not prove" in html


def test_comparison_view_safely_escapes_user_supplied_names():
    from app import compare_catalog_lyrics

    catalog = SongIndex()
    catalog.add(Song("<script>alert(1)</script>", "A", "one two three"),
                np.array([1.0, 0.0]))
    catalog.add(Song("Safe", "B", "another verse entirely"),
                np.array([0.0, 1.0]))
    _, html = compare_catalog_lyrics(catalog, 5)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_comparison_view_requires_two_comparable_songs():
    from app import compare_catalog_lyrics

    catalog = SongIndex()
    catalog.add(Song("Only metadata", "A", "", text_origin="none"))
    status, html = compare_catalog_lyrics(catalog, 10)
    assert "Add at least two comparable songs" in status
    assert "No comparable song pairs" in html


def test_melody_metric_is_present_and_shows_na_for_links():
    from app import compare_catalog_lyrics

    catalog = SongIndex()
    catalog.add(
        Song("One", "Singer", "Some lyric words", source="youtube"),
        np.array([1.0, 0.0]),
    )
    catalog.add(
        Song("Two", "Singer", "Some lyric words", source="youtube"),
        np.array([1.0, 0.0]),
    )
    _status, html = compare_catalog_lyrics(catalog, 10)
    assert "Melody similarity" in html
    assert "N/A" in html


def test_audio_only_songs_can_be_compared_without_lyrics():
    from app import compare_catalog_lyrics

    catalog = SongIndex()
    first = Song("Song A", "A", "", text_origin="none")
    second = Song("Song B", "B", "", text_origin="none")
    catalog.add(first)
    catalog.add(second)
    chroma = np.zeros((12, 12), dtype=np.float32)
    chroma[np.arange(12), np.arange(12)] = 1
    catalog.index_melody(first, chroma)
    catalog.index_melody(second, chroma)
    status, html = compare_catalog_lyrics(catalog, 10)
    assert "1 pair compared" in status
    assert "Melody similarity" in html
    assert html.count("N/A") >= 2
    assert "100.0" in html
