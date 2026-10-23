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

    assert "1 pairs compared" in status
    assert "1 metadata-only song excluded" in status
    assert "Hallelujah" in html
    assert "Text: LRCLIB" in html
    assert "Text: ASR" in html
    assert "100.0%" in html
    assert "scs-match-grid" in html
    assert "scs-meter" in html
    assert "not proof" in html


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


def test_comparison_view_requires_two_searchable_songs():
    from app import compare_catalog_lyrics

    catalog = SongIndex()
    catalog.add(Song("Only metadata", "A", "", text_origin="none"))
    status, html = compare_catalog_lyrics(catalog, 10)
    assert "Add at least two songs" in status
    assert "No song pairs" in html
