"""Presentation regressions: metadata is escaped and card layout is responsive."""


def test_youtube_cards_have_images_and_safe_links():
    from app import render_video_cards

    html = render_video_cards([
        {
            "title": "<script>Hallelujah</script>",
            "channel": "Test & channel",
            "url": "https://www.youtube.com/watch?v=QTAyVS3uYq8",
        },
    ])
    assert "scs-video-grid" in html
    assert "scs-video-thumb" in html
    assert "i.ytimg.com/vi/QTAyVS3uYq8/mqdefault.jpg" in html
    assert "&lt;script&gt;" in html
    assert "<script>" not in html
    assert "Test &amp; channel" in html
    assert 'rel="noopener noreferrer"' in html


def test_youtube_cards_show_safe_fallback_for_nonstandard_video_id():
    from app import render_video_cards

    html = render_video_cards([
        {"title": "Song", "channel": "Channel",
         "url": "https://www.youtube.com/watch?v=not11"},
    ])
    assert "scs-video-thumb" in html
    assert "<img " not in html


def test_visual_theme_contains_mobile_layout_and_keyboard_focus():
    from cover_search.ui_theme import APP_CSS, HERO_HTML

    assert "@media (max-width: 600px)" in APP_CSS
    assert "focus-visible" in APP_CSS
    assert "scs-video-grid" in APP_CSS
    assert "Semantic Cover Search" in HERO_HTML


def test_comparison_metric_label_is_understandable_without_changing_formula():
    import numpy as np

    from app import render_lyric_comparison_cards
    from cover_search.index import SongIndex
    from cover_search.models import Song

    index = SongIndex()
    index.add(Song("One", "Artist A", "some song lyrics"), np.array([1.0, 0.0]))
    index.add(Song("Two", "Artist B", "other music lyrics"), np.array([0.6, 0.8]))
    html = render_lyric_comparison_cards(index.compare_all())
    assert "Semantic similarity" in html
    assert "Not a probability" in html
    assert "60.0" in html  # cosine(1,0 and 0.6,0.8) * 100


def test_status_area_reserves_loading_space():
    from cover_search.ui_theme import APP_CSS

    assert ".gradio-container .scs-status" in APP_CSS
    assert "min-height: 52px" in APP_CSS
