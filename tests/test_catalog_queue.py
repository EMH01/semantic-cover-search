"""Catalog imports must never cancel or overwrite each other through parallel jobs."""


def test_all_catalog_writers_share_one_serial_queue():
    import app

    handlers = (
        app.add_upload,
        app.add_youtube,
        app.add_selected_video,
        app.add_confirmed_recording_result,
    )
    for handler in handlers:
        matched = [event for event in app.demo.fns.values() if event.fn is handler]
        assert len(matched) == 1, f"Expected one registered event for {handler.__name__}"
        assert matched[0].concurrency_id == app.CATALOG_QUEUE_ID
        assert matched[0].concurrency_limit == 1


def test_catalog_comparison_waits_for_imports_to_complete():
    import app

    for handler in (app.compare_catalog_lyrics, app.search_recording_on_youtube):
        matched = [event for event in app.demo.fns.values() if event.fn is handler]
        assert len(matched) == 1
        assert matched[0].concurrency_id == app.CATALOG_QUEUE_ID
        assert matched[0].concurrency_limit == 1

    assert app.demo._queue is not None


def test_public_video_discovery_is_not_blocked_by_catalog_imports():
    import app

    matched = [
        event for event in app.demo.fns.values()
        if event.fn is app.search_public_versions
    ]
    assert len(matched) == 1
    assert matched[0].concurrency_id != app.CATALOG_QUEUE_ID
