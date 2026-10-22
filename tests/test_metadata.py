from pathlib import Path
from unittest.mock import patch

from cover_search.metadata import audio_metadata


def test_blank_file_clears_metadata():
    assert audio_metadata(None) == ("", "")


def test_embedded_metadata_is_read(tmp_path):
    path = tmp_path / "filename.mp3"
    path.write_bytes(b"fixture")
    fake = type("Audio", (), {"tags": {"title": ["Real title"], "artist": ["Real artist"]}})()
    with patch("cover_search.metadata.MutagenFile", return_value=fake):
        assert audio_metadata(str(path)) == ("Real title", "Real artist")


def test_missing_tags_fall_back_to_filename(tmp_path):
    path = tmp_path / "My Song.mp3"
    path.write_bytes(b"fixture")
    with patch("cover_search.metadata.MutagenFile", return_value=None):
        assert audio_metadata(str(path)) == ("My Song", "")


def test_invalid_audio_metadata_does_not_block_upload(tmp_path):
    path = tmp_path / "Recording.wav"
    path.write_bytes(b"fixture")
    with patch("cover_search.metadata.MutagenFile", side_effect=ValueError("bad metadata")):
        assert audio_metadata(str(path)) == ("Recording", "")
