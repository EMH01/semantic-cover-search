from types import SimpleNamespace

import numpy as np
import pytest

from cover_search import local


def test_local_embedding_is_normalized(monkeypatch):
    class FakeModel:
        def encode(self, text):
            assert text == "a similar lyric"
            return [3, 4]
    monkeypatch.setattr(local, "_embedding_model", lambda _: FakeModel())
    vector = local.embed_local("a similar lyric")
    assert np.isclose(np.linalg.norm(vector), 1)
    assert np.allclose(vector, [0.6, 0.8])


def test_empty_transcript_is_not_embedded():
    with pytest.raises(ValueError, match="empty"):
        local.embed_local("  ")


def test_local_transcription_uses_cpu_model_and_joins_segments(monkeypatch):
    class FakeWhisper:
        def transcribe(self, path, **kwargs):
            assert path.endswith(".wav")
            return iter([SimpleNamespace(text=" hello "), SimpleNamespace(text=" world ")]), None
    monkeypatch.setattr(local, "_whisper_model", lambda _: FakeWhisper())
    assert local.transcribe_local(b"synthetic fixture", "demo.wav") == "hello world"


def test_unsupported_audio_extension_is_rejected_before_model_load():
    with pytest.raises(ValueError, match="Unsupported"):
        local.transcribe_local(b"content", "demo.exe")
