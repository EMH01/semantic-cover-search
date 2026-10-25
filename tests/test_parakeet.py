from types import SimpleNamespace

import numpy as np
import pytest

from cover_search import parakeet


def test_parakeet_transcribes_decoded_pcm(monkeypatch):
    class FakeStream:
        result = SimpleNamespace(text="  hello   world  ")

        def accept_waveform(self, sample_rate, pcm):
            assert sample_rate == 16000
            assert np.array_equal(pcm, np.array([0.1, -0.1], dtype=np.float32))

    stream = FakeStream()

    class FakeRecognizer:
        def create_stream(self):
            return stream

        def decode_stream(self, received):
            assert received is stream

    monkeypatch.setattr(parakeet, "decode_audio", lambda _: np.array([0.1, -0.1], dtype=np.float32))
    monkeypatch.setattr(parakeet, "_recognizer", lambda: FakeRecognizer())
    assert parakeet.transcribe_parakeet(b"audio", "track.wav") == "hello world"


def test_parakeet_rejects_empty_audio():
    with pytest.raises(ValueError, match="empty"):
        parakeet.transcribe_parakeet(b"")


def test_parakeet_rejects_empty_transcript(monkeypatch):
    class FakeRecognizer:
        def create_stream(self):
            return SimpleNamespace(
                accept_waveform=lambda *args: None, result=SimpleNamespace(text=" ")
            )

        def decode_stream(self, stream):
            pass

    monkeypatch.setattr(parakeet, "decode_audio", lambda _: np.ones(4, dtype=np.float32))
    monkeypatch.setattr(parakeet, "_recognizer", lambda: FakeRecognizer())
    with pytest.raises(ValueError, match="empty transcript") as exc_info:
        parakeet.transcribe_parakeet(b"audio")
    assert "clearer spoken words" in str(exc_info.value)


def test_app_transcription_always_delegates_to_parakeet(monkeypatch):
    import app

    calls = []

    def fake_parakeet(audio, filename):
        calls.append((audio, filename))
        return "recognized words"

    monkeypatch.setattr(app, "transcribe_parakeet", fake_parakeet)
    assert app.transcribe(b"audio", "song.wav") == "recognized words"
    assert calls == [(b"audio", "song.wav")]


def test_settings_do_not_expose_transcription_backend_switch():
    from cover_search.config import Settings

    settings = Settings()
    assert not hasattr(settings, "asr_engine")
    assert not hasattr(settings, "local_whisper_model")
