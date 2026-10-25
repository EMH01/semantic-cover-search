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
    assert "ASR_ENGINE" not in str(exc_info.value)
    assert "clearer spoken words" in str(exc_info.value)
