from types import SimpleNamespace

import numpy as np

from cover_search.services import embed_text, transcribe_audio


class FakeTranscriptions:
    def create(self, **kwargs):
        assert kwargs["model"] == "test-transcriber"
        return SimpleNamespace(text="  hello   world  ")


class FakeEmbeddings:
    def create(self, **kwargs):
        assert kwargs["model"] == "test-embedding"
        return SimpleNamespace(data=[SimpleNamespace(embedding=[3.0, 4.0])])


def test_transcription_is_normalized():
    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=FakeTranscriptions()))
    assert transcribe_audio(client, b"audio", "song.mp3", "test-transcriber") == "hello world"


def test_embedding_is_l2_normalized():
    client = SimpleNamespace(embeddings=FakeEmbeddings())
    vector = embed_text(client, "lyrics", "test-embedding")
    assert np.isclose(np.linalg.norm(vector), 1.0)
