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
