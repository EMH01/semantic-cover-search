import faiss
import numpy as np

from .models import Song


class SongIndex:
    """Small in-memory FAISS index using cosine similarity through normalized inner product."""

    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP | None = None
        self._songs: list[Song] = []

    @property
    def songs(self) -> tuple[Song, ...]:
        return tuple(self._songs)

    def add(self, song: Song, embedding: np.ndarray) -> None:
        vector = self._prepare_vector(embedding)
        if self._index is None:
            self._index = faiss.IndexFlatIP(vector.shape[1])
        elif self._index.d != vector.shape[1]:
            raise ValueError("Embedding dimension does not match the existing index.")

        if any(existing.label == song.label for existing in self._songs):
            raise ValueError(f"Song already indexed: {song.label}")

        self._index.add(vector)
        self._songs.append(song)

    def search(self, embedding: np.ndarray, top_k: int = 5) -> list[tuple[Song, float]]:
        if self._index is None or not self._songs:
            return []
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        vector = self._prepare_vector(embedding)
        k = min(top_k, len(self._songs))
        scores, indices = self._index.search(vector, k)
        return [
            (self._songs[index], float(score))
            for index, score in zip(indices[0], scores[0], strict=True)
            if index >= 0
        ]

    @staticmethod
    def _prepare_vector(embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vector)
        return vector
