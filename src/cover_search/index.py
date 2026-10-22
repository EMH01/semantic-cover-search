import faiss
import numpy as np

from .comparison import LyricComparison, compare_all_lyrics
from .models import Song


class SongIndex:
    """Small in-memory FAISS index using cosine similarity through normalized inner product."""

    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP | None = None
        self._songs: list[Song] = []
        self._indexed_songs: list[Song] = []
        self._vectors: list[np.ndarray] = []

    @property
    def songs(self) -> tuple[Song, ...]:
        return tuple(self._songs)

    @property
    def indexed_count(self) -> int:
        return len(self._indexed_songs)

    def add(self, song: Song, embedding: np.ndarray | None = None) -> None:
        """Save metadata even when no reliable lyrics embedding is available."""
        identity = (song.title.casefold(), song.artist.casefold())
        if any((item.title.casefold(), item.artist.casefold()) == identity for item in self._songs):
            raise ValueError(f"Song already in catalog: {song.label}")

        if embedding is not None:
            vector = self._prepare_vector(embedding)
            if self._index is None:
                self._index = faiss.IndexFlatIP(vector.shape[1])
            elif self._index.d != vector.shape[1]:
                raise ValueError("Embedding dimension does not match the existing index.")
            self._index.add(vector)
            self._indexed_songs.append(song)
            self._vectors.append(vector.reshape(-1).copy())
        self._songs.append(song)

    def search(self, embedding: np.ndarray, top_k: int = 5) -> list[tuple[Song, float]]:
        if self._index is None or not self._indexed_songs:
            return []
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        vector = self._prepare_vector(embedding)
        k = min(top_k, len(self._indexed_songs))
        scores, indices = self._index.search(vector, k)
        return [
            (self._indexed_songs[index], float(score))
            for index, score in zip(indices[0], scores[0], strict=True)
            if index >= 0
        ]

    def compare_all(self) -> list[LyricComparison]:
        """Rank all unique pairs that have text embeddings, excluding metadata-only."""
        return compare_all_lyrics(self._indexed_songs, self._vectors)

    @staticmethod
    def _prepare_vector(embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] == 0 or not np.all(np.isfinite(vector)):
            raise ValueError("Embedding must contain finite values and at least one dimension.")
        if float(np.linalg.norm(vector)) <= 0:
            raise ValueError("Embedding must not be a zero vector.")
        faiss.normalize_L2(vector)
        return vector
