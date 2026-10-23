import faiss
import numpy as np

from .comparison import LyricComparison, compare_all_lyrics
from .melody import MelodyError, melody_match
from .models import Song


class SongIndex:
    """Small in-memory FAISS index using cosine similarity through normalized inner product."""

    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP | None = None
        self._songs: list[Song] = []
        self._indexed_songs: list[Song] = []
        self._vectors: list[np.ndarray] = []
        self._melodies: dict[Song, np.ndarray] = {}

    @property
    def songs(self) -> tuple[Song, ...]:
        return tuple(self._songs)

    @property
    def indexed_count(self) -> int:
        return len(self._indexed_songs)

    @property
    def searchable_songs(self) -> tuple[Song, ...]:
        return tuple(self._indexed_songs)

    @property
    def melody_songs(self) -> tuple[Song, ...]:
        return tuple(self._melodies)

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
        return compare_all_lyrics(self._indexed_songs, self._vectors, self._melodies)

    @property
    def melody_count(self) -> int:
        return len(self._melodies)

    def has_melody(self, song: Song) -> bool:
        return song in self._melodies

    def index_melody(self, song: Song, chroma: np.ndarray) -> None:
        """Keep short pitch-class references only in this user's catalog session."""
        if song not in self._songs:
            raise ValueError("Add this song to the local catalog first.")
        if song not in self._melodies and len(self._melodies) >= 12:
            raise ValueError("The experimental catalog supports up to 12 melody references.")
        features = np.asarray(chroma, dtype=np.float32)
        if (
            features.ndim != 2
            or features.shape[1] != 12
            or features.shape[0] < 7
            or not np.all(np.isfinite(features))
        ):
            raise MelodyError("Melody reference must contain valid pitch-class frames.")
        self._melodies[song] = features.copy()

    def search_melody(
        self, query: np.ndarray, top_k: int = 5
    ) -> list[tuple[Song, float, float, int]]:
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        scores = []
        for song, reference in self._melodies.items():
            try:
                score, offset, shift = melody_match(query, reference)
                scores.append((song, score, offset, shift))
            except MelodyError:
                continue
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    @staticmethod
    def _prepare_vector(embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] == 0 or not np.all(np.isfinite(vector)):
            raise ValueError("Embedding must contain finite values and at least one dimension.")
        if float(np.linalg.norm(vector)) <= 0:
            raise ValueError("Embedding must not be a zero vector.")
        faiss.normalize_L2(vector)
        return vector
