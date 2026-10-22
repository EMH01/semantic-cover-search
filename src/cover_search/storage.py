"""Durable local song catalog backed by SQLite.

Catalogs are shared by all sessions connected to the same local app instance.
Store transcripts and embedding vectors, not original audio files.
"""
import os
from contextlib import closing
import sqlite3
from pathlib import Path

import numpy as np

from .index import SongIndex
from .models import Song

DEFAULT_DB = Path(__file__).resolve().parents[2] / ".local_data" / "catalog.sqlite3"


def database_path() -> Path:
    return Path(os.environ.get("COVER_CATALOG_DB", str(DEFAULT_DB))).expanduser()


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=20)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS songs (
            identity TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            artist TEXT NOT NULL,
            transcript TEXT NOT NULL,
            source TEXT NOT NULL,
            embedding BLOB NOT NULL,
            dimensions INTEGER NOT NULL
        )"""
    )
    return conn


def _identity(song: Song) -> str:
    return song.title.strip().casefold() + "\u0000" + song.artist.strip().casefold()


def store_song(song: Song, embedding: np.ndarray, path: Path | None = None) -> None:
    """Commit a record atomically. Reject duplicates and malformed embeddings."""
    vec = SongIndex._prepare_vector(embedding).reshape(-1)
    target = path if path is not None else database_path()
    with closing(_connect(target)) as conn, conn:
        try:
            conn.execute(
                "INSERT INTO songs VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    _identity(song), song.title, song.artist, song.transcript,
                    song.source, vec.tobytes(), vec.size,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Song already indexed: {song.label}") from exc


def load_catalog(path: Path | None = None) -> SongIndex:
    """Recreate FAISS from saved vectors when the app/session starts."""
    target = path if path is not None else database_path()
    index = SongIndex()
    with closing(_connect(target)) as conn:
        rows = conn.execute(
            "SELECT title, artist, transcript, source, embedding, dimensions "
            "FROM songs ORDER BY rowid"
        ).fetchall()
    for title, artist, transcript, source, vector_data, dimensions in rows:
        vector = np.frombuffer(vector_data, dtype=np.float32)
        if vector.size != dimensions:
            raise ValueError("Saved catalog contains an invalid embedding.")
        index.add(Song(title, artist, transcript, source), vector)
    return index
