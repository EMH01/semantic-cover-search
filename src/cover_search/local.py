"""Free, on-device transcription and multilingual text embeddings.

The local backends are loaded lazily and reused. No inference API token is
required. Initial model downloads require internet access on the host.
"""
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


@lru_cache(maxsize=1)
def _whisper_model(model_name: str):
    from faster_whisper import WhisperModel
    return WhisperModel(model_name, device="cpu", compute_type="int8")


def transcribe_local(audio_bytes: bytes, filename: str, model_name: str = "base") -> str:
    if not audio_bytes:
        raise ValueError("Audio input is empty.")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".webm", ".mp4"}:
        raise ValueError("Unsupported audio format.")
    with TemporaryDirectory() as directory:
        path = Path(directory) / ("audio" + suffix)
        path.write_bytes(audio_bytes)
        segments, _ = _whisper_model(model_name).transcribe(
            str(path), beam_size=2, vad_filter=True
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    if not text:
        raise ValueError("No speech or lyrics could be transcribed from this audio.")
    return text


@lru_cache(maxsize=1)
def _embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device="cpu")


def embed_local(text: str, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> np.ndarray:
    if not text.strip():
        raise ValueError("Cannot embed an empty transcript.")
    vector = np.asarray(_embedding_model(model_name).encode(text), dtype=np.float32)
    norm = np.linalg.norm(vector)
    if not vector.size or not np.isfinite(norm) or norm <= 0 or not np.all(np.isfinite(vector)):
        raise ValueError("Local embedding model returned an invalid vector.")
    return vector / norm
