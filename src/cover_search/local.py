"""Free, on-device transcription and multilingual text embeddings.

The local backends are loaded lazily and reused. No inference API token is
required. Initial model downloads require internet access on the host.
"""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def embed_local(
    text: str,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> np.ndarray:
    if not text.strip():
        raise ValueError("Cannot embed an empty transcript.")
    vector = np.asarray(_embedding_model(model_name).encode(text), dtype=np.float32)
    norm = np.linalg.norm(vector)
    if not vector.size or not np.isfinite(norm) or norm <= 0 or not np.all(np.isfinite(vector)):
        raise ValueError("Local embedding model returned an invalid vector.")
    return vector / norm
