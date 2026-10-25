from typing import Any

import numpy as np


def embed_text(client: Any, text: str, model: str) -> np.ndarray:
    response = client.embeddings.create(model=model, input=[text])
    vector = np.asarray(response.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if vector.size == 0 or not np.all(np.isfinite(vector)) or not np.isfinite(norm) or norm == 0:
        raise ValueError("The embedding service returned an invalid vector.")
    return vector / norm
