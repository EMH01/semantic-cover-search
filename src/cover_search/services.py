from typing import Any

import numpy as np


def transcribe_audio(client: Any, audio_bytes: bytes, filename: str, model: str) -> str:
    response = client.audio.transcriptions.create(
        model=model,
        file=(filename, audio_bytes),
    )
    text = response if isinstance(response, str) else response.text
    transcript = " ".join(str(text).split())
    if not transcript:
        raise ValueError("The transcription service returned an empty transcript.")
    return transcript


def embed_text(client: Any, text: str, model: str) -> np.ndarray:
    response = client.embeddings.create(model=model, input=[text])
    vector = np.asarray(response.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("The embedding service returned a zero vector.")
    return vector / norm
