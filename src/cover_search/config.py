import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True, slots=True)
class Settings:
    provider: str = os.getenv("AI_PROVIDER", "local")
    local_whisper_model: str = os.getenv("LOCAL_WHISPER_MODEL", "base")
    local_embedding_model: str = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    transcription_model: str = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-transcribe")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
