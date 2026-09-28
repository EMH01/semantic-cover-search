from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Song:
    title: str
    artist: str
    transcript: str
    source: str = "upload"

    @property
    def label(self) -> str:
        return f"{self.title} — {self.artist}"
