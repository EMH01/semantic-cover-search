"""Read optional embedded song metadata from local audio files."""
from pathlib import Path

from mutagen import File as MutagenFile


def audio_metadata(path: str | None) -> tuple[str, str]:
    """Return an editable title and artist, falling back to the filename."""
    if not path:
        return "", ""
    source = Path(path)
    title = source.stem.strip() or "Untitled recording"
    artist = ""
    try:
        audio = MutagenFile(source, easy=True)
        if audio is not None and audio.tags:
            titles = audio.tags.get("title", [])
            artists = audio.tags.get("artist", [])
            if titles and str(titles[0]).strip():
                title = str(titles[0]).strip()
            if artists and str(artists[0]).strip():
                artist = str(artists[0]).strip()
    except (OSError, ValueError, TypeError, KeyError):
        # Metadata is not required to index a valid audio file.
        pass
    return title, artist
