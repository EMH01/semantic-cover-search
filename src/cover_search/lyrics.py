"""Best-effort song-lyrics lookup from public LRCLIB metadata.

Lyrics are fetched on demand and kept only in the current app session.
Copyright in underlying songs remains with the rights holders.
"""
import re
import unicodedata
from dataclasses import dataclass

import httpx

API = "https://lrclib.net/api/search"
HEADERS = {"User-Agent": "SemanticCoverSearch/0.1 (https://github.com/EMH01/semantic-cover-search)"}


@dataclass(frozen=True)
class LyricsMatch:
    title: str
    reference_artist: str
    text: str
    url: str


def normalized(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", text))


def title_from_video(video_title: str) -> str:
    """Extract a candidate song title, never claiming verified identification."""
    title = video_title.strip()
    parts = re.split(r"\s+[-–—]\s+", title, maxsplit=1)
    if len(parts) == 2:
        title = parts[1]
    title = re.sub(r"\s*[\[(].*?[\])]", "", title).strip()
    title = re.sub(\n        r"\\b(official music video|lyric video|cover version|acoustic cover|cover)\\b.*$",\n        "", title, flags=re.I,\n    ).strip(" -–—")
    return title or video_title.strip()


def find_lyrics(title: str, artist: str = "") -> LyricsMatch | None:
    """Return only an exact title match; prefer matching the supplied artist.

    With a missing artist, choose the first exact-title LRCLIB result.
    This is a *candidate* rendition: verses may differ between cover versions.
    """
    query = title.strip()
    if len(query) < 2 or len(query) > 150:
        return None
    with httpx.Client(timeout=12, headers=HEADERS) as client:
        response = client.get(API, params={"track_name": query})
        response.raise_for_status()
        records = response.json()
    if not isinstance(records, list):
        return None
    candidates = [
        item for item in records
        if isinstance(item, dict)
        and normalized(str(item.get("trackName") or "")) == normalized(query)
        and isinstance(item.get("plainLyrics"), str)
        and item["plainLyrics"].strip()
    ]
    if not candidates:
        return None
    if artist.strip():
        matching = [
            item for item in candidates
            if normalized(str(item.get("artistName") or "")) == normalized(artist)
        ]
        if matching:
            candidates = matching
    item = candidates[0]
    return LyricsMatch(
        title=str(item.get("trackName") or query),
        reference_artist=str(item.get("artistName") or "Unknown reference artist"),
        text=item["plainLyrics"].strip(),
        url=f"https://lrclib.net/api/get/{item['id']}",
    )
