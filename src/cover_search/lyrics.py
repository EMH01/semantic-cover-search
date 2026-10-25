"""Look up reference lyrics using LRCLIB without downloading song audio.

Lyrics are reference text and may differ from a cover's performed verses.
Rights to song lyrics remain with their respective copyright holders.
"""

import re
import time
import unicodedata
from dataclasses import dataclass
from html import escape

import httpx

API = "https://lrclib.net/api/search"
HEADERS = {
    "User-Agent": "SemanticCoverSearch/0.1 (https://github.com/EMH01/semantic-cover-search)"
}


class LyricsLookupError(RuntimeError):
    """The provider failed; distinct from a successful search without lyrics."""


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
    """Extract a tentative title from common 'Artist - Track (Video)' patterns."""
    title = video_title.strip()
    parts = re.split(r"\s+[-–—]\s+", title, maxsplit=1)
    if len(parts) == 2:
        title = parts[1]
    title = re.sub(r"\s*[\[(].*?[\])]", "", title).strip()
    title = re.sub(
        r"\b(official music video|lyric video|cover version|acoustic cover|cover)\b.*$",
        "",
        title,
        flags=re.I,
    ).strip(" -–—")
    return title or video_title.strip()


def lyric_title_for_result(video_title: str, searched_title: str = "") -> str:
    """Infer title cautiously using both discovery query and video title.

    'Leonard Cohen Hallelujah' plus 'Leonard Cohen - Hallelujah' -> 'Hallelujah'.
    'Hallelujah' plus 'Hallelujah - Lucy Thomas' -> 'Hallelujah'.
    """
    searched = normalized(searched_title)
    candidate = title_from_video(video_title)
    cleaned = normalized(candidate)
    if searched and cleaned and f" {cleaned} " in f" {searched} ":
        return candidate
    if searched and f" {searched} " in f" {normalized(video_title)} ":
        return searched_title.strip()
    return candidate


def _get_results(client: httpx.Client, params: dict[str, str]) -> list[dict]:
    """Bounded retry only on LRCLIB 503; observe rate limit and other errors."""
    for attempt in range(2):
        try:
            response = client.get(API, params=params)
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, list):
                raise LyricsLookupError("LRCLIB returned an unexpected response.")
            return [r for r in result if isinstance(r, dict)]
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code == 503 and attempt == 0:
                delay = exc.response.headers.get("Retry-After", "1")
                try:
                    time.sleep(max(0.5, min(float(delay), 2.0)))
                except ValueError:
                    time.sleep(1.0)
                continue
            if code == 429:
                raise LyricsLookupError(
                    "LRCLIB is rate-limiting requests; retry later."
                ) from exc
            raise LyricsLookupError(
                f"LRCLIB returned HTTP {code}; try again later."
            ) from exc
        except httpx.RequestError as exc:
            if attempt == 0:
                time.sleep(0.6)
                continue
            raise LyricsLookupError(
                "LRCLIB could not be reached after a retry."
            ) from exc
        except ValueError as exc:
            raise LyricsLookupError("LRCLIB returned invalid JSON.") from exc
    return []


def _plain_text(record: dict) -> str:
    plain = record.get("plainLyrics")
    if isinstance(plain, str) and plain.strip():
        return plain.strip()
    synced = record.get("syncedLyrics")
    if not isinstance(synced, str):
        return ""
    # Fallback for records with timed lyrics only.
    lines = [re.sub(r"^(?:\[\d{1,2}:\d{2}(?:\.\d+)?\]\s*)+", "", line).strip()
             for line in synced.splitlines()]
    return "\n".join(line for line in lines if line)


def _matching_record(records: list[dict], title: str, artist: str) -> dict | None:
    candidates = [
        record for record in records
        if normalized(str(record.get("trackName") or "")) == normalized(title)
        and _plain_text(record)
        and not record.get("instrumental", False)
    ]
    if artist.strip():
        candidates = [
            record for record in candidates
            if normalized(str(record.get("artistName") or "")) == normalized(artist)
        ]
    return candidates[0] if candidates else None


def find_lyrics(title: str, artist: str = "") -> LyricsMatch | None:
    """Look up an exact title, then fall back to a broad query if necessary.

    No artist supplied means the first exact-title match can be another version.
    Do not assume that reference lyrics are the exact words of a YouTube cover.
    """
    title = title.strip()
    artist = artist.strip()
    if not 2 <= len(title) <= 150:
        return None
    with httpx.Client(timeout=12, headers=HEADERS) as client:
        params = {"track_name": title}
        if artist:
            params["artist_name"] = artist
        records = _get_results(client, params)
        record = _matching_record(records, title, artist)
        if record is None:
            # Limit to two sequential requests. LRCLIB recommends a short pause.
            time.sleep(0.35)
            records = _get_results(client, {"q": f"{title} {artist}".strip()})
            record = _matching_record(records, title, artist)
    if record is None:
        return None
    return LyricsMatch(
        title=str(record.get("trackName") or title),
        reference_artist=str(record.get("artistName") or "Unknown reference artist"),
        text=_plain_text(record),
        url=f"https://lrclib.net/api/get/{record['id']}",
    )


def render_lyrics(match: LyricsMatch | None) -> str:
    """Readable, source-attributed lyric text with HTML-escaped content."""
    if match is None:
        return (
            '<p style="padding:14px;opacity:.72;">No reference lyrics found for the '
            'latest selection. The entry may remain metadata-only.</p>'
        )
    title = escape(match.title)
    artist = escape(match.reference_artist)
    url = escape(match.url, quote=True)
    lines = match.text.strip().splitlines()
    verses: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if not line.strip() or len(current) >= 5:
            if current:
                verses.append(current)
                current = []
        if line.strip():
            current.append(escape(line.strip()))
    if current:
        verses.append(current)
    formatted = "".join(
        '<p style="margin:0 0 20px;line-height:1.85;">'
        + "<br>".join(verse)
        + "</p>"
        for verse in verses
    )
    return (
        '<section style="max-width:760px;padding:12px 10px;">'
        f'<h3 style="margin:0 0 6px;">{title}</h3>'
        f'<p style="opacity:.72;margin:0 0 8px;">Reference artist: {artist}</p>'
        '<p style="font-size:.88em;opacity:.75;margin:0 0 20px;">'
        "Lyrics from LRCLIB; a cover may use different verses. "
        f'<a href="{url}" target="_blank" rel="noopener noreferrer">View source ↗</a>'
        "</p>"
        '<div style="overflow-y:auto;max-height:450px;'
        'border-top:1px solid var(--border-color-primary);'
        f'padding:18px 4px;">{formatted}</div></section>'
    )


def render_transcription(transcript: str | None, message: str = "") -> str:
    """Show a machine-generated ASR fallback without presenting it as official lyrics."""
    if not transcript:
        detail = escape(message or "No usable text could be extracted from this song.")
        return (
            '<p style="padding:14px;opacity:.75;">'
            f'{detail} The catalog entry is metadata-only.</p>'
        )
    words = escape(transcript.strip())
    # ASR produces prose; preserve word wrapping instead of inventing song verses.
    return (
        '<section style="max-width:760px;padding:12px 10px;">'
        '<h3 style="margin:0 0 8px;">Automatic audio transcription</h3>'
        '<p style="opacity:.75;font-size:.91em;margin:0 0 16px;">'
        'Generated by speech recognition because no reference lyrics were available. '
        'Singing and backing instruments can cause omissions or incorrect words.'
        '</p>'
        '<div style="line-height:1.9;max-height:450px;overflow-y:auto;'
        'padding:18px 14px;border:1px solid var(--border-color-primary);'
        'border-radius:10px;white-space:pre-wrap;overflow-wrap:anywhere;">'
        f'{words}</div></section>'
    )
