"""Discover recordings of a musical work through MusicBrainz metadata.

No audio identification or lyric search is performed by this module.
"""
import threading
import time
from urllib.parse import quote

import httpx

BASE_URL = "https://musicbrainz.org/ws/2"
HEADERS = {
    "User-Agent": "SemanticCoverSearch/0.1 (https://github.com/EMH01/semantic-cover-search)",
    "Accept": "application/json",
}
_request_lock = threading.Lock()
_last_request = 0.0


def _fetch(path: str, params: dict[str, str | int]) -> dict:
    """Rate-limit attempts and retry temporary MusicBrainz failures."""
    global _last_request
    retry_codes = {429, 500, 502, 503, 504}
    with _request_lock:
        with httpx.Client(timeout=15, headers=HEADERS, follow_redirects=True) as client:
            for attempt in range(3):
                delay = max(0.0, 1.1 - (time.monotonic() - _last_request))
                if delay:
                    time.sleep(delay)
                try:
                    response = client.get(
                        f"{BASE_URL}/{path}", params={**params, "fmt": "json"}
                    )
                finally:
                    _last_request = time.monotonic()
                if response.status_code not in retry_codes or attempt == 2:
                    response.raise_for_status()
                    return response.json()
                # Back off in addition to the regular per-attempt rate limit.
                time.sleep(min(2 ** attempt, 4))
    raise RuntimeError("MusicBrainz request failed after retries.")


def find_works(title: str, limit: int = 8) -> list[dict]:
    title = title.strip()
    if len(title) < 2 or len(title) > 100:
        raise ValueError("Enter a song title between 2 and 100 characters.")
    data = _fetch("work", {"query": title, "limit": min(max(limit, 1), 20)})
    return [
        {"id": item["id"], "title": item.get("title", ""), "score": item.get("score", 0)}
        for item in data.get("works", [])
        if item.get("id")
    ]


def recordings_for_work(work_id: str, limit: int = 50) -> list[dict]:
    import uuid

    try:
        work_id = str(uuid.UUID(work_id))
    except (ValueError, AttributeError) as exc:
        raise ValueError("Invalid MusicBrainz work identifier.") from exc
    data = _fetch("recording", {"work": work_id, "limit": min(max(limit, 1), 100)})
    recordings = []
    for item in data.get("recordings", []):
        artist = "".join(
            credit if isinstance(credit, str) else credit.get("name", "")
            for credit in item.get("artist-credit", [])
        )
        if item.get("id"):
            recordings.append({
                "title": item.get("title", ""),
                "artist": artist or "Unknown artist",
                "url": f"https://musicbrainz.org/recording/{quote(item['id'])}",
                "id": item["id"],
            })
    return recordings


def discover_versions(title: str) -> tuple[list[dict], list[dict]]:
    """Return candidate musical works and recordings of the top-scoring work."""
    works = find_works(title)
    if not works:
        return [], []
    return works, recordings_for_work(works[0]["id"])
