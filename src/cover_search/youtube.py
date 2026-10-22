from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlparse


@dataclass(frozen=True, slots=True)
class YouTubeAudio:
    title: str
    artist: str
    filename: str
    data: bytes


def validate_youtube_url(url: str) -> str:
    """Accept canonical video, shortened and YouTube Music links only."""
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or parsed.username or parsed.password:
        raise ValueError("Enter a valid HTTPS YouTube video URL.")
    if host in {"www.youtube.com", "youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path != "/watch" or not parse_qs(parsed.query).get("v"):
            raise ValueError("Enter a YouTube video URL containing a video ID.")
    elif host == "youtu.be":
        if not parsed.path.strip("/") or "/" in parsed.path.strip("/"):
            raise ValueError("Enter a valid shortened YouTube video URL.")
    else:
        raise ValueError("Only YouTube and YouTube Music video URLs are supported.")
    return url.strip()


def download_audio(url: str) -> YouTubeAudio:
    """Download a bounded audio source using the optional yt-dlp adapter."""
    url = validate_youtube_url(url)
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            'YouTube support is optional. Install with: pip install -e ".[youtube]"'
        ) from exc

    with TemporaryDirectory() as temp_dir:
        output_template = str(Path(temp_dir) / "%(id)s.%(ext)s")
        options = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "quiet": True,
            "noplaylist": True,
            "max_filesize": 20 * 1024 * 1024,
            "match_filter": lambda info, *, incomplete=False: (
                "Video exceeds the 10-minute duration limit."
                if info.get("duration") and info["duration"] > 600 else None
            ),
        }
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))
            if not path.exists():
                requested = info.get("requested_downloads") or []
                if requested and requested[0].get("filepath"):
                    path = Path(requested[0]["filepath"])
            if not path.exists():
                raise RuntimeError("yt-dlp completed but no audio file could be located.")

            if path.stat().st_size > 20 * 1024 * 1024:
                raise ValueError("Downloaded audio exceeds the 20 MB limit.")
            return YouTubeAudio(
                title=info.get("title") or "Unknown title",
                artist=info.get("artist") or info.get("uploader") or "Unknown artist",
                filename=path.name,
                data=path.read_bytes(),
            )
