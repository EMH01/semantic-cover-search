from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory


@dataclass(frozen=True, slots=True)
class YouTubeAudio:
    title: str
    artist: str
    filename: str
    data: bytes


def download_audio(url: str) -> YouTubeAudio:
    """Download the best available audio stream using the optional yt-dlp dependency."""
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

            return YouTubeAudio(
                title=info.get("title") or "Unknown title",
                artist=info.get("artist") or info.get("uploader") or "Unknown artist",
                filename=path.name,
                data=path.read_bytes(),
            )
