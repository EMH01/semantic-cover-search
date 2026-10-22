"""Semantic Cover Search: online cover discovery and session-only lyric similarity."""
import os
from html import escape
from pathlib import Path

import gradio as gr

from cover_search.config import Settings
from cover_search.index import SongIndex
from cover_search.local import embed_local, transcribe_local
from cover_search.lyrics import (
    LyricsLookupError,
    find_lyrics,
    lyric_title_for_result,
    render_lyrics,
    title_from_video,
)
from cover_search.metadata import audio_metadata
from cover_search.models import Song
from cover_search.parakeet import transcribe_parakeet
from cover_search.services import embed_text, transcribe_audio
from cover_search.youtube import search_youtube_videos, video_metadata

settings = Settings()
MAX_AUDIO_BYTES = 20 * 1024 * 1024
SUPPORTED_AUDIO = {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".webm", ".mp4"}


def read_audio(path: str) -> bytes:
    file = Path(path)
    if file.suffix.lower() not in SUPPORTED_AUDIO:
        raise gr.Error("Unsupported audio format.")
    if not 0 < file.stat().st_size <= MAX_AUDIO_BYTES:
        raise gr.Error("Audio must be nonempty and at most 20 MB.")
    return file.read_bytes()


def transcribe(data: bytes, filename: str) -> str:
    if settings.provider not in {"local", "openai"}:
        raise gr.Error("AI_PROVIDER must be local or openai.")
    if settings.provider == "local":
        if settings.asr_engine == "parakeet":
            return transcribe_parakeet(data, filename)
        if settings.asr_engine == "whisper":
            return transcribe_local(data, filename, settings.local_whisper_model)
        raise gr.Error("ASR_ENGINE must be parakeet or whisper.")
    return transcribe_audio(get_client(), data, filename, settings.transcription_model)


def embed(text: str):
    if settings.provider == "local":
        return embed_local(text, settings.local_embedding_model)
    return embed_text(get_client(), text, settings.embedding_model)


def get_client():
    if settings.provider != "openai":
        raise gr.Error("Select AI_PROVIDER=openai to use the hosted provider.")
    if not os.getenv("OPENAI_API_KEY"):
        raise gr.Error("OpenAI mode requires OPENAI_API_KEY. Use local mode for free inference.")
    from openai import OpenAI

    return OpenAI()


def catalog_rows(catalog: SongIndex) -> list[list[str]]:
    return [
        [song.title, song.artist, song.source, "Lyrics" if song.transcript else "Metadata only"]
        for song in catalog.songs
    ]


def add_known_song(
    catalog: SongIndex,
    title: str,
    artist: str,
    source: str,
    lyric_title: str | None = None,
    lyric_artist: str = "",
) -> tuple[str, list[list[str]], SongIndex, str]:
    """Add metadata, optionally finding lyrics. Never download/transcribe full songs."""
    title = title.strip() or "Untitled recording"
    artist = artist.strip() or "Unknown performer"
    if any(
        song.title.casefold() == title.casefold()
        and song.artist.casefold() == artist.casefold()
        for song in catalog.songs
    ):
        return (
            f"**Already in local catalog:** {escape(title)} — {escape(artist)}.",
            catalog_rows(catalog), catalog, "",
        )
    requested_lyrics_title = (lyric_title or title).strip()
    match = None
    lookup_error = ""
    try:
        match = find_lyrics(requested_lyrics_title, lyric_artist)
    except LyricsLookupError as exc:
        lookup_error = str(exc)
    lyrics = match.text if match else ""
    vector = None
    index_error = False
    if lyrics:
        try:
            vector = embed(lyrics)
        except Exception:
            index_error = True
            lyrics = ""
    song = Song(title, artist, lyrics, source=source)
    try:
        catalog.add(song, vector)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    status = f"**Added to local catalog:** {escape(song.label)}."
    if match is not None and vector is not None:
        status += (
            f" Lyrics matched against **{escape(match.title)} — "
            f"{escape(match.reference_artist)}** "
            f"([LRCLIB source]({match.url})). "
            "The reference may differ from this performance."
        )
    elif match is not None and index_error:
        status += (
            " Reference lyrics were found, but semantic indexing is unavailable. "
            "This entry is metadata-only."
        )
    elif lookup_error:
        status += (
            f" **Lyrics service problem:** {escape(lookup_error)} "
            "The song was added as metadata-only."
        )
    else:
        status += (
            f" No exact LRCLIB lyrics match for **{escape(requested_lyrics_title)}**. "
            "The entry is metadata-only."
        )
    return status, catalog_rows(catalog), catalog, render_lyrics(match)

def add_upload(audio_path: str | None, title: str, artist: str, catalog: SongIndex):
    if not audio_path:
        raise gr.Error("Upload an audio file first.")
    path = Path(audio_path)
    if path.suffix.lower() not in SUPPORTED_AUDIO:
        raise gr.Error("Unsupported audio format.")
    if not 0 < path.stat().st_size <= MAX_AUDIO_BYTES:
        raise gr.Error("Audio must be nonempty and at most 20 MB.")
    detected_title, detected_artist = audio_metadata(audio_path)
    return add_known_song(
        catalog, title or detected_title, artist or detected_artist, "upload",
        lyric_artist=artist or detected_artist,
    )


def add_youtube(url: str, catalog: SongIndex):
    if not url.strip():
        raise gr.Error("Enter a YouTube URL first.")
    try:
        video = video_metadata(url)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error("YouTube video metadata is unavailable. Check the link.") from exc
    return add_known_song(
        catalog, video["title"], f"Uploader: {video['channel']}", "youtube",
        lyric_title=title_from_video(video["title"]),
    )


def format_results(results):
    return [
        [rank, song.title, song.artist, f"{score * 100:.1f}%", song.source]
        for rank, (song, score) in enumerate(results, start=1)
    ]


def search_upload(audio_path: str | None, top_k: int, catalog: SongIndex):
    if not audio_path:
        raise gr.Error("Record a short fragment first.")
    if not catalog.indexed_count:
        raise gr.Error(
            "No songs with matching lyrics are indexed. Add songs with identifiable "
            "lyrics to the local catalog first."
        )
    path = Path(audio_path)
    try:
        transcript = transcribe(read_audio(str(path)), path.name)
        query = embed(transcript)
    except ValueError as exc:
        raise gr.Error(
            "Speech recognition could not extract words from this recording. "
            "Try a clear, short sung phrase; humming recognition is not supported yet."
        ) from exc
    return format_results(catalog.search(query, int(top_k)))


def render_video_cards(videos: list[dict[str, str]]) -> str:
    """Create readable, safely escaped cards from third-party video metadata."""
    if not videos:
        return "<p>No matching videos found.</p>"
    cards = []
    for item in videos:
        title = escape(item["title"])
        channel = escape(item["channel"])
        url = escape(item["url"], quote=True)
        cards.append(
            '<article style="padding:14px 16px;border:1px solid var(--border-color-primary);'
            'border-radius:10px;margin:10px 0;">'
            f'<div style="font-weight:600;margin-bottom:6px;">{title}</div>'
            f'<div style="opacity:.75;font-size:.9em;margin-bottom:9px;">Channel: {channel}</div>'
            f'<a href="{url}" target="_blank" rel="noopener noreferrer">'
            'Watch on YouTube ↗</a></article>'
        )
    return "".join(cards)


def search_public_versions(title: str):
    try:
        videos = search_youtube_videos(title)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error(
            "YouTube search is temporarily unavailable. Check connectivity or yt-dlp."
        ) from exc
    for item in videos:
        item["search_title"] = title.strip()
    note = (
        f"Showing {len(videos)} YouTube video candidates for **{escape(title.strip())}**. "
        "Channels are uploaders, not necessarily performing artists. "
        "These are not verified cover recordings."
    )
    options = [
        (f"{i + 1}. {item['title']} — {item['channel']}", str(i))
        for i, item in enumerate(videos)
    ]
    return (
        note,
        render_video_cards(videos),
        videos,
        gr.update(choices=options, value=None, visible=bool(videos)),
        gr.update(visible=bool(videos)),
    )


def add_selected_video(selection: str | None, videos: list[dict], catalog: SongIndex):
    if selection is None:
        raise gr.Error("Select a video from the search results first.")
    try:
        item = videos[int(selection)]
    except (TypeError, ValueError, IndexError) as exc:
        raise gr.Error("Selection has expired. Run the search again.") from exc
    return add_known_song(
        catalog, item["title"], f"Uploader: {item['channel']}", "youtube",
        lyric_title=lyric_title_for_result(item["title"], item.get("search_title", "")),
    )


with gr.Blocks(title="Semantic Cover Search") as demo:
    catalog_state = gr.State(value=SongIndex)
    youtube_results_state = gr.State(value=[])
    gr.Markdown("# 🎵 Semantic Cover Search")

    with gr.Tab("Discover covers on YouTube"):
        gr.Markdown(
            "Find cover video candidates on YouTube by song title. "
            "Channels identify uploaders; verify the performers by opening each video. "
            "No personal catalog or API key required."
        )
        public_title = gr.Textbox(label="Song title", placeholder="e.g. Hallelujah")
        public_search = gr.Button("Search covers on YouTube", variant="primary")
        public_status = gr.Markdown()
        public_results = gr.HTML(label="YouTube results")
        result_selection = gr.Dropdown(
            label="Choose a video to add to your local catalog",
            choices=[],
            interactive=True,
            visible=False,
        )
        add_selected_button = gr.Button(
            "Add selected video to local catalog", visible=False
        )
        selected_status = gr.Markdown()
        public_search.click(
            search_public_versions,
            inputs=public_title,
            outputs=[
                public_status,
                public_results,
                youtube_results_state,
                result_selection,
                add_selected_button,
            ],
            show_progress="minimal",
        )

    with gr.Tab("Local catalog"):
        gr.Markdown(
            "Session-only catalog: titles and references are kept until Gradio restarts. "
            "Known songs use online lyrics when available; full-song transcription "
            "and audio downloads are not required to add a song."
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Upload audio metadata")
                add_file = gr.Audio(
                    type="filepath", sources=["upload"], label="Audio file to add"
                )
                add_title = gr.Textbox(label="Song title (editable, optional)")
                add_artist = gr.Textbox(label="Performing artist (editable, optional)")
                add_file.change(
                    audio_metadata,
                    inputs=add_file,
                    outputs=[add_title, add_artist],
                )
                add_file_button = gr.Button("Add song to catalog", variant="primary")
            with gr.Column():
                gr.Markdown("### Add a song from YouTube")
                add_url = gr.Textbox(label="YouTube URL")
                add_url_button = gr.Button("Add link to catalog")

        add_status = gr.Markdown()
        with gr.Accordion("Reference lyrics · latest match (LRCLIB)", open=False):
            lyrics_view = gr.HTML(value=render_lyrics(None))
        catalog_table = gr.Dataframe(
            headers=["Title", "Performer / uploader", "Source", "Searchable"],
            datatype=["str", "str", "str", "str"],
            interactive=False,
            label="Session catalog",
        )
        add_selected_button.click(
            add_selected_video,
            inputs=[result_selection, youtube_results_state, catalog_state],
            outputs=[selected_status, catalog_table, catalog_state, lyrics_view],
        )
        add_file_button.click(
            add_upload,
            inputs=[add_file, add_title, add_artist, catalog_state],
            outputs=[add_status, catalog_table, catalog_state, lyrics_view],
        )
        add_url_button.click(
            add_youtube,
            inputs=[add_url, catalog_state],
            outputs=[add_status, catalog_table, catalog_state, lyrics_view],
        )

    with gr.Tab("Experimental singing search"):
        top_k = gr.Slider(1, 10, value=5, step=1, label="Top results")
        results_table = gr.Dataframe(
            headers=["Rank", "Title", "Performer / uploader", "Similarity", "Source"],
            datatype=["number", "str", "str", "str", "str"],
            interactive=False,
            label="Closest lyric matches in the session",
        )
        gr.Markdown(
            "Record a short sung or spoken phrase. Parakeet transcribes your recording "
            "and searches the lyrics indexed in this session. "
            "It cannot yet identify unknown songs from the whole internet or recognize humming."
        )
        query_file = gr.Audio(
            type="filepath",
            sources=["microphone"],
            label="Record a fragment to search",
        )
        search_file_button = gr.Button("Find similar songs", variant="primary")
        search_file_button.click(
            search_upload,
            inputs=[query_file, top_k, catalog_state],
            outputs=results_table,
        )


if __name__ == "__main__":
    demo.launch()
