import os
from pathlib import Path

import gradio as gr

from cover_search.config import Settings
from cover_search.discovery import discover_versions
from cover_search.index import SongIndex
from cover_search.local import embed_local, transcribe_local
from cover_search.models import Song
from cover_search.parakeet import transcribe_parakeet
from cover_search.services import embed_text, transcribe_audio
from cover_search.youtube import download_audio

settings = Settings()
MAX_AUDIO_BYTES = 20 * 1024 * 1024


def read_audio(path: str) -> bytes:
    file = Path(path)
    if file.stat().st_size > MAX_AUDIO_BYTES:
        raise gr.Error("Audio exceeds the 20 MB upload limit.")
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


def embed(transcript: str):
    if settings.provider == "local":
        return embed_local(transcript, settings.local_embedding_model)
    return embed_text(get_client(), transcript, settings.embedding_model)


def get_client():
    if settings.provider != "openai":
        raise gr.Error("Select AI_PROVIDER=openai to use the hosted provider.")
    if not os.getenv("OPENAI_API_KEY"):
        raise gr.Error("OpenAI mode requires OPENAI_API_KEY. Use local mode for free inference.")
    from openai import OpenAI

    return OpenAI()


def catalog_rows(catalog: SongIndex) -> list[list[str]]:
    return [[song.title, song.artist, song.source] for song in catalog.songs]


def add_audio_bytes(
    catalog: SongIndex, data: bytes, filename: str, title: str, artist: str, source: str
) -> tuple[str, str]:
    if not data or len(data) > MAX_AUDIO_BYTES:
        raise gr.Error("Audio must be nonempty and at most 20 MB.")
    if not title.strip() or not artist.strip():
        raise gr.Error("Title and artist are required.")
    transcript = transcribe(data, filename)
    embedding = embed(transcript)
    song = Song(title.strip(), artist.strip(), transcript, source=source)
    catalog.add(song, embedding)
    return f"Indexed {song.label} ({len(transcript)} transcript characters).", transcript


def add_upload(audio_path: str | None, title: str, artist: str, catalog: SongIndex):
    if not audio_path:
        raise gr.Error("Upload an audio file first.")
    path = Path(audio_path)
    status, transcript = add_audio_bytes(
        catalog, read_audio(str(path)), path.name, title, artist, "upload"
    )
    return status, catalog_rows(catalog), catalog, transcript


def add_youtube(url: str, catalog: SongIndex):
    if not url.strip():
        raise gr.Error("Enter a YouTube URL first.")
    audio = download_audio(url.strip())
    status, transcript = add_audio_bytes(
        catalog, audio.data, audio.filename, audio.title, audio.artist, "youtube"
    )
    return status, catalog_rows(catalog), catalog, transcript


def format_results(results):
    return [
        [rank, song.title, song.artist, f"{score * 100:.1f}%", song.source]
        for rank, (song, score) in enumerate(results, start=1)
    ]


def search_upload(audio_path: str | None, top_k: int, catalog: SongIndex):
    if not audio_path:
        raise gr.Error("Upload a query audio file first.")
    if not catalog.songs:
        raise gr.Error("Index at least one song before searching.")
    path = Path(audio_path)
    transcript = transcribe(read_audio(str(path)), path.name)
    embedding = embed(transcript)
    return format_results(catalog.search(embedding, int(top_k)))


def search_youtube(url: str, top_k: int, catalog: SongIndex):
    if not url.strip():
        raise gr.Error("Enter a YouTube URL first.")
    if not catalog.songs:
        raise gr.Error("Index at least one song before searching.")
    audio = download_audio(url.strip())
    transcript = transcribe(audio.data, audio.filename)
    embedding = embed(transcript)
    return format_results(catalog.search(embedding, int(top_k)))


def search_public_versions(title: str):
    try:
        works, recordings = discover_versions(title)
    except (ValueError, Exception) as exc:
        if isinstance(exc, ValueError):
            raise gr.Error(str(exc)) from exc
        raise gr.Error("MusicBrainz is unavailable. Please retry later.") from exc
    if not works:
        return "No matching musical work found.", []
    work = works[0]
    note = (
        f"Recordings associated with **{work['title']}** in MusicBrainz. "
        "Results are metadata links, not a verified ranking of covers. "
        "Other works with the same title may exist."
    )
    rows = [
        [record["title"], record["artist"], record["url"]]
        for record in recordings
    ]
    return note, rows


with gr.Blocks(title="Semantic Cover Search") as demo:
    catalog_state = gr.State(value=SongIndex)
    gr.Markdown(
        "# 🎵 Semantic Cover Search\n"
        "Compare lyrical content through transcription and embeddings. "
        "Parakeet CPU mode requires no API key."
    )

    with gr.Tab("Discover versions by song title"):
        gr.Markdown(
            "Search online MusicBrainz metadata for performances of a composition. "
            "No personal catalog or API key needed. This does not identify recordings from singing."
        )
        public_title = gr.Textbox(label="Song title", placeholder="e.g. Hallelujah")
        public_search = gr.Button("Find recorded versions", variant="primary")
        public_status = gr.Markdown()
        public_results = gr.Dataframe(
            headers=["Recording", "Artist", "MusicBrainz link"],
            datatype=["str", "str", "str"],
            interactive=False,
            label="Recorded versions (MusicBrainz)",
        )
        public_search.click(
            search_public_versions,
            inputs=public_title,
            outputs=[public_status, public_results],
        )

    with gr.Tab("Experimental local catalog"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Upload a song to the catalog")
                add_file = gr.Audio(type="filepath", sources=["upload"], label="Audio file to index")
                add_title = gr.Textbox(label="Title")
                add_artist = gr.Textbox(label="Artist")
                add_file_button = gr.Button("Transcribe & index", variant="primary")
            with gr.Column():
                gr.Markdown("### Add a song from YouTube")
                add_url = gr.Textbox(label="YouTube URL")
                add_url_button = gr.Button("Download, transcribe & index")

        add_status = gr.Markdown()
        with gr.Accordion("Latest indexed transcript", open=False):
            transcript_view = gr.Textbox(
                label="Recognized lyrics / speech",
                lines=8,
                max_lines=12,
                interactive=False,
            )
        catalog_table = gr.Dataframe(
            headers=["Title", "Artist", "Source"],
            datatype=["str", "str", "str"],
            interactive=False,
            label="Indexed catalog",
        )
        add_file_button.click(
            add_upload,
            inputs=[add_file, add_title, add_artist, catalog_state],
            outputs=[add_status, catalog_table, catalog_state, transcript_view],
        )
        add_url_button.click(
            add_youtube,
            inputs=[add_url, catalog_state],
            outputs=[add_status, catalog_table, catalog_state, transcript_view],
        )

    with gr.Tab("Experimental singing search"):
        top_k = gr.Slider(1, 10, value=5, step=1, label="Top results")
        results_table = gr.Dataframe(
            headers=["Rank", "Title", "Artist", "Similarity", "Source"],
            datatype=["number", "str", "str", "str", "str"],
            interactive=False,
            label="Most similar indexed songs",
        )
        gr.Markdown(
            "Record yourself singing or speaking a short fragment to search "
            "the songs already in your catalog."
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
