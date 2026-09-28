import os
from pathlib import Path

import gradio as gr
from openai import OpenAI

from cover_search.config import Settings
from cover_search.index import SongIndex
from cover_search.models import Song
from cover_search.services import embed_text, transcribe_audio
from cover_search.youtube import download_audio

settings = Settings()
catalog = SongIndex()


def get_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise gr.Error("Set OPENAI_API_KEY before using transcription or embeddings.")
    return OpenAI()


def catalog_rows() -> list[list[str]]:
    return [[song.title, song.artist, song.source] for song in catalog.songs]


def add_audio_bytes(data: bytes, filename: str, title: str, artist: str, source: str) -> str:
    if not title.strip() or not artist.strip():
        raise gr.Error("Title and artist are required.")
    client = get_client()
    transcript = transcribe_audio(client, data, filename, settings.transcription_model)
    embedding = embed_text(client, transcript, settings.embedding_model)
    song = Song(title.strip(), artist.strip(), transcript, source=source)
    catalog.add(song, embedding)
    return f"Indexed {song.label} ({len(transcript)} transcript characters)."


def add_upload(audio_path: str | None, title: str, artist: str):
    if not audio_path:
        raise gr.Error("Upload an audio file first.")
    path = Path(audio_path)
    status = add_audio_bytes(path.read_bytes(), path.name, title, artist, "upload")
    return status, catalog_rows()


def add_youtube(url: str):
    if not url.strip():
        raise gr.Error("Enter a YouTube URL first.")
    audio = download_audio(url.strip())
    status = add_audio_bytes(audio.data, audio.filename, audio.title, audio.artist, "youtube")
    return status, catalog_rows()


def format_results(results):
    return [
        [rank, song.title, song.artist, f"{score * 100:.1f}%", song.source]
        for rank, (song, score) in enumerate(results, start=1)
    ]


def search_upload(audio_path: str | None, top_k: int):
    if not audio_path:
        raise gr.Error("Upload a query audio file first.")
    if not catalog.songs:
        raise gr.Error("Index at least one song before searching.")
    path = Path(audio_path)
    client = get_client()
    transcript = transcribe_audio(
        client,
        path.read_bytes(),
        path.name,
        settings.transcription_model,
    )
    embedding = embed_text(client, transcript, settings.embedding_model)
    return format_results(catalog.search(embedding, int(top_k)))


def search_youtube(url: str, top_k: int):
    if not url.strip():
        raise gr.Error("Enter a YouTube URL first.")
    if not catalog.songs:
        raise gr.Error("Index at least one song before searching.")
    audio = download_audio(url.strip())
    client = get_client()
    transcript = transcribe_audio(client, audio.data, audio.filename, settings.transcription_model)
    embedding = embed_text(client, transcript, settings.embedding_model)
    return format_results(catalog.search(embedding, int(top_k)))


with gr.Blocks(title="Semantic Cover Search") as demo:
    gr.Markdown(
        "# 🎵 Semantic Cover Search\n"
        "Discover likely covers or lyrically similar performances using transcription + embeddings."
    )

    with gr.Tab("Build catalog"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Upload audio")
                add_file = gr.Audio(type="filepath", label="Audio file")
                add_title = gr.Textbox(label="Title")
                add_artist = gr.Textbox(label="Artist")
                add_file_button = gr.Button("Transcribe & index", variant="primary")
            with gr.Column():
                gr.Markdown("### Optional YouTube source")
                add_url = gr.Textbox(label="YouTube URL")
                add_url_button = gr.Button("Download, transcribe & index")
                gr.Markdown('Requires the optional extra: `pip install -e ".[youtube]"`.')

        add_status = gr.Markdown()
        catalog_table = gr.Dataframe(
            headers=["Title", "Artist", "Source"],
            datatype=["str", "str", "str"],
            interactive=False,
            label="Indexed catalog",
        )
        add_file_button.click(
            add_upload,
            inputs=[add_file, add_title, add_artist],
            outputs=[add_status, catalog_table],
        )
        add_url_button.click(
            add_youtube,
            inputs=add_url,
            outputs=[add_status, catalog_table],
        )

    with gr.Tab("Search"):
        top_k = gr.Slider(1, 10, value=5, step=1, label="Top results")
        results_table = gr.Dataframe(
            headers=["Rank", "Title", "Artist", "Similarity", "Source"],
            datatype=["number", "str", "str", "str", "str"],
            interactive=False,
            label="Most similar indexed songs",
        )
        with gr.Row():
            with gr.Column():
                query_file = gr.Audio(type="filepath", label="Query audio")
                search_file_button = gr.Button("Search from upload", variant="primary")
            with gr.Column():
                query_url = gr.Textbox(label="Query YouTube URL")
                search_url_button = gr.Button("Search from YouTube")

        search_file_button.click(
            search_upload,
            inputs=[query_file, top_k],
            outputs=results_table,
        )
        search_url_button.click(
            search_youtube,
            inputs=[query_url, top_k],
            outputs=results_table,
        )


if __name__ == "__main__":
    demo.launch()
