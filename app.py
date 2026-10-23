"""Semantic Cover Search: online cover discovery and session-only lyric similarity."""
import os
import re
from html import escape
from inspect import signature
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import gradio as gr

from cover_search.config import Settings
from cover_search.index import SongIndex
from cover_search.local import embed_local, transcribe_local
from cover_search.lyrics import (
    LyricsLookupError,
    find_lyrics,
    lyric_title_for_result,
    render_lyrics,
    render_transcription,
    title_from_video,
)
from cover_search.metadata import audio_metadata
from cover_search.models import Song
from cover_search.parakeet import transcribe_parakeet
from cover_search.services import embed_text, transcribe_audio
from cover_search.ui_theme import APP_CSS, HERO_HTML
from cover_search.youtube import (
    download_audio,
    search_youtube_lyrics_fragment,
    search_youtube_videos,
    video_metadata,
)

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
    names = {
        "lrclib": "Reference lyrics (LRCLIB)",
        "asr": "Automatic transcription",
        "none": "Metadata only",
    }
    return [
        [song.title, song.artist, song.source, names.get(song.text_origin, "Metadata only")]
        for song in catalog.songs
    ]


def add_known_song(
    catalog: SongIndex,
    title: str,
    artist: str,
    source: str,
    lyric_title: str | None = None,
    lyric_artist: str = "",
    fallback_audio_path: str | None = None,
    fallback_youtube_url: str | None = None,
    progress: gr.Progress | None = None,
) -> tuple[str, list[list[str]], SongIndex, str]:
    """Try reference lyrics first, then speech recognition for unavailable lyrics."""
    title = title.strip() or "Untitled recording"
    artist = artist.strip() or "Unknown performer"
    if any(
        song.title.casefold() == title.casefold()
        and song.artist.casefold() == artist.casefold()
        for song in catalog.songs
    ):
        return (
            f"**Already in local catalog:** {escape(title)} — {escape(artist)}.",
            catalog_rows(catalog),
            catalog,
            render_transcription(None, "This song is already in the catalog."),
        )

    if progress is not None:
        progress(0.12, desc="Searching for reference lyrics…")
    requested_title = (lyric_title or title).strip()
    match = None
    lookup_error = ""
    try:
        match = find_lyrics(requested_title, lyric_artist)
    except LyricsLookupError as exc:
        lookup_error = str(exc)

    text = match.text if match else ""
    origin = "lrclib" if text else "none"
    fallback_error = ""
    if not text and (fallback_audio_path or fallback_youtube_url):
        if progress is not None:
            progress(0.35, desc="No reference lyrics; preparing audio…")
        try:
            if fallback_audio_path:
                file = Path(fallback_audio_path)
                audio_bytes, audio_name = read_audio(str(file)), file.name
            else:
                # yt-dlp enforces duration and file-size bounds.
                audio = download_audio(fallback_youtube_url)
                audio_bytes, audio_name = audio.data, audio.filename
            if progress is not None:
                progress(0.62, desc="Transcribing audio with local ASR…")
            text = transcribe(audio_bytes, audio_name).strip()
            if not text:
                raise ValueError("The speech recognizer did not detect any words.")
            origin = "asr"
        except Exception as exc:
            # A poor recording or missing YouTube format must not prevent saving metadata.
            fallback_error = type(exc).__name__
            text = ""

    display_text = text
    embedding = None
    index_error = False
    if text:
        if progress is not None:
            progress(0.82, desc="Creating text embedding for similarity search…")
        try:
            embedding = embed(text)
        except Exception:
            index_error = True
            text = ""
    if embedding is None:
        origin = "none"

    song = Song(title, artist, text, source=source, text_origin=origin)
    try:
        catalog.add(song, embedding)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    status = f"**Added to local catalog:** {escape(song.label)}."
    if origin == "lrclib" and match is not None:
        status += (
            f" Reference lyrics: **{escape(match.title)} — "
            f"{escape(match.reference_artist)}** "
            f"([LRCLIB source]({match.url})). "
            "The recording may use different verses."
        )
        view = render_lyrics(match)
    elif origin == "asr":
        status += (
            " LRCLIB did not supply usable lyrics, so the audio was transcribed "
            "automatically and indexed. Singing recognition may be inaccurate."
        )
        view = render_transcription(text)
    else:
        status += " No searchable text was obtained; saved as **metadata only**."
        if index_error:
            status += " Text was obtained, but vector indexing failed."
        elif fallback_error:
            status += (
                " Audio download or speech recognition failed "
                f"({escape(fallback_error)})."
            )
        if match is not None:
            view = render_lyrics(match)
        else:
            view = render_transcription(display_text or None)

    if lookup_error:
        status += f" LRCLIB issue: {escape(lookup_error)}"
    if progress is not None:
        progress(1.0, desc="Catalog updated")
    return status, catalog_rows(catalog), catalog, view

def add_upload(
    audio_path: str | None, title: str, artist: str, catalog: SongIndex,
    progress: gr.Progress = gr.Progress(),
):
    if not audio_path:
        raise gr.Error("Upload an audio file first.")
    path = Path(audio_path)
    if path.suffix.lower() not in SUPPORTED_AUDIO:
        raise gr.Error("Unsupported audio format.")
    if not 0 < path.stat().st_size <= MAX_AUDIO_BYTES:
        raise gr.Error("Audio must be nonempty and at most 20 MB.")
    progress(0.05, desc="Reading audio metadata…")
    detected_title, detected_artist = audio_metadata(audio_path)
    return add_known_song(
        catalog, title or detected_title, artist or detected_artist, "upload",
        lyric_artist=artist or detected_artist,
        fallback_audio_path=audio_path,
        progress=progress,
    )


def add_youtube(
    url: str, catalog: SongIndex, progress: gr.Progress = gr.Progress()
):
    if not url.strip():
        raise gr.Error("Enter a YouTube URL first.")
    progress(0.05, desc="Reading YouTube video metadata…")
    try:
        video = video_metadata(url)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error("YouTube video metadata is unavailable. Check the link.") from exc
    return add_known_song(
        catalog, video["title"], f"Uploader: {video['channel']}", "youtube",
        lyric_title=title_from_video(video["title"]),
        fallback_youtube_url=video["url"],
        progress=progress,
    )


def format_results(results):
    return [
        [rank, song.title, song.artist, f"{score * 100:.1f}%", song.source]
        for rank, (song, score) in enumerate(results, start=1)
    ]


def render_lyric_comparison_cards(comparisons) -> str:
    """Compact comparison cards; lexical and semantic metrics stay distinct."""
    if not comparisons:
        return (
            '<div class="scs-empty">No song pairs to compare. '
            'Add two songs with lyrics first.</div>'
        )
    cards = []
    for rank, pair in enumerate(comparisons, start=1):
        left, right = escape(pair.first.label), escape(pair.second.label)
        lexical = pair.phrase_overlap * 100
        cosine = pair.semantic_similarity * 100
        cosine_width = max(0, min(100, cosine))
        first_source = "LRCLIB" if pair.first.text_origin == "lrclib" else "ASR"
        second_source = "LRCLIB" if pair.second.text_origin == "lrclib" else "ASR"
        note = (
            "Identical indexed text is not proof of identical performances."
            if pair.exact_text else
            "Scores describe indexed text, not verified cover identity."
        )
        cards.append(
            '<article class="scs-match">'
            f'<div class="scs-match-head">Comparison {rank:02d}</div>'
            f'<div class="scs-track-name">{left}</div>'
            f'<div class="scs-track-source">Text: {first_source}</div>'
            '<div class="scs-match-divider"></div>'
            f'<div class="scs-track-name">{right}</div>'
            f'<div class="scs-track-source">Text: {second_source}</div>'
            '<div class="scs-metrics">'
            '<div class="scs-metric">'
            '<div class="scs-metric-label">Shared lyric phrases</div>'
            f'<div class="scs-metric-value">{lexical:.1f}%</div>'
            f'<div class="scs-meter"><div class="scs-meter-fill" '
            f'style="width:{lexical:.1f}%"></div></div></div>'
            '<div class="scs-metric">'
            '<div class="scs-metric-label" '
            'title="Cosine similarity between normalized lyric embeddings, '
            'multiplied by 100. Not a probability.">'
            'Semantic similarity ⓘ</div>'
            f'<div class="scs-metric-value">{cosine:.1f}</div>'
            '<div class="scs-meter"><div class="scs-meter-fill scs-meter-fill-alt" '
            f'style="width:{cosine_width:.1f}%"></div></div></div></div>'
            f'<p class="scs-match-note">{escape(note)}</p>'
            '</article>'
        )
    return '<div class="scs-match-grid">' + "".join(cards) + '</div>'

def compare_catalog_lyrics(catalog: SongIndex, top_pairs: int):
    """Evaluate every eligible pair; render only the best N to avoid long screens."""
    total = len(catalog.songs)
    eligible = catalog.indexed_count
    excluded = total - eligible
    pairs = catalog.compare_all()
    if not pairs:
        return (
            f"**{eligible} of {total} songs have searchable text.** "
            "Add at least two songs with reference lyrics or automatic transcription "
            "to compare them. Songs without text are excluded.",
            render_lyric_comparison_cards([]),
        )
    displayed = pairs[:max(1, min(int(top_pairs), 50))]
    excluded_label = "song" if excluded == 1 else "songs"
    pair_label = "pair" if len(pairs) == 1 else "pairs"
    note = (
        f"**{len(pairs)} {pair_label} compared** · {eligible} songs with lyrics · "
        f"{excluded} metadata-only {excluded_label} excluded. "
        f"Showing {len(displayed)} highest-ranked pairs, prioritizing shared phrases. "
        "Semantic similarity = embedding cosine × 100, not a probability. "
        "Neither score confirms a cover relationship."
    )
    return note, render_lyric_comparison_cards(displayed)


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
    """Compact responsive video cards with escaped metadata and thumbnails."""
    if not videos:
        return (
            '<div class="scs-empty">No matching videos found. '
            'Try another song title or a clearer recorded phrase.</div>'
        )
    cards = []
    for item in videos:
        title = escape(item["title"])
        channel = escape(item["channel"])
        url = escape(item["url"], quote=True)
        video_id = parse_qs(urlparse(item["url"]).query).get("v", [""])[0]
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            img = (
                f'<img class="scs-video-thumb" loading="lazy" alt="" '
                f'src="https://i.ytimg.com/vi/{video_id}/mqdefault.jpg">'
            )
        else:
            img = '<div class="scs-video-thumb" aria-hidden="true"></div>'
        cards.append(
            '<article class="scs-video">'
            f'{img}<div class="scs-video-body">'
            f'<div class="scs-video-title">{title}</div>'
            f'<div class="scs-video-channel">{channel}</div>'
            f'<a class="scs-video-link" href="{url}" '
            'target="_blank" rel="noopener noreferrer">'
            'Watch on YouTube ↗</a></div></article>'
        )
    return '<div class="scs-video-grid">' + "".join(cards) + '</div>'

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


def search_recording_on_youtube(
    audio_path: str | None, progress: gr.Progress = gr.Progress()
):
    """Transcribe a recording and search YouTube for lyric-text candidates.

    This does not identify the recording from its melody or guarantee song ID.
    """
    if not audio_path:
        raise gr.Error("Record or upload a short sung or spoken fragment first.")
    path = Path(audio_path)
    progress(0.12, desc="Recognizing sung or spoken words locally…")
    try:
        recognized = transcribe(read_audio(str(path)), path.name).strip()
    except (ValueError, RuntimeError) as exc:
        raise gr.Error(
            "No recognizable words were extracted. Try a clearer phrase; "
            "humming and melodic matching are not implemented."
        ) from exc
    if not recognized:
        raise gr.Error("The recognizer returned no words to search for.")
    progress(0.7, desc="Searching YouTube with the recognized lyric phrase…")
    try:
        videos = search_youtube_lyrics_fragment(recognized)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error("YouTube phrase search is unavailable. Try again later.") from exc
    note = (
        f"**{len(videos)} possible YouTube results** from the recognized words. "
        "These are **unverified song candidates**, not audio/melody matches. "
        "Check video titles and listen to confirm the song before adding it."
    )
    options = [
        (f"{i + 1}. {item['title']} — {item['channel']}", str(i))
        for i, item in enumerate(videos)
    ]
    progress(1.0, desc="Candidate search finished")
    return (
        note,
        render_video_cards(videos),
        videos,
        gr.update(choices=options, value=None, visible=bool(videos)),
        gr.update(visible=bool(videos)),
        recognized,
    )


def add_selected_video(
    selection: str | None, videos: list[dict], catalog: SongIndex,
    progress: gr.Progress = gr.Progress(),
):
    if selection is None:
        raise gr.Error("Select a video from the search results first.")
    try:
        item = videos[int(selection)]
    except (TypeError, ValueError, IndexError) as exc:
        raise gr.Error("Selection has expired. Run the search again.") from exc
    progress(0.05, desc="Preparing selected YouTube video…")
    return add_known_song(
        catalog, item["title"], f"Uploader: {item['channel']}", "youtube",
        lyric_title=lyric_title_for_result(item["title"], item.get("search_title", "")),
        fallback_youtube_url=item["url"],
        progress=progress,
    )


UI_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
)
UI_OPTIONS = {"theme": UI_THEME, "css": APP_CSS}
# Gradio 6 moved theme/css from Blocks(...) to launch(...).
STYLE_ON_BLOCKS = "theme" in signature(gr.Blocks).parameters

with gr.Blocks(
    title="Semantic Cover Search",
    **(UI_OPTIONS if STYLE_ON_BLOCKS else {}),
) as demo:
    catalog_state = gr.State(value=SongIndex)
    youtube_results_state = gr.State(value=[])
    recording_results_state = gr.State(value=[])
    gr.HTML(value=HERO_HTML)

    with gr.Tab("Discover covers on YouTube"):
        gr.Markdown(
            "Discover cover versions by song title. Preview results and open "
            "the original videos, then add selected recordings to your catalog.",
            elem_classes=["scs-intro"],
        )
        public_title = gr.Textbox(label="Song title", placeholder="e.g. Hallelujah")
        public_search = gr.Button(
            "Search covers on YouTube", variant="primary", elem_classes=["scs-primary"]
        )
        public_status = gr.Markdown()
        public_results = gr.HTML(
            value='<div class="scs-empty">Search for a song to explore recordings.</div>',
            label="YouTube results",
        )
        result_selection = gr.Dropdown(
            label="Choose a video to add to your local catalog",
            choices=[],
            interactive=True,
            visible=False,
        )
        add_selected_button = gr.Button(
            "Add selected video to local catalog", visible=False,
            elem_classes=["scs-secondary"]
        )
        selected_status = gr.Markdown(elem_classes=["scs-status"])
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
            "Collect recordings and compare their words. We try reference lyrics "
            "first, then local transcription if needed. The catalog resets on restart.",
            elem_classes=["scs-intro"],
        )
        with gr.Row():
            with gr.Column(elem_classes=["scs-panel"]):
                gr.Markdown("### Upload a song", elem_classes=["scs-section-title"])
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
                add_file_button = gr.Button(
                    "Add song to catalog", variant="primary", elem_classes=["scs-primary"]
                )
            with gr.Column(elem_classes=["scs-panel"]):
                gr.Markdown(
                    "### Add a song from YouTube", elem_classes=["scs-section-title"]
                )
                add_url = gr.Textbox(label="YouTube URL")
                add_url_button = gr.Button(
                    "Add link to catalog", elem_classes=["scs-secondary"]
                )
        with gr.Accordion("Search by singing or humming · experimental", open=False):
            gr.Markdown(
                "Record a short phrase to search YouTube by recognized words, "
                "or compare its musical pattern with analyzed catalog recordings. "
                "YouTube searches are lyric-based; melody matching is local only."
            )
            query_file = gr.Audio(
                type="filepath",
                sources=["microphone"],
                label="Record a sung or spoken fragment",
            )
            record_search_button = gr.Button(
                "Search YouTube using recognized words", elem_classes=["scs-secondary"]
            )
            recording_text = gr.Textbox(
                label="Words recognized from your recording (may be inaccurate)",
                lines=3,
                interactive=False,
            )
            gr.Markdown("#### Compare against your catalog")
            top_k = gr.Slider(1, 10, value=5, step=1, label="Local top results")
            search_file_button = gr.Button(
                "Compare recording with catalog lyrics", elem_classes=["scs-secondary"]
            )
            results_table = gr.Dataframe(
                headers=["Rank", "Title", "Performer / uploader", "Similarity", "Source"],
                datatype=["number", "str", "str", "str", "str"],
                interactive=False,
                label="Closest lyric matches in local catalog",
            )
            search_file_button.click(
                search_upload,
                inputs=[query_file, top_k, catalog_state],
                outputs=results_table,
                show_progress="minimal",
            )
            gr.Markdown("#### YouTube candidates from recognized words")
            recording_status = gr.Markdown()
            recording_results = gr.HTML(
                value='<div class="scs-empty">Record to search for candidates.</div>'
            )
            recording_selection = gr.Dropdown(
                label="Choose a candidate to add",
                choices=[], interactive=True, visible=False,
            )
            add_recording_button = gr.Button(
                "Add selected recording candidate to catalog",
                visible=False, elem_classes=["scs-secondary"],
            )
            recording_add_status = gr.Markdown(elem_classes=["scs-status"])


        add_status = gr.Markdown(elem_classes=["scs-status"])
        with gr.Accordion("Lyrics or automatic transcription · latest song", open=False):
            lyrics_view = gr.HTML(value=render_transcription(None))
        catalog_table = gr.Dataframe(
            headers=["Title", "Performer / uploader", "Source", "Searchable"],
            datatype=["str", "str", "str", "str"],
            interactive=False,
            label="Session catalog",
            elem_classes=["scs-dataframe"],
        )
        gr.Markdown(
            "### Find songs with similar lyrics", elem_classes=["scs-section-title"]
        )
        gr.Markdown(
            "Compare every available song pair using shared lyric phrases and "
            "cached semantic embeddings. No further audio processing is needed.",
            elem_classes=["scs-intro"],
        )
        comparison_top = gr.Slider(
            minimum=1, maximum=50, value=10, step=1, label="Show top song pairs"
        )
        compare_button = gr.Button(
            "Compare all catalog lyrics", variant="primary", elem_classes=["scs-primary"]
        )
        comparison_status = gr.Markdown()
        comparison_results = gr.HTML(
            value='<p style="opacity:.75;">Add at least two songs, then compare.</p>',
            label="Similar lyrics across songs",
        )
        compare_button.click(
            compare_catalog_lyrics,
            inputs=[catalog_state, comparison_top],
            outputs=[comparison_status, comparison_results],
            show_progress="minimal",
        )
        record_search_button.click(
            search_recording_on_youtube,
            inputs=[query_file],
            outputs=[
                recording_status, recording_results, recording_results_state,
                recording_selection, add_recording_button, recording_text,
            ],
            show_progress="minimal",
        )
        add_recording_button.click(
            add_selected_video,
            inputs=[recording_selection, recording_results_state, catalog_state],
            outputs=[recording_add_status, catalog_table, catalog_state, lyrics_view],
            show_progress="full",
            show_progress_on=[recording_add_status],
        )
        # Use Gradio's single native progress indicator for each import.
        # Restrict its overlay to the status component: never cover the
        # catalog and lyric panel or stack it on a second loading message.
        add_selected_button.click(
            add_selected_video,
            inputs=[result_selection, youtube_results_state, catalog_state],
            outputs=[selected_status, catalog_table, catalog_state, lyrics_view],
            show_progress="full",
            show_progress_on=[selected_status],
        )
        add_file_button.click(
            add_upload,
            inputs=[add_file, add_title, add_artist, catalog_state],
            outputs=[add_status, catalog_table, catalog_state, lyrics_view],
            show_progress="full",
            show_progress_on=[add_status],
        )
        add_url_button.click(
            add_youtube,
            inputs=[add_url, catalog_state],
            outputs=[add_status, catalog_table, catalog_state, lyrics_view],
            show_progress="full",
            show_progress_on=[add_status],
        )


if __name__ == "__main__":
    demo.launch(**({} if STYLE_ON_BLOCKS else UI_OPTIONS))
