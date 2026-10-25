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
from cover_search.melody import MelodyError, extract_chroma
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
        [
            song.title,
            song.artist,
            song.source,
            "Ready" if catalog.has_melody(song) else "Not analyzed",
            names.get(song.text_origin, "Metadata only"),
        ]
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
    existing = next(
        (
            song for song in catalog.songs
            if song.title.casefold() == title.casefold()
            and song.artist.casefold() == artist.casefold()
        ),
        None,
    )
    if existing is not None and existing in catalog.searchable_songs:
        return (
            f"**Already indexed:** {escape(existing.label)}. "
            "This entry already has searchable lyrics.",
            catalog_rows(catalog),
            catalog,
            render_transcription(None, "This song already has indexed lyrics."),
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
    audio_error = ""
    audio_bytes: bytes | None = None
    audio_name = ""

    # One bounded audio acquisition per song. Reuse the same bytes for ASR
    # (only if LRCLIB failed) and for optional melody features (all sources).
    if fallback_audio_path or fallback_youtube_url:
        if progress is not None:
            progress(0.35, desc="Preparing audio for melody analysis…")
        try:
            if fallback_audio_path:
                path = Path(fallback_audio_path)
                audio_bytes, audio_name = read_audio(str(path)), path.name
            else:
                audio = download_audio(fallback_youtube_url)
                audio_bytes, audio_name = audio.data, audio.filename
        except Exception as exc:
            # A blocked/unavailable video cannot prevent lyrics from being indexed.
            audio_error = type(exc).__name__

    if not text and audio_bytes is not None:
        try:
            if progress is not None:
                progress(0.62, desc="Transcribing audio with local ASR…")
            text = transcribe(audio_bytes, audio_name).strip()
            if not text:
                raise ValueError("The recognizer did not detect any words.")
            origin = "asr"
        except Exception as exc:
            fallback_error = type(exc).__name__
            text = ""
    elif not text and audio_error:
        fallback_error = audio_error

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
        if existing is None:
            catalog.add(song, embedding)
        else:
            song = catalog.retry_metadata(song, embedding)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    # Audio from uploads AND YouTube contributes optional melody features.
    # Keep only pitch-class frames; never retain raw media in SongIndex.
    melody_problem = ""
    if audio_bytes is not None:
        try:
            if progress is not None:
                progress(0.9, desc="Extracting melody features from audio…")
            features = extract_chroma(audio_bytes)
            catalog.index_melody(song, features)
        except (MelodyError, ValueError, OSError, RuntimeError, ImportError) as exc:
            melody_problem = type(exc).__name__

    action = "Retried in local catalog" if existing is not None else "Added to local catalog"
    status = f"**{action}:** {escape(song.label)}."
    if catalog.has_melody(song):
        status += " Melody features available for comparisons."
    elif audio_error:
        status += (
            " Melody unavailable: YouTube audio could not be obtained. "
            "Check the terminal for yt-dlp details and the JavaScript runtime."
        )
    elif melody_problem:
        status += " Melody extraction was unavailable; lyric analysis is unaffected."
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
        elif fallback_error and not audio_error:
            status += (
                " Audio transcription failed "
                f"({escape(fallback_error)})."
            )
        if match is not None:
            view = render_lyrics(match)
        else:
            view = render_transcription(display_text or None)

    if lookup_error:
        status += f" Lyrics service issue: {escape(lookup_error)}"
    if progress is not None:
        progress(1.0, desc="Catalog updated")
    return status, catalog_rows(catalog), catalog, view

# Gradio injects Progress from a constructed default; Ruff B008 is not applicable.
def add_upload(
    audio_path: str | None, title: str, artist: str, catalog: SongIndex,
    progress: gr.Progress = gr.Progress()  # noqa: B008,
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
    url: str, catalog: SongIndex, progress: gr.Progress = gr.Progress()  # noqa: B008
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


def render_lyric_comparison_cards(comparisons) -> str:
    """Three independent metrics; missing audio/text is never scored as zero."""
    if not comparisons:
        return (
            '<div class="scs-empty">No comparable song pairs yet. '
            'Add two songs with indexed lyrics or uploaded audio.</div>'
        )
    cards = []
    for rank, pair in enumerate(comparisons, start=1):
        left, right = escape(pair.first.label), escape(pair.second.label)
        first_source = (
            "LRCLIB" if pair.first.text_origin == "lrclib"
            else "ASR" if pair.first.text_origin == "asr" else "No lyrics"
        )
        second_source = (
            "LRCLIB" if pair.second.text_origin == "lrclib"
            else "ASR" if pair.second.text_origin == "asr" else "No lyrics"
        )
        metrics = [
            ("Shared lyric phrases", pair.phrase_overlap, "scs-meter-fill"),
            ("Semantic similarity ⓘ", pair.semantic_similarity, "scs-meter-fill-alt"),
            ("Melody similarity · experimental", pair.melody_similarity, "scs-meter-fill"),
        ]
        metric_html = []
        for label, value, color in metrics:
            # Cosine is signed; preserve negative values, cap only bar width.
            display = "N/A" if value is None else f"{value * 100:.1f}"
            if value is not None and label == "Shared lyric phrases":
                display += "%"
            fraction = 0 if value is None else max(0.0, min(100.0, value * 100))
            explanation = (
                "N/A: audio was not available for both songs"
                if value is None and "Melody" in label else
                "Not enough indexed lyrics to compare this pair"
                if value is None else
                "Cosine similarity between normalized lyric embeddings, "
                "multiplied by 100. Not a probability."
                if "Semantic" in label else
                "Experimental similarity, not song identification confidence."
            )
            metric_html.append(
                '<div class="scs-metric">'
                f'<div class="scs-metric-label" title="{escape(explanation)}">'
                f'{escape(label)}</div>'
                f'<div class="scs-metric-value">{display}</div>'
                f'<div class="scs-meter"><div class="scs-meter-fill {color}" '
                f'style="width:{fraction:.1f}%"></div></div></div>'
            )
        note = (
            "Identical indexed lyrics do not prove identical performances."
            if pair.exact_text else
            "Lyrics and melody are separate, uncalibrated measurements."
        )
        cards.append(
            '<article class="scs-match">'
            f'<div class="scs-match-head">Comparison {rank:02d}</div>'
            f'<div class="scs-track-name">{left}</div>'
            f'<div class="scs-track-source">Text: {first_source}</div>'
            '<div class="scs-match-divider"></div>'
            f'<div class="scs-track-name">{right}</div>'
            f'<div class="scs-track-source">Text: {second_source}</div>'
            '<div class="scs-metrics">' + "".join(metric_html) + '</div>'
            f'<p class="scs-match-note">{escape(note)}</p>'
            '</article>'
        )
    return '<div class="scs-match-grid">' + "".join(cards) + '</div>'


def compare_catalog_lyrics(catalog: SongIndex, top_pairs: int):
    """Compare lyrics and audio automatically if those features are available."""
    total = len(catalog.songs)
    eligible = len({
        *catalog.searchable_songs,
        *catalog.melody_songs,
    })
    excluded = total - eligible
    pairs = catalog.compare_all()
    if not pairs:
        return (
            f"**{eligible} of {total} songs have text or audio features.** "
            "Add at least two comparable songs. "
            "Audio unavailable from YouTube may leave melody unmeasured.",
            render_lyric_comparison_cards([]),
        )
    displayed = pairs[:max(1, min(int(top_pairs), 50))]
    pair_label = "pair" if len(pairs) == 1 else "pairs"
    song_label = "song" if excluded == 1 else "songs"
    note = (
        f"**{len(pairs)} {pair_label} compared** · "
        f"{catalog.indexed_count} lyric-indexed · "
        f"{catalog.melody_count} audio-analyzed · "
        f"{excluded} metadata-only {song_label} excluded. "
        "Sorted primarily by shared lyric phrases. "
        "Melody similarity is N/A when either recording is unavailable; "
        "scores are not probabilities of a cover match."
    )
    return note, render_lyric_comparison_cards(displayed)


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
    audio_path: str | None, progress: gr.Progress = gr.Progress()  # noqa: B008
):
    """Transcribe a recording and search YouTube for lyric-text candidates.

    This does not identify the recording from its melody or guarantee song ID.
    """
    if not audio_path:
        raise gr.Error("Record a short sung or spoken phrase first.")
    path = Path(audio_path)
    progress(0.12, desc="Recognizing sung or spoken words locally…")
    try:
        recognized = transcribe(read_audio(str(path)), path.name).strip()
    except (ValueError, RuntimeError) as exc:
        raise gr.Error(
            "No recognizable words were extracted. Try singing a clear phrase; "
            "YouTube search here uses words, not melody."
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


def add_confirmed_recording_result(
    selection: str | None,
    videos: list[dict[str, str]],
    catalog: SongIndex,
    progress: gr.Progress = gr.Progress(),  # noqa: B008
):
    """Add a confirmed YouTube recording through the same URL flow as manual paste."""
    if selection is None:
        raise gr.Error("Choose and confirm a YouTube result first.")
    try:
        video = videos[int(selection)]
        url = video["url"]
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        raise gr.Error("Selection has expired. Search again.") from exc
    return add_youtube(url, catalog, progress=progress)


def add_selected_video(
    selection: str | None, videos: list[dict], catalog: SongIndex,
    progress: gr.Progress = gr.Progress()  # noqa: B008,
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

                with gr.Accordion("Find a song by singing · experimental", open=False):
                    gr.Markdown(
                        "Sing a recognizable phrase. We search YouTube using the "
                        "words detected, then you choose the correct video. "
                        "Humming alone is not supported for YouTube discovery.",
                        elem_classes=["scs-intro"],
                    )
                    query_file = gr.Audio(
                        type="filepath",
                        sources=["microphone"],
                        label="Record a sung phrase",
                    )
                    record_search_button = gr.Button(
                        "Find song on YouTube", elem_classes=["scs-secondary"]
                    )
                    recording_text = gr.Textbox(
                        label="Recognized words (may be inaccurate)",
                        lines=2,
                        interactive=False,
                    )
                    recording_status = gr.Markdown()
                    recording_results = gr.HTML(
                        value='<div class="scs-empty">Record a phrase to find candidates.</div>'
                    )
                    recording_selection = gr.Dropdown(
                        label="Confirm the correct YouTube video",
                        choices=[], interactive=True, visible=False,
                    )
                    add_recording_button = gr.Button(
                        "Confirm and add to catalog",
                        visible=False, elem_classes=["scs-primary"],
                    )
                    recording_add_status = gr.Markdown(elem_classes=["scs-status"])
        add_status = gr.Markdown(elem_classes=["scs-status"])
        with gr.Accordion("Lyrics or automatic transcription · latest song", open=False):
            lyrics_view = gr.HTML(value=render_transcription(None))
        catalog_table = gr.Dataframe(
            headers=["Title", "Performer / uploader", "Source", "Melody", "Lyrics"],
            datatype=["str", "str", "str", "str", "str"],
            interactive=False,
            label="Session catalog",
            elem_classes=["scs-dataframe"],
        )
        gr.Markdown(
            "### Find songs with similar lyrics", elem_classes=["scs-section-title"]
        )
        gr.Markdown(
            "Compare shared lyric phrases, semantic meaning and available audio "
            "melodies. Uploaded files and accessible YouTube audio are analyzed "
            "automatically; unavailable recordings display N/A.",
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
            add_confirmed_recording_result,
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
