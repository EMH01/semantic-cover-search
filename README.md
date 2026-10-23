# Semantic Cover Search

A local-first research prototype for discovering song covers on YouTube and experimenting with retrieval from sung phrases. The current implementation **does not identify arbitrary songs by melody or search the entire internet from a recording**.

## Features

- **Discover covers on YouTube:** search YouTube for a song title plus "cover" using `yt-dlp`. Results show video title, channel and watch link. Search metadata is not confirmation that a video is a cover or that the uploader is the performing artist.
- **Local catalog:** add a result from YouTube search, paste a YouTube URL, or upload an audio file. Uploaded MP3 tags can prefill an editable song title and performing artist. YouTube URLs are resolved to public title/uploader metadata via oEmbed.
- **Lyrics-first, ASR fallback:** retrieve title-matched reference lyrics through the LRCLIB no-key API. If none are available **or the provider fails**, transcribe the uploaded audio or a bounded download of the selected YouTube video with the configured speech recognizer (Parakeet TDT on CPU by default). Index recognized text with local embeddings in FAISS. If both routes fail, retain a metadata-only entry rather than indexing invented words. LRCLIB lyrics may represent a different rendition, and speech recognition of singing may be inaccurate.
- **Metadata-only entries:** if neither reference lyrics nor the fallback transcription produces usable text (or embedding fails), retain the song in the catalog without a searchable vector. Status and catalog columns distinguish **reference lyrics**, **automatic transcription**, and **metadata only**.
- **Compare all catalog lyrics:** in **Local catalog**, click **Compare all catalog lyrics** to evaluate every unique song pair with indexed text. Rank pairs by shared consecutive three-word phrases (Jaccard overlap), then local embedding cosine similarity, displayed as separate measurements. Uses already-computed vectors; no repeat API calls, transcription or download. Metadata-only songs are excluded, and the view reports their count. Identical reference lyrics are **not proof** that two uploaded or linked performances are covers.
- **Experimental recording search (in Discover covers on YouTube):** record or upload a short sung/spoken phrase, transcribe its words with Parakeet TDT 0.6B v3 on CPU, and search YouTube for *candidate* videos based on those recognized words. Optionally compare the recording against indexed lyrics in Local catalog. This is not melody recognition or internet-wide acoustic matching.
- **Session-only state:** no database and no persistence. Restarting Gradio clears the catalog; this is intentional for a small demo.

## Architecture

```mermaid
flowchart LR
    A[Song title] --> Y[YouTube search metadata]
    Y --> U[Select cover candidate]
    B[Audio file with tags] --> C[Session catalog]
    D[Manual YouTube URL] --> O[YouTube oEmbed metadata]
    U --> C
    O --> C
    C --> L[LRCLIB lyric lookup]
    L -->|Matched lyrics| E[Multilingual embedding]
    L -->|Unavailable or no match| ASR[Audio ASR fallback]
    ASR -->|Words detected| E
    ASR -->|No words or download failure| M[Metadata-only record]
    E --> F[FAISS vector index]
    F --> Cmp[Compare all catalog lyric pairs]
    Q[Recorded sung phrase] --> P[Local Parakeet ASR]
    P --> YQ[YouTube lyric-phrase search]
    P --> QE[Optional local query embedding]
    QE --> F
    F --> R[Closest lyric matches]
```

**Important:** Neither YouTube metadata nor LRCLIB guarantees that a result is a genuine cover or identifies an unknown song. Parakeet was designed for speech recognition; singing, instrumentals and humming are unreliable. Comparisons against whole lyrics can be weak for very short recorded excerpts. No melodic fingerprinting, chroma/DTW matching, or internet-wide audio identification has been implemented.

## Installation

Use Python 3.11 or 3.12 and a per-project virtual environment.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

The app runs at `http://127.0.0.1:7860` by default. The Gradio interface uses a responsive light theme, compact YouTube cards with thumbnails, and paired lyric-comparison cards. The first **singing-search** request can download the Parakeet model (approximately 670 MB) if it is not cached. If LRCLIB provides lyrics, adding a song does **not** download its YouTube audio or run full-song transcription. Otherwise the app attempts one size/duration-limited download and speech transcription. This fallback can take significantly longer on CPU.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

Tests mock outside services and do not prove real YouTube or LRCLIB availability, nor melodic/lyric matching accuracy.

## Privacy, provider limitations and rights

- YouTube search uses third-party extraction. Depending on the version, `yt-dlp` may need a JavaScript runtime for complete YouTube support. See the [official setup guide](https://github.com/yt-dlp/yt-dlp/wiki/EJS).
- [LRCLIB](https://lrclib.net/docs) provides a no-key API, with published rate limits and identification requirements. This demo fetches lyrics on demand rather than redistributing a local database. Song lyrics may be copyrighted independently of the API/software license: **check lyric display/indexing rights before public deployment**, even when data comes from an API. Each lyric panel links to the source. Other no-key services such as lyrics.ovh scrape third-party lyric pages and are not enabled as a fallback because their data provenance/reuse terms do not provide a clear basis for a public showcase.
- Uploaded file metadata is processed locally. The singing recording is processed locally by Parakeet/Whisper; known song titles can be sent to LRCLIB; video titles and URLs can be sent to YouTube services.
- The catalog stays in Gradio's server/session memory only. Audio used for fallback is processed transiently and is not stored in a project database.
- An old `.local_data/catalog.sqlite3` created by previous development versions is no longer used. `.local_data/` stays in `.gitignore` to prevent accidental commits.

## Potential Enhancements

- Melody-oriented song identification using a licensed reference corpus, robust pitch/chroma representations and sequence alignment.
- Segment-level lyric indexing to improve short-fragment retrieval and distinguish partial shared verses from generic semantic proximity.
- Test sets of original recordings, covers and unrelated songs, with Recall@K and ranking quality metrics.
- Rights-cleared, user-facing lyric references for a public deployment.
