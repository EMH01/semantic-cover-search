# Development and troubleshooting notes

This document preserves detailed operational guidance from earlier README revisions. The [main README](../README.md) is the public project overview.

## External services

If a YouTube song is added as **Metadata only** with an LRCLIB connection error and a `DownloadError`, **both independent providers failed**. That does not mean the song lacks lyrics. Check the terminal for the underlying LRCLIB / yt-dlp exception and network/proxy/TLS information. LRCLIB connection failures are retried once, but an outage or network block may persist.

If YouTube warns about the missing JavaScript runtime, this is a **yt-dlp/YouTube extraction warning**, not a Parakeet problem. The Python extra `yt-dlp[default]` installs EJS helper scripts, but does **not** install a JavaScript runtime. Follow the [official yt-dlp EJS guide](https://github.com/yt-dlp/yt-dlp/wiki/EJS). On Windows, Deno can be installed with `winget install -e --id DenoLand.Deno` (system-level dependency, not a Python virtual environment package), then restart the terminal and check `deno --version`. Install the updated Python dependencies inside the project's existing `.venv` using `.\\.venv\\Scripts\\python.exe -m pip install -U -e .`; the `yt-dlp[default]` extra supplies matching EJS scripts. A supported runtime can improve extraction but cannot override video restrictions or guarantee every download.

You can retry a failed **metadata-only** catalog entry by selecting the same video or adding its URL again, without restarting Gradio. If LRCLIB recovers but YouTube does not, lyrics can still be indexed; melody remains *Not analyzed*. Audio extraction is bounded by duration/file-size controls. To isolate external availability from catalog code, upload a locally accessible song audio file.

## Interpreting tonal-pattern scores

The previous implementation compared normalized 12-bin chroma vectors from the **whole mixed audio**. Because most notes contain broad harmonic energy, unrelated recordings could produce almost identical chroma vectors. Searching over time offsets, transpositions and tempo variants could then return misleading ~99% alignment scores.

The experimental matcher now subtracts each recording's stable pitch-class background and compares **changing tonal patterns** instead. If the audio contains no discriminative tonal changes, the metric is **N/A** rather than an artificial high match. It does not isolate the singer or identify a tune from its true melodic contour. A high score means the tonal patterns aligned under the tested transformations, **not** that the songs are confirmed covers. Real-world validation requires an annotated benchmark of same-song/different-song recordings and ranking metrics such as Recall@K. Lyrics matching remains independent.

## Manual validation: catalog queue

1. Start importing a YouTube result that requires lyric lookup, embedding, or audio analysis.
2. Before it finishes, switch to **Local catalog**, upload a different audio file, and click **Add song to catalog**.
3. The second import must **wait**, while the first finishes and updates the session catalog. Both entries should appear afterward; one job must not cancel the other.
4. Repeat with **Add link to catalog** and **Compare all catalog lyrics**. Comparison must run only after queued imports finish. **Discover covers on YouTube** can still search video metadata independently.

The queue is intentionally serial for CPU reliability and to avoid concurrent writes to the session catalog. Queued work is not persistent across browser/server restarts; do not close/restart the running app while testing.

## Manual validation: melody prototype

1. In **Local catalog**, add two uploaded audio files or accessible YouTube recordings. Melody features are extracted automatically during addition, even when LRCLIB supplies lyrics.
2. Verify that the catalog marks **Melody** as *Ready* for accessible analyzed audio. A YouTube URL may display *Not analyzed* if the audio cannot be obtained, is too long/large, or melody extraction fails. Reference lyrics can still be available in those cases.
3. Click **Compare all catalog lyrics**. The same comparison cards now show **Shared lyric phrases**, **Semantic similarity** and **Tonal pattern similarity · experimental**. The last metric should show N/A for pairs missing audio features.
4. Add an MP3 with sung vocals. If Parakeet cannot recognize the lyrics, the song stays **Metadata only** but may still have **Melody: Ready**. This means audio processing worked, not that a transcription succeeded.
5. To compare songs, use **Compare all catalog lyrics** in Local catalog. Session data resets when Gradio restarts.

This is an experimental **temporal chroma-change** baseline, not isolated melody extraction or robust song identification. Previously, raw cosine of broad-spectrum chroma could exceed 99% even for unrelated songs; stable tonal background is now removed before sequence alignment. The corrected number is still uncalibrated, and similar chord progressions or backing instruments can mislead it. The app does not search unindexed songs on YouTube by audio. A future version could use predominant-melody extraction and a rights-cleared reference corpus.

## Download and model diagnostics

Hugging Face progress lines such as `Download complete` or `Reconstruction complete: 0.00B` can reflect cache/download activity; they do not, on their own, prove a transcription failure.

An audio entry labeled **Melody: Ready** indicates successful tonal feature extraction, not successful lyric transcription. A temporary LRCLIB HTTP 503 is a service failure rather than proof that lyrics do not exist.
