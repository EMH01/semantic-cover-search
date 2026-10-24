"""Experimental pitch-class matching for short hums against local audio references.

No acoustic fingerprinting or internet-wide song identification is claimed.
The reference's accompaniment can mask the melody, so scores need validation.
"""
from __future__ import annotations

import io

import numpy as np

SAMPLE_RATE = 11025
FFT_SIZE = 4096
HOP = 2205  # Five chroma frames per second
MAX_REFERENCE_SECONDS = 180
MAX_QUERY_SECONDS = 35


class MelodyError(ValueError):
    """Audio could not be decoded into useful musical features."""


def decode_mono(data: bytes, max_seconds: int) -> np.ndarray:
    """Bound audio duration and memory before any spectral processing."""
    if not data or len(data) > 20 * 1024 * 1024:
        raise MelodyError("Provide an audio file between 1 byte and 20 MB.")
    try:
        import av

        collected = []
        length = 0
        limit = SAMPLE_RATE * max_seconds
        with av.open(io.BytesIO(data)) as container:
            if not container.streams.audio:
                raise MelodyError("No audio stream was found.")
            resampler = av.audio.resampler.AudioResampler(
                format="fltp", layout="mono", rate=SAMPLE_RATE
            )
            for frame in container.decode(audio=0):
                for converted in resampler.resample(frame):
                    chunk = converted.to_ndarray().reshape(-1).astype(np.float32)
                    need = limit - length
                    if need <= 0:
                        break
                    chunk = chunk[:need]
                    collected.append(chunk)
                    length += chunk.size
                if length >= limit:
                    break
            if length < limit:
                for converted in resampler.resample(None):
                    chunk = converted.to_ndarray().reshape(-1).astype(np.float32)
                    collected.append(chunk[: limit - length])
                    length += min(chunk.size, limit - length)
                    if length >= limit:
                        break
        if not collected:
            raise MelodyError("The audio stream is empty.")
        output = np.concatenate(collected)
    except MelodyError:
        raise
    except (OSError, ValueError, RuntimeError, AttributeError) as exc:
        raise MelodyError("The audio could not be decoded.") from exc
    if output.size < SAMPLE_RATE * 2:
        raise MelodyError("Provide at least 2 seconds of audio.")
    return np.nan_to_num(output, copy=False)


def extract_chroma(data: bytes, max_seconds: int = MAX_REFERENCE_SECONDS) -> np.ndarray:
    """Convert audio to normalized 12-bin pitch-class features at 5 Hz.

    Uses FFT spectral bins; no external model or microphone transcript is required.
    Polyphonic audio has a mixture of pitches rather than a clean vocal melody.
    """
    waveform = decode_mono(data, max_seconds)
    waveform = waveform - np.mean(waveform)
    if float(np.max(np.abs(waveform))) < 1e-4:
        raise MelodyError("The recording is silent.")
    if waveform.size < FFT_SIZE:
        waveform = np.pad(waveform, (0, FFT_SIZE - waveform.size))

    starts = np.arange(0, waveform.size - FFT_SIZE + 1, HOP)
    windowed = (
        np.lib.stride_tricks.sliding_window_view(waveform, FFT_SIZE)[starts]
        * np.hanning(FFT_SIZE).astype(np.float32)
    )
    spectra = np.abs(np.fft.rfft(windowed, axis=1)).astype(np.float32)
    frequencies = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
    within = (frequencies >= 100) & (frequencies <= 1500)
    chosen_frequencies = frequencies[within]
    pitch_classes = (
        np.rint(69 + 12 * np.log2(chosen_frequencies / 440)).astype(int) % 12
    )
    energy = spectra[:, within]
    pitch = np.zeros((energy.shape[0], 12), dtype=np.float32)
    for note in range(12):
        pitch[:, note] = np.sum(energy[:, pitch_classes == note], axis=1)
    # Compress dominant harmonics, retaining pitch-class texture.
    pitch = np.log1p(pitch)
    norm = np.linalg.norm(pitch, axis=1)
    voiced = norm > max(float(np.max(norm)) * 0.08, 1e-5)
    if np.count_nonzero(voiced) < 7:
        raise MelodyError("Not enough musical information was detected.")
    pitch[~voiced] = 0
    pitch[voiced] /= norm[voiced, None]
    return pitch


def melody_match(
    query: np.ndarray, reference: np.ndarray
) -> tuple[float, float, int]:
    """Find strongest subsequence chroma alignment with tempo/pitch tolerance.

    Returns (alignment score 0..1, offset seconds, pitch-class shift).
    The score is NOT a calibrated probability of identifying a song.
    """
    a = np.asarray(query, dtype=np.float32)
    b = np.asarray(reference, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 12 or b.shape[1] != 12:
        raise MelodyError("Expected pitch-class matrices with 12 columns.")
    if a.shape[0] < 7 or b.shape[0] < 7:
        raise MelodyError("Not enough pitch-class frames to compare.")
    best_score, best_offset, best_shift = -1.0, 0.0, 0
    # Mild variations of performance speed, plus transposed singing.
    for tempo in (0.8, 1.0, 1.2):
        query_length = max(7, int(round(a.shape[0] * tempo)))
        if query_length > b.shape[0]:
            continue
        at = np.linspace(0, a.shape[0] - 1, query_length)
        original = np.arange(a.shape[0])
        stretched = np.stack(
            [np.interp(at, original, a[:, index]) for index in range(12)], axis=1
        )
        norms = np.linalg.norm(stretched, axis=1)
        valid = norms > 1e-5
        if np.count_nonzero(valid) < 7:
            continue
        stretched[valid] /= norms[valid, None]
        stretched[~valid] = 0
        for shift in range(12):
            candidate = np.roll(stretched, shift, axis=1)
            # Sum correlations over all 12 pitch classes for each start offset.
            sums = np.zeros(b.shape[0] - query_length + 1, dtype=np.float32)
            for note in range(12):
                sums += np.correlate(b[:, note], candidate[:, note], mode="valid")
            # Select the segment with highest average per voiced query frame.
            averages = sums / np.count_nonzero(valid)
            location = int(np.argmax(averages))
            if averages[location] > best_score:
                best_score = float(averages[location])
                best_offset = location * HOP / SAMPLE_RATE
                best_shift = shift
    if best_score < 0:
        raise MelodyError("The query is longer than the available reference audio.")
    return float(np.clip(best_score, 0, 1)), best_offset, best_shift
