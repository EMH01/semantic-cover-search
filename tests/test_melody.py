"""Synthetic melody regressions: no downloaded songs, models or external services."""
import io
import wave

import numpy as np
import pytest

from cover_search.index import SongIndex
from cover_search.melody import (
    HOP,
    SAMPLE_RATE,
    MelodyError,
    extract_chroma,
    melody_match,
)
from cover_search.models import Song


def _notes(notes):
    output = np.zeros((len(notes), 12), dtype=np.float32)
    for frame, note in enumerate(notes):
        output[frame, note % 12] = 1
    return output


def test_melody_finds_a_query_inside_longer_reference():
    notes = [0, 2, 4, 5, 7, 9, 7, 5, 4, 2, 0, 11]
    reference = _notes([11] * 20 + notes + [6] * 20)
    score, offset, shift = melody_match(_notes(notes), reference)
    # The shared phrase is still located despite different static intros.
    assert score > 0.75
    assert offset == pytest.approx(20 * HOP / SAMPLE_RATE)
    assert shift == 0


def test_melody_search_accepts_transposed_humming():
    notes = [0, 2, 4, 5, 7, 9, 7, 5, 4, 2, 0, 11]
    reference = _notes(notes)
    query = _notes([(pitch + 3) % 12 for pitch in notes])
    score, _offset, shift = melody_match(query, reference)
    assert score == pytest.approx(1.0, abs=1e-5)
    assert shift == 9


def test_song_index_stores_melodies_independently_of_lyrics():
    catalog = SongIndex()
    first = Song("First", "A", "", text_origin="none")
    second = Song("Second", "B", "lyric words", text_origin="lrclib")
    catalog.add(first)
    catalog.add(second, np.array([1.0, 0.0]))
    notes = _notes([0, 2, 4, 5, 7, 9, 7, 5, 4, 2, 0, 11])
    catalog.index_melody(first, notes)
    catalog.index_melody(second, _notes([6] * 12))
    assert catalog.melody_count == 2
    assert catalog.indexed_count == 1
    matches = catalog.search_melody(notes, top_k=2)
    assert matches[0][0] == first
    assert matches[0][1] == pytest.approx(1.0)


def test_index_rejects_unknown_song_or_invalid_features():
    catalog = SongIndex()
    unknown = Song("Unknown", "Artist", "")
    with pytest.raises(ValueError, match="Add this song"):
        catalog.index_melody(unknown, _notes([1] * 9))
    catalog.add(unknown)
    with pytest.raises(MelodyError, match="pitch-class"):
        catalog.index_melody(unknown, np.zeros((5, 11)))


def test_no_reference_raises_comparison_error():
    with pytest.raises(MelodyError, match="Not enough"):
        melody_match(_notes([0, 1, 2]), _notes([0, 1, 2]))


def test_static_reference_cannot_be_used_as_melodic_evidence():
    query = _notes([0, 1, 4, 6, 7, 10, 7, 6, 4, 1, 0, 1])
    other = _notes([9] * 35)
    with pytest.raises(MelodyError, match="Not enough changing tonal"):
        melody_match(query, other)


def test_unrelated_flat_spectra_are_not_99_percent_melody_matches():
    # Reproduces the previous failure: normalized broad-spectrum chroma
    # can give >99% raw cosine for unrelated random tonal sequences.
    def flattened_chroma(frames: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        samples = 1.0 + 0.04 * rng.normal(size=(frames, 12))
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        return samples.astype(np.float32)

    first = flattened_chroma(125, 44)
    second = flattened_chroma(350, 56)
    # Old algorithm using raw chroma cosine would score approximately 0.999.
    naive_frame_cosine = np.mean(np.sum(first * second[:125], axis=1))
    assert naive_frame_cosine > 0.99
    score, _, _ = melody_match(first, second)
    assert 0 <= score < 0.35


def test_same_notes_in_different_order_do_not_look_like_same_tune():
    # Both pieces use exactly the same pitches with the same frequencies,
    # yet their changing sequences are independent.
    rng = np.random.default_rng(88)
    material = np.tile(np.arange(12), 10)
    first = _notes(rng.permutation(material))
    second = _notes(
        np.concatenate([rng.permutation(material), rng.permutation(material)])
    )
    score, _, _ = melody_match(first, second)
    assert score < 0.35


def test_same_sequence_remains_high_after_background_centering():
    notes = [0, 2, 4, 5, 7, 9, 7, 5, 4, 2, 0, 11]
    reference = _notes(notes * 9)
    query = reference[:60].copy()
    score, _, _ = melody_match(query, reference)
    assert score > 0.95


def _wav_tone(frequency: int = 440, seconds: int = 4) -> bytes:
    time = np.arange(SAMPLE_RATE * seconds) / SAMPLE_RATE
    samples = (0.4 * np.sin(2 * np.pi * frequency * time) * 32767).astype(
        np.int16
    )
    destination = io.BytesIO()
    with wave.open(destination, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(samples.tobytes())
    return destination.getvalue()


def test_chroma_extraction_from_synthetic_melody_audio():
    pytest.importorskip("av")
    chroma = extract_chroma(_wav_tone())
    assert chroma.ndim == 2 and chroma.shape[1] == 12
    assert chroma.shape[0] >= 7
    # A4 is pitch class 9 in our MIDI-to-pitch-class convention.
    assert int(np.argmax(chroma.mean(axis=0))) == 9


def test_silent_audio_is_rejected_for_melody_matching():
    pytest.importorskip("av")
    samples = np.zeros(SAMPLE_RATE * 4, dtype=np.int16)
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(samples.tobytes())
    with pytest.raises(MelodyError, match="silent"):
        extract_chroma(output.getvalue())
