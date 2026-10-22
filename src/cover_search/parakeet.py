"""Local NVIDIA Parakeet TDT 0.6B v3 speech recognition through sherpa-onnx."""
from functools import lru_cache
from io import BytesIO

import numpy as np

MODEL_REPO = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
MODEL_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")


@lru_cache(maxsize=1)
def _recognizer():
    import sherpa_onnx
    from huggingface_hub import snapshot_download

    directory = snapshot_download(repo_id=MODEL_REPO, allow_patterns=list(MODEL_FILES))
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=f"{directory}/encoder.int8.onnx",
        decoder=f"{directory}/decoder.int8.onnx",
        joiner=f"{directory}/joiner.int8.onnx",
        tokens=f"{directory}/tokens.txt",
        num_threads=4,
        provider="cpu",
        decoding_method="greedy_search",
        model_type="nemo_transducer",
    )


def decode_audio(audio_bytes: bytes) -> np.ndarray:
    """Decode common media containers to 16 kHz mono float32 PCM using PyAV."""
    import av

    samples = []
    with av.open(BytesIO(audio_bytes), mode="r") as container:
        if not container.streams.audio:
            raise ValueError("No audio stream found in uploaded media.")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
        for frame in container.decode(audio=0):
            for output in resampler.resample(frame):
                samples.append(np.asarray(output.to_ndarray(), dtype=np.float32).reshape(-1))
        for output in resampler.resample(None):
            samples.append(np.asarray(output.to_ndarray(), dtype=np.float32).reshape(-1))
    if not samples:
        raise ValueError("The audio contains no decodable samples.")
    return np.concatenate(samples)


def transcribe_parakeet(audio_bytes: bytes, filename: str = "audio") -> str:
    if not audio_bytes:
        raise ValueError("Audio input is empty.")
    pcm = decode_audio(audio_bytes)
    recognizer = _recognizer()
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, pcm)
    recognizer.decode_stream(stream)
    transcript = " ".join(stream.result.text.split())
    if not transcript:
        raise ValueError(
            "Parakeet produced an empty transcript. "
            "Try clear speech or compare with ASR_ENGINE=whisper."
        )
    return transcript
