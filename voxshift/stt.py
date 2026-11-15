"""Speech-to-text with faster-whisper (open source, runs offline on CPU or GPU)."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3", "distil-large-v3"]


@dataclass
class Transcript:
    text: str
    language: str
    probability: float


class SpeechToText:
    def __init__(self, model_size: str = "base", device: str = "auto", compute_type: str = "default"):
        from faster_whisper import WhisperModel

        if device == "auto":
            device = _best_device()
        if compute_type == "default":
            compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.model_size = model_size
        self.device = device
        self._lock = threading.Lock()

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> Transcript:
        """`audio` is float32 mono at 16 kHz. `language=None` means auto-detect."""
        with self._lock:
            segments, info = self.model.transcribe(
                audio,
                language=language or None,
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
        return Transcript(text=text, language=info.language, probability=info.language_probability)


def _best_device() -> str:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"
