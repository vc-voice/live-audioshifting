"""Speech-to-text with faster-whisper (open source, runs offline on CPU or GPU)."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import numpy as np

# Rough CPU timings for a short sentence on a 4-core laptop:
#   tiny ~0.7 s, base ~1.8 s, small ~5 s   (auto language detection doubles these)
MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3", "distil-large-v3"]

# Re-check the spoken language every N utterances when auto-detecting.
REDETECT_EVERY = 6


@dataclass
class Transcript:
    text: str
    language: str
    probability: float


class SpeechToText:
    def __init__(self, model_size: str = "tiny", device: str = "auto", compute_type: str = "default"):
        from faster_whisper import WhisperModel

        if device == "auto":
            device = _best_device()
        if compute_type == "default":
            compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type, cpu_threads=os.cpu_count() or 4
        )
        self.model_size = model_size
        self.device = device
        self._lock = threading.Lock()
        self._sticky_language: str | None = None
        self._since_detect = 0

    def warm_up(self) -> None:
        """Run once on silence so the first real utterance isn't slow."""
        self.transcribe(np.zeros(16000, dtype=np.float32), "en")

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> Transcript:
        """`audio` is float32 mono at 16 kHz. `language=None` means auto-detect.

        Auto-detection costs a second encoder pass, so the detected language is
        reused for the next few utterances and only re-checked periodically.
        """
        hint = language
        if hint is None and self._sticky_language and self._since_detect < REDETECT_EVERY:
            hint = self._sticky_language
        with self._lock:
            segments, info = self.model.transcribe(
                audio,
                language=hint,
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
        if language is None:
            if hint is None:
                self._sticky_language = info.language
                self._since_detect = 0
            else:
                self._since_detect += 1
        return Transcript(text=text, language=info.language, probability=info.language_probability)


def _best_device() -> str:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"
