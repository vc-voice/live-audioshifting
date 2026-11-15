"""Text-to-speech with Piper voices (fast, offline, many languages and accents)."""

from __future__ import annotations

import threading

import numpy as np

from .voices import ensure_voice, load_catalog


class TextToSpeech:
    def __init__(self, voice_key: str):
        self._lock = threading.Lock()
        self._catalog = load_catalog()
        self._voice = None
        self.voice_key = ""
        self.language = "en"
        self.set_voice(voice_key)

    def set_voice(self, voice_key: str) -> None:
        if voice_key == self.voice_key:
            return
        from piper import PiperVoice

        if voice_key not in self._catalog:
            raise ValueError(f"Unknown voice '{voice_key}'. Run `voxshift voices` to list them.")
        info = self._catalog[voice_key]
        model = PiperVoice.load(ensure_voice(info))
        with self._lock:
            self._voice = model
            self.voice_key = voice_key
            self.language = info.family

    @property
    def sample_rate(self) -> int:
        return int(self._voice.config.sample_rate)

    def synthesize_stream(self, text: str, speed: float = 1.0):
        """Yield (audio, rate) one sentence at a time so playback can start early."""
        from piper import SynthesisConfig

        cfg = SynthesisConfig(length_scale=1.0 / max(0.25, speed))
        with self._lock:
            voice, rate = self._voice, self.sample_rate
        for chunk in voice.synthesize(text, syn_config=cfg):
            yield chunk.audio_float_array.astype(np.float32).reshape(-1), rate

    def synthesize(self, text: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        rate = self.sample_rate
        chunks = []
        for audio, rate in self.synthesize_stream(text, speed):
            chunks.append(audio)
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return audio, rate
