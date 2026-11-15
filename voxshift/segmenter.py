"""Split a live microphone stream into utterances using an adaptive energy gate.

The segmenter is fed small blocks of float32 mono audio. When it decides that an
utterance has finished (speech followed by enough silence, or the maximum
length was reached) it returns the whole utterance as one numpy array.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class UtteranceSegmenter:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        silence_ms: int = 600,
        min_speech_ms: int = 250,
        max_utterance_s: float = 8.0,
        pre_roll_ms: int = 300,
        sensitivity: float = 3.0,
        min_threshold: float = 0.006,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_len = int(sample_rate * frame_ms / 1000)
        self.silence_frames = max(1, silence_ms // frame_ms)
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.max_frames = int(max_utterance_s * 1000 / frame_ms)
        self.sensitivity = sensitivity
        self.min_threshold = min_threshold

        self._pending = np.zeros(0, dtype=np.float32)
        self._pre_roll: deque[np.ndarray] = deque(maxlen=max(1, pre_roll_ms // frame_ms))
        self._frames: list[np.ndarray] = []
        self._speech_frames = 0
        self._silent_run = 0
        self._noise_floor = 0.002
        self.level = 0.0  # last frame RMS, handy for a level meter

    @property
    def in_speech(self) -> bool:
        return bool(self._frames)

    @property
    def threshold(self) -> float:
        return max(self.min_threshold, self._noise_floor * self.sensitivity)

    def reset(self) -> None:
        self._pending = np.zeros(0, dtype=np.float32)
        self._pre_roll.clear()
        self._frames = []
        self._speech_frames = 0
        self._silent_run = 0

    def feed(self, block: np.ndarray) -> list[np.ndarray]:
        """Feed audio; return any utterances that completed during this block."""
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        self._pending = np.concatenate([self._pending, block])
        done: list[np.ndarray] = []
        while len(self._pending) >= self.frame_len:
            frame = self._pending[: self.frame_len]
            self._pending = self._pending[self.frame_len :]
            out = self._process_frame(frame)
            if out is not None:
                done.append(out)
        return done

    def flush(self) -> np.ndarray | None:
        """Return whatever speech is buffered (e.g. when stopping)."""
        out = self._finish() if self._speech_frames >= self.min_speech_frames else None
        self.reset()
        return out

    def _process_frame(self, frame: np.ndarray) -> np.ndarray | None:
        rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
        self.level = rms
        is_speech = rms > self.threshold

        if not self._frames:
            if is_speech:
                self._frames = list(self._pre_roll) + [frame]
                self._pre_roll.clear()
                self._speech_frames = 1
                self._silent_run = 0
            else:
                # Track background noise slowly while nobody is talking.
                self._noise_floor = 0.95 * self._noise_floor + 0.05 * rms
                self._pre_roll.append(frame)
            return None

        self._frames.append(frame)
        if is_speech:
            self._speech_frames += 1
            self._silent_run = 0
        else:
            self._silent_run += 1

        if len(self._frames) >= self.max_frames:
            return self._finish()
        if self._silent_run >= self.silence_frames:
            if self._speech_frames >= self.min_speech_frames:
                return self._finish()
            self.reset_utterance()  # just a click or a breath
        return None

    def reset_utterance(self) -> None:
        self._frames = []
        self._speech_frames = 0
        self._silent_run = 0

    def _finish(self) -> np.ndarray:
        # Trim most of the trailing silence but keep a short tail.
        keep_tail = min(self._silent_run, 5)
        frames = self._frames[: len(self._frames) - self._silent_run + keep_tail]
        audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
        self.reset_utterance()
        return audio
