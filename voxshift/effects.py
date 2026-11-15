"""Small, dependency-free audio effects used on the synthesized voice."""

from __future__ import annotations

import numpy as np


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resampler (good enough for speech)."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if src_rate == dst_rate or len(audio) == 0:
        return audio
    n_out = int(round(len(audio) * dst_rate / src_rate))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def time_stretch(audio: np.ndarray, factor: float, sample_rate: int) -> np.ndarray:
    """Change duration by `factor` (>1 longer) without changing pitch (WSOLA)."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if abs(factor - 1.0) < 1e-3 or len(audio) == 0:
        return audio
    win = int(sample_rate * 0.04)
    hop_out = win // 2
    hop_in = hop_out / factor
    tol = win // 4
    window = np.hanning(win).astype(np.float32)
    n_frames = max(1, int((len(audio) - win) / hop_in) + 1)
    padded = np.concatenate([np.zeros(tol, np.float32), audio, np.zeros(win + 2 * tol, np.float32)])
    out = np.zeros(n_frames * hop_out + win, dtype=np.float32)
    norm = np.zeros_like(out)
    prev = 0  # input position (in padded) of the previous frame
    for i in range(n_frames):
        nominal = int(round(i * hop_in)) + tol
        if i == 0:
            pos = nominal
        else:
            # Pick the offset whose start best continues the previous frame's waveform.
            natural = padded[prev + hop_out : prev + hop_out + win]
            lo, hi = nominal - tol, nominal + tol
            region = padded[lo : hi + win]
            corr = np.correlate(region, natural, mode="valid")
            pos = lo + int(np.argmax(corr))
        b = i * hop_out
        out[b : b + win] += padded[pos : pos + win] * window
        norm[b : b + win] += window
        prev = pos
    norm[norm < 1e-6] = 1.0
    return out / norm


def pitch_shift(audio: np.ndarray, semitones: float, sample_rate: int) -> np.ndarray:
    """Shift pitch by `semitones`, keeping roughly the same duration."""
    if abs(semitones) < 0.05:
        return np.asarray(audio, dtype=np.float32)
    ratio = 2.0 ** (semitones / 12.0)
    stretched = time_stretch(audio, ratio, sample_rate)
    return resample(stretched, int(round(sample_rate * ratio)), sample_rate)


def robot(audio: np.ndarray, sample_rate: int, freq: float = 50.0) -> np.ndarray:
    t = np.arange(len(audio), dtype=np.float32) / sample_rate
    return (audio * np.sign(np.sin(2 * np.pi * freq * t))).astype(np.float32)


def normalize(audio: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if m < 1e-6:
        return audio
    return (audio * (peak / m)).astype(np.float32)


class StreamingPitchShifter:
    """Low-latency pitch shifter for live audio (two-tap rotating delay line).

    Feed it consecutive blocks; each call returns a block of the same length.
    Latency is about one window (~40 ms).
    """

    def __init__(self, sample_rate: int, semitones: float = 0.0, window_ms: float = 40.0):
        self.window = int(sample_rate * window_ms / 1000)
        self.sample_rate = sample_rate
        self._hist = np.zeros(self.window * 2 + 2, dtype=np.float32)
        self._phase = 0.0
        self._robot_t = 0
        self.set_semitones(semitones)

    def set_semitones(self, semitones: float) -> None:
        self.semitones = float(semitones)
        self.ratio = 2.0 ** (self.semitones / 12.0)

    def process(self, block: np.ndarray, robot_voice: bool = False) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        n = len(block)
        work = np.concatenate([self._hist, block])
        self._hist = work[-len(self._hist) :]
        if abs(self.semitones) < 0.05:
            out = block.copy()
        else:
            W = self.window
            step = (1.0 - self.ratio) / W
            phases = (self._phase + step * np.arange(1, n + 1)) % 1.0
            self._phase = float(phases[-1])
            base = len(work) - n + np.arange(n)
            out = np.zeros(n, dtype=np.float32)
            for offset in (0.0, 0.5):
                p = (phases + offset) % 1.0
                pos = base - p * W - 1
                i0 = np.floor(pos).astype(np.int64)
                frac = (pos - i0).astype(np.float32)
                tap = work[i0] * (1 - frac) + work[i0 + 1] * frac
                out += tap * np.sin(np.pi * p).astype(np.float32) ** 2
        if robot_voice:
            t = (self._robot_t + np.arange(n)) / self.sample_rate
            out = out * np.sign(np.sin(2 * np.pi * 50.0 * t)).astype(np.float32)
        self._robot_t += n
        return out


EFFECTS = ["none", "robot", "radio"]


def apply_effects(
    audio: np.ndarray,
    sample_rate: int,
    pitch: float = 0.0,
    effect: str = "none",
) -> np.ndarray:
    audio = pitch_shift(audio, pitch, sample_rate)
    if effect == "robot":
        audio = robot(audio, sample_rate)
    elif effect == "radio":
        # crude band-pass: remove lows by differencing, soften highs by averaging
        audio = np.diff(audio, prepend=audio[:1]) * 3.0
        audio = np.convolve(audio, np.ones(4, dtype=np.float32) / 4, mode="same")
        audio = np.tanh(audio * 2.0)
    return normalize(audio)
