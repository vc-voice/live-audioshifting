"""Make the generated speech follow *how* you said it.

For every phrase we measure your pitch (F0), how much it moves, whether it rises
or falls at the end (questions, emphasis), your loudness over time and your
speaking pace. The synthesized phrase is then reshaped to match:

* its pitch contour is moved to your pitch level and range, and its ending
  follows your rise or fall,
* its loudness follows your loudness envelope,
* its length follows your speaking rate.

All changes are clamped so the result can only ever sound like you on a slightly
off day, never broken. Everything is plain NumPy and takes a few tens of ms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HOP_S = 0.01
WIN_S = 0.04
F0_MIN, F0_MAX = 60.0, 420.0


def _frames(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def track(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (f0 in semitones re 100 Hz, NaN where unvoiced; frame level in dB), 10 ms hop."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    win, hop = int(sr * WIN_S), int(sr * HOP_S)
    fr = _frames(x, win, hop).astype(np.float64)
    fr -= fr.mean(axis=1, keepdims=True)
    level = 10 * np.log10(np.mean(fr**2, axis=1) + 1e-10)

    # Normalised autocorrelation via FFT.
    nfft = 1 << int(np.ceil(np.log2(2 * win)))
    spec = np.fft.rfft(fr * np.hanning(win), nfft)
    ac = np.fft.irfft(np.abs(spec) ** 2, nfft)[:, :win]
    ac /= ac[:, :1] + 1e-12
    lo, hi = int(sr / F0_MAX), min(win - 2, int(sr / F0_MIN))
    seg = ac[:, lo:hi]
    lag = np.argmax(seg, axis=1) + lo
    peak = seg[np.arange(len(seg)), lag - lo]
    # Parabolic interpolation for sub-sample lag.
    rows = np.arange(len(ac))
    a, b, c = ac[rows, lag - 1], ac[rows, lag], ac[rows, np.minimum(lag + 1, win - 1)]
    denom = a - 2 * b + c
    frac = np.where(np.abs(denom) > 1e-9, 0.5 * (a - c) / denom, 0.0)
    f0 = sr / (lag + np.clip(frac, -0.5, 0.5))

    loud_enough = level > (np.max(level) - 35)
    voiced = (peak > 0.45) & loud_enough
    st = np.where(voiced, 12 * np.log2(f0 / 100.0), np.nan)
    return _clean(st), level


def _clean(st: np.ndarray) -> np.ndarray:
    """Remove octave jumps and isolated voiced blips."""
    st = st.copy()
    v = ~np.isnan(st)
    if v.sum() < 5:
        return st
    med = np.nanmedian(st)
    st[v & (np.abs(st - med) > 9)] = np.nan  # > ~octave away from the median: likely an error
    # 5-frame running median over voiced frames.
    out = st.copy()
    for i in np.flatnonzero(~np.isnan(st)):
        w = st[max(0, i - 2) : i + 3]
        w = w[~np.isnan(w)]
        out[i] = np.median(w) if len(w) >= 3 else np.nan
    return out


@dataclass
class Delivery:
    """How a phrase was spoken."""

    pitch: float  # median, semitones re 100 Hz
    spread: float  # pitch variation (semitones, robust std)
    ending: float  # pitch movement over the last part (semitones; + = rising)
    level_db: float  # loudness of the voiced/speech part
    speech_s: float  # seconds of actual speech (pauses excluded)
    contour: np.ndarray  # per-frame semitones (NaN unvoiced), normalised time
    envelope: np.ndarray  # per-frame dB relative to level_db

    @property
    def ok(self) -> bool:
        return np.isfinite(self.pitch) and np.count_nonzero(~np.isnan(self.contour)) >= 10


def analyze(audio: np.ndarray, sr: int) -> Delivery:
    st, level = track(audio, sr)
    speech = level > (np.max(level) - 30)
    speech_s = max(HOP_S, speech.sum() * HOP_S)
    voiced = st[~np.isnan(st)]
    if len(voiced) >= 5:
        pitch = float(np.median(voiced))
        spread = float(np.subtract(*np.percentile(voiced, [84, 16])) / 2)
    else:
        pitch, spread = float("nan"), 0.0
    ending = _ending(st)
    lvl = float(np.mean(level[speech])) if speech.any() else -60.0
    # Trim leading/trailing silence so contours line up with the words.
    idx = np.flatnonzero(speech)
    a, b = (idx[0], idx[-1] + 1) if len(idx) else (0, len(level))
    return Delivery(pitch, spread, ending, lvl, speech_s, st[a:b], level[a:b] - lvl)


def _ending(st: np.ndarray) -> float:
    idx = np.flatnonzero(~np.isnan(st))
    if len(idx) < 10:
        return 0.0
    tail = idx[int(len(idx) * 0.7) :]
    if len(tail) < 4:
        return 0.0
    slope = np.polyfit(tail * HOP_S, st[tail], 1)[0]  # semitones per second
    return float(np.clip(slope * (tail[-1] - tail[0]) * HOP_S, -8, 8))


def _resize(x: np.ndarray, n: int) -> np.ndarray:
    """Stretch a per-frame curve to n points (NaN-aware)."""
    if n <= 0:
        return np.zeros(0)
    if len(x) == 0:
        return np.full(n, np.nan)
    src = np.linspace(0, 1, len(x))
    dst = np.linspace(0, 1, n)
    v = ~np.isnan(x)
    if v.sum() < 2:
        return np.full(n, np.nan)
    out = np.interp(dst, src[v], x[v])
    # keep NaN where the source is mostly unvoiced
    mask = np.interp(dst, src, v.astype(float)) > 0.5
    return np.where(mask, out, np.nan)


def _smooth(x: np.ndarray, frames: int) -> np.ndarray:
    if frames <= 1 or len(x) == 0:
        return x
    k = np.hanning(frames + 2)[1:-1]
    k /= k.sum()
    return np.convolve(np.pad(x, frames, mode="edge"), k, mode="same")[frames:-frames]


# ------------------------------------------------------------------ transform
def _pitch_marks(x: np.ndarray, sr: int, st: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One mark per glottal cycle in voiced parts (every 10 ms elsewhere).

    Returns (mark positions, voiced flag per mark).
    """
    hop = int(sr * HOP_S)
    n = len(x)
    marks, voiced = [], []
    t = 0
    while t < n:
        f = int(min(len(st) - 1, t // hop)) if len(st) else 0
        if len(st) and not np.isnan(st[f]):
            period = int(sr / (100.0 * 2 ** (st[f] / 12)))
            # snap to the strongest peak near the expected position
            lo, hi = max(0, t - period // 4), min(n, t + period // 4 + 1)
            if marks and hi > lo:
                t = lo + int(np.argmax(x[lo:hi]))
            marks.append(t)
            voiced.append(True)
            t += max(period, 20)
        else:
            marks.append(t)
            voiced.append(False)
            t += hop
    return np.asarray(marks), np.asarray(voiced)


def psola(audio: np.ndarray, sr: int, st: np.ndarray, ratio: np.ndarray, stretch: np.ndarray) -> np.ndarray:
    """TD-PSOLA: change pitch by `ratio` and length by `stretch` (both per 10 ms frame).

    Works one pitch cycle at a time, so the vocal-tract resonances (what makes a
    voice sound like *you*) stay where they are; only the pitch moves.
    """
    x = np.asarray(audio, dtype=np.float32)
    hop = int(sr * HOP_S)
    marks, voiced = _pitch_marks(x, sr, st)
    if len(marks) < 3:
        return x
    periods = np.diff(marks, append=marks[-1] + (marks[-1] - marks[-2]))
    periods = np.clip(periods, 20, int(sr / F0_MIN))
    frame_of = lambda pos: int(min(len(ratio) - 1, max(0, pos // hop)))  # noqa: E731

    total = int(len(x) * float(np.mean(stretch))) + 4 * hop
    out = np.zeros(total + 2 * int(sr / F0_MIN), np.float32)
    norm = np.zeros_like(out)
    ts, ta = float(marks[0]), float(marks[0])
    while ta < len(x) and ts < total:
        i = int(np.searchsorted(marks, ta))
        if i >= len(marks) or (i > 0 and ta - marks[i - 1] < marks[i] - ta):
            i -= 1
        i = max(0, min(i, len(marks) - 1))
        p = int(periods[i])
        c = int(marks[i])
        lo, hi = max(0, c - p), min(len(x), c + p)
        grain = x[lo:hi] * np.hanning(2 * p + 1)[lo - (c - p) : hi - (c - p)].astype(np.float32)
        dst = int(round(ts)) - (c - lo)
        if dst >= 0 and dst + len(grain) <= len(out):
            w = np.hanning(2 * p + 1)[lo - (c - p) : hi - (c - p)]
            out[dst : dst + len(grain)] += grain
            norm[dst : dst + len(grain)] += w
        f = frame_of(ta)
        r = ratio[f] if voiced[i] else 1.0
        step = p / max(0.3, r)
        ts += step
        ta += step / max(0.3, stretch[f])
    end = int(ts) + p
    norm = np.maximum(norm, 0.5)  # avoid boosting the quiet edges between grains
    return (out / norm)[: min(end, len(out))]


def match(
    synth: np.ndarray,
    sr: int,
    you: Delivery,
    span: tuple[float, float] = (0.0, 1.0),
    pace: float | None = None,
    strength: float = 1.0,
) -> np.ndarray:
    """Reshape `synth` so its pitch, intonation, loudness and pace follow `you`.

    `span` is which part of your phrase this piece of synthesized audio covers
    (0-1), used when a phrase is spoken as several sentences. `pace` is the
    desired length change (>1 = longer), already clamped by the caller.
    """
    synth = np.asarray(synth, dtype=np.float32).reshape(-1)
    if len(synth) < sr * 0.15 or strength <= 0:
        return synth
    mine = analyze(synth, sr)
    st, level = track(synth, sr)
    n = len(st)

    # Your contour for this span, laid over the synthesized frames that hold speech.
    a = int(round(span[0] * len(you.contour)))
    b = max(a + 1, int(round(span[1] * len(you.contour))))
    your_c = you.contour[a:b]
    your_env = you.envelope[a:b]
    speech = level > (np.max(level) - 30)
    idx = np.flatnonzero(speech)
    s0, s1 = (idx[0], idx[-1] + 1) if len(idx) else (0, n)

    semis = np.zeros(n)
    if you.ok and mine.ok:
        # a) overall pitch level and range
        shift = np.clip(you.pitch - mine.pitch, -9, 9)
        scale = np.clip((you.spread + 0.3) / (mine.spread + 0.3), 0.6, 1.7)
        base = np.where(np.isnan(st), mine.pitch, st)
        target = you.pitch + (base - mine.pitch) * scale
        semis = target - base
        # b) follow your phrase shape (rise/fall), but only softly: the words differ
        yours = _resize(your_c, s1 - s0)
        theirs = st[s0:s1]
        both = ~np.isnan(yours) & ~np.isnan(theirs)
        if both.sum() > 10:
            yr = yours - np.nanmedian(yours)
            tr = (theirs - mine.pitch) * scale
            shape = np.where(both, yr - tr, 0.0)
            shape = _smooth(shape, 25)  # follow the phrase melody, not syllable detail
            semis[s0:s1] += 0.5 * np.clip(shape, -3, 3)
        # c) your ending: questions rise, statements fall
        #    (anchored on the last *voiced* stretch: trailing consonants have no pitch)
        vidx = np.flatnonzero(~np.isnan(st))
        v0, v1 = (vidx[0], vidx[-1] + 1) if len(vidx) else (s0, s1)
        tail = max(1, int((v1 - v0) * 0.3))
        ramp = np.linspace(0, 1, tail) ** 1.5
        extra = np.clip(you.ending - mine.ending, -8, 8) * 0.8
        semis[v1 - tail : v1] += ramp * extra
        semis[v1:] = semis[v1 - 1]
        semis[:s0] = semis[s0]
        # d) re-centre: the shaping above must not drag your overall pitch level
        voiced_now = base[vidx] + semis[vidx] if len(vidx) else np.array([])
        if len(voiced_now):
            semis += np.clip(you.pitch - np.median(voiced_now), -3, 3)
        semis = _smooth(np.clip(semis, -10, 10), 5) * strength
    ratio = 2.0 ** (semis / 12.0)

    stretch = np.full(n, 1.0)
    if pace is not None:
        stretch[:] = 1.0 + (np.clip(pace, 0.75, 1.3) - 1.0) * strength

    changed = np.any(np.abs(semis) > 0.05) or np.any(np.abs(stretch - 1) > 0.01)
    out = psola(synth, sr, st, ratio, stretch) if changed else synth

    # d) loudness follows yours (relative), in 10 ms steps, smoothed
    hop = int(sr * HOP_S)
    _, lvl_out = track(out, sr)
    m = len(lvl_out)
    if m > 4 and len(your_env):
        sp = lvl_out > (np.max(lvl_out) - 30)
        mine_rel = lvl_out - (np.mean(lvl_out[sp]) if sp.any() else 0)
        want = _resize(np.clip(your_env, -20, 6), m)
        gain_db = np.where(sp, np.clip(want - mine_rel, -6, 6), 0.0) * 0.6 * strength
        gain_db = _smooth(np.nan_to_num(gain_db), 15)
        g = 10 ** (np.interp(np.arange(len(out)), (np.arange(m) + 0.5) * hop, gain_db) / 20)
        out = out * g.astype(np.float32)
    return out.astype(np.float32)


def pace_ratio(you: Delivery, your_text: str, synth_seconds: float, synth_text: str) -> float:
    """How much longer (>1) or shorter the synthesized speech should be to match your pace."""
    yc = len("".join(your_text.split()))
    sc = len("".join(synth_text.split()))
    if yc < 4 or sc < 4 or you.speech_s < 0.3 or synth_seconds < 0.3:
        return 1.0
    your_rate = yc / you.speech_s  # characters per second while you speak
    wanted = sc / your_rate
    return float(np.clip(wanted / synth_seconds, 0.8, 1.25))
