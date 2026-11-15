import numpy as np

from voxshift import effects
from voxshift.segmenter import UtteranceSegmenter

SR = 16000


def tone(freq, seconds, sr=SR, amp=0.3):
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_freq(x, sr):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / sr)[np.argmax(spec)]


def test_segmenter_finds_one_utterance():
    seg = UtteranceSegmenter(sample_rate=SR, silence_ms=300)
    audio = np.concatenate([np.zeros(SR, np.float32), tone(220, 1.0), np.zeros(SR, np.float32)])
    out = []
    for i in range(0, len(audio), 480):
        out += seg.feed(audio[i : i + 480])
    assert len(out) == 1
    assert 0.9 * SR < len(out[0]) < 1.6 * SR


def test_segmenter_ignores_click():
    seg = UtteranceSegmenter(sample_rate=SR, silence_ms=300)
    audio = np.concatenate([np.zeros(SR, np.float32), tone(220, 0.06), np.zeros(SR, np.float32)])
    out = []
    for i in range(0, len(audio), 480):
        out += seg.feed(audio[i : i + 480])
    assert out == []


def test_pitch_shift_octave_up_keeps_length():
    x = tone(220, 1.0)
    y = effects.pitch_shift(x, 12, SR)
    assert abs(len(y) - len(x)) < 0.05 * len(x)
    assert abs(dominant_freq(y, SR) - 440) < 15


def test_streaming_pitch_shifter():
    sr = 48000
    x = tone(300, 1.0, sr)
    ps = effects.StreamingPitchShifter(sr, semitones=-12)
    y = np.concatenate([ps.process(x[i : i + 256]) for i in range(0, len(x), 256)])
    assert len(y) == len(x)
    assert abs(dominant_freq(y[sr // 4 :], sr) - 150) < 10


def test_resample_length():
    assert len(effects.resample(np.zeros(22050, np.float32), 22050, 48000)) == 48000


def test_enrollment_rejects_short_sample(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setenv("VOXSHIFT_HOME", str(tmp_path))
    from voxshift.clone import save_enrollment

    with pytest.raises(ValueError):
        save_enrollment(np.zeros(SR * 3, np.float32), SR)
    path = save_enrollment(tone(150, 10.0), SR)
    assert path.exists() and path.parent == tmp_path


def voiced_like(f0_start, f0_end, seconds, sr=SR):
    """Glottal-pulse-like signal (harmonics) with a gliding pitch."""
    n = int(sr * seconds)
    f0 = np.linspace(f0_start, f0_end, n)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sig = sum(np.sin(k * phase) / k for k in range(1, 12))
    return (0.2 * sig).astype(np.float32)


def test_pitch_tracker_and_ending():
    from voxshift import prosody

    d = prosody.analyze(voiced_like(140, 200, 1.5), SR)
    assert abs(100 * 2 ** (d.pitch / 12) - 168) < 15
    assert d.ending > 1.0  # rising
    assert prosody.analyze(voiced_like(200, 140, 1.5), SR).ending < -1.0


def test_psola_moves_pitch_keeps_length():
    from voxshift import prosody

    x = voiced_like(150, 150, 1.0)
    st, _ = prosody.track(x, SR)
    n = len(st)
    y = prosody.psola(x, SR, st, np.full(n, 2 ** (4 / 12)), np.ones(n))
    assert abs(len(y) - len(x)) < 0.05 * len(x)
    got = 100 * 2 ** (prosody.analyze(y, SR).pitch / 12)
    assert abs(got - 150 * 2 ** (4 / 12)) < 10


def test_match_follows_speaker_level():
    from voxshift import prosody

    you = prosody.analyze(voiced_like(110, 120, 1.2), SR)
    synth = voiced_like(200, 210, 1.2)
    out = prosody.match(synth, SR, you)
    assert abs(prosody.analyze(out, SR).pitch - you.pitch) < 1.5
