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
