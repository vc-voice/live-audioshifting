"""The live voice pipeline.

AI mode:  mic -> utterance segmenter -> Whisper (speech-to-text)
          -> Argos (translate, if needed) -> Piper (accent voice) -> effects -> output
FX mode:  mic -> streaming pitch shifter -> output   (~40 ms latency, no AI)

Send the output to a virtual cable (e.g. VB-Audio "CABLE Input") and pick the
cable as your microphone in Discord / Zoom / OBS / games.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import sounddevice as sd

from . import effects, prosody
from .audio import device_rate, parse_device
from .segmenter import UtteranceSegmenter

log = logging.getLogger(__name__)

STT_RATE = 16000


@dataclass
class Settings:
    mode: str = "ai"  # "ai" or "fx"
    voice: str = "en_GB-alan-medium"  # Piper voice key: language + accent + speaker
    source_language: str | None = None  # None = auto-detect
    translate: bool = True  # translate into the voice's language when they differ
    clone_voice: bool = False  # re-colour the output with your own recorded voice
    follow_delivery: bool = True  # copy your pitch, intonation, loudness and pace per phrase
    whisper_model: str = "tiny"
    input_device: str | int | None = None
    output_device: str | int | None = None
    pitch: float = 0.0  # semitones
    speed: float = 1.0  # >1 faster
    effect: str = "none"
    silence_ms: int = 450
    sensitivity: float = 3.0


@dataclass
class Event:
    kind: str  # "status" | "utterance" | "error"
    message: str = ""
    heard: str = ""
    heard_language: str = ""
    spoken: str = ""
    latency: float = 0.0
    timestamp: float = field(default_factory=time.time)


class VoicePipeline:
    def __init__(self, settings: Settings, on_event: Callable[[Event], None] | None = None):
        self.settings = settings
        self.on_event = on_event or (lambda e: log.info("%s", e))
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._stream: sd.InputStream | sd.Stream | None = None
        self._mic_q: queue.Queue[np.ndarray] = queue.Queue()
        self._utt_q: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(maxsize=4)
        self._out_q: queue.Queue[tuple[np.ndarray, int]] = queue.Queue()
        self.segmenter: UtteranceSegmenter | None = None
        self._stt = None
        self._tts = None
        self._translator = None
        self._cloner = None
        self._shifter: effects.StreamingPitchShifter | None = None
        # Live meters (RMS, 0..1) and what the pipeline is doing right now.
        self.in_level = 0.0
        self.out_level = 0.0
        self._busy = False
        self._out_until = 0.0

    # ---------------------------------------------------------------- control
    @property
    def running(self) -> bool:
        return self._stream is not None

    @property
    def level(self) -> float:
        return self.in_level

    @property
    def state(self) -> str:
        """idle | listening | hearing | thinking | speaking"""
        if not self.running:
            return "idle"
        if time.time() < self._out_until:
            return "speaking"
        if self._busy or not self._utt_q.empty():
            return "thinking"
        if self.segmenter is not None and self.segmenter.in_speech:
            return "hearing"
        if self.settings.mode == "fx" and self.in_level > 0.02:
            return "speaking"
        return "listening"

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        if self.settings.mode == "fx":
            self._start_fx()
        else:
            self._start_ai()

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        for t in self._threads:
            t.join(timeout=3)
        self._threads = []
        self.in_level = self.out_level = 0.0
        self._emit("status", "Stopped")

    def update(self, **changes) -> None:
        """Change settings while running (pitch/speed/effect/voice/language apply live)."""
        for k, v in changes.items():
            setattr(self.settings, k, v)
        if self._shifter is not None and "pitch" in changes:
            self._shifter.set_semitones(self.settings.pitch)
        if "voice" in changes and self._tts is not None:
            self._tts.set_voice(self.settings.voice)
        if changes.get("clone_voice") and self._cloner is None and self._tts is not None:
            self._load_cloner()

    def _load_cloner(self) -> None:
        from .clone import VoiceCloner

        self._emit("status", "Loading voice cloning model...")
        cloner = VoiceCloner()
        if not cloner.load_my_voice():
            self.settings.clone_voice = False
            self._emit("error", "Record your voice first (\"Use my own voice\" panel) - cloning is off")
            return
        src = cloner.source_embedding(self._tts.voice_key, self._tts)
        cloner.convert(*self._tts.synthesize("Hello there."), src)  # warm up
        self._cloner = cloner

    # ---------------------------------------------------------------- FX mode
    def _start_fx(self) -> None:
        s = self.settings
        inp, out = parse_device(s.input_device), parse_device(s.output_device)
        rate = device_rate(out, "output")
        self._shifter = effects.StreamingPitchShifter(rate, s.pitch)

        def callback(indata, outdata, frames, t, status):
            mono = indata[:, 0]
            self.in_level = float(np.sqrt(np.mean(mono * mono)))
            processed = self._shifter.process(mono, robot_voice=self.settings.effect == "robot")
            if self.settings.effect == "radio":
                processed = np.tanh(processed * 3.0) * 0.6
            outdata[:] = processed.reshape(-1, 1).repeat(outdata.shape[1], axis=1)
            self.out_level = float(np.sqrt(np.mean(processed * processed)))

        self._stream = sd.Stream(
            device=(inp, out), samplerate=rate, channels=(1, 1), dtype="float32",
            blocksize=256, latency="low", callback=callback,
        )
        self._stream.start()
        self._emit("status", f"Live FX running ({rate} Hz, pitch {s.pitch:+.1f})")

    # ---------------------------------------------------------------- AI mode
    def _start_ai(self) -> None:
        s = self.settings
        self._emit("status", f"Loading speech model '{s.whisper_model}'...")
        from .stt import SpeechToText
        from .translate import Translator
        from .tts import TextToSpeech

        self._stt = SpeechToText(s.whisper_model)
        self._emit("status", f"Loading voice '{s.voice}'...")
        self._tts = TextToSpeech(s.voice)
        self._translator = Translator()
        self._emit("status", "Warming up...")
        self._stt.warm_up()
        self._tts.synthesize("ok")
        if s.clone_voice:
            self._load_cloner()
        if s.translate and s.source_language and s.source_language != self._tts.language:
            self._translator.translate("hello", s.source_language, self._tts.language)
        self.segmenter = UtteranceSegmenter(
            sample_rate=STT_RATE, silence_ms=s.silence_ms, sensitivity=s.sensitivity
        )

        def callback(indata, frames, t, status):
            self._mic_q.put(indata[:, 0].copy())

        self._stream = sd.InputStream(
            device=parse_device(s.input_device), samplerate=STT_RATE, channels=1,
            dtype="float32", blocksize=480, callback=callback,
        )
        for target in (self._segment_loop, self._process_loop, self._playback_loop):
            th = threading.Thread(target=target, daemon=True, name=target.__name__)
            th.start()
            self._threads.append(th)
        self._stream.start()
        self._emit("status", f"Listening - speak, then pause ({self._stt.device})")

    def _segment_loop(self) -> None:
        while not self._stop.is_set():
            try:
                block = self._mic_q.get(timeout=0.1)
            except queue.Empty:
                continue
            utts = self.segmenter.feed(block)
            self.in_level = self.segmenter.level
            for utt in utts:
                try:
                    self._utt_q.put_nowait((utt, time.time()))
                except queue.Full:
                    self._emit("error", "Falling behind - dropped an utterance (try a smaller model)")

    def _process_loop(self) -> None:
        while not self._stop.is_set():
            try:
                audio, t_end = self._utt_q.get(timeout=0.1)
            except queue.Empty:
                continue
            self._busy = True
            try:
                self.process_utterance(audio, t_end)
            except Exception as exc:  # keep the pipeline alive
                log.exception("processing failed")
                self._emit("error", f"{type(exc).__name__}: {exc}")
            finally:
                self._busy = False

    def process_utterance(self, audio: np.ndarray, t_end: float | None = None) -> tuple[np.ndarray, int] | None:
        """Run one utterance through STT -> translate -> TTS. Also used by tests/CLI."""
        t_end = t_end or time.time()
        s = self.settings
        heard = self._stt.transcribe(audio, s.source_language)
        if not heard.text or _is_noise(heard.text):
            return None
        target = self._tts.language
        text = heard.text
        if s.translate and heard.language != target:
            text = self._translator.translate(text, heard.language, target)
        you = prosody.analyze(audio, STT_RATE) if s.follow_delivery else None
        same_language = heard.language == self._tts.language

        # Queue each sentence for playback as soon as it is ready.
        parts, rate, first_audio = [], self._tts.sample_rate, None
        for sentence, span in _sentences(text):
            wav, rate = self._tts.synthesize(sentence, speed=s.speed)
            if len(wav) == 0:
                continue
            if s.clone_voice and self._cloner is not None:
                src_se = self._cloner.source_embedding(self._tts.voice_key, self._tts)
                wav, rate = self._cloner.convert(wav, rate, src_se)
            if you is not None and you.ok:
                pace = None
                if same_language and s.speed == 1.0:
                    synth_s = prosody.analyze(wav, rate).speech_s
                    pace = prosody.pace_ratio(you, heard.text, synth_s / (span[1] - span[0]), text)
                wav = prosody.match(wav, rate, you, span=span, pace=pace)
            wav = effects.apply_effects(wav, rate, pitch=s.pitch, effect=s.effect)
            self._out_q.put((wav, rate))
            parts.append(wav)
            first_audio = first_audio or time.time()
        self._emit(
            "utterance", heard=heard.text, heard_language=heard.language, spoken=text,
            latency=(first_audio or time.time()) - t_end,
        )
        return (np.concatenate(parts) if parts else np.zeros(0, np.float32)), rate

    def _playback_loop(self) -> None:
        out = parse_device(self.settings.output_device)
        rate = device_rate(out, "output")
        with sd.OutputStream(device=out, samplerate=rate, channels=1, dtype="float32") as stream:
            block = rate // 40  # 25 ms pieces so the output meter moves with the voice
            while not self._stop.is_set():
                try:
                    wav, src_rate = self._out_q.get(timeout=0.1)
                except queue.Empty:
                    self.out_level = 0.0
                    continue
                audio = effects.resample(wav, src_rate, rate)
                self._out_until = time.time() + len(audio) / rate + 0.1
                for i in range(0, len(audio), block):
                    piece = audio[i : i + block]
                    self.out_level = float(np.sqrt(np.mean(piece * piece)))
                    stream.write(piece.reshape(-1, 1))
                    if self._stop.is_set():
                        break
                self.out_level = 0.0

    # ---------------------------------------------------------------- utils
    def _emit(self, kind: str, message: str = "", **kw) -> None:
        try:
            self.on_event(Event(kind=kind, message=message, **kw))
        except Exception:
            log.exception("event handler failed")


_SENTENCE_END = re.compile(r"(?<=[.!?。！？।])\s+")


def _sentences(text: str) -> list[tuple[str, tuple[float, float]]]:
    """Split into sentences, each with the share (0-1) of the phrase it covers."""
    parts = [p for p in _SENTENCE_END.split(text.strip()) if p.strip()] or [text]
    total = sum(len(p) for p in parts) or 1
    out, pos = [], 0
    for p in parts:
        out.append((p, (pos / total, (pos + len(p)) / total)))
        pos += len(p)
    return out


_NOISE = {"", ".", "...", "you", "thanks for watching!", "thank you for watching."}


def _is_noise(text: str) -> bool:
    """Whisper hallucinates these on silence/breaths."""
    return text.strip().lower() in _NOISE
