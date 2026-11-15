"""The VoxShift control panel: a small local web app (FastAPI + one HTML page).

Run:  voxshift   (opens http://127.0.0.1:7860)
Live meters and status are pushed to the page with Server-Sent Events.
"""

from __future__ import annotations

import asyncio
import collections
import io
import json
import threading
import time
import wave
import webbrowser
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from . import effects
from .audio import find_virtual_cable, list_devices, parse_device
from .pipeline import Event, Settings, VoicePipeline
from .stt import MODEL_SIZES
from .voices import load_catalog

SPOKEN_LANGUAGES = {
    "auto": "Detect automatically", "en": "English", "hi": "Hindi", "te": "Telugu",
    "ta": "Tamil", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian", "ar": "Arabic", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean",
}


class Options(BaseModel):
    mode: str = "ai"
    voice: str = "en_GB-alan-medium"
    source_language: str = "auto"
    translate: bool = True
    clone_voice: bool = False
    follow_delivery: bool = True
    whisper_model: str = "tiny"
    input_device: str = "default"
    output_device: str = "default"
    pitch: float = 0.0
    speed: float = 1.0
    effect: str = "none"
    silence_ms: int = 450

    def settings(self) -> Settings:
        return Settings(
            mode=self.mode, voice=self.voice,
            source_language=None if self.source_language == "auto" else self.source_language,
            translate=self.translate, clone_voice=self.clone_voice,
            follow_delivery=self.follow_delivery, whisper_model=self.whisper_model,
            input_device=self.input_device, output_device=self.output_device, pitch=self.pitch,
            speed=self.speed, effect=self.effect, silence_ms=self.silence_ms,
        )


class SayRequest(BaseModel):
    text: str
    options: Options


class EnrollRequest(BaseModel):
    seconds: float = 15.0
    input_device: str = "default"


class Controller:
    def __init__(self) -> None:
        self.catalog = load_catalog()
        self.pipeline: VoicePipeline | None = None
        self.lock = threading.Lock()
        self.status = "Ready"
        self.error = ""
        self.loading = False
        self.history: collections.deque[dict] = collections.deque(maxlen=20)
        self.version = 0  # bumps when history/status changes
        self.enroll = {"active": False, "progress": 0.0, "level": 0.0, "message": ""}
        self._say_tts = None
        self._say_cloner = None

    # ------------------------------------------------------------ events
    def on_event(self, e: Event) -> None:
        if e.kind == "status":
            self.status, self.error = e.message, ""
        elif e.kind == "error":
            self.error = e.message
        elif e.kind == "utterance":
            self.history.appendleft({
                "heard": e.heard, "lang": e.heard_language, "spoken": e.spoken,
                "delay": round(e.latency, 2), "t": e.timestamp,
            })
        self.version += 1

    def snapshot(self) -> dict:
        p = self.pipeline
        state = "loading" if self.loading else (p.state if p else "idle")
        if self.enroll["active"]:
            state = "recording"
        return {
            "state": state,
            "in": self.enroll["level"] if self.enroll["active"] else (p.in_level if p else 0.0),
            "out": p.out_level if p else 0.0,
            "status": self.status, "error": self.error, "version": self.version,
            "enroll": self.enroll,
        }

    # ------------------------------------------------------------ actions
    def start(self, opts: Options) -> None:
        def run():
            with self.lock:
                if self.pipeline:
                    self.pipeline.stop()
                self.loading, self.error = True, ""
                self.pipeline = VoicePipeline(opts.settings(), self.on_event)
                try:
                    self.pipeline.start()
                except Exception as exc:
                    self.on_event(Event("error", f"{type(exc).__name__}: {exc}"))
                    self.pipeline = None
                finally:
                    self.loading = False
                    self.version += 1

        threading.Thread(target=run, daemon=True).start()

    def stop(self) -> None:
        with self.lock:
            if self.pipeline:
                self.pipeline.stop()
                self.pipeline = None
        self.status = "Ready"
        self.version += 1

    def update(self, opts: Options) -> None:
        p = self.pipeline
        if p and p.running:
            s = opts.settings()
            p.update(pitch=s.pitch, speed=s.speed, effect=s.effect, voice=s.voice,
                     translate=s.translate, source_language=s.source_language,
                     clone_voice=s.clone_voice, follow_delivery=s.follow_delivery)

    def record_my_voice(self, req: EnrollRequest) -> None:
        """Record the user's own voice from the selected microphone (never from a file)."""
        import sounddevice as sd

        from .clone import save_enrollment

        if self.enroll["active"]:
            return
        if self.pipeline and self.pipeline.running:
            self.stop()
        rate, seconds = 22050, max(8.0, min(30.0, req.seconds))
        chunks: list[np.ndarray] = []
        self.enroll.update(active=True, progress=0.0, level=0.0, message="Recording...")

        def callback(indata, frames, t, status):
            mono = indata[:, 0].copy()
            chunks.append(mono)
            self.enroll["level"] = float(np.sqrt(np.mean(mono * mono)))
            self.enroll["progress"] = min(1.0, sum(len(c) for c in chunks) / (rate * seconds))

        def run():
            try:
                with sd.InputStream(device=parse_device(req.input_device), samplerate=rate,
                                    channels=1, dtype="float32", callback=callback):
                    while self.enroll["progress"] < 1.0:
                        time.sleep(0.05)
                audio = np.concatenate(chunks)
                if float(np.sqrt(np.mean(audio**2))) < 0.003:
                    raise ValueError("That was almost silent - check the microphone and try again")
                save_enrollment(audio, rate)
                self._say_cloner = None
                self.enroll["message"] = "Saved. \"Sound like me\" is ready."
            except Exception as exc:
                self.enroll["message"] = f"Couldn't record: {exc}"
            finally:
                self.enroll.update(active=False, level=0.0)
                self.version += 1

        threading.Thread(target=run, daemon=True).start()

    def say(self, req: SayRequest) -> bytes:
        from .translate import Translator
        from .tts import TextToSpeech

        o = req.options
        text = req.text.strip()[:400]
        if not text:
            raise HTTPException(400, "Type something first")
        if self._say_tts is None:
            self._say_tts = TextToSpeech(o.voice)
        self._say_tts.set_voice(o.voice)
        tts = self._say_tts
        src = "en" if o.source_language == "auto" else o.source_language
        spoken = Translator().translate(text, src, tts.language) if o.translate else text
        wav, rate = tts.synthesize(spoken, speed=o.speed)
        if o.clone_voice:
            from .clone import VoiceCloner

            if self._say_cloner is None:
                cloner = VoiceCloner()
                if not cloner.load_my_voice():
                    raise HTTPException(400, "Record your voice first")
                self._say_cloner = cloner
            wav, rate = self._say_cloner.convert(
                wav, rate, self._say_cloner.source_embedding(o.voice, tts))
        wav = effects.apply_effects(wav, rate, pitch=o.pitch, effect=o.effect)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes((np.clip(wav, -1, 1) * 32767).astype(np.int16).tobytes())
        return buf.getvalue(), spoken

    def options(self) -> dict:
        from .clone import my_voice_path

        inputs, outputs = list_devices()
        langs: dict[str, dict] = {}
        for v in self.catalog.values():
            lang = langs.setdefault(v.language, {"name": v.language, "code": v.family, "voices": []})
            lang["voices"].append({
                "key": v.key, "accent": v.country, "speaker": v.speaker.replace("_", " "),
                "quality": v.quality,
            })
        return {
            "languages": sorted(langs.values(), key=lambda l: l["name"]),
            "spoken": SPOKEN_LANGUAGES,
            "inputs": inputs, "outputs": outputs, "cable": find_virtual_cable(),
            "models": MODEL_SIZES, "effects": effects.EFFECTS,
            "enrolled": my_voice_path().exists(),
        }


def create_app() -> FastAPI:
    app = FastAPI(title="VoxShift", docs_url=None, redoc_url=None)
    ctl = Controller()
    page = Path(__file__).with_name("static") / "index.html"

    @app.get("/", response_class=HTMLResponse)
    def index():
        return page.read_text(encoding="utf-8")

    @app.get("/api/options")
    def options():
        return ctl.options()

    @app.get("/api/history")
    def history():
        return list(ctl.history)

    @app.post("/api/start")
    def start(opts: Options):
        ctl.start(opts)
        return {"ok": True}

    @app.post("/api/stop")
    def stop():
        ctl.stop()
        return {"ok": True}

    @app.post("/api/update")
    def update(opts: Options):
        ctl.update(opts)
        return {"ok": True}

    @app.post("/api/enroll")
    def enroll(req: EnrollRequest):
        ctl.record_my_voice(req)
        return {"ok": True}

    @app.post("/api/say")
    def say(req: SayRequest):
        audio, spoken = ctl.say(req)
        return Response(audio, media_type="audio/wav",
                        headers={"X-Spoken-Text": json.dumps(spoken)})

    @app.get("/api/live")
    async def live():
        async def stream():
            while True:
                yield f"data: {json.dumps(ctl.snapshot())}\n\n"
                await asyncio.sleep(0.05)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    return app


def main(host: str = "127.0.0.1", port: int = 7860, open_browser: bool = True) -> None:
    import uvicorn

    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
