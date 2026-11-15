"""Web control panel (Gradio). Run with:  python -m voxshift  (opens http://127.0.0.1:7860)."""

from __future__ import annotations

import collections
import threading
import time

import gradio as gr

from . import effects
from .audio import find_virtual_cable, list_devices
from .pipeline import Event, Settings, VoicePipeline
from .stt import MODEL_SIZES
from .voices import load_catalog

MODE_AI = "AI voice - change accent / language"
MODE_FX = "Live FX - pitch & effects only (no AI, ~40 ms)"

SOURCE_LANGS = {
    "Auto-detect": None, "English": "en", "Hindi": "hi", "Telugu": "te", "Tamil": "ta",
    "Spanish": "es", "French": "fr", "German": "de", "Italian": "it", "Portuguese": "pt",
    "Russian": "ru", "Arabic": "ar", "Chinese": "zh", "Japanese": "ja", "Korean": "ko",
}


class App:
    def __init__(self) -> None:
        self.catalog = load_catalog()
        self.pipeline: VoicePipeline | None = None
        self.events: collections.deque[Event] = collections.deque(maxlen=50)
        self.status = "Idle"
        self.lock = threading.Lock()
        self.preview_tts = None

    # -- catalog helpers
    def languages(self) -> list[str]:
        return sorted({v.language for v in self.catalog.values()})

    def voices_for(self, language: str) -> list[tuple[str, str]]:
        return [(f"{v.country} - {v.speaker} [{v.quality}]", v.key)
                for v in self.catalog.values() if v.language == language]

    # -- events
    def on_event(self, e: Event) -> None:
        if e.kind == "status":
            self.status = e.message
        elif e.kind == "error":
            self.status = "Error: " + e.message
        self.events.appendleft(e)

    def transcript_md(self) -> str:
        rows = [e for e in self.events if e.kind == "utterance"]
        if not rows:
            return "_Nothing yet - press Start and say something, then pause._"
        lines = ["| You said | Lang | Spoken as | Delay |", "|---|---|---|---|"]
        for e in rows[:15]:
            lines.append(f"| {e.heard} | {e.heard_language} | {e.spoken} | {e.latency:.1f}s |")
        return "\n".join(lines)

    def level_text(self) -> str:
        p = self.pipeline
        if not p or not p.running or not p.segmenter:
            return self.status
        bars = int(min(1.0, p.level * 20) * 20)
        talk = "speaking" if p.segmenter.in_speech else "listening"
        return f"{self.status}\n\nMic: {'#' * bars}{'.' * (20 - bars)}  ({talk})"

    # -- actions
    def start(self, mode, voice, src, translate, model, inp, out, pitch, speed, effect, silence):
        with self.lock:
            if self.pipeline and self.pipeline.running:
                self.pipeline.stop()
            settings = Settings(
                mode="fx" if mode == MODE_FX else "ai", voice=voice,
                source_language=SOURCE_LANGS.get(src), translate=translate,
                whisper_model=model, input_device=inp, output_device=out, pitch=pitch,
                speed=speed, effect=effect, silence_ms=int(silence),
            )
            self.pipeline = VoicePipeline(settings, self.on_event)
            try:
                self.pipeline.start()
            except Exception as exc:
                self.on_event(Event("error", f"{type(exc).__name__}: {exc}"))
                self.pipeline = None
        return self.level_text()

    def stop(self):
        with self.lock:
            if self.pipeline:
                self.pipeline.stop()
                self.pipeline = None
        return self.status

    def live_update(self, pitch, speed, effect, voice, translate, src):
        if self.pipeline and self.pipeline.running:
            self.pipeline.update(pitch=pitch, speed=speed, effect=effect, voice=voice,
                                 translate=translate, source_language=SOURCE_LANGS.get(src))

    def preview(self, text, voice, pitch, speed, effect, translate):
        """Type text instead of speaking - hear it in the chosen voice/language."""
        from .translate import Translator
        from .tts import TextToSpeech

        if not text.strip():
            return None, ""
        if self.preview_tts is None:
            self.preview_tts = TextToSpeech(voice)
        else:
            self.preview_tts.set_voice(voice)
        spoken = text
        if translate and self.preview_tts.language != "en":
            spoken = Translator().translate(text, "en", self.preview_tts.language)
        wav, rate = self.preview_tts.synthesize(spoken, speed=speed)
        wav = effects.apply_effects(wav, rate, pitch=pitch, effect=effect)
        return (rate, wav), spoken


def build(app: App | None = None) -> gr.Blocks:
    app = app or App()
    inputs, outputs = list_devices()
    cable = find_virtual_cable()
    langs = app.languages()
    default_lang = "English" if "English" in langs else langs[0]
    default_voices = app.voices_for(default_lang)
    default_voice = next((k for _, k in default_voices if k.startswith("en_GB")), default_voices[0][1])

    with gr.Blocks(title="VoxShift - live voice & accent changer") as demo:
        gr.Markdown(
            "# VoxShift\nLive voice, accent and language changer. Speak, pause, and your words "
            "come out in the chosen accent or language. Route **Output** to a virtual cable "
            "(VB-Audio *CABLE Input*) and select *CABLE Output* as the mic in Discord/Zoom/OBS."
        )
        with gr.Row():
            with gr.Column(scale=3):
                mode = gr.Radio([MODE_AI, MODE_FX], value=MODE_AI, label="Mode")
                with gr.Row():
                    language = gr.Dropdown(langs, value=default_lang, label="Speak in language")
                    voice = gr.Dropdown(default_voices, value=default_voice,
                                        label="Accent / voice (country - speaker)")
                with gr.Row():
                    src = gr.Dropdown(list(SOURCE_LANGS), value="Auto-detect", label="I am speaking")
                    translate = gr.Checkbox(True, label="Translate to the voice's language")
                    model = gr.Dropdown(MODEL_SIZES, value="base", label="Recognition model")
                with gr.Row():
                    inp = gr.Dropdown(["default"] + inputs, value="default", label="Microphone")
                    out = gr.Dropdown(["default"] + outputs, value=cable or "default",
                                      label="Output (virtual cable = mic for other apps)")
                with gr.Row():
                    pitch = gr.Slider(-12, 12, 0, step=0.5, label="Pitch (semitones)")
                    speed = gr.Slider(0.5, 2.0, 1.0, step=0.05, label="Speed")
                    effect = gr.Dropdown(effects.EFFECTS, value="none", label="Effect")
                    silence = gr.Slider(250, 1500, 600, step=50, label="Pause before speaking (ms)")
                with gr.Row():
                    start = gr.Button("Start", variant="primary")
                    stop = gr.Button("Stop", variant="stop")
            with gr.Column(scale=2):
                status = gr.Textbox(value="Idle", label="Status", lines=3, interactive=False)
                transcript = gr.Markdown(app.transcript_md())
                with gr.Accordion("Try it without a mic", open=False):
                    text = gr.Textbox("Hello, how are you today?", label="Type something")
                    say = gr.Button("Speak it")
                    said = gr.Textbox(label="Spoken text", interactive=False)
                    audio = gr.Audio(label="Result", autoplay=True)

        language.change(
            lambda lang: gr.update(choices=app.voices_for(lang), value=app.voices_for(lang)[0][1]),
            language, voice,
        )
        all_inputs = [mode, voice, src, translate, model, inp, out, pitch, speed, effect, silence]
        start.click(app.start, all_inputs, status)
        stop.click(app.stop, None, status)
        live = [pitch, speed, effect, voice, translate, src]
        for c in live:
            c.change(app.live_update, live, None)
        say.click(app.preview, [text, voice, pitch, speed, effect, translate], [audio, said])

        timer = gr.Timer(0.5)
        timer.tick(lambda: (app.level_text(), app.transcript_md()), None, [status, transcript])
    return demo


def main(host: str = "127.0.0.1", port: int = 7860, open_browser: bool = True) -> None:
    demo = build()
    demo.queue().launch(server_name=host, server_port=port, inbrowser=open_browser)
