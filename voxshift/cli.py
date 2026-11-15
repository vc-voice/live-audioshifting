"""Command line entry point.

  voxshift                       open the web control panel
  voxshift devices               list audio devices
  voxshift voices [--lang en]    list available voices / accents
  voxshift run --voice fr_FR-siwis-medium --output "CABLE Input"
  voxshift say "Hello" --voice hi_IN-pratham-medium -o hello.wav
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import wave

import numpy as np


def _match_device(name: str | None, outputs: bool) -> str | None:
    if not name:
        return None
    from .audio import list_devices

    if name.isdigit():
        return name
    ins, outs = list_devices()
    for label in outs if outputs else ins:
        if name.lower() in label.lower():
            return label
    raise SystemExit(f"No {'output' if outputs else 'input'} device matching '{name}'")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="voxshift", description="Live voice, accent & language changer")
    sub = p.add_subparsers(dest="cmd")

    ui = sub.add_parser("ui", help="open the web control panel (default)")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=7860)
    ui.add_argument("--no-browser", action="store_true")

    sub.add_parser("devices", help="list audio devices")

    v = sub.add_parser("voices", help="list voices")
    v.add_argument("--lang", help="language family, e.g. en, hi, fr")

    r = sub.add_parser("run", help="run live from the terminal")
    r.add_argument("--mode", choices=["ai", "fx"], default="ai")
    r.add_argument("--voice", default="en_GB-alan-medium")
    r.add_argument("--source", help="language you speak (default: auto-detect)")
    r.add_argument("--no-translate", action="store_true")
    r.add_argument("--clone", action="store_true", help="sound like your recorded voice (see 'enroll')")
    r.add_argument("--no-follow", action="store_true", help="don't copy your pitch/intonation/pace")
    r.add_argument("--model", default="tiny")
    r.add_argument("--input")
    r.add_argument("--output")
    r.add_argument("--pitch", type=float, default=0.0)
    r.add_argument("--speed", type=float, default=1.0)
    r.add_argument("--effect", default="none")

    e = sub.add_parser("enroll", help="record your own voice for --clone")
    e.add_argument("--seconds", type=float, default=15)
    e.add_argument("--input")

    s = sub.add_parser("say", help="speak typed text in a voice (no mic needed)")
    s.add_argument("text")
    s.add_argument("--voice", default="en_GB-alan-medium")
    s.add_argument("--from-lang", default="en")
    s.add_argument("--pitch", type=float, default=0.0)
    s.add_argument("--speed", type=float, default=1.0)
    s.add_argument("--effect", default="none")
    s.add_argument("-o", "--out", help="write a .wav instead of playing")

    args = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # print Hindi/French etc. on Windows consoles
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.cmd in (None, "ui"):
        from .ui import main as ui_main

        if args.cmd is None:
            ui_main()
        else:
            ui_main(args.host, args.port, not args.no_browser)
    elif args.cmd == "devices":
        from .audio import find_virtual_cable, list_devices

        ins, outs = list_devices()
        print("Inputs:\n  " + "\n  ".join(ins))
        print("Outputs:\n  " + "\n  ".join(outs))
        print(f"\nVirtual cable: {find_virtual_cable() or 'not found (install VB-Audio Cable)'}")
    elif args.cmd == "voices":
        from .voices import load_catalog

        for voice in load_catalog().values():
            if not args.lang or voice.family == args.lang:
                print(f"{voice.key:45} {voice.label}")
    elif args.cmd == "run":
        from .pipeline import Settings, VoicePipeline

        def show(e):
            if e.kind == "utterance":
                print(f"[{e.heard_language}] {e.heard}  ->  {e.spoken}   ({e.latency:.1f}s)")
            else:
                print(f"* {e.message}")

        settings = Settings(
            mode=args.mode, voice=args.voice, source_language=args.source,
            translate=not args.no_translate, clone_voice=args.clone, follow_delivery=not args.no_follow, whisper_model=args.model,
            input_device=_match_device(args.input, False), output_device=_match_device(args.output, True),
            pitch=args.pitch, speed=args.speed, effect=args.effect,
        )
        pipe = VoicePipeline(settings, show)
        pipe.start()
        print("Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pipe.stop()
    elif args.cmd == "enroll":
        import sounddevice as sd

        from .audio import parse_device
        from .clone import save_enrollment

        print("Read aloud in your normal voice, e.g.: \"Hi, this is my voice. I'm recording a short "
              "sample so the app can learn how I sound. The weather is nice today...\"")
        input("Press Enter to start recording...")
        rate = 22050
        rec = sd.rec(int(args.seconds * rate), samplerate=rate, channels=1, dtype="float32",
                     device=parse_device(_match_device(args.input, False)))
        sd.wait()
        print(f"Saved {save_enrollment(rec[:, 0], rate)}. Now use: voxshift run --clone")
    elif args.cmd == "say":
        from . import effects
        from .translate import Translator
        from .tts import TextToSpeech

        tts = TextToSpeech(args.voice)
        text = Translator().translate(args.text, args.from_lang, tts.language)
        print(f"Speaking: {text}")
        wav, rate = tts.synthesize(text, speed=args.speed)
        wav = effects.apply_effects(wav, rate, pitch=args.pitch, effect=args.effect)
        if args.out:
            with wave.open(args.out, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes((np.clip(wav, -1, 1) * 32767).astype(np.int16).tobytes())
            print(f"Wrote {args.out}")
        else:
            import sounddevice as sd

            sd.play(wav, rate)
            sd.wait()


if __name__ == "__main__":
    main(sys.argv[1:])
