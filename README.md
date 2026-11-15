# VoxShift

**Live voice, accent and language changer** built entirely on open-source models.
Speak into your mic, pause, and your words come out in the accent, language and voice you picked:
say *"hello"* and it can come out as a British, American or Scottish voice, or as French *"bonjour"*,
Spanish *"hola"* or Hindi *"नमस्ते"*. Route the output to a virtual cable and it works as a microphone
in Discord, Zoom, Meet, OBS or games.

| Open-source piece | Used for |
|---|---|
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (OpenAI Whisper) | Speech-to-text and auto language detection |
| [Argos Translate](https://github.com/argosopentech/argos-translate) | Offline translation (packs download on first use) |
| [OpenVoice V2](https://github.com/myshell-ai/OpenVoice) (optional) | Voice cloning: makes the output sound like **you** |
| [Piper](https://github.com/OHF-Voice/piper1-gpl) + [piper-voices](https://huggingface.co/rhasspy/piper-voices) | Text-to-speech in 100+ voices over 40+ languages and regional accents |
| [sounddevice](https://github.com/spatialaudio/python-sounddevice) / PortAudio | Live mic capture and playback |
| [Gradio](https://github.com/gradio-app/gradio) | Control panel in the browser |

Everything runs locally. After the first download of models, no internet is needed.

## Modes

- **AI voice:** mic → Whisper → (translate) → Piper voice → pitch/effects → output.
  Pick a language plus an accent/speaker (for example English: United States, Great Britain,
  northern English, Scottish; Spanish: Spain or Mexico; Hindi, Telugu, French, German and many more).
  Measured on a 4-core laptop CPU: about **0.6–0.9 s** from the end of your pause to the first sound.
- **Sound like me (voice cloning):** record about 15 s of your voice once. Each sentence Piper speaks is
  then converted to your own timbre, so you sound like *yourself* with the new accent or language. This
  adds about 0.2–1.5 s per sentence on a CPU (total about 0.8–2.4 s) and much less on an NVIDIA GPU.
- **Live FX:** mic → streaming pitch shifter (deeper/higher, robot, radio) → output with about 40 ms delay. No AI.

Pitch, speed, effect, voice and translation can be changed while it is running.

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/vc-voice/live-audioshifting.git
cd live-audioshifting
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -e .
```

To enable voice cloning, also run:

```bash
pip install -e .[clone]
pip install --no-deps git+https://github.com/myshell-ai/OpenVoice.git
```

The `--no-deps` flag matters. OpenVoice pins very old packages, and VoxShift only uses its converter
network. The 128 MB of converter weights download on first use.

### Use it as a mic in other apps

1. Install a free virtual audio cable:
   [VB-Audio Cable](https://vb-audio.com/Cable/) (Windows/macOS) or
   [BlackHole](https://github.com/ExistentialAudio/BlackHole) (macOS).
   On Linux, use a PulseAudio/PipeWire null sink.
2. In VoxShift, set **Output** to `CABLE Input`. It is picked automatically if found.
3. In Discord, Zoom or another app, choose `CABLE Output` as the microphone.

## Use

```bash
voxshift                     # opens the control panel at http://127.0.0.1:7860
voxshift devices             # list mics/speakers and detect the virtual cable
voxshift voices --lang en    # list English accents/voices
voxshift voices --lang hi    # Hindi voices
voxshift say "Hello, how are you?" --voice fr_FR-siwis-medium      # -> "Bonjour, comment allez-vous ?"
voxshift run --voice en_GB-alan-medium --output "CABLE Input"     # live, British accent
voxshift run --voice es_MX-claude-high --source en                # speak English, come out Spanish
voxshift run --mode fx --pitch -5 --effect robot                  # live deep robot voice
voxshift enroll                                                   # record ~15 s of your own voice
voxshift run --voice fr_FR-siwis-medium --clone                   # speak French in *your* voice
```

The first run downloads the Whisper model (`base` is about 150 MB), the selected voice
(about 60 MB) and any translation pack. They are stored in `~/.voxshift` and the
Hugging Face cache.

### Tips

- `tiny` (the default) is fastest on a CPU. Use `base` or `small` for accuracy, ideally on a GPU.
- Choosing your language under **I am speaking** avoids an extra language check about every 6 sentences.
- For cloning, record in a quiet room in your normal speaking voice. 15–20 s is enough.
- Lower **Pause before speaking** for snappier responses. Raise it if sentences get cut off.
- Use headphones so the output does not feed back into the mic.

## How it works

```
mic (16 kHz) ─► energy VAD segmenter ─► faster-whisper ─► Argos Translate ─► Piper TTS ─► pitch/FX ─► output device
                                         (text + language)  (if language differs)   (accent voice)          (e.g. CABLE Input)
```

- `voxshift/segmenter.py`: adaptive-noise-floor utterance detector
- `voxshift/stt.py`, `translate.py`, `tts.py`, `voices.py`: model wrappers
- `voxshift/clone.py`: OpenVoice V2 tone-colour converter (decoder runs in ONNX Runtime on CPU)
- `voxshift/effects.py`: resampling, pitch shift, streaming pitch shifter, robot and radio effects
- `voxshift/pipeline.py`: threads that tie it all together
- `voxshift/ui.py`, `cli.py`: interfaces

## Roadmap / contributing

- Speech-to-speech timbre conversion via [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- Streaming partial transcripts for lower latency
- Packaged Windows installer

PRs are welcome. Run the tests with `pip install -e .[dev] && pytest`.

## Responsible use

Don't use VoxShift to impersonate real people or to deceive anyone. Voice cloning is meant for
**your own** voice. The control panel only enrolls from a live microphone recording. Some voice models have
their own licenses. Check the model card of each Piper voice before commercial use.

## License

MIT for the VoxShift code. Model licenses vary.
