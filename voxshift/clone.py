"""Make the AI voice sound like *you*: OpenVoice V2 tone-colour conversion.

Piper says the words in the chosen language/accent; this module then re-colours
that audio with the timbre of your own voice, taken from a short recording of
you (~10-20 s). Model: https://github.com/myshell-ai/OpenVoice (MIT licence),
weights from https://huggingface.co/myshell-ai/OpenVoiceV2.

Only the converter network is used (plain PyTorch); OpenVoice's own text
front-end and its old pinned dependencies are not needed.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .effects import resample
from .voices import _download, data_dir

log = logging.getLogger(__name__)

HF_BASE = "https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/main/converter"
MIN_ENROLL_SECONDS = 8.0


def _ns(d):
    return SimpleNamespace(**{k: _ns(v) if isinstance(v, dict) else v for k, v in d.items()})


def model_dir() -> Path:
    path = data_dir() / "openvoice_v2"
    path.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "checkpoint.pth"):
        if not (path / name).exists():
            _download(f"{HF_BASE}/{name}", path / name)
    return path


def my_voice_path() -> Path:
    return data_dir() / "my_voice.wav"


def save_enrollment(audio: np.ndarray, sample_rate: int) -> Path:
    """Store the user's own voice sample (mono float32)."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) / sample_rate < MIN_ENROLL_SECONDS:
        raise ValueError(f"Record at least {MIN_ENROLL_SECONDS:.0f} seconds of your voice")
    path = my_voice_path()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    # Any cached embedding is now stale.
    (data_dir() / "my_voice.se.npy").unlink(missing_ok=True)
    return path


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        rate, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768
    return data.reshape(-1, ch).mean(axis=1), rate


class VoiceCloner:
    """Converts any speech so it sounds like the enrolled speaker."""

    def __init__(self, device: str = "auto", tau: float = 0.3):
        import torch
        from openvoice.models import SynthesizerTrn

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tau = tau
        folder = model_dir()
        hps = _ns(json.loads((folder / "config.json").read_text(encoding="utf-8")))
        self.hps = hps
        self.rate = hps.data.sampling_rate
        model = SynthesizerTrn(
            len(getattr(hps, "symbols", [])),
            hps.data.filter_length // 2 + 1,
            n_speakers=hps.data.n_speakers,
            **vars(hps.model),
        ).to(device)
        state = torch.load(folder / "checkpoint.pth", map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        model.eval().requires_grad_(False)
        for m in model.modules():  # fold weight-norm: same output, less work per call
            try:
                torch.nn.utils.remove_weight_norm(m)
            except (ValueError, AttributeError):
                pass
        self.model = model
        self._window = torch.hann_window(hps.data.win_length).to(device)
        self._lock = threading.Lock()
        self._source_se: dict[str, object] = {}
        self.target_se = None
        torch.set_num_threads(os.cpu_count() or 4)
        self._onnx_dec = self._load_onnx_decoder(folder) if device == "cpu" else None

    def _load_onnx_decoder(self, folder: Path):
        """The audio decoder is ~90% of the cost; ONNX Runtime runs it ~1.5x faster on CPU."""
        torch = self.torch
        path = folder / "decoder.onnx"
        try:
            import onnxruntime as ort

            if not path.exists():
                dec = self.model.dec
                g0 = torch.zeros(1, dec.cond.in_channels, 1) if hasattr(dec, "cond") else None

                class Decoder(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.dec = dec

                    def forward(self, z):
                        return self.dec(z, g=g0)

                with torch.no_grad():
                    torch.onnx.export(
                        Decoder().eval(),(torch.randn(1, dec.conv_pre.in_channels, 100),), str(path),
                        input_names=["z"], output_names=["wav"],
                        dynamic_axes={"z": {2: "frames"}, "wav": {2: "samples"}},
                        opset_version=17, dynamo=False,
                    )
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = os.cpu_count() or 4
            return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        except Exception as exc:  # fall back to PyTorch
            path.unlink(missing_ok=True)
            log.warning("ONNX decoder unavailable (%s); using PyTorch", str(exc).splitlines()[0][:200])
            return None

    # ------------------------------------------------------------ embeddings
    def _spec(self, audio: np.ndarray):
        torch = self.torch
        d = self.hps.data
        y = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).to(self.device)[None]
        pad = (d.filter_length - d.hop_length) // 2
        y = torch.nn.functional.pad(y[:, None], (pad, pad), mode="reflect")[:, 0]
        spec = torch.stft(
            y, d.filter_length, hop_length=d.hop_length, win_length=d.win_length,
            window=self._window, center=False, return_complex=True,
        )
        return torch.sqrt(spec.real**2 + spec.imag**2 + 1e-6)

    def embed(self, audio: np.ndarray, rate: int):
        """Speaker embedding averaged over ~3 s pieces of `audio`."""
        audio = resample(audio, rate, self.rate)
        piece = 3 * self.rate
        pieces = [audio[i : i + piece] for i in range(0, len(audio), piece)]
        pieces = [p for p in pieces if len(p) > self.rate] or [audio]
        with self._lock, self.torch.no_grad():
            gs = [self.model.ref_enc(self._spec(p).transpose(1, 2)).unsqueeze(-1) for p in pieces]
        return self.torch.stack(gs).mean(0)

    def load_my_voice(self) -> bool:
        """Load the enrolled voice; returns False if the user hasn't recorded one yet."""
        cache = data_dir() / "my_voice.se.npy"
        if cache.exists():
            self.target_se = self.torch.from_numpy(np.load(cache)).to(self.device)
            return True
        path = my_voice_path()
        if not path.exists():
            return False
        audio, rate = load_wav(path)
        self.target_se = self.embed(audio, rate)
        np.save(cache, self.target_se.cpu().numpy())
        return True

    def source_embedding(self, voice_key: str, tts) -> object:
        """Embedding of a Piper voice, measured once from a calibration sentence."""
        if voice_key not in self._source_se:
            cache = data_dir() / "voices" / f"{voice_key}.se.npy"
            if cache.exists():
                se = self.torch.from_numpy(np.load(cache)).to(self.device)
            else:
                text = {
                    "hi": "नमस्ते, आज मौसम बहुत अच्छा है और मैं आपसे बात करके खुश हूँ। "
                          "हम सब मिलकर काम करेंगे।",
                    "fr": "Bonjour, il fait très beau aujourd'hui et je suis content de vous parler. "
                          "Nous allons travailler ensemble.",
                    "es": "Hola, hoy hace muy buen tiempo y me alegra hablar contigo. "
                          "Vamos a trabajar juntos.",
                }.get(tts.language, "The quick brown fox jumps over the lazy dog. Today the weather "
                      "is lovely, and I am happy to talk with you about our plans for the week.")
                audio, rate = tts.synthesize(text)
                se = self.embed(audio, rate)
                np.save(cache, se.cpu().numpy())
            self._source_se[voice_key] = se
        return self._source_se[voice_key]

    # ------------------------------------------------------------ conversion
    def convert(self, audio: np.ndarray, rate: int, source_se) -> tuple[np.ndarray, int]:
        if self.target_se is None:
            raise RuntimeError("No voice sample recorded yet")
        audio = resample(audio, rate, self.rate)
        torch = self.torch
        m = self.model
        with self._lock, torch.inference_mode():
            spec = self._spec(audio)
            lengths = torch.LongTensor([spec.size(-1)]).to(self.device)
            if self._onnx_dec is None:
                out = m.voice_conversion(
                    spec, lengths, sid_src=source_se, sid_tgt=self.target_se, tau=self.tau
                )[0][0, 0]
                return out.float().cpu().numpy(), self.rate
            # Same steps as SynthesizerTrn.voice_conversion, decoder in ONNX Runtime.
            g_enc = torch.zeros_like(source_se) if m.zero_g else source_se
            z, _, _, mask = m.enc_q(spec, lengths, g=g_enc, tau=self.tau)
            z_p = m.flow(z, mask, g=source_se)
            z_hat = m.flow(z_p, mask, g=self.target_se, reverse=True)
            zin = (z_hat * mask).cpu().numpy().astype(np.float32)
        out = self._onnx_dec.run(None, {"z": zin})[0]
        return out.reshape(-1).astype(np.float32), self.rate
