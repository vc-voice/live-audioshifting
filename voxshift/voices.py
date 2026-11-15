"""Catalog of open-source Piper voices (https://huggingface.co/rhasspy/piper-voices).

Each voice is a language + region ("accent") + speaker, e.g. ``en_GB-alan-medium``
is a British English male voice. The catalog is downloaded once and cached.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def data_dir() -> Path:
    root = os.environ.get("VOXSHIFT_HOME") or Path.home() / ".voxshift"
    path = Path(root)
    (path / "voices").mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class Voice:
    key: str  # e.g. en_GB-alan-medium
    lang_code: str  # e.g. en_GB
    family: str  # e.g. en
    language: str  # e.g. English
    country: str  # e.g. Great Britain
    speaker: str
    quality: str
    files: tuple[str, ...]

    @property
    def label(self) -> str:
        return f"{self.language} ({self.country}) - {self.speaker} [{self.quality}]"


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "voxshift"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 16):
            f.write(chunk)
    tmp.replace(dest)


def load_catalog(refresh: bool = False) -> dict[str, Voice]:
    cache = data_dir() / "voices.json"
    if refresh or not cache.exists():
        _download(f"{HF_BASE}/voices.json", cache)
    raw = json.loads(cache.read_text(encoding="utf-8"))
    voices: dict[str, Voice] = {}
    for key, v in raw.items():
        lang = v.get("language", {})
        voices[key] = Voice(
            key=key,
            lang_code=lang.get("code", key.split("-")[0]),
            family=lang.get("family", key.split("_")[0]),
            language=lang.get("name_english", lang.get("family", "?")),
            country=lang.get("country_english", lang.get("region", "")),
            speaker=v.get("name", key),
            quality=v.get("quality", ""),
            files=tuple(f for f in v.get("files", {}) if f.endswith((".onnx", ".onnx.json"))),
        )
    return dict(sorted(voices.items(), key=lambda kv: (kv[1].language, kv[1].country, kv[0])))


def voices_for_language(catalog: dict[str, Voice], family: str) -> list[Voice]:
    return [v for v in catalog.values() if v.family == family]


def ensure_voice(voice: Voice) -> Path:
    """Download the voice model (once) and return the path to its .onnx file."""
    folder = data_dir() / "voices"
    onnx_path = None
    for rel in voice.files:
        dest = folder / Path(rel).name
        if not dest.exists():
            _download(f"{HF_BASE}/{rel}", dest)
        if rel.endswith(".onnx"):
            onnx_path = dest
    if onnx_path is None:
        raise RuntimeError(f"Voice {voice.key} has no model file")
    return onnx_path
