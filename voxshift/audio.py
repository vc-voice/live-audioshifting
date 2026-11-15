"""Audio device helpers (sounddevice / PortAudio)."""

from __future__ import annotations

import sounddevice as sd


def list_devices() -> tuple[list[str], list[str]]:
    """Return (input_names, output_names) formatted as 'index: name (hostapi)'."""
    inputs, outputs = [], []
    apis = sd.query_hostapis()
    for idx, dev in enumerate(sd.query_devices()):
        label = f"{idx}: {dev['name']} ({apis[dev['hostapi']]['name']})"
        if dev["max_input_channels"] > 0:
            inputs.append(label)
        if dev["max_output_channels"] > 0:
            outputs.append(label)
    return inputs, outputs


def parse_device(label: str | int | None) -> int | None:
    if label is None or label == "" or label == "default":
        return None
    if isinstance(label, int):
        return label
    return int(str(label).split(":", 1)[0])


def device_rate(device: int | None, kind: str) -> int:
    info = sd.query_devices(device, kind)
    return int(info["default_samplerate"])


def find_virtual_cable() -> str | None:
    """Find a virtual-mic output (VB-Audio Cable, VoiceMeeter, BlackHole) if installed."""
    _, outputs = list_devices()
    for name in outputs:
        low = name.lower()
        if any(k in low for k in ("cable input", "voicemeeter input", "blackhole", "vb-audio")):
            return name
    return None
