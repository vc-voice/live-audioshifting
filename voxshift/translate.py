"""Offline translation with Argos Translate. Language packs download on first use."""

from __future__ import annotations

import logging
import os
import threading

# Argos' default sentence splitter (Stanza) loads a PyTorch model per language,
# hundreds of MB each. MiniSBD is tiny and plenty for short spoken phrases.
os.environ.setdefault("ARGOS_CHUNK_TYPE", "MINISBD")

log = logging.getLogger(__name__)
for _noisy in ("argostranslate", "argos-translate", "stanza"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# Whisper language codes that differ from Argos codes.
_ALIASES = {"zh": "zh", "jw": "jv", "he": "he", "iw": "he", "nb": "nb", "no": "nb"}


def _code(lang: str) -> str:
    lang = (lang or "").lower().split("_")[0].split("-")[0]
    return _ALIASES.get(lang, lang)


class Translator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._index_updated = False

    def translate(self, text: str, source: str, target: str) -> str:
        src, dst = _code(source), _code(target)
        if not text.strip() or src == dst:
            return text
        import argostranslate.translate as at

        logging.getLogger("argostranslate.utils").setLevel(logging.WARNING)
        with self._lock:
            if src != "en" and dst != "en":
                # Argos packs are mostly X<->English, so pivot through English.
                self._ensure(src, "en")
                self._ensure("en", dst)
                mid = at.translate(text, src, "en")
                return at.translate(mid, "en", dst)
            self._ensure(src, dst)
            return at.translate(text, src, dst)

    def _ensure(self, src: str, dst: str) -> None:
        import argostranslate.package as ap

        installed = {(p.from_code, p.to_code) for p in ap.get_installed_packages()}
        if (src, dst) in installed:
            return
        if not self._index_updated:
            log.info("Updating Argos Translate package index")
            ap.update_package_index()
            self._index_updated = True
        for pkg in ap.get_available_packages():
            if pkg.from_code == src and pkg.to_code == dst:
                log.info("Downloading translation pack %s -> %s", src, dst)
                ap.install_from_path(pkg.download())
                return
        raise RuntimeError(f"No offline translation pack for {src} -> {dst}")
