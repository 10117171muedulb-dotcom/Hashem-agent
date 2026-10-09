"""Font discovery and selection, with first-class Arabic support.

The renderer must produce beautiful Arabic typography on a stock Windows install
without shipping a font manager, so this module:

* indexes the OS font directories plus bundled and user fonts,
* knows which families cover Arabic,
* falls back through a ranked chain instead of ever returning ``None``.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ..utils.log import get_logger
from ..utils.paths import bundled_fonts_dir, fonts_dir

log = get_logger("motion.fonts")

FONT_SUFFIXES = {".ttf", ".otf", ".ttc", ".woff", ".woff2"}

# Ordered best-first. Tahoma/Segoe ship with Windows and both cover Arabic; Cairo
# and Tajawal are the modern open-source choices when the user has them.
ARABIC_CHAIN = [
    "cairo", "tajawal", "almarai", "amiri", "noto naskh arabic", "noto kufi arabic",
    "ibm plex sans arabic", "dubai", "gesh", "geeza pro", "tahoma", "segoe ui",
    "arial", "dejavu sans", "liberation sans",
]

LATIN_CHAIN = [
    "cairo", "tajawal", "inter", "roboto", "segoe ui", "arial", "helvetica",
    "dejavu sans", "liberation sans", "noto sans",
]

MONO_CHAIN = ["jetbrains mono", "fira code", "consolas", "cascadia code", "dejavu sans mono", "courier new"]

_ARABIC_RANGE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")


@dataclass(frozen=True)
class FontEntry:
    family: str
    path: str
    bold: bool = False
    italic: bool = False
    weight: str = "regular"

    @property
    def key(self) -> tuple[str, bool, bool]:
        return (self.family.lower(), self.bold, self.italic)


@dataclass
class FontIndex:
    entries: list[FontEntry] = field(default_factory=list)
    by_family: dict[str, list[FontEntry]] = field(default_factory=dict)
    scanned_dirs: list[str] = field(default_factory=list)

    def add(self, entry: FontEntry) -> None:
        self.entries.append(entry)
        self.by_family.setdefault(entry.family.lower(), []).append(entry)

    def families(self) -> list[str]:
        return sorted(self.by_family)

    def find(self, family: str | None, bold: bool = False, italic: bool = False) -> FontEntry | None:
        if not family:
            return None
        candidates = self.by_family.get(family.strip().lower())
        if not candidates:
            # tolerate "Cairo Bold" / "Cairo-Bold" style lookups
            slug = re.sub(r"[-\s]+(bold|black|heavy|light|thin|medium|semibold|regular|italic)$", "", family.strip().lower())
            candidates = self.by_family.get(slug)
        if not candidates:
            return None

        def score(entry: FontEntry) -> tuple[int, int]:
            return (0 if entry.bold == bold else 1, 0 if entry.italic == italic else 1)

        return sorted(candidates, key=score)[0]


_WEIGHT_TOKENS = {
    "thin": 100, "hairline": 100, "extralight": 200, "ultralight": 200, "light": 300,
    "regular": 400, "normal": 400, "book": 400, "medium": 500, "semibold": 600,
    "demibold": 600, "bold": 700, "extrabold": 800, "ultrabold": 800, "black": 900,
    "heavy": 900,
}


def _classify(name: str) -> tuple[str, bool, bool, str]:
    """Split a file stem into (family, bold, italic, weight)."""
    stem = Path(name).stem
    tokens = re.split(r"[-_\s]+", stem)
    flags: list[str] = []
    family_tokens: list[str] = []
    for token in tokens:
        low = token.lower()
        if low in _WEIGHT_TOKENS or low in {"italic", "oblique", "it"}:
            flags.append(low)
        else:
            family_tokens.append(token)
    family = " ".join(family_tokens).strip() or stem
    text = " ".join(flags)
    bold = any(t in text for t in ("bold", "black", "heavy", "extrabold", "semibold"))
    italic = "italic" in text or "oblique" in text or text.strip() == "it"
    weight = "bold" if bold else "regular"
    for token in tokens:
        if token.lower() in _WEIGHT_TOKENS:
            weight = token.lower()
            break
    return family, bold, italic, weight


def system_font_dirs() -> list[Path]:
    """Directories that typically hold usable fonts on the current OS."""
    dirs: list[Path] = []
    if sys.platform == "win32":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        dirs += [Path(windir) / "Fonts", Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Windows/Fonts"]
    elif sys.platform == "darwin":
        dirs += [Path("/System/Library/Fonts"), Path("/Library/Fonts"), Path.home() / "Library/Fonts"]
    else:
        dirs += [
            Path("/usr/share/fonts"),
            Path("/usr/local/share/fonts"),
            Path.home() / ".fonts",
            Path.home() / ".local/share/fonts",
        ]
    dirs.append(bundled_fonts_dir())
    dirs.append(fonts_dir())
    return [d for d in dirs if d and Path(d).is_dir()]


def scan_fonts(dirs: list[Path] | None = None) -> FontIndex:
    """Walk the font directories and build an index. Missing dirs are skipped."""
    index = FontIndex()
    targets = dirs if dirs is not None else system_font_dirs()
    for directory in targets:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        index.scanned_dirs.append(str(directory))
        for root, _dirs, files in os.walk(directory):
            for name in files:
                if Path(name).suffix.lower() not in FONT_SUFFIXES:
                    continue
                full = Path(root) / name
                family, bold, italic, weight = _classify(name)
                index.add(FontEntry(family=family, path=str(full), bold=bold, italic=italic, weight=weight))
    log.debug("Indexed %d fonts across %d directories", len(index.entries), len(index.scanned_dirs))
    return index


@lru_cache(maxsize=1)
def _default_index() -> FontIndex:
    return scan_fonts()


def get_index(rescan: bool = False) -> FontIndex:
    if rescan:
        _default_index.cache_clear()
    return _default_index()


def contains_arabic(text: str) -> bool:
    return bool(_ARABIC_RANGE.search(text or ""))


def _pick_from_chain(index: FontIndex, chain: list[str], bold: bool, italic: bool) -> FontEntry | None:
    for family in chain:
        entry = index.find(family, bold=bold, italic=italic)
        if entry:
            return entry
    return None


def resolve_font(
    text: str,
    family: str | None = None,
    bold: bool = False,
    italic: bool = False,
    index: FontIndex | None = None,
) -> str:
    """Return a font file path suitable for ``text``.

    Always returns a usable path: explicit family → Arabic/Latin chain → any
    indexed font → PIL's built-in bitmap font (last resort, still renders).
    """
    idx = index or get_index()

    if family:
        entry = idx.find(family, bold=bold, italic=italic)
        if entry:
            return entry.path
        log.debug("Requested family %r not found; falling back to the auto chain", family)

    chain = ARABIC_CHAIN if contains_arabic(text) else LATIN_CHAIN
    entry = _pick_from_chain(idx, chain, bold, italic)
    if entry:
        return entry.path

    # Anything at all is better than nothing.
    if idx.entries:
        return idx.entries[0].path
    return ""


def resolve_mono(bold: bool = False, index: FontIndex | None = None) -> str:
    idx = index or get_index()
    entry = _pick_from_chain(idx, MONO_CHAIN, bold=bold, italic=False)
    return entry.path if entry else resolve_font("mono", index=idx, bold=bold)


def available_families(arabic_only: bool = False) -> list[str]:
    """Families present on this machine (used to populate the UI font picker)."""
    idx = get_index()
    names = idx.families()
    if not arabic_only:
        return names
    known = set(ARABIC_CHAIN)
    return [n for n in names if n in known]


def weight_to_synthetic(weight: str | int | None) -> tuple[bool, str]:
    """Map a CSS-ish weight to ``(bold, canonical_name)``.

    PIL cannot synthesise weights, so anything >= 600 is treated as bold.
    """
    if isinstance(weight, int):
        value = weight
    else:
        value = _WEIGHT_TOKENS.get(str(weight or "regular").lower(), 400)
    return value >= 600, ("bold" if value >= 600 else "regular")


def font_report() -> dict[str, object]:
    """Diagnostics payload for ``hashem doctor`` and the settings panel."""
    idx = get_index()
    return {
        "total_fonts": len(idx.entries),
        "families": len(idx.by_family),
        "scanned_dirs": idx.scanned_dirs,
        "arabic_available": [n for n in ARABIC_CHAIN if idx.find(n) is not None][:5],
        "latin_available": [n for n in LATIN_CHAIN if idx.find(n) is not None][:5],
        "bundled_dir": str(bundled_fonts_dir()),
        "user_dir": str(fonts_dir()),
    }


__all__ = [
    "FontEntry",
    "FontIndex",
    "scan_fonts",
    "get_index",
    "system_font_dirs",
    "resolve_font",
    "resolve_mono",
    "contains_arabic",
    "available_families",
    "weight_to_synthetic",
    "font_report",
    "ARABIC_CHAIN",
    "LATIN_CHAIN",
]
