"""Colour parsing, palettes and gradient generation.

Everything works in 8-bit RGBA tuples.  Gradients are produced as numpy arrays so
the compositor can blend them without a Python-level pixel loop.
"""

from __future__ import annotations

import colorsys
import math
import re
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "parse_color",
    "to_hex",
    "with_alpha",
    "mix",
    "linear_gradient",
    "radial_gradient",
    "conic_gradient",
    "PALETTES",
    "palette",
    "readable_text_color",
    "contrast_ratio",
]

_NAMED = {
    "transparent": (0, 0, 0, 0),
    "white": (255, 255, 255, 255),
    "black": (0, 0, 0, 255),
    "red": (239, 68, 68, 255),
    "green": (34, 197, 94, 255),
    "blue": (59, 130, 246, 255),
    "yellow": (250, 204, 21, 255),
    "orange": (249, 115, 22, 255),
    "purple": (168, 85, 247, 255),
    "pink": (236, 72, 153, 255),
    "cyan": (34, 211, 238, 255),
    "teal": (20, 184, 166, 255),
    "gold": (212, 175, 55, 255),
    "navy": (15, 23, 42, 255),
    "slate": (100, 116, 139, 255),
    "indigo": (99, 102, 241, 255),
    "emerald": (16, 185, 129, 255),
    "rose": (244, 63, 94, 255),
    "amber": (245, 158, 11, 255),
}

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{3,8})$")


def parse_color(value: object, default: tuple[int, int, int, int] = (255, 255, 255, 255)) -> tuple[int, int, int, int]:
    """Parse hex strings, CSS names, ``rgb()/rgba()`` and sequences into RGBA.

    Never raises: anything unparseable returns ``default``, which keeps a bad
    model-generated scene from killing a long render.
    """
    if value is None:
        return default

    if isinstance(value, (tuple, list)):
        try:
            parts = [int(round(float(v))) for v in value]
        except (TypeError, ValueError):
            return default
        if len(parts) == 3:
            parts.append(255)
        if len(parts) < 4:
            return default
        return (_c(parts[0]), _c(parts[1]), _c(parts[2]), _c(parts[3]))

    if isinstance(value, str):
        text = value.strip()
        lowered = text.lower()
        if lowered in _NAMED:
            return _NAMED[lowered]

        match = _HEX_RE.match(text)
        if match:
            digits = match.group(1)
            if len(digits) in (3, 4):
                digits = "".join(ch * 2 for ch in digits)
            if len(digits) == 6:
                digits += "ff"
            if len(digits) == 8:
                try:
                    r, g, b, a = (int(digits[i : i + 2], 16) for i in (0, 2, 4, 6))
                    return (r, g, b, a)
                except ValueError:
                    return default
            return default

        fn_match = re.match(r"^rgba?\(([^)]+)\)$", lowered)
        if fn_match:
            chunks = [c.strip() for c in fn_match.group(1).split(",")]
            try:
                rgb = [_component(c) for c in chunks[:3]]
                alpha = _component(chunks[3], alpha=True) if len(chunks) > 3 else 255
                return (rgb[0], rgb[1], rgb[2], alpha)
            except (ValueError, IndexError):
                return default

    return default


def _component(chunk: str, alpha: bool = False) -> int:
    if chunk.endswith("%"):
        return _c(round(float(chunk[:-1]) / 100.0 * 255.0))
    value = float(chunk)
    if alpha and value <= 1.0 and "." in chunk:
        return _c(round(value * 255.0))
    return _c(round(value))


def _c(v: int) -> int:
    return max(0, min(255, int(v)))


def to_hex(color: object) -> str:
    """Return ``#rrggbb`` for any parseable colour (alpha is dropped)."""
    r, g, b, _ = parse_color(color)
    return f"#{r:02x}{g:02x}{b:02x}"


def with_alpha(color: object, alpha: float) -> tuple[int, int, int, int]:
    r, g, b, a = parse_color(color)
    return (r, g, b, _c(round(a * max(0.0, min(1.0, alpha)))))


def mix(a: object, b: object, t: float) -> tuple[int, int, int, int]:
    """Linear blend between two colours; ``t`` is clamped to ``[0, 1]``."""
    t = max(0.0, min(1.0, float(t)))
    ca, cb = parse_color(a), parse_color(b)
    return tuple(_c(round(ca[i] + (cb[i] - ca[i]) * t)) for i in range(4))  # type: ignore[return-value]


def _channel_array(colors: Sequence[object]) -> np.ndarray:
    return np.array([parse_color(c) for c in colors], dtype=np.float64)


def _ramp(stops: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Build a ``(N, 4)`` colour ramp by interpolating through ``stops``."""
    count = 256
    t = np.linspace(0.0, 1.0, count)
    ramp = np.zeros((count, 4), dtype=np.float64)
    for channel in range(4):
        ramp[:, channel] = np.interp(t, positions, stops[:, channel])
    return ramp


def _stop_positions(stops: np.ndarray, explicit: Sequence[float] | None) -> np.ndarray:
    n = len(stops)
    if explicit and len(explicit) == n:
        pos = np.array([max(0.0, min(1.0, float(p))) for p in explicit], dtype=np.float64)
    else:
        pos = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.0, 1.0])
        if n == 1:
            stops = np.vstack([stops, stops])
            pos = np.array([0.0, 1.0])
    # np.interp needs non-decreasing xp
    return np.maximum.accumulate(pos)


def linear_gradient(
    colors: Sequence[object],
    width: int,
    height: int,
    angle: float = 90.0,
    positions: Sequence[float] | None = None,
) -> np.ndarray:
    """Diagonal linear gradient as an ``(H, W, 4)`` uint8 array.

    ``angle`` follows CSS convention: 0deg points up, 90deg points right.
    """
    width, height = max(1, int(width)), max(1, int(height))
    stops = _channel_array(colors)
    ramp = _ramp(stops, _stop_positions(stops, positions))

    radians = math.radians(float(angle) - 90.0)
    dx, dy = math.cos(radians), math.sin(radians)

    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    if width > 1:
        xs /= width - 1
    if height > 1:
        ys /= height - 1
    xs -= 0.5
    ys -= 0.5

    projection = xs * dx + ys * dy
    span = abs(dx) * 0.5 + abs(dy) * 0.5
    if span <= 1e-9:
        norm = np.zeros_like(projection)
    else:
        norm = np.clip((projection / (2.0 * span)) + 0.5, 0.0, 1.0)

    indices = np.clip((norm * 255.0).astype(np.int32), 0, 255)
    return ramp[indices].astype(np.uint8)


def radial_gradient(
    colors: Sequence[object],
    width: int,
    height: int,
    center: tuple[float, float] = (0.5, 0.5),
    radius: float = 0.75,
    positions: Sequence[float] | None = None,
) -> np.ndarray:
    width, height = max(1, int(width)), max(1, int(height))
    stops = _channel_array(colors)
    ramp = _ramp(stops, _stop_positions(stops, positions))

    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    if width > 1:
        xs /= width - 1
    if height > 1:
        ys /= height - 1
    aspect = width / height if height else 1.0
    dx = (xs - center[0]) * aspect
    dy = ys - center[1]
    distance = np.sqrt(dx * dx + dy * dy) / max(1e-6, radius * aspect)
    indices = np.clip((np.clip(distance, 0.0, 1.0) * 255.0).astype(np.int32), 0, 255)
    return ramp[indices].astype(np.uint8)


def conic_gradient(
    colors: Sequence[object],
    width: int,
    height: int,
    center: tuple[float, float] = (0.5, 0.5),
    start_angle: float = 0.0,
) -> np.ndarray:
    width, height = max(1, int(width)), max(1, int(height))
    stops = _channel_array(colors)
    ramp = _ramp(stops, _stop_positions(stops, None))

    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    if width > 1:
        xs /= width - 1
    if height > 1:
        ys /= height - 1
    angle = np.arctan2(ys - center[1], xs - center[0])
    norm = ((angle - math.radians(start_angle)) / (2.0 * math.pi)) % 1.0
    indices = np.clip((norm * 255.0).astype(np.int32), 0, 255)
    return ramp[indices].astype(np.uint8)


def contrast_ratio(a: object, b: object) -> float:
    """WCAG contrast ratio between two colours (1.0 .. 21.0)."""

    def luminance(color: object) -> float:
        r, g, b, _ = parse_color(color)
        channels = []
        for value in (r, g, b):
            v = value / 255.0
            channels.append(v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4)
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    l1, l2 = luminance(a), luminance(b)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def readable_text_color(background: object) -> tuple[int, int, int, int]:
    """Pick black or white text for maximum legibility on ``background``."""
    white = (255, 255, 255, 255)
    black = (17, 24, 39, 255)
    return white if contrast_ratio(background, white) >= contrast_ratio(background, black) else black


def complement(color: object) -> tuple[int, int, int, int]:
    r, g, b, a = parse_color(color)
    h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    r2, g2, b2 = colorsys.hls_to_rgb((h + 0.5) % 1.0, l, s)
    return (round(r2 * 255), round(g2 * 255), round(b2 * 255), a)


def harmonize(base: object, count: int = 4) -> list[tuple[int, int, int, int]]:
    """Generate ``count`` colours evenly spaced around the hue wheel from ``base``."""
    r, g, b, a = parse_color(base)
    h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    out = []
    for i in range(max(1, count)):
        hue = (h + i / max(1, count)) % 1.0
        r2, g2, b2 = colorsys.hls_to_rgb(hue, max(0.25, min(0.8, l)), max(0.4, s))
        out.append((round(r2 * 255), round(g2 * 255), round(b2 * 255), a))
    return out


PALETTES: dict[str, list[str]] = {
    "midnight": ["#0f172a", "#1e293b", "#38bdf8", "#e2e8f0"],
    "sunset": ["#7c2d12", "#ea580c", "#fbbf24", "#fef3c7"],
    "ocean": ["#082f49", "#0369a1", "#22d3ee", "#e0f2fe"],
    "emerald": ["#022c22", "#059669", "#34d399", "#ecfdf5"],
    "royal": ["#1e1b4b", "#4f46e5", "#a78bfa", "#f5f3ff"],
    "rose": ["#4c0519", "#e11d48", "#fb7185", "#fff1f2"],
    "mono": ["#09090b", "#27272a", "#a1a1aa", "#fafafa"],
    "gold": ["#1c1917", "#78350f", "#d4af37", "#fef9c3"],
    "neon": ["#0b0f1a", "#7c3aed", "#22d3ee", "#f0abfc"],
    "sand": ["#292524", "#a8a29e", "#e7e5e4", "#fafaf9"],
    "forest": ["#14532d", "#166534", "#84cc16", "#f7fee7"],
    "candy": ["#581c87", "#db2777", "#fb923c", "#fef9c3"],
}


def palette(name: str) -> list[str]:
    """Return a named palette, falling back to ``midnight``."""
    return PALETTES.get((name or "midnight").lower(), PALETTES["midnight"])


def palette_names() -> list[str]:
    return sorted(PALETTES)


def gradient_colors(spec: Iterable[object] | str | None, fallback: str = "midnight") -> list[str]:
    """Normalise a user/model-supplied gradient spec into a list of colours."""
    if isinstance(spec, str):
        return palette(spec)
    if isinstance(spec, (list, tuple)) and spec:
        return [c if isinstance(c, str) else to_hex(c) for c in spec]
    return palette(fallback)
