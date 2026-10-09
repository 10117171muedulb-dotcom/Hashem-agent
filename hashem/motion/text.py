"""Text measurement, Arabic shaping and rasterisation.

PIL has no complex text shaping and (without libraqm) no bidirectional engine, so
Arabic is handled the way every offline renderer does it:

1. ``arabic_reshaper`` converts logical characters to their contextual
   presentation forms (initial/medial/final/isolated).
2. ``python-bidi`` reorders the line into visual order.
3. PIL then draws it left-to-right, which is now correct.

Both steps are optional imports: if they are missing the text still renders, just
unshaped — better a slightly wrong glyph run than a crashed render.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

from ..utils.log import get_logger
from .colors import parse_color
from .fonts import contains_arabic, resolve_font

log = get_logger("motion.text")

try:  # pragma: no cover - import guard
    import arabic_reshaper

    _HAS_RESHAPER = True
except Exception:  # pragma: no cover
    arabic_reshaper = None
    _HAS_RESHAPER = False

try:  # pragma: no cover - import guard
    from bidi.algorithm import get_display as _bidi_display

    _HAS_BIDI = True
except Exception:  # pragma: no cover
    _bidi_display = None
    _HAS_BIDI = False

# Characters that carry no advance width of their own and must never be left
# dangling at the start of a wrapped line.
_NO_START = set("،؛،.)]}»'\"٪%،!?؟:-")
_NO_END = set("([«'\"")


def has_arabic_support() -> bool:
    """True when full Arabic shaping + bidi is available."""
    return _HAS_RESHAPER and _HAS_BIDI


def shape_text(text: str) -> str:
    """Return ``text`` in visual order, ready for PIL.

    Latin-only text is returned untouched so nothing regresses for English scenes.
    """
    if not text:
        return ""
    if not contains_arabic(text):
        return text
    out = text
    if _HAS_RESHAPER:
        try:
            out = arabic_reshaper.reshape(out)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("arabic_reshaper failed (%s); using raw text", exc)
    if _HAS_BIDI:
        try:
            out = _bidi_display(out)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("python-bidi failed (%s); using reshaped text only", exc)
    return out


@lru_cache(maxsize=512)
def load_font(path: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Cached font loader. ``size`` is clamped so a bad scene cannot exhaust memory."""
    size = max(4, min(int(size), 600))
    if path:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, ValueError) as exc:
            log.warning("Could not load font %s (%s); falling back to default", path, exc)
    return ImageFont.load_default(size=size)


@dataclass
class TextMetrics:
    width: int
    height: int
    ascent: int
    descent: int

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


def measure(text: str, font_path: str, size: int, letter_spacing: float = 0.0) -> TextMetrics:
    """Measure a single logical line (no wrapping) including letter spacing."""
    font = load_font(font_path, size)
    shaped = shape_text(text)
    if not shaped:
        return TextMetrics(0, 0, 0, 0)
    try:
        left, top, right, bottom = font.getbbox(shaped)
        advance = int(font.getlength(shaped))
    except AttributeError:  # pragma: no cover - very old PIL
        left, top = 0, 0
        right, bottom = font.getsize(shaped)
        advance = right
    extra = int(round(letter_spacing * max(0, len(shaped) - 1)))
    return TextMetrics(
        width=max(0, advance + extra),
        height=max(1, bottom - top),
        ascent=max(0, -top),
        descent=max(0, bottom),
    )


def wrap_text(
    text: str,
    font_path: str,
    size: int,
    max_width: int,
    letter_spacing: float = 0.0,
) -> list[str]:
    """Greedy word wrap that respects explicit newlines and never overflows.

    Words longer than ``max_width`` are hard-split so a long URL or an unbroken
    Arabic word cannot blow past the canvas.
    """
    if max_width <= 0:
        return [line for line in (text or "").split("\n")]

    lines: list[str] = []
    for paragraph in (text or "").split("\n"):
        paragraph = paragraph.rstrip()
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for word in paragraph.split(" "):
            candidate = f"{current} {word}".strip() if current else word
            if measure(candidate, font_path, size, letter_spacing).width <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
                current = ""
            if measure(word, font_path, size, letter_spacing).width > max_width:
                chunk = ""
                for char in word:
                    if measure(chunk + char, font_path, size, letter_spacing).width > max_width and chunk:
                        lines.append(chunk)
                        chunk = char
                    else:
                        chunk += char
                current = chunk
            else:
                current = word
        if current:
            lines.append(current)
    return lines or [""]


def fit_font_size(
    text: str,
    font_path: str,
    max_width: int,
    max_height: int,
    start: int = 200,
    minimum: int = 10,
    line_spacing: float = 1.2,
    letter_spacing: float = 0.0,
) -> int:
    """Largest font size that fits ``text`` inside the box (binary search)."""
    lo, hi = minimum, max(minimum, int(start))
    best = minimum
    while lo <= hi:
        mid = (lo + hi) // 2
        lines = wrap_text(text, font_path, mid, max_width, letter_spacing)
        height = int(mid * line_spacing * len(lines))
        if height <= max_height:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def render_text(
    text: str,
    *,
    font_path: str,
    size: int,
    fill: object = "#ffffff",
    stroke_fill: object | None = None,
    stroke_width: int = 0,
    letter_spacing: float = 0.0,
    line_spacing: float = 1.2,
    align: str = "center",
    max_width: int | None = None,
    shadow: object | None = None,
    shadow_offset: tuple[int, int] = (0, 0),
    padding: int = 0,
) -> Image.Image:
    """Rasterise ``text`` onto a transparent RGBA image sized to its content."""
    resolved_path = font_path or resolve_font(text)
    font = load_font(resolved_path, size)

    width_limit = int(max_width) if max_width and max_width > 0 else 100000
    lines = wrap_text(text, resolved_path, size, width_limit, letter_spacing)

    line_metrics = [measure(line, resolved_path, size, letter_spacing) for line in lines]
    text_width = max((m.width for m in line_metrics), default=0)
    line_height = int(size * line_spacing)
    text_height = max(1, line_height * len(lines))

    stroke_width = max(0, int(stroke_width))
    pad = max(0, int(padding)) + stroke_width
    shadow_pad = 0
    if shadow is not None:
        shadow_pad = max(abs(shadow_offset[0]), abs(shadow_offset[1])) + 4

    canvas_w = text_width + 2 * (pad + shadow_pad)
    canvas_h = text_height + 2 * (pad + shadow_pad)
    image = Image.new("RGBA", (max(1, canvas_w), max(1, canvas_h)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    fill_rgba = parse_color(fill)

    for index, (line, metrics) in enumerate(zip(lines, line_metrics)):
        if not line:
            continue
        y = pad + int(index * line_height + (line_height - size) / 2)
        if align == "right":
            x = pad + text_width - metrics.width
        elif align == "left":
            x = pad
        else:
            x = pad + (text_width - metrics.width) // 2

        shaped = shape_text(line)

        if shadow is not None:
            shadow_color = parse_color(shadow)
            _draw_letterspaced(
                draw, shaped, (x + shadow_offset[0], y + shadow_offset[1]), font,
                shadow_color, letter_spacing, stroke_fill=None, stroke_width=0,
            )

        _draw_letterspaced(
            draw, shaped, (x, y), font, fill_rgba, letter_spacing,
            stroke_fill=parse_color(stroke_fill) if stroke_fill is not None else None,
            stroke_width=stroke_width,
        )

    return image


def _draw_letterspaced(
    draw: ImageDraw.ImageDraw,
    shaped: str,
    position: tuple[int, int],
    font,
    fill,
    letter_spacing: float,
    stroke_fill=None,
    stroke_width: int = 0,
) -> None:
    """Draw a line, optionally with per-character tracking.

    Character-by-character drawing is only used when tracking is requested; the
    fast path keeps whole-string rendering (which preserves hinting).
    """
    x, y = int(position[0]), int(position[1])
    kwargs = {}
    if stroke_fill is not None and stroke_width > 0:
        kwargs = {"stroke_width": stroke_width, "stroke_fill": stroke_fill}

    if letter_spacing <= 0:
        draw.text((x, y), shaped, font=font, fill=fill, **kwargs)
        return

    cursor = x
    for char in shaped:
        draw.text((cursor, y), char, font=font, fill=fill, **kwargs)
        try:
            cursor += int(round(font.getlength(char))) + int(letter_spacing)
        except AttributeError:  # pragma: no cover - very old PIL
            cursor += int(font.getsize(char)[0]) + int(letter_spacing)


def typewriter_slice(text: str, progress: float) -> str:
    """Return the prefix of ``text`` revealed by ``progress`` in ``[0, 1]``."""
    if progress <= 0:
        return ""
    if progress >= 1:
        return text
    count = int(round(len(text) * max(0.0, min(1.0, progress))))
    return text[:count]


def split_words(text: str) -> list[str]:
    """Word list used by per-word stagger animations."""
    return [w for w in (text or "").split(" ") if w]


def estimate_reading_time(text: str, wpm: float = 160.0) -> float:
    """Seconds a viewer needs to read ``text`` — used for auto scene durations."""
    words = len((text or "").split())
    if words == 0:
        return 1.5
    return max(1.2, words / max(20.0, wpm) * 60.0)


__all__ = [
    "shape_text",
    "has_arabic_support",
    "load_font",
    "measure",
    "TextMetrics",
    "wrap_text",
    "fit_font_size",
    "render_text",
    "typewriter_slice",
    "split_words",
    "estimate_reading_time",
]
