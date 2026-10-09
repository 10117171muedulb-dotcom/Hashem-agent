"""Static graphic design composition.

Everything here is deterministic PIL/numpy work over the same colour, font and
Arabic-shaping foundations the motion engine uses, so Arabic headlines on a
poster look as good as they do in a video.

Each builder returns a PIL ``Image``.  :func:`render_design` is the single
dispatcher the agent calls, taking a loose dict of options.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from ..media import image_ops
from ..motion.colors import gradient_colors, linear_gradient, mix, palette, parse_color, readable_text_color
from ..motion.fonts import resolve_font, weight_to_synthetic
from ..motion.text import fit_font_size, render_text, wrap_text
from ..utils.log import get_logger

log = get_logger("design")

PLATFORM_SIZES = {
    "square": (1080, 1080),
    "instagram": (1080, 1080),
    "story": (1080, 1920),
    "reel": (1080, 1920),
    "facebook": (1200, 630),
    "twitter": (1600, 900),
    "x": (1600, 900),
    "linkedin": (1200, 627),
    "youtube": (1280, 720),
    "thumbnail": (1280, 720),
    "banner": (1500, 500),
    "a4": (1240, 1754),
    "poster": (1080, 1350),
}

_DECOR = {"circles", "grid", "rays", "bokeh", "none", "diagonal", "dots"}


def design_kinds() -> list[str]:
    return ["poster", "social", "thumbnail", "quote", "logo", "banner", "book_cover"]


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #


def _background(size: tuple[int, int], style: str, colors: list[str], seed: int = 7) -> Image.Image:
    width, height = size
    palette_colors = gradient_colors(colors, "midnight")
    style = (style or "gradient").lower()

    if style == "solid":
        return Image.new("RGB", size, parse_color(palette_colors[0])[:3])
    if style == "radial":
        from ..motion.colors import radial_gradient

        return Image.fromarray(radial_gradient(palette_colors, width, height)[..., :3], "RGB")
    if style == "duotone":
        base = linear_gradient([palette_colors[0], palette_colors[1] if len(palette_colors) > 1 else palette_colors[0]], width, height, 90)
        return Image.fromarray(base[..., :3], "RGB")
    if style in _DECOR or True:
        array = linear_gradient(palette_colors, width, height, random.Random(seed).choice([90, 135, 180]))
        image = Image.fromarray(array[..., :3], "RGB")
        overlay = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        rng = random.Random(seed)
        accent = parse_color(palette_colors[-1] if palette_colors else "#ffffff")

        if style in {"circles", "bokeh", "dots"}:
            count = 14 if style != "dots" else 60
            for _ in range(count):
                r = rng.randint(20, width // 6) if style != "dots" else rng.randint(3, 8)
                x, y = rng.randint(-r, width), rng.randint(-r, height)
                alpha = rng.randint(18, 70) if style != "dots" else rng.randint(40, 90)
                fill = (accent[0], accent[1], accent[2], alpha)
                if style == "bokeh":
                    blob = Image.new("RGBA", (r * 2, r * 2), (0, 0, 0, 0))
                    ImageDraw.Draw(blob).ellipse([0, 0, r * 2, r * 2], fill=fill)
                    blob = blob.filter(ImageFilter.GaussianBlur(r // 3 + 1))
                    overlay.alpha_composite(blob, (x - r, y - r))
                else:
                    draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)
        elif style == "grid":
            step = max(40, width // 14)
            for gx in range(0, width, step):
                draw.line([gx, 0, gx, height], fill=(255, 255, 255, 16), width=1)
            for gy in range(0, height, step):
                draw.line([0, gy, width, gy], fill=(255, 255, 255, 16), width=1)
        elif style == "rays":
            cx, cy = width // 2, int(height * 0.4)
            for i in range(12):
                angle = i * math.pi / 6
                length = max(width, height) * 1.5
                draw.polygon(
                    [
                        (cx, cy),
                        (cx + length * math.cos(angle), cy + length * math.sin(angle)),
                        (cx + length * math.cos(angle + 0.18), cy + length * math.sin(angle + 0.18)),
                    ],
                    fill=(255, 255, 255, 14),
                )
        elif style == "diagonal":
            for i in range(-height, width, max(30, width // 12)):
                draw.line([i, 0, i + height, height], fill=(255, 255, 255, 20), width=max(6, width // 90))

        return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _fit_text(
    text: str,
    max_width: int,
    max_height: int,
    fill: str,
    weight: str = "bold",
    family: str = "",
    align: str = "center",
    stroke_color: str = "",
    stroke_width: int = 0,
    shadow: str = "",
    start: int = 190,
) -> Image.Image:
    bold, _ = weight_to_synthetic(weight)
    font_path = resolve_font(text, family=family or None, bold=bold)
    size = fit_font_size(text, font_path, max_width, max_height, start=start)
    return render_text(
        text,
        font_path=font_path,
        size=size,
        fill=fill,
        align=align,
        max_width=max_width,
        stroke_fill=stroke_color or None,
        stroke_width=stroke_width,
        shadow=shadow or None,
        shadow_offset=(0, int(size * 0.12)) if shadow else (0, 0),
    )


def _paste_centered(base: Image.Image, tile: Image.Image, cx: float, cy: float) -> None:
    left = int(cx * base.width - tile.width / 2)
    top = int(cy * base.height - tile.height / 2)
    base.paste(tile, (left, top), tile if tile.mode == "RGBA" else None)


def _fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image_ops.resize(image, size[0], size[1], fit="cover")


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #


def render_poster(spec: dict[str, Any]) -> Image.Image:
    size = tuple(spec.get("size") or PLATFORM_SIZES.get(str(spec.get("platform", "poster")), PLATFORM_SIZES["poster"]))
    style = spec.get("style", "gradient")
    colors = gradient_colors(spec.get("palette") or spec.get("colors"), "midnight")
    seed = int(spec.get("seed", 7))

    base = _background(size, style, colors, seed)

    image_path = spec.get("image")
    if image_path and Path(image_path).exists():
        photo = image_ops.load_image(image_path)
        photo = _fit_image(photo, size)
        photo = ImageEnhance.Brightness(photo).enhance(0.75)
        base = Image.blend(base.convert("RGB"), photo.convert("RGB"), 0.55)
        scrim = Image.new("RGBA", size, (0, 0, 0, 0))
        scrim_draw = ImageDraw.Draw(scrim)
        for i in range(size[1]):
            alpha = int(200 * (i / size[1]))
            scrim_draw.line([0, i, size[0], i], fill=(5, 8, 18, alpha))
        base = Image.alpha_composite(base.convert("RGBA"), scrim).convert("RGB")

    width, height = base.size
    text_color = "#ffffff"
    title = str(spec.get("title", spec.get("text", "")))
    subtitle = str(spec.get("subtitle", ""))
    tag = str(spec.get("tag", spec.get("badge", "")))

    if tag:
        tag_tile = render_text(tag, font_path=resolve_font(tag), size=max(20, width // 34),
                               fill=colors[-1], align="center", padding=18)
        pad = Image.new("RGBA", (tag_tile.width + 48, tag_tile.height + 24), (0, 0, 0, 0))
        ImageDraw.Draw(pad).rounded_rectangle([0, 0, pad.width - 1, pad.height - 1], radius=pad.height // 2, fill=(255, 255, 255, 40))
        pad.alpha_composite(tag_tile, (24, 12))
        _paste_centered(base, pad, 0.5, 0.16)

    if title:
        title_tile = _fit_text(title, int(width * 0.82), int(height * 0.34), text_color,
                               weight=spec.get("weight", "bold"), shadow="#000000aa", align="center")
        _paste_centered(base, title_tile, 0.5, 0.52 if not image_path else 0.62)

    if subtitle:
        sub_tile = _fit_text(subtitle, int(width * 0.7), int(height * 0.14), "#e2e8f0",
                             weight="regular", align="center")
        _paste_centered(base, sub_tile, 0.5, 0.8 if not image_path else 0.78)

    if spec.get("brand"):
        brand = str(spec["brand"])
        brand_tile = render_text(brand, font_path=resolve_font(brand), size=max(16, width // 44), fill="#cbd5e1")
        _paste_centered(base, brand_tile, 0.5, 0.94)

    return base


def render_social(spec: dict[str, Any]) -> Image.Image:
    platform = str(spec.get("platform", "square"))
    merged = {**spec}
    merged.setdefault("size", PLATFORM_SIZES.get(platform, PLATFORM_SIZES["square"]))
    merged.setdefault("style", spec.get("style", "circles"))
    return render_poster(merged)


def render_thumbnail(spec: dict[str, Any]) -> Image.Image:
    merged = {**spec}
    merged["size"] = tuple(spec.get("size") or PLATFORM_SIZES["thumbnail"])
    merged.setdefault("style", spec.get("style", "rays"))
    merged.setdefault("weight", "bold")
    base = render_poster(merged)

    if spec.get("accent") or spec.get("highlight"):
        width, height = base.size
        highlight = str(spec.get("highlight", ""))
        if highlight:
            hl = _fit_text(highlight, int(width * 0.7), int(height * 0.22), "#fbbf24",
                           weight="bold", stroke_color="#000000", stroke_width=max(2, width // 220), align="center")
            _paste_centered(base, hl, 0.5, 0.8)
    return base


def render_quote(spec: dict[str, Any]) -> Image.Image:
    size = tuple(spec.get("size") or (1080, 1080))
    colors = gradient_colors(spec.get("palette") or spec.get("colors"), "royal")
    base = _background(size, spec.get("style", "gradient"), colors, int(spec.get("seed", 3)))
    width, height = base.size

    quote = str(spec.get("text", spec.get("quote", "")))
    author = str(spec.get("author", spec.get("by", "")))

    # giant decorative quotation mark
    mark_font = resolve_font('"')
    mark = render_text("❞" if not quote else '"', font_path=mark_font, size=int(height * 0.24),
                       fill=(255, 255, 255, 70))
    _paste_centered(base, mark, 0.14, 0.16)

    if quote:
        quote_tile = _fit_text(quote, int(width * 0.76), int(height * 0.5), "#ffffff", weight="bold", align="center")
        _paste_centered(base, quote_tile, 0.5, 0.48)

    if author:
        author_tile = render_text(f"— {author}", font_path=resolve_font(author), size=max(22, width // 30), fill=colors[-1])
        _paste_centered(base, author_tile, 0.5, 0.82)
    return base


def render_logo(spec: dict[str, Any]) -> Image.Image:
    size = tuple(spec.get("size") or (800, 800))
    colors = gradient_colors(spec.get("palette") or spec.get("colors"), "neon")
    name = str(spec.get("text", spec.get("name", "H")))
    shape = str(spec.get("shape", "circle"))
    transparent = bool(spec.get("transparent", True))

    width, height = size
    if transparent:
        base = Image.new("RGBA", size, (0, 0, 0, 0))
    else:
        base = _background(size, "solid", ["#0f172a"]).convert("RGBA")
    draw = ImageDraw.Draw(base)

    cx, cy = width / 2, height / 2
    radius = min(width, height) * 0.4
    a, b = parse_color(colors[0]), parse_color(colors[-1] if len(colors) > 1 else colors[0])
    gradient_array = linear_gradient(colors, width, height, 135)
    grad_image = Image.fromarray(gradient_array[..., :3], "RGB").convert("RGBA")

    mask = Image.new("L", size, 0)
    mask_draw = ImageDraw.Draw(mask)
    if shape == "hexagon":
        points = [
            (cx + radius * math.cos(math.radians(60 * i - 30)), cy + radius * math.sin(math.radians(60 * i - 30)))
            for i in range(6)
        ]
        mask_draw.polygon(points, fill=255)
    elif shape in {"square", "rounded"}:
        pad = radius * 0.2
        mask_draw.rounded_rectangle(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            radius=int(radius * (0.3 if shape == "rounded" else 0.05)), fill=255,
        )
    elif shape == "star":
        points = []
        for i in range(10):
            r = radius if i % 2 == 0 else radius * 0.5
            angle = math.radians(36 * i - 90)
            points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        mask_draw.polygon(points, fill=255)
    else:
        mask_draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=255)

    ring = Image.new("RGBA", size, (0, 0, 0, 0))
    ring_draw = ImageDraw.Draw(ring)
    ring_draw.ellipse([cx - radius - 8, cy - radius - 8, cx + radius + 8, cy + radius + 8],
                      outline=(255, 255, 255, 90), width=6)

    grad_image.putalpha(mask)
    base = Image.alpha_composite(base, grad_image)
    if not transparent:
        base = Image.alpha_composite(base, ring)

    initial = (name or "H").strip()
    monogram = _fit_text(initial, int(radius * 1.2), int(radius * 1.2),
                         readable_text_color(colors[0]), weight="bold", align="center")
    _paste_centered(base, monogram, 0.5, 0.5)

    if spec.get("wordmark"):
        word = str(spec["wordmark"])
        word_tile = render_text(word, font_path=resolve_font(word), size=int(height * 0.1), fill="#e2e8f0")
        _paste_centered(base, word_tile, 0.5, 0.93)

    return base.convert("RGBA") if transparent else base.convert("RGB")


def render_banner(spec: dict[str, Any]) -> Image.Image:
    size = tuple(spec.get("size") or PLATFORM_SIZES["banner"])
    colors = gradient_colors(spec.get("palette") or spec.get("colors"), "ocean")
    base = _background(size, spec.get("style", "diagonal"), colors, int(spec.get("seed", 5)))
    width, height = base.size
    title = str(spec.get("title", spec.get("text", "")))
    if title:
        tile = _fit_text(title, int(width * 0.86), int(height * 0.6), "#ffffff", weight="bold")
        _paste_centered(base, tile, 0.5, 0.5)
    if spec.get("subtitle"):
        sub = str(spec["subtitle"])
        sub_tile = render_text(sub, font_path=resolve_font(sub), size=max(20, height // 12), fill="#dbeafe")
        _paste_centered(base, sub_tile, 0.5, 0.78)
    return base


def render_book_cover(spec: dict[str, Any]) -> Image.Image:
    size = tuple(spec.get("size") or (1240, 1754))
    colors = gradient_colors(spec.get("palette") or spec.get("colors"), "midnight")
    base = _background(size, spec.get("style", "gradient"), colors, int(spec.get("seed", 11)))
    width, height = base.size

    # decorative frame
    frame = Image.new("RGBA", size, (0, 0, 0, 0))
    frame_draw = ImageDraw.Draw(frame)
    accent = parse_color(colors[-1] if colors else "#ffffff")
    inset = int(width * 0.06)
    frame_draw.rectangle([inset, inset, width - inset, height - inset], outline=(accent[0], accent[1], accent[2], 160), width=4)
    frame_draw.rectangle([inset + 14, inset + 14, width - inset - 14, height - inset - 14], outline=(accent[0], accent[1], accent[2], 80), width=2)
    base = Image.alpha_composite(base.convert("RGBA"), frame).convert("RGB")

    title = str(spec.get("title", spec.get("text", "")))
    author = str(spec.get("author", ""))
    subtitle = str(spec.get("subtitle", ""))

    if title:
        title_tile = _fit_text(title, int(width * 0.72), int(height * 0.3), "#ffffff", weight="bold", align="center")
        _paste_centered(base, title_tile, 0.5, 0.4)
    if subtitle:
        sub_tile = _fit_text(subtitle, int(width * 0.6), int(height * 0.08), colors[-1], weight="regular", align="center")
        _paste_centered(base, sub_tile, 0.5, 0.56)
    if author:
        author_tile = render_text(author, font_path=resolve_font(author), size=max(24, width // 26), fill="#e2e8f0")
        _paste_centered(base, author_tile, 0.5, 0.86)
    return base


_BUILDERS = {
    "poster": render_poster,
    "social": render_social,
    "thumbnail": render_thumbnail,
    "quote": render_quote,
    "logo": render_logo,
    "banner": render_banner,
    "book_cover": render_book_cover,
    "cover": render_book_cover,
}


def render_design(kind: str, spec: dict[str, Any]) -> Image.Image:
    """Dispatch a design request. Unknown kinds fall back to a poster."""
    builder = _BUILDERS.get((kind or "poster").lower(), render_poster)
    return builder(spec or {})


__all__ = [
    "PLATFORM_SIZES",
    "design_kinds",
    "render_design",
    "render_poster",
    "render_social",
    "render_thumbnail",
    "render_quote",
    "render_logo",
    "render_banner",
    "render_book_cover",
]
