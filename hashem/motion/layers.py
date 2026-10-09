"""Layer painters and per-frame transform evaluation.

Two responsibilities:

* :func:`compute_state` turns ``(layer, time)`` into an opacity/offset/scale/
  rotation state by composing the layer's enter, loop and exit animations.
* ``paint_*`` functions rasterise a layer into an RGBA *tile*.  Tiles are cached
  whenever their content is time-invariant, which is what makes a 1080p render
  fast: the expensive glyph rasterisation happens once, and each frame only does
  an affine transform plus an alpha composite.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..utils.log import get_logger
from .colors import conic_gradient, linear_gradient, parse_color, radial_gradient
from .easing import ease
from .fonts import resolve_font, weight_to_synthetic
from .scene import Canvas, Layer, anchor_offset
from .text import measure, render_text, shape_text, typewriter_slice, wrap_text

log = get_logger("motion.layers")


@dataclass
class LayerState:
    opacity: float = 1.0
    dx: float = 0.0
    dy: float = 0.0
    scale: float = 1.0
    rotation: float = 0.0
    progress: float = 1.0   # 0..1 reveal progress, used by typewriter

    def visible(self) -> bool:
        return self.opacity > 0.003 and self.scale > 0.001


# --------------------------------------------------------------------------- #
# animation composition
# --------------------------------------------------------------------------- #

_SLIDE_DIRECTIONS = {
    "up": (0.0, -1.0),
    "down": (0.0, 1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "top": (0.0, -1.0),
    "bottom": (0.0, 1.0),
}


def _enter_state(kind: str, p: float, spec) -> tuple[float, float, float, float, float]:
    """Return ``(opacity, dx, dy, scale, rotation)`` for an enter/exit animation."""
    dx = dy = 0.0
    opacity, scale, rotation = 1.0, 1.0, 0.0

    if kind in {"fade", "fade_in", "fade_out", "dissolve"}:
        opacity = p
    elif kind.startswith("fade_"):
        direction = kind.split("_", 1)[1]
        vx, vy = _SLIDE_DIRECTIONS.get(direction, (0.0, -1.0))
        opacity = p
        dx, dy = vx * spec.distance * (1.0 - p), vy * spec.distance * (1.0 - p)
    elif kind.startswith("slide"):
        direction = kind.split("_", 1)[1] if "_" in kind else "up"
        vx, vy = _SLIDE_DIRECTIONS.get(direction, (0.0, -1.0))
        dx, dy = vx * spec.distance * (1.0 - p), vy * spec.distance * (1.0 - p)
    elif kind in {"zoom_in", "zoom"}:
        scale = spec.from_scale + (1.0 - spec.from_scale) * p
        opacity = min(1.0, p * 1.4)
    elif kind == "zoom_out":
        scale = (2.0 - spec.from_scale) - (1.0 - spec.from_scale) * p
        opacity = min(1.0, p * 1.4)
    elif kind == "pop":
        scale = spec.from_scale + (1.0 - spec.from_scale) * p
        opacity = min(1.0, p * 2.0)
    elif kind in {"rotate_in", "rotate"}:
        rotation = spec.from_rotation * (1.0 - p)
        opacity = p
        scale = spec.from_scale + (1.0 - spec.from_scale) * p
    elif kind == "flip":
        scale = max(0.02, abs(math.cos(math.pi * (1.0 - p))))
        opacity = min(1.0, p * 1.5)
    elif kind == "wipe":
        opacity = 1.0
    elif kind in {"none", "", "instant"}:
        opacity = 1.0
    else:
        opacity = p
    return opacity, dx, dy, scale, rotation


def _loop_state(kind: str, t: float, spec) -> tuple[float, float, float, float, float]:
    """Continuous loops. Returns multipliers/adders, not absolute values."""
    if kind in {"pulse", "breathe"}:
        scale = 1.0 + spec.amplitude * 0.5 * math.sin(2.0 * math.pi * spec.speed * t)
        return 1.0, 0.0, 0.0, scale, 0.0
    if kind in {"float", "hover", "bob"}:
        return 1.0, 0.0, spec.amplitude * 100.0 * math.sin(2.0 * math.pi * spec.speed * t), 1.0, 0.0
    if kind in {"sway", "swing"}:
        return 1.0, 0.0, 0.0, 1.0, spec.amplitude * 60.0 * math.sin(2.0 * math.pi * spec.speed * t)
    if kind in {"glow", "flicker", "blink"}:
        return 1.0 - spec.amplitude * 0.5 * (0.5 + 0.5 * math.sin(2.0 * math.pi * spec.speed * t)), 0.0, 0.0, 1.0, 0.0
    if kind in {"drift", "pan"}:
        return 1.0, spec.amplitude * 120.0 * math.sin(2.0 * math.pi * spec.speed * t * 0.5), 0.0, 1.0, 0.0
    if kind in {"spin", "rotate"}:
        return 1.0, 0.0, 0.0, 1.0, 360.0 * spec.speed * t
    return 1.0, 0.0, 0.0, 1.0, 0.0


def compute_state(layer: Layer, t: float, scene_duration: float) -> LayerState:
    """Evaluate every animation attached to ``layer`` at scene-local time ``t``."""
    opacity = layer.opacity
    dx = dy = 0.0
    scale = 1.0
    rotation = layer.rotation
    progress = 1.0

    enter = layer.anim.enter
    if enter and enter.type not in {"none", "", "instant"}:
        start = enter.delay
        end = start + max(0.0001, enter.duration)
        if t < start:
            p = 0.0
        elif t >= end:
            p = 1.0
        else:
            p = ease(enter.easing, (t - start) / (end - start))
        o, ex, ey, s, r = _enter_state(enter.type, p, enter)
        opacity *= o
        dx += ex
        dy += ey
        scale *= s
        rotation += r
        if enter.type == "typewriter":
            progress = p

    exit_anim = layer.anim.exit
    if exit_anim and exit_anim.type not in {"none", "", "instant"}:
        end = max(0.0, scene_duration - exit_anim.delay)
        start = max(0.0, end - exit_anim.duration)
        if t >= end:
            p = 0.0
        elif t > start:
            p = 1.0 - ease(exit_anim.easing, (t - start) / max(0.0001, end - start))
        else:
            p = 1.0
        o, ex, ey, s, r = _enter_state(exit_anim.type, p, exit_anim)
        opacity *= o
        dx += ex
        dy += ey
        scale *= s
        rotation += r

    loop = layer.anim.loop
    if loop:
        o, lx, ly, s, r = _loop_state(loop.type, max(0.0, t), loop)
        opacity *= o
        dx += lx
        dy += ly
        scale *= s
        rotation += r

    return LayerState(
        opacity=max(0.0, min(1.0, opacity)),
        dx=dx,
        dy=dy,
        scale=max(0.0, scale),
        rotation=rotation,
        progress=progress,
    )


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #


def _resolve_box(layer: Layer, canvas: Canvas) -> tuple[float, float, float, float]:
    """Return ``(cx, cy, w, h)`` in pixels for a layer."""
    w, h = canvas.size
    if layer.unit == "px":
        cx, cy = layer.x, layer.y
        bw = layer.width if layer.width else 0.0
        bh = layer.height if layer.height else 0.0
    else:
        cx, cy = layer.x * w, layer.y * h
        bw = layer.width * w if layer.width else 0.0
        bh = layer.height * h if layer.height else 0.0
    return cx, cy, bw, bh


def _max_width_px(layer: Layer, canvas: Canvas) -> int:
    if layer.unit == "px":
        return int(layer.width or layer.max_width)
    if layer.max_width <= 1.5:  # clearly normalised
        return max(8, int(layer.max_width * canvas.width))
    return max(8, int(layer.max_width))


# --------------------------------------------------------------------------- #
# painters
# --------------------------------------------------------------------------- #


def _shadow_tile(
    tile: Image.Image, color: object, blur: float, offset: tuple[float, float]
) -> tuple[Image.Image, tuple[int, int]]:
    """Build a blurred silhouette behind ``tile``."""
    r, g, b, a = parse_color(color)
    alpha = np.array(tile.split()[-1], dtype=np.float32) * (a / 255.0)
    shadow = Image.fromarray(alpha.astype(np.uint8), mode="L")
    if blur > 0:
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(0.1, blur)))
    layer = Image.new("RGBA", tile.size, (r, g, b, 0))
    layer.putalpha(shadow)
    return layer, (int(offset[0]), int(offset[1]))  # type: ignore[return-value]


def paint_text(layer: Layer, canvas: Canvas, t: float, state: LayerState) -> Image.Image | None:
    raw_text = layer.text or ""
    if state.progress < 1.0:
        raw_text = typewriter_slice(raw_text, state.progress)
    if not raw_text.strip():
        return None

    bold, _ = weight_to_synthetic(layer.font_weight)
    font_path = resolve_font(raw_text, family=layer.font_family or None, bold=bold)
    max_w = _max_width_px(layer, canvas)

    kwargs = dict(
        font_path=font_path,
        size=int(round(layer.font_size)),
        fill=layer.fill,
        letter_spacing=layer.letter_spacing,
        line_spacing=layer.line_spacing,
        align=layer.align,
        max_width=max_w,
    )
    if layer.stroke_color and layer.stroke_width > 0:
        kwargs["stroke_fill"] = layer.stroke_color
        kwargs["stroke_width"] = int(round(layer.stroke_width))

    tile = render_text(raw_text, **kwargs)

    if layer.shadow:
        shadow_tile, offset = _shadow_tile(tile, layer.shadow, layer.shadow_blur, layer.shadow_offset)
        pad = max(abs(offset[0]), abs(offset[1])) + int(layer.shadow_blur * 3) + 4
        combined = Image.new("RGBA", (tile.width + 2 * pad, tile.height + 2 * pad), (0, 0, 0, 0))
        combined.alpha_composite(shadow_tile, (pad + offset[0], pad + offset[1]))
        combined.alpha_composite(tile, (pad, pad))
        tile = combined
    return tile


_SHAPE_GRADIENT_CACHE: dict[tuple, np.ndarray] = {}


def _gradient_for(layer: Layer, size: tuple[int, int]) -> np.ndarray | None:
    if not layer.gradient or len(layer.gradient) < 2:
        return None
    key = (tuple(layer.gradient), size, round(layer.gradient_angle))
    if key not in _SHAPE_GRADIENT_CACHE:
        _SHAPE_GRADIENT_CACHE[key] = linear_gradient(layer.gradient, size[0], size[1], layer.gradient_angle)
    return _SHAPE_GRADIENT_CACHE[key]


def _star_points(cx: float, cy: float, outer: float, inner: float, points: int, rotation: float = -90.0):
    coords = []
    step = math.pi / max(3, points)
    for i in range(max(3, points) * 2):
        radius = outer if i % 2 == 0 else inner
        angle = rotation * math.pi / 180.0 + i * step
        coords.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return coords


def paint_shape(layer: Layer, canvas: Canvas, t: float, state: LayerState) -> Image.Image | None:
    cx, cy, bw, bh = _resolve_box(layer, canvas)
    kind = layer.shape

    if kind in {"circle", "ellipse", "ring"}:
        size = bw or min(canvas.width, canvas.height) * 0.2
        w = int(max(2, size))
        h = int(max(2, bh or size))
    elif kind == "line":
        w = int(max(2, bw or canvas.width * 0.4))
        h = int(max(1, layer.thickness))
    else:
        w = int(max(2, bw or canvas.width * 0.32))
        h = int(max(2, bh or max(24.0, canvas.height * 0.12)))

    w, h = min(w, canvas.width * 4), min(h, canvas.height * 4)
    tile = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    base = parse_color(layer.fill, default=(255, 255, 255, 255))
    width, height = tile.size

    def fill_with_gradient(mask: Image.Image) -> None:
        gradient = _gradient_for(layer, (width, height))
        if gradient is None:
            return
        rgb = Image.fromarray(gradient[..., :3], "RGB").convert("RGBA")
        rgb.putalpha(mask.split()[-1])
        tile.paste(rgb, (0, 0), mask)

    if kind in {"rect", "rectangle", "square"}:
        draw.rectangle([0, 0, width - 1, height - 1], fill=base)
        fill_with_gradient(tile)
    elif kind in {"rounded", "round_rect", "card", "panel"}:
        radius = min(layer.radius, min(width, height) / 2)
        draw.rounded_rectangle([0, 0, width - 1, height - 1], radius=radius, fill=base)
        fill_with_gradient(tile)
    elif kind in {"pill", "badge", "chip"}:
        radius = min(width, height) / 2
        draw.rounded_rectangle([0, 0, width - 1, height - 1], radius=radius, fill=base)
        fill_with_gradient(tile)
    elif kind in {"circle", "ellipse"}:
        draw.ellipse([0, 0, width - 1, height - 1], fill=base)
        fill_with_gradient(tile)
    elif kind == "ring":
        thickness = max(1.0, min(layer.thickness, min(width, height) / 2))
        draw.ellipse([0, 0, width - 1, height - 1], outline=base, width=int(thickness))
    elif kind == "triangle":
        draw.polygon([(width / 2, 0), (width - 1, height - 1), (0, height - 1)], fill=base)
        fill_with_gradient(tile)
    elif kind == "star":
        draw.polygon(_star_points(width / 2, height / 2, min(width, height) / 2, min(width, height) / 4.5, int(layer.points)), fill=base)
        fill_with_gradient(tile)
    elif kind == "polygon":
        n = max(3, int(layer.points))
        pts = [
            (width / 2 + (min(width, height) / 2) * math.cos(-math.pi / 2 + 2 * math.pi * i / n),
             height / 2 + (min(width, height) / 2) * math.sin(-math.pi / 2 + 2 * math.pi * i / n))
            for i in range(n)
        ]
        draw.polygon(pts, fill=base)
        fill_with_gradient(tile)
    elif kind == "line":
        draw.rectangle([0, 0, width - 1, max(0, height - 1)], fill=base)
        fill_with_gradient(tile)
    elif kind == "arrow":
        shaft_h = max(1, int(height * 0.35))
        head_w = int(width * 0.3)
        top = (height - shaft_h) // 2
        draw.rectangle([0, top, width - head_w, top + shaft_h], fill=base)
        draw.polygon([(width - head_w, 0), (width - 1, height / 2), (width - head_w, height - 1)], fill=base)
    elif kind in {"dots", "grid"}:
        step = max(6, int(layer.radius))
        dot = max(1, int(layer.thickness))
        for gx in range(0, width, step):
            for gy in range(0, height, step):
                draw.ellipse([gx, gy, gx + dot, gy + dot], fill=base)
    else:
        draw.rectangle([0, 0, width - 1, height - 1], fill=base)
        fill_with_gradient(tile)

    if layer.border_color and layer.border_width > 0:
        border = parse_color(layer.border_color)
        bw_px = max(1, int(layer.border_width))
        outline_draw = ImageDraw.Draw(tile)
        outline_draw.rectangle([0, 0, width - 1, height - 1], outline=border, width=bw_px)

    if layer.text.strip():
        # shapes can carry a centred label (badges, buttons, counters)
        bold, _ = weight_to_synthetic(layer.font_weight)
        font_path = resolve_font(layer.text, family=layer.font_family or None, bold=bold)
        label = render_text(
            layer.text,
            font_path=font_path,
            size=int(round(layer.font_size)),
            fill=layer.fill,
            align="center",
            max_width=max(8, width - 8),
        )
        canvas_tile = Image.new("RGBA", tile.size, (0, 0, 0, 0))
        canvas_tile.alpha_composite(
            label, (max(0, (width - label.width) // 2), max(0, (height - label.height) // 2))
        )
        tile = Image.alpha_composite(tile, canvas_tile)

    if layer.shadow:
        shadow_tile, offset = _shadow_tile(tile, layer.shadow, layer.shadow_blur, layer.shadow_offset)
        pad = max(abs(offset[0]), abs(offset[1])) + int(layer.shadow_blur * 3) + 4
        combined = Image.new("RGBA", (tile.width + 2 * pad, tile.height + 2 * pad), (0, 0, 0, 0))
        combined.alpha_composite(shadow_tile, (pad + offset[0], pad + offset[1]))
        combined.alpha_composite(tile, (pad, pad))
        tile = combined

    return tile


def paint_image(layer: Layer, canvas: Canvas, t: float, state: LayerState, base_dir=None) -> Image.Image | None:
    from pathlib import Path

    path = layer.image
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute() and base_dir:
        candidate = Path(base_dir) / path
    if not candidate.exists():
        log.warning("Image layer references a missing file: %s", candidate)
        return None
    try:
        source = Image.open(candidate).convert("RGBA")
    except (OSError, ValueError) as exc:
        log.warning("Could not open image %s (%s)", candidate, exc)
        return None

    target_w = int(layer.width * (canvas.width if layer.unit != "px" else 1)) if layer.width else int(canvas.width * 0.4)
    target_h = int(layer.height * (canvas.height if layer.unit != "px" else 1)) if layer.height else 0
    target_w = max(2, min(target_w, canvas.width * 4))

    if layer.fit == "stretch" and target_h:
        return source.resize((target_w, max(2, target_h)), Image.LANCZOS)

    ratio = min(target_w / source.width, (target_h or 10**9) / source.height) if layer.fit == "contain" else max(
        target_w / source.width, (target_h or source.height) / source.height
    )
    new_size = (max(2, int(source.width * ratio)), max(2, int(source.height * ratio)))
    resized = source.resize(new_size, Image.LANCZOS)

    if layer.fit == "cover" and target_h:
        left = max(0, (resized.width - target_w) // 2)
        top = max(0, (resized.height - target_h) // 2)
        resized = resized.crop((left, top, left + target_w, top + target_h))
    return resized


def paint_particles(layer: Layer, canvas: Canvas, t: float, state: LayerState) -> Image.Image | None:
    rng = random.Random(layer.seed)
    count = max(1, min(int(layer.count), 2000))
    width, height = canvas.size
    tile = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    color = parse_color(layer.particle_color, default=(255, 255, 255, 200))

    for i in range(count):
        base_x = rng.random()
        base_y = rng.random()
        size = rng.uniform(3.0, 16.0)
        drift = rng.uniform(-40.0, 40.0)
        speed = rng.uniform(0.05, 0.35) * layer.speed_factor
        phase = rng.uniform(0.0, math.tau)

        y = ((base_y + t * speed) % 1.2) - 0.1
        x = base_x + (drift / width) * math.sin(t * 0.7 + phase)
        px, py = x * width, y * height
        alpha = int(color[3] * (0.35 + 0.65 * abs(math.sin(phase + t * 0.9))))
        fill = (color[0], color[1], color[2], alpha)

        shape = layer.particle_shape
        if shape == "square":
            draw.rectangle([px, py, px + size, py + size], fill=fill)
        elif shape == "star":
            draw.polygon(_star_points(px, py, size, size / 2.4, 5), fill=fill)
        elif shape == "sparkle":
            draw.line([px - size, py, px + size, py], fill=fill, width=max(1, int(size / 4)))
            draw.line([px, py - size, px, py + size], fill=fill, width=max(1, int(size / 4)))
        elif shape == "snow":
            for k in range(3):
                angle = math.pi * k / 3.0
                draw.line(
                    [px - size * math.cos(angle), py - size * math.sin(angle),
                     px + size * math.cos(angle), py + size * math.sin(angle)],
                    fill=fill, width=max(1, int(size / 5)),
                )
        else:
            draw.ellipse([px, py, px + size, py + size], fill=fill)
    return tile


def paint_progress(layer: Layer, canvas: Canvas, t: float, state: LayerState, scene_duration: float) -> Image.Image | None:
    cx, cy, bw, bh = _resolve_box(layer, canvas)
    width = int(max(4, bw or canvas.width * 0.6))
    height = int(max(4, bh or max(8.0, canvas.height * 0.012)))
    ratio = 0.0 if scene_duration <= 0 else max(0.0, min(1.0, t / scene_duration))

    track = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(track)
    radius = height / 2
    track_color = parse_color(layer.border_color or "#ffffff33")
    draw.rounded_rectangle([0, 0, width - 1, height - 1], radius=radius, fill=track_color)

    fill_w = int(width * ratio)
    if fill_w > 2:
        bar = Image.new("RGBA", (fill_w, height), (0, 0, 0, 0))
        bar_draw = ImageDraw.Draw(bar)
        bar_draw.rounded_rectangle([0, 0, fill_w - 1, height - 1], radius=radius, fill=parse_color(layer.fill))
        gradient = _gradient_for(layer, (fill_w, height))
        if gradient is not None:
            rgb = Image.fromarray(gradient[..., :3], "RGB").convert("RGBA")
            bar = Image.alpha_composite(bar, rgb)
        track.alpha_composite(bar, (0, 0))
    return track


def paint_counter(layer: Layer, canvas: Canvas, t: float, state: LayerState, scene_duration: float) -> Image.Image | None:
    span = max(0.0001, layer.anim.enter.duration + layer.anim.enter.delay)
    p = max(0.0, min(1.0, t / span))
    eased = ease(layer.anim.enter.easing or "easeOutCubic", p)
    value = layer.value_from + (layer.value_to - layer.value_from) * eased
    decimals = 1 if abs(layer.value_to) % 1 else 0
    text = f"{layer.prefix}{value:,.{decimals}f}{layer.suffix}"
    clone = Layer(**{**layer.__dict__, "text": text, "kind": "text"})
    return paint_text(clone, canvas, t, state)


PAINTERS: dict[str, Callable[..., Image.Image | None]] = {
    "text": paint_text,
    "shape": paint_shape,
    "image": paint_image,
    "particles": paint_particles,
    "progress": paint_progress,
    "counter": paint_counter,
    "badge": paint_shape,
    "line": paint_shape,
}


def paint_layer(
    layer: Layer,
    canvas: Canvas,
    t: float,
    state: LayerState,
    base_dir=None,
    scene_duration: float = 0.0,
) -> Image.Image | None:
    """Dispatch to the right painter. Never raises — a bad layer is skipped."""
    painter = PAINTERS.get(layer.kind)
    if painter is None:
        log.warning("Unknown layer type %r — skipped", layer.kind)
        return None
    try:
        if layer.kind == "image":
            return painter(layer, canvas, t, state, base_dir)
        if layer.kind in {"progress", "counter"}:
            return painter(layer, canvas, t, state, scene_duration)
        return painter(layer, canvas, t, state)
    except Exception as exc:  # noqa: BLE001 - one broken layer must not kill the render
        log.exception("Layer %r failed to paint (%s) — skipped", layer.id, exc)
        return None


def place(tile: Image.Image, layer: Layer, canvas: Canvas, state: LayerState) -> tuple[Image.Image, tuple[int, int]]:
    """Scale/rotate a tile and return ``(transformed_tile, top_left_position)``."""
    cx, cy, _, _ = _resolve_box(layer, canvas)
    ax, ay = anchor_offset(layer.anchor)

    if abs(state.scale - 1.0) > 0.001:
        new_size = (
            max(1, int(round(tile.width * state.scale))),
            max(1, int(round(tile.height * state.scale))),
        )
        tile = tile.resize(new_size, Image.LANCZOS)

    if abs(state.rotation) > 0.05:
        tile = tile.rotate(-state.rotation, resample=Image.BICUBIC, expand=True)

    left = cx - tile.width * ax + state.dx
    top = cy - tile.height * ay + state.dy
    return tile, (int(round(left)), int(round(top)))


def is_static(layer: Layer) -> bool:
    """True when the tile can be cached for the whole scene."""
    return layer.kind not in {"particles", "progress", "counter"} and layer.anim.enter.type != "typewriter"


__all__ = [
    "LayerState",
    "compute_state",
    "paint_layer",
    "paint_text",
    "paint_shape",
    "paint_image",
    "paint_particles",
    "paint_progress",
    "paint_counter",
    "place",
    "is_static",
    "PAINTERS",
]
