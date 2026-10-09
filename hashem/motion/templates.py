"""Auto-composers: turn a topic + a few bullet points into a finished scene graph.

These matter because a 3–7B local model is far more reliable at producing a
title and a bullet list than a perfect 200-line JSON document.  The agent can
therefore call ``create_video`` with ``topic`` + ``points`` and get a polished,
correctly-animated project for free.
"""

from __future__ import annotations

import random
from typing import Any

from .colors import PALETTES, palette_names
from .scene import parse_project

_TRANSITIONS = ["fade", "slide_left", "slide_right", "wipe_right", "zoom", "circle", "blur"]
_LOOPS = ["float", "pulse", "sway", "glow"]


def template_names() -> list[str]:
    return ["intro_points_outro", "promo", "quote_reel", "countdown", "story"]


def _pick(rng: random.Random, seq, default=None):
    return rng.choice(seq) if seq else default


def compose_project(
    topic: str,
    points: list[str] | None = None,
    subtitle: str = "",
    palette: str = "",
    style: str = "gradient",
    preset: str = "1080p",
    fps: int = 30,
    scene_duration: float = 3.2,
    with_music: bool = True,
    music_style: str = "corporate",
    seed: int | None = None,
    outro: str = "",
    brand: str = "",
) -> dict[str, Any]:
    """Build a full project dict from high-level inputs."""
    rng = random.Random(seed if seed is not None else abs(hash(topic)) % 10_000)
    palette = palette or rng.choice(palette_names())
    colors = PALETTES.get(palette, PALETTES["midnight"])
    points = [p for p in (points or []) if str(p).strip()]

    scenes: list[dict[str, Any]] = []

    # ---------------------------------------------------------- intro
    intro_layers = [
        {
            "type": "particles", "count": 36, "particle_color": "#ffffff40", "z": -1,
            "speed_factor": 0.8, "seed": rng.randint(1, 9999),
        },
        {
            "type": "shape", "shape": "ring", "width": 0.16, "height": 0.16,
            "fill": colors[2] if len(colors) > 2 else "#ffffff", "thickness": 6, "y": 0.3,
            "animation": {"enter": {"type": "pop", "duration": 0.7, "easing": "easeOutElastic"}, "loop": {"type": "spin", "speed": 0.15}},
        },
        {
            "type": "text", "text": topic, "y": 0.48, "font_size": 86, "font_weight": "bold",
            "fill": "#ffffff", "max_width": 0.84, "shadow": "#000000aa",
            "animation": {"enter": {"type": "fade_up", "duration": 0.8, "easing": "easeOutBack"}},
        },
    ]
    if subtitle:
        intro_layers.append({
            "type": "text", "text": subtitle, "y": 0.66, "font_size": 40, "fill": colors[3] if len(colors) > 3 else "#e2e8f0",
            "max_width": 0.8, "animation": {"enter": {"type": "fade_up", "duration": 0.7, "delay": 0.35}},
        })
    scenes.append({"duration": scene_duration + 0.6, "transition": {"type": "fade", "duration": 0.4}, "layers": intro_layers})

    # ---------------------------------------------------------- points
    for index, point in enumerate(points):
        progress = (index + 1) / max(1, len(points))
        layers = [
            {
                "type": "shape", "shape": "pill", "x": 0.5, "y": 0.22, "width": 0.16, "height": 0.09,
                "gradient": [colors[2], colors[1]], "text": f"{index + 1} / {len(points)}",
                "font_size": 30, "fill": "#ffffff",
                "animation": {"enter": {"type": "pop", "duration": 0.5, "easing": "easeOutBack"}},
            },
            {
                "type": "text", "text": point, "y": 0.5, "font_size": 54, "font_weight": "bold",
                "fill": "#ffffff", "max_width": 0.8, "align": "center", "shadow": "#00000088",
                "animation": {"enter": {"type": "fade_up", "duration": 0.7, "easing": "easeOutCubic"}, "loop": {"type": "float", "amplitude": 0.012, "speed": 0.4}},
            },
            {"type": "progress", "y": 0.85, "width": 0.55, "gradient": [colors[2], colors[3] if len(colors) > 3 else colors[2]]},
        ]
        scenes.append({
            "duration": scene_duration,
            "transition": {"type": _pick(rng, _TRANSITIONS, "fade"), "duration": 0.5},
            "layers": layers,
        })

    # ---------------------------------------------------------- outro
    outro_text = outro or (brand or topic)
    outro_layers = [
        {"type": "particles", "count": 50, "particle_color": colors[2] + "66", "particle_shape": "sparkle", "seed": rng.randint(1, 9999), "z": -1},
        {
            "type": "text", "text": "شكراً للمشاهدة" if not outro else outro_text, "y": 0.42,
            "font_size": 72, "font_weight": "bold", "fill": "#ffffff", "max_width": 0.8,
            "animation": {"enter": {"type": "zoom_in", "duration": 0.8, "easing": "easeOutBack"}},
        },
        {
            "type": "shape", "shape": "pill", "y": 0.62, "width": 0.4, "height": 0.085,
            "gradient": [colors[2], colors[3] if len(colors) > 3 else colors[2]],
            "text": "تابعنا للمزيد", "font_size": 28, "fill": "#ffffff",
            "animation": {"enter": {"type": "pop", "duration": 0.6, "delay": 0.4, "easing": "easeOutElastic"}},
        },
    ]
    scenes.append({"duration": scene_duration, "transition": {"type": "fade", "duration": 0.5}, "layers": outro_layers})

    project = {
        "name": topic,
        "canvas": _preset_size(preset, fps),
        "background": {"type": "gradient", "colors": colors, "angle": 135, "animated": True, "speed": 10, "vignette": 0.3},
        "scenes": scenes,
        "audio": {"music": {"enabled": bool(with_music), "style": music_style, "volume": 0.28}},
        "meta": {"palette": palette, "template": "intro_points_outro", "generator": "hashem"},
    }
    return project


def compose_promo(
    product: str,
    tagline: str = "",
    price: str = "",
    palette: str = "gold",
    preset: str = "1080p",
    fps: int = 30,
    **_ignored: Any,
) -> dict[str, Any]:
    """A punchy 3-scene promo."""
    colors = PALETTES.get(palette, PALETTES["gold"])
    scenes = [
        {
            "duration": 2.4, "transition": {"type": "zoom", "duration": 0.5},
            "layers": [
                {"type": "particles", "count": 40, "particle_color": "#ffd70055", "particle_shape": "sparkle", "z": -1, "seed": 42},
                {"type": "text", "text": product, "y": 0.45, "font_size": 96, "font_weight": "bold", "fill": "#ffffff", "max_width": 0.85,
                 "shadow": "#000000cc", "animation": {"enter": {"type": "pop", "duration": 0.7, "easing": "easeOutElastic"}, "loop": {"type": "glow", "amplitude": 0.1, "speed": 1.2}}},
            ],
        },
        {
            "duration": 2.4, "transition": {"type": "slide_left", "duration": 0.5},
            "layers": [
                {"type": "text", "text": tagline or "جودة لا تُضاهى", "y": 0.42, "font_size": 60, "fill": colors[3] if len(colors) > 3 else "#fff", "max_width": 0.8,
                 "animation": {"enter": {"type": "fade_up", "duration": 0.7}}},
                {"type": "shape", "shape": "star", "width": 0.14, "height": 0.14, "fill": colors[2], "y": 0.68, "points": 5,
                 "animation": {"enter": {"type": "pop", "duration": 0.6, "delay": 0.3, "easing": "easeOutBack"}, "loop": {"type": "spin", "speed": 0.2}}},
            ],
        },
        {
            "duration": 2.6, "transition": {"type": "fade", "duration": 0.5},
            "layers": [
                {"type": "text", "text": price or "اطلبه الآن", "y": 0.4, "font_size": 84, "font_weight": "bold", "fill": colors[2], "max_width": 0.8,
                 "animation": {"enter": {"type": "zoom_in", "duration": 0.7, "easing": "easeOutBack"}}},
                {"type": "shape", "shape": "pill", "y": 0.64, "width": 0.45, "height": 0.1, "gradient": [colors[2], colors[1]], "text": "اشترِ الآن", "font_size": 32, "fill": "#0b0f1a",
                 "animation": {"enter": {"type": "pop", "duration": 0.6, "delay": 0.4, "easing": "easeOutElastic"}}},
            ],
        },
    ]
    return {
        "name": product,
        "canvas": _preset_size(preset, fps),
        "background": {"type": "radial", "colors": [colors[0], colors[1]], "animated": False, "vignette": 0.4},
        "scenes": scenes,
        "audio": {"music": {"enabled": True, "style": "upbeat", "volume": 0.3}},
        "meta": {"template": "promo", "palette": palette},
    }


def compose_quote_reel(quote: str, author: str = "", palette: str = "royal", preset: str = "vertical", fps: int = 30, **_ignored: Any) -> dict[str, Any]:
    colors = PALETTES.get(palette, PALETTES["royal"])
    layers = [
        {"type": "particles", "count": 30, "particle_color": "#ffffff33", "particle_shape": "bokeh", "z": -1, "seed": 7},
        {"type": "text", "text": "❞", "y": 0.18, "font_size": 140, "fill": colors[2] + "aa"},
        {"type": "text", "text": quote, "y": 0.48, "font_size": 52, "fill": "#ffffff", "max_width": 0.8, "line_spacing": 1.5,
         "animation": {"enter": {"type": "typewriter", "duration": 2.2, "easing": "linear"}}},
        {"type": "text", "text": f"— {author}" if author else "", "y": 0.78, "font_size": 34, "fill": colors[2],
         "animation": {"enter": {"type": "fade_up", "duration": 0.7, "delay": 2.0}}},
    ]
    return {
        "name": "اقتباس",
        "canvas": _preset_size(preset, fps),
        "background": {"type": "gradient", "colors": colors, "angle": 160, "animated": True, "speed": 8, "vignette": 0.35},
        "scenes": [{"duration": 4.5, "transition": {"type": "fade", "duration": 0.4}, "layers": layers}],
        "audio": {"music": {"enabled": True, "style": "ambient", "volume": 0.3}},
        "meta": {"template": "quote_reel", "palette": palette},
    }


def compose_countdown(title: str, start: int = 5, palette: str = "neon", preset: str = "1080p", fps: int = 30, **_ignored: Any) -> dict[str, Any]:
    colors = PALETTES.get(palette, PALETTES["neon"])
    scenes = []
    for n in range(int(start), 0, -1):
        scenes.append({
            "duration": 1.0,
            "transition": {"type": "zoom", "duration": 0.3},
            "layers": [
                {"type": "counter", "value_from": n, "value_to": n, "y": 0.5, "font_size": 220, "font_weight": "bold", "fill": colors[2],
                 "animation": {"enter": {"type": "zoom_in", "duration": 0.5, "easing": "easeOutBack"}}},
            ],
        })
    scenes.append({
        "duration": 1.6, "transition": {"type": "fade", "duration": 0.4},
        "layers": [
            {"type": "text", "text": title, "y": 0.5, "font_size": 96, "font_weight": "bold", "fill": "#ffffff", "max_width": 0.8,
             "animation": {"enter": {"type": "pop", "duration": 0.6, "easing": "easeOutElastic"}}},
        ],
    })
    return {
        "name": title,
        "canvas": _preset_size(preset, fps),
        "background": {"type": "gradient", "colors": colors, "angle": 135, "animated": True, "speed": 30, "vignette": 0.3},
        "scenes": scenes,
        "audio": {"music": {"enabled": True, "style": "tech", "volume": 0.3}},
        "meta": {"template": "countdown", "palette": palette},
    }


def _preset_size(preset: str, fps: int) -> dict[str, int]:
    from .scene import PRESETS

    width, height = PRESETS.get((preset or "1080p").lower(), (1920, 1080))
    return {"width": width, "height": height, "fps": fps}


def build(kind: str, **kwargs: Any) -> dict[str, Any]:
    """Dispatch to the right composer."""
    kind = (kind or "intro_points_outro").lower()
    if kind == "promo":
        return compose_promo(**kwargs)
    if kind in {"quote_reel", "quote"}:
        return compose_quote_reel(**kwargs)
    if kind == "countdown":
        return compose_countdown(**kwargs)
    return compose_project(**kwargs)


def validate_project_dict(raw: dict[str, Any]) -> list[str]:
    from .scene import parse_project, validate

    return validate(parse_project(raw))


__all__ = [
    "template_names",
    "compose_project",
    "compose_promo",
    "compose_quote_reel",
    "compose_countdown",
    "build",
    "validate_project_dict",
]
