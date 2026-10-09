"""The motion-graphics project model.

A project is a plain JSON document.  That matters more than it sounds: the whole
point of the app is that an LLM writes this document, so the parser has to be
*extremely* forgiving.  Every field has a default, unknown keys are ignored, and
wrong types are coerced rather than raising.  :func:`validate` then reports
problems as human-readable warnings instead of exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .colors import gradient_colors, parse_color
from .easing import EASINGS

# --------------------------------------------------------------------------- #
# coercion helpers
# --------------------------------------------------------------------------- #


def _num(value: Any, default: float, lo: float | None = None, hi: float | None = None) -> float:
    """Best-effort float coercion with optional clamping."""
    try:
        if isinstance(value, bool):
            out = float(default)
        elif value is None or value == "":
            out = float(default)
        else:
            out = float(value)
    except (TypeError, ValueError):
        out = float(default)
    if out != out:  # NaN
        out = float(default)
    if lo is not None:
        out = max(float(lo), out)
    if hi is not None:
        out = min(float(hi), out)
    return out


def _int(value: Any, default: int, lo: int | None = None, hi: int | None = None) -> int:
    return int(round(_num(value, default, lo if lo is None else float(lo), hi if hi is None else float(hi))))


def _str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "نعم"}
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str) and value.strip():
        return [value]
    return []


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


@dataclass
class Canvas:
    width: int = 1920
    height: int = 1080
    fps: int = 30

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 1.0

    def frame_count(self, duration: float) -> int:
        return max(1, int(round(duration * self.fps)))


PRESETS = {
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "4k": (3840, 2160),
    "vertical": (1080, 1920),
    "reels": (1080, 1920),
    "square": (1080, 1080),
    "twitter": (1600, 900),
    "story": (1080, 1920),
}


@dataclass
class Background:
    kind: str = "gradient"          # solid | gradient | radial | conic | image
    colors: list[str] = field(default_factory=lambda: ["#0f172a", "#1e293b"])
    angle: float = 135.0
    animated: bool = False
    speed: float = 12.0             # degrees per second when animated
    center: tuple[float, float] = (0.5, 0.5)
    radius: float = 0.8
    image: str = ""
    overlay: str = ""               # colour multiplied over an image background
    overlay_opacity: float = 0.35
    vignette: float = 0.0           # 0..1 darkening towards the edges
    grain: float = 0.0              # 0..1 film grain amount


@dataclass
class AnimSpec:
    type: str = "fade"
    duration: float = 0.6
    delay: float = 0.0
    easing: str = "easeOutCubic"
    distance: float = 60.0          # px for slide-type animations
    from_scale: float = 0.85
    from_rotation: float = -12.0
    # loop-specific
    amplitude: float = 0.05
    speed: float = 1.0


@dataclass
class LayerAnim:
    enter: AnimSpec = field(default_factory=lambda: AnimSpec(type="fade_up"))
    exit: AnimSpec = field(default_factory=lambda: AnimSpec(type="fade", duration=0.4))
    loop: AnimSpec | None = None


@dataclass
class Layer:
    kind: str = "text"

    # geometry — normalised by default (0..1 of the canvas)
    x: float = 0.5
    y: float = 0.5
    unit: str = "norm"              # norm | px
    anchor: str = "center"          # center | top | bottom | left | right | top-left ...
    width: float | None = None
    height: float | None = None
    rotation: float = 0.0
    opacity: float = 1.0
    z: int = 0

    # text
    text: str = ""
    font_family: str = ""
    font_size: float = 72.0
    font_weight: str = "bold"
    align: str = "center"
    line_spacing: float = 1.25
    letter_spacing: float = 0.0
    max_width: float = 0.8          # normalised unless unit == px
    fill: str = "#ffffff"
    stroke_color: str = ""
    stroke_width: float = 0.0
    shadow: str = ""
    shadow_blur: float = 0.0
    shadow_offset: tuple[float, float] = (0.0, 6.0)

    # shapes
    shape: str = "rect"             # rect | rounded | circle | ellipse | ring | triangle | star | line | arrow | polygon | pill
    radius: float = 24.0
    points: int = 5
    thickness: float = 8.0
    gradient: list[str] = field(default_factory=list)
    gradient_angle: float = 90.0

    # images
    image: str = ""
    fit: str = "contain"            # contain | cover | stretch

    # particles
    count: int = 40
    particle_shape: str = "circle"  # circle | square | star | snow | sparkle
    particle_color: str = "#ffffff"
    speed_factor: float = 1.0
    seed: int = 1234

    # numeric / progress
    value_from: float = 0.0
    value_to: float = 100.0
    suffix: str = ""
    prefix: str = ""

    # padding / decoration
    padding: float = 28.0
    border_color: str = ""
    border_width: float = 0.0

    anim: LayerAnim = field(default_factory=LayerAnim)

    # resolved at normalisation time
    id: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Transition:
    type: str = "fade"              # fade | none | slide_left | slide_right | slide_up | slide_down | wipe | zoom | fade_black
    duration: float = 0.5
    easing: str = "easeInOutCubic"


@dataclass
class Scene:
    duration: float = 3.0
    layers: list[Layer] = field(default_factory=list)
    background: Background | None = None
    transition: Transition = field(default_factory=Transition)
    name: str = ""


@dataclass
class MusicSpec:
    enabled: bool = True
    style: str = "corporate"
    volume: float = 0.28
    fade_out: float = 1.2


@dataclass
class NarrationSpec:
    enabled: bool = False
    text: str = ""
    voice: str = ""
    rate: int = 0                   # -10..10 (SAPI)
    volume: float = 0.95


@dataclass
class AudioSpec:
    music: MusicSpec = field(default_factory=MusicSpec)
    narration: NarrationSpec = field(default_factory=NarrationSpec)
    music_file: str = ""


@dataclass
class Project:
    name: str = "Untitled"
    author: str = "Hashem Agent"
    canvas: Canvas = field(default_factory=Canvas)
    background: Background = field(default_factory=Background)
    scenes: list[Scene] = field(default_factory=list)
    audio: AudioSpec = field(default_factory=AudioSpec)
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ derived
    @property
    def duration(self) -> float:
        return sum(max(0.0, s.duration) for s in self.scenes)

    @property
    def frame_count(self) -> int:
        return self.canvas.frame_count(self.duration)

    def scene_at(self, t: float) -> tuple[int, float]:
        """Return ``(scene_index, local_time)`` for absolute time ``t``."""
        if not self.scenes:
            return 0, 0.0
        cursor = 0.0
        for index, scene in enumerate(self.scenes):
            if t < cursor + scene.duration or index == len(self.scenes) - 1:
                return index, max(0.0, t - cursor)
            cursor += scene.duration
        return len(self.scenes) - 1, 0.0

    def scene_bounds(self, index: int) -> tuple[float, float]:
        start = sum(s.duration for s in self.scenes[:index])
        return start, start + self.scenes[index].duration


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #

_ANCHORS = {
    "center": (0.5, 0.5),
    "top": (0.5, 0.0),
    "bottom": (0.5, 1.0),
    "left": (0.0, 0.5),
    "right": (1.0, 0.5),
    "top-left": (0.0, 0.0),
    "top-right": (1.0, 0.0),
    "bottom-left": (0.0, 1.0),
    "bottom-right": (1.0, 1.0),
}


def anchor_offset(anchor: str) -> tuple[float, float]:
    """Fraction of the layer box that sits left/above the anchor point."""
    return _ANCHORS.get((anchor or "center").lower().replace("_", "-").replace(" ", "-"), (0.5, 0.5))


def _parse_anim(raw: Any, default_type: str) -> AnimSpec:
    if isinstance(raw, str):
        return AnimSpec(type=raw or default_type)
    data = _dict(raw)
    spec = AnimSpec(
        type=_str(data.get("type"), default_type).lower(),
        duration=_num(data.get("duration"), 0.6, 0.0, 30.0),
        delay=_num(data.get("delay"), 0.0, 0.0, 120.0),
        easing=_str(data.get("easing"), "easeOutCubic"),
        distance=_num(data.get("distance"), 60.0, -4000.0, 4000.0),
        from_scale=_num(data.get("from_scale", data.get("scale")), 0.85, 0.0, 8.0),
        from_rotation=_num(data.get("from_rotation", data.get("rotation")), -12.0, -720.0, 720.0),
        amplitude=_num(data.get("amplitude"), 0.05, 0.0, 4.0),
        speed=_num(data.get("speed"), 1.0, 0.0, 20.0),
    )
    if spec.easing not in EASINGS:
        spec.easing = "easeOutCubic"
    return spec


def _parse_layer_anim(raw: Any) -> LayerAnim:
    data = _dict(raw)
    loop_raw = data.get("loop")
    return LayerAnim(
        enter=_parse_anim(data.get("enter"), "fade_up"),
        exit=_parse_anim(data.get("exit"), "fade"),
        loop=_parse_anim(loop_raw, "pulse") if loop_raw else None,
    )


def _parse_layer(raw: Any, index: int) -> Layer:
    data = _dict(raw)
    font = _dict(data.get("font"))
    stroke = _dict(data.get("stroke"))
    shadow = _dict(data.get("shadow"))
    size = data.get("size")

    kind = _str(data.get("type", data.get("kind")), "text").lower()
    if kind in {"title", "subtitle", "heading", "caption", "label"}:
        kind = "text"

    shadow_offset = shadow.get("offset") or data.get("shadow_offset") or [0, 6]
    offset_list = _list(shadow_offset) or [0, 6]
    offset = (
        _num(offset_list[0], 0.0, -4000, 4000),
        _num(offset_list[1] if len(offset_list) > 1 else 6.0, 6.0, -4000, 4000),
    )

    layer = Layer(
        kind=kind,
        x=_num(data.get("x"), 0.5),
        y=_num(data.get("y"), 0.5),
        unit=_str(data.get("unit"), "norm").lower(),
        anchor=_str(data.get("anchor"), "center"),
        width=_num(data["width"], 0.0) if data.get("width") is not None else None,
        height=_num(data["height"], 0.0) if data.get("height") is not None else None,
        rotation=_num(data.get("rotation"), 0.0, -720.0, 720.0),
        opacity=_num(data.get("opacity"), 1.0, 0.0, 1.0),
        z=_int(data.get("z"), 0),
        text=_str(data.get("text"), ""),
        font_family=_str(data.get("font_family", font.get("family", data.get("family"))), ""),
        font_size=_num(
            font.get("size", size if not isinstance(size, (dict, list)) else None) or data.get("font_size"),
            72.0, 4.0, 600.0,
        ),
        font_weight=_str(font.get("weight", data.get("font_weight")), "bold"),
        align=_str(data.get("align", data.get("text_align")), "center").lower(),
        line_spacing=_num(data.get("line_spacing"), 1.25, 0.5, 4.0),
        letter_spacing=_num(data.get("letter_spacing"), 0.0, -50.0, 200.0),
        max_width=_num(data.get("max_width"), 0.8, 0.0, 100000.0),
        fill=_str(data.get("fill", data.get("color")), "#ffffff"),
        stroke_color=_str(stroke.get("color", data.get("stroke_color")), ""),
        stroke_width=_num(stroke.get("width", data.get("stroke_width")), 0.0, 0.0, 100.0),
        shadow=_str(shadow.get("color", data.get("shadow_color")), ""),
        shadow_blur=_num(shadow.get("blur", data.get("shadow_blur")), 0.0, 0.0, 200.0),
        shadow_offset=offset,
        shape=_str(data.get("shape", kind if kind != "text" else "rect"), "rect").lower(),
        radius=_num(data.get("radius"), 24.0, 0.0, 4000.0),
        points=_int(data.get("points"), 5, 3, 24),
        thickness=_num(data.get("thickness", data.get("stroke_px")), 8.0, 0.0, 4000.0),
        gradient=gradient_colors(data.get("gradient")) if data.get("gradient") else [],
        gradient_angle=_num(data.get("gradient_angle"), 90.0, -720.0, 720.0),
        image=_str(data.get("image", data.get("src", data.get("path"))), ""),
        fit=_str(data.get("fit"), "contain").lower(),
        count=_int(data.get("count"), 40, 1, 2000),
        particle_shape=_str(data.get("particle_shape", data.get("shape_type")), "circle").lower(),
        particle_color=_str(data.get("particle_color", data.get("color")), "#ffffff"),
        speed_factor=_num(data.get("speed_factor", data.get("speed")), 1.0, 0.0, 20.0),
        seed=_int(data.get("seed"), 1234),
        value_from=_num(data.get("value_from", data.get("from")), 0.0),
        value_to=_num(data.get("value_to", data.get("to")), 100.0),
        suffix=_str(data.get("suffix"), ""),
        prefix=_str(data.get("prefix"), ""),
        padding=_num(data.get("padding"), 28.0, 0.0, 1000.0),
        border_color=_str(data.get("border_color"), ""),
        border_width=_num(data.get("border_width"), 0.0, 0.0, 200.0),
        anim=_parse_layer_anim(data.get("animation", data.get("anim"))),
        id=_str(data.get("id"), f"layer-{index}"),
    )

    # A shape layer that forgot to say `shape` should not silently become a rect.
    if layer.kind not in {"text", "image", "particles", "progress", "counter", "shape", "badge", "line"}:
        layer.kind = "shape"

    known = set(Layer.__dataclass_fields__)
    layer.extras = {k: v for k, v in data.items() if k not in known and not isinstance(v, (dict, list))}
    return layer


def _parse_background(raw: Any, fallback: Background | None = None) -> Background:
    data = _dict(raw)
    if not data:
        return fallback or Background()
    kind = _str(data.get("type", data.get("kind")), "gradient").lower()
    if kind in {"linear", "grad"}:
        kind = "gradient"
    colors = data.get("colors") or data.get("color") or data.get("palette")
    center_raw = _list(data.get("center")) or [0.5, 0.5]
    return Background(
        kind=kind,
        colors=gradient_colors(colors, "midnight"),
        angle=_num(data.get("angle"), 135.0, -720.0, 720.0),
        animated=_bool(data.get("animated"), False),
        speed=_num(data.get("speed"), 12.0, -360.0, 360.0),
        center=(_num(center_raw[0], 0.5), _num(center_raw[1] if len(center_raw) > 1 else 0.5, 0.5)),
        radius=_num(data.get("radius"), 0.8, 0.05, 8.0),
        image=_str(data.get("image"), ""),
        overlay=_str(data.get("overlay"), ""),
        overlay_opacity=_num(data.get("overlay_opacity"), 0.35, 0.0, 1.0),
        vignette=_num(data.get("vignette"), 0.0, 0.0, 1.0),
        grain=_num(data.get("grain"), 0.0, 0.0, 1.0),
    )


def _parse_transition(raw: Any) -> Transition:
    if isinstance(raw, str):
        raw = {"type": raw}
    data = _dict(raw)
    kind = _str(data.get("type"), "fade").lower().replace(" ", "_")
    if kind in {"cut", "none", ""}:
        kind = "none"
    easing = _str(data.get("easing"), "easeInOutCubic")
    if easing not in EASINGS:
        easing = "easeInOutCubic"
    return Transition(
        type=kind,
        duration=_num(data.get("duration"), 0.5, 0.0, 10.0),
        easing=easing,
    )


def _parse_scene(raw: Any, index: int) -> Scene:
    data = _dict(raw)
    layers_raw = data.get("layers") or data.get("elements") or []
    if not isinstance(layers_raw, list):
        layers_raw = [layers_raw]
    return Scene(
        duration=_num(data.get("duration", data.get("length")), 3.0, 0.1, 600.0),
        layers=[_parse_layer(item, i) for i, item in enumerate(layers_raw)],
        background=_parse_background(data.get("background")) if data.get("background") else None,
        transition=_parse_transition(data.get("transition", {})),
        name=_str(data.get("name"), f"Scene {index + 1}"),
    )


def _parse_canvas(raw: Any, preset: str = "") -> Canvas:
    data = _dict(raw)
    width = _num(data.get("width"), 0.0, 0.0, 8192.0)
    height = _num(data.get("height"), 0.0, 0.0, 8192.0)
    if (not width or not height) and preset.lower() in PRESETS:
        width, height = PRESETS[preset.lower()]
    return Canvas(
        width=int(width) if width else 1920,
        height=int(height) if height else 1080,
        fps=_int(data.get("fps"), 30, 1, 120),
    )


def _parse_audio(raw: Any) -> AudioSpec:
    data = _dict(raw)
    music = _dict(data.get("music"))
    narration = _dict(data.get("narration", data.get("tts", data.get("voiceover"))))
    return AudioSpec(
        music=MusicSpec(
            enabled=_bool(music.get("enabled"), True),
            style=_str(music.get("style"), "corporate"),
            volume=_num(music.get("volume"), 0.28, 0.0, 1.0),
            fade_out=_num(music.get("fade_out"), 1.2, 0.0, 30.0),
        ),
        narration=NarrationSpec(
            enabled=_bool(narration.get("enabled"), False),
            text=_str(narration.get("text"), ""),
            voice=_str(narration.get("voice"), ""),
            rate=_int(narration.get("rate"), 0, -10, 10),
            volume=_num(narration.get("volume"), 0.95, 0.0, 1.0),
        ),
        music_file=_str(data.get("music_file", data.get("file")), ""),
    )


def parse_project(raw: dict[str, Any] | str) -> Project:
    """Build a :class:`Project` from a (possibly sloppy) dict or JSON string."""
    if isinstance(raw, str):
        import json

        raw = json.loads(raw)
    data = _dict(raw)
    meta = _dict(data.get("meta"))
    canvas = _parse_canvas(data.get("canvas"), _str(data.get("preset"), ""))
    scenes_raw = data.get("scenes") or data.get("slides") or []
    if not isinstance(scenes_raw, list):
        scenes_raw = [scenes_raw]

    return Project(
        name=_str(data.get("name", meta.get("name")), "Untitled"),
        author=_str(data.get("author", meta.get("author")), "Hashem Agent"),
        canvas=canvas,
        background=_parse_background(data.get("background")),
        scenes=[_parse_scene(item, i) for i, item in enumerate(scenes_raw)],
        audio=_parse_audio(data.get("audio")),
        meta=meta,
    )


def to_dict(project: Project) -> dict[str, Any]:
    """Serialise a project back to JSON-ready primitives (round-trips)."""
    from dataclasses import asdict

    payload = asdict(project)
    payload.pop("meta", None)
    return {
        "name": project.name,
        "author": project.author,
        "canvas": {"width": project.canvas.width, "height": project.canvas.height, "fps": project.canvas.fps},
        "background": asdict(project.background),
        "scenes": [
            {
                "name": scene.name,
                "duration": scene.duration,
                "transition": asdict(scene.transition),
                "background": asdict(scene.background) if scene.background else None,
                "layers": [asdict(layer) for layer in scene.layers],
            }
            for scene in project.scenes
        ],
        "audio": asdict(project.audio),
        "meta": project.meta,
    }


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


def validate(project: Project) -> list[str]:
    """Return human-readable warnings. Empty list means the project looks sane."""
    warnings: list[str] = []
    if not project.scenes:
        warnings.append("المشروع لا يحتوي على أي مشهد (scenes فارغة).")
    if project.canvas.width % 2 or project.canvas.height % 2:
        warnings.append("أبعاد h264 يجب أن تكون زوجية — سيتم تقريبها تلقائيًا عند التصدير.")
    if project.canvas.fps < 12:
        warnings.append(f"معدل الإطارات {project.canvas.fps} منخفض؛ 24–30 أفضل للحركة السلسة.")

    for index, scene in enumerate(project.scenes, start=1):
        if scene.duration <= 0.2:
            warnings.append(f"المشهد {index} قصير جدًا ({scene.duration:.2f} ثانية).")
        if not scene.layers:
            warnings.append(f"المشهد {index} فارغ — لا يحتوي على أي عنصر.")
        for layer in scene.layers:
            if layer.kind == "text" and not layer.text.strip():
                warnings.append(f"المشهد {index}: عنصر نصي بدون نص.")
            if layer.kind == "image" and not layer.image.strip():
                warnings.append(f"المشهد {index}: عنصر صورة بدون مسار.")
            if parse_color(layer.fill, default=None) is None:  # type: ignore[arg-type]
                warnings.append(f"المشهد {index}: لون غير مفهوم {layer.fill!r}.")
    return warnings


__all__ = [
    "Canvas",
    "Background",
    "AnimSpec",
    "LayerAnim",
    "Layer",
    "Transition",
    "Scene",
    "AudioSpec",
    "MusicSpec",
    "NarrationSpec",
    "Project",
    "PRESETS",
    "parse_project",
    "to_dict",
    "validate",
    "anchor_offset",
]
