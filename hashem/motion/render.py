"""Frame compositor — turns a project + time into pixels.

Design notes that matter for both quality and speed:

* Backgrounds, vignettes and static layer tiles are cached.  A 10-second 1080p
  clip is 300 frames; without caching you would rasterise the same glyphs 300
  times.
* Everything is composited with PIL's C-level ``alpha_composite``, and only the
  effects that genuinely need it (vignette, grain, colour grade) go through
  numpy.
* A single broken layer or scene degrades to "skip it" — a long render never dies
  95% of the way through.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from ..utils.log import get_logger
from .colors import conic_gradient, linear_gradient, parse_color, radial_gradient
from .easing import ease
from .layers import LayerState, compute_state, is_static, paint_layer, place
from .scene import Background, Canvas, Layer, Project, Scene

log = get_logger("motion.render")

ProgressCallback = Callable[[int, int, float], None]


class Renderer:
    """Renders a :class:`Project` one frame at a time."""

    def __init__(
        self,
        project: Project,
        base_dir: str | None = None,
        progress: ProgressCallback | None = None,
    ):
        self.project = project
        self.base_dir = base_dir
        self.progress = progress
        self.canvas: Canvas = project.canvas
        self._bg_cache: dict[tuple, np.ndarray] = {}
        self._tile_cache: dict[tuple, Image.Image] = {}
        self._vignette: Optional[np.ndarray] = None
        self._frames_rendered = 0
        self._t0 = time.time()

    # ------------------------------------------------------------------ basics
    @property
    def duration(self) -> float:
        return self.project.duration

    @property
    def frame_count(self) -> int:
        return self.canvas.frame_count(self.duration)

    def time_at(self, index: int) -> float:
        return index / float(self.canvas.fps)

    # ------------------------------------------------------------- backgrounds
    def _vignette_mask(self) -> np.ndarray:
        if self._vignette is None:
            w, h = self.canvas.size
            ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
            xs = (xs / max(1, w - 1) - 0.5) * 2.0
            ys = (ys / max(1, h - 1) - 0.5) * 2.0
            distance = np.sqrt(xs * xs + ys * ys) / math.sqrt(2.0)
            self._vignette = np.clip(1.0 - np.power(np.clip(distance, 0, 1), 2.2), 0.0, 1.0)
        return self._vignette

    def render_background(self, background: Background, t: float) -> Image.Image:
        """Rasterise a background spec into an RGB image (cached when static)."""
        w, h = self.canvas.size
        angle = background.angle + (background.speed * t if background.animated else 0.0)
        key = (
            background.kind, tuple(background.colors), round(angle, 1), background.center,
            background.radius, background.image, background.overlay, w, h,
        )
        cached = self._bg_cache.get(key)
        if cached is None:
            if background.kind == "solid":
                array = np.zeros((h, w, 4), dtype=np.uint8)
                array[...] = parse_color(background.colors[0] if background.colors else "#0f172a")
            elif background.kind == "radial":
                array = radial_gradient(background.colors, w, h, background.center, background.radius)
            elif background.kind == "conic":
                array = conic_gradient(background.colors, w, h, background.center, background.angle)
            elif background.kind == "image" and background.image:
                array = self._background_from_image(background.image, w, h)
            else:
                array = linear_gradient(background.colors, w, h, angle)
            self._bg_cache[key] = array
            cached = array

        image = Image.fromarray(cached[..., :3], "RGB")

        if background.kind == "image" and background.overlay:
            overlay = Image.new("RGB", (w, h), parse_color(background.overlay)[:3])
            image = Image.blend(image, overlay, max(0.0, min(1.0, background.overlay_opacity)))

        if background.vignette > 0:
            array = np.asarray(image, dtype=np.float32)
            factor = 1.0 - (1.0 - self._vignette_mask()) * max(0.0, min(1.0, background.vignette))
            array *= factor[..., None]
            image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")

        if background.grain > 0:
            noise = np.random.default_rng(int(t * 1000) % 100000).normal(
                0.0, 18.0 * max(0.0, min(1.0, background.grain)), (h, w, 1)
            ).astype(np.float32)
            array = np.asarray(image, dtype=np.float32) + noise
            image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")

        return image

    def _background_from_image(self, path: str, w: int, h: int) -> np.ndarray:
        from pathlib import Path

        candidate = Path(path)
        if not candidate.is_absolute() and self.base_dir:
            candidate = Path(self.base_dir) / path
        try:
            source = Image.open(candidate).convert("RGB")
        except (OSError, ValueError) as exc:
            log.warning("Background image %s unavailable (%s); using gradient", candidate, exc)
            return linear_gradient(["#0f172a", "#1e293b"], w, h, 135.0)
        ratio = max(w / source.width, h / source.height)
        resized = source.resize((max(1, int(source.width * ratio)), max(1, int(source.height * ratio))), Image.LANCZOS)
        left = (resized.width - w) // 2
        top = (resized.height - h) // 2
        return np.asarray(resized.crop((left, top, left + w, top + h)), dtype=np.uint8)

    # ------------------------------------------------------------------- scenes
    def _scene_background(self, index: int) -> Background:
        scene = self.project.scenes[index]
        return scene.background or self.project.background

    def _tile_for(self, scene_index: int, layer: Layer, local_t: float, state: LayerState, scene: Scene):
        """Return the layer tile, using the cache when the layer is time-invariant."""
        if not is_static(layer):
            return paint_layer(layer, self.canvas, local_t, state, self.base_dir, scene.duration)
        key = (scene_index, layer.id, layer.kind, layer.text, round(layer.font_size, 1))
        tile = self._tile_cache.get(key)
        if tile is None:
            tile = paint_layer(layer, self.canvas, local_t, state, self.base_dir, scene.duration)
            if tile is not None:
                self._tile_cache[key] = tile
        return tile

    def render_scene_frame(self, index: int, local_t: float) -> Image.Image:
        """Composite one scene at scene-local time ``local_t``."""
        if not self.project.scenes:
            return Image.new("RGB", self.canvas.size, (0, 0, 0))
        index = max(0, min(index, len(self.project.scenes) - 1))
        scene = self.project.scenes[index]

        frame = self.render_background(self._scene_background(index), local_t).convert("RGBA")

        ordered = sorted(scene.layers, key=lambda l: l.z)
        for layer in ordered:
            state = compute_state(layer, local_t, scene.duration)
            if not state.visible():
                continue
            tile = self._tile_for(index, layer, local_t, state, scene)
            if tile is None:
                continue
            try:
                transformed, position = place(tile, layer, self.canvas, state)
                if state.opacity < 0.999:
                    alpha = transformed.split()[-1].point(lambda v: int(v * state.opacity))
                    transformed.putalpha(alpha)
                frame.alpha_composite(transformed, position)
            except Exception as exc:  # noqa: BLE001
                log.warning("Compositing layer %r failed (%s) — skipped", layer.id, exc)

        return frame.convert("RGB")

    # -------------------------------------------------------------- transitions
    def render_frame_at(self, t: float) -> Image.Image:
        """Render the absolute timeline position ``t`` including transitions."""
        index, local = self.project.scene_at(max(0.0, t))
        current = self.render_scene_frame(index, local)

        transition = self.project.scenes[index].transition
        if index == 0 or transition.type == "none" or transition.duration <= 0 or local >= transition.duration:
            return current

        previous = self.render_scene_frame(index - 1, self.project.scenes[index - 1].duration)
        p = ease(transition.easing, local / max(0.0001, transition.duration))
        return apply_transition(previous, current, p, transition.type)

    def render_frame_index(self, index: int) -> Image.Image:
        frame = self.render_frame_at(self.time_at(index))
        self._frames_rendered += 1
        if self.progress:
            elapsed = time.time() - self._t0
            fps = self._frames_rendered / elapsed if elapsed > 0 else 0.0
            try:
                self.progress(self._frames_rendered, self.frame_count, fps)
            except Exception:  # noqa: BLE001 - a broken callback must not stop the render
                log.debug("progress callback raised", exc_info=True)
        return frame

    def frames(self):
        """Yield every frame as an RGB image."""
        for i in range(self.frame_count):
            yield self.render_frame_index(i)

    def preview(self, t: float | None = None, scale: float = 0.35) -> Image.Image:
        """A fast, scaled-down single frame — used by the live UI preview."""
        moment = self.duration / 2.0 if t is None else max(0.0, min(t, max(0.0, self.duration - 1e-3)))
        frame = self.render_frame_at(moment)
        scale = max(0.05, min(1.0, scale))
        if scale >= 0.999:
            return frame
        size = (max(1, int(self.canvas.width * scale)), max(1, int(self.canvas.height * scale)))
        return frame.resize(size, Image.BILINEAR)

    def thumbnail_grid(self, columns: int = 4, width: int = 1280) -> Image.Image:
        """Contact sheet of evenly spaced frames — great for reviewing a storyboard."""
        count = max(1, min(self.frame_count, columns * 6))
        thumbs = []
        for i in range(count):
            t = (i + 0.5) * self.duration / count
            frame = self.render_frame_at(max(0.0, min(t, max(0.0, self.duration - 1e-3))))
            tw = max(64, width // columns)
            th = max(1, int(tw / self.canvas.aspect))
            thumbs.append(frame.resize((tw, th), Image.BILINEAR))
        rows = math.ceil(len(thumbs) / columns)
        tw, th = thumbs[0].size
        sheet = Image.new("RGB", (tw * columns, th * rows), (12, 12, 16))
        for i, thumb in enumerate(thumbs):
            sheet.paste(thumb, ((i % columns) * tw, (i // columns) * th))
        return sheet

    def clear_caches(self) -> None:
        self._tile_cache.clear()
        self._bg_cache.clear()


# --------------------------------------------------------------------------- #
# transitions
# --------------------------------------------------------------------------- #


def apply_transition(
    previous: Image.Image, current: Image.Image, progress: float, kind: str
) -> Image.Image:
    """Blend two full frames. ``progress`` runs 0 (all previous) → 1 (all current)."""
    p = max(0.0, min(1.0, float(progress)))
    if p <= 0.0:
        return previous
    if p >= 1.0:
        return current

    kind = (kind or "fade").lower()
    w, h = current.size

    if kind in {"fade", "crossfade", "dissolve"}:
        return Image.blend(previous, current, p)

    if kind == "fade_black":
        black = Image.new("RGB", (w, h), (0, 0, 0))
        if p < 0.5:
            return Image.blend(previous, black, p * 2.0)
        return Image.blend(black, current, (p - 0.5) * 2.0)

    if kind.startswith("slide"):
        direction = kind.split("_", 1)[1] if "_" in kind else "left"
        previous_arr = np.asarray(previous, dtype=np.uint8)
        current_arr = np.asarray(current, dtype=np.uint8)
        out = np.zeros_like(current_arr)
        if direction in {"left", "right"}:
            shift = int(w * p)
            if direction == "left":
                out[:, : w - shift] = previous_arr[:, shift:]
                out[:, w - shift :] = current_arr[:, :shift]
            else:
                out[:, shift:] = previous_arr[:, : w - shift]
                out[:, :shift] = current_arr[:, w - shift :]
        else:
            shift = int(h * p)
            if direction == "up":
                out[: h - shift, :] = previous_arr[shift:, :]
                out[h - shift :, :] = current_arr[:shift, :]
            else:
                out[shift:, :] = previous_arr[: h - shift, :]
                out[:shift, :] = current_arr[h - shift :, :]
        return Image.fromarray(out, "RGB")

    if kind in {"wipe", "wipe_left", "wipe_right", "wipe_up", "wipe_down"}:
        direction = kind.split("_", 1)[1] if "_" in kind else "right"
        previous_arr = np.asarray(previous, dtype=np.uint8).astype(np.float32)
        current_arr = np.asarray(current, dtype=np.uint8).astype(np.float32)
        if direction in {"left", "right"}:
            ramp = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
            if direction == "left":
                ramp = 1.0 - ramp
        else:
            ramp = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
            if direction == "up":
                ramp = 1.0 - ramp
        edge = max(0.02, 0.12)
        mask = np.clip((p * (1.0 + edge) - ramp) / edge, 0.0, 1.0)
        out = previous_arr * (1.0 - mask) + current_arr * mask
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")

    if kind in {"zoom", "zoom_in"}:
        scale = 1.0 + 0.25 * (1.0 - p)
        zoomed = previous.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        left = (zoomed.width - w) // 2
        top = (zoomed.height - h) // 2
        zoomed = zoomed.crop((left, top, left + w, top + h))
        return Image.blend(zoomed, current, p)

    if kind == "circle":
        previous_arr = np.asarray(previous, dtype=np.uint8).astype(np.float32)
        current_arr = np.asarray(current, dtype=np.uint8).astype(np.float32)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        distance = np.sqrt(((xs - w / 2) / (w / 2)) ** 2 + ((ys - h / 2) / (h / 2)) ** 2)
        mask = (distance <= (p * math.sqrt(2.0))).astype(np.float32)[..., None]
        out = previous_arr * (1.0 - mask) + current_arr * mask
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")

    if kind in {"blur", "blur_fade"}:
        radius = 18.0 * (1.0 - abs(p - 0.5) * 2.0)
        blurred_current = current.filter(ImageFilter.GaussianBlur(radius)) if radius > 0.2 else current
        return Image.blend(previous, blurred_current, p)

    return Image.blend(previous, current, p)


TRANSITION_KINDS = [
    "none", "fade", "fade_black", "slide_left", "slide_right", "slide_up", "slide_down",
    "wipe_right", "wipe_left", "wipe_up", "wipe_down", "zoom", "circle", "blur",
]


__all__ = ["Renderer", "apply_transition", "TRANSITION_KINDS"]
