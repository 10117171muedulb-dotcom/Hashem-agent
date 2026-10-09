"""Tests for the motion-graphics engine — easing, colours, shaping, render, encode."""

from __future__ import annotations

import numpy as np
import pytest

from hashem.motion import easing, colors
from hashem.motion import text as mtext
from hashem.motion import parse_project, validate, Renderer, encode_frames
from hashem.motion.encoder import probe


# ------------------------------------------------------------------ easing
def test_easing_endpoints_and_overshoot():
    assert easing.ease("linear", 0.0) == 0.0
    assert easing.ease("linear", 1.0) == 1.0
    assert easing.ease("easeOutCubic", 1.0) == 1.0
    # back/elastic intentionally overshoot mid-flight but land on 1
    assert easing.ease("easeOutBack", 1.0) == pytest.approx(1.0, abs=1e-6)
    assert easing.ease("easeOutBack", 0.7) > 1.0
    assert easing.ease("nope", 0.5) == pytest.approx(easing.ease("easeOutCubic", 0.5))


def test_easing_nan_and_inf_safe():
    for name in easing.easing_names():
        for t in (0.0, 0.37, 1.0, float("nan"), -2.0, 3.0):
            value = easing.ease(name, t)
            assert value == value and abs(value) < 1e6


def test_cubic_bezier_identity():
    fn = easing.cubic_bezier(0, 0, 1, 1)
    assert fn(0.5) == pytest.approx(0.5, abs=1e-3)
    assert fn(1.0) == pytest.approx(1.0, abs=1e-3)


# ------------------------------------------------------------------ colors
def test_color_parsing():
    assert colors.parse_color("#ff0000") == (255, 0, 0, 255)
    assert colors.parse_color("rgba(10,20,30,.5)")[3] == 128
    assert colors.parse_color("garbage", default=(1, 2, 3, 4)) == (1, 2, 3, 4)
    assert colors.contrast_ratio("#ffffff", "#000000") == pytest.approx(21.0, abs=0.1)


def test_gradients_shapes():
    assert colors.linear_gradient(["#000", "#fff"], 64, 32).shape == (32, 64, 4)
    assert colors.radial_gradient(["#000", "#fff"], 32, 32).shape == (32, 32, 4)
    assert colors.conic_gradient(["#000", "#fff"], 32, 32).shape == (32, 32, 4)


# ------------------------------------------------------------------ shaping
def test_arabic_shaping_produces_presentation_forms():
    shaped = mtext.shape_text("مرحبا")
    # first logical letter م must become an initial presentation form (FEE3)
    assert "\ufee3" in shaped
    assert mtext.has_arabic_support()


def test_wrap_and_fit():
    path = mtext.resolve_font_for if hasattr(mtext, "resolve_font_for") else None
    from hashem.motion.fonts import resolve_font

    font = resolve_font("نص عربي طويل جدا للالتفاف")
    lines = mtext.wrap_text("نص عربي طويل جدا للالتفاف على اكثر من سطر", font, 24, 120)
    assert len(lines) >= 2


# ------------------------------------------------------------------ project
def test_parse_and_validate_forgiving():
    raw = {"canvas": {"width": 300, "height": 200, "fps": 12},
           "scenes": [{"duration": 1, "layers": [{"type": "text", "text": "", "fill": "notacolor"}]}]}
    project = parse_project(raw)
    assert project.canvas.width == 300
    warnings = validate(project)
    assert any("بدون نص" in w for w in warnings)


def test_scene_at_and_bounds(small_project):
    index, local = small_project.scene_at(0.5)
    assert index == 0
    index2, _ = small_project.scene_at(1.5)
    assert index2 == 1


# ------------------------------------------------------------------ render
def test_render_frame_non_trivial(small_project):
    renderer = Renderer(small_project)
    frame = renderer.render_frame_at(0.5)
    array = np.asarray(frame)
    assert array.shape[:2] == (90, 160)
    assert array.std() > 1.0  # not a flat frame


def test_transition_blend():
    from hashem.motion.render import apply_transition
    from PIL import Image

    a = Image.new("RGB", (32, 32), (255, 0, 0))
    b = Image.new("RGB", (32, 32), (0, 0, 255))
    for kind in ("fade", "slide_left", "wipe_right", "zoom", "circle"):
        out = apply_transition(a, b, 0.5, kind)
        assert out.size == (32, 32)


# ------------------------------------------------------------------ encode
def test_encode_and_probe(small_project, tmp_path):
    renderer = Renderer(small_project)
    frames = list(renderer.frames())
    out = encode_frames(frames, tmp_path / "out.mp4", (160, 90), fps=10, crf=28)
    assert out.exists() and out.stat().st_size > 500
    info = probe(out)
    assert info["has_video"]
    assert info["width"] == 160 and info["height"] == 90
