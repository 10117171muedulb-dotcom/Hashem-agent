"""Motion-graphics engine: JSON scene graph → rendered video."""

from __future__ import annotations

from .colors import PALETTES, palette, parse_color, to_hex
from .easing import EASINGS, easing_names
from .encoder import encode_frames, ffmpeg_available, ffmpeg_path, mux_audio, probe
from .layers import compute_state, paint_layer
from .render import TRANSITION_KINDS, Renderer, apply_transition
from .scene import PRESETS, Project, parse_project, to_dict, validate
from .text import has_arabic_support

__all__ = [
    "Renderer",
    "Project",
    "parse_project",
    "to_dict",
    "validate",
    "PRESETS",
    "PALETTES",
    "palette",
    "parse_color",
    "to_hex",
    "EASINGS",
    "easing_names",
    "TRANSITION_KINDS",
    "apply_transition",
    "compute_state",
    "paint_layer",
    "encode_frames",
    "mux_audio",
    "probe",
    "ffmpeg_path",
    "ffmpeg_available",
    "has_arabic_support",
]
