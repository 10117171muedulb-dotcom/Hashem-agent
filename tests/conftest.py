"""Shared pytest fixtures."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Keep tests hermetic and cross-platform: redirect the workspace into a temp tree.
os.environ["HASHEM_WORKSPACE"] = os.path.join(tempfile.gettempdir(), "hashem_test_ws")


@pytest.fixture()
def tmp_ws(tmp_path):
    return tmp_path


@pytest.fixture()
def small_project():
    from hashem.motion import parse_project

    return parse_project(
        {
            "canvas": {"width": 160, "height": 90, "fps": 10},
            "background": {"type": "gradient", "colors": ["#0f172a", "#7c3aed"], "animated": True},
            "scenes": [
                {
                    "duration": 1.0,
                    "transition": {"type": "fade", "duration": 0.3},
                    "layers": [
                        {"type": "text", "text": "مرحبا", "font_size": 24, "fill": "#ffffff",
                         "animation": {"enter": {"type": "fade_up", "duration": 0.4}}},
                    ],
                },
                {
                    "duration": 1.0,
                    "transition": {"type": "slide_left", "duration": 0.3},
                    "layers": [{"type": "shape", "shape": "circle", "width": 0.3, "fill": "#22d3ee"}],
                },
            ],
        }
    )
