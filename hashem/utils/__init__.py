"""Small, dependency-free helpers shared across the application."""

from __future__ import annotations

from .log import get_logger, setup_logging, tail_log
from .paths import app_data_dir, assets_dir, bundled_fonts_dir, ensure_dir, is_frozen, workspace_dir
from .safety import PathGuard, looks_destructive

__all__ = [
    "get_logger",
    "setup_logging",
    "tail_log",
    "app_data_dir",
    "assets_dir",
    "bundled_fonts_dir",
    "ensure_dir",
    "is_frozen",
    "workspace_dir",
    "PathGuard",
    "looks_destructive",
]
