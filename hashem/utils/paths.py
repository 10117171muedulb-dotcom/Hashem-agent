"""Cross-platform path resolution.

Works in three very different situations and has to get all of them right:

1. Running from a source checkout (``python -m hashem``).
2. Running as a PyInstaller *onefile* bundle, where ``sys._MEIPASS`` points at a
   temporary extraction directory that holds the read-only assets.
3. Running as a PyInstaller *onedir* bundle, where the assets sit next to the exe.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_APP_DIR_NAME = "HashemAgent"


def is_frozen() -> bool:
    """True when executing inside a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> Path:
    """Directory that holds the read-only assets shipped with the app."""
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:  # onefile
            return Path(meipass)
        return Path(sys.executable).resolve().parent  # onedir
    return Path(__file__).resolve().parent.parent


def app_data_dir() -> Path:
    """Writable per-user directory for settings, logs, sessions and exports."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        root = Path(base) / _APP_DIR_NAME
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support" / _APP_DIR_NAME
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
        root = Path(base) / _APP_DIR_NAME
    return ensure_dir(root)


def ensure_dir(path: Path | str) -> Path:
    """Create ``path`` (and parents) if missing and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def logs_dir() -> Path:
    return ensure_dir(app_data_dir() / "logs")


def cache_dir() -> Path:
    return ensure_dir(app_data_dir() / "cache")


def sessions_dir() -> Path:
    return ensure_dir(app_data_dir() / "sessions")


def exports_dir() -> Path:
    return ensure_dir(app_data_dir() / "exports")


def fonts_dir() -> Path:
    """Writable directory where the user can drop their own .ttf/.otf files."""
    return ensure_dir(app_data_dir() / "fonts")


def assets_dir() -> Path:
    return bundle_dir() / "assets"


def bundled_fonts_dir() -> Path:
    return assets_dir() / "fonts"


def workspace_dir() -> Path:
    """Default working directory the agent is allowed to touch.

    Resolution order: ``HASHEM_WORKSPACE`` env var, ``--workspace`` handled by the
    CLI, then a ``workspace`` folder inside the app data directory.
    """
    env = os.environ.get("HASHEM_WORKSPACE")
    if env:
        return ensure_dir(env)
    return ensure_dir(app_data_dir() / "workspace")


def log_file() -> Path:
    return logs_dir() / "hashem.log"


__all__ = [
    "is_frozen",
    "bundle_dir",
    "app_data_dir",
    "ensure_dir",
    "logs_dir",
    "cache_dir",
    "sessions_dir",
    "exports_dir",
    "fonts_dir",
    "assets_dir",
    "bundled_fonts_dir",
    "workspace_dir",
    "log_file",
]
