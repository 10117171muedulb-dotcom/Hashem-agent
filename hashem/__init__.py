"""Hashem Agent — a free, offline-first AI agent with a motion-graphics studio.

The public surface of the package is intentionally small; everything else lives in
sub-packages (`agent`, `motion`, `llm`, `tools`, `ui`).
"""

from __future__ import annotations

__version__ = "1.0.0"
__app_name__ = "Hashem Agent"
__app_slug__ = "hashem-agent"
__author__ = "Hashem"

VERSION_TUPLE = (1, 0, 0)


def version_string() -> str:
    """Return a human readable version banner."""
    return f"{__app_name__} {__version__}"


__all__ = [
    "__version__",
    "__app_name__",
    "__app_slug__",
    "__author__",
    "VERSION_TUPLE",
    "version_string",
]
