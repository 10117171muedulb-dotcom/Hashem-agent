"""Logging setup that is safe inside a windowed PyInstaller bundle.

A windowed Windows executable has no console, so ``sys.stdout``/``sys.stderr`` can
be ``None``.  Every write in this module therefore tolerates that.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from .paths import log_file

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


class _NullStream:
    """Stand-in for a missing stdout/stderr so logging never raises."""

    def write(self, _msg: str) -> int:
        return 0

    def flush(self) -> None:  # pragma: no cover - trivial
        return None

    def isatty(self) -> bool:
        return False


def setup_logging(level: int = logging.INFO, file: Optional[Path] = None) -> Path:
    """Configure the root ``hashem`` logger. Returns the log file path."""
    global _configured
    target = Path(file) if file else log_file()
    target.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("hashem")
    logger.setLevel(level)
    logger.propagate = False

    if not _configured:
        logger.handlers.clear()

        file_handler = RotatingFileHandler(
            target, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(file_handler)

        if level <= logging.DEBUG:
            stream = logging.StreamHandler(sys.stderr or _NullStream())
            stream.setFormatter(logging.Formatter(_FORMAT))
            logger.addHandler(stream)

        _configured = True
    return target


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``hashem`` namespace."""
    if not name.startswith("hashem"):
        name = f"hashem.{name}"
    if not logging.getLogger("hashem").handlers:
        setup_logging()
    return logging.getLogger(name)


def tail_log(lines: int = 200) -> str:
    """Return the last ``lines`` lines of the log file (for the UI diagnostics view)."""
    path = log_file()
    if not path.exists():
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-lines:])
    except OSError:
        return ""


__all__ = ["setup_logging", "get_logger", "tail_log", "_NullStream"]
