"""Self-diagnostics. Run by ``hashem doctor`` and by the Windows .exe smoke test."""

from __future__ import annotations

import platform
import sys
from typing import Any


def _check(name: str, fn) -> dict[str, Any]:
    try:
        value = fn()
        return {"name": name, "ok": bool(value), "detail": str(value)}
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "ok": False, "detail": f"{type(exc).__name__}: {exc}"}


def run_diagnostics() -> dict[str, Any]:
    from . import __version__
    from .motion import encoder, ffmpeg_available
    from .motion.fonts import font_report
    from .motion.text import has_arabic_support
    from .audio import has_edge_tts
    from .utils.paths import app_data_dir, workspace_dir

    checks = [
        {"name": "python", "ok": True, "detail": platform.python_version()},
        {"name": "platform", "ok": True, "detail": f"{platform.system()} {platform.release()} {platform.machine()}"},
        _check("pillow", lambda: __import__("PIL").__version__),
        _check("numpy", lambda: __import__("numpy").__version__),
        _check("requests", lambda: __import__("requests").__version__),
        _check("ffmpeg", lambda: encoder.ffmpeg_path() or ""),
        _check("arabic_shaping", lambda: has_arabic_support()),
        _check("fpdf2", lambda: __import__("fpdf").__version__ if hasattr(__import__("fpdf"), "__version__") else True),
        _check("ebooklib", lambda: bool(__import__("ebooklib"))),
        _check("python_docx", lambda: bool(__import__("docx"))),
        _check("edge_tts", lambda: has_edge_tts()),
        _check("qrcode", lambda: bool(__import__("qrcode"))),
        {"name": "version", "ok": True, "detail": __version__},
    ]

    fonts = font_report()
    report = {
        "app": "Hashem Agent",
        "version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "checks": checks,
        "fonts": {k: fonts[k] for k in ("total_fonts", "arabic_available", "latin_available")},
        "ffmpeg_available": ffmpeg_available(),
        "app_data": str(app_data_dir()),
        "workspace": str(workspace_dir()),
        "all_ok": all(c["ok"] for c in checks),
    }
    return report


def format_report(report: dict[str, Any]) -> str:
    lines = [f"{report['app']} {report['version']}  —  {'OK' if report['all_ok'] else 'ISSUES FOUND'}", ""]
    for check in report["checks"]:
        mark = "[+]" if check["ok"] else "[x]"
        lines.append(f"  {mark} {check['name']:<16} {check['detail']}")
    lines.append("")
    lines.append(f"  Arabic fonts: {report['fonts']['arabic_available']}")
    lines.append(f"  FFmpeg: {'available' if report['ffmpeg_available'] else 'MISSING'}")
    lines.append(f"  Workspace: {report['workspace']}")
    return "\n".join(lines)


__all__ = ["run_diagnostics", "format_report"]
