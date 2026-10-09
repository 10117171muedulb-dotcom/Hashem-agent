"""Build the standalone Windows executable with PyInstaller.

Run on a Windows machine (or the CI windows-latest runner):

    python build/windows/build_exe.py

Produces ``dist/HashemAgent.exe`` — a single file that needs no Python install.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def _datas() -> list[tuple[str, str]]:
    datas: list[tuple[str, str]] = []

    static = ROOT / "hashem" / "ui" / "static"
    if static.is_dir():
        datas.append((str(static), "hashem/ui/static"))

    assets = ROOT / "hashem" / "assets"
    if assets.is_dir():
        datas.append((str(assets), "assets"))

    # Bundled static FFmpeg so the exe can encode video with no external install.
    try:
        import imageio_ffmpeg

        binaries = Path(imageio_ffmpeg.__file__).parent / "binaries"
        datas.append((str(binaries), "imageio_ffmpeg/binaries"))
    except Exception as exc:
        print("WARN imageio_ffmpeg binaries:", exc)

    # python-docx ships a default template it needs at runtime.
    try:
        import docx

        templates = Path(docx.__file__).parent / "templates"
        if templates.is_dir():
            datas.append((str(templates), "docx/templates"))
    except Exception as exc:
        print("WARN docx templates:", exc)

    return datas


def _hidden() -> list[str]:
    return [
        "edge_tts", "aiohttp", "aiosignal", "frozenlist", "async_timeout",
        "multidict", "yarl", "attr", "certifi",
        "ebooklib", "docx", "fpdf", "qrcode",
        "arabic_reshaper", "bidi", "bidi.algorithm", "bidi._bidi", "PIL", "numpy", "requests",
        "imageio_ffmpeg", "jinja2", "markdown", "rich", "fontTools",
    ]


def main() -> int:
    try:
        import PyInstaller.__main__
    except ImportError:
        print("PyInstaller is required: pip install pyinstaller", file=sys.stderr)
        return 1

    icon = ROOT / "hashem" / "assets" / "icon.ico"
    # Console subsystem: shows live logs + the studio URL, returns proper exit
    # codes (needed for CI smoke tests), and still auto-opens the browser.
    args = [
        str(ROOT / "hashem" / "__main__.py"),
        "--name=HashemAgent",
        "--onefile",
        "--noconfirm",
        "--clean",
        f"--distpath={ROOT / 'dist'}",
        f"--workpath={ROOT / 'build' / 'windows' / 'obj'}",
        f"--specpath={ROOT / 'build' / 'windows'}",
    ]
    if icon.exists():
        args.append(f"--icon={icon}")
    for source, dest in _datas():
        args.append(f"--add-data={source}{os.pathsep}{dest}")
    for module in _hidden():
        args.append(f"--hidden-import={module}")

    PyInstaller.__main__.run(args)

    exe = ROOT / "dist" / "HashemAgent.exe"
    if exe.exists():
        size_mb = exe.stat().st_size / 1e6
        print(f"\nOK  {exe}  ({size_mb:.1f} MB)")
        return 0
    print("\nFAIL: executable was not produced", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
