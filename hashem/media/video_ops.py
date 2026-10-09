"""Video operations — thin, friendly wrappers over the FFmpeg encoder layer."""

from __future__ import annotations

from pathlib import Path

from ..motion import encoder
from ..utils.log import get_logger

log = get_logger("media.video")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".srt", ".vtt", ".json", ".csv", ".html"}
BOOK_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}
DOCX_SUFFIXES = {".docx"}


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_SUFFIXES


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def is_text(path: str | Path) -> bool:
    return Path(path).suffix.lower() in TEXT_SUFFIXES


def probe(path: str | Path) -> dict:
    """Full metadata for any media file (video or image)."""
    path = Path(path)
    if is_video(path):
        return encoder.probe(path)
    if is_image(path):
        from PIL import Image

        with Image.open(path) as img:
            return {
                "path": str(path),
                "exists": True,
                "kind": "image",
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "size_bytes": path.stat().st_size,
            }
    return {"path": str(path), "exists": path.exists(), "kind": "unknown"}


def trim(path, output, start: float = 0.0, duration: float | None = None) -> Path:
    return encoder.trim(path, output, start, duration)


def concat(paths, output) -> Path:
    return encoder.concat_videos(paths, output)


def thumbnail(path, output, at: float = 0.0, width: int = 640) -> Path:
    return encoder.thumbnail_at(path, output, at, width)


def extract_frames(path, out_dir, every: float = 1.0) -> list[Path]:
    return encoder.extract_frames(path, out_dir, every)


def video_to_gif(path: str | Path, output: str | Path, fps: int = 12, width: int = 480, duration: float | None = None) -> Path:
    """Convert a video (or part of it) to an animated GIF."""
    output = Path(output)
    args = ["-i", str(path)]
    if duration:
        args += ["-t", str(duration)]
    args += ["-vf", f"fps={fps},scale={width}:-2:flags=lanczos", str(output)]
    result = encoder._run(args, timeout=900)
    if result.returncode != 0 or not output.exists():
        raise RuntimeError(f"gif conversion failed: {(result.stderr or b'').decode('utf-8', 'replace')[-400:]}")
    return output


def speed_change(path: str | Path, output: str | Path, factor: float = 2.0) -> Path:
    """Speed up (factor>1) or slow down (factor<1) a video."""
    factor = max(0.25, min(8.0, float(factor)))
    result = encoder._run(
        ["-i", str(path), "-filter:v", f"setpts={1 / factor:.4f}*PTS", "-an", str(output)],
        timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError("speed change failed")
    return Path(output)


__all__ = [
    "IMAGE_SUFFIXES", "VIDEO_SUFFIXES", "TEXT_SUFFIXES", "BOOK_SUFFIXES", "DOCX_SUFFIXES",
    "is_video", "is_image", "is_text", "probe", "trim", "concat", "thumbnail",
    "extract_frames", "video_to_gif", "speed_change",
]
