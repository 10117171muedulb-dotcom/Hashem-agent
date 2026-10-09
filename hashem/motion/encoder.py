"""Video encoding and media operations, built on a bundled FFmpeg.

``imageio-ffmpeg`` ships a static FFmpeg binary for every platform, which means
the packaged .exe can encode H.264/WebM/GIF without asking the user to install
anything.  Every function here degrades gracefully if the binary is missing.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
from PIL import Image

from ..utils.log import get_logger

log = get_logger("motion.encoder")

ProgressCallback = Callable[[int, int], None]

_ffmpeg_override: str | None = None


def set_ffmpeg(path: str | None) -> None:
    """Force a specific ffmpeg binary (used by tests and by ``--ffmpeg``)."""
    global _ffmpeg_override
    _ffmpeg_override = path


def ffmpeg_path() -> str | None:
    """Locate an ffmpeg binary: override → PATH → bundled imageio-ffmpeg copy."""
    if _ffmpeg_override:
        return _ffmpeg_override
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        log.warning("No ffmpeg available (%s)", exc)
        return None


def ffmpeg_available() -> bool:
    return ffmpeg_path() is not None


def _run(args: Sequence[str], timeout: int = 600, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    binary = ffmpeg_path()
    if not binary:
        raise RuntimeError("ffmpeg is not available on this system")
    command = [binary, "-hide_banner", "-nostdin", "-y", *args]
    log.debug("ffmpeg %s", " ".join(command[:12]))
    return subprocess.run(
        command,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


# --------------------------------------------------------------------------- #
# probing
# --------------------------------------------------------------------------- #

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_STREAM_RE = re.compile(r"Stream.*?:\s*(\w+):", re.IGNORECASE)
_RESOLUTION_RE = re.compile(r"(\d{2,5})x(\d{2,5})")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s+fps")


def probe(path: str | Path) -> dict:
    """Read duration/streams/resolution by parsing ``ffmpeg -i`` output."""
    result = _run(["-i", str(path)], timeout=60)
    text = (result.stderr or b"").decode("utf-8", errors="replace")

    info: dict = {"path": str(path), "exists": Path(path).exists()}
    match = _DURATION_RE.search(text)
    if match:
        hours, minutes, seconds = match.groups()
        info["duration"] = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    info["streams"] = [s.lower() for s in _STREAM_RE.findall(text)]
    info["has_video"] = "video" in info["streams"]
    info["has_audio"] = "audio" in info["streams"]
    res = _RESOLUTION_RE.search(text)
    if res:
        info["width"], info["height"] = int(res.group(1)), int(res.group(2))
    fps = _FPS_RE.search(text)
    if fps:
        info["fps"] = float(fps.group(1))
    info["size_bytes"] = Path(path).stat().st_size if info["exists"] else 0
    return info


def read_frames(path: str | Path, max_frames: int | None = None) -> Iterator[np.ndarray]:
    """Decode a video file into RGB numpy frames."""
    binary = ffmpeg_path()
    if not binary:
        raise RuntimeError("ffmpeg is not available on this system")
    try:
        import imageio_ffmpeg

        reader = imageio_ffmpeg.read_frames(str(path))
        _meta = next(reader)  # metadata dict
        count = 0
        for frame in reader:
            yield frame
            count += 1
            if max_frames and count >= max_frames:
                break
        reader.close()
    except Exception as exc:  # noqa: BLE001
        log.warning("read_frames(%s) failed: %s — falling back to pipe decode", path, exc)
        yield from _read_frames_pipe(path, max_frames)


def _read_frames_pipe(path: str | Path, max_frames: int | None) -> Iterator[np.ndarray]:
    """Fallback decoder used when imageio-ffmpeg's helper is unavailable."""
    info = probe(path)
    width, height = int(info.get("width", 0)), int(info.get("height", 0))
    if not width or not height:
        return
    binary = ffmpeg_path()
    command = [
        binary, "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-v", "error", "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_size = width * height * 3
    count = 0
    try:
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(frame_size)
            if len(chunk) < frame_size:
                break
            yield np.frombuffer(chunk, dtype=np.uint8).reshape((height, width, 3))
            count += 1
            if max_frames and count >= max_frames:
                break
    finally:
        process.stdout.close()
        process.wait(timeout=30)


# --------------------------------------------------------------------------- #
# encoding
# --------------------------------------------------------------------------- #


def _even(value: int) -> int:
    value = int(value)
    return value if value % 2 == 0 else max(2, value - 1)


class VideoWriter:
    """Pipe frames straight into ffmpeg. Use as a context manager."""

    def __init__(
        self,
        output: str | Path,
        size: tuple[int, int],
        fps: int = 30,
        crf: int = 20,
        codec: str = "libx264",
        pix_fmt: str = "yuv420p",
        preset: str = "medium",
        extra: Sequence[str] | None = None,
    ):
        if not ffmpeg_available():
            raise RuntimeError("ffmpeg is not available on this system")
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.size = (_even(size[0]), _even(size[1]))
        self.fps = max(1, int(fps))
        self.frames_written = 0
        self._command = [
            ffmpeg_path() or "ffmpeg",
            "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{self.size[0]}x{self.size[1]}",
            "-pix_fmt", "rgb24", "-r", str(self.fps),
            "-i", "-",
            "-an",
            "-vcodec", codec,
            "-pix_fmt", pix_fmt,
            "-crf", str(int(crf)),
            "-preset", preset,
            *(extra or []),
            str(self.output),
        ]
        self._process: subprocess.Popen | None = None

    def __enter__(self) -> "VideoWriter":
        self._process = subprocess.Popen(
            self._command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        return self

    def write(self, frame: Image.Image | np.ndarray) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("VideoWriter used outside its context manager")
        if isinstance(frame, Image.Image):
            if frame.mode != "RGB":
                frame = frame.convert("RGB")
            if frame.size != self.size:
                frame = frame.resize(self.size, Image.BILINEAR)
            data = frame.tobytes()
        else:
            array = np.ascontiguousarray(frame, dtype=np.uint8)
            if array.shape[1] != self.size[0] or array.shape[0] != self.size[1]:
                array = np.asarray(Image.fromarray(array).resize(self.size, Image.BILINEAR))
            data = array.tobytes()
        try:
            self._process.stdin.write(data)
            self.frames_written += 1
        except BrokenPipeError as exc:
            stderr = self._process.stderr.read().decode("utf-8", "replace") if self._process.stderr else ""
            raise RuntimeError(f"ffmpeg rejected the stream: {stderr[:400]}") from exc

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._process is not None
        if self._process.stdin:
            try:
                self._process.stdin.close()
            except OSError:
                pass
        code = self._process.wait(timeout=300)
        stderr = self._process.stderr.read().decode("utf-8", "replace") if self._process.stderr else ""
        if code != 0 and exc_type is None:
            raise RuntimeError(f"ffmpeg exited with {code}: {stderr[-800:]}")
        self._process = None


def _codec_for(fmt: str) -> tuple[str, str, list[str]]:
    fmt = (fmt or "mp4").lower().lstrip(".")
    if fmt in {"webm", "vp9"}:
        return "libvpx-vp9", "yuv420p", ["-b:v", "0"]
    if fmt in {"gif",}:
        return "gif", "rgb8", []
    if fmt in {"mov",}:
        return "libx264", "yuv420p", ["-movflags", "+faststart"]
    return "libx264", "yuv420p", ["-movflags", "+faststart"]


def encode_frames(
    frames: Iterable[Image.Image | np.ndarray],
    output: str | Path,
    size: tuple[int, int],
    fps: int = 30,
    fmt: str = "mp4",
    crf: int = 20,
    progress: ProgressCallback | None = None,
    total: int | None = None,
) -> Path:
    """Encode an iterable of frames into a video file."""
    output = Path(output)
    fmt = (fmt or output.suffix.lstrip(".") or "mp4").lower().lstrip(".")
    codec, pix_fmt, extra = _codec_for(fmt)
    if output.suffix.lstrip(".").lower() != fmt:
        output = output.with_suffix("." + fmt)

    with VideoWriter(output, size, fps=fps, crf=crf, codec=codec, pix_fmt=pix_fmt, extra=extra) as writer:
        for index, frame in enumerate(frames, start=1):
            writer.write(frame)
            if progress and total:
                progress(index, total)
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"encoding produced an empty file: {output}")
    log.info("Encoded %s frames -> %s (%d bytes)", writer.frames_written, output, output.stat().st_size)
    return output


def mux_audio(video: str | Path, audio: str | Path, output: str | Path, shortest: bool = True) -> Path:
    """Combine a silent video with an audio track."""
    output = Path(output)
    args = ["-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
    if shortest:
        args.append("-shortest")
    args.append(str(output))
    result = _run(args, timeout=900)
    if result.returncode != 0 or not output.exists():
        raise RuntimeError(f"audio mux failed: {(result.stderr or b'').decode('utf-8', 'replace')[-500:]}")
    return output


def trim(input_path: str | Path, output: str | Path, start: float = 0.0, duration: float | None = None) -> Path:
    """Cut a segment without re-encoding (fast, stream copy)."""
    args = ["-ss", f"{max(0.0, start):.3f}", "-i", str(input_path)]
    if duration:
        args += ["-t", f"{max(0.01, duration):.3f}"]
    args += ["-c", "copy", str(output)]
    result = _run(args, timeout=900)
    if result.returncode != 0:
        # some containers refuse stream copy — retry with re-encode
        args = ["-ss", f"{max(0.0, start):.3f}", "-i", str(input_path)]
        if duration:
            args += ["-t", f"{max(0.01, duration):.3f}"]
        args += ["-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-c:a", "aac", str(output)]
        result = _run(args, timeout=1800)
    if result.returncode != 0 or not Path(output).exists():
        raise RuntimeError(f"trim failed: {(result.stderr or b'').decode('utf-8', 'replace')[-500:]}")
    return Path(output)


def thumbnail_at(input_path: str | Path, output: str | Path, at: float = 0.0, width: int = 640) -> Path:
    """Grab one frame as a JPEG thumbnail."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = _run(
        ["-ss", f"{max(0.0, at):.3f}", "-i", str(input_path), "-frames:v", "1", "-vf", f"scale={width}:-2", str(output)],
        timeout=180,
    )
    if result.returncode != 0 or not output.exists():
        raise RuntimeError(f"thumbnail failed: {(result.stderr or b'').decode('utf-8', 'replace')[-400:]}")
    return output


def extract_frames(input_path: str | Path, out_dir: str | Path, every: float = 1.0, fmt: str = "png") -> list[Path]:
    """Extract one frame every ``every`` seconds into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / f"frame_%05d.{fmt}"
    result = _run(
        ["-i", str(input_path), "-vf", f"fps=1/{max(0.04, every)}", str(pattern)],
        timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError(f"frame extraction failed: {(result.stderr or b'').decode('utf-8', 'replace')[-400:]}")
    return sorted(out_dir.glob(f"frame_*.{fmt}"))


def images_to_video(
    images: Sequence[str | Path],
    output: str | Path,
    fps: int = 30,
    per_image: float = 1.0,
    size: tuple[int, int] | None = None,
    crf: int = 20,
) -> Path:
    """Slideshow: hold each image for ``per_image`` seconds."""
    frames: list[Image.Image] = []
    target = size
    for path in images:
        with Image.open(path) as source:
            frame = source.convert("RGB")
        if target is None:
            target = (frame.width, frame.height)
        if frame.size != target:
            frame = frame.resize(target, Image.LANCZOS)
        for _ in range(max(1, int(round(per_image * fps)))):
            frames.append(frame)
    if not frames:
        raise ValueError("no images supplied")
    return encode_frames(frames, output, target or (1920, 1080), fps=fps, fmt=Path(output).suffix.lstrip("."), crf=crf)


def to_gif(
    frames: Iterable[Image.Image],
    output: str | Path,
    size: tuple[int, int],
    fps: int = 15,
    palette_colors: int = 128,
) -> Path:
    """Encode an animated GIF with an optimised palette."""
    output = Path(output)
    materialised = [f.convert("RGB").resize(size, Image.BILINEAR) for f in frames]
    if not materialised:
        raise ValueError("no frames supplied")
    materialised[0].save(
        output,
        save_all=True,
        append_images=materialised[1:],
        duration=int(1000 / max(1, fps)),
        loop=0,
        optimize=True,
    )
    return output


def concat_videos(inputs: Sequence[str | Path], output: str | Path) -> Path:
    """Concatenate videos that share codec/geometry (stream copy)."""
    list_file = Path(output).with_suffix(".concat.txt")
    list_file.write_text(
        "".join(f"file '{Path(p).as_posix()}'\n" for p in inputs), encoding="utf-8"
    )
    result = _run(["-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(output)], timeout=1800)
    list_file.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"concat failed: {(result.stderr or b'').decode('utf-8', 'replace')[-400:]}")
    return Path(output)


__all__ = [
    "ffmpeg_path",
    "ffmpeg_available",
    "set_ffmpeg",
    "probe",
    "read_frames",
    "VideoWriter",
    "encode_frames",
    "mux_audio",
    "trim",
    "thumbnail_at",
    "extract_frames",
    "images_to_video",
    "to_gif",
    "concat_videos",
]
