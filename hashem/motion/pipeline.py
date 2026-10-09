"""End-to-end render pipeline: scene graph → frames → encoded video → audio.

Shared by the agent's ``create_video`` tool and the studio UI so both paths get
identical, verified output.  Returns a metrics dict the learning layer records.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..audio import (
    SAMPLE_RATE,
    mix_tracks,
    render_music,
    resample,
    save_wav,
    speech_to_wav,
)
from ..config import VideoSettings
from ..utils.log import get_logger
from .colors import contrast_ratio, parse_color
from .encoder import encode_frames, ffmpeg_available, mux_audio, probe
from .render import Renderer
from .scene import Project, parse_project, validate

log = get_logger("motion.pipeline")

Progress = Callable[[str, float], None]  # (stage, 0..1)


def _noop(_stage: str, _p: float) -> None:
    return None


def render_project(
    project: Project | dict[str, Any] | str,
    output: str | Path,
    video_settings: VideoSettings | None = None,
    base_dir: str | None = None,
    progress: Progress = _noop,
    include_audio: bool = True,
    tts: bool = True,
) -> dict[str, Any]:
    """Render everything and return a result/metrics dictionary."""
    started = time.time()
    settings = video_settings or VideoSettings()
    if not isinstance(project, Project):
        project = parse_project(project)

    warnings = validate(project)
    if not project.scenes:
        raise ValueError("المشروع لا يحتوي على مشاهد — لا يمكن التصدير.")

    # Honour the configured export size/fps by re-scaling the canvas.
    if settings.width and settings.height:
        project.canvas.width = int(settings.width)
        project.canvas.height = int(settings.height)
    if settings.fps:
        project.canvas.fps = int(settings.fps)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fmt = (settings.format or output.suffix.lstrip(".") or "mp4").lower().lstrip(".")

    progress("render", 0.05)
    renderer = Renderer(project, base_dir=base_dir)

    silent = output.with_name(output.stem + ".video." + ("mp4" if fmt in {"gif"} else fmt))
    progress("render", 0.1)

    def frame_progress(done: int, total: int, _fps: float) -> None:
        progress("render", 0.1 + 0.6 * (done / max(1, total)))

    renderer.progress = frame_progress
    video_path = encode_frames(
        renderer.frames(),
        silent,
        (project.canvas.width, project.canvas.height),
        fps=project.canvas.fps,
        fmt="mp4" if fmt == "gif" else fmt,
        crf=settings.quality,
    )
    progress("render", 0.72)

    final_path = video_path
    music_used = False
    narration_used = False

    if fmt == "gif":
        from .encoder import to_gif

        frames = list(Renderer(project, base_dir=base_dir).frames())
        final_path = output.with_suffix(".gif")
        to_gif(frames, final_path, (min(640, project.canvas.width), int(min(640, project.canvas.width) / project.canvas.aspect)), fps=12)
        silent.unlink(missing_ok=True)
    elif include_audio and ffmpeg_available():
        audio = _build_audio(project, settings, renderer.duration, progress, do_tts=tts)
        if audio is not None:
            progress("audio", 0.8)
            mix_path = output.with_name(output.stem + ".mix.wav")
            save_wav(mix_path, audio)
            try:
                final_path = mux_audio(video_path, mix_path, output)
                music_used = True
                narration_used = project.audio.narration.enabled
                silent.unlink(missing_ok=True)
                mix_path.unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001
                log.warning("audio mux failed (%s); delivering silent video", exc)
                final_path = video_path
                final_path = final_path.rename(output) if final_path != output else final_path

    if final_path != output and final_path.suffix != output.suffix:
        output = final_path  # e.g. requested .mp4 but a .gif was produced

    # When no audio track was muxed, deliver the file under the requested name.
    if final_path != output and not output.exists():
        try:
            final_path = final_path.rename(output)
        except OSError:
            pass

    progress("done", 1.0)
    metrics = _metrics(project, final_path)
    return {
        "path": str(final_path),
        "silent_path": str(silent) if silent.exists() else None,
        "format": fmt,
        "duration": round(renderer.duration, 2),
        "frames": renderer.frame_count,
        "size_bytes": final_path.stat().st_size if final_path.exists() else 0,
        "warnings": warnings,
        "music": music_used,
        "narration": narration_used,
        "render_seconds": round(time.time() - started, 2),
        "metrics": metrics,
    }


def _build_audio(project: Project, settings: VideoSettings, duration: float, progress: Progress, do_tts: bool) -> np.ndarray | None:
    music = None
    narration = None

    want_music = project.audio.music.enabled and not project.audio.music_file
    if want_music:
        progress("music", 0.74)
        volume = project.audio.music.volume or settings.music_volume
        music = render_music(duration + 0.5, style=project.audio.music.style or settings.music_style, volume=1.0)
        # fade handled by mixer; scale music volume there
        music_gain = volume
    elif project.audio.music_file and Path(project.audio.music_file).exists():
        loaded, rate = _load(project.audio.music_file)
        music = resample(loaded, rate, SAMPLE_RATE)
        music_gain = project.audio.music.volume or 0.3
    else:
        music_gain = project.audio.music.volume or settings.music_volume

    if project.audio.narration.enabled and project.audio.narration.text and do_tts:
        progress("tts", 0.76)
        from ..utils.paths import cache_dir

        tts_out = cache_dir() / "narration.wav"
        speech = speech_to_wav(project.audio.narration.text, tts_out, voice=project.audio.narration.voice, rate=project.audio.narration.rate)
        if speech is not None:
            loaded, rate = speech
            narration = resample(loaded, rate, SAMPLE_RATE)

    if music is None and narration is None:
        return None
    return mix_tracks(narration, music, music_volume=music_gain, fade_out=project.audio.music.fade_out)


def _load(path: str) -> tuple[np.ndarray, int]:
    from ..audio import load_wav

    return load_wav(path)


def _metrics(project: Project, path: Path) -> dict[str, Any]:
    """Cheap quality signals used for learning + self-critique."""
    metrics: dict[str, Any] = {}
    bg = project.background.colors[0] if project.background.colors else "#000000"

    text_layers = [l for scene in project.scenes for l in scene.layers if l.kind == "text" and l.text.strip()]
    if text_layers:
        ratios = [contrast_ratio(l.fill, bg) for l in text_layers]
        metrics["min_contrast"] = round(min(ratios), 2)
        metrics["avg_contrast"] = round(sum(ratios) / len(ratios), 2)
    metrics["scenes"] = len(project.scenes)
    metrics["layers"] = sum(len(s.layers) for s in project.scenes)
    metrics["duration"] = round(project.duration, 2)

    quality = 0.5
    contrast = metrics.get("min_contrast", 4.5)
    if contrast >= 4.5:
        quality += 0.25
    elif contrast >= 3.0:
        quality += 0.1
    if 1.0 <= project.duration <= 120:
        quality += 0.15
    if path.exists() and path.stat().st_size > 20_000:
        quality += 0.1
    metrics["quality"] = round(min(1.0, quality), 2)
    return metrics


__all__ = ["render_project"]
