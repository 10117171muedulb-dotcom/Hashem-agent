"""Audio: procedural royalty-free music, text-to-speech and mixing.

Three independent sources, each free:

* **Music** is synthesised locally with numpy — chord progressions, bass, pads,
  plucks and a drum kit.  No licence, no network, no royalties.
* **Narration** prefers Microsoft's Edge neural TTS (free, excellent Arabic
  voices) and falls back to the built-in Windows SAPI voice when offline.
* **Mixing** does loudness normalisation plus sidechain-style ducking so the
  music drops under speech.

Every function tolerates the optional dependency being absent.
"""

from __future__ import annotations

import asyncio
import math
import os
import struct
import subprocess
import sys
import wave
from pathlib import Path
from typing import Sequence

import numpy as np

from ..utils.log import get_logger

log = get_logger("audio")

SAMPLE_RATE = 44100

# --------------------------------------------------------------------------- #
# WAV I/O (stdlib only — no scipy/librosa dependency)
# --------------------------------------------------------------------------- #


def save_wav(path: str | Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    """Write float ``[-1, 1]`` samples (mono or stereo) to a 16-bit PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)
    data = np.clip(data, -1.0, 1.0)
    pcm = (data * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return path


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a WAV into a float ``[-1, 1]`` stereo array plus its sample rate."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())

    if width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:  # pragma: no cover - exotic bit depths
        raise ValueError(f"unsupported WAV sample width: {width}")

    if channels > 1:
        samples = samples.reshape(-1, channels)
        if channels > 2:
            samples = samples[:, :2]
    else:
        samples = np.stack([samples, samples], axis=1)
    return samples, rate


def resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Linear resample — good enough for voice/music beds."""
    if from_rate == to_rate or samples.size == 0:
        return samples
    ratio = to_rate / float(from_rate)
    new_length = max(1, int(round(len(samples) * ratio)))
    positions = np.linspace(0, len(samples) - 1, new_length)
    indices = np.clip(positions.astype(int), 0, len(samples) - 1)
    next_indices = np.clip(indices + 1, 0, len(samples) - 1)
    fraction = (positions - indices)[..., None]
    return (samples[indices] * (1 - fraction) + samples[next_indices] * fraction).astype(np.float32)


# --------------------------------------------------------------------------- #
# synthesis primitives
# --------------------------------------------------------------------------- #


def _adsr(length: int, rate: int, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    env = np.ones(length, dtype=np.float32)
    a = max(1, int(attack * rate))
    d = max(1, int(decay * rate))
    r = max(1, int(release * rate))
    if a < length:
        env[:a] = np.linspace(0, 1, a, dtype=np.float32)
    if a + d < length:
        env[a : a + d] = np.linspace(1, sustain, d, dtype=np.float32)
        env[a + d :] = sustain
    if r < length:
        tail = max(0, length - r)
        env[tail:] *= np.linspace(1, 0, length - tail, dtype=np.float32)
    return env


def _waveform(kind: str, phase: np.ndarray) -> np.ndarray:
    two_pi_phase = 2.0 * np.pi * phase
    if kind == "sine":
        return np.sin(two_pi_phase)
    if kind == "square":
        return np.sign(np.sin(two_pi_phase))
    if kind == "saw":
        return 2.0 * (phase % 1.0) - 1.0
    if kind == "triangle":
        return 2.0 * np.abs(2.0 * (phase % 1.0) - 1.0) - 1.0
    if kind in {"pluck", "karplus"}:
        # a couple of decaying harmonics reads as a plucked string
        return (
            np.sin(two_pi_phase) * 1.0
            + np.sin(two_pi_phase * 2) * 0.45
            + np.sin(two_pi_phase * 3) * 0.22
            + np.sin(two_pi_phase * 5) * 0.08
        )
    if kind == "pad":
        return np.sin(two_pi_phase) * 0.7 + np.sin(two_pi_phase * 1.005) * 0.5 + np.sin(two_pi_phase * 2) * 0.18
    return np.sin(two_pi_phase)


def _tone(
    freq: float,
    duration: float,
    rate: int,
    kind: str = "sine",
    amplitude: float = 0.3,
    attack: float = 0.01,
    decay: float = 0.08,
    sustain: float = 0.7,
    release: float = 0.12,
    detune: float = 0.0,
) -> np.ndarray:
    length = max(1, int(duration * rate))
    t = np.arange(length, dtype=np.float32) / rate
    phase = np.cumsum(np.full(length, freq, dtype=np.float32)) / rate
    left = _waveform(kind, phase)
    right = _waveform(kind, phase * (1.0 + detune)) if detune else left
    env = _adsr(length, rate, attack, decay, sustain, release)
    stereo = np.stack([left * env, right * env], axis=1).astype(np.float32)
    return stereo * amplitude


def _noise(duration: float, rate: int, amplitude: float = 0.3) -> np.ndarray:
    length = max(1, int(duration * rate))
    noise = np.random.default_rng(7).uniform(-1, 1, length).astype(np.float32)
    env = np.exp(-np.linspace(0, 12, length, dtype=np.float32))
    mono = noise * env * amplitude
    return np.stack([mono, mono], axis=1).astype(np.float32)


def _kick(duration: float, rate: int, amplitude: float = 0.9) -> np.ndarray:
    length = max(1, int(duration * rate))
    t = np.arange(length, dtype=np.float32) / rate
    freq = 120.0 * np.exp(-t * 28.0) + 45.0
    phase = np.cumsum(freq) / rate
    body = np.sin(2 * np.pi * phase) * np.exp(-t * 9.0)
    click = np.random.default_rng(3).uniform(-1, 1, length).astype(np.float32) * np.exp(-t * 220.0) * 0.35
    mono = (body + click) * amplitude
    return np.stack([mono, mono], axis=1).astype(np.float32)


def _lowpass(samples: np.ndarray, cutoff: float, rate: int) -> np.ndarray:
    """One-pole low-pass, applied per channel. Cheap and stable."""
    rc = 1.0 / (2.0 * math.pi * max(20.0, cutoff))
    alpha = (1.0 / rate) / (rc + 1.0 / rate)
    out = np.empty_like(samples)
    previous = np.zeros(samples.shape[1], dtype=np.float32)
    for i in range(len(samples)):
        previous = previous + alpha * (samples[i] - previous)
        out[i] = previous
    return out


# --------------------------------------------------------------------------- #
# music styles
# --------------------------------------------------------------------------- #

STYLES: dict[str, dict] = {
    "corporate": {
        "bpm": 100, "root": 0, "scale": [0, 2, 4, 7, 9],
        "progression": [[0, 4, 7], [5, 9, 12], [-3, 0, 4], [-5, -1, 2]],
        "lead": "pluck", "pad": True, "drums": "soft", "bass_wave": "sine",
    },
    "upbeat": {
        "bpm": 124, "root": 2, "scale": [0, 2, 3, 5, 7, 10],
        "progression": [[0, 3, 7], [5, 8, 12], [-2, 2, 5], [3, 7, 10]],
        "lead": "pluck", "pad": True, "drums": "four", "bass_wave": "square",
    },
    "cinematic": {
        "bpm": 76, "root": -3, "scale": [0, 2, 3, 5, 7, 8, 11],
        "progression": [[0, 3, 7], [-2, 2, 5], [-5, -1, 2], [-4, 0, 3]],
        "lead": "pad", "pad": True, "drums": "epic", "bass_wave": "sine",
    },
    "lofi": {
        "bpm": 82, "root": -1, "scale": [0, 2, 3, 5, 7, 10],
        "progression": [[0, 3, 7, 10], [-2, 2, 5, 9], [-4, 0, 3, 7], [-5, -1, 2, 6]],
        "lead": "pad", "pad": True, "drums": "lofi", "bass_wave": "triangle",
    },
    "tech": {
        "bpm": 128, "root": 0, "scale": [0, 3, 5, 7, 10],
        "progression": [[0, 7], [-2, 5], [-4, 3], [-5, 2]],
        "lead": "saw", "pad": False, "drums": "four", "bass_wave": "saw",
    },
    "ambient": {
        "bpm": 60, "root": -5, "scale": [0, 2, 4, 7, 9],
        "progression": [[0, 7, 12], [-3, 4, 9], [-5, 2, 7], [-1, 6, 11]],
        "lead": "pad", "pad": True, "drums": "none", "bass_wave": "sine",
    },
    "epic": {
        "bpm": 92, "root": -5, "scale": [0, 2, 3, 5, 7, 8, 10],
        "progression": [[0, 3, 7], [-5, -1, 2], [-3, 0, 4], [-7, -3, 0]],
        "lead": "saw", "pad": True, "drums": "epic", "bass_wave": "saw",
    },
    "minimal": {
        "bpm": 110, "root": 0, "scale": [0, 3, 7],
        "progression": [[0, 7], [0, 7], [-2, 5], [-4, 3]],
        "lead": "sine", "pad": False, "drums": "soft", "bass_wave": "sine",
    },
}


def music_styles() -> list[str]:
    return sorted(STYLES)


def _midi_to_freq(root: int, semitones: float) -> float:
    return 440.0 * (2.0 ** ((root + semitones - 9.0) / 12.0))


def render_music(
    duration: float,
    style: str = "corporate",
    seed: int = 1,
    sample_rate: int = SAMPLE_RATE,
    volume: float = 1.0,
) -> np.ndarray:
    """Synthesise ``duration`` seconds of stereo music as float ``[-1, 1]``."""
    duration = max(0.5, float(duration))
    cfg = STYLES.get((style or "corporate").lower(), STYLES["corporate"])
    rng = np.random.default_rng(seed)
    bpm = float(cfg["bpm"])
    beat = 60.0 / bpm
    total = int(duration * sample_rate)
    mix = np.zeros((total, 2), dtype=np.float32)

    def add(samples: np.ndarray, at_seconds: float, gain: float = 1.0) -> None:
        start = int(max(0.0, at_seconds) * sample_rate)
        if start >= total or samples.size == 0:
            return
        length = min(len(samples), total - start)
        mix[start : start + length] += samples[:length] * gain

    beats = int(math.ceil(duration / beat)) + 2
    progression = cfg["progression"]

    for bar in range(beats // 4 + 1):
        bar_time = bar * 4 * beat
        if bar_time >= duration + beat:
            break
        chord = progression[bar % len(progression)]

        # pad / chord bed
        if cfg["pad"]:
            for note in chord:
                freq = _midi_to_freq(cfg["root"] + 60, note)
                add(_tone(freq, 4 * beat * 0.98, sample_rate, "pad", 0.075, 0.4, 0.6, 0.9, 0.8, detune=0.003), bar_time, 0.9)

        # bass line
        for step in range(4):
            if cfg["drums"] == "lofi" and step % 2 == 1:
                continue
            freq = _midi_to_freq(cfg["root"] + 36, chord[0])
            add(_tone(freq, beat * 0.85, sample_rate, cfg["bass_wave"], 0.34, 0.005, 0.05, 0.75, 0.1), bar_time + step * beat, 0.85)

        # arpeggio / lead
        for step in range(8):
            if rng.random() < (0.25 if cfg["drums"] == "none" else 0.12):
                continue
            note = chord[step % len(chord)] + (12 if step % 4 >= 2 else 0)
            freq = _midi_to_freq(cfg["root"] + 60, note)
            add(_tone(freq, beat * 0.45, sample_rate, cfg["lead"], 0.11, 0.004, 0.09, 0.35, 0.16), bar_time + step * beat * 0.5, 0.8)

        # drums
        kit = cfg["drums"]
        for step in range(8):
            t = bar_time + step * beat * 0.5
            if kit == "four" and step % 2 == 0:
                add(_kick(0.28, sample_rate, 0.85), t, 0.9)
            elif kit == "epic" and step in (0, 3, 6):
                add(_kick(0.42, sample_rate, 1.0), t, 1.0)
            elif kit == "soft" and step == 0:
                add(_kick(0.24, sample_rate, 0.6), t, 0.7)
            elif kit == "lofi" and step in (0, 5):
                add(_kick(0.22, sample_rate, 0.5), t, 0.6)

            if kit in {"four", "epic", "lofi"} and step % 2 == 1:
                add(_noise(0.16, sample_rate, 0.16), t, 0.55)
            if kit in {"four", "lofi"}:
                add(_noise(0.05, sample_rate, 0.07), t + beat * 0.25, 0.4)

    # gentle fades so the bed never clicks at the edges
    fade = int(min(1.5, duration * 0.15) * sample_rate)
    if fade > 0:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)[..., None]
        mix[:fade] *= ramp
        mix[-fade:] *= ramp[::-1]

    return normalize(mix, target=0.89) * max(0.0, min(1.0, volume))


def normalize(samples: np.ndarray, target: float = 0.95) -> np.ndarray:
    """Peak-normalise without clipping."""
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak < 1e-6:
        return samples
    return (samples * (target / peak)).astype(np.float32)


def mix_tracks(
    narration: np.ndarray | None,
    music: np.ndarray | None,
    music_volume: float = 0.3,
    duck: float = 0.45,
    fade_out: float = 1.2,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Mix narration over music with sidechain-style ducking under speech."""
    tracks = [t for t in (narration, music) if t is not None and len(t)]
    if not tracks:
        return np.zeros((int(0.5 * sample_rate), 2), dtype=np.float32)

    length = max(len(t) for t in tracks)
    out = np.zeros((length, 2), dtype=np.float32)

    if music is not None and len(music):
        bed = music.astype(np.float32).copy()
        if narration is not None and len(narration):
            envelope = np.abs(narration).mean(axis=1)
            envelope = np.convolve(envelope, np.ones(int(0.12 * sample_rate)) / max(1, int(0.12 * sample_rate)), mode="same")
            gate = (envelope > 0.01).astype(np.float32)
            smoothed = np.convolve(gate, np.ones(int(0.18 * sample_rate)) / max(1, int(0.18 * sample_rate)), mode="same")
            gain = 1.0 - np.clip(smoothed, 0, 1) * max(0.0, min(0.95, duck))
            bed[: len(gain)] *= gain[..., None]
        out[: len(bed)] += bed * max(0.0, min(1.0, music_volume))

    if narration is not None and len(narration):
        out[: len(narration)] += narration.astype(np.float32)

    if fade_out > 0:
        tail = min(len(out), int(fade_out * sample_rate))
        if tail > 0:
            out[-tail:] *= np.linspace(1, 0, tail, dtype=np.float32)[..., None]

    return normalize(out, 0.95)


# --------------------------------------------------------------------------- #
# text to speech
# --------------------------------------------------------------------------- #

ARABIC_VOICES = [
    "ar-SA-HamedNeural", "ar-SA-ZariyahNeural", "ar-EG-ShakirNeural", "ar-EG-SalmaNeural",
    "ar-JO-SanaNeural", "ar-JO-TaimNeural", "ar-AE-HamdanNeural", "ar-AE-FatimaNeural",
    "ar-IQ-BasselNeural", "ar-IQ-RanaNeural", "ar-LB-LaylaNeural", "ar-LB-RamiNeural",
]
ENGLISH_VOICES = ["en-US-GuyNeural", "en-US-JennyNeural", "en-GB-RyanNeural", "en-GB-SoniaNeural"]


def has_edge_tts() -> bool:
    try:
        import edge_tts  # noqa: F401

        return True
    except Exception:
        return False


def list_voices(language: str = "ar") -> list[dict]:
    """List Edge TTS voices for a language (needs network on first call)."""
    if not has_edge_tts():
        return []

    async def _fetch() -> list[dict]:
        import edge_tts

        voices = await edge_tts.list_voices()
        return [v for v in voices if str(v.get("Locale", "")).startswith(language)]

    try:
        return asyncio.run(_fetch())
    except Exception as exc:  # noqa: BLE001 - offline is a normal case
        log.info("Could not list Edge TTS voices (%s)", exc)
        return []


def _contains_arabic(text: str) -> bool:
    return any("\u0600" <= ch <= "\u06FF" for ch in text or "")


def synthesize_edge(text: str, output: str | Path, voice: str = "", rate: int = 0) -> Path | None:
    """Neural TTS through Microsoft Edge's free service. Returns ``None`` offline."""
    if not has_edge_tts():
        return None
    chosen = voice or (ARABIC_VOICES[0] if _contains_arabic(text) else ENGLISH_VOICES[0])
    rate_pct = f"{'+' if rate >= 0 else ''}{int(rate) * 5}%"

    async def _run() -> None:
        import edge_tts

        communicate = edge_tts.Communicate(text, chosen, rate=rate_pct)
        await communicate.save(str(output))

    try:
        asyncio.run(_run())
        path = Path(output)
        if path.exists() and path.stat().st_size > 1000:
            log.info("Edge TTS produced %s with voice %s", path, chosen)
            return path
        log.warning("Edge TTS returned an empty file; falling back")
        return None
    except Exception as exc:  # noqa: BLE001
        log.info("Edge TTS unavailable (%s)", exc)
        return None


def synthesize_sapi(text: str, output: str | Path, voice: str = "", rate: int = 0) -> Path | None:
    """Offline Windows speech synthesis via the built-in SAPI voices."""
    if sys.platform != "win32":
        return None
    output = Path(output).with_suffix(".wav")
    output.parent.mkdir(parents=True, exist_ok=True)
    selector = f"Where-Object {{ $_.Name -like '*{voice}*' }} | Select-Object -First 1 | " if voice else ""
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$v = $s.GetInstalledVoices() | {selector}Select-Object -First 1; "
        "if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }; "
        f"$s.Rate = {max(-10, min(10, int(rate)))}; "
        f"$s.SetOutputToWaveFile('{output.as_posix()}'); "
        "$s.Speak([Console]::In.ReadToEnd()); "
        "$s.Dispose()"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.info("SAPI TTS failed (%s)", exc)
        return None
    if result.returncode == 0 and output.exists() and output.stat().st_size > 1000:
        return output
    log.info("SAPI TTS produced no audio: %s", (result.stderr or b"").decode("utf-8", "replace")[:200])
    return None


def synthesize_speech(
    text: str,
    output: str | Path,
    voice: str = "",
    rate: int = 0,
    provider: str = "auto",
) -> Path | None:
    """Best available TTS. Returns the WAV path, or ``None`` if nothing worked."""
    text = (text or "").strip()
    if not text:
        return None
    output = Path(output)
    if provider in {"auto", "edge"}:
        result = synthesize_edge(text, output, voice, rate)
        if result:
            return result
    if provider in {"auto", "sapi"}:
        return synthesize_sapi(text, output, voice, rate)
    return None


def speech_to_wav(
    text: str, output: str | Path, voice: str = "", rate: int = 0
) -> tuple[np.ndarray, int] | None:
    """Synthesise speech and return it as a float array ready for mixing."""
    path = synthesize_speech(text, output, voice, rate)
    if not path:
        return None
    try:
        samples, rate_hz = load_wav(path)
        return samples, rate_hz
    except (OSError, ValueError, wave.Error) as exc:
        log.warning("Could not decode TTS output (%s)", exc)
        return None


def silence(duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros((max(1, int(duration * sample_rate)), 2), dtype=np.float32)


__all__ = [
    "SAMPLE_RATE",
    "STYLES",
    "music_styles",
    "render_music",
    "normalize",
    "mix_tracks",
    "save_wav",
    "load_wav",
    "resample",
    "silence",
    "has_edge_tts",
    "list_voices",
    "synthesize_speech",
    "synthesize_edge",
    "synthesize_sapi",
    "speech_to_wav",
    "ARABIC_VOICES",
    "ENGLISH_VOICES",
]
