"""Persistent, atomic application settings.

Settings live in a single JSON document under the per-user app data directory so
the app is portable and can be inspected or hand-edited.  Writes go through a
temporary file + ``os.replace`` so a crash mid-write can never leave a truncated
file behind.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .utils.log import get_logger
from .utils.paths import app_data_dir, ensure_dir, exports_dir, workspace_dir

log = get_logger("config")

CONFIG_VERSION = 1


@dataclass
class ProviderSettings:
    """Which model brain the agent talks to."""

    provider: str = "ollama"          # ollama | openai | openai_compatible | lmstudio
    model: str = "qwen2.5:7b"
    base_url: str = "http://127.0.0.1:11434"
    api_key: str = ""                 # stored locally only, never transmitted elsewhere
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 300
    stream: bool = True

    def describe(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("api_key"):
            data["api_key"] = "***set***"
        return data


@dataclass
class AgentSettings:
    max_steps: int = 12
    auto_approve: bool = False        # auto-run commands the guard flags as risky
    allow_shell: bool = True
    allow_web: bool = True
    system_prompt_extra: str = ""
    language: str = "ar"              # ar | en


@dataclass
class VideoSettings:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    quality: int = 20                 # ffmpeg CRF (lower = better)
    format: str = "mp4"               # mp4 | webm | gif
    preview_scale: float = 0.5        # live-preview render scale
    output_dir: str = ""              # empty => app_data/exports
    auto_music: bool = True
    music_style: str = "corporate"
    music_volume: float = 0.28

    def resolved_output_dir(self) -> Path:
        if self.output_dir:
            return ensure_dir(self.output_dir)
        return exports_dir()


@dataclass
class UISettings:
    host: str = "127.0.0.1"
    port: int = 8765
    open_browser: bool = True
    theme: str = "dark"


@dataclass
class Settings:
    version: int = CONFIG_VERSION
    provider: ProviderSettings = field(default_factory=ProviderSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    video: VideoSettings = field(default_factory=VideoSettings)
    ui: UISettings = field(default_factory=UISettings)
    workspace: str = ""

    # -------------------------------------------------------------- serialising
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        """Build settings from a dict, ignoring unknown keys (forward compatible)."""
        def pick(name: str, model: type) -> Any:
            payload = data.get(name) or {}
            if not isinstance(payload, dict):
                return model()
            valid = {k: v for k, v in payload.items() if k in model.__dataclass_fields__}
            try:
                return model(**valid)
            except TypeError:  # pragma: no cover - defensive
                log.warning("Falling back to defaults for section %r", name)
                return model()

        return cls(
            version=int(data.get("version", CONFIG_VERSION)),
            provider=pick("provider", ProviderSettings),
            agent=pick("agent", AgentSettings),
            video=pick("video", VideoSettings),
            ui=pick("ui", UISettings),
            workspace=str(data.get("workspace", "")),
        )

    # ------------------------------------------------------------------ helpers
    def resolved_workspace(self) -> Path:
        if self.workspace:
            return ensure_dir(self.workspace)
        return workspace_dir()

    def redacted(self) -> dict[str, Any]:
        data = self.to_dict()
        if data["provider"].get("api_key"):
            data["provider"]["api_key"] = "***set***"
        return data


_LOCK = threading.RLock()


def config_path() -> Path:
    return app_data_dir() / "settings.json"


def load_settings(path: Path | None = None) -> Settings:
    """Load settings, falling back to defaults for anything unreadable."""
    target = Path(path) if path else config_path()
    with _LOCK:
        if not target.exists():
            settings = Settings()
            save_settings(settings, target)
            return settings
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not read %s (%s) — using defaults", target, exc)
            return Settings()
    return Settings.from_dict(raw if isinstance(raw, dict) else {})


def save_settings(settings: Settings, path: Path | None = None) -> Path:
    """Atomically write settings and return the path written."""
    target = Path(path) if path else config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    payload = json.dumps(settings.to_dict(), indent=2, ensure_ascii=False)
    with _LOCK:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, target)
    return target


# Free / no-cost model presets shown in the UI. All of them are either local
# (Ollama, LM Studio) or require only a free-tier key from the provider.
MODEL_PRESETS: list[dict[str, str]] = [
    {
        "id": "ollama-local",
        "label": "Ollama (محلي — مجاني 100٪ وبدون إنترنت)",
        "provider": "ollama",
        "base_url": "http://127.0.0.1:11434",
        "model": "qwen2.5:7b",
        "api_key": "",
        "note": "Requires `ollama serve` running locally.",
    },
    {
        "id": "ollama-small",
        "label": "Ollama — نموذج خفيف للأجهزة الضعيفة",
        "provider": "ollama",
        "base_url": "http://127.0.0.1:11434",
        "model": "qwen2.5:3b",
        "api_key": "",
        "note": "Faster on 8 GB RAM machines.",
    },
    {
        "id": "lmstudio",
        "label": "LM Studio (محلي)",
        "provider": "openai_compatible",
        "base_url": "http://127.0.0.1:1234/v1",
        "model": "local-model",
        "api_key": "",
        "note": "Start the LM Studio local server first.",
    },
    {
        "id": "groq-free",
        "label": "Groq — مجاني بمفتاح",
        "provider": "openai_compatible",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "api_key": "",
        "note": "Free tier, needs GROQ_API_KEY.",
    },
    {
        "id": "openrouter-free",
        "label": "OpenRouter — نماذج مجانية",
        "provider": "openai_compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "api_key": "",
        "note": "Free models, needs OPENROUTER_API_KEY.",
    },
    {
        "id": "gemini-free",
        "label": "Google Gemini — مجاني بمفتاح",
        "provider": "openai_compatible",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.0-flash",
        "api_key": "",
        "note": "Free tier, needs a Google AI Studio key.",
    },
    {
        "id": "openai",
        "label": "OpenAI (مدفوع)",
        "provider": "openai_compatible",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_key": "",
        "note": "Paid.",
    },
]


def preset_by_id(preset_id: str) -> dict[str, str] | None:
    for preset in MODEL_PRESETS:
        if preset["id"] == preset_id:
            return preset
    return None


def apply_preset(settings: Settings, preset_id: str, api_key: str = "") -> Settings:
    """Copy a preset into the live settings (optionally attaching an API key)."""
    preset = preset_by_id(preset_id)
    if not preset:
        raise KeyError(f"unknown preset: {preset_id}")
    settings.provider.provider = preset["provider"]
    settings.provider.base_url = preset["base_url"]
    settings.provider.model = preset["model"]
    if api_key:
        settings.provider.api_key = api_key
    return settings


__all__ = [
    "Settings",
    "ProviderSettings",
    "AgentSettings",
    "VideoSettings",
    "UISettings",
    "CONFIG_VERSION",
    "MODEL_PRESETS",
    "config_path",
    "load_settings",
    "save_settings",
    "preset_by_id",
    "apply_preset",
]
