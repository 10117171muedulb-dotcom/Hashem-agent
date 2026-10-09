"""The application context handed to every tool handler."""

from __future__ import annotations

from pathlib import Path

from ..config import Settings, save_settings
from ..learning import SkillStore
from ..llm.provider import ChatProvider
from ..utils.log import get_logger
from ..utils.paths import app_data_dir, cache_dir, exports_dir
from ..utils.safety import PathGuard

log = get_logger("agent.context")


class AppContext:
    def __init__(self, settings: Settings | None = None):
        from ..config import load_settings

        self.settings = settings or load_settings()
        self.provider = ChatProvider(self.settings.provider)
        self.skills = SkillStore()
        self.workspace = self.settings.resolved_workspace()
        self.exports = exports_dir()
        self.cache = cache_dir()
        self.guard = PathGuard([self.workspace, self.exports, self.cache, app_data_dir()])

    # --------------------------------------------------------------- lifecycle
    def rebuild_provider(self) -> None:
        self.provider = ChatProvider(self.settings.provider)

    def save(self) -> None:
        save_settings(self.settings)

    def refresh_paths(self) -> None:
        self.workspace = self.settings.resolved_workspace()
        self.guard = PathGuard([self.workspace, self.exports, self.cache, app_data_dir()])

    # ----------------------------------------------------------------- helpers
    def output_path(self, name: str, suffix: str) -> Path:
        safe = "".join(ch for ch in name if ch.isalnum() or ch in " -_").strip() or "output"
        return self.exports / f"{safe[:50]}{suffix}"


__all__ = ["AppContext"]
