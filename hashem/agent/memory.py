"""Conversation persistence — sessions survive restarts."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils.log import get_logger
from ..utils.paths import sessions_dir

log = get_logger("agent.memory")


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    title: str = "محادثة جديدة"
    messages: list[dict[str, Any]] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    def add(self, role: str, content: str, **extra: Any) -> None:
        message = {"role": role, "content": content, **extra}
        self.messages.append(message)
        self.updated = time.time()

    def to_messages(self, max_messages: int = 40) -> list[dict[str, Any]]:
        return [m for m in self.messages if m.get("role") in {"user", "assistant", "tool", "system"}][-max_messages:]


class Memory:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else sessions_dir()

    # ------------------------------------------------------------------ paths
    def _path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    # ------------------------------------------------------------------ basic
    def new_session(self, title: str = "محادثة جديدة") -> Session:
        session = Session(title=title)
        self.save(session)
        return session

    def list_sessions(self) -> list[dict[str, Any]]:
        items = []
        for path in sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                items.append({"id": data.get("id"), "title": data.get("title"),
                              "messages": len(data.get("messages", [])), "updated": data.get("updated")})
            except (json.JSONDecodeError, OSError):
                continue
        return items

    def load(self, session_id: str) -> Session | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return Session(id=data.get("id", session_id), title=data.get("title", ""),
                       messages=data.get("messages", []), created=data.get("created", 0), updated=data.get("updated", 0))

    def save(self, session: Session) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(session.id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(session.__dict__, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False


__all__ = ["Session", "Memory"]
