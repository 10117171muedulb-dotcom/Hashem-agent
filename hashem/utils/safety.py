"""Safety rails for anything the agent is allowed to execute or touch.

Two independent guards live here:

* :class:`PathGuard` — a filesystem jail.  Every tool that accepts a path resolves
  it through the guard first, so a model that hallucinates ``C:\\Windows\\system32``
  (or a prompt-injection payload) cannot escape the workspace.
* :func:`looks_destructive` — a conservative classifier for shell commands that
  forces an explicit confirmation step.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

from .log import get_logger
from .paths import ensure_dir

log = get_logger("safety")

# Commands that can destroy data or the machine. Matched against the first token
# and, for a few, anywhere in the command line.
_DESTRUCTIVE_PREFIXES = (
    "rm", "rmdir", "del", "erase", "format", "mkfs", "dd", "shred",
    "rd", "remove-item", "ri", "diskpart", "reg", "regedit", "bcdedit",
    "shutdown", "restart-computer", "stop-computer", "taskkill",
    "chmod", "chown", "takeown", "icacls", "cipher",
    "curl|sh", "wget|sh",
)

_DESTRUCTIVE_PATTERNS = (
    re.compile(r"\brm\s+-[a-z]*[rf]"),
    re.compile(r"\bgit\s+(push\s+.*(--force|-f)|reset\s+--hard|clean\s+-[a-z]*f)"),
    re.compile(r">\s*/dev/sd[a-z]"),
    re.compile(r":\(\)\s*\{.*\};\s*:"),  # fork bomb
    re.compile(r"\bRemove-Item\b.*-Recurse", re.IGNORECASE),
    re.compile(r"\bFormat-Volume\b", re.IGNORECASE),
    re.compile(r"\bdrop\s+(database|table)\b", re.IGNORECASE),
    re.compile(r"\btruncate\s+table\b", re.IGNORECASE),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bcurl\b.*\|\s*(ba)?sh"),
    re.compile(r"\bInvoke-Expression\b", re.IGNORECASE),
    re.compile(r"\biex\b", re.IGNORECASE),
)

# Never allow these to be read or written, even inside the workspace.
_FORBIDDEN_NAMES = {".git", ".ssh", ".aws", ".gnupg", "id_rsa", "id_ed25519"}
_FORBIDDEN_SUFFIXES = (".pem", ".key", ".p12", ".pfx", "credentials", ".netrc")


class PathViolation(Exception):
    """Raised when a path escapes the allowed roots."""


class PathGuard:
    """Resolve and validate paths against a set of allowed root directories."""

    def __init__(self, roots: Iterable[Path | str]):
        resolved: list[Path] = []
        for root in roots:
            try:
                resolved.append(Path(root).expanduser().resolve())
            except (OSError, RuntimeError):
                continue
        if not resolved:
            raise ValueError("PathGuard needs at least one valid root directory")
        self.roots = resolved

    # ------------------------------------------------------------------ helpers
    @property
    def primary(self) -> Path:
        """The first root — used as the default for relative paths."""
        return self.roots[0]

    def _is_within(self, path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------- public
    def resolve(self, raw: str | Path, *, must_exist: bool = False, create_parents: bool = False) -> Path:
        """Turn ``raw`` into an absolute, validated path.

        Relative paths are resolved against the primary root.
        """
        candidate = Path(os.path.expandvars(str(raw)))
        if not candidate.is_absolute():
            candidate = self.primary / candidate

        try:
            candidate = candidate.resolve()
        except (OSError, RuntimeError) as exc:  # pragma: no cover - platform quirks
            raise PathViolation(f"Cannot resolve path {raw!r}: {exc}") from exc

        if not any(self._is_within(candidate, root) for root in self.roots):
            raise PathViolation(
                f"{candidate} is outside the allowed workspace "
                f"({', '.join(str(r) for r in self.roots)})"
            )

        self._check_sensitive(candidate)

        if must_exist and not candidate.exists():
            raise PathViolation(f"{candidate} does not exist")

        if create_parents:
            ensure_dir(candidate.parent)

        return candidate

    def check(self, raw: str | Path) -> Path:
        """Alias of :meth:`resolve` reading better at call sites that only validate."""
        return self.resolve(raw)

    def _check_sensitive(self, path: Path) -> None:
        lowered = {p.lower() for p in path.parts}
        if lowered & _FORBIDDEN_NAMES:
            raise PathViolation(f"{path} is a protected path and cannot be accessed")
        name = path.name.lower()
        if name.endswith(_FORBIDDEN_SUFFIXES):
            raise PathViolation(f"{path} looks like a credential file and cannot be accessed")

    def allows(self, raw: str | Path) -> bool:
        try:
            self.resolve(raw)
            return True
        except PathViolation:
            return False


def looks_destructive(command: str) -> tuple[bool, str]:
    """Return ``(is_risky, reason)`` for a shell command.

    Deliberately over-approximates: a false positive only costs the user one
    confirmation click, a false negative can cost them their disk.
    """
    if not command or not command.strip():
        return False, ""

    text = command.strip()
    lowered = text.lower()

    for pattern in _DESTRUCTIVE_PATTERNS:
        if pattern.search(text):
            return True, f"matches destructive pattern {pattern.pattern!r}"

    first = re.split(r"[\s;&|]+", lowered)[0]
    if first in _DESTRUCTIVE_PREFIXES:
        return True, f"command {first!r} can delete or reconfigure data"

    if ";" in text or "&&" in text or "|" in text:
        for part in re.split(r"[;&|]+", lowered):
            token = part.strip().split(" ")[0]
            if token in _DESTRUCTIVE_PREFIXES:
                return True, f"chained command {token!r} can delete or reconfigure data"

    return False, ""


def redact(text: str) -> str:
    """Mask anything that looks like a secret before it reaches logs or the UI."""
    patterns = (
        (re.compile(r"(sk|pk|ghp|gho|xox[baprs]|AIza)[A-Za-z0-9_\-]{8,}"), "***REDACTED***"),
        (re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*\S+"), r"\1=***"),
    )
    for pattern, replacement in patterns:
        text = pattern.sub(replacement, text)
    return text


__all__ = ["PathGuard", "PathViolation", "looks_destructive", "redact"]
