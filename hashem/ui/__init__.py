"""Local web studio (REST + Server-Sent Events). No external framework."""

from __future__ import annotations

from .server import serve

__all__ = ["serve"]
