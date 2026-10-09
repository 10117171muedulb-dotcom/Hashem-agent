"""Ingest any input the user drops in: images, video, .txt/.md/.docx/.srt.

Normalises everything into either pixels, frames metadata or clean paragraphs of
text so the rest of the studio only deals with one vocabulary.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..utils.log import get_logger
from . import video_ops

log = get_logger("media.ingest")

_ENCODINGS = ("utf-8-sig", "utf-8", "utf-16", "cp1256", "windows-1252", "latin-1")


def detect_kind(path: str | Path) -> str:
    """Return ``image`` | ``video`` | ``text`` | ``docx`` | ``other``."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in video_ops.VIDEO_SUFFIXES:
        return "video"
    if suffix in video_ops.IMAGE_SUFFIXES:
        return "image"
    if suffix in video_ops.DOCX_SUFFIXES:
        return "docx"
    if suffix in video_ops.TEXT_SUFFIXES:
        return "text"
    if suffix in video_ops.BOOK_SUFFIXES:
        return "text"
    return "other"


def read_text_file(path: str | Path) -> str:
    """Read a text file, trying a sensible chain of encodings for Arabic content."""
    path = Path(path)
    raw = path.read_bytes()
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def _strip_srt(text: str) -> str:
    """Reduce subtitle content to plain dialogue lines."""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "-->" in line or re.fullmatch(r"\d+", line):
            continue
        lines.append(re.sub(r"<[^>]+>", "", line))
    return "\n".join(lines)


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"[ \t]+", " ", text)


def _docx_text(path: Path) -> str:
    try:
        import docx
    except Exception as exc:  # noqa: BLE001
        log.warning("python-docx unavailable (%s)", exc)
        return ""
    document = docx.Document(str(path))
    return "\n\n".join(p.text for p in document.paragraphs if p.text.strip())


def text_from_any(path: str | Path) -> str:
    """Best-effort extraction of readable text from any supported file."""
    path = Path(path)
    kind = detect_kind(path)
    if kind == "docx":
        return _docx_text(path)
    if kind == "text":
        text = read_text_file(path)
        if path.suffix.lower() in {".srt", ".vtt"}:
            return _strip_srt(text)
        if path.suffix.lower() == ".html":
            return _strip_html(text)
        if path.suffix.lower() == ".json":
            try:
                data = json.loads(text)
                return json.dumps(data, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                return text
        return text
    return ""


def split_paragraphs(text: str, min_length: int = 1) -> list[str]:
    """Split raw text into tidy paragraphs, collapsing stray blank lines."""
    paragraphs = []
    for chunk in re.split(r"\n\s*\n", text or ""):
        chunk = re.sub(r"[ \t]+", " ", chunk).replace("\n", " ").strip()
        if len(chunk) >= min_length:
            paragraphs.append(chunk)
    return paragraphs


def split_lines(text: str) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def ingest(path: str | Path) -> dict:
    """One entry point for any dropped file. Returns a normalised descriptor."""
    path = Path(path)
    if not path.exists():
        return {"kind": "missing", "path": str(path), "text": "", "error": "file not found"}

    kind = detect_kind(path)
    descriptor: dict = {"kind": kind, "path": str(path), "text": "", "name": path.stem}

    if kind == "image":
        descriptor.update(video_ops.probe(path))
    elif kind == "video":
        try:
            descriptor.update(video_ops.probe(path))
        except Exception as exc:  # noqa: BLE001
            descriptor["error"] = str(exc)
    elif kind in {"text", "docx"}:
        descriptor["text"] = text_from_any(path)
        descriptor["paragraphs"] = split_paragraphs(descriptor["text"])
        descriptor["words"] = len((descriptor["text"] or "").split())
    return descriptor


__all__ = [
    "detect_kind",
    "read_text_file",
    "text_from_any",
    "split_paragraphs",
    "split_lines",
    "ingest",
]
