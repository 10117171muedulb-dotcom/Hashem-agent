"""Parse raw text / markdown / docx into a structured book document."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils.log import get_logger

log = get_logger("book.document")

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_RE = re.compile(r"^(\s*)([-*+]|\d+\.)\s+(.*)$")
_HR_RE = re.compile(r"^[-*_]{3,}\s*$")
_QUOTE_RE = re.compile(r"^>\s?(.*)$")
_CODE_RE = re.compile(r"^(```|~~~)")


@dataclass
class Block:
    kind: str            # title | heading | subheading | paragraph | quote | list | code | hr
    text: str = ""
    level: int = 0
    lang: str = ""

    @property
    def is_heading(self) -> bool:
        return self.kind in {"title", "heading", "subheading"}


@dataclass
class BookDocument:
    title: str = "كتاب بلا عنوان"
    author: str = ""
    blocks: list[Block] = field(default_factory=list)
    language: str = "ar"
    source: str = ""
    word_count: int = 0

    @property
    def headings(self) -> list[Block]:
        return [b for b in self.blocks if b.is_heading]

    def stats(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "author": self.author,
            "blocks": len(self.blocks),
            "headings": len(self.headings),
            "words": self.word_count,
            "language": self.language,
        }


def _contains_arabic(text: str) -> bool:
    return any("\u0600" <= ch <= "\u06FF" for ch in text or "")


def _clean(line: str) -> str:
    line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
    line = re.sub(r"__([^_]+)__", r"\1", line)
    line = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", line)
    line = re.sub(r"`([^`]+)`", r"\1", line)
    return line.strip()


def parse_document(
    text: str,
    title: str | None = None,
    author: str | None = None,
    source: str = "",
) -> BookDocument:
    """Build a :class:`BookDocument` from markdown-ish text."""
    doc = BookDocument(source=source)
    blocks: list[Block] = []
    buffer: list[str] = []
    in_code = False
    code_lines: list[str] = []
    first_heading_used = False

    def flush() -> None:
        if buffer:
            blocks.append(Block(kind="paragraph", text=" ".join(buffer).strip()))
            buffer.clear()

    for raw in (text or "").splitlines():
        line = raw.rstrip()
        if not line.strip():
            flush()
            continue

        if _CODE_RE.match(line.strip()):
            if in_code:
                blocks.append(Block(kind="code", text="\n".join(code_lines)))
                code_lines = []
                in_code = False
            else:
                flush()
                in_code = True
            continue
        if in_code:
            code_lines.append(raw)
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            content = _clean(heading.group(2))
            if not first_heading_used and level == 1 and not title:
                doc.title = content
                blocks.append(Block(kind="title", text=content, level=1))
                first_heading_used = True
            else:
                blocks.append(Block(kind="heading" if level <= 2 else "subheading", text=content, level=level))
            continue

        if _HR_RE.match(line):
            flush()
            blocks.append(Block(kind="hr"))
            continue

        quote = _QUOTE_RE.match(line)
        if quote:
            flush()
            blocks.append(Block(kind="quote", text=_clean(quote.group(1))))
            continue

        listed = _LIST_RE.match(line)
        if listed:
            flush()
            blocks.append(Block(kind="list", text=_clean(listed.group(3))))
            continue

        buffer.append(_clean(line))

    flush()
    if in_code and code_lines:
        blocks.append(Block(kind="code", text="\n".join(code_lines)))

    if title:
        doc.title = title
    elif not first_heading_used and blocks:
        first_text = next((b for b in blocks if b.kind == "paragraph"), None)
        if first_text and len(first_text.text) < 80:
            doc.title = first_text.text
    doc.author = author or ""

    doc.blocks = blocks
    doc.word_count = sum(len(b.text.split()) for b in blocks if b.text)
    doc.language = "ar" if _contains_arabic(doc.title + " " + " ".join(b.text for b in blocks[:20])) else "en"
    if not any(b.kind == "title" for b in blocks):
        blocks.insert(0, Block(kind="title", text=doc.title, level=1))
    return doc


def parse_file(path: str | Path, author: str | None = None) -> BookDocument:
    """Read any supported file and parse it."""
    from ..media.ingest import text_from_any

    path = Path(path)
    text = text_from_any(path)
    return parse_document(text, title=path.stem if path.stem else None, author=author, source=str(path))


__all__ = ["Block", "BookDocument", "parse_document", "parse_file"]
