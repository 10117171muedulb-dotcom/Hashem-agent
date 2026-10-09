"""Typesetting + pagination + page rendering for books.

Pages are rendered as raster images by the same Arabic-capable text engine the
video studio uses, so a PDF produced here shows flawless Arabic, headings, quotes,
code, page numbers, running heads and a generated table of contents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image, ImageDraw

from ..motion.colors import parse_color
from ..motion.fonts import resolve_font, resolve_mono, weight_to_synthetic
from ..motion.text import measure, render_text, wrap_text
from ..utils.log import get_logger
from .document import BookDocument

log = get_logger("book.typesetter")


@dataclass
class Theme:
    name: str = "classic"
    page_w: int = 1240
    page_h: int = 1754
    margin: int = 110
    paper: str = "#fdfcf8"
    ink: str = "#1c1917"
    accent: str = "#b45309"
    heading_color: str = "#0c0a09"
    body_size: int = 30
    line_spacing: float = 1.6
    body_font: str = ""
    heading_font: str = ""
    rtl: bool = True
    page_numbers: bool = True
    running_head: bool = True


THEMES: dict[str, Theme] = {
    "classic": Theme(),
    "modern": Theme(name="modern", paper="#ffffff", ink="#111827", accent="#4f46e5",
                    heading_color="#111827", body_size=30),
    "dark": Theme(name="dark", paper="#0f172a", ink="#e2e8f0", accent="#38bdf8",
                  heading_color="#f8fafc", body_size=30),
    "sepia": Theme(name="sepia", paper="#f4ecd8", ink="#5b4636", accent="#8b5e34",
                   heading_color="#4a3728", body_size=30),
    "minimal": Theme(name="minimal", paper="#ffffff", ink="#374151", accent="#9ca3af",
                     heading_color="#111827", body_size=28, margin=140),
}


def theme_names() -> list[str]:
    return sorted(THEMES)


@dataclass
class PageItem:
    kind: str
    lines: list[str] = field(default_factory=list)
    size: int = 30
    color: str = "#000000"
    bold: bool = False
    align: str = "right"
    spacing_after: int = 26
    indent: int = 0
    mono: bool = False
    accent_bar: bool = False


@dataclass
class Page:
    number: int
    items: list[PageItem] = field(default_factory=list)
    kind: str = "body"          # cover | toc | body
    heading: str = ""


@dataclass
class TypesetResult:
    doc: BookDocument
    theme: Theme
    pages: list[Page] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)


def _item_height(item: PageItem, theme: Theme) -> int:
    height = 0
    for _ in item.lines:
        height += int(item.size * theme.line_spacing)
    return height + item.spacing_after


def _wrap_block(text: str, font_path: str, size: int, max_width: int, theme: Theme) -> list[str]:
    return wrap_text(text, font_path, size, max_width)


def _layout_block(block, theme: Theme, max_width: int) -> list[PageItem]:
    items: list[PageItem] = []
    rtl_align = "right" if theme.rtl else "left"

    if block.kind == "title":
        font = resolve_font(block.text, family=theme.heading_font or None, bold=True)
        items.append(PageItem(kind="title", lines=[block.text], size=int(theme.body_size * 2.1),
                              color=theme.heading_color, bold=True, align="center", spacing_after=20))
        return items
    if block.kind == "heading":
        items.append(PageItem(kind="heading", lines=[block.text], size=int(theme.body_size * 1.5),
                              color=theme.heading_color, bold=True, align=rtl_align,
                              spacing_after=34, accent_bar=True))
        return items
    if block.kind == "subheading":
        items.append(PageItem(kind="subheading", lines=[block.text], size=int(theme.body_size * 1.22),
                              color=theme.accent, bold=True, align=rtl_align, spacing_after=22))
        return items
    if block.kind == "quote":
        font = resolve_font(block.text, family=theme.body_font or None, bold=False)
        lines = _wrap_block(block.text, font, int(theme.body_size * 0.96), max_width - 80, theme)
        items.append(PageItem(kind="quote", lines=lines, size=int(theme.body_size * 0.96),
                              color=theme.accent, align=rtl_align, spacing_after=26, indent=60))
        return items
    if block.kind == "list":
        font = resolve_font(block.text, family=theme.body_font or None, bold=False)
        prefixed = f"•  {block.text}"
        lines = _wrap_block(prefixed, font, theme.body_size, max_width - 60, theme)
        items.append(PageItem(kind="list", lines=lines, size=theme.body_size, color=theme.ink,
                              align=rtl_align, spacing_after=14, indent=40))
        return items
    if block.kind == "code":
        font = resolve_mono()
        lines = []
        for code_line in block.text.split("\n"):
            lines.extend(_wrap_block(code_line or " ", font, int(theme.body_size * 0.8), max_width - 80, theme))
        items.append(PageItem(kind="code", lines=lines, size=int(theme.body_size * 0.8),
                              color=theme.ink, align="left", spacing_after=24, indent=40, mono=True))
        return items
    if block.kind == "hr":
        items.append(PageItem(kind="hr", lines=[""], size=0, color=theme.accent, spacing_after=30))
        return items
    if block.kind == "paragraph":
        font = resolve_font(block.text, family=theme.body_font or None, bold=False)
        lines = _wrap_block(block.text, font, theme.body_size, max_width, theme)
        items.append(PageItem(kind="paragraph", lines=lines, size=theme.body_size, color=theme.ink,
                              align=rtl_align, spacing_after=26))
        return items
    return items


def typeset(doc: BookDocument, theme: Theme | str = "classic") -> TypesetResult:
    """Lay out the whole document into pages (cover + TOC + body)."""
    theme = theme if isinstance(theme, Theme) else THEMES.get(theme, THEMES["classic"])
    result = TypesetResult(doc=doc, theme=theme)
    max_width = theme.page_w - 2 * theme.margin
    content_top = theme.margin + 60
    content_bottom = theme.page_h - theme.margin - 60
    usable = content_bottom - content_top

    # -------- body pagination
    body_pages: list[Page] = []
    current = Page(number=0, kind="body")
    used = 0

    def new_page() -> None:
        nonlocal current, used
        if current.items:
            body_pages.append(current)
        current = Page(number=0, kind="body")
        used = 0

    for block in doc.blocks:
        if block.kind == "title":
            continue
        items = _layout_block(block, theme, max_width)
        for item in items:
            height = _item_height(item, theme)
            if used + height > usable and current.items:
                new_page()
            # keep headings from being stranded at the very bottom
            if item.kind in {"heading", "subheading"} and used + height > usable - int(theme.body_size * 2.4):
                new_page()
            current.items.append(item)
            used += height
    if current.items:
        body_pages.append(current)

    # -------- table of contents
    toc_pages: list[Page] = []
    heading_pages: list[tuple[str, int, int]] = []
    toc_offset = 2  # cover + at least one toc page placeholder
    for page_index, page in enumerate(body_pages):
        first_heading = next((i for i in page.items if i.kind in {"heading", "title"}), None)
        for item in page.items:
            if item.kind == "heading" and item.lines:
                heading_pages.append((item.lines[0], page_index + toc_offset, 0))

    # build toc items
    toc_items: list[PageItem] = [PageItem(kind="heading", lines=["المحتويات" if doc.language == "ar" else "Contents"],
                                          size=int(theme.body_size * 1.5), color=theme.heading_color, bold=True, align="center", spacing_after=40)]
    for name, page_no, _ in heading_pages:
        toc_items.append(PageItem(kind="toc_line", lines=[f"{name}  ....  {page_no}"], size=int(theme.body_size * 0.95),
                                  color=theme.ink, align="right", spacing_after=16))

    # paginate toc
    used = 0
    toc = Page(number=1, kind="toc")
    for item in toc_items:
        height = _item_height(item, theme)
        if used + height > usable and toc.items:
            toc_pages.append(toc)
            toc = Page(number=len(toc_pages) + 1, kind="toc")
            used = 0
        toc.items.append(item)
        used += height
    if toc.items:
        toc_pages.append(toc)

    final_toc_count = max(1, len(toc_pages))

    # -------- assign final numbers
    cover = Page(number=0, kind="cover")
    result.pages.append(cover)
    for page in toc_pages:
        page.number = len(result.pages)
        result.pages.append(page)
    # pad toc to final_toc_count (already equal)
    for page_index, page in enumerate(body_pages):
        page.number = 1 + final_toc_count + page_index
        page.heading = next((i.lines[0] for i in page.items if i.kind in {"heading", "title"}), "")
        result.pages.append(page)

    return result


def render_page(result: TypesetResult, index: int) -> Image.Image:
    """Rasterise a single page of a typeset result."""
    theme = result.theme
    doc = result.doc
    page = result.pages[index]
    paper = parse_color(theme.paper)[:3]
    image = Image.new("RGB", (theme.page_w, theme.page_h), paper)
    draw = ImageDraw.Draw(image)
    max_width = theme.page_w - 2 * theme.margin

    if page.kind == "cover":
        return _render_cover(result, image)

    y = theme.margin + 60

    # running head + page number
    if theme.running_head and page.kind == "body" and doc.title:
        head = render_text(doc.title, font_path=resolve_font(doc.title), size=int(theme.body_size * 0.7),
                           fill=theme.accent)
        image.paste(head, (theme.margin, theme.margin // 2), head)
    if theme.page_numbers:
        number_text = str(page.number)
        num = render_text(number_text, font_path=resolve_mono(), size=int(theme.body_size * 0.8), fill=theme.accent)
        image.paste(num, (theme.page_w // 2 - num.width // 2, theme.page_h - theme.margin // 2 - num.height), num)

    for item in page.items:
        if item.kind == "hr":
            draw.line([theme.margin, y + 8, theme.page_w - theme.margin, y + 8], fill=parse_color(theme.accent)[:3], width=3)
            y += item.spacing_after + 16
            continue

        font_family = theme.heading_font if item.kind in {"title", "heading", "subheading"} else theme.body_font
        if item.mono:
            font_path = resolve_mono()
        else:
            font_path = resolve_font(" ".join(item.lines) or "text", family=font_family or None, bold=item.bold)

        if item.accent_bar:
            bar_h = int(item.size * 1.2)
            draw.rectangle([theme.page_w - theme.margin - 6, y, theme.page_w - theme.margin, y + bar_h],
                           fill=parse_color(theme.accent)[:3])

        line_h = int(item.size * theme.line_spacing)
        for line in item.lines:
            tile = render_text(
                line,
                font_path=font_path,
                size=item.size,
                fill=item.color,
                align="left",
                max_width=max_width,
            )
            if item.align == "center":
                x = (theme.page_w - tile.width) // 2
            elif item.align == "right":
                x = theme.page_w - theme.margin - tile.width
            else:
                x = theme.margin + item.indent
            if tile.mode == "RGBA":
                image.paste(tile, (max(0, x), y), tile)
            else:
                image.paste(tile, (max(0, x), y))
            y += line_h
        y += item.spacing_after

    return image


def _render_cover(result: TypesetResult, image: Image.Image) -> Image.Image:
    theme = result.theme
    doc = result.doc
    draw = ImageDraw.Draw(image)
    w, h = image.size
    accent = parse_color(theme.accent)[:3]

    # elegant double frame
    inset = int(w * 0.07)
    draw.rectangle([inset, inset, w - inset, h - inset], outline=accent, width=5)
    draw.rectangle([inset + 16, inset + 16, w - inset - 16, h - inset - 16], outline=accent, width=2)

    # ornament
    draw.ellipse([w // 2 - 40, int(h * 0.18) - 40, w // 2 + 40, int(h * 0.18) + 40], outline=accent, width=4)

    title_tile = render_text(doc.title, font_path=resolve_font(doc.title, bold=True),
                             size=int(theme.body_size * 2.0), fill=theme.heading_color, align="center",
                             max_width=int(w * 0.7))
    image.paste(title_tile, ((w - title_tile.width) // 2, int(h * 0.38)), title_tile)

    if doc.author:
        author_tile = render_text(doc.author, font_path=resolve_font(doc.author),
                                  size=int(theme.body_size * 1.0), fill=theme.accent)
        image.paste(author_tile, ((w - author_tile.width) // 2, int(h * 0.6)), author_tile)

    return image


__all__ = ["Theme", "THEMES", "theme_names", "Page", "PageItem", "TypesetResult", "typeset", "render_page"]
