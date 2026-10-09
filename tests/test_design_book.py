"""Tests for the image studio and the book pipeline."""

from __future__ import annotations

import os

from hashem.design import render_design, PLATFORM_SIZES
from hashem.book import parse_document, typeset, render_page, export_book


def test_design_kinds_produce_images(tmp_path):
    for kind, spec in {
        "poster": {"title": "عنوان", "palette": "sunset"},
        "logo": {"text": "هـ", "shape": "hexagon"},
        "quote": {"text": "نص", "author": "م"},
        "thumbnail": {"title": "عنوان", "palette": "neon"},
        "banner": {"title": "لافتة"},
        "book_cover": {"title": "كتاب", "author": "م"},
    }.items():
        image = render_design(kind, spec)
        assert image.width > 10 and image.height > 10, kind


def test_social_platform_sizes():
    image = render_design("social", {"title": "تجربة", "platform": "story"})
    assert image.size == PLATFORM_SIZES["story"]


def test_parse_and_typeset_book():
    md = "# كتابي\n\n## فصل\n\nنص فقرة طويلة بعض الشيء لكي تلتف على اكثر من سطر عند التنسيق.\n\n> اقتباس\n\n## فصل ثان\n\nنهاية."
    doc = parse_document(md, author="مؤلف")
    assert doc.title == "كتابي"
    assert doc.word_count > 5
    result = typeset(doc, "classic")
    assert result.page_count >= 2
    page = render_page(result, 0)
    assert page.size == (1240, 1754)


def test_export_book_formats(tmp_path):
    doc = parse_document("# ت\n\n## ف\n\nنص تجريبي بسيط.", author="م")
    result = typeset(doc, "classic")
    outputs = export_book(result, tmp_path, ["pdf", "epub", "docx", "html"])
    assert set(outputs) == {"pdf", "epub", "docx", "html"}
    for path in outputs.values():
        assert os.path.getsize(path) > 200


def test_dark_theme_page_renders():
    doc = parse_document("# داكن\n\nنص على خلفية داكنة.")
    result = typeset(doc, "dark")
    page = render_page(result, 1)
    pixels = list(page.getdata())
    # background should be dark
    assert sum(pixels[0][:3]) < 200
