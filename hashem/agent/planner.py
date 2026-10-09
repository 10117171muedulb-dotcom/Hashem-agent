"""A deterministic, offline intent planner.

When no LLM provider is reachable (no Ollama, no API key, no internet) the agent
still has to be useful.  This module maps a natural-language request — Arabic or
English — onto concrete studio tool calls.  It is deliberately conservative: it
only acts on recognisable design intents and otherwise explains the capabilities.

When a real LLM *is* configured the agent prefers it; this is purely the fallback
that keeps the product functional 100% offline and free.
"""

from __future__ import annotations

import re
from typing import Any

# --------------------------------------------------------------------------- #
# vocabularies
# --------------------------------------------------------------------------- #

_PALETTE_WORDS = {
    "gold": ["ذهبي", "ذهبى", "gold", "فاخر"],
    "sunset": ["غروب", "برتقال", "دافئ", "sunset", "احمر"],
    "ocean": ["محيط", "ازرق", "أزرق", "بحر", "ocean", "سماوي"],
    "emerald": ["زمردي", "اخضر", "أخضر", "emerald", "طبيعي"],
    "royal": ["ملكي", "بنفسج", "royal", "ارجوان"],
    "rose": ["وردي", "وردي", "rose", "زهري"],
    "neon": ["نيون", "neon", "توهج", "سايبر"],
    "midnight": ["ليلي", "داكن", "midnight", "اسود", "أسود"],
    "candy": ["حلوى", "مرح", "candy", "باستيل"],
}

_VIDEO_WORDS = ["فيديو", "موشن", "فيدي", "clip", "video", "مقطع", "ريل", "اعلان متحرك"]
_IMAGE_KINDS = [
    ("logo", ["شعار", "لوغو", "لوجو", "logo"]),
    ("book_cover", ["غلاف", "cover"]),
    ("thumbnail", ["مصغر", "مصغرة", "thumbnail", "يوتيوب", "youtube"]),
    ("quote", ["اقتباس", "حكمه", "حكمة", "quote", "عبارة"]),
    ("banner", ["لافتة", "بانر", "banner", "غلاف فيسبوك"]),
    ("social", ["منشور", "بوست", "سوشيال", "انستقرام", "instagram", "تويتر", "post"]),
    ("poster", ["بوستر", "بوسترات", "poster", "ملصق", "اعلان"]),
]
_BOOK_WORDS = ["كتاب", "كتابا", "book", "روايه", "رواية", "مخطوط"]
_MUSIC_WORDS = ["موسيقى", "موسيقي", "music", "لحن", "اغنيه بدون كلمات"]
_TTS_WORDS = ["انطق", "نطق", "صوت معلق", "تعليق صوتي", "tts", "علق بصوت"]
_IMAGE_EDIT_WORDS = ["فلتر", "ازالة الخلفية", "إزالة الخلفية", "علامة مائية", "watermark", "تحسين الصوره", "حسن الصورة"]


def _contains(text: str, words: list[str]) -> bool:
    return any(w in text for w in words)


def _palette(text: str) -> str:
    for name, words in _PALETTE_WORDS.items():
        if _contains(text, words):
            return name
    return ""


def _topic(text: str) -> str:
    """Strip imperative verbs/filler to get a usable title/topic."""
    cleaned = text
    for token in ["اصنع", "اعمل", "سوي", "صمم", "صمّم", "انشئ", "أنشئ", "اكتب", "ولد", "generate",
                  "create", "make", "design", "لي", "لى", "من فضلك", "please", "video", "فيديو",
                  "بوستر", "poster", "صورة", "صوره", "شعار", "كتاب", "عن"]:
        cleaned = cleaned.replace(token, " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -.،")
    return cleaned or text.strip()


def _points(text: str) -> list[str]:
    """Extract bullet points if the user listed them."""
    lines = [l.strip(" -•*.\t") for l in text.splitlines() if l.strip()]
    if len(lines) >= 2:
        return [l for l in lines[1:] if l][:8]
    if "،" in text:
        parts = [p.strip() for p in text.split("،") if p.strip()]
        if len(parts) >= 2:
            return parts[1:4]
    return []


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #

def plan(text: str) -> list[dict[str, Any]]:
    """Return a list of ``{"tool": name, "arguments": {...}, "say": str}`` steps."""
    lowered = (text or "").lower()
    palette = _palette(lowered)
    steps: list[dict[str, Any]] = []

    if _contains(lowered, _VIDEO_WORDS):
        args = {"topic": _topic(text), "points": _points(text), "template": "intro_points_outro"}
        if palette:
            args["palette"] = palette
        if _contains(lowered, ["عد تنازلي", "countdown"]):
            args["template"] = "countdown"
        elif _contains(lowered, ["اقتباس", "quote"]):
            args["template"] = "quote_reel"
            args["topic"] = _topic(text)
        elif _contains(lowered, ["ترويج", "منتج", "promo", "اعلان"]):
            args["template"] = "promo"
        if _contains(lowered, _TTS_WORDS):
            args["narration"] = _topic(text)
        steps.append({"tool": "create_video", "arguments": args, "say": "🎬 أصنع الفيديو الآن…"})
        return steps

    for kind, words in _IMAGE_KINDS:
        if _contains(lowered, words):
            args = {"kind": kind, "title": _topic(text)}
            if palette:
                args["palette"] = palette
            if kind == "quote":
                args["text"] = _topic(text)
            steps.append({"tool": "create_image", "arguments": args, "say": "🖼️ أصمم الصورة الآن…"})
            return steps

    if _contains(lowered, _IMAGE_EDIT_WORDS):
        steps.append({"tool": "capabilities", "arguments": {}, "say": "لتحرير صورة موجودة استخدم أداة edit_image بمسار الصورة."})
        return steps

    if _contains(lowered, _BOOK_WORDS) and len(text) > 300:
        steps.append({"tool": "create_book", "arguments": {"text": text, "title": _topic(text.splitlines()[0])[:60]}, "say": "📚 أُنسق الكتاب الآن…"})
        return steps

    if _contains(lowered, _MUSIC_WORDS):
        steps.append({"tool": "generate_music", "arguments": {"style": "corporate", "duration": 12}, "say": "🎵 أولد الموسيقى…"})
        return steps

    # default: explain what it can do so the user isn't stranded
    steps.append({"tool": "capabilities", "arguments": {},
                  "say": "أنا أعمل الآن بالمخطط المحلي المجاني (بدون نموذج ذكاء متصل)."})
    return steps


def offline_reply(text: str, results: list[dict[str, Any]]) -> str:
    """Compose a friendly Arabic summary after executing fallback steps."""
    lines = ["✅ تم التنفيذ عبر المخطط المحلي المجاني (لا يحتاج إنترنت ولا مفاتيح):"]
    for result in results:
        if not isinstance(result, dict):
            continue
        path = result.get("path")
        outputs = result.get("outputs")
        if path:
            lines.append(f"• {path}")
        if outputs:
            for fmt, p in outputs.items():
                lines.append(f"• {fmt.upper()}: {p}")
        if result.get("duration"):
            lines.append(f"• المدة: {result['duration']} ثانية")
    if len(lines) == 1:
        lines.append("• لم أتعرف على طلب تصميم محدد. جرّب: «اصنع فيديو عن …» أو «صمم بوستر …» أو «حوّل هذا النص إلى كتاب».")
    lines.append("\nلتفعيل ذكاء كامل (اختياري): ثبّت Ollama محليًا أو ضع مفتاح Groq/Gemini من الإعدادات.")
    return "\n".join(lines)


__all__ = ["plan", "offline_reply"]
