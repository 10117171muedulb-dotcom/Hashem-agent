"""Tool registry: every studio capability exposed to the LLM as a function.

Handlers always return JSON-serialisable dicts and never raise — an exception in
one tool becomes an error payload the model can react to, keeping the loop alive.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .. import media, motion
from ..design import render_design
from ..book import export_book, parse_document, parse_file, typeset
from ..motion import templates
from ..motion.pipeline import render_project
from ..utils.log import get_logger
from ..utils.safety import PathViolation, looks_destructive

log = get_logger("agent.tools")


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    category: str = "studio"

    def schema(self) -> dict[str, Any]:
        properties = {}
        required = []
        for pname, pspec in self.parameters.items():
            properties[pname] = {
                "type": pspec.get("type", "string"),
                "description": pspec.get("description", ""),
            }
            if pspec.get("items"):
                properties[pname]["items"] = pspec["items"]
            if pspec.get("required"):
                required.append(pname)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        }


def _err(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}


def _ok(**kwargs: Any) -> dict[str, Any]:
    payload = {"ok": True}
    payload.update(kwargs)
    return payload


# =========================================================================== #
# video
# =========================================================================== #

def t_create_video(ctx, project: Any = None, topic: str = "", points: Any = None,
                   template: str = "intro_points_outro", palette: str = "",
                   narration: str = "", music_style: str = "", format: str = "",
                   **_: Any) -> dict[str, Any]:
    points = points if isinstance(points, list) else ([p for p in str(points or "").split("|") if p.strip()])
    raw = project
    if raw is None:
        profile = ctx.skills.style_profile()
        chosen_palette = palette or profile.get("favourite_palette") or ""
        raw = templates.build(
            template, topic=topic or narration or "فيديو جديد", points=points,
            palette=chosen_palette, music_style=music_style or ctx.settings.video.music_style,
            fps=ctx.settings.video.fps,
        )
    if isinstance(raw, str):
        import json as _json

        raw = _json.loads(raw)
    if narration:
        raw.setdefault("audio", {}).setdefault("narration", {})["enabled"] = True
        raw["audio"]["narration"]["text"] = narration

    name = str(raw.get("name") or topic or "video")
    settings = ctx.settings.video
    if format:
        settings.format = format
    result = render_project(raw, ctx.output_path(name, ".mp4"), settings)
    ctx.skills.remember("video", topic or name, params={"palette": (raw.get("meta") or {}).get("palette", palette),
                                                       "template": template, "style": music_style},
                        metrics=result["metrics"])
    result["skill_id"] = ctx.skills.skills[-1].id
    return result


def t_render_preview(ctx, project: Any = None, **_: Any) -> dict[str, Any]:
    if project is None:
        return _err("project is required")
    from ..motion.scene import parse_project
    from ..motion.render import Renderer

    renderer = Renderer(parse_project(project))
    preview = renderer.preview(scale=0.4)
    path = ctx.cache / "preview.png"
    preview.save(path)
    return _ok(path=str(path), size=list(preview.size))


def t_edit_video(ctx, path: str = "", action: str = "probe", output: str = "",
                 start: float = 0.0, duration: float = 0.0, factor: float = 2.0,
                 fps: int = 12, width: int = 480, **_: Any) -> dict[str, Any]:
    try:
        source = ctx.guard.resolve(path, must_exist=True)
    except PathViolation as exc:
        return _err(str(exc))
    action = (action or "probe").lower()
    if action == "probe":
        return _ok(**media.probe(source))
    out = ctx.output_path(Path(source).stem + "_" + action, Path(source).suffix) if not output else ctx.guard.resolve(output, create_parents=True)
    if action == "trim":
        return _ok(path=str(media.trim(source, out, start, duration or None)))
    if action == "thumbnail":
        return _ok(path=str(media.thumbnail(source, ctx.output_path(Path(source).stem + "_thumb", ".jpg"), start)))
    if action == "gif":
        return _ok(path=str(media.video_to_gif(source, ctx.output_path(Path(source).stem, ".gif"), fps, width)))
    if action == "speed":
        return _ok(path=str(media.speed_change(source, out, factor)))
    if action == "frames":
        frames = media.extract_frames(source, ctx.cache / "frames", every=duration or 1.0)
        return _ok(frames=[str(f) for f in frames], count=len(frames))
    return _err(f"unknown action {action}")


def t_video_from_images(ctx, images: Any = None, per_image: float = 1.5, fps: int = 30, **_: Any) -> dict[str, Any]:
    images = images or []
    resolved = []
    for item in images:
        try:
            resolved.append(ctx.guard.resolve(item, must_exist=True))
        except PathViolation as exc:
            log.warning("skip %s: %s", item, exc)
    if not resolved:
        return _err("no valid images")
    out = ctx.output_path("slideshow", ".mp4")
    path = motion.encoder.images_to_video(resolved, out, fps=fps, per_image=per_image)
    return _ok(path=str(path), count=len(resolved))


# =========================================================================== #
# image design
# =========================================================================== #

def t_create_image(ctx, kind: str = "poster", title: str = "", subtitle: str = "",
                   text: str = "", author: str = "", palette: str = "", style: str = "",
                   platform: str = "", image: str = "", shape: str = "", **spec: Any) -> dict[str, Any]:
    full = {
        "title": title or text, "subtitle": subtitle, "text": text, "author": author,
        "palette": palette or ctx.skills.style_profile().get("favourite_palette") or "",
        "style": style, "platform": platform, "shape": shape,
    }
    if image:
        try:
            full["image"] = str(ctx.guard.resolve(image, must_exist=True))
        except PathViolation as exc:
            return _err(str(exc))
    full.update({k: v for k, v in spec.items() if v not in (None, "")})
    image_obj = render_design(kind, full)
    suffix = ".png" if image_obj.mode == "RGBA" else ".png"
    path = ctx.output_path(title or text or kind, suffix)
    image_obj.save(path)
    ctx.skills.remember("image", title or text or kind, params=full, metrics={"quality": 0.7})
    return _ok(path=str(path), size=list(image_obj.size), mode=image_obj.mode)


def t_edit_image(ctx, path: str = "", filter: str = "", width: int = 0, height: int = 0,
                 watermark: str = "", remove_bg: bool = False, qr: str = "",
                 border: int = 0, rounded: int = 0, enhance: bool = False, **_: Any) -> dict[str, Any]:
    try:
        source = ctx.guard.resolve(path, must_exist=True)
    except PathViolation as exc:
        return _err(str(exc))
    image = media.load_image(source)
    if enhance:
        image = media.auto_enhance(image)
    if filter:
        image = media.apply_filter(image, filter)
    if width or height:
        image = media.resize(image, width or None, height or None)
    if remove_bg:
        image = media.remove_background(image)
    if rounded:
        image = media.rounded_corners(image, rounded)
    if border:
        image = media.add_border(image, width=border)
    if watermark:
        image = media.overlay_watermark(image, watermark)
    if qr:
        image = media.embed_qr(image, qr)
    out = ctx.output_path(Path(source).stem + "_edited", ".png")
    media.save_image(image, out)
    return _ok(path=str(out), size=list(image.size))


# =========================================================================== #
# book
# =========================================================================== #

def t_create_book(ctx, text: str = "", file: str = "", title: str = "", author: str = "",
                  theme: str = "classic", formats: Any = None, **_: Any) -> dict[str, Any]:
    if file:
        try:
            doc = parse_file(ctx.guard.resolve(file, must_exist=True), author=author)
        except PathViolation as exc:
            return _err(str(exc))
    elif text:
        doc = parse_document(text, title=title or None, author=author)
    else:
        return _err("either text or file is required")
    if title:
        doc.title = title
    result = typeset(doc, theme)
    formats = formats if isinstance(formats, list) else ["pdf", "html"]
    outputs = export_book(result, ctx.exports, formats)
    ctx.skills.remember("book", doc.title, params={"theme": theme}, metrics={"pages": result.page_count})
    return _ok(pages=result.page_count, outputs={k: str(v) for k, v in outputs.items()}, stats=doc.stats())


# =========================================================================== #
# audio
# =========================================================================== #

def t_generate_music(ctx, style: str = "corporate", duration: float = 12.0, volume: float = 0.6, **_: Any) -> dict[str, Any]:
    from ..audio import render_music, save_wav

    samples = render_music(duration, style=style, volume=volume)
    path = ctx.output_path(f"music_{style}", ".wav")
    save_wav(path, samples)
    return _ok(path=str(path), duration=round(duration, 2), style=style)


def t_text_to_speech(ctx, text: str = "", voice: str = "", rate: int = 0, **_: Any) -> dict[str, Any]:
    from ..audio import synthesize_speech

    if not text:
        return _err("text is required")
    path = ctx.output_path("speech", ".mp3")
    result = synthesize_speech(text, path, voice=voice, rate=rate)
    if not result:
        return _err("no TTS engine available (needs internet for Edge TTS or Windows SAPI)")
    return _ok(path=str(result))


def t_list_voices(ctx, language: str = "ar", **_: Any) -> dict[str, Any]:
    from ..audio import list_voices

    voices = list_voices(language)
    return _ok(count=len(voices), voices=[{"name": v.get("ShortName"), "gender": v.get("Gender")} for v in voices[:30]])


# =========================================================================== #
# files / system
# =========================================================================== #

def t_read_file(ctx, path: str = "", limit: int = 200, **_: Any) -> dict[str, Any]:
    try:
        target = ctx.guard.resolve(path, must_exist=True)
    except PathViolation as exc:
        return _err(str(exc))
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return _err(str(exc))
    return _ok(path=str(target), total_lines=len(lines), content="\n".join(lines[: max(1, int(limit))]))


def t_write_file(ctx, path: str = "", content: str = "", **_: Any) -> dict[str, Any]:
    try:
        target = ctx.guard.resolve(path, create_parents=True)
    except PathViolation as exc:
        return _err(str(exc))
    target.write_text(content, encoding="utf-8")
    return _ok(path=str(target), bytes=len(content.encode("utf-8")))


def t_list_dir(ctx, path: str = ".", **_: Any) -> dict[str, Any]:
    try:
        target = ctx.guard.resolve(path)
    except PathViolation as exc:
        return _err(str(exc))
    entries = []
    for item in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))[:200]:
        entries.append({"name": item.name, "type": "dir" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else 0})
    return _ok(path=str(target), entries=entries)


def t_run_command(ctx, command: str = "", timeout: int = 120, **_: Any) -> dict[str, Any]:
    if not ctx.settings.agent.allow_shell:
        return _err("shell access is disabled in settings")
    risky, reason = looks_destructive(command)
    if risky and not ctx.settings.agent.auto_approve:
        return _err(f"refused (destructive: {reason}). Enable auto_approve to allow.")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True,
                                timeout=max(1, int(timeout)), cwd=str(ctx.workspace))
    except subprocess.TimeoutExpired:
        return _err("command timed out")
    return _ok(exit_code=result.returncode, stdout=result.stdout[-4000:], stderr=result.stderr[-2000:])


def t_system_info(ctx, **_: Any) -> dict[str, Any]:
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "ffmpeg": motion.encoder.ffmpeg_available(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import shutil as _sh

        info["disk_free_gb"] = round(_sh.disk_usage(str(ctx.workspace)).free / 1e9, 1)
    except OSError:
        pass
    return _ok(**info)


def t_open_path(ctx, path: str = "", **_: Any) -> dict[str, Any]:
    try:
        target = ctx.guard.resolve(path, must_exist=True)
    except PathViolation as exc:
        return _err(str(exc))
    try:
        if platform.system() == "Windows":
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
        return _ok(opened=str(target))
    except OSError as exc:
        return _err(str(exc))


# =========================================================================== #
# web
# =========================================================================== #

def t_web_search(ctx, query: str = "", max_results: int = 5, **_: Any) -> dict[str, Any]:
    if not ctx.settings.agent.allow_web:
        return _err("web access is disabled in settings")
    try:
        import requests

        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        response.raise_for_status()
        import re as _re

        results = []
        for match in _re.finditer(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', response.text):
            url, title = match.groups()
            title = _re.sub(r"<[^>]+>", "", title)
            results.append({"title": title.strip(), "url": url})
            if len(results) >= max_results:
                break
        return _ok(results=results)
    except Exception as exc:  # noqa: BLE001
        return _err(f"search failed: {exc}")


def t_fetch_url(ctx, url: str = "", max_chars: int = 4000, **_: Any) -> dict[str, Any]:
    if not ctx.settings.agent.allow_web:
        return _err("web access is disabled")
    try:
        import re as _re

        import requests

        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        response.raise_for_status()
        text = _re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", response.text)
        text = _re.sub(r"(?s)<[^>]+>", " ", text)
        text = _re.sub(r"\s+", " ", text).strip()
        return _ok(url=url, text=text[: max(200, int(max_chars))])
    except Exception as exc:  # noqa: BLE001
        return _err(f"fetch failed: {exc}")


# =========================================================================== #
# learning / info
# =========================================================================== #

def t_capabilities(ctx, **_: Any) -> dict[str, Any]:
    from ..design import PLATFORM_SIZES
    from ..motion.colors import palette_names
    from ..motion.scene import PRESETS
    from ..book.typesetter import theme_names
    from ..book.exporters import book_formats
    from ..audio import music_styles

    return _ok(
        video_templates=templates.template_names(),
        video_presets=list(PRESETS),
        design_kinds=["poster", "social", "thumbnail", "quote", "logo", "banner", "book_cover"],
        platforms=list(PLATFORM_SIZES),
        palettes=palette_names(),
        book_themes=theme_names(),
        book_formats=book_formats(),
        music_styles=music_styles(),
        transitions=motion.render.TRANSITION_KINDS,
    )


def t_learning_stats(ctx, **_: Any) -> dict[str, Any]:
    return _ok(**ctx.skills.style_profile(), recent=[{"prompt": s.prompt[:60], "kind": s.kind, "rating": s.rating} for s in ctx.skills.skills[-8:]])


def t_rate_task(ctx, skill_id: str = "", rating: float = 1.0, **_: Any) -> dict[str, Any]:
    skill = ctx.skills.rate(skill_id, rating)
    if not skill:
        return _err("skill not found")
    return _ok(rated=skill_id, rating=skill.rating)


_HANDLERS: list[Tool] = [
    Tool("create_video", "اصنع فيديو موشن جرافيك كامل. مرر topic + points (قائمة) أو project JSON كامل. يعيد مسار ملف MP4.", {
        "topic": {"type": "string", "description": "عنوان/موضوع الفيديو"},
        "points": {"type": "array", "items": {"type": "string"}, "description": "نقاط/شرائح الفيديو"},
        "project": {"type": "string", "description": "JSON كامل لمشروع المشهد (اختياري)"},
        "template": {"type": "string", "description": "intro_points_outro | promo | quote_reel | countdown"},
        "palette": {"type": "string", "description": "اسم لوحة ألوان"},
        "narration": {"type": "string", "description": "نص تعليق صوتي عربي (اختياري)"},
        "music_style": {"type": "string", "description": "نمط الموسيقى"},
    }, t_create_video),
    Tool("render_preview", "اعرض معاينة صورة واحدة لمشروع فيديو قبل التصدير.", {
        "project": {"type": "string", "description": "JSON المشروع"},
    }, t_render_preview),
    Tool("edit_video", "عمليات على ملف فيديو: probe/trim/thumbnail/gif/speed/frames.", {
        "path": {"type": "string", "description": "مسار الفيديو"},
        "action": {"type": "string", "description": "probe|trim|thumbnail|gif|speed|frames"},
        "start": {"type": "number", "description": "ثانية البداية"},
        "duration": {"type": "number", "description": "المدة"},
        "factor": {"type": "number", "description": "معامل السرعة"},
    }, t_edit_video),
    Tool("video_from_images", "اصنع فيديو سلايدشو من قائمة صور.", {
        "images": {"type": "array", "items": {"type": "string"}, "description": "مسارات الصور"},
        "per_image": {"type": "number", "description": "ثوانٍ لكل صورة"},
    }, t_video_from_images),
    Tool("create_image", "صمم صورة: poster/social/thumbnail/quote/logo/banner/book_cover.", {
        "kind": {"type": "string", "description": "نوع التصميم"},
        "title": {"type": "string", "description": "العنوان"},
        "subtitle": {"type": "string", "description": "نص فرعي"},
        "author": {"type": "string", "description": "للاغلفة/الاقتباسات"},
        "palette": {"type": "string", "description": "لوحة ألوان"},
        "style": {"type": "string", "description": "نمط الخلفية"},
        "platform": {"type": "string", "description": "instagram/story/twitter/..."},
        "image": {"type": "string", "description": "صورة خلفية اختيارية"},
    }, t_create_image),
    Tool("edit_image", "حرر صورة: فلاتر/تحجيم/إزالة خلفية/علامة مائية/QR/حواف.", {
        "path": {"type": "string", "description": "مسار الصورة"},
        "filter": {"type": "string", "description": "sepia/noir/warm/cool/pop/..."},
        "width": {"type": "integer"}, "height": {"type": "integer"},
        "watermark": {"type": "string"}, "remove_bg": {"type": "boolean"},
        "qr": {"type": "string"}, "border": {"type": "integer"}, "rounded": {"type": "integer"},
        "enhance": {"type": "boolean"},
    }, t_edit_image),
    Tool("create_book", "حوّل نص/ملف إلى كتاب منسق وصدّره PDF/EPUB/DOCX/HTML.", {
        "text": {"type": "string", "description": "نص الكتاب (markdown)"},
        "file": {"type": "string", "description": "أو مسار ملف txt/md/docx"},
        "title": {"type": "string"}, "author": {"type": "string"},
        "theme": {"type": "string", "description": "classic/modern/dark/sepia/minimal"},
        "formats": {"type": "array", "items": {"type": "string"}, "description": "pdf/epub/docx/html"},
    }, t_create_book),
    Tool("generate_music", "ولّد موسيقى خلفية خالية من الحقوق.", {
        "style": {"type": "string", "description": "corporate/upbeat/cinematic/lofi/tech/ambient/epic/minimal"},
        "duration": {"type": "number", "description": "المدة بالثواني"},
    }, t_generate_music),
    Tool("text_to_speech", "حوّل نصًا إلى كلام عربي بصوت عصبي مجاني.", {
        "text": {"type": "string", "required": True}, "voice": {"type": "string"}, "rate": {"type": "integer"},
    }, t_text_to_speech),
    Tool("list_voices", "اعرض أصوات النطق المتاحة للغة.", {"language": {"type": "string"}}, t_list_voices),
    Tool("read_file", "اقرأ ملفًا من مساحة العمل.", {"path": {"type": "string", "required": True}, "limit": {"type": "integer"}}, t_read_file),
    Tool("write_file", "اكتب ملفًا في مساحة العمل.", {"path": {"type": "string", "required": True}, "content": {"type": "string", "required": True}}, t_write_file),
    Tool("list_dir", "اعرض محتويات مجلد.", {"path": {"type": "string"}}, t_list_dir),
    Tool("run_command", "نفّذ أمر shell في مساحة العمل (مع حماية).", {"command": {"type": "string", "required": True}, "timeout": {"type": "integer"}}, t_run_command),
    Tool("system_info", "معلومات النظام وffmpeg.", {}, t_system_info),
    Tool("open_path", "افتح ملفًا/مجلدًا في المستعرض الافتراضي.", {"path": {"type": "string", "required": True}}, t_open_path),
    Tool("web_search", "ابحث في الويب (DuckDuckGo).", {"query": {"type": "string", "required": True}, "max_results": {"type": "integer"}}, t_web_search),
    Tool("fetch_url", "اجلب نص صفحة ويب.", {"url": {"type": "string", "required": True}}, t_fetch_url),
    Tool("capabilities", "اعرض كل القوالب والأنماط والألوان والصيغ المتاحة.", {}, t_capabilities),
    Tool("learning_stats", "اعرض ما تعلمه الوكيل (ملف الأسلوب والتقييمات).", {}, t_learning_stats),
    Tool("rate_task", "قيّم مهمة سابقة ليتعلم منها الوكيل.", {"skill_id": {"type": "string", "required": True}, "rating": {"type": "number"}}, t_rate_task),
]


def all_tools() -> list[Tool]:
    return _HANDLERS


def tool_schemas() -> list[dict[str, Any]]:
    return [tool.schema() for tool in _HANDLERS]


def get_tool(name: str) -> Tool | None:
    for tool in _HANDLERS:
        if tool.name == name:
            return tool
    return None


def execute(ctx, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    tool = get_tool(name)
    if not tool:
        return _err(f"unknown tool: {name}")
    try:
        result = tool.handler(ctx, **arguments)
        return result if isinstance(result, dict) else _ok(result=result)
    except Exception as exc:  # noqa: BLE001 - keep the loop alive
        log.exception("tool %s raised", name)
        return _err(f"{type(exc).__name__}: {exc}")


__all__ = ["Tool", "all_tools", "tool_schemas", "get_tool", "execute"]
