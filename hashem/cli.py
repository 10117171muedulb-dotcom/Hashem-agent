"""Command line entry point.

The Windows .exe is built *windowed* (no console), so every user-facing write goes
through :func:`emit`, which falls back to the log file when stdout is ``None``.
``--version`` and ``doctor`` accept ``--out FILE`` so the CI smoke test can read a
deterministic artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .utils.log import get_logger, setup_logging

log = get_logger("cli")


def emit(text: str) -> None:
    if sys.stdout is not None:
        try:
            print(text)
            return
        except Exception:  # noqa: BLE001
            pass
    log.info("CLI: %s", text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hashem", description="Hashem Agent — free AI design studio")
    parser.add_argument("--version", action="store_true", help="print version")
    parser.add_argument("--out", help="write machine output to this file (for CI)")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="start the web studio (default)")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--no-browser", action="store_true")

    chat = sub.add_parser("chat", help="interactive terminal chat")
    chat.add_argument("--model", default=None)

    render = sub.add_parser("render", help="render a motion-graphics project to video")
    render.add_argument("--project", help="path to a project JSON")
    render.add_argument("--topic", help="or build from a topic")
    render.add_argument("--points", help="pipe-separated bullet points")
    render.add_argument("--template", default="intro_points_outro")
    render.add_argument("--palette", default="")
    render.add_argument("--out", default=None)
    render.add_argument("--no-audio", action="store_true")

    book = sub.add_parser("book", help="build a book from a file or text")
    book.add_argument("--file")
    book.add_argument("--text")
    book.add_argument("--title", default="")
    book.add_argument("--author", default="")
    book.add_argument("--theme", default="classic")
    book.add_argument("--formats", default="pdf,html")

    doctor = sub.add_parser("doctor", help="run diagnostics")
    doctor.add_argument("--json", action="store_true", dest="as_json")
    doctor.add_argument("--out", default=None)

    sub.add_parser("models", help="list models from the configured provider")
    return parser


def _write_out(path: str | None, text: str) -> None:
    if path:
        Path(path).write_text(text, encoding="utf-8")
        emit(f"wrote {path}")


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    from .config import load_settings, save_settings
    from .doctor import format_report, run_diagnostics

    settings = load_settings()

    if args.version:
        _write_out(args.out, json.dumps({"version": __version__}))
        emit(f"Hashem Agent {__version__}")
        return 0

    command = args.command or "serve"

    if command == "doctor":
        report = run_diagnostics()
        if args.as_json:
            _write_out(getattr(args, "out", None) or None, json.dumps(report, ensure_ascii=False, indent=2))
            emit(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            _write_out(getattr(args, "out", None) or None, json.dumps(report, ensure_ascii=False, indent=2))
            emit(format_report(report))
        return 0 if report["all_ok"] else 1

    if command == "models":
        from .llm.provider import ChatProvider

        provider = ChatProvider(settings.provider)
        models = provider.list_models()
        emit(json.dumps({"provider": settings.provider.provider, "models": models}, ensure_ascii=False))
        return 0

    if command == "render":
        return _cmd_render(args, settings)

    if command == "book":
        return _cmd_book(args, settings)

    if command == "chat":
        return _cmd_chat(args, settings)

    return _cmd_serve(args, settings)


# --------------------------------------------------------------------------- #


def _cmd_render(args, settings) -> int:
    from .agent.context import AppContext
    from .motion import templates
    from .motion.pipeline import render_project

    ctx = AppContext(settings)
    if args.project:
        raw = json.loads(Path(args.project).read_text(encoding="utf-8"))
    else:
        points = [p for p in (args.points or "").split("|") if p.strip()]
        raw = templates.build(args.template, topic=args.topic or "فيديو", points=points, palette=args.palette)
    output = args.out or ctx.output_path(str(raw.get("name", "video")), ".mp4")
    result = render_project(raw, output, settings.video, include_audio=not args.no_audio)
    emit(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _cmd_book(args, settings) -> int:
    from .agent.context import AppContext
    from .book import export_book, parse_document, parse_file, typeset

    ctx = AppContext(settings)
    if args.file:
        doc = parse_file(args.file, author=args.author)
    else:
        doc = parse_document(args.text or "", title=args.title or None, author=args.author)
    result = typeset(doc, args.theme)
    outputs = export_book(result, ctx.exports, [f.strip() for f in args.formats.split(",") if f.strip()])
    emit(json.dumps({"pages": result.page_count, "outputs": {k: str(v) for k, v in outputs.items()}}, ensure_ascii=False, indent=2))
    return 0


def _cmd_chat(args, settings) -> int:
    from .agent import AgentLoop, AppContext, Memory

    ctx = AppContext(settings)
    memory = Memory()
    session = memory.new_session()
    loop = AgentLoop(ctx, on_event=_print_event)
    emit("هاشم جاهز. اكتب 'exit' للخروج.")
    while True:
        try:
            emit("\nأنت: ")
            user = input() if sys.stdin and not sys.stdin.closed else ""
        except (EOFError, KeyboardInterrupt):
            break
        if not user or user.strip().lower() in {"exit", "quit", "خروج"}:
            break
        loop.run(session, user)
        memory.save(session)
    return 0


def _print_event(event: dict[str, Any]) -> None:
    kind = event.get("type")
    if kind == "tool_call":
        emit(f"  ⟶ أداة: {event['name']}")
    elif kind == "tool_result":
        path = event.get("result", {}).get("path")
        if path:
            emit(f"  ✓ {path}")
    elif kind == "final":
        emit(f"\nهاشم: {event['text']}")
    elif kind == "error":
        emit(f"  ! {event['message']}")


def _cmd_serve(args, settings) -> int:
    from .ui.server import serve

    host = args.host or settings.ui.host
    port = args.port or settings.ui.port
    serve(settings, host=host, port=port, open_browser=not args.no_browser)
    return 0


__all__ = ["main", "emit", "build_parser"]
