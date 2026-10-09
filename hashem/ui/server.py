"""A dependency-free local HTTP studio.

Why stdlib ``http.server``: it ships with Python, packs cleanly into the .exe, and
needs zero configuration to run on Windows.  The frontend is a single Arabic RTL
single-page app served from ``hashem/ui/static``.
"""

from __future__ import annotations

import json
import mimetypes
import queue
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .. import __version__
from ..agent import AgentLoop, AppContext, Memory
from ..agent.tools import execute as execute_tool
from ..config import MODEL_PRESETS, apply_preset, save_settings
from ..utils.log import get_logger
from ..utils.paths import bundle_dir
from ..utils.safety import PathViolation

log = get_logger("ui")

STATIC_DIR = Path(__file__).resolve().parent / "static"


class Studio:
    def __init__(self, settings):
        self.settings = settings
        self.context = AppContext(settings)
        self.memory = Memory()
        self.sessions: dict[str, Any] = {}
        self.httpd: ThreadingHTTPServer | None = None
        self.shutdown_flag = threading.Event()

    def session(self, session_id: str | None) -> Any:
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        loaded = self.memory.load(session_id) if session_id else None
        if loaded is None:
            loaded = self.memory.new_session()
        self.sessions[loaded.id] = loaded
        return loaded


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def make_handler(studio: Studio):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"HashemAgent/{__version__}"
        protocol_version = "HTTP/1.1"

        # ------------------------------------------------------------- helpers
        def log_message(self, fmt, *args):  # keep logs tidy
            log.debug("HTTP %s", fmt % args)

        def _send(self, status: int, body: bytes, content_type: str = "application/json; charset=utf-8") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _send_json(self, payload: Any, status: int = 200) -> None:
            self._send(status, _json_bytes(payload))

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError:
                return {}

        def _not_found(self) -> None:
            self._send_json({"ok": False, "error": "not found"}, 404)

        # ------------------------------------------------------------ dispatch
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            if path in {"/", "/index.html"}:
                return self._serve_static("index.html")
            if path.startswith("/static/"):
                return self._serve_static(path[len("/static/"):])
            if path == "/api/health":
                return self._send_json({"ok": True, "version": __version__, "name": "Hashem Agent"})
            if path == "/api/state":
                return self._state()
            if path == "/api/sessions":
                return self._send_json({"sessions": studio.memory.list_sessions()})
            if path == "/api/file":
                return self._serve_file(query.get("path", [""])[0])
            return self._not_found()

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            body = self._body()

            if path == "/api/settings":
                return self._update_settings(body)
            if path == "/api/preset":
                return self._apply_preset(body)
            if path == "/api/sessions":
                session = studio.memory.new_session(body.get("title", "محادثة جديدة"))
                return self._send_json({"id": session.id, "title": session.title})
            if path == "/api/chat":
                return self._chat(body)
            if path == "/api/tool":
                result = execute_tool(studio.context, body.get("name", ""), body.get("arguments", {}) or {})
                return self._send_json(result)
            if path == "/api/render":
                return self._render(body)
            if path == "/api/shutdown":
                self._send_json({"ok": True, "bye": True})
                threading.Thread(target=studio.shutdown_flag.set, daemon=True).start()
                return
            return self._not_found()

        def do_DELETE(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/sessions/"):
                session_id = parsed.path.rsplit("/", 1)[-1]
                studio.sessions.pop(session_id, None)
                return self._send_json({"deleted": studio.memory.delete(session_id)})
            return self._not_found()

        # ----------------------------------------------------------- endpoints
        def _serve_static(self, name: str) -> None:
            safe = Path(name).name
            target = STATIC_DIR / safe
            if not target.exists():
                return self._not_found()
            ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            if safe.endswith(".js"):
                ctype = "text/javascript; charset=utf-8"
            if safe.endswith(".css"):
                ctype = "text/css; charset=utf-8"
            if safe.endswith(".html"):
                ctype = "text/html; charset=utf-8"
            self._send(200, target.read_bytes(), ctype)

        def _serve_file(self, raw_path: str) -> None:
            try:
                target = studio.context.guard.resolve(raw_path, must_exist=True)
            except PathViolation as exc:
                return self._send_json({"ok": False, "error": str(exc)}, 403)
            if not target.is_file():
                return self._not_found()
            ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            if target.suffix in {".mp4", ".webm"}:
                ctype = "video/mp4"
            if target.suffix == ".png":
                ctype = "image/png"
            self._send(200, target.read_bytes(), ctype)

        def _state(self) -> None:
            from ..agent.tools import execute as _exec

            capabilities = _exec(studio.context, "capabilities", {})
            learning = _exec(studio.context, "learning_stats", {})
            from ..motion.fonts import available_families

            self._send_json({
                "version": __version__,
                "settings": studio.settings.redacted(),
                "presets": [{k: v for k, v in p.items()} for p in MODEL_PRESETS],
                "capabilities": capabilities,
                "learning": learning,
                "fonts": available_families()[:40],
                "workspace": str(studio.context.workspace),
                "exports": str(studio.context.exports),
            })

        def _update_settings(self, body: dict[str, Any]) -> None:
            for section, values in body.items():
                current = getattr(studio.settings, section, None)
                if current is None or not isinstance(values, dict):
                    continue
                for key, value in values.items():
                    if hasattr(current, key):
                        setattr(current, key, value)
            save_settings(studio.settings)
            studio.context.settings = studio.settings
            studio.context.rebuild_provider()
            self._send_json({"ok": True, "settings": studio.settings.redacted()})

        def _apply_preset(self, body: dict[str, Any]) -> None:
            try:
                apply_preset(studio.settings, body.get("preset_id", ""), body.get("api_key", ""))
            except KeyError as exc:
                return self._send_json({"ok": False, "error": str(exc)}, 400)
            save_settings(studio.settings)
            studio.context.rebuild_provider()
            self._send_json({"ok": True, "settings": studio.settings.redacted()})

        def _render(self, body: dict[str, Any]) -> None:
            from ..motion.pipeline import render_project

            project = body.get("project")
            if project is None:
                from ..motion import templates

                project = templates.build(
                    body.get("template", "intro_points_outro"),
                    topic=body.get("topic", "فيديو"),
                    points=body.get("points") or [],
                    palette=body.get("palette", ""),
                )
            name = str(body.get("name") or (project or {}).get("name") or "video")
            try:
                result = render_project(project, studio.context.output_path(name, ".mp4"), studio.settings.video,
                                        include_audio=bool(body.get("audio", True)))
            except Exception as exc:  # noqa: BLE001
                return self._send_json({"ok": False, "error": str(exc)}, 500)
            self._send_json(result)

        def _chat(self, body: dict[str, Any]) -> None:
            session = studio.session(body.get("session"))
            message = str(body.get("message", "")).strip()
            if not message:
                return self._send_json({"ok": False, "error": "empty message"}, 400)

            events: queue.Queue = queue.Queue()
            loop = AgentLoop(studio.context, on_event=events.put)

            def run() -> None:
                try:
                    loop.run(session, message)
                finally:
                    studio.memory.save(session)
                    events.put(None)

            threading.Thread(target=run, daemon=True).start()

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                while True:
                    event = events.get(timeout=600)
                    if event is None:
                        break
                    payload = f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    self.wfile.write(payload.encode("utf-8"))
                    self.wfile.flush()
                self.wfile.write(b"data: {\"type\": \"done\"}\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    return Handler


def serve(settings, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    """Start the studio and block until shutdown is requested."""
    studio = Studio(settings)
    handler = make_handler(studio)
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    studio.httpd = httpd
    url = f"http://{host if host not in {'0.0.0.0', '::'} else '127.0.0.1'}:{port}"
    log.info("Studio listening on %s", url)
    print(f"Hashem Agent studio: {url}")

    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    def shutdown_watcher() -> None:
        studio.shutdown_flag.wait()
        httpd.shutdown()

    threading.Thread(target=shutdown_watcher, daemon=True).start()
    try:
        httpd.serve_forever(poll_interval=0.4)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        log.info("Studio stopped")


__all__ = ["serve", "Studio"]
