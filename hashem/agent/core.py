"""The agent loop: reason → act → observe, until a final answer.

Kept deliberately provider-agnostic and event-driven so the same loop powers the
CLI and the web UI.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from ..utils.log import get_logger
from . import planner
from .context import AppContext
from .memory import Session
from .prompts import build_system
from .tools import execute, tool_schemas

log = get_logger("agent.core")

Event = Callable[[dict[str, Any]], None]


def _noop(_event: dict[str, Any]) -> None:
    return None


class AgentLoop:
    def __init__(self, context: AppContext, on_event: Event = _noop):
        self.context = context
        self.on_event = on_event

    # ------------------------------------------------------------------ events
    def _emit(self, event: dict[str, Any]) -> None:
        try:
            self.on_event(event)
        except Exception:  # noqa: BLE001 - a broken UI callback must not kill the loop
            log.debug("event callback raised", exc_info=True)

    # ----------------------------------------------------- offline fallback
    def _offline_fallback(self, session: Session, user_message: str) -> str:
        """Execute the request with the deterministic local planner (no LLM)."""
        self._emit({
            "type": "token",
            "text": "🤖 لا يوجد مزوّد ذكاء متصل (Ollama/سحابي) — أستخدم المخطط المحلي المجاني دون إنترنت.\n",
        })
        results: list[dict[str, Any]] = []
        for step in planner.plan(user_message):
            name = step["tool"]
            arguments = step.get("arguments", {})
            self._emit({"type": "tool_call", "name": name, "arguments": arguments})
            if step.get("say"):
                self._emit({"type": "token", "text": step["say"] + "\n"})
            result = execute(self.context, name, arguments)
            self._emit({"type": "tool_result", "name": name, "result": result})
            results.append(result)
        final_text = planner.offline_reply(user_message, results)
        self._emit({"type": "final", "text": final_text})
        session.add("assistant", final_text)
        return final_text

    # ------------------------------------------------------------------- entry
    def run(self, session: Session, user_message: str) -> str:
        settings = self.context.settings
        session.add("user", user_message)
        self.context.save_session if hasattr(self.context, "save_session") else None

        system = build_system(extra=settings.agent.system_prompt_extra, language=settings.agent.language)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        messages += [m for m in session.to_messages() if m.get("content") or m.get("role") == "assistant"]

        schemas = tool_schemas()
        final_text = ""

        for step in range(max(1, settings.agent.max_steps)):
            self._emit({"type": "thinking", "step": step + 1})
            try:
                result = self.context.provider.chat(messages, tools=schemas)
            except Exception as exc:  # noqa: BLE001
                log.info("Provider unavailable (%s); using the offline planner", exc)
                return self._offline_fallback(session, user_message)

            if result.has_tool_calls:
                assistant_note = result.content or ""
                tool_calls_payload = [
                    {"id": call.id or f"call_{step}_{i}", "type": "function",
                     "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)}}
                    for i, call in enumerate(result.tool_calls)
                ]
                messages.append({"role": "assistant", "content": assistant_note, "tool_calls": tool_calls_payload})
                if assistant_note:
                    self._emit({"type": "token", "text": assistant_note + "\n"})

                for call in result.tool_calls:
                    self._emit({"type": "tool_call", "name": call.name, "arguments": call.arguments})
                    tool_result = execute(self.context, call.name, call.arguments)
                    self._emit({"type": "tool_result", "name": call.name, "result": tool_result})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id or "",
                        "name": call.name,
                        "content": json.dumps(tool_result, ensure_ascii=False)[:6000],
                    })
                continue

            final_text = result.content.strip()
            break

        if not final_text:
            final_text = "تم تنفيذ المطلوب. راجع النتائج أعلاه."
        self._emit({"type": "final", "text": final_text})
        session.add("assistant", final_text)
        return final_text


__all__ = ["AgentLoop"]
