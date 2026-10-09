"""A single resilient, streaming, tool-calling client for OpenAI-compatible APIs.

Covers Ollama (``/v1``), LM Studio, Groq, OpenRouter, Gemini and OpenAI.  There is
deliberately no SDK dependency: ``requests`` plus careful parsing keeps the
footprint small and the packaged executable self-contained.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import requests

from ..config import ProviderSettings
from ..utils.log import get_logger
from ..utils.safety import redact

log = get_logger("llm")


class ProviderError(RuntimeError):
    """Raised when the provider is unreachable or returns an unrecoverable error."""


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def summary(self) -> str:
        keys = ", ".join(list(self.arguments)[:4])
        return f"{self.name}({keys})"


@dataclass
class ChatResult:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


def normalise_base_url(provider: str, base_url: str) -> str:
    """Return the ``/v1`` root for an OpenAI-compatible backend."""
    url = (base_url or "").strip().rstrip("/")
    if not url:
        url = "http://127.0.0.1:11434" if provider == "ollama" else url
    if provider == "ollama":
        if not url.endswith("/v1"):
            url += "/v1"
    return url


def _extract_json_block(text: str) -> list[ToolCall] | None:
    """Fallback parser for models that emit tool calls as fenced JSON instead of
    using the native ``tool_calls`` field."""
    calls = []
    for chunk in text.split("```"):
        chunk = chunk.strip()
        if chunk.startswith("json"):
            chunk = chunk[4:].strip()
        if not chunk.startswith("{") and not chunk.startswith("["):
            continue
        try:
            data = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict) or "name" not in item:
                continue
            args = item.get("arguments", item.get("args", item.get("parameters", {})))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(name=str(item["name"]), arguments=args or {}))
    return calls or None


class ChatProvider:
    """Streaming + tool-calling chat against any OpenAI-compatible endpoint."""

    def __init__(self, settings: ProviderSettings):
        self.settings = settings
        self.base_url = normalise_base_url(settings.provider, settings.base_url)

    # ------------------------------------------------------------------ plumbing
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    def _url(self) -> str:
        return f"{self.base_url}/chat/completions"

    # -------------------------------------------------------------------- health
    def health(self) -> dict[str, Any]:
        """Cheap probe used by ``doctor`` and the connection test button."""
        try:
            url = f"{self.base_url}/models"
            response = requests.get(url, headers=self._headers(), timeout=5)
            if response.ok:
                data = response.json()
                models = [m.get("id") for m in data.get("data", [])][:20]
                return {"ok": True, "models": models, "base_url": self.base_url}
            return {"ok": False, "status": response.status_code, "base_url": self.base_url}
        except requests.RequestException as exc:
            return {"ok": False, "error": str(exc), "base_url": self.base_url}

    def list_models(self) -> list[str]:
        try:
            response = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=8)
            if response.ok:
                return [m.get("id", "") for m in response.json().get("data", [])]
        except requests.RequestException:
            pass
        return []

    # ---------------------------------------------------------------------- chat
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_retries: int = 2,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "stream": False,
        }
        if self.settings.max_tokens:
            payload["max_tokens"] = self.settings.max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    self._url(),
                    json=payload,
                    headers=self._headers(),
                    timeout=self.settings.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(0.6 * (attempt + 1))
                continue

            if response.status_code in (401, 403):
                raise ProviderError(f"Authentication failed ({response.status_code}). Check the API key.")
            if response.status_code == 404:
                raise ProviderError(f"Endpoint not found at {self._url()} — is the server running?")
            if not response.ok:
                last_error = ProviderError(f"HTTP {response.status_code}: {response.text[:300]}")
                time.sleep(0.6 * (attempt + 1))
                continue

            return self._parse(response.json())

        raise ProviderError(f"Provider unreachable after retries: {last_error}")

    def _parse(self, data: dict[str, Any]) -> ChatResult:
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"Malformed provider response: {str(data)[:200]}") from exc
        message = choice.get("message", {}) or {}
        content = message.get("content") or ""

        tool_calls: list[ToolCall] = []
        for call in message.get("tool_calls") or []:
            function = call.get("function", {}) or {}
            raw_args = function.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(name=function.get("name", ""), arguments=args or {}, id=call.get("id", "")))

        if not tool_calls and content:
            parsed = _extract_json_block(content)
            if parsed:
                tool_calls = parsed

        usage = data.get("usage", {}) or {}
        return ChatResult(
            content=content,
            tool_calls=tool_calls,
            model=data.get("model", self.settings.model),
            usage={"prompt": usage.get("prompt_tokens", 0), "completion": usage.get("completion_tokens", 0)},
            finish_reason=choice.get("finish_reason", ""),
        )

    # ------------------------------------------------------------------ streaming
    def stream(self, messages: list[dict[str, Any]]) -> Iterator[str]:
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "stream": True,
        }
        try:
            with requests.post(self._url(), json=payload, headers=self._headers(),
                               timeout=self.settings.timeout, stream=True) as response:
                if not response.ok:
                    raise ProviderError(f"HTTP {response.status_code}: {response.text[:200]}")
                for raw in response.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data_text = line[5:].strip()
                    if data_text == "[DONE]":
                        break
                    try:
                        data = json.loads(data_text)
                    except json.JSONDecodeError:
                        continue
                    delta = (data.get("choices") or [{}])[0].get("delta", {}) or {}
                    token = delta.get("content")
                    if token:
                        yield token
        except requests.RequestException as exc:
            raise ProviderError(f"stream interrupted: {exc}") from exc

    def ask_simple(self, prompt: str, system: str = "") -> str:
        """Convenience for one-shot completion (used by the learning self-critique)."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages).content


__all__ = ["ChatProvider", "ChatResult", "ToolCall", "ProviderError", "normalise_base_url"]
