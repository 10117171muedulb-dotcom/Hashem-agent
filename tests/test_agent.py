"""Tests for learning, tool execution and the safety rails."""

from __future__ import annotations

import pytest

from hashem.learning import SkillStore
from hashem.agent.context import AppContext
from hashem.agent.tools import execute, get_tool, tool_schemas
from hashem.utils.safety import PathGuard, PathViolation, looks_destructive, redact
from hashem.config import Settings
from hashem.llm.provider import normalise_base_url, _extract_json_block


# ------------------------------------------------------------------ learning
def test_skill_store_roundtrip(tmp_path):
    store = SkillStore(tmp_path / "skills.json")
    skill = store.remember("video", "فيديو عن القهوة", {"palette": "gold"}, {"quality": 0.8})
    store.rate(skill.id, 1.0)
    again = SkillStore(tmp_path / "skills.json")
    assert len(again.skills) == 1
    assert again.skills[0].rating == 1.0
    similar = again.similar("فيديو عن القهوة العربية", "video")
    assert similar and similar[0].id == skill.id
    profile = again.style_profile()
    assert profile["favourite_palette"] == "gold"


# ------------------------------------------------------------------ tools
def test_tool_schemas_valid():
    for schema in tool_schemas():
        assert schema["type"] == "function"
        assert schema["function"]["parameters"]["type"] == "object"


def test_execute_create_image(tmp_path):
    ctx = AppContext(Settings())
    result = execute(ctx, "create_image", {"kind": "quote", "text": "نص", "palette": "royal"})
    assert result["ok"]
    assert result["path"].endswith(".png")


def test_execute_unknown_tool():
    ctx = AppContext(Settings())
    result = execute(ctx, "does_not_exist", {})
    assert not result["ok"]


def test_execute_guarded_read_blocks_escape(tmp_path):
    ctx = AppContext(Settings())
    result = execute(ctx, "read_file", {"path": "../../etc/passwd"})
    assert not result["ok"]


def test_system_info():
    ctx = AppContext(Settings())
    result = execute(ctx, "system_info", {})
    assert result["ok"] and "ffmpeg" in result


# ------------------------------------------------------------------ safety
def test_path_guard_jail(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    guard = PathGuard([root])
    assert guard.resolve("inner/file.txt").is_relative_to(root)
    with pytest.raises(PathViolation):
        guard.resolve("/etc/passwd")
    with pytest.raises(PathViolation):
        guard.resolve("../secret")


def test_destructive_detection():
    assert looks_destructive("rm -rf /")[0]
    assert looks_destructive("git push --force")[0]
    assert looks_destructive("echo hello")[0] is False


def test_redact():
    assert "***" in redact("api_key=sk-abcdefgh12345678")


# ------------------------------------------------------------------ llm
def test_normalise_base_url():
    assert normalise_base_url("ollama", "http://127.0.0.1:11434").endswith("/v1")
    assert normalise_base_url("openai_compatible", "https://api.groq.com/openai/v1").endswith("/v1")


def test_extract_json_block_tool_calls():
    text = 'بعض النص\n```json\n{"name": "create_video", "arguments": {"topic": "ق"}}\n```'
    calls = _extract_json_block(text)
    assert calls and calls[0].name == "create_video"
    assert calls[0].arguments["topic"] == "ق"
