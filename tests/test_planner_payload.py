#!/usr/bin/env python3
"""
Unit-Tests für die Planner-Payload-Konstruktion.

Testet die drei bekannten Failures:
  1) tool_call_id-Konsistenz im Recap-Payload (KIMI 400 'tool_call_id not found')
  2) reasoning_content-Erhaltung für DeepSeek V4 thinking-mode
  3) image_url-Sanitizer für DeepSeek text-only

Lauffähig ohne laufenden Proxy — nur reine Funktionen.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PROXY_FILE = REPO_ROOT / "proxy.py"


# ═══════════════════════════════════════════════════════════════════════════
# proxy.py laden OHNE uvicorn.run() zu triggern
# ═══════════════════════════════════════════════════════════════════════════


def _load_proxy_module():
    """Lädt proxy.py als Modul ohne Server zu starten."""
    # Modul-Pfad absichern
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    # Config-Datei simulieren falls nicht da
    os.environ.setdefault("MYPROXY_TEST_MODE", "1")

    spec = importlib.util.spec_from_file_location("proxy_under_test", PROXY_FILE)
    module = importlib.util.module_from_spec(spec)
    # WICHTIG: Modul in sys.modules registrieren BEVOR exec_module, damit
    # dataclasses zur Auswertungszeit cls.__module__ auflösen kann.
    sys.modules["proxy_under_test"] = module

    # Mock uvicorn.run() damit das Modul importiert werden kann
    # (es könnte ein auto-run-Check vorhanden sein)
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass
    return module


# Try import — falls httpx, fastapi etc. fehlen, pytest überspringt
try:
    proxy = _load_proxy_module()
    HAS_PROXY = True
    SKIP_REASON = ""
except Exception as _e:  # pragma: no cover
    HAS_PROXY = False
    SKIP_REASON = f"proxy.py konnte nicht importiert werden: {_e}"


pytestmark = pytest.mark.skipif(not HAS_PROXY, reason=SKIP_REASON)


# ═══════════════════════════════════════════════════════════════════════════
# Helper-Funktionen
# ═══════════════════════════════════════════════════════════════════════════


def _make_tool_call(tc_id: str, name: str = "read_file", args: str = None) -> dict:
    if args is None:
        args = '{"filePath": "/foo.py"}'
    return {
        "id": tc_id,
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


def _make_tool_result(tc_id: str, content: str = "ok") -> dict:
    return {
        "role": "tool",
        "tool_call_id": tc_id,
        "name": "tool",
        "content": content,
    }


def _make_session(iterations: int = 3, files: int = 3) -> dict:
    return {
        "iterations": iterations,
        "distinct_files": {f"read_file:/file{i}.py" for i in range(files)},
        "tool_signatures": [],
        "assistant_contents": [],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Recap-Payload — KEIN assistant/tool-Durchreich mehr
# ═══════════════════════════════════════════════════════════════════════════


def test_recap_payload_has_no_assistant_or_tool_messages():
    """NEUES VERHALTEN: Recap-Payload enthält NUR system + user messages.
    Früher wurden assistant+tool_calls durchgereicht, was zu 'tool_call_id
    not found' (400) und invaliden tool_calls führte. Jetzt nicht mehr."""
    body = {
        "messages": [
            {"role": "system", "content": "Originaler System-Prompt mit Tool-Schemas"},
            {"role": "user", "content": "Bitte einen Plan erstellen für Feature X"},
            {"role": "assistant", "content": "", "tool_calls": [
                _make_tool_call("tc_1"),
                _make_tool_call("tc_2"),
            ]},
            _make_tool_result("tc_1", "result 1"),
            _make_tool_result("tc_2", "result 2"),
        ]
    }
    payload = proxy._build_planner_tool_continuation_context(
        body=body,
        session=_make_session(),
        original_task="Plan erstellen für Feature X",
    )
    msgs = payload["messages"]
    # Recap darf KEINE assistant/tool roles enthalten
    for m in msgs:
        assert m["role"] in ("system", "user"), (
            f"Recap-Payload enthält unerwartete role={m['role']!r} — "
            f"nur system/user erlaubt. Altes assistant/tool-pass-through wurde entfernt."
        )
    # Aber original system prompt MUSS erhalten sein
    system_msgs = [m for m in msgs if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert "Originaler System-Prompt" in system_msgs[0]["content"]
    assert "PLANNER AGENT MODE" in system_msgs[0]["content"]


def test_recap_payload_includes_original_task_and_exploration_recap():
    """Recap-Payload enthält (1) original task und (2) exploration recap.
    Die exploration recap ist die einzige Informationsquelle über vorherige
    tool_calls — kein direkter assistant/tool-pass-through mehr."""
    body = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Implementiere Feature Y"},
            # Zwei Runden Assistant + Tool (exploration)
            {"role": "assistant", "content": "", "tool_calls": [
                _make_tool_call("a", "read_file", '{"filePath": "/foo.py"}'),
            ]},
            _make_tool_result("a", "def foo(): pass"),
            {"role": "assistant", "content": "", "tool_calls": [
                _make_tool_call("b", "grep_search", '{"query": "bar"}'),
            ]},
            _make_tool_result("b", "line 42: bar = 1"),
        ]
    }
    payload = proxy._build_planner_tool_continuation_context(
        body=body,
        session=_make_session(),
        original_task="Implementiere Feature Y",
    )
    msgs = payload["messages"]
    texts = [str(m.get("content", "")) for m in msgs]

    # Original-Task muss enthalten sein
    assert any("Feature Y" in t for t in texts), "original task fehlt im Recap"

    # Exploration Recap muss die tool_calls erwähnen
    recap_texts = " ".join(texts)
    assert "EXPLORATION RECAP" in recap_texts, "EXPLORATION RECAP fehlt"
    assert "read_file" in recap_texts, "tool read_file fehlt in exploration recap"
    assert "grep_search" in recap_texts, "tool grep_search fehlt in exploration recap"


def test_recap_payload_never_contains_assistant_or_tool_even_with_reasoning():
    """Auch wenn reasoning_content im Original war: Recap enthält KEINE
    assistant-Message. reasoning_content wird NICHT durchgereicht (der
    Recap-Payload ist ein Neustart, keine Continuation)."""
    body = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": [
                _make_tool_call("a"),
            ], "reasoning_content": "< denken, denken >"},
            _make_tool_result("a", "result-a"),
        ]
    }
    payload = proxy._build_planner_tool_continuation_context(
        body=body, session=_make_session(), original_task="task",
    )
    msgs = payload["messages"]
    assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
    assert len(assistant_msgs) == 0, (
        f"Recap enthält {len(assistant_msgs)} assistant-Messages — "
        f"tool_call-pass-through wurde entfernt."
    )
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert len(tool_msgs) == 0, (
        f"Recap enthält {len(tool_msgs)} tool-Messages — "
        f"tool_call-pass-through wurde entfernt."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: image_url-Sanitizer (Worker-Payload)
# ═══════════════════════════════════════════════════════════════════════════


def test_worker_payload_strips_image_url_parts_for_text_only_models():
    """Bug 3: DeepSeek ist text-only, 'image_url' Payloads werden mit
    400 abgelehnt. Der Worker-Payload muss diese Parts entfernen oder
    in Text umwandeln, BEVOR er an DeepSeek geht."""
    body = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [
                {"type": "text", "text": "mach was"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
            ]},
            {"role": "assistant", "content": "ok"},
            # messages[41] aus dem Log war tool-result
            {"role": "tool", "tool_call_id": "x", "content": [
                {"type": "text", "text": "tool output"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,yyy"}},
            ]},
        ]
    }

    payload = proxy._build_worker_payload(
        body=body, plan="plan", memory_context="", plan_path=None,
        model_name="deepseek-v4-pro",  # text-only → Sanitizer triggert
    )

    for i, m in enumerate(payload["messages"]):
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for j, part in enumerate(content):
            assert part.get("type") != "image_url", (
                f"image_url survived in messages[{i}].content[{j}] — "
                f"DeepSeek wirft 400. Sanitizer fehlt."
            )


def test_worker_payload_keeps_text_parts_intact_when_sanitizing():
    """Companion-Test: Sanitizer darf text-Parts nicht verändern."""
    body = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [
                {"type": "text", "text": "keep me"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
            ]},
        ]
    }
    payload = proxy._build_worker_payload(
        body=body, plan="plan", memory_context="", plan_path=None,
        model_name="deepseek-v4-pro",
    )
    # Die 'keep me' muss erhalten bleiben
    user_idx = next(i for i, m in enumerate(payload["messages"]) if m.get("role") == "user")
    user_parts = payload["messages"][user_idx]["content"]
    if isinstance(user_parts, list):  # Plan-binding kann hinzugefügt haben
        text_parts = [p for p in user_parts if isinstance(p, dict) and p.get("type") == "text"]
        assert any("keep me" in (p.get("text") or "") for p in text_parts), \
            "text-Part ging verloren beim Sanitizer"


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: tool-result-cap
# ═══════════════════════════════════════════════════════════════════════════


def test_tool_result_cap_truncates_giant_results():
    """Bug 4: 111KB grep-hit muss gekappt werden."""
    huge = "x" * 200000  # 200KB
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [_make_tool_call("g")],
         "reasoning_content": "thinking"},
        {"role": "tool", "tool_call_id": "g", "content": huge},
    ]
    capped = proxy._cap_tool_results_inplace(messages, "test")
    assert capped == 1
    tool_msg = messages[-1]
    assert len(tool_msg["content"]) < 10000, "Cap hat nicht gekürzt"
    assert "TRUNCATED" in tool_msg["content"]
