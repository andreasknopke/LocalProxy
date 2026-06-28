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
# Test 1: tool_call_id-Konsistenz im Recap-Payload
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


def test_recap_payload_only_includes_tool_results_matching_last_assistant_tool_calls():
    """Bug 1: Recap-Payload darf NUR tool_results für die tool_calls der
    LETZTEN assistant-Message enthalten, sonst → 'tool_call_id not found'.
    """
    body = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Bitte einen Plan erstellen für Feature X"},
            # Erste Assistent-Runde (älter) mit 2 tool_calls
            {"role": "assistant", "content": "", "tool_calls": [
                _make_tool_call("tc_old_1"),
                _make_tool_call("tc_old_2"),
            ]},
            _make_tool_result("tc_old_1", "result for old 1"),
            _make_tool_result("tc_old_2", "result for old 2"),
            # Letzte Assistent-Runde: 4 tool_calls, ABER wir fügen nur die
            # letzen 2 tool_results hinzu — Müll!
            {"role": "assistant", "content": "", "tool_calls": [
                _make_tool_call("tc_new_1"),
                _make_tool_call("tc_new_2"),
                _make_tool_call("tc_new_3"),
                _make_tool_call("tc_new_4"),
            ]},
            # Erwartet: Recap-Payload darf hier NUR tool_results anhängen,
            # die zu {tc_new_1..4} gehören.angingefügte alte results müssen
            # ignoriert werden.
            _make_tool_result("tc_new_1", "r1"),
            _make_tool_result("tc_new_2", "r2"),
            _make_tool_result("tc_new_3", "r3"),
            _make_tool_result("tc_new_4", "r4"),
        ]
    }

    payload = proxy._build_planner_tool_continuation_context(
        body=body,
        session=_make_session(),
        original_task="Plan erstellen für Feature X",
    )

    msgs = payload["messages"]
    # Finde die letzte assistant-Message
    last_asst_idx = max(i for i, m in enumerate(msgs) if m.get("role") == "assistant")
    # Alle Messages danach müssen tool-Results sein — UND deren IDs müssen
    # in den tool_calls der last-assistant stehen.
    expected_ids = {tc["id"] for tc in msgs[last_asst_idx].get("tool_calls", [])}
    for m in msgs[last_asst_idx + 1:]:
        assert m["role"] == "tool", f"message after last assistant must be tool, got {m.get('role')}"
        tc_id = m.get("tool_call_id")
        assert tc_id in expected_ids, (
            f"tool_call_id {tc_id!r} nicht in last-assistant-tool_calls {expected_ids}. "
            f"DAS IST DER KIMI-400-BUG."
        )


def test_recap_payload_never_forgets_tool_result_for_last_assistant_call():
    """Bug 1 (Spiegel): Jede tool_call_id der letzten assistant-Message
    MUSS einen matching tool_result haben. Sonst 400."""
    body = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": [
                _make_tool_call("a"),
                _make_tool_call("b"),
                _make_tool_call("c"),
            ]},
            _make_tool_result("a", "r-a"),
            _make_tool_result("c", "r-c"),  # 'b' fehlt im Input
        ]
    }
    payload = proxy._build_planner_tool_continuation_context(
        body=body, session=_make_session(), original_task="task",
    )
    msgs = payload["messages"]
    last_asst_idx = max(i for i, m in enumerate(msgs) if m.get("role") == "assistant")
    last_asst_tc_ids = {tc["id"] for tc in msgs[last_asst_idx].get("tool_calls", [])}
    provided_ids = {m.get("tool_call_id") for m in msgs[last_asst_idx + 1:] if m.get("role") == "tool"}
    # 'b' kann nicht lieferbar sein (Input unvollständig) - aber dann muss
    # der Recap-Payload 'b' aus den tool_calls der last-assistant entfernen!
    orphan_ids = last_asst_tc_ids - provided_ids
    assert not orphan_ids, (
        f"tool_call_ids ohne matching result: {orphan_ids}. "
        f"Recap-Payload muss orphan tool_calls DROPPEN statt result leer zu lassen."
    )


def test_recap_payload_preserves_reasoning_content_for_thinking_mode():
    """Bug 2: DeepSeek V4 im thinking mode braucht reasoning_content auch
    im Recap-Payload, sonst: 400 'reasoning_content must be passed back'.
    """
    body = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "arbeite", "tool_calls": [
                _make_tool_call("a"),
            ], "reasoning_content": "< denken, denken, denken >"},
            _make_tool_result("a", "result-a"),
        ]
    }
    payload = proxy._build_planner_tool_continuation_context(
        body=body, session=_make_session(), original_task="task",
    )
    msgs = payload["messages"]
    last_asst_idx = max(i for i, m in enumerate(msgs) if m.get("role") == "assistant")
    last_asst = msgs[last_asst_idx]
    assert "reasoning_content" in last_asst, (
        "letzte assistant-Message im Recap-Payload hat KEIN reasoning_content — "
        "DeepSeek-V4 thinking mode wirft 400."
    )
    assert last_asst["reasoning_content"], "reasoning_content ist leer/None"


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
