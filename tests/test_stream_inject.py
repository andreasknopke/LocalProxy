"""
Unit-Tests fuer das Live-Streaming (Kategorie=local) inkl. Co-Worker-Stream-Inject.

Testet:
  1) Tool-Call-Akkumulation aus Streamed-Deltas (OpenAI-Format)
  2) _build_forward_tool_calls (VS-Code-taugliches Format)
  3) _coworker_task_preview (Task-Extraktion aus JSON-Args)
  4) _stream_backend_turn: reasoning/content fliessen live durch, Finish wird
     erfasst, Keepalives erscheinen bei langsamen Modellen
  5) _stream_local_events: Co-Worker-Delegation mit Stream-Inject
     (Delegations-Hinweis + Co-Worker-Antwort + Hauptmodell-Antwort)
  6) _stream_local_events: VS-Code-Tools werden durchgereicht, Fehlerfaelle
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT))


def _load_proxy_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("proxy_stream_test", REPO_ROOT / "proxy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["proxy_stream_test"] = module
    spec.loader.exec_module(module)
    return module


try:
    proxy = _load_proxy_module()
    HAS_PROXY = True
    SKIP_REASON = ""
except Exception as e:
    HAS_PROXY = False
    SKIP_REASON = f"proxy.py konnte nicht importiert werden: {e}"

pytestmark = pytest.mark.skipif(not HAS_PROXY, reason=SKIP_REASON)


def _collect_sse(agen) -> List[str]:
    """Sammelt alle Strings eines Async-Generators in eine Liste."""
    return asyncio.run(_collect_async(agen))


async def _collect_async(agen) -> List[str]:
    return [s async for s in agen]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Tool-Call-Akkumulation aus Streamed-Deltas
# ═══════════════════════════════════════════════════════════════════════════

def test_accumulate_tool_call_deltas():
    state: Dict[str, Any] = {"tool_calls": {}}
    # Typische OpenAI-Stream-Deltas: erst id+name, dann arguments in Teilen
    proxy._accumulate_stream_tool_calls(state, [
        {"index": 0, "id": "call_abc", "type": "function",
         "function": {"name": "ask_coworker", "arguments": ""}},
    ])
    proxy._accumulate_stream_tool_calls(state, [
        {"index": 0, "function": {"arguments": "{\"task\": \"Review"}},
    ])
    proxy._accumulate_stream_tool_calls(state, [
        {"index": 0, "function": {"arguments": " file X\"}"}},
    ])
    calls = proxy._finalize_stream_tool_calls(state)
    assert calls is not None
    assert len(calls) == 1
    tc = calls[0]
    assert tc["id"] == "call_abc"
    assert tc["function"]["name"] == "ask_coworker"
    assert json.loads(tc["function"]["arguments"]) == {"task": "Review file X"}
    assert tc["index"] == 0


def test_accumulate_multiple_tool_calls():
    state: Dict[str, Any] = {"tool_calls": {}}
    proxy._accumulate_stream_tool_calls(state, [
        {"index": 0, "id": "call_1", "function": {"name": "read_file", "arguments": "{}"}},
        {"index": 1, "id": "call_2", "function": {"name": "grep_search", "arguments": "{}"}},
    ])
    calls = proxy._finalize_stream_tool_calls(state)
    assert calls is not None
    assert [c["function"]["name"] for c in calls] == ["read_file", "grep_search"]
    # Reihenfolge nach Index
    assert [c["index"] for c in calls] == [0, 1]


def test_finalize_empty_returns_none():
    state: Dict[str, Any] = {"tool_calls": {}}
    assert proxy._finalize_stream_tool_calls(state) is None
    assert state["tool_calls"] == []


def test_finalize_skips_call_without_name():
    state: Dict[str, Any] = {"tool_calls": {0: {"id": "call_x", "function": {"name": "", "arguments": ""}}}}
    assert proxy._finalize_stream_tool_calls(state) is None


# ═══════════════════════════════════════════════════════════════════════════
# 2. _build_forward_tool_calls — VS-Code-Format
# ═══════════════════════════════════════════════════════════════════════════

def test_build_forward_tool_calls():
    calls = [{"index": 0, "id": "call_1", "type": "function",
              "function": {"name": "read_file", "arguments": '{"filePath": "a.py"}'}}]
    fwd = proxy._build_forward_tool_calls(calls)
    assert fwd[0]["index"] == 0
    assert fwd[0]["id"] == "call_1"
    assert fwd[0]["type"] == "function"
    assert fwd[0]["function"]["name"] == "read_file"
    assert json.loads(fwd[0]["function"]["arguments"]) == {"filePath": "a.py"}


def test_build_forward_tool_calls_stringifies_dict_args():
    calls = [{"index": 0, "id": "call_1", "type": "function",
              "function": {"name": "t", "arguments": {"a": 1}}}]
    fwd = proxy._build_forward_tool_calls(calls)
    assert isinstance(fwd[0]["function"]["arguments"], str)
    assert json.loads(fwd[0]["function"]["arguments"]) == {"a": 1}


# ═══════════════════════════════════════════════════════════════════════════
# 3. _coworker_task_preview
# ═══════════════════════════════════════════════════════════════════════════

def test_coworker_task_preview():
    calls = [{"id": "call_1", "function": {"name": "ask_coworker",
             "arguments": '{"task": "Review file X", "context": "code"}'}}]
    assert proxy._coworker_task_preview(calls) == "Review file X"


def test_coworker_task_preview_truncates():
    calls = [{"id": "call_1", "function": {"name": "ask_coworker",
             "arguments": '{"task": "' + "y" * 500 + '"}'}}]
    preview = proxy._coworker_task_preview(calls, max_chars=200)
    assert len(preview) <= 201  # 200 + "…"


def test_coworker_task_preview_fallback():
    calls = [{"id": "call_1", "function": {"name": "ask_coworker", "arguments": "not-json"}}]
    assert proxy._coworker_task_preview(calls) == "Sub-Task"


# ═══════════════════════════════════════════════════════════════════════════
# 4. _stream_backend_turn — Live-Durchreichung + Keepalive
# ═══════════════════════════════════════════════════════════════════════════

def test_backend_turn_streams_reasoning_and_content(monkeypatch):
    async def fake_single(body, category, def_idx, inject_hindsight=True):
        yield {"type": "chunk", "choice": {"delta": {"role": "assistant", "reasoning_content": "Ich denke "}, "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {"reasoning_content": "weiter"}, "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {"content": "Antwort"}, "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {}, "finish_reason": "stop"}}
        yield {"type": "done"}

    monkeypatch.setattr(proxy, "_stream_single_model_events", fake_single)
    state: Dict[str, Any] = {"stream_id": "test-stream", "role_sent": False}
    sse = _collect_sse(proxy._stream_backend_turn({"messages": []}, "local", None, state))

    joined = "\n".join(sse)
    assert "Ich denke" in joined
    assert "weiter" in joined
    assert "Antwort" in joined
    assert state["content"] == "Antwort"
    assert state["reasoning"] == "Ich denke weiter"
    assert state["finish_reason"] == "stop"
    assert state["role_sent"] is True
    assert state["all_failed"] is False


def test_backend_turn_emits_keepalive_when_backend_slow(monkeypatch):
    async def fake_single(body, category, def_idx, inject_hindsight=True):
        await asyncio.sleep(0.25)  # laenger als der (gekuerzte) Keepalive-Intervall
        yield {"type": "chunk", "choice": {"delta": {"content": "A"}, "finish_reason": None}}
        await asyncio.sleep(0.25)
        yield {"type": "done"}

    monkeypatch.setattr(proxy, "_stream_single_model_events", fake_single)
    monkeypatch.setattr(proxy, "_STREAM_KEEPALIVE_INTERVAL", 0.05)
    state: Dict[str, Any] = {"stream_id": "test-stream", "role_sent": False}
    sse = _collect_sse(proxy._stream_backend_turn({"messages": []}, "local", None, state))

    assert any(s.strip() == ": keepalive" for s in sse)
    assert state["content"] == "A"


# ═══════════════════════════════════════════════════════════════════════════
# 4b. Reasoning-Mapping: verschiedene Backend-Strukturen + <think>-Tags
# ═══════════════════════════════════════════════════════════════════════════

def test_extract_reasoning_from_delta_string():
    assert proxy._extract_reasoning_from_delta({"reasoning_content": "abc"}) == "abc"
    assert proxy._extract_reasoning_from_delta({"reasoning": "abc"}) == "abc"
    assert proxy._extract_reasoning_from_delta({"thinking": "abc"}) == "abc"
    assert proxy._extract_reasoning_from_delta({"content": "abc"}) is None
    assert proxy._extract_reasoning_from_delta({}) is None


def test_extract_reasoning_from_delta_list_parts():
    delta = {"reasoning": [{"type": "text", "text": "denke "}, {"type": "text", "text": "weiter"}]}
    assert proxy._extract_reasoning_from_delta(delta) == "denke weiter"


def test_extract_reasoning_from_delta_dict():
    delta = {"reasoning": {"content": "ueberlege"}}
    assert proxy._extract_reasoning_from_delta(delta) == "ueberlege"


def test_split_think_content_complete():
    content = "Vorwort\n<think>ueberlege</think>\nAntwort"
    clean, reasoning = proxy._split_think_content(content)
    assert reasoning == "ueberlege"
    assert "think" not in clean
    assert "Vorwort" in clean and "Antwort" in clean


def test_split_think_chunk_across_boundaries():
    state: Dict[str, Any] = {"in_think": False, "pending": ""}
    # Tag ueber Chunks verteilt, Inhalt danach
    r1, c1 = proxy._split_think_chunk("Vorwort <thi", state)
    assert c1 == "Vorwort "
    assert r1 is None
    r2, c2 = proxy._split_think_chunk("nk>ueberleg", state)
    assert r2 == "ueberleg"
    assert c2 is None
    r3, c3 = proxy._split_think_chunk("en</think>Antwort", state)
    assert r3 == "en"
    assert c3 == "Antwort"
    assert state["in_think"] is False
    assert state["pending"] == ""


def test_backend_turn_maps_think_tags_to_reasoning(monkeypatch):
    """Qwen3/vLLM liefert Reasoning als <think>...</think> im content — der
    Proxy muss das als reasoning_content an VS Code mappen (Issue 2)."""
    async def fake_single(body, category, def_idx, inject_hindsight=True):
        yield {"type": "chunk", "choice": {"delta": {"content": "Start <thi"}, "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {"content": "nk>ueberleg"}, "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {"content": "en</think>Antwort"}, "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {}, "finish_reason": "stop"}}
        yield {"type": "done"}

    monkeypatch.setattr(proxy, "_stream_single_model_events", fake_single)
    state: Dict[str, Any] = {"stream_id": "test-stream", "role_sent": False}
    sse = _collect_sse(proxy._stream_backend_turn({"messages": []}, "local", None, state))

    joined = "\n".join(sse)
    assert "ueberleg" in joined
    assert "en" in joined
    assert '"reasoning_content"' in joined
    assert "<think>" not in joined
    assert "Antwort" in joined
    assert state["reasoning"] == "ueberlegen"
    assert state["content"] == "Start Antwort"
    assert state["has_explicit_reasoning"] is False


def test_backend_turn_prefers_explicit_reasoning_field(monkeypatch):
    """Wenn das Backend reasoning_content liefert, wird content NICHT nochmal
    als <think> geparst (kein Doppel-Reasoning)."""
    async def fake_single(body, category, def_idx, inject_hindsight=True):
        yield {"type": "chunk", "choice": {"delta": {"reasoning_content": "explizit"}, "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {"content": "Antwort <think>fake</think>"}, "finish_reason": None}}
        yield {"type": "done"}

    monkeypatch.setattr(proxy, "_stream_single_model_events", fake_single)
    state: Dict[str, Any] = {"stream_id": "test-stream", "role_sent": False}
    sse = _collect_sse(proxy._stream_backend_turn({"messages": []}, "local", None, state))

    joined = "\n".join(sse)
    assert state["reasoning"] == "explizit"
    assert state["content"] == "Antwort <think>fake</think>"
    assert state["has_explicit_reasoning"] is True


# ═══════════════════════════════════════════════════════════════════════════
# 4c. Co-Worker-Phase — Reasoning live streamen (Issue 1)
# ═══════════════════════════════════════════════════════════════════════════

def test_coworker_phase_streams_reasoning_live(monkeypatch):
    """Das reasoning_content des Co-Workers wird als eigener Reasoning-Context
    live an VS Code gestreamt; die Antwort landet in den tool results."""
    async def fake_single(body, category, def_idx, inject_hindsight=True):
        assert category == "coworker"
        assert inject_hindsight is False
        yield {"type": "chunk", "choice": {"delta": {"reasoning_content": "Co-Worker denkt "}, "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {"reasoning_content": "weiter"}, "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {"content": "Ergebnis: mach X"}, "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {}, "finish_reason": "stop"}}
        yield {"type": "done"}

    monkeypatch.setattr(proxy, "_stream_single_model_events", fake_single)
    calls = [{"id": "call_1", "function": {"name": "ask_coworker",
             "arguments": '{"task": "T", "context": "C"}'}}]
    coworker_state: Dict[str, Any] = {"stream_id": "test-stream", "model": "local-model"}
    sse = _collect_sse(proxy._stream_coworker_phase(calls, coworker_state))
    joined = "\n".join(sse)

    assert "Co-Worker denkt" in joined
    assert "weiter" in joined
    assert '"reasoning_content"' in joined

    results = coworker_state["coworker_results"]
    assert len(results) == 1
    assert results[0]["role"] == "tool"
    assert results[0]["content"] == "Ergebnis: mach X"
    assert results[0]["reasoning_content"] == "Co-Worker denkt weiter"


def test_local_events_strips_coworker_reasoning_from_history(monkeypatch):
    """reasoning_content der Co-Worker-tool-results wird live als
    Reasoning-Context gestreamt, aber NICHT in die History geschrieben
    (kein unbekanntes Feld am Backend des Hauptmodells)."""
    calls = {"call_count": 0}

    async def fake_backend_turn(body, category, force_start_idx, state):
        calls["call_count"] += 1
        if calls["call_count"] == 1:
            state.update({
                "content": "", "reasoning": "", "tool_calls": {
                    0: {"id": "call_1", "type": "function",
                        "function": {"name": "ask_coworker",
                                     "arguments": '{"task": "T"}'}},
                },
                "finish_reason": "tool_calls", "all_failed": False,
                "mid_stream_error": None, "error_content": None,
                "def_idx": 0, "model": "local-model",
            })
            yield proxy._format_openai_stream_chunk("local-model", content="Delegiere",
                                                    include_role=True, chunk_id="test")
        else:
            state.update({
                "content": "Endergebnis", "reasoning": "", "tool_calls": {},
                "finish_reason": "stop", "all_failed": False,
                "mid_stream_error": None, "error_content": None,
                "def_idx": 0, "model": "local-model",
            })
            yield proxy._format_openai_stream_chunk("local-model", content="Endergebnis",
                                                    include_role=True, chunk_id="test")

    async def fake_single(body, category, def_idx, inject_hindsight=True):
        yield {"type": "chunk", "choice": {"delta": {"reasoning_content": "denkt"}, "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {"content": "Antwort"}, "finish_reason": None}}
        yield {"type": "done"}

    async def noop_retain(body, content):
        pass

    monkeypatch.setattr(proxy, "_stream_backend_turn", fake_backend_turn)
    monkeypatch.setattr(proxy, "_stream_single_model_events", fake_single)
    monkeypatch.setattr(proxy._hindsight, "retain_async", noop_retain)

    body: Dict[str, Any] = {"messages": [{"role": "user", "content": "mach was\n--local"}]}
    sse = _collect_sse(proxy._stream_local_events(body, "local", None))
    joined = "\n".join(sse)

    # Reasoning live als Reasoning-Context gestreamt + Antwort-Inject
    assert "denkt" in joined
    assert '"reasoning_content"' in joined
    assert "[Proxy] Co-Worker-Antwort:" in joined
    # History ohne reasoning_content
    tool_msgs = [m for m in body["messages"] if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "reasoning_content" not in tool_msgs[0]
    assert tool_msgs[0]["content"] == "Antwort"


# ═══════════════════════════════════════════════════════════════════════════
# 5. _stream_local_events — Co-Worker-Delegation mit Stream-Inject
# ═══════════════════════════════════════════════════════════════════════════

def test_local_events_coworker_stream_inject(monkeypatch):
    """Turn 1 liefert ask_coworker → Inject-Hinweise, Turn 2 liefert die finale
    Antwort des Hauptmodells."""
    calls = {"call_count": 0}

    async def fake_backend_turn(body, category, force_start_idx, state):
        calls["call_count"] += 1
        if calls["call_count"] == 1:
            state.update({
                "content": "", "reasoning": "Delegiere das",
                "tool_calls": {
                    0: {"id": "call_1", "type": "function",
                        "function": {"name": "ask_coworker",
                                     "arguments": '{"task": "Review file X"}'}},
                },
                "finish_reason": "tool_calls", "all_failed": False,
                "mid_stream_error": None, "error_content": None,
                "def_idx": 0, "model": "local-model",
            })
            yield proxy._format_openai_stream_chunk("local-model", content="Delegiere das",
                                                    include_role=True, chunk_id="test")
        else:
            state.update({
                "content": "Hier ist das Endergebnis", "reasoning": "Verarbeite Antwort",
                "tool_calls": {}, "finish_reason": "stop", "all_failed": False,
                "mid_stream_error": None, "error_content": None,
                "def_idx": 0, "model": "local-model",
            })
            yield proxy._format_openai_stream_chunk("local-model", reasoning_content="Verarbeite Antwort",
                                                    include_role=True, chunk_id="test")
            yield proxy._format_openai_stream_chunk("local-model", content="Hier ist das Endergebnis",
                                                    chunk_id="test")

    async def fake_coworker_phase(coworker_calls_norm, coworker_state):
        coworker_state["coworker_results"] = [
            {"role": "tool", "tool_call_id": "call_1", "name": "ask_coworker",
             "content": "Co-Worker sagt: mach X"}
        ]
        yield ": keepalive\n\n"

    async def noop_retain(body, content):
        pass

    monkeypatch.setattr(proxy, "_stream_backend_turn", fake_backend_turn)
    monkeypatch.setattr(proxy, "_stream_coworker_phase", fake_coworker_phase)
    monkeypatch.setattr(proxy._hindsight, "retain_async", noop_retain)

    body: Dict[str, Any] = {"messages": [{"role": "user", "content": "mach was\n--local"}]}
    sse = _collect_sse(proxy._stream_local_events(body, "local", None))
    joined = "\n".join(sse)

    # 1) Delegations-Hinweis mit Task-Preview
    assert "[Proxy] Delegation an Co-Worker: Review file X" in joined
    # 2) Co-Worker-Antwort sichtbar
    assert "[Proxy] Co-Worker-Antwort:" in joined
    assert "Co-Worker sagt: mach X" in joined
    # 3) Hauptmodell verarbeitet (live) — finale Antwort + Finish
    assert "Verarbeite Antwort" in joined
    assert "Hier ist das Endergebnis" in joined
    assert '"finish_reason": "stop"' in joined

    # History wurde korrekt erweitert (assistant tool_calls + tool result)
    msgs = body["messages"]
    assert len(msgs) == 3
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "ask_coworker"
    assert msgs[2]["role"] == "tool"
    assert "Co-Worker sagt: mach X" in msgs[2]["content"]


def test_local_events_forwards_vscode_tools(monkeypatch):
    """Turn liefert read_file (kein ask_coworker) → Tool-Calls werden an
    Copilot durchgereicht (finish_reason=tool_calls)."""

    async def fake_backend_turn(body, category, force_start_idx, state):
        state.update({
            "content": "", "reasoning": "",
            "tool_calls": {
                0: {"id": "call_read", "type": "function",
                    "function": {"name": "read_file", "arguments": '{"filePath": "a.py"}'}},
            },
            "finish_reason": "tool_calls", "all_failed": False,
            "mid_stream_error": None, "error_content": None,
            "def_idx": 0, "model": "local-model",
        })
        yield ": keepalive\n\n"

    monkeypatch.setattr(proxy, "_stream_backend_turn", fake_backend_turn)
    body: Dict[str, Any] = {"messages": [{"role": "user", "content": "lies a.py\n--local"}]}
    sse = _collect_sse(proxy._stream_local_events(body, "local", None))
    joined = "\n".join(sse)

    assert "read_file" in joined
    assert '"finish_reason": "tool_calls"' in joined
    # Keine Delegation, keine History-Manipulation
    assert "[Proxy] Delegation" not in joined
    assert len(body["messages"]) == 1


def test_local_events_all_failed(monkeypatch):
    async def fake_backend_turn(body, category, force_start_idx, state):
        state.update({
            "content": "", "reasoning": "", "tool_calls": {}, "finish_reason": None,
            "all_failed": True, "mid_stream_error": None,
            "error_content": "Modell down", "def_idx": 0, "model": "local-model",
        })
        yield ": keepalive\n\n"

    monkeypatch.setattr(proxy, "_stream_backend_turn", fake_backend_turn)
    sse = _collect_sse(proxy._stream_local_events({"messages": []}, "local", None))
    joined = "\n".join(sse)
    assert "[Proxy: Stream-Fehler]" in joined
    assert "Modell down" in joined
    assert '"finish_reason": "stop"' in joined


def test_local_events_mid_stream_error(monkeypatch):
    async def fake_backend_turn(body, category, force_start_idx, state):
        state.update({
            "content": "teil", "reasoning": "", "tool_calls": {}, "finish_reason": None,
            "all_failed": False, "mid_stream_error": "Connection reset",
            "error_content": None, "def_idx": 0, "model": "local-model",
        })
        yield ": keepalive\n\n"

    monkeypatch.setattr(proxy, "_stream_backend_turn", fake_backend_turn)
    sse = _collect_sse(proxy._stream_local_events({"messages": []}, "local", None))
    joined = "\n".join(sse)
    assert "[Proxy: Stream abgebrochen]" in joined
    assert "Connection reset" in joined
