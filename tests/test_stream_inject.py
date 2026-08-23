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
import time

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


def test_backend_turn_reasoning_cap_stops_and_injects_note(monkeypatch):
    """Reasoning-Cap: ab REASONING_CAP_CHARS wird Reasoning verworfen und
    einmalig eine Abschluss-Aufforderung als content injiziert."""
    monkeypatch.setattr(proxy, "REASONING_CAP_CHARS", 10)

    async def fake_single(body, category, def_idx, inject_hindsight=True):
        yield {"type": "chunk", "choice": {"delta": {"reasoning_content": "0123456789"}, "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {"reasoning_content": "XX"}, "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {"reasoning_content": "YY"}, "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {"content": "Finale Antwort"}, "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {}, "finish_reason": "stop"}}
        yield {"type": "done"}

    monkeypatch.setattr(proxy, "_stream_single_model_events", fake_single)
    state: Dict[str, Any] = {"stream_id": "test-stream", "role_sent": False}
    sse = _collect_sse(proxy._stream_backend_turn({"messages": []}, "local", None, state))

    joined = "\n".join(sse)
    # Reasoning wurde bis zum Cap durchgereicht (die ersten 10 Zeichen),
    # danach verworfen: "XX"/"YY" duerfen NICHT als reasoning_content erscheinen.
    assert '"reasoning_content": "0123456789"' in joined
    assert '"reasoning_content": "XX"' not in joined
    assert '"reasoning_content": "YY"' not in joined
    # Abschluss-Aufforderung wurde als content injiziert (genau einmal).
    assert joined.count("Reasoning-Limit erreicht") == 1
    # content danach fliesst normal weiter.
    assert "Finale Antwort" in joined
    assert state["content"].endswith("Finale Antwort")
    # state.reasoning haelt den VOLLEN internen Text (fuer Reporting), auch wenn
    # die Deltas danach nicht mehr an den Client gingen.
    assert state["reasoning"] == "0123456789XXYY"


def test_backend_turn_reasoning_cap_disabled_by_default(monkeypatch):
    """REASONING_CAP_CHARS=0 (Default) -> kein Cap, alles fliesst durch."""
    monkeypatch.setattr(proxy, "REASONING_CAP_CHARS", 0)

    async def fake_single(body, category, def_idx, inject_hindsight=True):
        yield {"type": "chunk", "choice": {"delta": {"reasoning_content": "0123456789"}, "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {"reasoning_content": "XX"}, "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {"content": "A"}, "finish_reason": None}}
        yield {"type": "done"}

    monkeypatch.setattr(proxy, "_stream_single_model_events", fake_single)
    state: Dict[str, Any] = {"stream_id": "test-stream", "role_sent": False}
    sse = _collect_sse(proxy._stream_backend_turn({"messages": []}, "local", None, state))

    joined = "\n".join(sse)
    assert '"reasoning_content": "XX"' in joined
    assert "Reasoning-Limit erreicht" not in joined


def test_backend_turn_reasoning_cap_restart_mode(monkeypatch):
    """Restart-Mode: Backend-Stream wird beim Cap ABGEBROCHEN und ein
    Folgeturn mit Anti-Loop-Hinweis gestartet — Antwort kommt trotzdem."""
    monkeypatch.setattr(proxy, "REASONING_CAP_CHARS", 10)
    monkeypatch.setattr(proxy, "REASONING_CAP_MODE", "restart")
    monkeypatch.setattr(proxy, "REASONING_CAP_MAX_RESTARTS", 1)

    captured_messages = []

    async def fake_single(body, category, def_idx, inject_hindsight=True):
        captured_messages.append([dict(m) for m in (body.get("messages") or [])])
        if len(captured_messages) == 1:
            # Erste Runde: endlos Reasoning, ueber das Cap hinaus.
            for _ in range(3):
                yield {"type": "chunk", "choice": {"delta": {"reasoning_content": "0123456789"}, "finish_reason": None}}
            # Haengt danach fest (kein done) — wird per Restart abgebrochen.
            await asyncio.Event().wait()
            yield {"type": "done"}
        else:
            # Zweite Runde: kurze finale Antwort.
            yield {"type": "chunk", "choice": {"delta": {"content": "Endlich Antwort"}, "finish_reason": None}}
            yield {"type": "chunk", "choice": {"delta": {}, "finish_reason": "stop"}}
            yield {"type": "done"}

    monkeypatch.setattr(proxy, "_stream_single_model_events", fake_single)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    state: Dict[str, Any] = {"stream_id": "test-stream", "role_sent": False}
    sse = _collect_sse(proxy._stream_backend_turn(body, "local", None, state))

    joined = "\n".join(sse)
    # Zwei Aufrufe = Restart hat stattgefunden.
    assert len(captured_messages) == 2
    # Anti-Loop-Hinweis wurde vor dem Folgeturn an die History angehaengt.
    assert captured_messages[1][-1]["role"] == "user"
    assert "zu lange nachgedacht" in captured_messages[1][-1]["content"]
    # Der erste (abgebrochene) Turn hatte den Hinweis noch NICHT.
    assert all("zu lange nachgedacht" not in str(m.get("content", ""))
               for m in captured_messages[0])
    # Finale Antwort kommt durch.
    assert "Endlich Antwort" in joined
    assert state["content"].endswith("Endlich Antwort")


def test_backend_turn_reasoning_cap_restart_exhausted_aborts(monkeypatch):
    """Restart-Mode mit 0 Restarts: Turn endet nach Abbruch (kein Neustart)."""
    monkeypatch.setattr(proxy, "REASONING_CAP_CHARS", 10)
    monkeypatch.setattr(proxy, "REASONING_CAP_MODE", "restart")
    monkeypatch.setattr(proxy, "REASONING_CAP_MAX_RESTARTS", 0)

    call_count = {"n": 0}

    async def fake_single(body, category, def_idx, inject_hindsight=True):
        call_count["n"] += 1
        for _ in range(3):
            yield {"type": "chunk", "choice": {"delta": {"reasoning_content": "0123456789"}, "finish_reason": None}}
        await asyncio.Event().wait()
        yield {"type": "done"}

    monkeypatch.setattr(proxy, "_stream_single_model_events", fake_single)
    state: Dict[str, Any] = {"stream_id": "test-stream", "role_sent": False}
    sse = _collect_sse(proxy._stream_backend_turn({"messages": []}, "local", None, state))

    joined = "\n".join(sse)
    # Nur EIN Aufruf (kein Restart), kein finaler content.
    assert call_count["n"] == 1
    assert "Endlich" not in joined


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

    # Reasoning streamt live als eigener Context
    assert "Co-Worker denkt" in joined
    assert "weiter" in joined
    assert '"reasoning_content"' in joined
    # Antwort streamt LIVE token-fuer-token (Header + Content als Content-Chunks)
    assert "[Proxy] Co-Worker-Antwort:" in joined
    assert "Ergebnis: mach X" in joined
    assert "<think>" not in joined

    results = coworker_state["coworker_results"]
    assert len(results) == 1
    assert results[0]["role"] == "tool"
    assert results[0]["content"] == "Ergebnis: mach X"
    assert results[0]["reasoning_content"] == "Co-Worker denkt weiter"


def test_coworker_phase_streams_think_tags_live(monkeypatch):
    """Co-Worker mit <think>-Tags im content (vLLM Qwen3): Reasoning wird als
    reasoning_content gemappt und gestreamt, Antwort-Content streamt live."""
    async def fake_single(body, category, def_idx, inject_hindsight=True):
        yield {"type": "chunk", "choice": {"delta": {"content": "<think>ueberleg</think>Ergebnis"}, "finish_reason": None}}
        yield {"type": "done"}

    monkeypatch.setattr(proxy, "_stream_single_model_events", fake_single)
    calls = [{"id": "call_1", "function": {"name": "ask_coworker",
             "arguments": '{"task": "T"}'}}]
    coworker_state: Dict[str, Any] = {"stream_id": "test-stream", "model": "local-model"}
    sse = _collect_sse(proxy._stream_coworker_phase(calls, coworker_state))
    joined = "\n".join(sse)

    assert "ueberleg" in joined
    assert '"reasoning_content"' in joined
    assert "<think>" not in joined
    assert "Ergebnis" in joined
    assert "[Proxy] Co-Worker-Antwort:" in joined

    results = coworker_state["coworker_results"]
    assert results[0]["content"] == "Ergebnis"
    assert results[0]["reasoning_content"] == "ueberleg"


def test_coworker_phase_streams_error(monkeypatch):
    """Fehler beim Co-Worker werden ebenfalls live sichtbar gestreamt."""
    async def fake_single(body, category, def_idx, inject_hindsight=True):
        yield {"type": "error", "status_code": 500, "content": "Model kaputt",
               "trigger_fallback": True}

    monkeypatch.setattr(proxy, "_stream_single_model_events", fake_single)
    calls = [{"id": "call_1", "function": {"name": "ask_coworker",
             "arguments": '{"task": "T"}'}}]
    coworker_state: Dict[str, Any] = {"stream_id": "test-stream", "model": "local-model"}
    sse = _collect_sse(proxy._stream_coworker_phase(calls, coworker_state))
    joined = "\n".join(sse)

    assert "[Proxy] Co-Worker-Antwort:" in joined
    assert "Model kaputt" in joined
    results = coworker_state["coworker_results"]
    assert len(results) == 1
    assert results[0]["role"] == "tool"
    assert "Model kaputt" in results[0]["content"]


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
        # Realer Pfad: Header + Antwort-Content streamen LIVE waehrend der Phase
        yield proxy._format_openai_stream_chunk("local-model", content="\n\n[Proxy] Co-Worker-Antwort:\n",
                                                include_role=False, chunk_id="test")
        yield proxy._format_openai_stream_chunk("local-model", content="Co-Worker sagt: mach X",
                                                include_role=False, chunk_id="test")
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


# ═══════════════════════════════════════════════════════════════════════════
# 6. _stream_local_events — Fork-Join: dispatch / collect (v3.2)
# ═══════════════════════════════════════════════════════════════════════════

def _bg_state_defaults() -> Dict[str, Any]:
    """Gueltiger Default-Status fuer _stream_backend_turn-Mocks."""
    return {
        "content": "", "reasoning": "", "tool_calls": {}, "finish_reason": None,
        "all_failed": False, "mid_stream_error": None, "error_content": None,
        "def_idx": 0, "model": "local-model",
    }


def test_local_events_dispatch_streams_note_and_history(monkeypatch):
    """Turn 1: dispatch_coworker → dispatched-Chunk + tool-result(task_id);
    Turn 2: finale Antwort."""
    calls = {"call_count": 0}

    async def fake_backend_turn(body, category, force_start_idx, state):
        calls["call_count"] += 1
        state.update(_bg_state_defaults())
        if calls["call_count"] == 1:
            state["tool_calls"] = {
                0: {"id": "call_d1", "type": "function",
                    "function": {"name": "dispatch_coworker",
                                 "arguments": '{"task": "Review file X"}'}},
            }
            state["finish_reason"] = "tool_calls"
            yield proxy._format_openai_stream_chunk(
                "local-model", content="Dispatch im Hintergrund", include_role=True,
                chunk_id="test")
        else:
            state["content"] = "Endergebnis nach Fork"
            state["finish_reason"] = "stop"
            yield proxy._format_openai_stream_chunk(
                "local-model", content="Endergebnis nach Fork", include_role=True,
                chunk_id="test")

    async def fake_bg(task, args):
        task.status = "done"
        task.result = "bg-result"
        task.finished_at = time.time()

    monkeypatch.setattr(proxy, "_stream_backend_turn", fake_backend_turn)
    monkeypatch.setattr(proxy, "_run_bg_coworker_task", fake_bg)
    monkeypatch.setattr(proxy, "_COWORKER_BG_TASKS", {})
    monkeypatch.setattr(proxy, "COWORKER_FORK_JOIN", True)
    monkeypatch.setattr(proxy, "COWORKER_ENABLED", True)
    monkeypatch.setattr(proxy, "COWORKER_DISPATCH_CAP", 12)
    monkeypatch.setattr(proxy, "COWORKER_MAX_DELEGATIONS", 2)

    body: Dict[str, Any] = {"messages": [{"role": "user", "content": "dispatch\n--local"}]}
    sse = _collect_sse(proxy._stream_local_events(body, "local", None))
    joined = "\n".join(sse)

    # 1) Live-Hinweis mit task_id + preview
    assert "[Proxy] Co-Worker dispatched:" in joined
    assert "Review file X" in joined
    assert "cw_" in joined
    # 2) Nächste Runde: finale Antwort
    assert "Endergebnis nach Fork" in joined
    assert '"finish_reason": "stop"' in joined
    # 3) History: assistant tool_calls + tool-mini-result mit task_id
    msgs = body["messages"]
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert payload_status_ok(tool_msgs[0])


def payload_status_ok(tool_msg):
    data = json.loads(tool_msg["content"])
    return (data.get("status") == "dispatched"
            and str(data.get("task_id", "")).startswith("cw_")
            and tool_msg.get("name") == "dispatch_coworker")


def test_local_events_collect_streams_summary(monkeypatch):
    """Turn 1: collect_coworker → Sammel-Chunk + JSON-summary tool-result;
    Turn 2: finale Antwort."""
    calls = {"call_count": 0}

    async def fake_backend_turn(body, category, force_start_idx, state):
        calls["call_count"] += 1
        state.update(_bg_state_defaults())
        if calls["call_count"] == 1:
            state["tool_calls"] = {
                0: {"id": "call_c1", "type": "function",
                    "function": {"name": "collect_coworker",
                                 "arguments": '{"task_ids": ["cw_1"]}'}},
            }
            state["finish_reason"] = "tool_calls"
            yield proxy._format_openai_stream_chunk(
                "local-model", content="Sammle", include_role=True, chunk_id="test")
        else:
            state["content"] = "alles gesammelt"
            state["finish_reason"] = "stop"
            yield proxy._format_openai_stream_chunk(
                "local-model", content="alles gesammelt", include_role=True,
                chunk_id="test")

    async def fake_await(task_ids, timeout_seconds=600.0):
        assert task_ids == ["cw_1"]
        return [{"task_id": "cw_1", "status": "done", "result": "ergebnis-1"}]

    monkeypatch.setattr(proxy, "_stream_backend_turn", fake_backend_turn)
    monkeypatch.setattr(proxy, "_await_bg_tasks", fake_await)
    monkeypatch.setattr(proxy, "_COWORKER_BG_TASKS", {})
    monkeypatch.setattr(proxy, "COWORKER_FORK_JOIN", True)
    monkeypatch.setattr(proxy, "COWORKER_ENABLED", True)
    monkeypatch.setattr(proxy, "COWORKER_MAX_DELEGATIONS", 2)

    body: Dict[str, Any] = {"messages": [{"role": "user", "content": "sammle\n--local"}]}
    sse = _collect_sse(proxy._stream_local_events(body, "local", None))
    joined = "\n".join(sse)

    # 1) Live-Sammel-Hinweis
    assert "[Proxy] Sammle Co-Worker-Ergebnisse" in joined
    # 2) Nächste Runde: finale Antwort
    assert "alles gesammelt" in joined
    assert '"finish_reason": "stop"' in joined
    # 3) History: assistant tool_calls + tool-result mit summaries
    msgs = body["messages"]
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["name"] == "collect_coworker"
    summaries = json.loads(tool_msgs[0]["content"])
    assert summaries[0]["task_id"] == "cw_1"
    assert summaries[0]["result"] == "ergebnis-1"


def test_local_events_dispatch_with_vstool_forwards(monkeypatch):
    """dispatch + read_file im selben Turn: read_file wird an Client
    durchgereicht (finish_reason=tool_calls), dispatch intern."""
    async def fake_backend_turn(body, category, force_start_idx, state):
        state.update(_bg_state_defaults())
        state["tool_calls"] = {
            0: {"id": "call_d1", "type": "function",
                "function": {"name": "dispatch_coworker",
                             "arguments": '{"task": "Hintergrundjob"}'}},
            1: {"id": "call_r1", "type": "function",
                "function": {"name": "read_file",
                             "arguments": '{"filePath": "a.py"}'}},
        }
        state["finish_reason"] = "tool_calls"
        yield ": keepalive\n\n"

    async def fake_bg(task, args):
        task.status = "done"
        task.result = "ok"
        task.finished_at = time.time()

    monkeypatch.setattr(proxy, "_stream_backend_turn", fake_backend_turn)
    monkeypatch.setattr(proxy, "_run_bg_coworker_task", fake_bg)
    monkeypatch.setattr(proxy, "_COWORKER_BG_TASKS", {})
    monkeypatch.setattr(proxy, "COWORKER_FORK_JOIN", True)
    monkeypatch.setattr(proxy, "COWORKER_ENABLED", True)
    monkeypatch.setattr(proxy, "COWORKER_DISPATCH_CAP", 12)
    monkeypatch.setattr(proxy, "COWORKER_MAX_DELEGATIONS", 2)

    body: Dict[str, Any] = {"messages": [{"role": "user", "content": "dispatch+lies\n--local"}]}
    sse = _collect_sse(proxy._stream_local_events(body, "local", None))
    joined = "\n".join(sse)

    assert "[Proxy] Co-Worker dispatched:" in joined
    # read_file raus, dispatch intern beantwortet
    assert "read_file" in joined
    assert '"finish_reason": "tool_calls"' in joined
    # History enthält dispatch-mini-result
    tool_msgs = [m for m in body["messages"] if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["name"] == "dispatch_coworker"


def test_local_events_collect_with_vstools_hint(monkeypatch):
    """collect + read_file: read_file wird NICHT ausgeführt, Hinweis-Message
    + nächste Runde."""
    calls = {"call_count": 0}

    async def fake_backend_turn(body, category, force_start_idx, state):
        calls["call_count"] += 1
        state.update(_bg_state_defaults())
        if calls["call_count"] == 1:
            state["tool_calls"] = {
                0: {"id": "call_c1", "type": "function",
                    "function": {"name": "collect_coworker", "arguments": '{}'}},
                1: {"id": "call_r1", "type": "function",
                    "function": {"name": "read_file",
                                 "arguments": '{"filePath": "a.py"}'}},
            }
            state["finish_reason"] = "tool_calls"
            yield ": keepalive\n\n"
        else:
            state["content"] = "fertig nach hinweis"
            state["finish_reason"] = "stop"
            yield proxy._format_openai_stream_chunk(
                "local-model", content="fertig nach hinweis", include_role=True,
                chunk_id="test")

    async def fake_await(task_ids, timeout_seconds=600.0):
        return [{"task_id": "cw_1", "status": "done", "result": "r"}]

    monkeypatch.setattr(proxy, "_stream_backend_turn", fake_backend_turn)
    monkeypatch.setattr(proxy, "_await_bg_tasks", fake_await)
    monkeypatch.setattr(proxy, "_COWORKER_BG_TASKS", {})
    monkeypatch.setattr(proxy, "COWORKER_FORK_JOIN", True)
    monkeypatch.setattr(proxy, "COWORKER_ENABLED", True)
    monkeypatch.setattr(proxy, "COWORKER_MAX_DELEGATIONS", 3)

    body: Dict[str, Any] = {"messages": [{"role": "user", "content": "sammel+lies\n--local"}]}
    sse = _collect_sse(proxy._stream_local_events(body, "local", None))
    joined = "\n".join(sse)

    assert "[Proxy] Sammle Co-Worker-Ergebnisse" in joined
    assert "fertig nach hinweis" in joined
    # Hinweis-Message in der History (andere Tools nicht ausgefuehrt)
    hint = [m for m in body["messages"] if m.get("role") == "user"
            and "nicht ausgefuehrt" in m.get("content", "")]
    assert len(hint) == 1
    # tool-result der collect-Abfrage
    tool_msgs = [m for m in body["messages"] if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["name"] == "collect_coworker"
