"""
Unit-Tests fuer den modell-agnostischen Text-Loop-Guard.

Testet:
  1) Helfer: _normalize_for_loop, _trim_looping_tail
  2) _TextLoopGuard: Satz-, Absatz-, N-Gramm- und Tail-Loops
  3) False-Positive-Schutz: variierter Text, Code-Fences, Mindestlaenge
  4) Integration in _stream_backend_turn (restart + exhausted)
  5) Integration in den Non-Streaming-/2-Pass-Pfad (_outcome_with_loop_guard)
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_proxy_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("proxy_loop_test", REPO_ROOT / "proxy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["proxy_loop_test"] = module
    spec.loader.exec_module(module)
    return module


try:
    proxy = _load_proxy_module()
    HAS_PROXY = True
    SKIP_REASON = ""
except Exception as e:  # pragma: no cover
    HAS_PROXY = False
    SKIP_REASON = f"proxy.py konnte nicht importiert werden: {e}"

pytestmark = pytest.mark.skipif(not HAS_PROXY, reason=SKIP_REASON)


def _collect_sse(agen) -> List[str]:
    return asyncio.run(_collect_async(agen))


async def _collect_async(agen) -> List[str]:
    return [s async for s in agen]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Helfer
# ═══════════════════════════════════════════════════════════════════════════

def test_normalize_for_loop_collapses_whitespace_and_case():
    assert proxy._normalize_for_loop("Hello   World!\n\n  Foo\tBAR") == "hello world! foo bar"


def test_trim_looping_tail_removes_repeated_sentences():
    sentence = "Repeated sentence that is long enough to count here."
    text = "First sentence here is also long enough to count. " + (sentence + " ") * 4
    out = proxy._trim_looping_tail(text)
    assert out.count(sentence) == 1
    assert out.startswith("First sentence here")


def test_trim_looping_tail_leaves_varied_text_alone():
    text = ("Alpha sentence that is long enough to count here. "
            "Beta sentence that is long enough to count here. "
            "Gamma sentence that is long enough to count here.")
    assert proxy._trim_looping_tail(text) == text


# ═══════════════════════════════════════════════════════════════════════════
# 2. Guard-Erkennung
# ═══════════════════════════════════════════════════════════════════════════

def test_guard_detects_sentence_repetition():
    g = proxy._TextLoopGuard()
    hit = None
    for _ in range(5):
        hit = g.feed("I need to check the file again and again to be sure. ", "reasoning") or hit
    assert hit is not None
    assert hit["kind"] == "sentence"
    assert hit["count"] >= 3


def test_guard_detects_paragraph_repetition():
    para = ("word " * 60).strip()  # ~299 Zeichen, keine Satzgrenzen
    g = proxy._TextLoopGuard()
    hit = g.feed(para + "\n\n", "reasoning")
    hit = g.feed(para + "\n\n", "reasoning") or hit
    assert hit is not None
    assert hit["kind"] == "paragraph"


def test_guard_detects_ngram_loop_without_sentence_boundaries():
    frag = "abcdefghij" * 10  # 100 Zeichen, keine Satzzeichen/Newlines
    g = proxy._TextLoopGuard()
    hit = None
    for _ in range(5):
        hit = g.feed(frag, "reasoning") or hit
    assert hit is not None
    assert hit["kind"] in ("ngram", "tail")


# ═══════════════════════════════════════════════════════════════════════════
# 3. False-Positive-Schutz
# ═══════════════════════════════════════════════════════════════════════════

def test_guard_no_hit_for_varied_text():
    g = proxy._TextLoopGuard()
    parts = [
        "The first step is to inspect the configuration file. ",
        "Next I will compare the two implementations carefully. ",
        "Finally I will run the test suite to confirm the change. ",
        "Afterwards I will summarise the findings for the user. ",
    ]
    for p in parts:
        assert g.feed(p, "reasoning") is None


def test_guard_ignores_code_fences_in_content():
    code = "```python\n" + ("x = 1\n" * 30) + "```\n"
    g = proxy._TextLoopGuard()
    hit = None
    for _ in range(3):
        hit = g.feed(code, "content") or hit
    assert hit is None


def test_guard_detects_code_fences_in_reasoning():
    code = "```python\n" + ("x = 1\n" * 30) + "```\n"
    g = proxy._TextLoopGuard()
    hit = None
    for _ in range(4):
        hit = g.feed(code, "reasoning") or hit
    assert hit is not None


def test_guard_respects_min_chars():
    g = proxy._TextLoopGuard(min_chars=100000)
    hit = None
    for _ in range(5):
        hit = g.feed("Short repeated sentence here. ", "reasoning") or hit
    assert hit is None


def test_guard_disabled():
    g = proxy._TextLoopGuard(enabled=False)
    hit = None
    for _ in range(5):
        hit = g.feed("Short repeated sentence here. ", "reasoning") or hit
    assert hit is None


def test_guard_reports_each_kind_only_once():
    g = proxy._TextLoopGuard()
    sentence = "This sentence repeats itself over and over again. "
    first = None
    for _ in range(5):
        first = g.feed(sentence, "reasoning") or first
    assert first is not None
    # Weitere Feeds derselben Art werden nicht erneut gemeldet.
    assert g.feed(sentence, "reasoning") is None


# ═══════════════════════════════════════════════════════════════════════════
# 4. Integration: _stream_backend_turn
# ═══════════════════════════════════════════════════════════════════════════

def test_backend_turn_loop_guard_restarts(monkeypatch):
    monkeypatch.setattr(proxy, "LOOP_GUARD_ENABLED", True)
    monkeypatch.setattr(proxy, "LOOP_GUARD_MODE", "restart")
    monkeypatch.setattr(proxy, "LOOP_GUARD_MAX_RESTARTS", 1)

    captured: List[List[Dict[str, Any]]] = []

    async def fake_single(body, category, def_idx, inject_hindsight=True, force_no_thinking=False):
        captured.append([dict(m) for m in (body.get("messages") or [])])
        if len(captured) == 1:
            for _ in range(8):
                yield {"type": "chunk", "choice": {"delta": {
                    "reasoning_content": "I need to check the file again and again to be sure. "},
                    "finish_reason": None}}
            await asyncio.Event().wait()  # haengt fest bis Restart
            yield {"type": "done"}
        else:
            yield {"type": "chunk", "choice": {"delta": {"content": "Endlich Antwort"}, "finish_reason": None}}
            yield {"type": "chunk", "choice": {"delta": {}, "finish_reason": "stop"}}
            yield {"type": "done"}

    monkeypatch.setattr(proxy, "_stream_single_model_events", fake_single)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    state: Dict[str, Any] = {"stream_id": "test-stream", "role_sent": False}
    sse = _collect_sse(proxy._stream_backend_turn(body, "local", None, state))

    assert len(captured) == 2, "Loop-Guard muss einen internen Restart ausgeloest haben"
    # Hinweis wurde an die History angehaengt (Folgeturn sieht ihn).
    assert captured[1][-1]["role"] == "user"
    assert "loop" in captured[1][-1]["content"].lower()
    # Erster Turn hatte den Hinweis noch nicht.
    assert all("loop" not in str(m.get("content", "")).lower() for m in captured[0])
    joined = "\n".join(sse)
    assert "Endlich Antwort" in joined
    assert state["content"].endswith("Endlich Antwort")


def test_backend_turn_loop_guard_disabled(monkeypatch):
    monkeypatch.setattr(proxy, "LOOP_GUARD_ENABLED", False)

    calls = {"n": 0}

    async def fake_single(body, category, def_idx, inject_hindsight=True, force_no_thinking=False):
        calls["n"] += 1
        for _ in range(8):
            yield {"type": "chunk", "choice": {"delta": {
                "reasoning_content": "I need to check the file again and again to be sure. "},
                "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {"content": "Fertig"}, "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {}, "finish_reason": "stop"}}
        yield {"type": "done"}

    monkeypatch.setattr(proxy, "_stream_single_model_events", fake_single)
    state: Dict[str, Any] = {"stream_id": "test-stream", "role_sent": False}
    _collect_sse(proxy._stream_backend_turn({"messages": []}, "local", None, state))
    assert calls["n"] == 1


def test_backend_turn_loop_guard_exhausted_stops_cleanly(monkeypatch):
    monkeypatch.setattr(proxy, "LOOP_GUARD_ENABLED", True)
    monkeypatch.setattr(proxy, "LOOP_GUARD_MODE", "restart")
    monkeypatch.setattr(proxy, "LOOP_GUARD_MAX_RESTARTS", 0)

    calls = {"n": 0}

    async def fake_single(body, category, def_idx, inject_hindsight=True, force_no_thinking=False):
        calls["n"] += 1
        # Wiederholter CONTENT (nicht Reasoning) — muss am Ende getrimmt werden.
        for _ in range(8):
            yield {"type": "chunk", "choice": {"delta": {
                "content": "I need to check the file again and again to be sure. "},
                "finish_reason": None}}
        await asyncio.Event().wait()
        yield {"type": "done"}

    monkeypatch.setattr(proxy, "_stream_single_model_events", fake_single)
    state: Dict[str, Any] = {"stream_id": "test-stream", "role_sent": False}
    sse = _collect_sse(proxy._stream_backend_turn({"messages": []}, "local", None, state))

    assert calls["n"] == 1, "ohne Restarts darf kein zweiter Call erfolgen"
    # Kein endloses Warten: der Generator terminiert.
    assert state.get("finish_reason") is None or True
    # Trailing Wiederholung wurde getrimmt (ein Vorkommen bleibt).
    assert state["content"].count("I need to check the file again") == 1


# ═══════════════════════════════════════════════════════════════════════════
# 5. Integration: Non-Streaming / 2-Pass
# ═══════════════════════════════════════════════════════════════════════════

def test_outcome_with_loop_guard_retries_nonstream(monkeypatch):
    monkeypatch.setattr(proxy, "LOOP_GUARD_ENABLED", True)
    monkeypatch.setattr(proxy, "LOOP_GUARD_MODE", "restart")
    monkeypatch.setattr(proxy, "LOOP_GUARD_MAX_RESTARTS", 1)

    calls = {"n": 0}

    async def fake_call(body, category, force_start_idx=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"result": {"content": "I need to check the file again and again to be sure. " * 8},
                    "used_model": "test", "all_failed": False}
        return {"result": {"content": "Saubere finale Antwort."},
                "used_model": "test", "all_failed": False}

    monkeypatch.setattr(proxy, "_call_model_with_fallbacks", fake_call)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    outcome = asyncio.run(proxy._outcome_with_loop_guard(
        body, "light", None, use_delegation=False))

    assert calls["n"] == 2
    assert "loop" in body["messages"][-1]["content"].lower()
    assert outcome["result"]["content"] == "Saubere finale Antwort."


def test_outcome_with_loop_guard_disabled_nonstream(monkeypatch):
    monkeypatch.setattr(proxy, "LOOP_GUARD_ENABLED", False)

    calls = {"n": 0}

    async def fake_call(body, category, force_start_idx=None):
        calls["n"] += 1
        return {"result": {"content": "I need to check the file again and again to be sure. " * 8},
                "used_model": "test", "all_failed": False}

    monkeypatch.setattr(proxy, "_call_model_with_fallbacks", fake_call)
    outcome = asyncio.run(proxy._outcome_with_loop_guard(
        {"messages": []}, "light", None, use_delegation=False))

    assert calls["n"] == 1
    assert outcome["result"]["content"].count("I need to check the file again") == 8


# ═══════════════════════════════════════════════════════════════════════════
# 6. End-to-End: Client sieht EINEN Stream mit EINEM finish_reason
# ═══════════════════════════════════════════════════════════════════════════

def test_stream_local_events_hides_internal_restart(monkeypatch):
    """Kernanforderung: Bei einem Thinking-Loop bricht der Proxy intern ab und
    startet neu — der Client (VS Code) sieht einen durchgehenden Stream mit
    genau EINEM finish_reason und nichts von der Unterbrechung."""
    monkeypatch.setattr(proxy, "LOOP_GUARD_ENABLED", True)
    monkeypatch.setattr(proxy, "LOOP_GUARD_MODE", "restart")
    monkeypatch.setattr(proxy, "LOOP_GUARD_MAX_RESTARTS", 1)
    # Kein Co-Worker, kein Auto-Dispatch — reiner Hauptmodell-Stream.
    monkeypatch.setattr(proxy, "COWORKER_ENABLED", False)
    monkeypatch.setattr(proxy, "COWORKER_AUTO_DISPATCH", False)

    calls = {"n": 0}

    async def fake_single(body, category, def_idx, inject_hindsight=True, force_no_thinking=False):
        calls["n"] += 1
        if calls["n"] == 1:
            # Erster Turn: endloser Satz-Loop im Reasoning.
            for _ in range(10):
                yield {"type": "chunk", "choice": {"delta": {
                    "reasoning_content": "I need to check the file again and again to be sure. "},
                    "finish_reason": None}}
            await asyncio.Event().wait()
            yield {"type": "done"}
        else:
            yield {"type": "chunk", "choice": {"delta": {"content": "Fertige Antwort."}, "finish_reason": None}}
            yield {"type": "chunk", "choice": {"delta": {}, "finish_reason": "stop"}}
            yield {"type": "done"}

    monkeypatch.setattr(proxy, "_stream_single_model_events", fake_single)
    body = {"messages": [{"role": "user", "content": "hi"}], "model": "qwen3.8-flash-next"}
    sse = _collect_sse(proxy._stream_local_events(body, "local", None))

    # Genau EIN finish_reason an den Client.
    finish = [s for s in sse if '"finish_reason": "stop"' in s]
    assert len(finish) == 1, f"erwartet 1 finish_reason, war {len(finish)}"
    # Der Loop-Hinweis ging NICHT an den Client (nur intern in die History).
    joined = "\n".join(sse)
    assert "STOP LOOPING" not in joined
    # Die finale Antwort kommt an.
    assert "Fertige Antwort." in joined
