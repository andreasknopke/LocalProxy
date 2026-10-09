"""
Unit-Tests fuer die Text-Tool-Call-Reparatur und den SSE-Terminator.

Hintergrund: Lokale Modelle (Qwen & Co.) emittieren Tool-Calls oft als Markup
im content statt als strukturierte message.tool_calls. Der Proxy repariert das:

  1) _parse_text_tool_calls: Markup -> strukturierte tool_calls + Cleanup
     (JSON/Hermes, Qwen <function=…><parameter=…>, Anthropic-invoke, DSML,
     fehlende Schluss-Tags)
  2) _repair_tool_calls_from_text: Non-Streaming-Fallback
  3) _ToolCallTextShield: haelt Markup im Live-Stream zurueck
  4) _io_tee: sendet den OpenAI-SSE-Terminator "data: [DONE]"
  5) Integration in _stream_backend_turn
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
    spec = importlib.util.spec_from_file_location("proxy_tc_repair_test", REPO_ROOT / "proxy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["proxy_tc_repair_test"] = module
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
    async def _collect():
        return [s async for s in agen]
    return asyncio.run(_collect())


def _call_pairs(tool_calls) -> List[tuple]:
    assert tool_calls
    return [(c["function"]["name"], json.loads(c["function"]["arguments"]))
            for c in tool_calls]


# ═══════════════════════════════════════════════════════════════════════════
# 1. _parse_text_tool_calls — Formate
# ═══════════════════════════════════════════════════════════════════════════

QWEN_UNCLOSED = (
    "Jetzt das Edit:\n"
    "<tool_call>\n"
    "<function=edit>\n"
    "<parameter=path>\n"
    "C:\\Temp\\pxa-bench.sh\n"
    "</parameter>\n"
    "<parameter=old_str>\n"
    "RESP=$(curl -s http://127.0.0.1:8084/v1/completion -d @/tmp/bench-req.json)\n"
    "</parameter>\n"
    "<parameter=new_str>\n"
    "RESP=$(curl -s http://127.0.0.1:8084/v1/chat/completions -d @/tmp/bench-req.json)\n"
    "</parameter>\n"
    "</function>\n"
)


def test_parse_qwen_function_with_missing_closing_tag():
    """Der vom lokalen Modell real erzeugte Defekt: </tool_call> fehlt."""
    clean, calls = proxy._parse_text_tool_calls(QWEN_UNCLOSED)
    assert clean == "Jetzt das Edit:"
    pairs = _call_pairs(calls)
    assert len(pairs) == 1
    name, args = pairs[0]
    assert name == "edit"
    assert args["path"] == "C:\\Temp\\pxa-bench.sh"
    # Code/Einrueckung im Wert bleibt unangetastet, nur Markup-Newlines fallen weg
    assert args["old_str"].startswith("RESP=$(curl -s")
    assert "/v1/chat/completions" in args["new_str"]
    assert not args["old_str"].startswith("\n")


def test_parse_qwen_function_closed_not_duplicated():
    text = ("<tool_call>\n<function=edit>\n<parameter=path>a.py</parameter>\n"
            "</function>\n</tool_call>")
    clean, calls = proxy._parse_text_tool_calls(text)
    assert _call_pairs(calls) == [("edit", {"path": "a.py"})]
    assert clean == ""


def test_parse_hermes_json_block():
    text = 'Ich lese.\n<tool_call>{"name": "read_file", "arguments": {"filePath": "a.py"}}</tool_call>\nFertig.'
    clean, calls = proxy._parse_text_tool_calls(text)
    assert _call_pairs(calls) == [("read_file", {"filePath": "a.py"})]
    assert clean == "Ich lese.\n\nFertig."


def test_parse_json_list_with_openai_nested_shape():
    text = ('<tool_calls>[{"name":"grep_search","arguments":{"query":"x"}},'
            '{"type":"function","function":{"name":"read_file","arguments":"{\\"filePath\\":\\"b.py\\"}"}}]'
            '</tool_calls>')
    clean, calls = proxy._parse_text_tool_calls(text)
    assert _call_pairs(calls) == [
        ("grep_search", {"query": "x"}),
        ("read_file", {"filePath": "b.py"}),
    ]
    assert clean == ""


def test_parse_invoke_style():
    text = '<invoke name="read_file"><parameter name="filePath">c.py</parameter></invoke>'
    _, calls = proxy._parse_text_tool_calls(text)
    assert _call_pairs(calls) == [("read_file", {"filePath": "c.py"})]


def test_parse_dsml_style():
    pipe = "\uff5c"
    d = f"{pipe}{pipe}DSML{pipe}{pipe}"
    text = (f'<{d}invoke name="read_file">\n'
            f'<{d}parameter name="filePath" string="true">d.py</{d}parameter>\n'
            f'</{d}invoke>')
    _, calls = proxy._parse_text_tool_calls(text)
    assert _call_pairs(calls) == [("read_file", {"filePath": "d.py"})]


def test_parse_bare_function_without_wrapper():
    text = 'Ok.\n<function=run_in_terminal><parameter=command>ls -la</parameter></function>'
    clean, calls = proxy._parse_text_tool_calls(text)
    assert _call_pairs(calls) == [("run_in_terminal", {"command": "ls -la"})]
    assert clean == "Ok."


def test_parse_truncated_json_block():
    text = 'Lesen:\n<tool_call>{"name": "read_file", "arguments": {"filePath": "x.py"}}'
    clean, calls = proxy._parse_text_tool_calls(text)
    assert _call_pairs(calls) == [("read_file", {"filePath": "x.py"})]
    assert clean == "Lesen:"


def test_parse_multiple_blocks():
    text = ('<tool_call><function=read_file><parameter=filePath>a.py</parameter></function></tool_call>'
            '<tool_call><function=read_file><parameter=filePath>b.py</parameter></function></tool_call>')
    _, calls = proxy._parse_text_tool_calls(text)
    assert _call_pairs(calls) == [("read_file", {"filePath": "a.py"}),
                                  ("read_file", {"filePath": "b.py"})]


def test_parse_leaves_text_without_calls_untouched():
    """Markup ohne Call-Inhalt darf NICHT entfernt werden (False-Positive-Schutz)."""
    for text in (
        "Der Tag <tool_call> ist nur erwaehnt. <tool_call></tool_call> Ende.",
        "Ganz normale Antwort ohne Markup.",
        "",
    ):
        clean, calls = proxy._parse_text_tool_calls(text)
        assert calls is None
        assert clean == text


def test_parse_rejects_implausible_tool_names():
    text = '<tool_call>{"name": "not a name!", "arguments": {}}</tool_call>'
    clean, calls = proxy._parse_text_tool_calls(text)
    assert calls is None
    assert clean == text


# ═══════════════════════════════════════════════════════════════════════════
# 2. _repair_tool_calls_from_text — Non-Streaming-Fallback
# ═══════════════════════════════════════════════════════════════════════════

def test_repair_keeps_structured_tool_calls():
    structured = [{"id": "call_1", "type": "function",
                   "function": {"name": "read_file", "arguments": "{}"}}]
    content, calls = proxy._repair_tool_calls_from_text("Text", structured)
    assert calls is structured
    assert content == "Text"


def test_repair_extracts_from_content():
    content, calls = proxy._repair_tool_calls_from_text(QWEN_UNCLOSED, None)
    assert _call_pairs(calls)[0][0] == "edit"
    assert "<tool_call>" not in content
    assert "<parameter=" not in content


def test_repair_noop_without_markup():
    content, calls = proxy._repair_tool_calls_from_text("nur text", None)
    assert calls is None
    assert content == "nur text"


# ═══════════════════════════════════════════════════════════════════════════
# 3. _ToolCallTextShield — Live-Stream-Schutz
# ═══════════════════════════════════════════════════════════════════════════

def test_shield_forwards_plain_text():
    shield = proxy._ToolCallTextShield()
    assert shield.feed("Hallo ") == "Hallo "
    assert shield.feed("Welt") == "Welt"
    flushed, calls = shield.finalize()
    assert flushed == ""
    assert calls is None


def test_shield_holds_markup_and_requires_via_stub():
    shield = proxy._ToolCallTextShield()
    visible = shield.feed("Ich lese die Datei")
    visible += shield.feed("\n<tool_call>{\"name\": \"read_file\", \"arguments\": ")
    visible += shield.feed("{\"filePath\": \"a.py\"}}</tool_call>")
    assert visible == "Ich lese die Datei\n"
    flushed, calls = shield.finalize()
    assert flushed == ""
    assert _call_pairs(calls) == [("read_file", {"filePath": "a.py"})]


def test_shield_flushes_non_tool_markup_unchanged():
    """Markup, das kein Tool-Call ist, darf nicht verschwinden."""
    shield = proxy._ToolCallTextShield()
    assert shield.feed("Text <tool_call>") == "Text "
    flushed, calls = shield.finalize()
    assert calls is None
    assert flushed == "<tool_call>"


def test_shield_handles_split_opener_across_chunks():
    """Der Opener kann mitten im Tag umbrechen ("<tool_cal" + "l>")."""
    shield = proxy._ToolCallTextShield()
    assert shield.feed("vor ") == "vor "
    assert shield.feed("<tool_cal") == ""
    assert shield.feed("l>{\"name\": \"grep_search\", \"arguments\": ") == ""
    assert shield.feed("{\"query\": \"q\"}}</tool_call>") == ""
    flushed, calls = shield.finalize()
    assert flushed == ""
    assert _call_pairs(calls) == [("grep_search", {"query": "q"})]


def test_shield_flushes_partial_tag_at_turn_end():
    """Ein am Turn-Ende angebrochener Tag darf nicht verloren gehen."""
    shield = proxy._ToolCallTextShield()
    assert shield.feed("Also ") == "Also "
    assert shield.feed("<tool") == ""      # Praefix eines echten Tag-Namens → halten
    flushed, calls = shield.finalize()
    assert calls is None
    assert flushed == "<tool"


def test_shield_releases_prose_mentioning_markup_immediately():
    """Prosa, die ein Tool-Call-Tag nur erwaehnt, darf den Stream nicht bis zum
    Turn-Ende blockieren (sonst 'kein Stream' aus Client-Sicht)."""
    shield = proxy._ToolCallTextShield()
    assert shield.feed("Siehe ") == "Siehe "
    assert shield.feed("<tool_call> ") == ""
    prose = "ist veraltet und wird nicht mehr verwendet. " * 10  # > Gate-Fenster
    released = shield.feed(prose)
    assert released == "<tool_call> " + prose
    # Danach fliesst alles sofort weiter, ein spaeterer echter Call wird
    # trotzdem noch erkannt.
    assert shield.feed("noch mehr Text") == "noch mehr Text"
    assert shield.feed('<tool_call>{"name": "read_file", "arguments": {"filePath": "a.py"}}'
                       '</tool_call>') == ""
    flushed, calls = shield.finalize()
    assert flushed == ""
    assert _call_pairs(calls) == [("read_file", {"filePath": "a.py"})]


def test_shield_commits_on_call_signal_within_gate():
    """Steht das Call-Signal sofort nach dem Opener, wird bis zum Turn-Ende
    gehalten (kein Markup-Leak)."""
    shield = proxy._ToolCallTextShield()
    shield.feed("<tool_call>")
    assert shield.feed("\n<function=edit>") == ""
    assert shield.feed("\n<parameter=path>a.sh</parameter>") == ""
    assert shield.feed("\n</function>") == ""
    flushed, calls = shield.finalize()
    assert flushed == ""
    assert _call_pairs(calls) == [("edit", {"path": "a.sh"})]


def test_shield_does_not_hold_plain_text_with_angle_bracket():
    """Normaler Text mit '<' darf nicht gepuffert werden (Streaming-Delay)."""
    shield = proxy._ToolCallTextShield()
    for text in ("a < b", "x<b", "5 <", "1 < 2 und 3 > 2"):
        shield = proxy._ToolCallTextShield()
        assert shield.feed(text) + shield.finalize()[0] == text


# ═══════════════════════════════════════════════════════════════════════════
# 4. _io_tee — SSE-Terminator
# ═══════════════════════════════════════════════════════════════════════════

def test_io_tee_appends_done_terminator():
    async def gen():
        yield "data: {\"a\": 1}\n\n"
        yield "data: {\"b\": 2}\n\n"

    out = _collect_sse(proxy._io_tee(gen()))
    assert out[-1] == "data: [DONE]\n\n"
    assert sum(1 for s in out if s == "data: [DONE]\n\n") == 1


def test_io_tee_does_not_append_done_on_error():
    async def gen():
        yield "data: {\"a\": 1}\n\n"
        raise RuntimeError("boom")

    async def _run():
        received = []
        with pytest.raises(RuntimeError):
            async for s in proxy._io_tee(gen()):
                received.append(s)
        return received

    received = asyncio.run(_run())
    assert received == ["data: {\"a\": 1}\n\n"]


def test_sse_done_constant_format():
    assert proxy.SSE_DONE == "data: [DONE]\n\n"


# ═══════════════════════════════════════════════════════════════════════════
# 5. Integration: _stream_backend_turn
# ═══════════════════════════════════════════════════════════════════════════

def test_backend_turn_repairs_leaked_text_tool_call(monkeypatch):
    """Tool-Call-Markup im content wird nicht an den Client gestreamt, sondern
    in strukturierte tool_calls umgewandelt (state['tool_calls'])."""
    monkeypatch.setattr(proxy, "TOOL_CALL_SHIELD_ENABLED", True)
    async def fake_single(body, category, def_idx, inject_hindsight=True, force_no_thinking=False):
        yield {"type": "chunk", "choice": {"delta": {"content": "Ich editiere die Datei.\n"}, "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {"content": "<tool_call>\n<function=edit>\n"}, "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {"content": "<parameter=path>a.sh</parameter>\n"}, "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {"content": "</function>\n"}, "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {}, "finish_reason": "stop"}}
        yield {"type": "done"}

    monkeypatch.setattr(proxy, "_stream_single_model_events", fake_single)
    state: Dict[str, Any] = {"stream_id": "s", "role_sent": False}
    body = {"messages": [], "tools": [{"type": "function", "function": {"name": "edit"}}]}
    sse = _collect_sse(proxy._stream_backend_turn(body, "local", None, state))
    joined = "\n".join(sse)

    assert "<tool_call>" not in joined
    assert "<parameter=" not in joined
    assert "Ich editiere die Datei." in joined
    calls = proxy._finalize_stream_tool_calls(state)
    assert _call_pairs(calls) == [("edit", {"path": "a.sh"})]
    assert "<tool_call>" not in state["content"]


def test_backend_turn_keeps_plain_content_streaming(monkeypatch):
    """Regulaerer Text fliesst unveraendert (und ohne Verzoegerung) durch."""
    async def fake_single(body, category, def_idx, inject_hindsight=True, force_no_thinking=False):
        yield {"type": "chunk", "choice": {"delta": {"content": "Teil1 "}, "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {"content": "Teil2"}, "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {}, "finish_reason": "stop"}}
        yield {"type": "done"}

    monkeypatch.setattr(proxy, "_stream_single_model_events", fake_single)
    state: Dict[str, Any] = {"stream_id": "s", "role_sent": False}
    sse = _collect_sse(proxy._stream_backend_turn({"messages": []}, "local", None, state))
    assert state["content"] == "Teil1 Teil2"
    assert any("Teil1 " in s for s in sse)
    assert proxy._finalize_stream_tool_calls(state) is None


def test_io_tee_end_to_end_repairs_markup_and_terminates(monkeypatch):
    """Voller Pfad _io_tee(_stream_backend_turn(...)) mit dem exakt gemeldeten
    Qwen-Markup (mehrere Chunks, fehlendes </tool_call>):
    kein Markup auf der Leitung, strukturierte tool_calls im state, und der
    Stream endet mit genau einem 'data: [DONE]'."""
    chunks = [
        "Der /v1/completion-Endpoint lieferte 404, ich passe das Skript an.\n\n",
        "<tool_call>\n<function=edit>\n",
        "<parameter=path>\n",
        "C:\\Temp\\pxa-bench.sh\n",
        "</parameter>\n<parameter=old_str>\n",
        "RESP=$(curl http://127.0.0.1:8084/v1/completion -d @/tmp/bench-req.json)\n",
        "</parameter>\n<parameter=new_str>\n",
        "RESP=$(curl http://127.0.0.1:8084/v1/chat/completions -d @/tmp/bench-req.json)\n",
        "</parameter>\n</function>\n",
    ]

    async def fake_single(body, category, def_idx, inject_hindsight=True, force_no_thinking=False):
        for c in chunks:
            yield {"type": "chunk", "choice": {"delta": {"content": c}, "finish_reason": None}}
        yield {"type": "chunk", "choice": {"delta": {}, "finish_reason": "stop"}}
        yield {"type": "done"}

    monkeypatch.setattr(proxy, "_stream_single_model_events", fake_single)
    monkeypatch.setattr(proxy, "TOOL_CALL_SHIELD_ENABLED", True)
    state: Dict[str, Any] = {"stream_id": "s", "role_sent": False}
    body = {"messages": [], "tools": [{"type": "function", "function": {"name": "edit"}}]}
    sse = _collect_sse(proxy._io_tee(
        proxy._stream_backend_turn(body, "local", None, state)))
    joined = "\n".join(sse)

    assert joined.rstrip().endswith("data: [DONE]")
    assert sum(1 for s in sse if s == proxy.SSE_DONE) == 1
    assert "<tool_call>" not in joined
    assert "<function=" not in joined
    assert "<parameter=" not in joined
    assert "/v1/completion-Endpoint lieferte 404" in joined

    calls = proxy._finalize_stream_tool_calls(state)
    assert _call_pairs(calls) == [("edit", {
        "path": "C:\\Temp\\pxa-bench.sh",
        "old_str": "RESP=$(curl http://127.0.0.1:8084/v1/completion -d @/tmp/bench-req.json)",
        "new_str": "RESP=$(curl http://127.0.0.1:8084/v1/chat/completions -d @/tmp/bench-req.json)",
    })]


def test_shield_handles_many_false_openers_in_one_chunk():
    """Viele Prosa-Opener in einem Chunk duerfen weder rekursiv noch haengen."""
    shield = proxy._ToolCallTextShield()
    text = ("<tool_call> nur Prosa ohne Call-Signal, weiter im Text. " * 500)
    released = shield.feed(text)
    flushed, calls = shield.finalize()
    assert calls is None
    assert released + flushed == text
