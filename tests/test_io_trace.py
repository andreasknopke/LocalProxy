"""Tests für das I/O-Trace-Modul (data/io_traces/<turn_id>/events.jsonl).

Konvention wie in den anderen Test-Files: plain def test_*(monkeypatch),
innere asyncio.run(...) für async Code. Kein pytest-asyncio, kein TestClient.
"""
import asyncio
import json
from pathlib import Path

import pytest

import proxy
from proxy import (
    _COWORKER_TOOL_NAMES,
    io_end_turn,
    io_log_backend_response,
    io_log_bg_result,
    io_log_client_sse,
    io_log_event,
    io_log_final,
    io_log_inbound,
    io_log_outbound,
    io_start_turn,
    io_trace_active,
    io_trace_analyze,
    io_trace_turn_list,
)


@pytest.fixture(autouse=True)
def _trace_env(tmp_path, monkeypatch):
    """Jeden Test auf ein frisches Temp-Trace-Dir isolieren."""
    trace_dir = tmp_path / "io_traces"
    monkeypatch.setattr(proxy, "IO_TRACE_DIR", trace_dir)
    monkeypatch.setattr(proxy, "IO_TRACE_ENABLED", True)
    proxy._ctx_turn_id.set("")
    yield trace_dir
    proxy._ctx_turn_id.set("")


def _read_events(turn_id: str):
    p = proxy.IO_TRACE_DIR / turn_id / "events.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# ─── Lifecycle ──────────────────────────────────────────────────────────────

def test_start_and_end_turn_lifecycle(monkeypatch):
    turn_id = io_start_turn("local")
    assert turn_id.startswith("turn_")
    assert proxy.io_trace_get_turn() == turn_id
    assert (proxy.IO_TRACE_DIR / turn_id / "meta.json").exists()
    assert (proxy.IO_TRACE_DIR / turn_id / "events.jsonl").exists()

    io_end_turn({"category": "local", "stream": True})
    meta = json.loads((proxy.IO_TRACE_DIR / turn_id / "meta.json").read_text(encoding="utf-8"))
    assert meta["category"] == "local"
    assert meta["stream"] is True
    assert "analysis" in meta
    assert meta["analysis"]["event_count"] == 0
    # ContextVar zurückgesetzt
    assert proxy.io_trace_get_turn() == ""


def test_disabled_trace_is_noop(monkeypatch):
    monkeypatch.setattr(proxy, "IO_TRACE_ENABLED", False)
    assert io_start_turn() == ""
    assert not io_trace_active()
    io_log_event(kind="note", text="sollte nicht landen")
    io_end_turn()  # darf nicht crashen (kein Turn aktiv)
    assert not (proxy.IO_TRACE_DIR).exists() or not list(proxy.IO_TRACE_DIR.iterdir())


# ─── Event-Logging ──────────────────────────────────────────────────────────

def test_log_events_append_jsonl(monkeypatch):
    turn_id = io_start_turn("local")
    io_log_inbound({"messages": [{"role": "user", "content": "hi"}], "tools": [
        {"type": "function", "function": {"name": "vscode_tool_1"}},
    ]})
    io_log_outbound(
        {"messages": [{"role": "system", "content": "sys"}], "tools": [
            {"type": "function", "function": {"name": "vscode_tool_1"}},
            {"type": "function", "function": {"name": "ask_coworker"}},
        ]},
        category="local", model="nemotron", req_id="model_local_0_1_abc")
    io_log_backend_response("model_local_0_1_abc", "nemotron", {
        "choices": [{"message": {"role": "assistant", "content": "ok",
                                 "tool_calls": [
                                     {"id": "c1", "type": "function",
                                      "function": {"name": "ask_coworker",
                                                   "arguments": "{\"task\": \"x\"}"}},
                                 ]}}],
    }, http_status=200)
    io_log_client_sse("data: {\"choices\":[]}")
    io_log_final({"choices": [{"message": {"content": "final"}}]})
    io_log_bg_result("cw_abc", "done", "ergebnis")
    io_end_turn()

    evts = _read_events(turn_id)
    kinds = [e["kind"] for e in evts]
    assert kinds == ["inbound", "outbound", "backend_resp", "client_sse",
                     "final", "bg_result"]
    # outbound muss Tool-Namen extrahiert haben
    out = evts[1]
    assert "ask_coworker" in out["tool_names"]
    assert "vscode_tool_1" in out["tool_names"]
    # bg_result content ist str
    assert evts[5]["content"] == "ergebnis"


# ─── Analyse: die Kernfragen des Co-Worker-Debugs ───────────────────────────

def test_analyze_coworker_tools_on_wire_and_calls(monkeypatch):
    turn_id = io_start_turn("local")
    io_log_outbound(
        {"messages": [
            {"role": "system", "content": "[PROXY DELEGATION GUIDANCE] two-machine team"},
            {"role": "user", "content": "mach was"},
        ], "tools": [
            {"type": "function", "function": {"name": "ask_coworker"}},
            {"type": "function", "function": {"name": "dispatch_coworker"}},
            {"type": "function", "function": {"name": "collect_coworker"}},
        ]},
        category="local", model="nemotron", req_id="r1")
    io_log_backend_response("r1", "nemotron", {
        "choices": [{"message": {"tool_calls": [
            {"function": {"name": "dispatch_coworker", "arguments": "{}"}},
            {"function": {"name": "ask_coworker", "arguments": "{}"}},
        ]}}],
    }, http_status=200)
    io_end_turn()

    a = io_trace_analyze(turn_id)
    assert a["coworker_tools_on_wire"] is True
    assert set(a["coworker_tool_names_found"]) >= {"ask_coworker", "dispatch_coworker"}
    assert a["guidance_in_system"] is True
    assert a["coworker_calls_seen"] == 2
    assert set(a["coworker_call_names"]) == {"dispatch_coworker", "ask_coworker"}
    assert "nemotron" in a["outbound_models"]
    assert a["backend_error"] is None
    assert a["event_count"] == 2


def test_analyze_without_coworker_tools(monkeypatch):
    turn_id = io_start_turn("light")
    io_log_outbound(
        {"messages": [{"role": "system", "content": "plain sys"}],
         "tools": [{"type": "function", "function": {"name": "str_replace"}}]},
        category="light", model="gpt-x", req_id="r2")
    io_log_backend_response("r2", "gpt-x", {
        "choices": [{"message": {"content": "plain",
            "tool_calls": [{"function": {"name": "str_replace", "arguments": "{}"}}]}}],
    }, http_status=200)
    io_end_turn()

    a = io_trace_analyze(turn_id)
    assert a["coworker_tools_on_wire"] is False
    assert a["coworker_tool_names_found"] == []
    assert a["guidance_in_system"] is False
    assert a["coworker_calls_seen"] == 0
    # client_tool_names = alle NON-coworker Tools im Outbound-Payload
    assert a["client_tool_names"] == ["str_replace"]


def test_analyze_stream_chunks_tool_calls(monkeypatch):
    """Stream-Path: tool_calls in sse_chunks Deltas must be recognized."""
    turn_id = io_start_turn("local")
    io_log_outbound(
        {"messages": [{"role": "user", "content": "stream"}],
         "tools": [{"type": "function", "function": {"name": "ask_coworker"}}]},
        category="local", model="m", req_id="r3")
    io_log_backend_response("r3", "m", {
        "sse_chunks": [
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c1", "function": {"name": "ask_coworker",
                                                       "arguments": ""}},
            ]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"name": "",
                                          "arguments": "{\"task\":"}},
            ]}}]},
            {"choices": [{"delta": {"content": "ok"}, "finish_reason": "tool_calls"}]},
        ],
    }, http_status=200)
    io_end_turn()

    a = io_trace_analyze(turn_id)
    assert a["coworker_tools_on_wire"] is True
    assert a["coworker_calls_seen"] == 1
    assert a["coworker_call_names"] == ["ask_coworker"]


def test_analyze_backend_error(monkeypatch):
    turn_id = io_start_turn("coworker")
    io_log_backend_response("r4", "cw", {
        "error": {"message": "connect refused", "note": "backend_error: timeout"}},
        http_status=0)
    io_end_turn()
    a = io_trace_analyze(turn_id)
    assert a["backend_error"] is not None
    assert "timeout" in str(a["backend_error"])


# ─── Turn-Liste / Fail-still / Rotation ──────────────────────────────────────

def test_turn_list_ordered_and_analyzed(monkeypatch):
    t1 = io_start_turn("local")
    io_end_turn()
    t2 = io_start_turn("light")
    io_end_turn()
    turns = io_trace_turn_list()
    ids = [t["turn_id"] for t in turns]
    # Neueste zuerst (Sortierung nach turn_id-Name = Zeitstempel)
    assert t2 in ids and t1 in ids
    assert ids.index(t2) < ids.index(t1)
    # Analyse-Felder sind angereichert
    assert all("coworker_tools_on_wire" in t for t in turns)


def test_fail_still_on_unwritable_dir(monkeypatch):
    # IO_TRACE_DIR auf eine Datei zeigen lassen → mkdir schlägt fehl
    blocker = proxy.IO_TRACE_DIR.parent / "blocker_file"
    blocker.write_text("not a dir", encoding="utf-8")
    monkeypatch.setattr(proxy, "IO_TRACE_DIR", blocker)
    # io_start_turn fängt den Fehler ab und liefert ''
    turn_id = io_start_turn("local")
    assert turn_id == ""
    # Alle io_log_* sind No-Ops ohne Turn
    io_log_inbound({"messages": []})
    io_log_outbound({}, "local", "m", "r")
    io_log_backend_response("r", "m", {})
    io_end_turn()  # kein Crash


def test_rotation_max_turns(monkeypatch):
    monkeypatch.setattr(proxy, "IO_TRACE_MAX_TURNS", 3)
    ids = []
    for i in range(6):
        t = io_start_turn("local")
        assert t, f"turn {i} konnte nicht gestartet werden"
        ids.append(t)
        io_end_turn()
        proxy._io_maybe_rotate(force=True)  # Throttle (1x/60s) im Test umgehen
    # Nach Rotation dürfen maximal IO_TRACE_MAX_TURNS übrig sein
    remaining = [d.name for d in proxy.IO_TRACE_DIR.iterdir() if d.is_dir()]
    assert len(remaining) <= 3


# ─── BG-Task-Integration: _run_bg_coworker_task loggt io_log_bg_result ──────

def test_run_bg_coworker_task_logs_bg_result(monkeypatch):
    """BG-Task-Ergebnis muss als bg_result-Event im Turn landen — auch wenn der
    Task in einem anderen Kontext läuft (bind_turn wurde mitgegeben)."""
    from proxy import CoworkerTask, _run_bg_coworker_task

    turn_id = io_start_turn("local")
    events_before = None

    async def run():
        task = CoworkerTask(task_id="cw_test1", preview="test task")
        # ContextVar weitergeben wie es _spawn tut
        await _run_bg_coworker_task(task, {"task": "mache xyz"})

    async def run_wrapped():
        # Der BG-Task läuft normalerweise ohne aktiven Turn — dann ist das
        # bg_result-Event ein No-Op (fail-still). Hier: Turn explizit binden.
        proxy.io_trace_bind_turn(turn_id)
        await run()

    asyncio.run(run_wrapped())

    evts = _read_events(turn_id)
    bg = [e for e in evts if e["kind"] == "bg_result"]
    # Coworker-Backend nicht erreichbar/konfiguriert → Fehler-Pfad logged auch
    assert len(bg) == 1
    assert bg[0]["task_id"] == "cw_test1"
    assert bg[0]["status"] in ("done", "error", "expired")


def test_coworker_tool_names_wired():
    """Die Analyse muss die echten Tool-Namen kennen (kein leerer Platzhalter)."""
    assert "ask_coworker" in _COWORKER_TOOL_NAMES
    assert "dispatch_coworker" in _COWORKER_TOOL_NAMES
    assert "collect_coworker" in _COWORKER_TOOL_NAMES


# ─── Debug-Endpoints: Response-Konstruktion (Regression JSONResponse) ────────

class _FakeURL:
    def __init__(self, path):
        self.path = path


class _FakeRequest:
    """Minimaler Request-Stub: _auth_or_raise liest nur request.url.path."""
    def __init__(self, path="/"):
        self.url = _FakeURL(path)


def _auth_off(monkeypatch):
    monkeypatch.setattr(proxy, "PROXY_AUTH_ENABLED", False)


def test_debug_streams_endpoint_returns_json(monkeypatch):
    """Regression: JSONResponse(content=..., default=str) crashte mit 500."""
    _auth_off(monkeypatch)
    turn_id = io_start_turn("local")
    io_end_turn({"category": "local", "stream": False, "status": "ok"})

    resp = asyncio.run(proxy.debug_streams(_FakeRequest(), limit=10))
    assert resp.status_code == 200
    data = json.loads(resp.body.decode("utf-8"))
    assert data["count"] == 1
    assert data["turns"][0]["turn_id"] == turn_id
    # Analyse-Felder sind top-level geflattet (coworker_tools_on_wire etc.)
    assert data["turns"][0]["coworker_tools_on_wire"] is False
    assert data["turns"][0]["event_count"] == 0


def test_debug_stream_detail_endpoint_returns_json(monkeypatch):
    """Regression: Detail-Endpoint hatte denselben default=str-Bug."""
    _auth_off(monkeypatch)
    turn_id = io_start_turn("light")
    io_log_inbound({"model": "m", "messages": []})
    io_log_outbound({"model": "m", "tools": [
        {"type": "function", "function": {"name": "str_replace"}}],
        "messages": []}, category="light", model="m", req_id="req1")
    io_log_final({"id": "x", "choices": []})
    io_end_turn({"category": "light", "stream": True, "status": "ok"})

    resp = asyncio.run(proxy.debug_stream_detail(_FakeRequest(), turn_id=turn_id))
    assert resp.status_code == 200
    data = json.loads(resp.body.decode("utf-8"))
    assert data["turn_id"] == turn_id
    assert data["meta"]["category"] == "light"
    assert data["meta"]["analysis"]["client_tool_names"] == ["str_replace"]
    kinds = [e["kind"] for e in data["events"]]
    assert kinds == ["inbound", "outbound", "final"]


def test_debug_stream_detail_invalid_id(monkeypatch):
    _auth_off(monkeypatch)
    resp = asyncio.run(proxy.debug_stream_detail(_FakeRequest(),
                                                 turn_id="..%2Fetc"))
    assert resp.status_code == 400


def test_debug_streams_delete_endpoint(monkeypatch):
    _auth_off(monkeypatch)
    io_start_turn("local")
    io_end_turn({"category": "local", "stream": False, "status": "ok"})
    io_start_turn("light")
    io_end_turn({"category": "light", "stream": False, "status": "ok"})

    resp = asyncio.run(proxy.debug_streams_cleanup(_FakeRequest()))
    assert resp.status_code == 200
    data = json.loads(resp.body.decode("utf-8"))
    # Rotation löscht nur über Caps hinaus — beide Turns bleiben (unter Limits).
    assert data["remaining_turns"] == 2
