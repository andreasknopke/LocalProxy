# -*- coding: utf-8 -*-
"""Tests fuer den Co-Worker Client-Tool-Tunnel (v5).

Der Runner ist entfernt: Der Proxy ist reiner Forwarder. Diese Tests decken
die Tunnel-Infrastruktur ab:
  * ID-Mapping (_cw_parse_tunnel_id, _cw_map_tool_calls_out)
  * Session-Store (_cw_session_new, _cw_absorb_tool_results)
  * History-Rewrite (_cw_strip_tunnel_from_messages, _cw_attach_finals)
  * Fallback-Pfad _run_coworker_agent (ohne Live-Stream / ohne Tools)
"""
import asyncio
import json

import pytest

import proxy


@pytest.fixture(autouse=True)
def _clean_tunnel_state(monkeypatch):
    """Isoliert die Prozess-globalen Tunnel-Stores pro Test."""
    monkeypatch.setattr(proxy, "_CW_SESSIONS", {})
    monkeypatch.setattr(proxy, "_CW_GROUPS", {})
    monkeypatch.setattr(proxy, "_CW_ARCHIVE", {})
    monkeypatch.setattr(proxy, "_CW_SESSIONS_LAST_CLEANUP", 0.0)
    monkeypatch.setattr(proxy, "_CW_RESUME_PENDING", [])
    yield


def _model_defs_fixture(monkeypatch):
    """coworker-Kategorie mit gueltiger Definition (Test-Umgebung ohne config)."""
    monkeypatch.setattr(proxy, "_MODEL_CATEGORIES", {
        **proxy._MODEL_CATEGORIES,
        "coworker": {"api_url": "http://spark:30000/v1/chat/completions",
                     "api_key": "", "model_name": "qwen3.8-27b",
                     "timeout_seconds": 60},
    })


# ── ID-Mapping ────────────────────────────────────────────────────────────

def test_parse_tunnel_id_roundtrip():
    sid, orig = proxy._cw_parse_tunnel_id("cws_abc12345_call_42")
    assert sid == "abc12345"
    assert orig == "call_42"


def test_parse_tunnel_id_rejects_foreign_ids():
    assert proxy._cw_parse_tunnel_id("call_42") is None
    assert proxy._cw_parse_tunnel_id("") is None
    assert proxy._cw_parse_tunnel_id(None) is None
    # Prefix ohne gueltige Reststruktur
    assert proxy._cw_parse_tunnel_id("cws_abc") is None
    assert proxy._cw_parse_tunnel_id("cws_") is None


def test_map_tool_calls_out_assigns_tunnel_ids():
    sess = proxy._cw_session_new("task", "ctx", client_tools=[])
    calls = [{"id": "call_1", "type": "function",
              "function": {"name": "read_file",
                           "arguments": json.dumps({"filePath": "a.py"})}}]
    fwd = proxy._cw_map_tool_calls_out(sess, calls)
    assert len(fwd) == 1
    tc = fwd[0]
    assert tc["index"] == 0
    assert tc["id"].startswith(proxy.CW_TUNNEL_ID_PREFIX)
    assert tc["function"]["name"] == "read_file"
    # pending wurde gesetzt
    assert tc["id"] in sess["pending"]
    assert sess["pending"][tc["id"]]["orig_id"] == "call_1"


def test_absorb_tool_results_roundtrip():
    sess = proxy._cw_session_new("task", "ctx", client_tools=[])
    calls = [{"id": "call_1", "type": "function",
              "function": {"name": "read_file",
                           "arguments": json.dumps({"filePath": "a.py"})}}]
    fwd = proxy._cw_map_tool_calls_out(sess, calls)
    tunnel_id = fwd[0]["id"]

    n = proxy._cw_absorb_tool_results(sess, [
        {"role": "tool", "tool_call_id": tunnel_id, "content": "inhalt"}])
    assert n == 1
    assert sess["pending"] == {}
    # History enthaelt die tool-Message mit ORIGINAL-ID
    tool_msg = sess["messages"][-1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_1"
    assert tool_msg["content"] == "inhalt"


def test_absorb_ignores_foreign_ids():
    sess = proxy._cw_session_new("task", "ctx", client_tools=[])
    n = proxy._cw_absorb_tool_results(sess, [
        {"role": "tool", "tool_call_id": "call_999", "content": "fremd"}])
    assert n == 0
    assert len(sess["messages"]) == 2  # nur system + user


def test_absorb_coerces_non_string_content():
    sess = proxy._cw_session_new("task", "ctx", client_tools=[])
    calls = [{"id": "c1", "type": "function",
              "function": {"name": "list_dir", "arguments": "{}"}}]
    fwd = proxy._cw_map_tool_calls_out(sess, calls)
    n = proxy._cw_absorb_tool_results(sess, [
        {"role": "tool", "tool_call_id": fwd[0]["id"],
         "content": {"files": ["a.py"]}}])
    assert n == 1
    assert isinstance(sess["messages"][-1]["content"], str)
    assert "a.py" in sess["messages"][-1]["content"]


# ── History-Rewrite ───────────────────────────────────────────────────────

def test_strip_removes_pure_tunnel_turn():
    msgs = [
        {"role": "user", "content": "mach was"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "cws_abc_call_1", "type": "function",
                         "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "cws_abc_call_1", "content": "x"},
    ]
    removed = proxy._cw_strip_tunnel_from_messages(msgs)
    assert removed == 2
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"


def test_strip_keeps_non_tunnel_calls_in_mixed_turn():
    msgs = [
        {"role": "assistant", "content": None,
         "tool_calls": [
             {"id": "call_real", "type": "function",
              "function": {"name": "read_file", "arguments": "{}"}},
             {"id": "cws_abc_call_1", "type": "function",
              "function": {"name": "grep_search", "arguments": "{}"}},
         ]},
        {"role": "tool", "tool_call_id": "cws_abc_call_1", "content": "x"},
        {"role": "tool", "tool_call_id": "call_real", "content": "y"},
    ]
    removed = proxy._cw_strip_tunnel_from_messages(msgs)
    assert removed == 1  # nur die Tunnel-tool-Message
    # Assistant behaelt den echten Call
    kept_calls = [t["id"] for t in msgs[0]["tool_calls"]]
    assert kept_calls == ["call_real"]
    # tool-Message mit echter ID bleibt
    assert msgs[1]["tool_call_id"] == "call_real"


def test_attach_finals_writes_ask_result_pair():
    sess = proxy._cw_session_new("task", "ctx", client_tools=[])
    sess["orig_ask"] = {"id": "call_ask", "type": "function",
                        "function": {"name": "ask_coworker", "arguments": "{}"}}
    sess["done"] = True
    sess["final"] = "fertig"
    msgs = [{"role": "user", "content": "start"}]
    proxy._cw_attach_finals(msgs, [sess])
    assert len(msgs) == 3
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["tool_calls"][0]["id"] == "call_ask"
    assert msgs[2]["role"] == "tool"
    assert msgs[2]["tool_call_id"] == "call_ask"
    assert msgs[2]["content"] == "fertig"


def test_attach_finals_skips_not_done_sessions():
    sess = proxy._cw_session_new("task", "ctx", client_tools=[])
    sess["orig_ask"] = {"id": "call_ask", "type": "function",
                        "function": {"name": "ask_coworker", "arguments": "{}"}}
    sess["done"] = False
    msgs = [{"role": "user", "content": "start"}]
    proxy._cw_attach_finals(msgs, [sess])
    assert len(msgs) == 1  # nichts angehaengt


# ── Resume-Routing ────────────────────────────────────────────────────────

def test_resume_sessions_groups_by_sid():
    s1 = proxy._cw_session_new("t1", "", client_tools=[])
    s2 = proxy._cw_session_new("t2", "", client_tools=[])
    calls1 = [{"id": "a", "type": "function",
               "function": {"name": "read_file", "arguments": "{}"}}]
    calls2 = [{"id": "b", "type": "function",
               "function": {"name": "write_file", "arguments": "{}"}}]
    fwd1 = proxy._cw_map_tool_calls_out(s1, calls1)
    fwd2 = proxy._cw_map_tool_calls_out(s2, calls2)

    tool_msgs = [
        {"role": "tool", "tool_call_id": fwd1[0]["id"], "content": "r1"},
        {"role": "tool", "tool_call_id": fwd2[0]["id"], "content": "r2"},
    ]
    resumed = proxy._cw_resume_sessions(tool_msgs)
    assert len(resumed) == 2
    assert s1["pending"] == {}
    assert s2["pending"] == {}


# ── Fallback _run_coworker_agent ──────────────────────────────────────────

class _FakeResp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def test_run_coworker_agent_fallback_without_tools(monkeypatch):
    _model_defs_fixture(monkeypatch)
    calls = []

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None):
            calls.append(json)
            return _FakeResp({"choices": [{"message": {
                "role": "assistant", "content": "Fertig!",
                "tool_calls": None}}]})

    monkeypatch.setattr(proxy.httpx, "AsyncClient", FakeClient)
    res = asyncio.run(proxy._run_coworker_agent("Schreibe tests/test_x.py", ""))
    assert res["status"] == "ok"
    assert "Fertig" in res["content"]
    # Fallback sendet KEINE Tools (plain Prompt)
    assert "tools" not in calls[0]


def test_run_coworker_agent_fallback_no_config(monkeypatch):
    monkeypatch.setattr(proxy, "_MODEL_CATEGORIES", {})
    res = asyncio.run(proxy._run_coworker_agent("task", ""))
    assert res["status"] == "error"
