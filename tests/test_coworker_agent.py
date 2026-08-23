# -*- coding: utf-8 -*-
"""Tests fuer den Co-Worker Agent-Mode (v4): Tool-Relay-Queue,
/agent/* Endpoints und der agentische Loop."""
import asyncio
import json
import time
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import proxy


@pytest.fixture()
def agent_env(monkeypatch):
    """Agent-Mode aktivieren + Runner als online markieren + Queue leeren."""
    monkeypatch.setattr(proxy, "COWORKER_AGENT_MODE", True)
    monkeypatch.setattr(proxy, "AGENT_TOOL_TIMEOUT", 5.0)
    monkeypatch.setattr(proxy, "_AGENT_RUNNER_SEEN", time.time())
    # Queue/Pending zuruecksetzen (Prozess-global!)
    fresh_queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    monkeypatch.setattr(proxy, "_AGENT_TOOL_QUEUE", fresh_queue)
    monkeypatch.setattr(proxy, "_AGENT_TOOL_PENDING", {})
    # coworker-Kategorie mit gueltiger Definition (Test-Umgebung hat keine config.json-Defs)
    monkeypatch.setattr(proxy, "_MODEL_CATEGORIES", {
        **proxy._MODEL_CATEGORIES,
        "coworker": {"api_url": "http://spark:30000/v1/chat/completions",
                     "api_key": "", "model_name": "qwen3.8-27b",
                     "timeout_seconds": 60},
    })
    yield


# ── _execute_tool_via_runner ──────────────────────────────────────────────

def test_execute_tool_runner_offline_raises(agent_env, monkeypatch):
    """Kein Runner verbunden -> RuntimeError mit Hinweis auf agent_runner.py."""
    monkeypatch.setattr(proxy, "_AGENT_RUNNER_SEEN", 0.0)
    with pytest.raises(RuntimeError, match="agent_runner"):
        asyncio.run(proxy._execute_tool_via_runner("read_file", {"path": "x.py"}))


def test_execute_tool_roundtrip_via_runner(agent_env):
    """Queue-Eintrag landet beim Runner, Ergebnis wird an den Caller geliefert."""
    async def scenario():
        async def runner_side():
            entry = await proxy._AGENT_TOOL_QUEUE.get()
            entry["result"] = "file content here"
            entry["done"].set()

        rt = asyncio.ensure_future(runner_side())
        res = await proxy._execute_tool_via_runner("read_file", {"path": "a.py"}, task_id="cw_x")
        await rt
        return res

    res = asyncio.run(scenario())
    assert res == "file content here"


def test_execute_tool_timeout(agent_env, monkeypatch):
    """Runner antwortet nicht -> RuntimeError Tool-Timeout."""
    monkeypatch.setattr(proxy, "AGENT_TOOL_TIMEOUT", 0.1)
    with pytest.raises(RuntimeError, match="Tool-Timeout"):
        asyncio.run(proxy._execute_tool_via_runner("read_file", {"path": "a.py"}))


# ── /agent/* Endpoints ────────────────────────────────────────────────────

@pytest.fixture()
def agent_app(monkeypatch):
    app = FastAPI()
    app.router.routes.extend([
        r for r in proxy.app.router.routes
        if getattr(r, "path", "").startswith("/agent/")
    ])
    monkeypatch.setattr(proxy, "AGENT_RUNNER_TOKEN", "runner-secret")
    return TestClient(app)


def test_agent_auth_rejects_bad_token(agent_app):
    assert agent_app.get("/agent/status",
                         headers={"Authorization": "Bearer falsch"}).status_code == 403
    assert agent_app.get("/agent/status").status_code == 401


def test_agent_status_and_claim_result_roundtrip(agent_app):
    h = {"Authorization": "Bearer runner-secret"}
    st = agent_app.get("/agent/status", headers=h).json()
    assert st["queue_depth"] == 0

    # Claim blockt bis Eintrag da ist — wir legen direkt einen Pending-Eintrag an
    entry = {
        "call_id": "tc_test123", "task_id": "cw_1", "name": "write_file",
        "arguments": {"path": "t.txt", "content": "hi"},
        "result": None, "error": None, "done": asyncio.Event(),
    }
    proxy._AGENT_TOOL_PENDING["tc_test123"] = entry
    proxy._AGENT_TOOL_QUEUE.put_nowait(entry)

    claimed = agent_app.get("/agent/tools/claim", params={"wait": 1}, headers=h)
    assert claimed.status_code == 200
    job = claimed.json()
    assert job["call_id"] == "tc_test123"
    assert job["name"] == "write_file"

    res = agent_app.post("/agent/tools/result", headers=h,
                         json={"call_id": "tc_test123", "result": "OK"})
    assert res.status_code == 200
    assert entry["done"].is_set()


def test_agent_claim_empty_returns_204(agent_app):
    h = {"Authorization": "Bearer runner-secret"}
    r = agent_app.get("/agent/tools/claim", params={"wait": 0.2}, headers=h)
    assert r.status_code == 204


# ── Agent-Loop ────────────────────────────────────────────────────────────

class _FakeResp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def test_agent_loop_final_answer_without_tools(agent_env, monkeypatch):
    """Coworker antwortet direkt ohne tool_calls -> Loop endet in Runde 1."""
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
                "role": "assistant", "content": "Fertig! Datei erstellt.",
                "tool_calls": None}}]})

    monkeypatch.setattr(proxy.httpx, "AsyncClient", FakeClient)
    res = asyncio.run(proxy._run_coworker_agent(
        "Schreibe tests/test_x.py", "", extra_context=None))
    assert res["status"] == "ok"
    assert res["rounds"] == 1
    assert "Fertig" in res["content"]
    body = calls[0]
    tool_names = [t["function"]["name"] for t in body["tools"]]
    assert set(tool_names) == {"read_file", "write_file", "list_dir", "web_search"}


def test_agent_loop_executes_tools_then_finishes(agent_env, monkeypatch):
    """Runde 1: write_file-Call -> Relay -> Runde 2: finale Antwort."""
    rounds = [
        {"choices": [{"message": {
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "write_file",
                             "arguments": json.dumps({"path": "tests/test_gen.py",
                                                      "content": "def test_ok():\n    assert True\n"})},
            }]}}]},
        {"choices": [{"message": {
            "role": "assistant", "content": "Datei tests/test_gen.py geschrieben.",
            "tool_calls": None}}]},
    ]
    state = {"n": 0}

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None):
            payload = rounds[state["n"]]
            state["n"] += 1
            return _FakeResp(payload)

    async def fake_relay(name, args, task_id=""):
        assert name == "write_file"
        assert args["path"] == "tests/test_gen.py"
        return "OK: geschrieben"

    monkeypatch.setattr(proxy.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(proxy, "_execute_tool_via_runner", fake_relay)
    res = asyncio.run(proxy._run_coworker_agent("Erstelle die Testdatei", ""))
    assert res["status"] == "ok"
    assert res["rounds"] == 2
    assert "test_gen.py" in res["content"]


def test_agent_loop_max_rounds_partial_result(agent_env, monkeypatch):
    """Loop bricht nach Max-Runden ab und liefert Teilergebnis."""
    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None):
            return _FakeResp({"choices": [{"message": {
                "role": "assistant", "content": "noch nicht fertig",
                "tool_calls": [{"id": "c0", "type": "function",
                                "function": {"name": "list_dir", "arguments": "{}"}}]}}]})

    async def fake_relay(name, args, task_id=""):
        return "(leer)"

    monkeypatch.setattr(proxy.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(proxy, "_execute_tool_via_runner", fake_relay)
    monkeypatch.setattr(proxy, "COWORKER_AGENT_MAX_ROUNDS", 3)
    res = asyncio.run(proxy._run_coworker_agent("Endlos-Task", ""))
    assert res["status"] == "ok"
    assert "Max-Runden" in res["content"]
