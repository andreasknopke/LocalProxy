"""Tests fuer Fork-Join (v3.2): dispatch_coworker / collect_coworker.

Deckt ab:
- Task-Store: register → await (done) → delivered → cleanup
- TTL-Ablauf (running → expired + cancel)
- _partition_tool_calls 4-Bucket (hier nochmal gegen echte proxy-Symbole)
- Status-Notiz (_coworker_status_line) Format
- _delegation_loop: dispatch → mini-result → naechste Runde → finale Antwort
- _delegation_loop: collect mit gemocktem _await_bg_tasks → tool-result JSON
- Fork-Join OFF → dispatch/collect werden NICHT injiziert (Rueckfall v3.1)
"""
import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_proxy_module():
    spec = importlib.util.spec_from_file_location("proxy_bg_test", REPO_ROOT / "proxy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["proxy_bg_test"] = module
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


def _tc(name: str, args: Dict[str, Any], tid: str = "call_x") -> Dict[str, Any]:
    return {"id": tid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


# ── Task-Store ─────────────────────────────────────────────────────────────

def test_register_and_await_done(monkeypatch):
    async def run():
        store: Dict[str, Any] = {}

        async def fake_bg(task, args):
            task.status = "done"
            task.result = "ergebnis-text"
            task.finished_at = time.time()

        monkeypatch.setattr(proxy, "_run_bg_coworker_task", fake_bg)
        monkeypatch.setattr(proxy, "_COWORKER_BG_TASKS", store)

        ct = proxy._register_bg_dispatch(
            _tc("dispatch_coworker", {"task": "mache was"}), files_context="")
        assert ct.task_id.startswith("cw_")
        assert ct.status == "running"

        summaries = await proxy._await_bg_tasks([ct.task_id], timeout_seconds=1.0)
        assert len(summaries) == 1
        assert summaries[0]["status"] == "done"
        assert summaries[0]["result"] == "ergebnis-text"
        assert ct.delivered is True
    asyncio.run(run())


def test_await_unknown_task_id(monkeypatch):
    async def run():
        monkeypatch.setattr(proxy, "_COWORKER_BG_TASKS", {})
        summaries = await proxy._await_bg_tasks(["cw_doesnotexist"], timeout_seconds=0.1)
        assert summaries[0]["status"] == "unknown"
    asyncio.run(run())


def test_await_running_stays_undelivered(monkeypatch):
    """Laufende Tasks werden nach Timeout als running gemeldet, nicht delivered."""
    async def run():
        release = asyncio.Event()

        async def fake_run(task, args):
            await release.wait()
            task.status = "done"
            task.result = "spaet"

        monkeypatch.setattr(proxy, "_run_bg_coworker_task", fake_run)
        store: Dict[str, Any] = {}
        monkeypatch.setattr(proxy, "_COWORKER_BG_TASKS", store)

        ct = proxy._register_bg_dispatch(
            _tc("dispatch_coworker", {"task": "langlaeufer"}), files_context="")
        summaries = await proxy._await_bg_tasks([ct.task_id], timeout_seconds=0.05)
        assert summaries[0]["status"] == "running"
        assert ct.delivered is False

        release.set()
        await asyncio.wait_for(ct.aio_task, timeout=1.0)
        assert ct.status == "done"
    asyncio.run(run())


def test_ttl_expiry_cancels_running(monkeypatch):
    async def run():
        store: Dict[str, Any] = {}

        async def never_finish(task, args):
            await asyncio.sleep(30)

        monkeypatch.setattr(proxy, "_run_bg_coworker_task", never_finish)
        monkeypatch.setattr(proxy, "_COWORKER_BG_TASKS", store)
        monkeypatch.setattr(proxy, "COWORKER_BG_TTL", 0.05)

        ct = proxy._register_bg_dispatch(
            _tc("dispatch_coworker", {"task": "endlos"}), files_context="")
        await asyncio.sleep(0.1)
        proxy._cleanup_bg_tasks()
        assert ct.status == "expired"
        with pytest.raises(asyncio.CancelledError):
            await ct.aio_task
    asyncio.run(run())


def test_cleanup_removes_delivered_after_60s(monkeypatch):
    store: Dict[str, Any] = {}
    monkeypatch.setattr(proxy, "_COWORKER_BG_TASKS", store)
    monkeypatch.setattr(proxy, "COWORKER_MAX_PARALLEL", 8)

    ct = proxy.CoworkerTask(task_id="cw_old", preview="alt",
                            created_at=time.time() - 500)
    ct.status = "done"
    ct.delivered = True
    ct.finished_at = time.time() - 120
    store["cw_old"] = ct

    proxy._cleanup_bg_tasks()
    assert "cw_old" not in store


def test_status_line_format(monkeypatch):
    store: Dict[str, Any] = {}
    monkeypatch.setattr(proxy, "_COWORKER_BG_TASKS", store)

    t1 = proxy.CoworkerTask(task_id="cw_a", preview="task a", created_at=time.time())
    t1.status = "done"
    t1.result = "x"
    t2 = proxy.CoworkerTask(task_id="cw_b", preview="task b", created_at=time.time())
    t2.status = "running"
    store["cw_a"] = t1
    store["cw_b"] = t2

    line = proxy._coworker_status_line()
    assert line is not None
    assert "cw_a" in line and "cw_b" in line
    assert "collect_coworker" in line
    assert "✅" in line and "⏳" in line


def test_status_line_empty(monkeypatch):
    monkeypatch.setattr(proxy, "_COWORKER_BG_TASKS", {})
    assert proxy._coworker_status_line() is None


# ── Injektion ──────────────────────────────────────────────────────────────

def test_partition_four_buckets_real_proxy():
    calls = [
        _tc("dispatch_coworker", {"task": "a"}, tid="1"),
        _tc("collect_coworker", {}, tid="2"),
        _tc("ask_coworker", {"task": "c"}, tid="3"),
        _tc("read_file", {"path": "x"}, tid="4"),
    ]
    d, c, a, o = proxy._partition_tool_calls(calls)
    assert [t["id"] for t in d] == ["1"]
    assert [t["id"] for t in c] == ["2"]
    assert [t["id"] for t in a] == ["3"]
    assert [t["id"] for t in o] == ["4"]


def test_inject_fork_join_off(monkeypatch):
    monkeypatch.setattr(proxy, "COWORKER_ENABLED", True)
    monkeypatch.setattr(proxy, "COWORKER_FORK_JOIN", False)
    monkeypatch.setattr(proxy, "_MODEL_CATEGORIES", {
        **proxy._MODEL_CATEGORIES,
        "coworker": {"api_url": "http://x:1/v1/chat/completions", "api_key": "",
                     "model_name": "m", "max_tokens": 4096, "is_vision": False,
                     "timeout_seconds": 300},
    })
    monkeypatch.setattr(proxy, "_COWORKER_HEALTH_CACHE", {
        "reachable": True, "checked_at": time.time(), "last_error": ""})
    payload = {"messages": []}
    assert proxy._inject_coworker_tool(payload) is True
    names = [t["function"]["name"] for t in payload["tools"]]
    assert names == ["ask_coworker"]


# ── _delegation_loop: dispatch + collect Fluesse ────────────────────────────

def _mk_outcome(content: str = "", tool_calls: Optional[List[Dict[str, Any]]] = None):
    return {"result": {"content": content, "tool_calls": tool_calls},
            "all_failed": False, "used_idx": 0, "used_model": "m", "attempts": []}


@pytest.fixture(autouse=True)
def _reset_bg_store(monkeypatch):
    """Isoliert den globalen Task-Store zwischen Tests."""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    mp.setattr(proxy, "_COWORKER_BG_TASKS", {})
    yield
    mp.undo()


def test_delegation_loop_dispatch_then_final(monkeypatch):
    """Runde 1: dispatch → mini-result; Runde 2: finale Antwort."""
    async def run():
        body = {"messages": [{"role": "user", "content": "tu was"}]}
        calls = {"n": 0}

        async def fake_fallbacks(b, cat, force_start_idx=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return _mk_outcome("", [
                    _tc("dispatch_coworker", {"task": "review file"}, tid="d1")])
            return _mk_outcome("fertig ohne tools")

        async def fake_bg(task, args):
            task.status = "done"
            task.result = "bg-ergebnis"
            task.finished_at = time.time()

        monkeypatch.setattr(proxy, "_call_model_with_fallbacks", fake_fallbacks)
        monkeypatch.setattr(proxy, "_run_bg_coworker_task", fake_bg)
        monkeypatch.setattr(proxy, "COWORKER_FORK_JOIN", True)
        monkeypatch.setattr(proxy, "COWORKER_ENABLED", True)
        monkeypatch.setattr(proxy, "COWORKER_MAX_DELEGATIONS", 2)
        monkeypatch.setattr(proxy, "COWORKER_DISPATCH_CAP", 12)

        outcome = await proxy._delegation_loop(body, "local")
        assert calls["n"] == 2
        assert outcome["result"]["content"] == "fertig ohne tools"
        msgs = body["messages"]
        # assistant-tool_calls + tool-mini-result in der History
        roles = [m["role"] for m in msgs]
        assert "assistant" in roles and "tool" in roles
        tool_msg = [m for m in msgs if m["role"] == "tool"][0]
        payload = json.loads(tool_msg["content"])
        assert payload["status"] == "dispatched"
        assert payload["task_id"].startswith("cw_")
        # BG-Task ist registriert und (durch fake) fertig
        assert len(proxy._COWORKER_BG_TASKS) == 1
    asyncio.run(run())


def test_delegation_loop_collect(monkeypatch):
    """collect_coworker blockt und schreibt Zusammenfassungen als tool-msg."""
    async def run():
        body = {"messages": [{"role": "user", "content": "sammle"}]}
        calls = {"n": 0}

        async def fake_fallbacks(b, cat, force_start_idx=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return _mk_outcome("", [
                    _tc("collect_coworker", {"task_ids": ["cw_1"]}, tid="c1")])
            return _mk_outcome("alles gesammelt")

        async def fake_await(task_ids, timeout_seconds=600.0):
            assert task_ids == ["cw_1"]
            return [{"task_id": "cw_1", "status": "done", "result": "r1"}]

        monkeypatch.setattr(proxy, "_call_model_with_fallbacks", fake_fallbacks)
        monkeypatch.setattr(proxy, "_await_bg_tasks", fake_await)
        monkeypatch.setattr(proxy, "COWORKER_FORK_JOIN", True)
        monkeypatch.setattr(proxy, "COWORKER_ENABLED", True)
        monkeypatch.setattr(proxy, "COWORKER_MAX_DELEGATIONS", 2)

        outcome = await proxy._delegation_loop(body, "local")
        assert outcome["result"]["content"] == "alles gesammelt"
        tool_msgs = [m for m in body["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        summaries = json.loads(tool_msgs[0]["content"])
        assert summaries[0]["task_id"] == "cw_1"
        assert summaries[0]["result"] == "r1"
    asyncio.run(run())


def test_delegation_loop_dispatch_with_vstools_forwards(monkeypatch):
    """dispatch + read_file im selben Turn: read_file wird durchgereicht."""
    async def run():
        body = {"messages": [{"role": "user", "content": "dispatch und lies"}]}

        async def fake_fallbacks(b, cat, force_start_idx=None):
            return _mk_outcome("", [
                _tc("dispatch_coworker", {"task": "hintergrund"}, tid="d1"),
                _tc("read_file", {"path": "a.py"}, tid="r1"),
            ])

        async def fake_bg(task, args):
            task.status = "done"
            task.result = "ok"
            task.finished_at = time.time()

        monkeypatch.setattr(proxy, "_call_model_with_fallbacks", fake_fallbacks)
        monkeypatch.setattr(proxy, "_run_bg_coworker_task", fake_bg)
        monkeypatch.setattr(proxy, "COWORKER_FORK_JOIN", True)
        monkeypatch.setattr(proxy, "COWORKER_ENABLED", True)
        monkeypatch.setattr(proxy, "COWORKER_MAX_DELEGATIONS", 2)
        monkeypatch.setattr(proxy, "COWORKER_DISPATCH_CAP", 12)

        outcome = await proxy._delegation_loop(body, "local")
        fw = outcome["result"]["tool_calls"]
        assert len(fw) == 1
        assert fw[0]["function"]["name"] == "read_file"
    asyncio.run(run())


def test_delegation_loop_dispatch_cap(monkeypatch):
    """Ueber-Cap dispatches werden blockt (Hinweis, kein Task angelegt)."""
    async def run():
        body = {"messages": [{"role": "user", "content": "flood"}]}
        calls = {"n": 0}
        registered: List[str] = []

        def fake_register(tc, files_context):
            registered.append(tc["id"])
            t = proxy.CoworkerTask(task_id=f"cw_{len(registered)}",
                                   preview="p", created_at=time.time())
            return t

        async def fake_fallbacks(b, cat, force_start_idx=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return _mk_outcome("", [
                    _tc("dispatch_coworker", {"task": "t1"}, tid="d1")])
            if calls["n"] == 2:
                # Modell ignoriert den Hinweis und dispatched erneut
                return _mk_outcome("", [
                    _tc("dispatch_coworker", {"task": "t2"}, tid="d2")])
            return _mk_outcome("ok aufgabe selbst erledigt")

        monkeypatch.setattr(proxy, "_call_model_with_fallbacks", fake_fallbacks)
        monkeypatch.setattr(proxy, "_register_bg_dispatch", fake_register)
        monkeypatch.setattr(proxy, "COWORKER_FORK_JOIN", True)
        monkeypatch.setattr(proxy, "COWORKER_ENABLED", True)
        monkeypatch.setattr(proxy, "COWORKER_MAX_DELEGATIONS", 3)
        monkeypatch.setattr(proxy, "COWORKER_DISPATCH_CAP", 1)

        outcome = await proxy._delegation_loop(body, "local")
        # Cap-Pfad: 1. dispatch registriert, 2. blockt → Hinweis → final
        assert registered == ["d1"]
    asyncio.run(run())


def test_register_bg_dispatch_real_semantics(monkeypatch):
    """Kein Mock auf _register_bg_dispatch: prueft task_id + Store-Eintrag."""
    async def run():
        store: Dict[str, proxy.CoworkerTask] = {}

        async def fake_bg(task, args):
            task.status = "done"
            task.result = "store-check"
            task.finished_at = time.time()

        monkeypatch.setattr(proxy, "_run_bg_coworker_task", fake_bg)
        monkeypatch.setattr(proxy, "_COWORKER_BG_TASKS", store)
        monkeypatch.setattr(proxy, "COWORKER_RESULT_CAP", 12000)
        ct = proxy._register_bg_dispatch(
            _tc("dispatch_coworker", {"task": "pruefe store"}), files_context="ctx")
        assert ct.task_id in store
        assert ct.file_context == "ctx"
        await asyncio.wait_for(ct.aio_task, timeout=1.0)
        assert ct.status == "done"
        assert ct.result == "store-check"
    asyncio.run(run())
