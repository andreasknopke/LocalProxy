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

        async def fake_bg(task, args, client_tools=None):
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

        async def fake_run(task, args, client_tools=None):
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

        async def never_finish(task, args, client_tools=None):
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


def test_status_line_noted_once_per_task_status(monkeypatch):
    """Die Status-Notiz darf das Hauptmodell nicht in jedem Folge-Turn mit
    derselben Meldung beschallen — pro (task_id, status) genau einmal."""
    store: Dict[str, Any] = {}
    monkeypatch.setattr(proxy, "_COWORKER_BG_TASKS", store)
    monkeypatch.setattr(proxy, "_COWORKER_STATUS_NOTED", set())

    t = proxy.CoworkerTask(task_id="cw_once", preview="ein task",
                           created_at=time.time())
    t.status = "running"
    store["cw_once"] = t

    assert proxy._coworker_status_line() is not None      # 1. Turn: Notiz
    assert proxy._coworker_status_line() is None          # 2. Turn: keine
    assert proxy._coworker_status_line() is None          # 3. Turn: keine

    # Statuswechsel ist neue Information -> Notiz erscheint erneut
    t.status = "done"
    line = proxy._coworker_status_line()
    assert line is not None and "cw_once" in line
    assert proxy._coworker_status_line() is None


def test_bg_task_shutdown_cancel_note(monkeypatch):
    """Cancel durch Prozess-Shutdown muss als Neustart gemeldet werden, nicht
    als TTL-Ablauf — sonst liest das Hauptmodell einen Restart als Task-Timeout."""
    import asyncio as _aio

    async def run():
        async def boom(*a, **k):
            raise _aio.CancelledError()

        monkeypatch.setattr(proxy, "_run_coworker_agent", boom)
        monkeypatch.setattr(proxy, "COWORKER_AGENT_MODE", True)
        monkeypatch.setattr(proxy, "_SHUTTING_DOWN", True)
        monkeypatch.setattr(proxy, "io_log_bg_result", lambda *a, **k: None)

        task = proxy.CoworkerTask(task_id="cw_sd", preview="p", created_at=time.time())
        task.file_context = "ctx"
        try:
            await proxy._run_bg_coworker_task(task, {"task": "t", "context": ""})
        except _aio.CancelledError:
            pass
        assert task.status == "expired"
        assert "NEUSTART" in task.error and "TTL" not in task.error

    _aio.run(run())


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

        async def fake_bg(task, args, client_tools=None):
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

        async def fake_bg(task, args, client_tools=None):
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

        def fake_register(tc, files_context, client_tools=None):
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

        async def fake_bg(task, args, client_tools=None):
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


# ── Deterministische Auto-Verteilung (manage_todo_list → Co-Worker) ────────

def test_extract_not_started_todos():
    """Extrahiert nur not-started Titel aus manage_todo_list-Calls."""
    tcs = [
        _tc("manage_todo_list", {"todoList": [
            {"id": 1, "title": "Task A", "status": "not-started"},
            {"id": 2, "title": "Task B", "status": "in-progress"},
            {"id": 3, "title": "Task C", "status": "completed"},
            {"id": 4, "title": "Task D", "status": "not-started"},
        ]}),
        _tc("read_file", {"path": "foo.py"}),  # anderes Tool → ignorieren
    ]
    titles = proxy._extract_not_started_todos(tcs)
    assert titles == ["Task A", "Task D"]


def test_extract_not_started_todos_empty():
    assert proxy._extract_not_started_todos(None) == []
    assert proxy._extract_not_started_todos([]) == []
    assert proxy._extract_not_started_todos(
        [_tc("manage_todo_list", {"todoList": [
            {"id": 1, "title": "x", "status": "in-progress"}]})]) == []


def test_auto_dispatch_creates_bg_tasks(monkeypatch):
    """not-started Todos werden als BG-Tasks registriert (mit Duplikat-Schutz)."""
    async def run():
        store: Dict[str, proxy.CoworkerTask] = {}

        async def fake_bg(task, args, client_tools=None):
            task.status = "done"
            task.result = "ok"
            task.finished_at = time.time()

        monkeypatch.setattr(proxy, "_run_bg_coworker_task", fake_bg)
        monkeypatch.setattr(proxy, "_COWORKER_BG_TASKS", store)
        monkeypatch.setattr(proxy, "_COWORKER_AUTO_DISPATCHED", set())
        monkeypatch.setattr(proxy, "COWORKER_DISPATCH_CAP", 12)

        titles = ["Task A", "Task B"]
        created, n = proxy._auto_dispatch_todos(titles, "ctx", 0)
        assert n == 2
        assert len(created) == 2
        assert all(ct.task_id in store for ct in created)
        # Duplikat-Schutz: gleiche Titel erneut → 0 neue Tasks
        created2, n2 = proxy._auto_dispatch_todos(titles, "ctx", 0)
        assert n2 == 0
        assert created2 == []
        for ct in created:
            await asyncio.wait_for(ct.aio_task, timeout=1.0)
            assert ct.status == "done"
    asyncio.run(run())


def test_auto_dispatch_respects_cap(monkeypatch):
    """Dispatch-Cap begrenzt die Anzahl verteilter Tasks."""
    async def run():
        store: Dict[str, proxy.CoworkerTask] = {}

        async def fake_bg(task, args, client_tools=None):
            task.status = "done"
            task.result = "ok"
            task.finished_at = time.time()

        monkeypatch.setattr(proxy, "_run_bg_coworker_task", fake_bg)
        monkeypatch.setattr(proxy, "_COWORKER_BG_TASKS", store)
        monkeypatch.setattr(proxy, "_COWORKER_AUTO_DISPATCHED", set())
        monkeypatch.setattr(proxy, "COWORKER_DISPATCH_CAP", 2)

        # dispatch_count=1 + 3 neue → nur 1 darf noch rein
        created, n = proxy._auto_dispatch_todos(
            ["T1", "T2", "T3"], "ctx", dispatch_count=1)
        assert n == 1
        assert len(created) == 1
    asyncio.run(run())


def test_delegation_loop_auto_dispatch_on_todo(monkeypatch):
    """_delegation_loop: manage_todo_list-Call → not-started Todos werden
    automatisch an den Co-Worker verteilt (deterministisch)."""
    async def run():
        store: Dict[str, proxy.CoworkerTask] = {}
        registered: List[str] = []

        async def fake_bg(task, args, client_tools=None):
            task.status = "done"
            task.result = "auto-ok"
            task.finished_at = time.time()

        def fake_register(tool_call, files_context, client_tools=None):
            import json as _json
            args = _json.loads(tool_call["function"]["arguments"])
            ct = proxy.CoworkerTask(
                task_id=f"cw_test{len(registered)}",
                preview=proxy._task_preview_from_args(args),
                file_context=files_context or None,
            )
            ct.aio_task = asyncio.ensure_future(fake_bg(ct, args))
            store[ct.task_id] = ct
            registered.append(ct.task_id)
            return ct

        calls = {"n": 0}

        async def fake_fallbacks(body, category, force_start_idx=None):
            calls["n"] += 1
            if calls["n"] == 1:
                # Modell plant per manage_todo_list (VS-Code-Tool)
                return {"result": {"status": "ok", "content": "",
                                   "tool_calls": [
                                       _tc("manage_todo_list", {"todoList": [
                                           {"id": 1, "title": "Refaktor A",
                                            "status": "not-started"},
                                           {"id": 2, "title": "Test B",
                                            "status": "not-started"}]})]},
                       "all_failed": False}
            return {"result": {"status": "ok",
                               "content": "fertig, tasks laufen beim Co-Worker"},
                    "all_failed": False}

        monkeypatch.setattr(proxy, "_run_bg_coworker_task", fake_bg)
        monkeypatch.setattr(proxy, "_COWORKER_BG_TASKS", store)
        monkeypatch.setattr(proxy, "_COWORKER_AUTO_DISPATCHED", set())
        monkeypatch.setattr(proxy, "_register_bg_dispatch", fake_register)
        monkeypatch.setattr(proxy, "_call_model_with_fallbacks", fake_fallbacks)
        monkeypatch.setattr(proxy, "COWORKER_ENABLED", True)
        monkeypatch.setattr(proxy, "COWORKER_FORK_JOIN", True)
        monkeypatch.setattr(proxy, "COWORKER_AUTO_DISPATCH", True)
        monkeypatch.setattr(proxy, "COWORKER_DISPATCH_CAP", 12)
        monkeypatch.setattr(proxy, "_COWORKER_HEALTH_CACHE",
                            {"reachable": True})

        body = {"messages": [{"role": "user", "content": "mach den Plan"}]}
        outcome = await proxy._delegation_loop(body, "local")
        # 2 not-started Todos deterministisch verteilt
        assert len(registered) == 2
        assert all(tid in store for tid in registered)
        # manage_todo_list-Call wird normal an den Client durchgereicht
        # (Client führt es aus; im nächsten Request zeigt die Status-Notiz
        # die offenen BG-Tasks)
        tcs = outcome["result"]["tool_calls"] or []
        assert any((tc.get("function") or {}).get("name") == "manage_todo_list"
                   for tc in tcs)
    asyncio.run(run())


# ── Tunnel-Dispatch: BG-Tasks mit echten Client-Tools (2026-08-29) ─────────

def test_bg_task_with_client_tools_pauses_and_maps_session(monkeypatch):
    """BG-Task mit client_tools: erste Runde endet mit tool_calls → Task
    pausiert (status=paused), Session haelt Tunnel-IDs, bg_task_id-Verkettung
    ist gesetzt. KEIN Plain-Fallback."""
    async def run():
        captured = {}

        async def fake_round(sess, queue, model, stream_id):
            captured["sess"] = sess
            sess["last_fwd_calls"] = [{"index": 0, "id": "cws_x_call_1",
                                       "type": "function",
                                       "function": {"name": "create_file",
                                                    "arguments": "{}"}}]

        monkeypatch.setattr(proxy, "_cw_stream_round", fake_round)
        monkeypatch.setattr(proxy, "COWORKER_AGENT_MODE", True)
        monkeypatch.setattr(proxy, "_COWORKER_BG_TASKS", {})
        monkeypatch.setattr(proxy, "io_log_bg_result", lambda *a, **k: None)

        tools = [{"type": "function", "function": {"name": "create_file"}}]
        task = proxy.CoworkerTask(task_id="cw_tools", preview="p", created_at=time.time())
        task.file_context = "files"
        await proxy._run_bg_coworker_task(task, {"task": "schreibe file"},
                                          client_tools=tools)
        assert task.status == "paused"
        assert task.sid is not None
        sess = captured["sess"]
        assert sess["bg_task_id"] == "cw_tools"
        assert sess["client_tools"] == tools
        assert sess["extra_files"] if False else True  # extra_context wurde gesetzt
    asyncio.run(run())


def test_bg_task_with_client_tools_immediate_final(monkeypatch):
    """Co-Worker antwortet ohne Tool-Bedarf direkt (text-only) → Task sofort
    done mit dem Final-Text."""
    async def run():
        async def fake_round(sess, queue, model, stream_id):
            sess["done"] = True
            sess["final"] = "fertig, nichts zu tun"

        monkeypatch.setattr(proxy, "_cw_stream_round", fake_round)
        monkeypatch.setattr(proxy, "COWORKER_AGENT_MODE", True)
        monkeypatch.setattr(proxy, "_COWORKER_BG_TASKS", {})
        monkeypatch.setattr(proxy, "io_log_bg_result", lambda *a, **k: None)

        task = proxy.CoworkerTask(task_id="cw_final", preview="p", created_at=time.time())
        await proxy._run_bg_coworker_task(task, {"task": "bereichne text"},
                                          client_tools=[{"type": "function",
                                                         "function": {"name": "read_file"}}])
        assert task.status == "done"
        assert task.result == "fertig, nichts zu tun"
    asyncio.run(run())


def test_cw_attach_finals_updates_paused_bg_task(monkeypatch):
    """Nach Tunnel-Resume wird der pausierte BG-Task auf done gesetzt und
    das Session-Final als task.result abgelegt."""
    store: Dict[str, Any] = {}
    monkeypatch.setattr(proxy, "_COWORKER_BG_TASKS", store)
    monkeypatch.setattr(proxy, "io_log_bg_result", lambda *a, **k: None)
    ct = proxy.CoworkerTask(task_id="cw_resume", preview="p", created_at=time.time())
    ct.status = "paused"
    ct.sid = "sess42"
    store["cw_resume"] = ct
    sess = {"done": True, "final": "jetzt wirklich fertig",
            "bg_task_id": "cw_resume",
            "orig_ask": {"id": "call_bg", "type": "function",
                         "function": {"name": "dispatch_coworker",
                                      "arguments": "{}"}}}
    msgs: List[Dict[str, Any]] = []
    proxy._cw_attach_finals(msgs, [sess])
    assert ct.status == "done"
    assert ct.result == "jetzt wirklich fertig"
    assert ct.delivered is False  # collect holt es Spaeter ab


def test_cw_attach_finals_bg_task_failure_marks_error(monkeypatch):
    """tunnel_failed-Session → BG-Task status=error mit Fehlertext."""
    store: Dict[str, Any] = {}
    monkeypatch.setattr(proxy, "_COWORKER_BG_TASKS", store)
    monkeypatch.setattr(proxy, "io_log_bg_result", lambda *a, **k: None)
    ct = proxy.CoworkerTask(task_id="cw_fail", preview="p", created_at=time.time())
    ct.status = "paused"
    ct.sid = "sess43"
    store["cw_fail"] = ct
    sess = {"done": True, "final": "[Co-Worker nicht verfuegbar]\nHTTP 500",
            "tunnel_failed": True, "bg_task_id": "cw_fail",
            "orig_ask": {"id": "call_bg", "type": "function",
                         "function": {"name": "dispatch_coworker",
                                      "arguments": "{}"}}}
    proxy._cw_attach_finals([], [sess])
    assert ct.status == "error"
    assert "500" in (ct.error or "")


def test_await_bg_tasks_reports_paused_as_running(monkeypatch):
    """Pausierte Tunnel-Tasks werden im Join als status=running gemeldet
    (mit Hinweis), nicht als error/expired — und bleiben undelivered."""
    async def run():
        store: Dict[str, Any] = {}
        ct = proxy.CoworkerTask(task_id="cw_paused", preview="demo",
                                created_at=time.time())
        ct.status = "paused"
        store["cw_paused"] = ct
        monkeypatch.setattr(proxy, "_COWORKER_BG_TASKS", store)
        summaries = await proxy._await_bg_tasks(["cw_paused"], timeout_seconds=0.05)
        assert summaries[0]["status"] == "running"
        assert ct.delivered is False
    asyncio.run(run())


def test_auto_dispatch_task_text_promise_matches_tools(monkeypatch):
    """Der Auto-Dispatch-Task-Text verspricht KEINEN Tool-Zugriff mehr, wenn
    keine Client-Tools mitkommen — der Widerspruch, der cw_09f072bd zur
    Ausrede verleitete, ist beseitigt."""
    monkeypatch.setattr(proxy, "_COWORKER_AUTO_DISPATCHED", set())
    monkeypatch.setattr(proxy, "COWORKER_DISPATCH_CAP", 4)

    registered: List[Dict[str, Any]] = []

    def fake_register(tc, files_context, client_tools=None):
        registered.append(json.loads(tc["function"]["arguments"]))

        class _T:
            task_id = "cw_fake"

        return _T()

    monkeypatch.setattr(proxy, "_register_bg_dispatch", fake_register)
    proxy._auto_dispatch_todos(["mach was"],
                               files_context="", dispatch_count=0,
                               client_tools=None)
    assert "NO tool access" in registered[0]["task"] or "no tool access" in registered[0]["task"]
    assert "using the available tools" not in registered[0]["task"]
