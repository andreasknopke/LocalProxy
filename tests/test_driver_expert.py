"""Tests fuer das Treiber/Experte-Fabric (Driver-Mode + Praefix-Sharing).

Deckt ab:
- Praefix-Sharing: Datei-Kontext steht VOR der Task-Injection (COWORKER_FILES_FIRST)
  -> parallele Tasks teilen einen byte-identischen Praefix (SGLang RadixAttention)
- Deaktiviertes files_first liefert die alte Reihenfolge (Task zuerst)
- Determinismus: gleicher History-Eintrag -> identischer Praefix (Cache-Treffer)
- Praefix-Sharing gilt auch fuer den Tunnel-Pfad (_cw_session_new)
- Driver-Mode: andere Guidance + eigener Marker, Merge-Idempotenz,
  kein Doppel-Marker beim Umschalten zwischen den Modi
- Semaphore-Gating: ask_coworker laeuft NICHT am COWORKER_MAX_PARALLEL-Limit vorbei
"""
import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_proxy_module():
    spec = importlib.util.spec_from_file_location("proxy_de_test", REPO_ROOT / "proxy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["proxy_de_test"] = module
    spec.loader.exec_module(module)
    return module


proxy = _load_proxy_module()

FILES = "### Datei: a.py\nprint('alpha')\n\n### Datei: b.py\nprint('beta')"


# ── Praefix-Sharing (RadixAttention) ───────────────────────────────────────

def test_build_coworker_body_files_first_by_default(monkeypatch):
    """Datei-Kontext vor der Task-Instruction — Voraussetzung fuer Praefix-Sharing."""
    monkeypatch.setattr(proxy, "COWORKER_FILES_FIRST", True)
    monkeypatch.setattr(proxy, "COWORKER_TASK_CAP", 0)
    body = proxy._build_coworker_body("Refaktoriere Modul A", "", extra_context=FILES)
    user = body["messages"][-1]["content"]
    assert user.index("Dateiinhalte") < user.index("Refaktoriere Modul A")
    assert "## Aufgabe" in user


def test_build_coworker_body_files_last_when_disabled(monkeypatch):
    """files_first=False -> alte Reihenfolge (Task zuerst, Dateien am Ende)."""
    monkeypatch.setattr(proxy, "COWORKER_FILES_FIRST", False)
    monkeypatch.setattr(proxy, "COWORKER_TASK_CAP", 0)
    body = proxy._build_coworker_body("Refaktoriere Modul A", "", extra_context=FILES)
    user = body["messages"][-1]["content"]
    assert user.index("Refaktoriere Modul A") < user.index("Dateiinhalte")


def test_parallel_tasks_share_identical_prefix(monkeypatch):
    """DER Kern des Boosters: zwei Tasks mit gleichem Datei-Kontext ergeben einen
    byte-identischen Praefix (system + Dateien), der sich erst ganz am Ende
    unterscheidet -> der Server prefillt den teuren Teil nur einmal."""
    monkeypatch.setattr(proxy, "COWORKER_FILES_FIRST", True)
    monkeypatch.setattr(proxy, "COWORKER_TASK_CAP", 0)
    b1 = proxy._build_coworker_body("Task EINS", "", extra_context=FILES)
    b2 = proxy._build_coworker_body("Task ZWEI", "", extra_context=FILES)
    m1, m2 = b1["messages"], b2["messages"]

    # system-Message identisch (sonst divergiert der Praefix schon dort)
    assert m1[0] == m2[0]
    u1, u2 = m1[-1]["content"], m2[-1]["content"]
    # laengster gemeinsamer Praefix = alles bis kurz vor das Task-Wort
    common = 0
    for c1, c2 in zip(u1, u2):
        if c1 != c2:
            break
        common += 1
    shared = u1[:common]
    # Der gemeinsame Teil enthaelt die KOMPLETTEN Dateien ...
    assert FILES in shared
    # ... und ist deutlich laenger als der unterscheidende Rest.
    assert len(shared) > 3 * len(u1) // 4, (len(shared), len(u1))


def test_extract_conversation_files_is_deterministic():
    """Praefix-Sharing setzt eine stabile Dateireihenfolge voraus (kein set-Iteration)."""
    msgs = [
        {"role": "tool", "name": "read_file", "content": "INHALT A"},
        {"role": "tool", "name": "read_file", "content": "INHALT B"},
        {"role": "tool", "name": "read_file", "content": "INHALT A"},
    ]
    first = proxy._extract_conversation_files(msgs, 60000)
    for _ in range(3):
        assert proxy._extract_conversation_files(msgs, 60000) == first
    # Dedup erhaelt die Reihenfolge der Erstbegegnung
    assert first.index("INHALT A") < first.index("INHALT B")


def test_tunnel_session_uses_files_first(monkeypatch):
    """Der Tunnel-Pfad (agentischer Coworker) nutzt dieselbe Reihenfolge."""
    monkeypatch.setattr(proxy, "COWORKER_FILES_FIRST", True)
    monkeypatch.setattr(proxy, "COWORKER_TASK_CAP", 0)
    sess = proxy._cw_session_new("Bau Feature X", "Kontext Y", extra_context=FILES)
    user = sess["messages"][-1]["content"]
    assert user.index("Dateiinhalte") < user.index("Bau Feature X")


def test_tunnel_session_files_first_when_disabled(monkeypatch):
    monkeypatch.setattr(proxy, "COWORKER_FILES_FIRST", False)
    monkeypatch.setattr(proxy, "COWORKER_TASK_CAP", 0)
    sess = proxy._cw_session_new("Bau Feature X", "Kontext Y", extra_context=FILES)
    user = sess["messages"][-1]["content"]
    assert user.index("Bau Feature X") < user.index("Dateiinhalte")


# ── Driver-Mode Guidance ───────────────────────────────────────────────────

def _setup_coworker_on(monkeypatch):
    monkeypatch.setattr(proxy, "COWORKER_ENABLED", True)
    monkeypatch.setattr(proxy, "COWORKER_FORK_JOIN", True)
    monkeypatch.setattr(proxy, "COWORKER_TEACH_DELEGATION", True)
    monkeypatch.setattr(proxy, "_coworker_configured", lambda: True)
    monkeypatch.setattr(proxy, "_COWORKER_HEALTH_CACHE", {"reachable": True})


def test_capacity_note_reflects_max_parallel(monkeypatch):
    """Die Kapazitaets-Info nennt die ECHTE Concurrency. Bei max_parallel=1
    warnt sie ausdruecklich vor falschem Fan-out (kein Speedup, queue)."""
    monkeypatch.setattr(proxy, "COWORKER_MAX_PARALLEL", 1)
    monkeypatch.setattr(proxy, "COWORKER_DISPATCH_CAP", 12)
    note = proxy._coworker_capacity_note()
    assert "at most 1 task" in note
    assert "NO speedup" in note
    assert "READ-ONLY" in note
    monkeypatch.setattr(proxy, "COWORKER_MAX_PARALLEL", 4)
    note4 = proxy._coworker_capacity_note()
    assert "at most 4 task" in note4
    assert "run in parallel" in note4


def test_inject_appends_capacity_note_to_dispatch_and_guidance(monkeypatch):
    """Die Kapazitaets-Info landet sowohl in der Dispatch-Tool-Beschreibung
    als auch in der Guidance-System-Message."""
    _setup_coworker_on(monkeypatch)
    monkeypatch.setattr(proxy, "COWORKER_DRIVER_MODE", True)
    monkeypatch.setattr(proxy, "COWORKER_MAX_PARALLEL", 1)
    payload = {"messages": [{"role": "system", "content": "You are Copilot."}],
               "tools": []}
    assert proxy._inject_coworker_tool(payload) is True
    names = [t["function"]["name"] for t in payload["tools"]]
    disp = next(t for t in payload["tools"]
                if t["function"]["name"] == "dispatch_coworker")
    assert "[CO-WORKER CAPACITY & PIPELINE]" in disp["function"]["description"]
    assert "at most 1 task" in disp["function"]["description"]
    assert "[CO-WORKER CAPACITY & PIPELINE]" in payload["messages"][0]["content"]


def test_driver_mode_guidance_replaces_default(monkeypatch):
    """Im Driver-Mode lehrt die Guidance das Treiber-Rollenbild, nicht das
    Delegations-Rollenbild (sonst delegiert der schnelle Treiber alles)."""
    _setup_coworker_on(monkeypatch)
    monkeypatch.setattr(proxy, "COWORKER_DRIVER_MODE", True)
    payload = {"messages": [{"role": "system", "content": "You are GitHub Copilot."}]}
    assert proxy._inject_coworker_tool(payload) is True
    sys_c = payload["messages"][0]["content"]
    assert proxy.COWORKER_DRIVER_GUIDANCE_MARKER in sys_c
    assert "[PROXY DELEGATION GUIDANCE]" not in sys_c
    assert "FAST DRIVER" in sys_c
    assert "DELEGATE FIRST" in sys_c
    # Die Guidance muss erlauben, nicht verbieten — ein 30B-Treiber hoert
    # "lieber nicht" und delegiert dann gar nicht (Evidenz 2026-08-28).
    assert "expected, good work, not an exception" in sys_c
    assert "Patterns to avoid" not in sys_c


def test_default_mode_keeps_delegation_guidance(monkeypatch):
    _setup_coworker_on(monkeypatch)
    monkeypatch.setattr(proxy, "COWORKER_DRIVER_MODE", False)
    payload = {"messages": [{"role": "system", "content": "You are GitHub Copilot."}]}
    proxy._inject_coworker_tool(payload)
    sys_c = payload["messages"][0]["content"]
    assert "[PROXY DELEGATION GUIDANCE]" in sys_c
    assert proxy.COWORKER_DRIVER_GUIDANCE_MARKER not in sys_c


def test_driver_mode_guidance_idempotent(monkeypatch):
    _setup_coworker_on(monkeypatch)
    monkeypatch.setattr(proxy, "COWORKER_DRIVER_MODE", True)
    payload = {"messages": [
        {"role": "system", "content": "You are GitHub Copilot."},
        {"role": "user", "content": "hi"},
    ]}
    proxy._inject_coworker_tool(payload)
    proxy._inject_coworker_tool(payload)
    assert payload["messages"][0]["content"].count(
        proxy.COWORKER_DRIVER_GUIDANCE_MARKER) == 1
    assert len(payload["messages"]) == 2


def test_mode_switch_does_not_stack_both_guidances(monkeypatch):
    """Umschalten von driver_mode innerhalb einer bestehenden History darf nicht
    beide Anleitungen in dieselbe system-Message schreiben."""
    _setup_coworker_on(monkeypatch)
    payload = {"messages": [
        {"role": "system", "content": "You are GitHub Copilot."},
        {"role": "user", "content": "hi"},
    ]}
    monkeypatch.setattr(proxy, "COWORKER_DRIVER_MODE", True)
    proxy._inject_coworker_tool(payload)
    monkeypatch.setattr(proxy, "COWORKER_DRIVER_MODE", False)
    proxy._inject_coworker_tool(payload)
    sys_c = payload["messages"][0]["content"]
    assert sys_c.count(proxy.COWORKER_DRIVER_GUIDANCE_MARKER) == 1
    assert "[PROXY DELEGATION GUIDANCE]" not in sys_c
    assert len(payload["messages"]) == 2


def test_driver_mode_no_client_system_inserts_at_zero(monkeypatch):
    _setup_coworker_on(monkeypatch)
    monkeypatch.setattr(proxy, "COWORKER_DRIVER_MODE", True)
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    proxy._inject_coworker_tool(payload)
    assert payload["messages"][0]["role"] == "system"
    assert proxy.COWORKER_DRIVER_GUIDANCE_MARKER in payload["messages"][0]["content"]


def test_io_trace_recognises_driver_marker(monkeypatch, tmp_path):
    """io_trace_analyze muss guidance_in_system auch im Driver-Mode melden,
    sonst zeigt proxy-status.ps1 '-Streams' Guidance=nein despite Injektion."""
    monkeypatch.setattr(proxy, "IO_TRACE_DIR", tmp_path / "io_traces")
    monkeypatch.setattr(proxy, "IO_TRACE_ENABLED", True)
    proxy._ctx_turn_id.set("")
    turn_id = proxy.io_start_turn("local")
    proxy.io_log_outbound(
        {"messages": [
            {"role": "system",
             "content": "Copilot prompt\n\n" + proxy.COWORKER_DRIVER_GUIDANCE_MARKER},
        ], "model": "m"},
        category="local", model="nemotron", req_id="r1")
    proxy.io_end_turn()
    a = proxy.io_trace_analyze(turn_id)
    assert a["guidance_in_system"] is True


# ── Position der Tools + Reihenfolge der Guidance ──────────────────────────

def test_coworker_tools_are_prepended_not_appended(monkeypatch):
    """Mit 56 Client-Tools standen die Delegationstools auf Index 56-58 und
    wurden in 65 Turns NIE gewaehlt. Sie muessen vorn stehen."""
    _setup_coworker_on(monkeypatch)
    monkeypatch.setattr(proxy, "COWORKER_DRIVER_MODE", True)
    client_tools = [{"type": "function",
                     "function": {"name": f"vs_tool_{i}", "parameters": {}}}
                    for i in range(56)]
    payload = {"messages": [{"role": "user", "content": "hi"}],
               "tools": client_tools}
    assert proxy._inject_coworker_tool(payload) is True
    names = [t["function"]["name"] for t in payload["tools"]]
    assert names[:3] == ["ask_coworker", "dispatch_coworker", "collect_coworker"]
    assert len(names) == 59  # nichts dupliziert/verloren


def test_execution_rules_merged_before_delegation_guidance(monkeypatch):
    """[EXECUTION RULES] ('prefer acting over drafting') darf NICHT zuletzt
    stehen — es widerspricht dem Delegieren und hat das Modell im Test-Turn
    zum Alleingang gebracht."""
    _setup_coworker_on(monkeypatch)
    monkeypatch.setattr(proxy, "COWORKER_DRIVER_MODE", True)
    payload = {"messages": [{"role": "system", "content": "You are GitHub Copilot."}],
               "tools": [{"type": "function",
                          "function": {"name": "read_file", "parameters": {}}}]}
    proxy._inject_tool_execution_guidance(payload)
    proxy._inject_coworker_tool(payload)
    sys_c = payload["messages"][0]["content"]
    assert sys_c.index("[EXECUTION RULES]") < sys_c.index(
        proxy.COWORKER_DRIVER_GUIDANCE_MARKER)


# ── Big-Build-Nudge (deterministischer Zweitter Trigger) ────────────────────

BIG_BUILD = ("Create a complete 3D horror game in a single HTML file, "
             "incorporating RPG and roguelike elements.")


def test_nudge_appended_to_user_message_on_big_build(monkeypatch):
    _setup_coworker_on(monkeypatch)
    monkeypatch.setattr(proxy, "COWORKER_DRIVER_MODE", True)
    payload = {"messages": [{"role": "user", "content": BIG_BUILD}],
               "tools": [{"type": "function",
                          "function": {"name": "ask_coworker", "parameters": {}}}]}
    assert proxy._inject_coworker_nudge(payload) is True
    assert proxy._COWORKER_NUDGE_MARKER in payload["messages"][0]["content"]


def test_nudge_idempotent(monkeypatch):
    _setup_coworker_on(monkeypatch)
    payload = {"messages": [{"role": "user", "content": BIG_BUILD}],
               "tools": [{"type": "function",
                          "function": {"name": "ask_coworker", "parameters": {}}}]}
    proxy._inject_coworker_nudge(payload)
    proxy._inject_coworker_nudge(payload)
    assert payload["messages"][0]["content"].count(proxy._COWORKER_NUDGE_MARKER) == 1


def test_nudge_skipped_for_small_request(monkeypatch):
    _setup_coworker_on(monkeypatch)
    payload = {"messages": [{"role": "user", "content": "fix the typo in README"}],
               "tools": [{"type": "function",
                          "function": {"name": "ask_coworker", "parameters": {}}}]}
    assert proxy._inject_coworker_nudge(payload) is False
    assert proxy._COWORKER_NUDGE_MARKER not in payload["messages"][0]["content"]


def test_big_build_detector_two_axes():
    """Beide Achsen muessen zusammentreffen — ein einzelnes Wort feuert nicht."""
    yes = ["Create a complete 3D horror game in a single HTML file",
           "build me a dashboard for the metrics",
           "implement the auth module from scratch"]
    no = ["fix the typo in README",
          "run the full test suite",          # Umfangs-Wort, aber kein Schaffens-Verb
          "what does this function do?",
          "rename the variable in this file"]
    for t in yes:
        assert proxy._is_big_build_request(t), f"muesste Grossbau sein: {t}"
    for t in no:
        assert not proxy._is_big_build_request(t), f"darf kein Grossbau sein: {t}"


def test_nudge_skipped_without_coworker_tools(monkeypatch):
    """Ohne Tools am Backend wuerde der Hinweis ins Leere fuehren."""
    _setup_coworker_on(monkeypatch)
    payload = {"messages": [{"role": "user", "content": BIG_BUILD}],
               "tools": [{"type": "function",
                          "function": {"name": "read_file", "parameters": {}}}]}
    assert proxy._inject_coworker_nudge(payload) is False


def test_nudge_handles_content_parts(monkeypatch):
    """VS Code schickt user-content als content-Array mit text-Parts."""
    _setup_coworker_on(monkeypatch)
    payload = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": BIG_BUILD},
        {"type": "file", "file": {"path": "a.py", "content": "x"}},
    ]}], "tools": [{"type": "function",
                    "function": {"name": "ask_coworker", "parameters": {}}}]}
    assert proxy._inject_coworker_nudge(payload) is True
    parts = payload["messages"][0]["content"]
    assert proxy._COWORKER_NUDGE_MARKER in parts[0]["text"]
    assert parts[1]["type"] == "file"  # file-Parts bleiben unangetastet


def test_nudge_disabled_by_flag(monkeypatch):
    _setup_coworker_on(monkeypatch)
    monkeypatch.setattr(proxy, "COWORKER_BIG_BUILD_NUDGE", False)
    payload = {"messages": [{"role": "user", "content": BIG_BUILD}],
               "tools": [{"type": "function",
                          "function": {"name": "ask_coworker", "parameters": {}}}]}
    assert proxy._inject_coworker_nudge(payload) is False


def test_nudge_fires_through_build_passthrough(monkeypatch):
    """End-to-End ueber den echten Choke-Point: category=local + Health-OK +
    Big-Build-Auftrag -> Nudge sitzt in der User-Message des Backend-Payloads."""
    _setup_coworker_on(monkeypatch)
    monkeypatch.setattr(proxy, "COWORKER_DRIVER_MODE", True)
    body = {"model": "driver", "messages": [{"role": "user", "content": BIG_BUILD}],
            "tools": [{"type": "function",
                       "function": {"name": "read_file", "parameters": {}}}]}
    out = proxy._build_passthrough_payload(body, "local")
    user = [m for m in out["messages"] if m.get("role") == "user"][-1]
    assert proxy._COWORKER_NUDGE_MARKER in user["content"]
    names = [t["function"]["name"] for t in out["tools"]]
    assert names[0] == "ask_coworker"


# ── Semaphore-Gating (hartes Server-Limit) ─────────────────────────────────

def test_ask_coworker_respects_max_parallel(monkeypatch):
    """ask_coworker lief frueher am Semaphore vorbei. Mit max_parallel=1 muessen
    zwei parallele Calls serialisiert werden — sonst laeuft der Coworker-Server
    (SGLang max_running_requests) ueber."""
    monkeypatch.setattr(proxy, "COWORKER_AGENT_MODE", False)
    monkeypatch.setattr(proxy, "COWORKER_MAX_PARALLEL", 1)
    monkeypatch.setattr(proxy, "_COWORKER_SEMAPHORE", None)
    monkeypatch.setattr(proxy, "COWORKER_RESULT_CAP", 0)

    live = 0
    peak = 0

    async def fake_call(body, category, def_idx=0, inject_hindsight=False):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.02)
        live -= 1
        return {"status": "ok", "content": "fertig"}

    monkeypatch.setattr(proxy, "_call_single_model", fake_call)

    tc = {"id": "c1", "function": {"name": "ask_coworker",
                                   "arguments": json.dumps({"task": "x"})}}

    async def run():
        return await asyncio.gather(*[proxy._run_coworker_call(tc) for _ in range(3)])

    results = asyncio.run(run())
    assert peak == 1, f"max_parallel=1 ignoriert (peak={peak})"
    assert all(r["content"] == "fertig" for r in results)


def test_tunnel_round_respects_max_parallel(monkeypatch):
    """Eine Tunnel-Runde = ein laufender Request; auch sie zaehlt zum Limit."""
    monkeypatch.setattr(proxy, "COWORKER_MAX_PARALLEL", 1)
    monkeypatch.setattr(proxy, "_COWORKER_SEMAPHORE", None)
    monkeypatch.setattr(proxy, "_model_defs",
                        lambda cat: [{"model_name": "expert"}])

    live = 0
    peak = 0

    async def fake_stream(body, category, def_idx=0, inject_hindsight=False):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.02)
        yield {"type": "chunk", "choice": {"delta": {"content": "ok"}}}
        yield {"type": "done"}
        live -= 1

    monkeypatch.setattr(proxy, "_stream_single_model_events", fake_stream)

    sessions = [{"sid": f"s{i}", "rounds": 0, "messages": [], "client_tools": [],
                 "done": False, "final": None, "pending": {}} for i in range(3)]
    state = {"model": "local", "stream_id": "s1"}

    async def run():
        out = []
        async for sse in proxy._stream_coworker_tunnel_phase(sessions, state):
            out.append(sse)
        return out

    asyncio.run(run())
    assert peak == 1, f"max_parallel=1 ignoriert (peak={peak})"


# ── Config-Oberflaeche ─────────────────────────────────────────────────────

def test_webui_defaults_expose_new_coworker_keys():
    """Default-Config (data/config.json ist gitignored, daher der Default als
    massgebliche, versionierte Oberflaeche)."""
    import webui
    cw = webui.DEFAULT_CONFIG["tokens"]["coworker"]
    assert cw["files_first"] is True
    assert cw["driver_mode"] is True
    assert cw["big_build_nudge"] is True
    # SGLang max_running_requests=4 -> einen Slot frei lassen
    assert cw["max_parallel"] == 3


def test_apply_config_file_reads_new_keys(monkeypatch):
    """_apply_config_file muss files_first/driver_mode aus tokens.coworker lesen."""
    cfg = {"tokens": {"coworker": {"files_first": False, "driver_mode": True,
                                   "max_parallel": 2}}}
    monkeypatch.setattr(proxy, "_webui_load_config", lambda: cfg)
    # Rueckgabe sichern: _apply_config_file schreibt auf Modul-Globals.
    before = (proxy.COWORKER_FILES_FIRST, proxy.COWORKER_DRIVER_MODE,
              proxy.COWORKER_MAX_PARALLEL)
    try:
        proxy._apply_config_file()
        assert proxy.COWORKER_FILES_FIRST is False
        assert proxy.COWORKER_DRIVER_MODE is True
        assert proxy.COWORKER_MAX_PARALLEL == 2
    finally:
        (proxy.COWORKER_FILES_FIRST, proxy.COWORKER_DRIVER_MODE,
         proxy.COWORKER_MAX_PARALLEL) = before


def test_webui_defaults_expose_new_keys():
    src = (REPO_ROOT / "webui.py").read_text(encoding="utf-8")
    for needle in ('"files_first": True', '"driver_mode": True',
                   '"big_build_nudge": True',
                   'id="coworker_files_first"', 'id="coworker_driver_mode"',
                   'id="coworker_big_build_nudge"',
                   "files_first: document.getElementById",
                   "driver_mode: document.getElementById",
                   "big_build_nudge: document.getElementById"):
        assert needle in src, f"WebUI-Oberflaeche fehlt: {needle}"
