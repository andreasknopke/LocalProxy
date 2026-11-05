"""Tests fuer den Co-Worker-Session-Schalter (-coworker-on / -coworker-off).

Deckt ab:
- Flag-Erkennung (-coworker-on/-off, --coworker-on, Gross-/Kleinschreibung,
  letztes Vorkommen gewinnt, kein Flag -> None)
- Strippen der Flags aus User-Nachrichten (string + content-Array)
- Override-Map: record/lookup, Ueberschreiben der globalen Einstellung
- _coworker_enabled_effective(): Session-OVERRIDE schlaegt COWORKER_ENABLED
- _coworker_active() / _inject_coworker_tool / _strip_coworker_tool_calls
  respektieren den Override in beide Richtungen
- Conversation-Key bleibt unabhaengig vom Schalter-Flag (History-Stabilitaet)
"""
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_proxy_module():
    spec = importlib.util.spec_from_file_location("proxy_cw_switch", REPO_ROOT / "proxy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["proxy_cw_switch"] = module
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


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Override-Map leer + ContextVar zuruecksetzen - Tests isolieren."""
    monkeypatch.setattr(proxy, "_COWORKER_SESSION_OVERRIDE", {})
    proxy._ctx_coworker_override.set(None)
    yield
    proxy._ctx_coworker_override.set(None)


def _setup_coworker_configured(monkeypatch, reachable: bool = True):
    monkeypatch.setattr(proxy, "_MODEL_CATEGORIES", {
        **proxy._MODEL_CATEGORIES,
        "coworker": {"api_url": "http://spark:30000/v1/chat/completions",
                     "api_key": "", "model_name": "qwen3.8-27b",
                     "max_tokens": 4096, "is_vision": False,
                     "timeout_seconds": 60},
    })
    monkeypatch.setattr(proxy, "COWORKER_ENABLED", True)
    monkeypatch.setattr(proxy, "_COWORKER_HEALTH_CACHE", {
        "reachable": reachable, "checked_at": time.time(), "last_error": "",
    })


# ── Flag-Erkennung ──────────────────────────────────────────────────────────

def test_detect_switch_on():
    assert proxy._detect_coworker_switch_flag("mach das -coworker-on") is True


def test_detect_switch_off():
    assert proxy._detect_coworker_switch_flag("mach das -coworker-off") is False


def test_detect_switch_double_dash_and_uppercase():
    assert proxy._detect_coworker_switch_flag("--COWORKER-ON bitte") is True
    assert proxy._detect_coworker_switch_flag("--Coworker-Off") is False


def test_detect_switch_last_wins():
    assert proxy._detect_coworker_switch_flag(
        "-coworker-on ... und dann -coworker-off") is False
    assert proxy._detect_coworker_switch_flag(
        "-coworker-off ... dann -coworker-on") is True


def test_detect_switch_absent():
    assert proxy._detect_coworker_switch_flag("ganz normaler text") is None
    assert proxy._detect_coworker_switch_flag("") is None


def test_detect_switch_not_confused_with_category_flag():
    # --coworker (Kategorie-Flag) ist KEIN Schalter
    assert proxy._detect_coworker_switch_flag("task --coworker") is None


# ── Strippen ────────────────────────────────────────────────────────────────

def test_strip_switch_from_string_content():
    msgs = [{"role": "user", "content": "bau mir was -coworker-on"}]
    proxy._strip_coworker_switch_flags_from_messages(msgs)
    assert msgs[0]["content"] == "bau mir was"


def test_strip_switch_from_array_content():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "hallo -coworker-off"},
        {"type": "file", "path": "x.py"},
    ]}]
    proxy._strip_coworker_switch_flags_from_messages(msgs)
    assert msgs[0]["content"][0]["text"] == "hallo"
    assert msgs[0]["content"][1]["type"] == "file"


def test_conversation_key_stable_across_switch_flag():
    msgs_a = [{"role": "user", "content": "erstelle ein projekt"}]
    msgs_b = [{"role": "user", "content": "erstelle ein projekt -coworker-on"}]
    assert proxy._conversation_key(msgs_a) == proxy._conversation_key(msgs_b)
    assert proxy._conversation_key(msgs_a) != ""


# ── Override-Map + effektive Aktivierung ────────────────────────────────────

def test_override_record_and_lookup(monkeypatch):
    monkeypatch.setattr(proxy, "COWORKER_ENABLED", False)
    proxy._coworker_override_record("conv1", True)
    assert proxy._coworker_override_lookup("conv1") is True
    assert proxy._coworker_override_lookup("conv2") is None
    proxy._ctx_coworker_override.set(proxy._coworker_override_lookup("conv1"))
    assert proxy._coworker_enabled_effective() is True


def test_override_off_beats_global_on(monkeypatch):
    monkeypatch.setattr(proxy, "COWORKER_ENABLED", True)
    proxy._coworker_override_record("conv1", False)
    proxy._ctx_coworker_override.set(proxy._coworker_override_lookup("conv1"))
    assert proxy._coworker_enabled_effective() is False


def test_no_override_uses_global(monkeypatch):
    monkeypatch.setattr(proxy, "COWORKER_ENABLED", False)
    proxy._ctx_coworker_override.set(None)
    assert proxy._coworker_enabled_effective() is False
    monkeypatch.setattr(proxy, "COWORKER_ENABLED", True)
    assert proxy._coworker_enabled_effective() is True


def test_override_expiry(monkeypatch):
    monkeypatch.setattr(proxy, "COWORKER_SWITCH_TTL_SECONDS", 0.0)
    proxy._coworker_override_record("conv1", True)
    assert proxy._coworker_override_lookup("conv1") is None


def test_override_set_at_recorded():
    before = time.time()
    proxy._coworker_override_record("conv1", True)
    assert proxy._coworker_override_set_at("conv1") >= before
    assert proxy._coworker_override_set_at("unbekannt") == 0.0


# ── Kopplung an die Co-Worker-Pfade ─────────────────────────────────────────

def test_coworker_active_respects_override(monkeypatch):
    _setup_coworker_configured(monkeypatch)
    proxy._coworker_override_record("c", False)
    proxy._ctx_coworker_override.set(False)
    assert proxy._coworker_active() is False
    monkeypatch.setattr(proxy, "COWORKER_ENABLED", False)
    proxy._ctx_coworker_override.set(True)
    assert proxy._coworker_active() is True


def test_inject_tool_global_off_override_on(monkeypatch):
    """Delegation global AUS, Session per -coworker-on AN -> Tools injiziert."""
    _setup_coworker_configured(monkeypatch)
    monkeypatch.setattr(proxy, "COWORKER_ENABLED", False)
    proxy._ctx_coworker_override.set(True)
    payload = {"messages": []}
    assert proxy._inject_coworker_tool(payload) is True
    names = [t["function"]["name"] for t in payload["tools"]]
    assert proxy._COWORKER_TOOL_NAME in names


def test_inject_tool_global_on_override_off(monkeypatch):
    """Delegation global AN, Session per -coworker-off AUS -> keine Tools."""
    _setup_coworker_configured(monkeypatch)
    proxy._ctx_coworker_override.set(False)
    payload = {"messages": []}
    assert proxy._inject_coworker_tool(payload) is False
    assert "tools" not in payload


def test_inject_tool_both_off(monkeypatch):
    """Global AUS + kein Override -> keine Tools (alter Zustand)."""
    _setup_coworker_configured(monkeypatch)
    monkeypatch.setattr(proxy, "COWORKER_ENABLED", False)
    payload = {"messages": []}
    assert proxy._inject_coworker_tool(payload) is False
    assert "tools" not in payload


def _tc(name: str, args: Dict[str, Any], tid: str = "call_x") -> Dict[str, Any]:
    return {"id": tid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def test_strip_tool_calls_respects_override(monkeypatch):
    """Halluzinierter ask_coworker-Call bei global AUS + Override AUS wird
    verworfen; bei Override AN bleibt er stehen."""
    _setup_coworker_configured(monkeypatch)
    calls = [_tc(proxy._COWORKER_TOOL_NAME, {"task": "x"})]
    monkeypatch.setattr(proxy, "COWORKER_ENABLED", False)
    proxy._ctx_coworker_override.set(None)
    assert proxy._strip_coworker_tool_calls(calls) is None
    proxy._ctx_coworker_override.set(True)
    assert proxy._strip_coworker_tool_calls(calls) == calls


def test_probe_skipped_when_globally_off_and_no_override(monkeypatch):
    """Bei global deaktivierter Delegation ohne Session-Override macht der
    Probe keinen HTTP-Call (Health-Check ruht komplett)."""
    import asyncio

    _setup_coworker_configured(monkeypatch)
    monkeypatch.setattr(proxy, "COWORKER_ENABLED", False)
    called = {"n": 0}

    class _NoClient:
        def __init__(self, *a, **k):
            called["n"] += 1

        async def __aenter__(self):
            raise AssertionError("darf nicht aufgerufen werden")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(proxy.httpx, "AsyncClient", _NoClient)
    ok = asyncio.run(proxy._probe_coworker())
    assert ok is False
    assert called["n"] == 0
