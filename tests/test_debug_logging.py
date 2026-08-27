"""Tests für den Debug-Logging-Master-Schalter (DEBUG_LOGGING / tokens.debug_logging).

Der Schalter steuert ALLE Debug-/Trace-Ausgaben:
  - proxy.log (Datei + stdout, via _log)
  - Payload-Dumps in data/debug/ (_dump_debug_payload)
  - I/O-Traces in data/io_traces/ (io_trace_active)
  - Debug-Ring (_register_debug_request)

AUS → es wird NICHTS mehr geschrieben; die Log-Anzeige im WebUI zeigt dann nur
noch die letzten Einträge vor dem Abschalten (Anzeige selbst bleibt unverändert).

Konvention wie in den anderen Test-Files: plain def test_*(monkeypatch).
"""
import builtins
import json

import pytest

import proxy


# ─── _log ───────────────────────────────────────────────────────────────────

def test_log_writes_when_debug_logging_enabled(monkeypatch):
    """DEBUG_LOGGING=True (Default): _log schreibt in stdout + Log-Handler."""
    monkeypatch.setattr(proxy, "DEBUG_LOGGING", True)
    printed: list = []
    handled: list = []
    monkeypatch.setattr(builtins, "print", lambda *a, **k: printed.append(a))
    monkeypatch.setattr(proxy._LOG_HANDLER, "handle",
                        lambda *a, **k: handled.append(a))
    proxy._log("test-zeile")
    assert printed, "_log muss bei aktivem Debug-Logging printen"
    assert handled, "_log muss den Datei-Handler aufrufen"


def test_log_silent_when_debug_logging_disabled(monkeypatch):
    """DEBUG_LOGGING=False: _log schreibt WEDER stdout NOCH Datei-Handler."""
    monkeypatch.setattr(proxy, "DEBUG_LOGGING", False)
    printed: list = []
    handled: list = []
    monkeypatch.setattr(builtins, "print", lambda *a, **k: printed.append(a))
    monkeypatch.setattr(proxy._LOG_HANDLER, "handle",
                        lambda *a, **k: handled.append(a))
    proxy._log("darf-nicht-landen")
    assert not printed
    assert not handled


# ─── _dump_debug_payload ───────────────────────────────────────────────────

def test_dump_debug_payload_writes_when_enabled(monkeypatch, tmp_path):
    """DEBUG_LOGGING=True + DEBUG_ENABLED=True: Payload-Dump wird geschrieben."""
    monkeypatch.setattr(proxy, "DEBUG_LOGGING", True)
    monkeypatch.setattr(proxy, "DEBUG_ENABLED", True)
    monkeypatch.setattr(proxy, "DEBUG_DIR", tmp_path / "debug")
    proxy._dump_debug_payload("req1", "model_local_0",
                              {"messages": [{"role": "user", "content": "hi"}]})
    files = list((tmp_path / "debug").glob("*.json"))
    assert len(files) == 1


def test_dump_debug_payload_skipped_when_debug_logging_disabled(monkeypatch, tmp_path):
    """DEBUG_LOGGING=False: Kein Payload-Dump, auch wenn DEBUG_ENABLED=True."""
    monkeypatch.setattr(proxy, "DEBUG_LOGGING", False)
    monkeypatch.setattr(proxy, "DEBUG_ENABLED", True)
    monkeypatch.setattr(proxy, "DEBUG_DIR", tmp_path / "debug")
    proxy._dump_debug_payload("req1", "model_local_0",
                              {"messages": [{"role": "user", "content": "hi"}]})
    assert not (tmp_path / "debug").exists() or not list((tmp_path / "debug").iterdir())


# ─── _register_debug_request (Ring) ────────────────────────────────────────

def test_debug_ring_not_filled_when_disabled(monkeypatch):
    """DEBUG_LOGGING=False: Der Debug-Ring bleibt leer."""
    monkeypatch.setattr(proxy, "DEBUG_LOGGING", False)
    monkeypatch.setattr(proxy, "_DEBUG_RING", [])
    proxy._register_debug_request("req1", {"type": "model_call_start",
                                           "category": "local"})
    assert proxy._DEBUG_RING == []


def test_debug_ring_filled_when_enabled(monkeypatch):
    """DEBUG_LOGGING=True: Eintrag landet im Ring."""
    monkeypatch.setattr(proxy, "DEBUG_LOGGING", True)
    monkeypatch.setattr(proxy, "_DEBUG_RING", [])
    proxy._register_debug_request("req1", {"type": "model_call_start"})
    assert len(proxy._DEBUG_RING) == 1
    assert proxy._DEBUG_RING[0]["req_id"] == "req1"


# ─── io_trace_active ───────────────────────────────────────────────────────

def test_io_trace_active_disabled_when_debug_logging_off(monkeypatch):
    """DEBUG_LOGGING=False gewinnt gegen IO_TRACE_ENABLED=True."""
    monkeypatch.setattr(proxy, "DEBUG_LOGGING", False)
    monkeypatch.setattr(proxy, "IO_TRACE_ENABLED", True)
    monkeypatch.setattr(proxy, "IO_TRACE_SECONDS", 0)
    assert not proxy.io_trace_active()


def test_io_trace_active_enabled(monkeypatch):
    """DEBUG_LOGGING=True + IO_TRACE_ENABLED=True → Tracing aktiv."""
    monkeypatch.setattr(proxy, "DEBUG_LOGGING", True)
    monkeypatch.setattr(proxy, "IO_TRACE_ENABLED", True)
    monkeypatch.setattr(proxy, "IO_TRACE_SECONDS", 0)
    assert proxy.io_trace_active()


# ─── _apply_config_file ────────────────────────────────────────────────────

def test_apply_config_file_reads_debug_logging(monkeypatch):
    """tokens.debug_logging=false aus config.json wird uebernommen."""
    monkeypatch.setattr(proxy, "_WEBUI_AVAILABLE", True)
    monkeypatch.setattr(proxy, "_webui_load_config",
                        lambda: {"tokens": {"debug_logging": False}},
                        raising=False)
    monkeypatch.setattr(proxy, "DEBUG_LOGGING", True)
    proxy._apply_config_file()
    assert proxy.DEBUG_LOGGING is False


def test_apply_config_file_keeps_debug_logging_default(monkeypatch):
    """Fehlt der Key, bleibt der aktuelle Wert (Default True) erhalten."""
    monkeypatch.setattr(proxy, "_WEBUI_AVAILABLE", True)
    monkeypatch.setattr(proxy, "_webui_load_config", lambda: {"tokens": {}},
                        raising=False)
    monkeypatch.setattr(proxy, "DEBUG_LOGGING", True)
    proxy._apply_config_file()
    assert proxy.DEBUG_LOGGING is True


# ─── WebUI-Defaults ────────────────────────────────────────────────────────

def test_webui_default_config_and_env_map():
    """WebUI kennt den Schalter: Default an + Env-Mapping vorhanden."""
    import webui
    assert webui.DEFAULT_CONFIG["tokens"]["debug_logging"] is True
    assert webui._ENV_TO_CONFIG["DEBUG_LOGGING"] == ("tokens", "debug_logging")
