"""
Unit-Tests fuer LocalProxy v3.0 Pass-Through Payload.

Testet:
  1) Model-Flag-Extraktion (--local/--light/--strong/--vision)
  2) Payload-Builder (image_url sanitizer, Moonshot-Patch)
  3) Tool-Call-Normalisierung
  4) Response-Payload (OpenAI-Shape)
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# proxy.py als Modul laden
sys.path.insert(0, str(REPO_ROOT))


def _load_proxy_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("proxy_test", REPO_ROOT / "proxy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["proxy_test"] = module
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


# ═══════════════════════════════════════════════════════════════════════════
# 1. Model-Flag-Extraktion
# ═══════════════════════════════════════════════════════════════════════════

def test_extract_flag_light():
    cleaned, cat, slot = proxy._extract_model_flag("hello --light test")
    assert cat == "light"
    assert slot is None
    assert "--light" not in cleaned


def test_extract_flag_vision():
    cleaned, cat, slot = proxy._extract_model_flag("--vision describe image")
    assert cat == "vision"
    assert slot is None
    assert "--vision" not in cleaned


def test_extract_flag_local():
    cleaned, cat, slot = proxy._extract_model_flag("use --local for this")
    assert cat == "local"
    assert slot is None
    assert "--local" not in cleaned


def test_extract_flag_strong():
    cleaned, cat, slot = proxy._extract_model_flag("--strong architect task")
    assert cat == "strong"
    assert slot is None
    assert "--strong" not in cleaned


def test_extract_flag_no_flag():
    cleaned, cat, slot = proxy._extract_model_flag("hello world no flag")
    assert cat is None
    assert slot is None


def test_extract_flag_last_wins():
    cleaned, cat, slot = proxy._extract_model_flag("--light but really --strong")
    assert cat == "strong"
    assert slot is None
    assert "--light" not in cleaned
    assert "--strong" not in cleaned


def test_extract_flag_with_slot_number():
    cleaned, cat, slot = proxy._extract_model_flag("--light 2 mach dies")
    assert cat == "light"
    assert slot == 2
    assert "--light" not in cleaned
    assert "2" not in cleaned


def test_extract_flag_slot_3():
    cleaned, cat, slot = proxy._extract_model_flag("test --strong 3 letzter slot")
    assert cat == "strong"
    assert slot == 3
    assert "--strong" not in cleaned


def test_extract_flag_slot_last_wins():
    cleaned, cat, slot = proxy._extract_model_flag("--light 1 but --light 3")
    assert cat == "light"
    assert slot == 3
    assert "--light" not in cleaned


def test_extract_flag_invalid_slot_stripped():
    cleaned, cat, slot = proxy._extract_model_flag("--light 5 ungültig")
    assert cat == "light"
    assert slot is None  # 5 nicht gültig (1-3)
    assert "--light" not in cleaned


def test_strip_flags_from_messages():
    msgs = [
        {"role": "user", "content": "hello --light world"},
        {"role": "assistant", "content": "ok"},
    ]
    proxy._strip_model_flags_from_messages(msgs)
    assert "--light" not in msgs[0]["content"]
    assert msgs[0]["content"].strip() == "hello world"


# ═══════════════════════════════════════════════════════════════════════════
# 2. image_url-Sanitizer
# ═══════════════════════════════════════════════════════════════════════════

def test_sanitize_removes_image_url():
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]},
    ]
    removed = proxy._sanitize_image_urls_inplace(msgs)
    assert removed == 1
    content = msgs[0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "hello"


def test_sanitize_keeps_text_only():
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "hello"},
        ]},
    ]
    removed = proxy._sanitize_image_urls_inplace(msgs)
    assert removed == 0
    assert len(msgs[0]["content"]) == 1


def test_sanitize_all_images_fallback():
    msgs = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "http://x/img.png"}},
        ]},
    ]
    removed = proxy._sanitize_image_urls_inplace(msgs)
    assert removed == 1
    content = msgs[0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert "omitted" in content[0]["text"]


# ═══════════════════════════════════════════════════════════════════════════
# 3. Moonshot-Patch
# ═══════════════════════════════════════════════════════════════════════════

def test_moonshot_patch_applies_on_moonshot_url():
    payload = {"temperature": 0.3, "top_p": 0.5, "top_k": 10,
               "presence_penalty": 1.0}
    original = dict(payload)
    proxy._patch_moonshot_payload(payload, "https://api.moonshot-ai.net/v1/chat/completions")
    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 0.95
    assert "top_k" not in payload
    assert payload["presence_penalty"] == 0.0


def test_moonshot_patch_skips_non_moonshot():
    payload = {"temperature": 0.3, "top_p": 0.5, "top_k": 10}
    original = dict(payload)
    proxy._patch_moonshot_payload(payload, "https://api.openai.com/v1/chat/completions")
    assert payload["temperature"] == 0.3  # unchanged
    assert payload["top_k"] == 10  # unchanged


# ═══════════════════════════════════════════════════════════════════════════
# 4. Tool-Call-Normalisierung
# ═══════════════════════════════════════════════════════════════════════════

def test_normalize_tool_calls_openai_format():
    tcs = [
        {"id": "call_123", "type": "function",
         "function": {"name": "read_file", "arguments": '{"filePath":"/x.py"}'}}
    ]
    result = proxy._normalize_tool_calls(tcs)
    assert result is not None
    assert len(result) == 1
    assert result[0]["function"]["name"] == "read_file"
    assert json.loads(result[0]["function"]["arguments"]) == {"filePath": "/x.py"}


def test_normalize_tool_calls_invalid_args():
    tcs = [
        {"function": {"name": "grep_search", "arguments": "not-json"}}
    ]
    result = proxy._normalize_tool_calls(tcs)
    assert result is not None
    assert len(result) == 1
    args = json.loads(result[0]["function"]["arguments"])
    assert "query" in args or "text" in args


def test_normalize_tool_calls_empty_list():
    result = proxy._normalize_tool_calls([])
    assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# 5. Response-Payload (OpenAI-Shape)
# ═══════════════════════════════════════════════════════════════════════════

def test_build_response_payload_text():
    body = {"model": "gpt-4.1-mini"}
    results: List[Dict[str, Any]] = [{"content": "Hello world", "status": "ok"}]
    resp = proxy._build_response_payload(body, "Hello world", results)
    assert resp["object"] == "chat.completion"
    assert resp["choices"][0]["finish_reason"] == "stop"
    assert resp["choices"][0]["message"]["content"] == "Hello world"


def test_build_response_payload_tool_calls():
    body = {"model": "test-model"}
    results: List[Dict[str, Any]] = [{
        "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "read_file",
                          "arguments": '{"filePath":"/test.py"}'}}
        ],
        "status": "ok",
    }]
    resp = proxy._build_response_payload(body, "", results)
    assert resp["choices"][0]["finish_reason"] == "tool_calls"
    assert resp["choices"][0]["message"]["content"] is None
    assert resp["choices"][0]["message"]["tool_calls"] is not None


# ═══════════════════════════════════════════════════════════════════════════
# 6. Hindsight-Kontext (minimal-smoke)
# ═══════════════════════════════════════════════════════════════════════════

def test_hindsight_format_context_empty():
    records: List[Any] = []
    result = proxy._hindsight.format_context(records)
    assert result == ""


# ═══════════════════════════════════════════════════════════════════════════
# 7. Passthrough-Payload (integration smoke)
# ═══════════════════════════════════════════════════════════════════════════

def test_build_passthrough_payload_basic():
    body = {
        "model": "ignored",
        "messages": [{"role": "user", "content": "hello --light"}],
    }
    payload = proxy._build_passthrough_payload(body, "light")
    # _MODEL_CATEGORIES["light"] ist jetzt ein Array; nimm erstes Element
    light_defs = proxy._model_defs("light")
    assert light_defs, "light sollte mindestens eine konfigurierte Definition haben"
    assert payload["model"] == light_defs[0]["model_name"]
    assert payload["stream"] is False
    assert payload["max_tokens"] > 0


def test_build_passthrough_payload_vision_keeps_image():
    vision_defs = proxy._model_defs("vision")
    if not vision_defs or not vision_defs[0].get("is_vision"):
        pytest.skip("vision category not configured with is_vision=True")
    body = {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]}],
    }
    payload = proxy._build_passthrough_payload(body, "vision")
    # Vision-Kategorie: is_vision=True => image_url sollte bleiben
    content = payload["messages"][0]["content"]
    has_image = any(p.get("type") == "image_url" for p in (content if isinstance(content, list) else []))
    assert has_image is True


# ═══════════════════════════════════════════════════════════════════════════
# 8. Fallback-System — Model-Defs, Cooldown, Reset
# ═══════════════════════════════════════════════════════════════════════════


def test_model_defs_legacy_dict():
    """local ist Single-Def Dict -> _model_defs gibt [dict] zurueck."""
    defs = proxy._model_defs("local")
    assert isinstance(defs, list)
    assert len(defs) == 1
    assert isinstance(defs[0], dict)
    assert defs[0].get("api_url")


def test_model_defs_array():
    """light/strong/vision sind Arrays -> gibt konfigurierte Defs zurueck."""
    for cat in ("light", "strong", "vision"):
        defs = proxy._model_defs(cat)
        assert isinstance(defs, list), f"{cat} sollte Liste sein"
        assert len(defs) >= 1, f"{cat} sollte mind. 1 Def haben"
        assert isinstance(defs[0], dict)
        assert defs[0].get("model_name")


def test_model_defs_skips_empty():
    """Leere Fallback-Eintraege (api_url='') werden gefiltert."""
    for cat in ("light", "strong", "vision"):
        defs = proxy._model_defs(cat)
        # Alle Defs muessen api_url + model_name haben
        for d in defs:
            assert d.get("api_url"), f"{cat} def hat leere api_url"
            assert d.get("model_name"), f"{cat} def hat leeren model_name"


def test_do_reset():
    """Reset setzt alle Kategorien auf 0 und loescht Cooldowns."""
    # Aktuelle Indices merken und setzen
    old = dict(proxy._CATEGORY_ACTIVE_IDX)
    proxy._CATEGORY_ACTIVE_IDX["light"] = 1
    proxy._CATEGORY_ACTIVE_IDX["strong"] = 2
    proxy._do_reset()
    assert proxy._CATEGORY_ACTIVE_IDX["light"] == 0
    assert proxy._CATEGORY_ACTIVE_IDX["strong"] == 0
    assert proxy._CATEGORY_ACTIVE_IDX["vision"] == 0
    # local bleibt auch 0
    assert proxy._CATEGORY_ACTIVE_IDX["local"] == 0
    # Wiederherstellen
    proxy._CATEGORY_ACTIVE_IDX.update(old)


def test_detect_reset_flag():
    """--reset wird erkannt, --light nicht als Reset."""
    assert proxy._detect_reset_flag("--reset")
    assert proxy._detect_reset_flag("bitte --reset machen")
    assert proxy._detect_reset_flag("--reset bitte --light")
    assert not proxy._detect_reset_flag("--light")
    assert not proxy._detect_reset_flag("kein flag")


def test_retry_after_seconds():
    """Retry-After Header wird korrekt extrahiert."""
    from types import SimpleNamespace
    headers = SimpleNamespace(get=lambda k: "60" if k.lower() == "retry-after" else None)
    val = proxy._retry_after_seconds(429, headers)
    assert val == 60.0

    headers_no = SimpleNamespace(get=lambda k: None)
    assert proxy._retry_after_seconds(429, headers_no) is None
    assert proxy._retry_after_seconds(200, headers) is None
    assert proxy._retry_after_seconds(503, headers) == 60.0


def test_model_key():
    """_model_key erzeugt stable ID category:model_name."""
    key = proxy._model_key("light", 0)
    assert key.startswith("light:")
    assert len(key) > 6
    assert "gpt" in key or "claude" in key or "Qwen" in key  # je nach Konfig


def test_cooldown_roundtrip():
    """Cooldown setzen und abfragen funktioniert."""
    # Cleanup
    if proxy.COOLDOWN_FILE.exists():
        proxy.COOLDOWN_FILE.unlink()
    assert not proxy._is_in_cooldown("light", 0)
    # Starte mit default Duration (300s) — min 10s greift
    proxy._start_cooldown("light", 0, duration_override=10.0)
    assert proxy._is_in_cooldown("light", 0)
    # Prüfe, ob Cooldown in Datei persistiert wurde
    data = proxy._load_cooldowns()
    key = proxy._model_key("light", 0)
    assert key in data
    assert data[key] > 0
    # Löschen um Test sauber zu beenden
    if proxy.COOLDOWN_FILE.exists():
        proxy.COOLDOWN_FILE.unlink()
    assert not proxy._is_in_cooldown("light", 0)
