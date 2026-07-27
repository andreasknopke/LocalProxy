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
    cleaned, cat, slot = proxy._extract_model_flag("hello world\n--light")
    assert cat == "light"
    assert slot is None
    assert "--light" not in cleaned


def test_extract_flag_vision():
    cleaned, cat, slot = proxy._extract_model_flag("describe image\n--vision 1")
    assert cat == "vision"
    assert slot == 1
    assert "--vision" not in cleaned


def test_extract_flag_local():
    cleaned, cat, slot = proxy._extract_model_flag("nutze lokales modell\n--local")
    assert cat == "local"
    assert slot is None
    assert "--local" not in cleaned


def test_extract_flag_strong():
    cleaned, cat, slot = proxy._extract_model_flag("architect task\n--strong")
    assert cat == "strong"
    assert slot is None
    assert "--strong" not in cleaned


def test_extract_flag_no_flag():
    cleaned, cat, slot = proxy._extract_model_flag("hello world no flag")
    assert cat is None
    assert slot is None


def test_extract_flag_last_wins():
    cleaned, cat, slot = proxy._extract_model_flag("some text\n--light\n--strong")
    assert cat == "strong"
    assert slot is None
    assert "--strong" not in cleaned
    # Nur das letzte Flag (--strong) wird entfernt, --light bleibt als Content


def test_extract_flag_with_slot_number():
    cleaned, cat, slot = proxy._extract_model_flag("mach dies\n--light 2")
    assert cat == "light"
    assert slot == 2
    assert "--light" not in cleaned


def test_extract_flag_slot_3():
    cleaned, cat, slot = proxy._extract_model_flag("test text\n--strong 3")
    assert cat == "strong"
    assert slot == 3
    assert "--strong" not in cleaned


def test_extract_flag_slot_last_wins():
    cleaned, cat, slot = proxy._extract_model_flag("some text\n--light 1\n--light 2")
    assert cat == "light"
    assert slot == 2
    # Nur --light 2 wird als Flag erkannt & entfernt; --light 1 ist Content


def test_extract_flag_invalid_slot_stripped():
    cleaned, cat, slot = proxy._extract_model_flag("text\n--light 5")
    assert cat == "light"
    assert slot is None  # 5 nicht gültig (1-3)
    assert "--light" not in cleaned


def test_extract_flag_xml_wrapped():
    """Flag mit XML-Wrapping (Copilot <userRequest>)"""
    cleaned, cat, slot = proxy._extract_model_flag("<userRequest>\nBaue ein Log-View\n--light 2\n</userRequest>")
    assert cat == "light"
    assert slot == 2
    assert "--light" not in cleaned


def test_extract_flag_proximity_guard():
    """Flag weit vor Text-Ende wird ignoriert (False-Positive-Schutz)"""
    long_suffix = "x" * 400
    cleaned, cat, slot = proxy._extract_model_flag(f"--light\n{long_suffix}")
    assert cat is None  # Zu weit vom Ende entfernt
    assert "--light" in cleaned  # Wurde nicht entfernt (bleibt als Content)


def test_extract_flag_multiline_last_valid():
    """Nur das letzte gueltige Flag nahe am Ende zaehlt"""
    cleaned, cat, slot = proxy._extract_model_flag("code with --light inside\nmore text here\n--strong 2")
    assert cat == "strong"
    assert slot == 2


def test_strip_flags_from_messages():
    msgs = [
        {"role": "user", "content": "hello world\n--light"},
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
    # auto-detect kann max_tokens → max_completion_tokens konvertieren
    token_key = "max_completion_tokens" if "max_completion_tokens" in payload else "max_tokens"
    assert payload[token_key] > 0


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


# ═══════════════════════════════════════════════════════════════════════════
# Read-Loop-Detection
# ═══════════════════════════════════════════════════════════════════════════

def _make_read_msg(file_path: str, start: int, end: int) -> Dict[str, Any]:
    """Erzeugt eine assistant-Message mit read_file tool_call."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_test",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": json.dumps({
                    "filePath": file_path, "startLine": start, "endLine": end
                }),
            },
        }],
    }


def test_extract_read_signature_basic():
    args = json.dumps({"filePath": "/a/b.py", "startLine": 1, "endLine": 10})
    sig = proxy._extract_read_signature(args)
    assert sig == "/a/b.py|1|10"


def test_extract_read_signature_dict():
    sig = proxy._extract_read_signature({"filePath": "/x.py", "startLine": 5, "endLine": 20})
    assert sig == "/x.py|5|20"


def test_extract_read_signature_no_file():
    assert proxy._extract_read_signature({"startLine": 1}) is None
    assert proxy._extract_read_signature("not json") is None
    assert proxy._extract_read_signature(None) is None


def test_extract_read_signature_missing_lines():
    sig = proxy._extract_read_signature({"filePath": "/f.py"})
    assert sig == "/f.py||"


def test_detect_read_loop_below_threshold():
    """3 identische Reads bei Threshold=3 → keine Intervention (erst >3)."""
    msgs = [
        {"role": "user", "content": "read it"},
        _make_read_msg("/a.py", 1, 10),
        {"role": "tool", "content": "file content"},
        _make_read_msg("/a.py", 1, 10),
        {"role": "tool", "content": "file content"},
        _make_read_msg("/a.py", 1, 10),
        {"role": "tool", "content": "file content"},
    ]
    old_threshold = proxy.READ_LOOP_THRESHOLD
    proxy.READ_LOOP_THRESHOLD = 3
    try:
        result = proxy._detect_read_loop_inplace(msgs, category="local")
        assert result is False
        assert len(msgs) == 7  # keine Intervention, 7 Messages unveraendert
    finally:
        proxy.READ_LOOP_THRESHOLD = old_threshold


def test_detect_read_loop_triggers():
    """4 identische Reads bei Threshold=3 → Intervention wird injiziert."""
    msgs = [
        {"role": "user", "content": "read it"},
        _make_read_msg("/a.py", 1, 10),
        {"role": "tool", "content": "c"},
        _make_read_msg("/a.py", 1, 10),
        {"role": "tool", "content": "c"},
        _make_read_msg("/a.py", 1, 10),
        {"role": "tool", "content": "c"},
        _make_read_msg("/a.py", 1, 10),
        {"role": "tool", "content": "c"},
    ]
    old_threshold = proxy.READ_LOOP_THRESHOLD
    proxy.READ_LOOP_THRESHOLD = 3
    try:
        result = proxy._detect_read_loop_inplace(msgs, category="local")
        assert result is True
        assert len(msgs) == 10  # +1 Intervention
        last = msgs[-1]
        assert last["role"] == "user"
        assert "STOP" in last["content"]
        assert "4" in last["content"]
    finally:
        proxy.READ_LOOP_THRESHOLD = old_threshold


def test_detect_read_loop_cloud_model_ignored():
    """Cloud-Modelle (light/strong/vision) → keine Detection."""
    msgs = [
        _make_read_msg("/a.py", 1, 10),
        _make_read_msg("/a.py", 1, 10),
        _make_read_msg("/a.py", 1, 10),
        _make_read_msg("/a.py", 1, 10),
        _make_read_msg("/a.py", 1, 10),
    ]
    old_threshold = proxy.READ_LOOP_THRESHOLD
    proxy.READ_LOOP_THRESHOLD = 3
    try:
        assert proxy._detect_read_loop_inplace(msgs, category="light") is False
        assert proxy._detect_read_loop_inplace(msgs, category="strong") is False
        assert proxy._detect_read_loop_inplace(msgs, category="vision") is False
        assert proxy._detect_read_loop_inplace(msgs, category="") is False
    finally:
        proxy.READ_LOOP_THRESHOLD = old_threshold


def test_detect_read_loop_different_files_no_trigger():
    """Unterschiedliche Dateien → kein Loop."""
    msgs = [
        _make_read_msg("/a.py", 1, 10),
        _make_read_msg("/b.py", 1, 10),
        _make_read_msg("/c.py", 1, 10),
        _make_read_msg("/d.py", 1, 10),
        _make_read_msg("/e.py", 1, 10),
    ]
    old_threshold = proxy.READ_LOOP_THRESHOLD
    proxy.READ_LOOP_THRESHOLD = 3
    try:
        assert proxy._detect_read_loop_inplace(msgs, category="local") is False
    finally:
        proxy.READ_LOOP_THRESHOLD = old_threshold


def test_detect_read_loop_different_lines_no_trigger():
    """Gleiche Datei aber andere Zeilen → kein exact Loop (aber file-crawl pruefen)."""
    msgs = [
        _make_read_msg("/a.py", 1, 10),
        _make_read_msg("/a.py", 11, 20),
        _make_read_msg("/a.py", 21, 30),
        _make_read_msg("/a.py", 31, 40),
    ]
    old_threshold = proxy.READ_LOOP_THRESHOLD
    old_file_threshold = proxy.READ_LOOP_FILE_THRESHOLD
    proxy.READ_LOOP_THRESHOLD = 3
    proxy.READ_LOOP_FILE_THRESHOLD = 8
    try:
        assert proxy._detect_read_loop_inplace(msgs, category="local") is False
    finally:
        proxy.READ_LOOP_THRESHOLD = old_threshold
        proxy.READ_LOOP_FILE_THRESHOLD = old_file_threshold


def test_detect_read_loop_disabled():
    """Threshold=0 → Detection deaktiviert."""
    msgs = [
        _make_read_msg("/a.py", 1, 10),
        _make_read_msg("/a.py", 1, 10),
        _make_read_msg("/a.py", 1, 10),
        _make_read_msg("/a.py", 1, 10),
        _make_read_msg("/a.py", 1, 10),
    ]
    old_threshold = proxy.READ_LOOP_THRESHOLD
    proxy.READ_LOOP_THRESHOLD = 0
    try:
        assert proxy._detect_read_loop_inplace(msgs, category="local") is False
    finally:
        proxy.READ_LOOP_THRESHOLD = old_threshold


def test_detect_read_loop_interrupted_sequence():
    """Loop wird durch anderen Read unterbrochen → Zaehler reset."""
    msgs = [
        _make_read_msg("/a.py", 1, 10),
        _make_read_msg("/a.py", 1, 10),
        _make_read_msg("/b.py", 1, 10),  # Unterbrechung
        _make_read_msg("/a.py", 1, 10),
        _make_read_msg("/a.py", 1, 10),
    ]
    old_threshold = proxy.READ_LOOP_THRESHOLD
    old_file_threshold = proxy.READ_LOOP_FILE_THRESHOLD
    proxy.READ_LOOP_THRESHOLD = 3
    proxy.READ_LOOP_FILE_THRESHOLD = 8
    try:
        # max consecutive = 2, nicht > 3; file-crawl: 4x /a.py in 5, nicht > 8
        assert proxy._detect_read_loop_inplace(msgs, category="local") is False
    finally:
        proxy.READ_LOOP_THRESHOLD = old_threshold
        proxy.READ_LOOP_FILE_THRESHOLD = old_file_threshold


def test_detect_file_crawl_triggers():
    """Gleiche Datei mit verschiedenen Zeilen >N mal im Fenster → file-crawl Intervention."""
    # 9 Reads der gleichen Datei mit verschiedenen Zeilen im Fenster von 12
    msgs = []
    for i in range(9):
        msgs.append(_make_read_msg("/big.tsx", i * 100 + 1, (i + 1) * 100))
        msgs.append({"role": "tool", "content": "content"})
    old_threshold = proxy.READ_LOOP_THRESHOLD
    old_file_threshold = proxy.READ_LOOP_FILE_THRESHOLD
    old_file_window = proxy.READ_LOOP_FILE_WINDOW
    proxy.READ_LOOP_THRESHOLD = 3
    proxy.READ_LOOP_FILE_THRESHOLD = 8
    proxy.READ_LOOP_FILE_WINDOW = 12
    try:
        result = proxy._detect_read_loop_inplace(msgs, category="local")
        assert result is True
        last = msgs[-1]
        assert last["role"] == "user"
        assert "crawling" in last["content"]
        assert "/big.tsx" in last["content"]
    finally:
        proxy.READ_LOOP_THRESHOLD = old_threshold
        proxy.READ_LOOP_FILE_THRESHOLD = old_file_threshold
        proxy.READ_LOOP_FILE_WINDOW = old_file_window


def test_detect_file_crawl_below_threshold():
    """7 Reads der gleichen Datei im Fenster → kein file-crawl (threshold=8)."""
    msgs = []
    for i in range(7):
        msgs.append(_make_read_msg("/big.tsx", i * 100 + 1, (i + 1) * 100))
        msgs.append({"role": "tool", "content": "content"})
    old_threshold = proxy.READ_LOOP_THRESHOLD
    old_file_threshold = proxy.READ_LOOP_FILE_THRESHOLD
    old_file_window = proxy.READ_LOOP_FILE_WINDOW
    proxy.READ_LOOP_THRESHOLD = 3
    proxy.READ_LOOP_FILE_THRESHOLD = 8
    proxy.READ_LOOP_FILE_WINDOW = 12
    try:
        assert proxy._detect_read_loop_inplace(msgs, category="local") is False
    finally:
        proxy.READ_LOOP_THRESHOLD = old_threshold
        proxy.READ_LOOP_FILE_THRESHOLD = old_file_threshold
        proxy.READ_LOOP_FILE_WINDOW = old_file_window
