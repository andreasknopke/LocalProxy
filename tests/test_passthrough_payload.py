"""
Unit-Tests fuer LocalProxy v3.0 Pass-Through Payload.

Testet:
  1) Model-Flag-Extraktion (--local/--light/--strong/--vision)
  2) Payload-Builder (image_url sanitizer)
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
# 3. Moonshot-Patch entfernt (Kimi-API akzeptiert jetzt Copilot-Temp-Settings)
# ═══════════════════════════════════════════════════════════════════════════

def test_moonshot_patch_function_removed():
    """Die Funktion _patch_moonshot_payload wurde entfernt (2026-08-23)."""
    assert not hasattr(proxy, "_patch_moonshot_payload")


# ═══════════════════════════════════════════════════════════════════════════
# 3b. Reasoning-Injektion entfernt (wurde verworfen — funktioniert nicht)
# ═══════════════════════════════════════════════════════════════════════════

def test_reasoning_injection_function_removed():
    """Die Funktion _patch_reasoning_injection_payload wurde entfernt."""
    assert not hasattr(proxy, "_patch_reasoning_injection_payload")
    assert not hasattr(proxy, "_REASONING_INJECTION_TEXT")


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

def _make_read_msg(file_path: str, start: int, end: int,
                   call_id: str = "call_test") -> Dict[str, Any]:
    """Erzeugt eine assistant-Message mit read_file tool_call."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
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
        result = proxy._detect_read_loop_inplace(msgs, category="local", model_name="poolside/Laguna-S-2.1-NVFP4")
        assert result is False
        assert len(msgs) == 7  # keine Intervention, 7 Messages unveraendert
    finally:
        proxy.READ_LOOP_THRESHOLD = old_threshold


def test_detect_read_loop_triggers():
    """4 identische Reads bei Threshold=3 → Truncation: Loop-Historie wird entfernt."""
    msgs = [
        {"role": "user", "content": "read it"},
        # --- 1. read (below threshold, bleibt erhalten) ---
        _make_read_msg("/a.py", 1, 10),
        {"role": "tool", "content": "c"},
        # --- ab hier: Loop (wiederholte Reads, werden trunkiert) ---
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
        result = proxy._detect_read_loop_inplace(msgs, category="local", model_name="poolside/Laguna-S-2.1-NVFP4")
        assert result is True
        # User-Msg + 1.Read + 1.ToolResult + Intervention = 4
        assert len(msgs) == 4
        last = msgs[-1]
        assert last["role"] == "user"
        assert "STOP" in last["content"]
        assert "4" in last["content"]
        # Der erste Read ist noch da (nicht Teil der Konsekutiv-Sequenz ab idx 2)
        assert msgs[1] is not None
    finally:
        proxy.READ_LOOP_THRESHOLD = old_threshold


def test_detect_read_loop_cloud_model_ignored():
    """Nicht-Laguna-Modelle → keine Detection (egal welche Kategorie)."""
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
        assert proxy._detect_read_loop_inplace(msgs, category="light", model_name="gpt-4.1-mini") is False
        assert proxy._detect_read_loop_inplace(msgs, category="strong", model_name="claude-sonnet-4-20250514") is False
        assert proxy._detect_read_loop_inplace(msgs, category="vision", model_name="gpt-4o") is False
        assert proxy._detect_read_loop_inplace(msgs, category="local", model_name="") is False
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
        assert proxy._detect_read_loop_inplace(msgs, category="local", model_name="poolside/Laguna-S-2.1-NVFP4") is False
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
        assert proxy._detect_read_loop_inplace(msgs, category="local", model_name="poolside/Laguna-S-2.1-NVFP4") is False
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
        assert proxy._detect_read_loop_inplace(msgs, category="local", model_name="poolside/Laguna-S-2.1-NVFP4") is False
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
        assert proxy._detect_read_loop_inplace(msgs, category="local", model_name="poolside/Laguna-S-2.1-NVFP4") is False
    finally:
        proxy.READ_LOOP_THRESHOLD = old_threshold
        proxy.READ_LOOP_FILE_THRESHOLD = old_file_threshold


def test_detect_file_crawl_triggers():
    """Gleiche Datei mit verschiedenen Zeilen >N mal im Fenster → file-crawl Truncation."""
    # 9 Reads der gleichen Datei mit verschiedenen Zeilen im Fenster von 12
    msgs = []
    for i in range(9):
        msgs.append(_make_read_msg("/big.tsx", i * 100 + 1, (i + 1) * 100))
        msgs.append({"role": "tool", "content": "content"})
    old_threshold = proxy.READ_LOOP_THRESHOLD
    old_file_threshold = proxy.READ_LOOP_FILE_THRESHOLD
    old_file_window = proxy.READ_LOOP_FILE_WINDOW
    old_keep = proxy.READ_LOOP_FILE_KEEP
    proxy.READ_LOOP_THRESHOLD = 3
    proxy.READ_LOOP_FILE_THRESHOLD = 8
    proxy.READ_LOOP_FILE_WINDOW = 12
    proxy.READ_LOOP_FILE_KEEP = 1
    try:
        result = proxy._detect_read_loop_inplace(msgs, category="local", model_name="poolside/Laguna-S-2.1-NVFP4")
        assert result is True
        last = msgs[-1]
        assert last["role"] == "user"
        assert "crawling" in last["content"]
        assert "/big.tsx" in last["content"]
        # Die ueberzaehligen Reads wurden trunkiert (keep=1 → 1 read + tool + intervention)
        assert len(msgs) < 18
        # Tail muss weg sein: Intervention ist letzte Message
        assert msgs[-1]["role"] == "user"
    finally:
        proxy.READ_LOOP_THRESHOLD = old_threshold
        proxy.READ_LOOP_FILE_THRESHOLD = old_file_threshold
        proxy.READ_LOOP_FILE_WINDOW = old_file_window
        proxy.READ_LOOP_FILE_KEEP = old_keep


def test_file_crawl_truncation_removes_tail():
    """File-crawl Truncation entfernt den gesamten Tail inkl. spaeterer Search-Loops."""
    msgs = [{"role": "user", "content": "analyze schedule"}]
    for i in range(9):
        msgs.append(_make_read_msg("/ScheduleBoard.tsx", i * 200 + 1, (i + 1) * 200))
        msgs.append({"role": "tool", "tool_call_id": f"r{i}", "content": "chunk"})
    # Nach dem Crawl: Search-Spam (waere sonst im Tail geblieben)
    for i in range(5):
        msgs.append(_make_search_msg("currentWeekShifts", call_id=f"s{i}"))
        msgs.append(_make_search_result(f"s{i}", "Found 64 matches."))

    old_threshold = proxy.READ_LOOP_THRESHOLD
    old_file_threshold = proxy.READ_LOOP_FILE_THRESHOLD
    old_file_window = proxy.READ_LOOP_FILE_WINDOW
    old_keep = proxy.READ_LOOP_FILE_KEEP
    proxy.READ_LOOP_THRESHOLD = 3
    proxy.READ_LOOP_FILE_THRESHOLD = 8
    proxy.READ_LOOP_FILE_WINDOW = 12
    proxy.READ_LOOP_FILE_KEEP = 1
    try:
        before = len(msgs)
        assert proxy._detect_read_loop_inplace(msgs, category="local", model_name="poolside/Laguna-S-2.1-NVFP4") is True
        assert len(msgs) < before
        # Kein Search-Tail mehr
        assert not any(
            m.get("role") == "assistant" and "currentWeekShifts" in json.dumps(m)
            for m in msgs
        )
        assert msgs[-1]["role"] == "user"
        assert "crawling" in msgs[-1]["content"]
    finally:
        proxy.READ_LOOP_THRESHOLD = old_threshold
        proxy.READ_LOOP_FILE_THRESHOLD = old_file_threshold
        proxy.READ_LOOP_FILE_WINDOW = old_file_window
        proxy.READ_LOOP_FILE_KEEP = old_keep


def test_detect_response_loop_flags_repeated_search():
    """Detect-only: wiederholte Search wird als Loop erkannt (nicht blockiert)."""
    msgs = []
    for i in range(4):
        cid = f"call_{i}"
        msgs.append(_make_search_msg("currentWeekShifts", call_id=cid))
        msgs.append(_make_search_result(cid, "Found 64 matches."))
    body = {"messages": msgs}
    new_tc = [{
        "id": "call_new",
        "type": "function",
        "function": {
            "name": "grep_search",
            "arguments": json.dumps({
                "query": "currentWeekShifts",
                "isRegexp": False,
            }),
        },
    }]
    old_thr = proxy._RESPONSE_LOOP_THRESHOLD
    proxy._RESPONSE_LOOP_THRESHOLD = 3
    try:
        reasons, blocked_names = proxy._detect_response_loop(
            body, new_tc, category="local", model_name="poolside/Laguna-S-2.1-NVFP4")
        assert reasons
        assert "grep_search" in blocked_names
    finally:
        proxy._RESPONSE_LOOP_THRESHOLD = old_thr


def test_detect_response_loop_allows_different_search():
    """Andere Search-Query wird nicht als Loop erkannt."""
    msgs = []
    for i in range(4):
        cid = f"call_{i}"
        msgs.append(_make_search_msg("currentWeekShifts", call_id=cid))
        msgs.append(_make_search_result(cid, "Found 64 matches."))
    body = {"messages": msgs}
    new_tc = [{
        "id": "call_new",
        "type": "function",
        "function": {
            "name": "grep_search",
            "arguments": json.dumps({
                "query": "totallyDifferentSymbol",
                "isRegexp": False,
            }),
        },
    }]
    old_thr = proxy._RESPONSE_LOOP_THRESHOLD
    proxy._RESPONSE_LOOP_THRESHOLD = 3
    try:
        reasons, blocked_names = proxy._detect_response_loop(
            body, new_tc, category="local", model_name="poolside/Laguna-S-2.1-NVFP4")
        assert reasons == []
        assert blocked_names == []
    finally:
        proxy._RESPONSE_LOOP_THRESHOLD = old_thr


def test_detect_response_loop_flags_file_crawl_continue():
    """Detect-only: weiterer read_file derselben gecrawlten Datei wird erkannt."""
    msgs = []
    for i in range(9):
        msgs.append(_make_read_msg("/big.tsx", i * 100 + 1, (i + 1) * 100, call_id=f"r{i}"))
        msgs.append({"role": "tool", "tool_call_id": f"r{i}", "content": "c"})
    body = {"messages": msgs}
    new_tc = [{
        "id": "call_new",
        "type": "function",
        "function": {
            "name": "read_file",
            "arguments": json.dumps({
                "filePath": "/big.tsx",
                "startLine": 900,
                "endLine": 1000,
            }),
        },
    }]
    old_thr = proxy._RESPONSE_LOOP_THRESHOLD
    old_ft = proxy.READ_LOOP_FILE_THRESHOLD
    old_fw = proxy.READ_LOOP_FILE_WINDOW
    proxy._RESPONSE_LOOP_THRESHOLD = 3
    proxy.READ_LOOP_FILE_THRESHOLD = 8
    proxy.READ_LOOP_FILE_WINDOW = 12
    try:
        reasons, blocked_names = proxy._detect_response_loop(
            body, new_tc, category="local", model_name="poolside/Laguna-S-2.1-NVFP4")
        assert reasons
        assert "read_file" in blocked_names
    finally:
        proxy._RESPONSE_LOOP_THRESHOLD = old_thr
        proxy.READ_LOOP_FILE_THRESHOLD = old_ft
        proxy.READ_LOOP_FILE_WINDOW = old_fw


# ═══════════════════════════════════════════════════════════════════════════
# Search-Loop-Detection Tests
# ═══════════════════════════════════════════════════════════════════════════

def _make_search_msg(query: str, include_pattern: str = "", call_id: str = "call_s1") -> Dict[str, Any]:
    """Erzeugt eine assistant-Message mit grep_search tool_call."""
    args = {"query": query, "isRegexp": False}
    if include_pattern:
        args["includePattern"] = include_pattern
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": "grep_search",
                "arguments": json.dumps(args),
            },
        }],
    }


def _make_search_result(call_id: str, text: str) -> Dict[str, Any]:
    """Erzeugt eine tool-Result-Message."""
    return {"role": "tool", "tool_call_id": call_id, "content": text}


_NO_MATCH_TEXT = (
    "No matches found. Your search pattern might be excluded completely by either "
    "the search.exclude settings or .*ignore files."
)


def test_extract_search_signature_basic():
    args = json.dumps({"query": "foo_bar", "isRegexp": False, "includePattern": "src/**"})
    sig = proxy._extract_search_signature(args)
    assert sig == "foo_bar|src/**"


def test_extract_search_signature_dict():
    sig = proxy._extract_search_signature({"query": "hello", "isRegexp": True})
    assert sig == "hello|"


def test_extract_search_signature_no_query():
    """Ohne query-Feld: faellt auf universelle Signatur zurueck (nicht None)."""
    sig = proxy._extract_search_signature({"isRegexp": False})
    assert sig is not None  # universelle Signatur: "isRegexp=False"
    assert proxy._extract_search_signature("not json") is None
    assert proxy._extract_search_signature(None) is None


def test_detect_search_loop_triggers():
    """4 identische Suchen mit No-Match bei Threshold=3 → Truncation ab 2. Wiederholung."""
    msgs = []
    for i in range(4):
        cid = f"call_{i}"
        msgs.append(_make_search_msg("nonexistent_thing", call_id=cid))
        msgs.append(_make_search_result(cid, _NO_MATCH_TEXT))

    old_threshold = proxy.SEARCH_LOOP_THRESHOLD
    proxy.SEARCH_LOOP_THRESHOLD = 3
    try:
        result = proxy._detect_search_loop_inplace(msgs, category="local", model_name="poolside/Laguna-S-2.1-NVFP4")
        assert result is True
        # 1.Search + 1.NoMatchResult + Intervention = 3
        assert len(msgs) == 3
        last = msgs[-1]
        assert last["role"] == "user"
        assert "STOP" in last["content"]
        assert "nonexistent_thing" in last["content"]
        assert "4" in last["content"]
    finally:
        proxy.SEARCH_LOOP_THRESHOLD = old_threshold


def test_detect_search_loop_below_threshold():
    """3 identische Suchen bei Threshold=3 → keine Intervention (erst >3)."""
    msgs = []
    for i in range(3):
        cid = f"call_{i}"
        msgs.append(_make_search_msg("foo", call_id=cid))
        msgs.append(_make_search_result(cid, _NO_MATCH_TEXT))

    old_threshold = proxy.SEARCH_LOOP_THRESHOLD
    proxy.SEARCH_LOOP_THRESHOLD = 3
    try:
        result = proxy._detect_search_loop_inplace(msgs, category="local", model_name="poolside/Laguna-S-2.1-NVFP4")
        assert result is False
        assert len(msgs) == 6  # unveraendert
    finally:
        proxy.SEARCH_LOOP_THRESHOLD = old_threshold


def test_detect_search_loop_cloud_model_ignored():
    """Nicht-Laguna-Modelle → keine Search-Loop-Detection."""
    msgs = []
    for i in range(5):
        cid = f"call_{i}"
        msgs.append(_make_search_msg("foo", call_id=cid))
        msgs.append(_make_search_result(cid, _NO_MATCH_TEXT))

    old_threshold = proxy.SEARCH_LOOP_THRESHOLD
    proxy.SEARCH_LOOP_THRESHOLD = 3
    try:
        assert proxy._detect_search_loop_inplace(msgs, category="light", model_name="gpt-4.1-mini") is False
        assert proxy._detect_search_loop_inplace(msgs, category="strong", model_name="claude-sonnet-4-20250514") is False
        assert proxy._detect_search_loop_inplace(msgs, category="local", model_name="") is False
    finally:
        proxy.SEARCH_LOOP_THRESHOLD = old_threshold


def test_detect_search_loop_with_results_no_trigger_when_repeat_disabled():
    """Suchen mit Treffern → kein No-Match-Loop; Repeat-Detection aus."""
    msgs = []
    for i in range(5):
        cid = f"call_{i}"
        msgs.append(_make_search_msg("foo", call_id=cid))
        msgs.append(_make_search_result(cid, "Found 3 matches in 2 files."))

    old_threshold = proxy.SEARCH_LOOP_THRESHOLD
    old_repeat = proxy.SEARCH_REPEAT_THRESHOLD
    proxy.SEARCH_LOOP_THRESHOLD = 3
    proxy.SEARCH_REPEAT_THRESHOLD = 0
    try:
        assert proxy._detect_search_loop_inplace(msgs, category="local", model_name="poolside/Laguna-S-2.1-NVFP4") is False
    finally:
        proxy.SEARCH_LOOP_THRESHOLD = old_threshold
        proxy.SEARCH_REPEAT_THRESHOLD = old_repeat


def test_detect_search_repeat_loop_with_results_triggers():
    """Gleiche Suche >N mal MIT Treffern → Repeat-Loop Truncation."""
    msgs = []
    for i in range(4):
        cid = f"call_{i}"
        msgs.append(_make_search_msg("currentWeekShifts", call_id=cid))
        msgs.append(_make_search_result(cid, "Found 64 matches in 12 files."))

    old_threshold = proxy.SEARCH_LOOP_THRESHOLD
    old_repeat = proxy.SEARCH_REPEAT_THRESHOLD
    proxy.SEARCH_LOOP_THRESHOLD = 3
    proxy.SEARCH_REPEAT_THRESHOLD = 3
    try:
        result = proxy._detect_search_loop_inplace(msgs, category="local", model_name="poolside/Laguna-S-2.1-NVFP4")
        assert result is True
        last = msgs[-1]
        assert last["role"] == "user"
        assert "STOP" in last["content"]
        assert "currentWeekShifts" in last["content"]
    finally:
        proxy.SEARCH_LOOP_THRESHOLD = old_threshold
        proxy.SEARCH_REPEAT_THRESHOLD = old_repeat


def test_detect_search_loop_interrupted_by_success():
    """No-Match-Loop wird durch erfolgreiche Suche unterbrochen → Zaehler reset.
    Repeat-Detection hier aus, damit nur No-Match-Logik getestet wird.
    """
    msgs = []
    # 2x no match
    msgs.append(_make_search_msg("foo", call_id="c0"))
    msgs.append(_make_search_result("c0", _NO_MATCH_TEXT))
    msgs.append(_make_search_msg("foo", call_id="c1"))
    msgs.append(_make_search_result("c1", _NO_MATCH_TEXT))
    # 1x success (unterbricht)
    msgs.append(_make_search_msg("foo", call_id="c2"))
    msgs.append(_make_search_result("c2", "Found 1 match."))
    # 2x no match
    msgs.append(_make_search_msg("foo", call_id="c3"))
    msgs.append(_make_search_result("c3", _NO_MATCH_TEXT))
    msgs.append(_make_search_msg("foo", call_id="c4"))
    msgs.append(_make_search_result("c4", _NO_MATCH_TEXT))

    old_threshold = proxy.SEARCH_LOOP_THRESHOLD
    old_repeat = proxy.SEARCH_REPEAT_THRESHOLD
    proxy.SEARCH_LOOP_THRESHOLD = 3
    proxy.SEARCH_REPEAT_THRESHOLD = 0
    try:
        # max consecutive no-match = 2, nicht > 3
        assert proxy._detect_search_loop_inplace(msgs, category="local", model_name="poolside/Laguna-S-2.1-NVFP4") is False
    finally:
        proxy.SEARCH_LOOP_THRESHOLD = old_threshold
        proxy.SEARCH_REPEAT_THRESHOLD = old_repeat


def test_detect_search_loop_different_queries_no_trigger():
    """Unterschiedliche Suchanfragen → kein Loop."""
    msgs = []
    for i, q in enumerate(["alpha", "beta", "gamma", "delta"]):
        cid = f"call_{i}"
        msgs.append(_make_search_msg(q, call_id=cid))
        msgs.append(_make_search_result(cid, _NO_MATCH_TEXT))

    old_threshold = proxy.SEARCH_LOOP_THRESHOLD
    old_repeat = proxy.SEARCH_REPEAT_THRESHOLD
    proxy.SEARCH_LOOP_THRESHOLD = 3
    proxy.SEARCH_REPEAT_THRESHOLD = 3
    try:
        assert proxy._detect_search_loop_inplace(msgs, category="local", model_name="poolside/Laguna-S-2.1-NVFP4") is False
    finally:
        proxy.SEARCH_LOOP_THRESHOLD = old_threshold
        proxy.SEARCH_REPEAT_THRESHOLD = old_repeat


def test_detect_search_loop_disabled():
    """Beide Thresholds=0 → Detection deaktiviert."""
    msgs = []
    for i in range(5):
        cid = f"call_{i}"
        msgs.append(_make_search_msg("foo", call_id=cid))
        msgs.append(_make_search_result(cid, _NO_MATCH_TEXT))

    old_threshold = proxy.SEARCH_LOOP_THRESHOLD
    old_repeat = proxy.SEARCH_REPEAT_THRESHOLD
    proxy.SEARCH_LOOP_THRESHOLD = 0
    proxy.SEARCH_REPEAT_THRESHOLD = 0
    try:
        assert proxy._detect_search_loop_inplace(msgs, category="local", model_name="poolside/Laguna-S-2.1-NVFP4") is False
    finally:
        proxy.SEARCH_LOOP_THRESHOLD = old_threshold
        proxy.SEARCH_REPEAT_THRESHOLD = old_repeat


def test_detect_search_loop_file_search_tool():
    """file_search Tool-Name wird ebenfalls erkannt."""
    msgs = []
    for i in range(4):
        cid = f"call_{i}"
        msgs.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": cid,
                "type": "function",
                "function": {
                    "name": "file_search",
                    "arguments": json.dumps({"query": "**/missing.ts"}),
                },
            }],
        })
        msgs.append(_make_search_result(cid, "No files found matching pattern."))

    old_threshold = proxy.SEARCH_LOOP_THRESHOLD
    proxy.SEARCH_LOOP_THRESHOLD = 3
    try:
        result = proxy._detect_search_loop_inplace(msgs, category="local", model_name="poolside/Laguna-S-2.1-NVFP4")
        assert result is True
        last = msgs[-1]
        assert "STOP" in last["content"]
    finally:
        proxy.SEARCH_LOOP_THRESHOLD = old_threshold


# ═══════════════════════════════════════════════════════════════════════════
# Generic-Tool-Loop-Detection Tests (z.B. manage_todo_list)
# ═══════════════════════════════════════════════════════════════════════════

def _make_generic_tool_msg(tool_name: str, args: Dict[str, Any],
                            call_id: str = "call_g1") -> Dict[str, Any]:
    """Erzeugt eine assistant-Message mit beliebigem tool_call."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(args),
            },
        }],
    }


_TODO_ARGS = {
    "todoList": [
        {"id": 1, "title": "Task 1", "status": "completed"},
        {"id": 2, "title": "Task 2", "status": "in-progress"},
    ],
}


def test_generic_tool_loop_triggers():
    """4x identischer manage_todo_list-Call bei Threshold=3 → Truncation."""
    msgs = []
    for i in range(4):
        cid = f"call_t{i}"
        msgs.append(_make_generic_tool_msg("manage_todo_list", _TODO_ARGS, call_id=cid))
        msgs.append({"role": "tool", "tool_call_id": cid, "content": "Aufgabenliste aktualisiert"})

    old_threshold = proxy.GENERIC_TOOL_LOOP_THRESHOLD
    proxy.GENERIC_TOOL_LOOP_THRESHOLD = 3
    try:
        result = proxy._detect_generic_tool_loop_inplace(msgs, category="local", model_name="poolside/Laguna-S-2.1-NVFP4")
        assert result is True
        last = msgs[-1]
        assert last["role"] == "user"
        assert "STOP" in last["content"]
        assert "manage_todo_list" in last["content"]
    finally:
        proxy.GENERIC_TOOL_LOOP_THRESHOLD = old_threshold


def test_generic_tool_loop_below_threshold():
    """3x identischer Call bei Threshold=3 → keine Intervention."""
    msgs = []
    for i in range(3):
        cid = f"call_t{i}"
        msgs.append(_make_generic_tool_msg("manage_todo_list", _TODO_ARGS, call_id=cid))
        msgs.append({"role": "tool", "tool_call_id": cid, "content": "ok"})

    old_threshold = proxy.GENERIC_TOOL_LOOP_THRESHOLD
    proxy.GENERIC_TOOL_LOOP_THRESHOLD = 3
    try:
        result = proxy._detect_generic_tool_loop_inplace(msgs, category="local", model_name="poolside/Laguna-S-2.1-NVFP4")
        assert result is False
    finally:
        proxy.GENERIC_TOOL_LOOP_THRESHOLD = old_threshold


def test_generic_tool_loop_different_args_no_trigger():
    """Gleiche Tool-Name, aber verschiedene Argumente → kein Loop."""
    msgs = []
    for i in range(5):
        cid = f"call_t{i}"
        args = {"todoList": [{"id": i, "title": f"Task {i}", "status": "in-progress"}]}
        msgs.append(_make_generic_tool_msg("manage_todo_list", args, call_id=cid))
        msgs.append({"role": "tool", "tool_call_id": cid, "content": "ok"})

    old_threshold = proxy.GENERIC_TOOL_LOOP_THRESHOLD
    proxy.GENERIC_TOOL_LOOP_THRESHOLD = 3
    try:
        result = proxy._detect_generic_tool_loop_inplace(msgs, category="local", model_name="poolside/Laguna-S-2.1-NVFP4")
        assert result is False
    finally:
        proxy.GENERIC_TOOL_LOOP_THRESHOLD = old_threshold


def test_generic_tool_loop_disabled():
    """Threshold=0 → deaktiviert."""
    msgs = []
    for i in range(6):
        cid = f"call_t{i}"
        msgs.append(_make_generic_tool_msg("manage_todo_list", _TODO_ARGS, call_id=cid))
        msgs.append({"role": "tool", "tool_call_id": cid, "content": "ok"})

    old_threshold = proxy.GENERIC_TOOL_LOOP_THRESHOLD
    proxy.GENERIC_TOOL_LOOP_THRESHOLD = 0
    try:
        result = proxy._detect_generic_tool_loop_inplace(msgs, category="local", model_name="poolside/Laguna-S-2.1-NVFP4")
        assert result is False
    finally:
        proxy.GENERIC_TOOL_LOOP_THRESHOLD = old_threshold


def test_generic_tool_loop_cloud_model_ignored():
    """Nicht-Laguna-Modelle → keine Detection."""
    msgs = []
    for i in range(5):
        cid = f"call_t{i}"
        msgs.append(_make_generic_tool_msg("manage_todo_list", _TODO_ARGS, call_id=cid))
        msgs.append({"role": "tool", "tool_call_id": cid, "content": "ok"})

    old_threshold = proxy.GENERIC_TOOL_LOOP_THRESHOLD
    proxy.GENERIC_TOOL_LOOP_THRESHOLD = 3
    try:
        assert proxy._detect_generic_tool_loop_inplace(msgs, category="light", model_name="gpt-4.1-mini") is False
        assert proxy._detect_generic_tool_loop_inplace(msgs, category="strong", model_name="claude-sonnet-4-20250514") is False
    finally:
        proxy.GENERIC_TOOL_LOOP_THRESHOLD = old_threshold


def test_generic_tool_loop_interrupted_by_different_tool():
    """Unterbrochene Sequenz (anderes Tool dazwischen) → kein Loop."""
    msgs = []
    for i in range(2):
        cid = f"call_t{i}"
        msgs.append(_make_generic_tool_msg("manage_todo_list", _TODO_ARGS, call_id=cid))
        msgs.append({"role": "tool", "tool_call_id": cid, "content": "ok"})
    # Anderes Tool dazwischen
    msgs.append(_make_generic_tool_msg("create_file", {"path": "/x.py", "content": "x"}, call_id="call_x"))
    msgs.append({"role": "tool", "tool_call_id": "call_x", "content": "created"})
    for i in range(2, 4):
        cid = f"call_t{i}"
        msgs.append(_make_generic_tool_msg("manage_todo_list", _TODO_ARGS, call_id=cid))
        msgs.append({"role": "tool", "tool_call_id": cid, "content": "ok"})

    old_threshold = proxy.GENERIC_TOOL_LOOP_THRESHOLD
    proxy.GENERIC_TOOL_LOOP_THRESHOLD = 3
    try:
        result = proxy._detect_generic_tool_loop_inplace(msgs, category="local", model_name="poolside/Laguna-S-2.1-NVFP4")
        assert result is False  # nur 2 trailing, nicht >3
    finally:
        proxy.GENERIC_TOOL_LOOP_THRESHOLD = old_threshold


def test_generic_tool_loop_skips_read_and_search():
    """read_file und grep_search werden nicht als generic loops erkannt."""
    msgs = []
    for i in range(4):
        msgs.append(_make_read_msg("/a.py", 1, 10, call_id=f"r{i}"))
        msgs.append({"role": "tool", "tool_call_id": f"r{i}", "content": "code"})
    for i in range(4):
        cid = f"s{i}"
        msgs.append(_make_search_msg("foo", call_id=cid))
        msgs.append(_make_search_result(cid, _NO_MATCH_TEXT))

    old_threshold = proxy.GENERIC_TOOL_LOOP_THRESHOLD
    proxy.GENERIC_TOOL_LOOP_THRESHOLD = 3
    try:
        result = proxy._detect_generic_tool_loop_inplace(msgs, category="local", model_name="poolside/Laguna-S-2.1-NVFP4")
        assert result is False  # read/search haben eigene Detection
    finally:
        proxy.GENERIC_TOOL_LOOP_THRESHOLD = old_threshold


def test_detect_response_loop_flags_generic_tool_repeat():
    """Response-Level: weiterer identischer manage_todo_list-Call wird erkannt."""
    msgs = []
    for i in range(4):
        cid = f"call_t{i}"
        msgs.append(_make_generic_tool_msg("manage_todo_list", _TODO_ARGS, call_id=cid))
        msgs.append({"role": "tool", "tool_call_id": cid, "content": "Aufgabenliste aktualisiert"})
    body = {"messages": msgs}
    new_tc = [{
        "id": "call_new",
        "type": "function",
        "function": {
            "name": "manage_todo_list",
            "arguments": json.dumps(_TODO_ARGS),
        },
    }]

    old_thr = proxy._RESPONSE_LOOP_THRESHOLD
    old_gen = proxy.GENERIC_TOOL_LOOP_THRESHOLD
    proxy._RESPONSE_LOOP_THRESHOLD = 3
    proxy.GENERIC_TOOL_LOOP_THRESHOLD = 3
    try:
        reasons, blocked_names = proxy._detect_response_loop(
            body, new_tc, category="local", model_name="poolside/Laguna-S-2.1-NVFP4")
        assert reasons
        assert "manage_todo_list" in blocked_names
    finally:
        proxy._RESPONSE_LOOP_THRESHOLD = old_thr
        proxy.GENERIC_TOOL_LOOP_THRESHOLD = old_gen


def test_detect_response_loop_allows_different_generic_args():
    """Response-Level: manage_todo_list mit ANDEREN Argumenten wird nicht geblockt."""
    msgs = []
    for i in range(4):
        cid = f"call_t{i}"
        msgs.append(_make_generic_tool_msg("manage_todo_list", _TODO_ARGS, call_id=cid))
        msgs.append({"role": "tool", "tool_call_id": cid, "content": "ok"})
    body = {"messages": msgs}
    new_tc = [{
        "id": "call_new",
        "type": "function",
        "function": {
            "name": "manage_todo_list",
            "arguments": json.dumps({"todoList": [{"id": 99, "title": "Neu", "status": "not-started"}]}),
        },
    }]

    old_thr = proxy._RESPONSE_LOOP_THRESHOLD
    old_gen = proxy.GENERIC_TOOL_LOOP_THRESHOLD
    proxy._RESPONSE_LOOP_THRESHOLD = 3
    proxy.GENERIC_TOOL_LOOP_THRESHOLD = 3
    try:
        reasons, blocked_names = proxy._detect_response_loop(
            body, new_tc, category="local", model_name="poolside/Laguna-S-2.1-NVFP4")
        assert reasons == []
        assert blocked_names == []
    finally:
        proxy._RESPONSE_LOOP_THRESHOLD = old_thr
        proxy.GENERIC_TOOL_LOOP_THRESHOLD = old_gen


# ═══════════════════════════════════════════════════════════════════════════
# Local-Model Sampling & Anti-Loop Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_patch_local_sampling_sets_params():
    """_patch_local_sampling_payload setzt alle empfohlenen Parameter."""
    payload = {"model": "test", "messages": []}
    proxy._patch_local_sampling_payload(payload)
    assert payload["temperature"] == proxy.LOCAL_TEMPERATURE
    assert payload["top_p"] == proxy.LOCAL_TOP_P
    assert payload["top_k"] == proxy.LOCAL_TOP_K
    assert payload["min_p"] == proxy.LOCAL_MIN_P
    assert payload["dry_multiplier"] == proxy.LOCAL_DRY_MULTIPLIER
    assert payload["dry_base"] == proxy.LOCAL_DRY_BASE
    assert payload["dry_allowed_length"] == proxy.LOCAL_DRY_ALLOWED_LENGTH
    assert payload["dry_penalty_last_n"] == proxy.LOCAL_DRY_PENALTY_LAST_N
    assert "dry_sequence_breaker" in payload
    assert payload["chat_template_kwargs"]["enable_thinking"] == proxy.LOCAL_ENABLE_THINKING
    assert payload["chat_template_kwargs"]["preserve_thinking"] == proxy.LOCAL_PRESERVE_THINKING


# ── Qwen-Anti-Loop-Sampling (qwen3.8-26b etc.) ─────────────────────────────

def test_patch_qwen_anti_loop_sets_params():
    """Qwen-Modell -> temp=0.3, presence_penalty=0.5, top_p=0.95 erzwungen."""
    payload = {"temperature": 1.0, "top_p": 1.0, "presence_penalty": 0.0}
    proxy._patch_qwen_anti_loop_payload(payload, "qwen3.8-26b")
    assert payload["temperature"] == 0.3
    assert payload["presence_penalty"] == 0.5
    assert payload["top_p"] == 0.95


def test_patch_qwen_anti_loop_idempotent():
    """Bereits korrekte Werte -> keine Aenderung, keine Fehler."""
    payload = {"temperature": 0.3, "top_p": 0.95, "presence_penalty": 0.5}
    proxy._patch_qwen_anti_loop_payload(payload, "Qwen/Qwen3-Next-80B")
    assert payload["temperature"] == 0.3
    assert payload["presence_penalty"] == 0.5
    assert payload["top_p"] == 0.95


def test_patch_qwen_anti_loop_ignores_non_qwen():
    """Nicht-Qwen-Modell -> Payload unveraendert."""
    payload = {"temperature": 1.0, "top_p": 1.0, "presence_penalty": 0.0}
    proxy._patch_qwen_anti_loop_payload(payload, "gpt-4.1-mini")
    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 1.0
    assert payload["presence_penalty"] == 0.0


def test_build_passthrough_payload_qwen_anti_loop(monkeypatch):
    """Passthrough-Payload fuer Qwen local-Modell enthaelt Anti-Loop-Werte."""
    monkeypatch.setitem(proxy._MODEL_CATEGORIES["local"], "model_name", "qwen3.8-26b")
    body = {"model": "irrelevant", "messages": [{"role": "user", "content": "hi"}],
            "temperature": 1.0, "top_p": 1.0, "presence_penalty": 0.0}
    payload = proxy._build_passthrough_payload(body, "local", 0)
    assert payload["temperature"] == 0.3
    assert payload["presence_penalty"] == 0.5
    assert payload["top_p"] == 0.95


def test_build_passthrough_payload_coworker_qwen_anti_loop(monkeypatch):
    """Passthrough-Payload fuer Qwen coworker-Modell enthaelt Anti-Loop-Werte."""
    monkeypatch.setitem(proxy._MODEL_CATEGORIES["coworker"], "model_name", "qwen3.8-26b")
    monkeypatch.setitem(proxy._MODEL_CATEGORIES["coworker"], "api_url", "http://localhost:9999/v1/chat/completions")
    body = {"model": "irrelevant", "messages": [{"role": "user", "content": "hi"}],
            "temperature": 1.0, "top_p": 1.0, "presence_penalty": 0.0}
    payload = proxy._build_passthrough_payload(body, "coworker", 0)
    assert payload["model"] == "qwen3.8-26b"
    assert payload["temperature"] == 0.3
    assert payload["presence_penalty"] == 0.5
    assert payload["top_p"] == 0.95


def test_inject_local_anti_loop_system_no_existing_system():
    """Anti-Loop-Prompt wird an Position 0 eingefuegt wenn kein system vorhanden."""
    msgs = [{"role": "user", "content": "hello"}]
    proxy._inject_local_anti_loop_system(msgs)
    assert msgs[0]["role"] == "system"
    assert "NEVER call the same tool" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"


def test_inject_local_anti_loop_system_after_existing_system():
    """Anti-Loop-Prompt wird NACH existierender system-Message eingefuegt."""
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hello"},
    ]
    proxy._inject_local_anti_loop_system(msgs)
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "You are helpful."
    assert msgs[1]["role"] == "system"
    assert "NEVER call the same tool" in msgs[1]["content"]
    assert msgs[2]["role"] == "user"


def test_clean_payload_keeps_top_k_for_local():
    """_clean_payload mit keep_top_k=True behaelt top_k."""
    payload = {"top_k": 20, "temperature": 0.7, "stop_sequences": ["x"]}
    proxy._clean_payload(payload, keep_tools=True, keep_top_k=True)
    assert payload["top_k"] == 20
    assert "stop_sequences" not in payload


def test_clean_payload_strips_top_k_for_cloud():
    """_clean_payload mit keep_top_k=False entfernt top_k."""
    payload = {"top_k": 20, "temperature": 0.7}
    proxy._clean_payload(payload, keep_tools=True, keep_top_k=False)
    assert "top_k" not in payload


# ═══════════════════════════════════════════════════════════════════════════
# Co-Worker-Delegation (ask_coworker)
# ═══════════════════════════════════════════════════════════════════════════

import asyncio
import time


def _coworker_def(api_url: str = "http://localhost:9999/v1/chat/completions",
                  model_name: str = "qwen3-coder") -> Dict[str, Any]:
    """Standard-Co-Worker-Definition fuer Tests."""
    return {
        "api_url": api_url,
        "api_key": "",
        "model_name": model_name,
        "max_tokens": 8192,
        "use_max_completion_tokens": False,
        "is_vision": False,
        "timeout_seconds": 300,
        "read_timeout_seconds": 120,
        "retry_on_timeout": 0,
        "retry_delay_seconds": 0,
    }


def _setup_coworker_configured(monkeypatch, reachable: bool = True,
                               api_url: str = "http://localhost:9999/v1/chat/completions",
                               fork_join: bool = False):
    """Monkeypatch: coworker konfiguriert + Health-Cache gesetzt.
    fork_join steuert explizit die Zahl der injizierten Tools (1 oder 3)."""
    monkeypatch.setattr(proxy, "COWORKER_ENABLED", True)
    monkeypatch.setattr(proxy, "COWORKER_FORK_JOIN", fork_join)
    monkeypatch.setattr(proxy, "_MODEL_CATEGORIES", {
        **proxy._MODEL_CATEGORIES,
        "coworker": _coworker_def(api_url=api_url),
    })
    monkeypatch.setattr(proxy, "_COWORKER_HEALTH_CACHE", {
        "reachable": reachable,
        "checked_at": time.time(),
        "last_error": "" if reachable else "conn refused",
    })


def test_extract_flag_coworker():
    cleaned, cat, slot = proxy._extract_model_flag("delegiere plan\n--coworker")
    assert cat == "coworker"
    assert slot is None
    assert "--coworker" not in cleaned


def test_extract_flag_coworker_strip_from_messages():
    msgs = [{"role": "user", "content": "brainschreibung\n--coworker"}]
    proxy._strip_model_flags_from_messages(msgs)
    assert msgs[0]["content"] == "brainschreibung"


def test_coworker_in_model_defs():
    """coworker ist Single-Def-Dict wie local -> _model_defs gibt [dict] zurueck."""
    proxy._MODEL_CATEGORIES["coworker"] = _coworker_def()
    defs = proxy._model_defs("coworker")
    assert len(defs) == 1
    assert defs[0]["model_name"] == "qwen3-coder"


def test_coworker_model_defs_empty_when_unconfigured():
    """Ohne Konfiguration (leere api_url/model_name) ist coworker nicht nutzbar."""
    proxy._MODEL_CATEGORIES["coworker"] = {
        "api_url": "", "api_key": "", "model_name": "", "max_tokens": 4096,
        "is_vision": False, "timeout_seconds": 300,
    }
    assert proxy._model_defs("coworker") == []


def test_inject_coworker_tool_disabled(monkeypatch):
    """COWORKER_ENABLED=False -> kein Tool injiziert."""
    _setup_coworker_configured(monkeypatch)
    monkeypatch.setattr(proxy, "COWORKER_ENABLED", False)
    payload = {"messages": []}
    assert proxy._inject_coworker_tool(payload) is False
    assert "tools" not in payload


def test_inject_coworker_tool_health_down(monkeypatch):
    """Health-Check nicht bestanden -> kein Tool injiziert."""
    _setup_coworker_configured(monkeypatch, reachable=False)
    payload = {"messages": []}
    assert proxy._inject_coworker_tool(payload) is False
    assert "tools" not in payload


def test_inject_coworker_tool_not_configured(monkeypatch):
    """coworker ohne gueltige Definition -> kein Tool injiziert."""
    monkeypatch.setattr(proxy, "COWORKER_ENABLED", True)
    monkeypatch.setattr(proxy, "_MODEL_CATEGORIES", {
        **proxy._MODEL_CATEGORIES,
        "coworker": {"api_url": "", "api_key": "", "model_name": ""},
    })
    monkeypatch.setattr(proxy, "_COWORKER_HEALTH_CACHE", {
        "reachable": True, "checked_at": time.time(), "last_error": "",
    })
    payload = {"messages": []}
    assert proxy._inject_coworker_tool(payload) is False
    assert "tools" not in payload


def test_inject_coworker_tool_ok(monkeypatch):
    """Health-OK + konfiguriert -> Tool mit korrektem Schema injiziert."""
    _setup_coworker_configured(monkeypatch)
    payload = {"messages": []}
    assert proxy._inject_coworker_tool(payload) is True
    tools = payload["tools"]
    assert len(tools) == 1
    fn = tools[0]["function"]
    assert fn["name"] == "ask_coworker"
    assert "task" in fn["parameters"]["properties"]
    assert "context" in fn["parameters"]["properties"]
    assert fn["parameters"]["required"] == ["task"]
    assert payload.get("tool_choice") == "auto"


def test_inject_coworker_guidance_system_message(monkeypatch):
    """Bootstrap-Guidance wird in die bestehende system-Message GEMERGET —
    Qwen-Jinja-Templates verbieten system nach Position 0 (500-Fehler)."""
    _setup_coworker_configured(monkeypatch, fork_join=True)
    payload = {"messages": [
        {"role": "system", "content": "You are GitHub Copilot."},
        {"role": "user", "content": "explore the workspace"},
    ]}
    assert proxy._inject_coworker_tool(payload) is True
    msgs = payload["messages"]
    assert len(msgs) == 2  # KEINE zusaetzliche Message — in system gemerged
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"].startswith("You are GitHub Copilot.")
    assert "[PROXY DELEGATION GUIDANCE]" in msgs[0]["content"]
    assert "dispatch_coworker" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"


def test_inject_coworker_guidance_no_client_system(monkeypatch):
    """Ohne client-system wird die Guidance als neue Message an Index 0 gesetzt."""
    _setup_coworker_configured(monkeypatch, fork_join=True)
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    proxy._inject_coworker_tool(payload)
    msgs = payload["messages"]
    assert msgs[0]["role"] == "system"
    assert "[PROXY DELEGATION GUIDANCE]" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"


def test_inject_coworker_guidance_idempotent_with_client_system(monkeypatch):
    """Merge-Idempotenz: bei bestehender system-Message nicht doppelt anhaengen."""
    _setup_coworker_configured(monkeypatch, fork_join=True)
    payload = {"messages": [
        {"role": "system", "content": "You are GitHub Copilot."},
        {"role": "user", "content": "hi"},
    ]}
    proxy._inject_coworker_tool(payload)
    proxy._inject_coworker_tool(payload)
    assert payload["messages"][0]["content"].count("[PROXY DELEGATION GUIDANCE]") == 1
    assert len(payload["messages"]) == 2


def test_inject_coworker_guidance_disabled(monkeypatch):
    """COWORKER_TEACH_DELEGATION=False -> keine Guidance-Message."""
    _setup_coworker_configured(monkeypatch, fork_join=True)
    monkeypatch.setattr(proxy, "COWORKER_TEACH_DELEGATION", False)
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    proxy._inject_coworker_tool(payload)
    assert all(not (isinstance(m, dict) and "[PROXY DELEGATION GUIDANCE]" in str(m.get("content", "")))
               for m in payload["messages"])


def test_inject_coworker_guidance_idempotent(monkeypatch):
    """Guidance wird nur einmal injiziert (Marker-Check)."""
    _setup_coworker_configured(monkeypatch, fork_join=True)
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    proxy._inject_coworker_tool(payload)
    proxy._inject_coworker_tool(payload)
    n = sum(1 for m in payload["messages"]
            if isinstance(m, dict) and "[PROXY DELEGATION GUIDANCE]" in str(m.get("content", "")))
    assert n == 1


def test_inject_coworker_guidance_marker_in_tool_result_ignored(monkeypatch):
    """REGRESSION (Turn 20260823_111957): Der Guidance-Marker in role=tool-
    Messages (z.B. gelesener proxy.py-Quellcode mit dem Literal im String)
    darf die Injection NICHT blockieren — der Check zaehlt nur system-Messages.
    Vorher: has_guidance=True (False Positive) -> Guidance nie gemergt."""
    _setup_coworker_configured(monkeypatch, fork_join=True)
    payload = {"messages": [
        {"role": "system", "content": "You are GitHub Copilot."},
        {"role": "user", "content": "lies proxy.py"},
        {"role": "tool", "tool_call_id": "t1",
         "content": "... _COWORKER_GUIDANCE_SYSTEM (\"[PROXY DELEGATION GUIDANCE]\") ..."},
    ]}
    assert proxy._inject_coworker_tool(payload) is True
    sys_msg = payload["messages"][0]["content"]
    assert "[PROXY DELEGATION GUIDANCE]" in sys_msg  # Guidance WURDE gemergt
    assert len(payload["messages"]) == 3  # keine zusaetzliche Message


def test_inject_coworker_tools_present_but_no_guidance_still_merges(monkeypatch):
    """REGRESSION: ask_coworker bereits in tools (z.B. frueherer Turn) ->
    frueher Early-Return VOR der Guidance. Jetzt: Guidance wird trotzdem
    gemergt, Tools werden nicht dupliziert."""
    _setup_coworker_configured(monkeypatch, fork_join=True)
    payload = {"messages": [
        {"role": "system", "content": "You are GitHub Copilot."},
        {"role": "user", "content": "hi"},
    ], "tools": [
        {"type": "function", "function": {"name": "read_file", "parameters": {}}},
        {"type": "function", "function": {"name": "ask_coworker",
                                          "parameters": {"properties": {}, "required": []}}},
    ]}
    assert proxy._inject_coworker_tool(payload) is True
    names = [t["function"]["name"] for t in payload["tools"]]
    assert names.count("ask_coworker") == 1  # kein Duplikat
    assert "[PROXY DELEGATION GUIDANCE]" in payload["messages"][0]["content"]


def test_inject_coworker_tool_fork_join(monkeypatch):
    """Fork-Join an -> ask + dispatch + collect werden injiziert."""
    _setup_coworker_configured(monkeypatch, fork_join=True)
    payload = {"messages": []}
    assert proxy._inject_coworker_tool(payload) is True
    tools = payload["tools"]
    names = [t["function"]["name"] for t in tools]
    assert names == ["ask_coworker", "dispatch_coworker", "collect_coworker"]
    # dispatch: task required, task_ids optional
    d_fn = tools[1]["function"]
    assert d_fn["parameters"]["required"] == ["task"]
    assert "task_ids" not in d_fn["parameters"].get("required", [])
    # collect: alles optional
    c_fn = tools[2]["function"]
    assert c_fn["parameters"].get("required", []) in ([], None)
    assert "task_ids" in c_fn["parameters"]["properties"]
    assert "timeout_seconds" in c_fn["parameters"]["properties"]


def test_inject_coworker_tool_no_override_choice(monkeypatch):
    """Bestehendes tool_choice wird nicht ueberschrieben."""
    _setup_coworker_configured(monkeypatch, fork_join=True)
    payload = {"messages": [], "tool_choice": "none"}
    proxy._inject_coworker_tool(payload)
    assert payload["tool_choice"] == "none"


def test_inject_coworker_tool_idempotent(monkeypatch):
    """Doppelter Aufruf injiziert die Tools nur je einmal."""
    _setup_coworker_configured(monkeypatch, fork_join=True)
    payload = {"messages": []}
    proxy._inject_coworker_tool(payload)
    proxy._inject_coworker_tool(payload)
    names = [t["function"]["name"] for t in payload["tools"]]
    assert names == ["ask_coworker", "dispatch_coworker", "collect_coworker"]


def test_partition_tool_calls_mixed():
    calls = [
        {"id": "a", "function": {"name": "ask_coworker", "arguments": "{}"}},
        {"id": "b", "function": {"name": "read_file", "arguments": "{}"}},
        {"id": "c", "function": {"name": "ask_coworker", "arguments": "{}"}},
    ]
    dispatch, collect, ask, other = proxy._partition_tool_calls(calls)
    assert len(ask) == 2
    assert len(other) == 1
    assert other[0]["function"]["name"] == "read_file"
    assert dispatch == []
    assert collect == []


def test_partition_tool_calls_fork_join():
    calls = [
        {"id": "a", "function": {"name": "dispatch_coworker", "arguments": "{}"}},
        {"id": "b", "function": {"name": "collect_coworker", "arguments": "{}"}},
        {"id": "c", "function": {"name": "ask_coworker", "arguments": "{}"}},
        {"id": "d", "function": {"name": "read_file", "arguments": "{}"}},
    ]
    dispatch, collect, ask, other = proxy._partition_tool_calls(calls)
    assert [tc["id"] for tc in dispatch] == ["a"]
    assert [tc["id"] for tc in collect] == ["b"]
    assert [tc["id"] for tc in ask] == ["c"]
    assert [tc["id"] for tc in other] == ["d"]


def test_partition_tool_calls_empty():
    dispatch, collect, ask, other = proxy._partition_tool_calls(None)
    assert dispatch == []
    assert collect == []
    assert ask == []
    assert other == []


def test_build_coworker_body_no_leakage(monkeypatch):
    """Frische Session: nur system+user, keine History/Tools/reasoning."""
    monkeypatch.setattr(proxy, "COWORKER_SYSTEM_PROMPT", "SysRolle")
    monkeypatch.setattr(proxy, "COWORKER_TASK_CAP", 0)
    body = proxy._build_coworker_body("plane task", "context snippet")
    assert body["stream"] is False
    assert "tools" not in body
    assert "tool_calls" not in body
    assert "reasoning_content" not in body
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user"]
    assert "context snippet" in body["messages"][1]["content"]


def test_build_coworker_body_caps(monkeypatch):
    """Task+Context werden auf task_cap_chars gekappt."""
    monkeypatch.setattr(proxy, "COWORKER_TASK_CAP", 30)
    monkeypatch.setattr(proxy, "COWORKER_SYSTEM_PROMPT", "SysRolle")
    body = proxy._build_coworker_body("t" * 100, "")
    content = body["messages"][1]["content"]
    assert len(content) <= 30 + len("\n…[gekappt]")
    assert content.endswith("…[gekappt]")


def test_build_coworker_body_appends_extra_context(monkeypatch):
    """Automatisch angehaengter Datei-Kontext landet in der User-Message —
    der Co-Worker bekommt die relevanten Dateiinhalte auch ohne task/context."""
    monkeypatch.setattr(proxy, "COWORKER_SYSTEM_PROMPT", "SysRolle")
    monkeypatch.setattr(proxy, "COWORKER_TASK_CAP", 0)
    body = proxy._build_coworker_body(
        "review this", "kurz",
        extra_context="### Datei: proxy.py\nprint('hi')")
    content = body["messages"][1]["content"]
    assert "review this" in content
    assert "### Datei: proxy.py" in content
    assert "print('hi')" in content


def test_build_coworker_body_task_cap_does_not_cut_files(monkeypatch):
    """Task/Context-Cap gilt nur fuer task+context; der angehaengte
    Datei-Kontext bleibt erhalten (sonst wuerden Dateien bei kleinem Cap
    komplett verloren gehen)."""
    monkeypatch.setattr(proxy, "COWORKER_TASK_CAP", 30)
    monkeypatch.setattr(proxy, "COWORKER_SYSTEM_PROMPT", "SysRolle")
    body = proxy._build_coworker_body(
        "t" * 100, "", extra_context="### Datei: a.py\n" + "x" * 500)
    content = body["messages"][1]["content"]
    assert "…[gekappt]" in content
    assert "### Datei: a.py" in content
    assert "x" * 500 in content


def test_extract_conversation_files_attachments():
    """file-Parts aus user-Messages (VS-Code-Attachments) werden extrahiert,
    inkl. verschachteltem part['file'] und dedupliziert nach Pfad."""
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "review das hier"},
            {"type": "file", "file": {"path": "proxy.py", "content": "def main(): pass"}},
        ]},
        {"role": "user", "content": [
            {"type": "file", "file": {"path": "proxy.py", "content": "def main(): pass"}},
            {"type": "file", "file": {"path": "webui.py", "content": "print('ui')"}},
        ]},
        {"role": "assistant", "content": "ich schau mir das an"},
    ]
    out = proxy._extract_conversation_files(msgs, max_chars=0)
    assert "### Datei: proxy.py" in out
    assert "def main(): pass" in out
    assert "### Datei: webui.py" in out
    assert "ich schau mir das an" not in out
    # proxy.py nur einmal (dedupliziert)
    assert out.count("### Datei: proxy.py") == 1


def test_extract_conversation_files_tool_results():
    """Tool-Ergebnisse (read_file etc.) zaehlen als Datei-Kontext."""
    msgs = [
        {"role": "user", "content": "mach was"},
        {"role": "tool", "name": "read_file", "tool_call_id": "c1",
         "content": "=== proxy.py (1-50) ===\nimport os"},
        {"role": "tool", "name": "read_file", "tool_call_id": "c2",
         "content": "=== webui.py (1-10) ===\nprint('ui')"},
        {"role": "tool", "name": "read_file", "tool_call_id": "c3",
         "content": "=== proxy.py (1-50) ===\nimport os"},
    ]
    out = proxy._extract_conversation_files(msgs, max_chars=0)
    assert "### Tool-Ergebnis (read_file)" in out
    assert "import os" in out
    assert "print('ui')" in out
    # identischer Tool-Text nur einmal
    assert out.count("import os") == 1


def test_extract_conversation_files_cap():
    """Budget kappt den Kontext und markiert das."""
    msgs = [
        {"role": "user", "content": [
            {"type": "file", "file": {"path": "a.py", "content": "x" * 2000}},
        ]},
        {"role": "tool", "name": "read_file", "tool_call_id": "c1",
         "content": "y" * 2000},
    ]
    out = proxy._extract_conversation_files(msgs, max_chars=500)
    assert "gekappt" in out
    assert len(out) < 600


def test_extract_conversation_files_empty():
    assert proxy._extract_conversation_files([], max_chars=0) == ""
    assert proxy._extract_conversation_files(None, max_chars=0) == ""
    assert proxy._extract_conversation_files(
        [{"role": "user", "content": "nur text"}], max_chars=0) == ""


def test_run_coworker_call_error_becomes_tool_result(monkeypatch):
    """Co-Worker-Ausfall wird tool-result mit Fehlertext, kein Crash.
    (Legacy-Pfad: COWORKER_AGENT_MODE=False — der Agent-Mode hat eigene
    Tests in tests/test_coworker_agent.py.)"""
    monkeypatch.setattr(proxy, "COWORKER_AGENT_MODE", False)
    monkeypatch.setattr(proxy, "_MODEL_CATEGORIES", {
        **proxy._MODEL_CATEGORIES,
        "coworker": _coworker_def(api_url="http://127.0.0.1:1/v1/chat/completions"),
    })
    tool_call = {"id": "call_1",
                 "function": {"name": "ask_coworker",
                              "arguments": json.dumps({"task": "x"})}}
    msg = asyncio.run(proxy._run_coworker_call(tool_call))
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_1"
    assert msg["name"] == "ask_coworker"
    assert "[Co-Worker nicht verfuegbar]" in msg["content"]


def test_delegation_loop_content_only(monkeypatch):
    """Keine tool_calls -> outcome wird direkt durchgereicht."""
    async def fake_call(body, category, force_start_idx=None):
        return {"result": {"status": "ok", "content": "direkt", "tool_calls": None},
                "used_idx": 0, "used_model": "main", "attempts": [], "all_failed": False}
    monkeypatch.setattr(proxy, "_call_model_with_fallbacks", fake_call)
    body = {"messages": [{"role": "user", "content": "test"}]}
    outcome = asyncio.run(proxy._delegation_loop(body, "local"))
    assert outcome["result"]["content"] == "direkt"


def test_delegation_loop_passthrough_other_tools(monkeypatch):
    """Nur VS-Code-Tools -> unveraendert an VS Code durchreichen."""
    async def fake_call(body, category, force_start_idx=None):
        return {"result": {"status": "ok", "content": "", "tool_calls": [
            {"id": "r1", "type": "function",
             "function": {"name": "read_file", "arguments": json.dumps({"filePath": "x"})}},
        ]}, "used_idx": 0, "used_model": "main", "attempts": [], "all_failed": False}
    monkeypatch.setattr(proxy, "_call_model_with_fallbacks", fake_call)
    body = {"messages": [{"role": "user", "content": "test"}]}
    outcome = asyncio.run(proxy._delegation_loop(body, "local"))
    tcs = outcome["result"]["tool_calls"]
    assert len(tcs) == 1
    assert tcs[0]["function"]["name"] == "read_file"


def test_delegation_loop_mixed_turn_forwarded(monkeypatch):
    """Gemischter Turn (ask_coworker + read_file): der ask laeuft durch den
    Tunnel (intern final beantwortet), read_file wird an den Client
    durchgereicht."""
    calls = {"n": 0}

    async def fake_call(body, category, force_start_idx=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"result": {"status": "ok", "content": "", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "ask_coworker", "arguments": json.dumps({"task": "t"})}},
                {"id": "r1", "type": "function",
                 "function": {"name": "read_file", "arguments": json.dumps({"filePath": "x"})}},
            ]}, "used_idx": 0, "used_model": "main", "attempts": [], "all_failed": False}
        return {"result": {"status": "ok", "content": "antwort", "tool_calls": None},
                "used_idx": 0, "used_model": "main", "attempts": [], "all_failed": False}

    monkeypatch.setattr(proxy, "_call_model_with_fallbacks", fake_call)
    monkeypatch.setattr(proxy, "COWORKER_MAX_DELEGATIONS", 5)
    body = {"messages": [{"role": "user", "content": "test"}]}
    outcome = asyncio.run(proxy._delegation_loop(body, "local"))
    # read_file wird an den Client durchgereicht (kein interner Block mehr)
    tcs = outcome["result"]["tool_calls"]
    assert tcs is not None
    assert any(t["function"]["name"] == "read_file" for t in tcs)
    # ask_coworker wurde intern als ask/result-Paar in die History gelegt
    tool_msgs = [m for m in body["messages"] if m.get("role") == "tool"]
    assert any(m.get("name") == "ask_coworker" for m in tool_msgs)


def test_delegation_loop_enforces_limit(monkeypatch):
    """Nach max_delegations wird das Modell zur direkten Antwort gezwungen."""
    calls = {"n": 0}

    async def fake_run(tool_call, extra_context=None):
        return {"role": "tool", "tool_call_id": tool_call.get("id"),
                "name": "ask_coworker", "content": "co-worker antwort"}

    async def fake_call(body, category, force_start_idx=None):
        calls["n"] += 1
        if calls["n"] <= 3:
            return {"result": {"status": "ok", "content": "", "tool_calls": [
                {"id": f"c{calls['n']}", "type": "function",
                 "function": {"name": "ask_coworker", "arguments": json.dumps({"task": "t"})}},
            ]}, "used_idx": 0, "used_model": "main", "attempts": [], "all_failed": False}
        return {"result": {"status": "ok", "content": "fertig", "tool_calls": None},
                "used_idx": 0, "used_model": "main", "attempts": [], "all_failed": False}

    monkeypatch.setattr(proxy, "_call_model_with_fallbacks", fake_call)
    monkeypatch.setattr(proxy, "_run_coworker_call", fake_run)
    monkeypatch.setattr(proxy, "COWORKER_MAX_DELEGATIONS", 2)
    body = {"messages": [{"role": "user", "content": "test"}]}
    outcome = asyncio.run(proxy._delegation_loop(body, "local"))
    assert outcome["result"]["content"] == "fertig"
    assert calls["n"] == 4  # 2 Delegations-Runden + Limit-Runde + Finale
    found_limit = any("Limit erreicht" in str(m.get("content", "")) for m in body["messages"])
    assert found_limit
