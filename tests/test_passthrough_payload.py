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


# ── Qwen-Anti-Loop-Sampling (NUR qwen3.8-26b) ──────────────────────────────

def test_patch_qwen_anti_loop_sets_params():
    """qwen3.8-26b -> temp=0.3, presence_penalty=0.5, top_p=0.95 erzwungen."""
    payload = {"temperature": 1.0, "top_p": 1.0, "presence_penalty": 0.0}
    proxy._patch_qwen_anti_loop_payload(payload, "qwen3.8-26b")
    assert payload["temperature"] == 0.3
    assert payload["presence_penalty"] == 0.5
    assert payload["top_p"] == 0.95


def test_patch_qwen_anti_loop_idempotent():
    """Bereits korrekte Werte -> keine Aenderung, keine Fehler."""
    payload = {"temperature": 0.3, "top_p": 0.95, "presence_penalty": 0.5}
    proxy._patch_qwen_anti_loop_payload(payload, "qwen3.8-26b")
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


def test_patch_qwen_anti_loop_ignores_other_qwen_models():
    """BUG-FIX: andere Qwen-Modelle durfen NICHT die Anti-Loop-Parameter bekommen.

    Vorher matchte das Muster pauschal "qwen" und patchte qwen3-coder,
    Qwen/Qwen3-Next-80B etc. mit — die Parameter sind eine Massnahme gegen
    Endlos-Denkschleifen des qwen3.8-26b, nicht der ganzen Familie.
    """
    for name in ("Qwen/Qwen3-Next-80B", "qwen3-coder", "Qwen3.8-Flash-Next",
                 "qwen3.5-27b", "qwen2.5-coder-32b", "Qwen/Qwen3-VL-30B"):
        payload = {"temperature": 1.0, "top_p": 1.0, "presence_penalty": 0.0}
        proxy._patch_qwen_anti_loop_payload(payload, name)
        assert payload == {"temperature": 1.0, "top_p": 1.0, "presence_penalty": 0.0}, name


def test_is_qwen_anti_loop_model_matches_26b_spelling_variants():
    """Schreibvarianten des 26b treffen, artverwandte Namen nicht."""
    for name in ("qwen3.8-26b", "Qwen3.8_26B", "qwen 3.8 26b", "dgx-qwen3.8-26b-nvfp4"):
        assert proxy._is_qwen_anti_loop_model(name), name
    for name in ("qwen3.8-27b", "qwen3.8-flash-next", "qwen3-next-80b", "", "gpt-4.1"):
        assert not proxy._is_qwen_anti_loop_model(name), name


# ── Thinking-OFF-Schalter (Worker / Co-Worker) ─────────────────────────────

def _thinking_off_body():
    return {"messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high",
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}


def test_local_thinking_off_overrides_everything(monkeypatch):
    """Worker-Schalter: kein reasoning_effort, enable_thinking=false — egal
    was Client-Request oder LOCAL_THINKING_MODE sagen."""
    monkeypatch.setattr(proxy, "LOCAL_THINKING_OFF", True)
    monkeypatch.setattr(proxy, "LOCAL_THINKING_MODE", "high")
    p = proxy._build_passthrough_payload(_thinking_off_body(), "local", 0)
    assert "reasoning_effort" not in p
    assert p["chat_template_kwargs"]["enable_thinking"] is False
    assert p["chat_template_kwargs"]["preserve_thinking"] is False


def test_local_thinking_off_does_not_touch_coworker(monkeypatch):
    """Der Worker-Schalter gilt NUR fuer Kategorie local."""
    monkeypatch.setattr(proxy, "LOCAL_THINKING_OFF", True)
    monkeypatch.setattr(proxy, "COWORKER_THINKING_OFF", False)
    monkeypatch.setattr(proxy, "LOCAL_THINKING_MODE", "")
    monkeypatch.setitem(proxy._MODEL_CATEGORIES["coworker"], "api_url",
                        "http://localhost:9999/v1/chat/completions")
    p = proxy._build_passthrough_payload(_thinking_off_body(), "coworker", 0)
    assert p.get("reasoning_effort") == "high"
    assert p["chat_template_kwargs"]["enable_thinking"] is True


def test_coworker_thinking_off_applies_on_coworker_category(monkeypatch):
    """Co-Worker-Schalter: greift auf Kategorie coworker (Tunnel, ask_coworker
    und dispatch laufen alle ueber diesen Payload-Builder)."""
    monkeypatch.setattr(proxy, "COWORKER_THINKING_OFF", True)
    monkeypatch.setattr(proxy, "LOCAL_THINKING_OFF", False)
    monkeypatch.setitem(proxy._MODEL_CATEGORIES["coworker"], "api_url",
                        "http://localhost:9999/v1/chat/completions")
    p = proxy._build_passthrough_payload(_thinking_off_body(), "coworker", 0)
    assert "reasoning_effort" not in p
    assert p["chat_template_kwargs"]["enable_thinking"] is False
    assert p["chat_template_kwargs"]["preserve_thinking"] is False


def test_coworker_thinking_off_off_by_default(monkeypatch):
    """Beide Schalter aus -> Thinking-Parameter bleiben unveraendert."""
    monkeypatch.setattr(proxy, "COWORKER_THINKING_OFF", False)
    monkeypatch.setattr(proxy, "LOCAL_THINKING_OFF", False)
    monkeypatch.setattr(proxy, "LOCAL_THINKING_MODE", "")
    p = proxy._build_passthrough_payload(_thinking_off_body(), "local", 0)
    assert p["reasoning_effort"] == "high"
    assert p["chat_template_kwargs"]["enable_thinking"] is True


def test_force_thinking_off_helper_creates_chat_template_kwargs():
    """Helper erzeugt chat_template_kwargs, wenn der Client keine mitschickt."""
    payload = {"reasoning": {"effort": "high"}}
    proxy._force_thinking_off_payload(payload, "test")
    assert "reasoning" not in payload and "reasoning_effort" not in payload
    assert payload["chat_template_kwargs"] == {"enable_thinking": False,
                                               "preserve_thinking": False}


def test_apply_config_file_reads_thinking_off_keys(monkeypatch):
    """_apply_config_file muss tokens.local_sampling.thinking_off und
    tokens.coworker.thinking_off lesen."""
    cfg = {"tokens": {"local_sampling": {"thinking_off": True},
                      "coworker": {"thinking_off": True}}}
    monkeypatch.setattr(proxy, "_WEBUI_AVAILABLE", True)
    monkeypatch.setattr(proxy, "_webui_load_config", lambda: cfg)
    old_local = proxy.LOCAL_THINKING_OFF
    old_cw = proxy.COWORKER_THINKING_OFF
    try:
        proxy._apply_config_file()
        assert proxy.LOCAL_THINKING_OFF is True
        assert proxy.COWORKER_THINKING_OFF is True
    finally:
        proxy.LOCAL_THINKING_OFF = old_local
        proxy.COWORKER_THINKING_OFF = old_cw


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
    fork_join steuert explizit die Zahl der injizierten Tools (1 oder 3).
    COWORKER_DRIVER_MODE wird auf False gepinnt: diese Tests pruefen die
    Default-Guidance, und data/config.json kann driver_mode=true ausliefern."""
    monkeypatch.setattr(proxy, "COWORKER_ENABLED", True)
    monkeypatch.setattr(proxy, "COWORKER_FORK_JOIN", fork_join)
    monkeypatch.setattr(proxy, "COWORKER_DRIVER_MODE", False)
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


# ── [EXECUTION RULES] Tool-Execution-Guidance ─────────────────────────────

def test_inject_tool_execution_guidance_merges_into_system():
    """EXECUTION RULES werden in die BESTEHENDE system-Message gemerged."""
    payload = {"messages": [
        {"role": "system", "content": "You are GitHub Copilot."},
        {"role": "user", "content": "hi"},
    ], "tools": [{"type": "function", "function": {"name": "write"}}]}
    proxy._inject_tool_execution_guidance(payload)
    msgs = payload["messages"]
    assert len(msgs) == 2  # keine zusaetzliche Message
    assert msgs[0]["content"].startswith("You are GitHub Copilot.")
    assert "[EXECUTION RULES]" in msgs[0]["content"]
    assert "NEVER write the actual code" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"


def test_inject_tool_execution_guidance_no_client_system():
    """Ohne client-system wird die Guidance als neue Message an Index 0 gesetzt."""
    payload = {"messages": [{"role": "user", "content": "hi"}],
               "tools": [{"type": "function", "function": {"name": "edit"}}]}
    proxy._inject_tool_execution_guidance(payload)
    assert payload["messages"][0]["role"] == "system"
    assert "[EXECUTION RULES]" in payload["messages"][0]["content"]


def test_inject_tool_execution_guidance_idempotent():
    """Merge-Idempotenz: Marker verhindert doppelte Injection."""
    payload = {"messages": [
        {"role": "system", "content": "You are GitHub Copilot."},
        {"role": "user", "content": "hi"},
    ], "tools": [{"type": "function", "function": {"name": "write"}}]}
    proxy._inject_tool_execution_guidance(payload)
    proxy._inject_tool_execution_guidance(payload)
    assert payload["messages"][0]["content"].count("[EXECUTION RULES]") == 1


def test_inject_tool_execution_guidance_no_tools_skipped():
    """Ohne tools im Payload wird nichts injiziert."""
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    proxy._inject_tool_execution_guidance(payload)
    assert all("[EXECUTION RULES]" not in str(m.get("content", ""))
               for m in payload["messages"])


def test_build_passthrough_payload_injects_execution_rules(monkeypatch):
    """_build_passthrough_payload injiziert [EXECUTION RULES] bei vorhandenen
    Tools — unabhängig vom Co-Worker-Health-Check (Regression: f23db6-Turn
    hatte KEINE Guidance, weil Health-Check beim Serverstart noch nicht durch)."""
    monkeypatch.setitem(proxy._MODEL_CATEGORIES["local"], "model_name", "qwen3.8-26b")
    monkeypatch.setattr(proxy, "COWORKER_ENABLED", True)
    monkeypatch.setattr(proxy, "_COWORKER_HEALTH_CACHE", {"reachable": False,
                                                          "last_error": "noch nicht geprueft"})
    body = {"model": "irrelevant", "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "write",
                                                        "parameters": {}}}]}
    payload = proxy._build_passthrough_payload(body, "local", 0)
    sys_content = payload["messages"][0]["content"]
    assert "[EXECUTION RULES]" in sys_content
    assert "[PROXY DELEGATION GUIDANCE]" not in sys_content  # Co-Worker nicht erreichbar


def test_build_passthrough_payload_no_execution_rules_when_delegation_disabled(monkeypatch):
    """BUGFIX: Bei ausgeschalteter Delegation (COWORKER_ENABLED=False) und
    deaktiviertem Fork-Join (COWORKER_FORK_JOIN=False) werden die
    [EXECUTION RULES] NICHT injiziert — der Prompt bleibt unveraendert
    (pure passthrough)."""
    monkeypatch.setitem(proxy._MODEL_CATEGORIES["local"], "model_name", "qwen3.8-26b")
    monkeypatch.setattr(proxy, "COWORKER_ENABLED", False)
    monkeypatch.setattr(proxy, "COWORKER_FORK_JOIN", False)
    monkeypatch.setattr(proxy, "_COWORKER_HEALTH_CACHE", {"reachable": True,
                                                          "last_error": ""})
    body = {"model": "irrelevant", "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "write",
                                                        "parameters": {}}}]}
    payload = proxy._build_passthrough_payload(body, "local", 0)
    assert all("[EXECUTION RULES]" not in str(m.get("content", ""))
               for m in payload["messages"])
    assert "[PROXY DELEGATION GUIDANCE]" not in payload["messages"][0]["content"]


def test_build_passthrough_payload_execution_rules_when_delegation_enabled_no_fork_join(monkeypatch):
    """Delegation AN, Fork-Join AUS: [EXECUTION RULES] werden weiterhin
    injiziert — die Regeln haengen an COWORKER_ENABLED, nicht an Fork-Join."""
    monkeypatch.setitem(proxy._MODEL_CATEGORIES["local"], "model_name", "qwen3.8-26b")
    monkeypatch.setattr(proxy, "COWORKER_ENABLED", True)
    monkeypatch.setattr(proxy, "COWORKER_FORK_JOIN", False)
    monkeypatch.setattr(proxy, "_COWORKER_HEALTH_CACHE", {"reachable": False,
                                                          "last_error": "noch nicht geprueft"})
    body = {"model": "irrelevant", "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "write",
                                                        "parameters": {}}}]}
    payload = proxy._build_passthrough_payload(body, "local", 0)
    assert "[EXECUTION RULES]" in payload["messages"][0]["content"]
