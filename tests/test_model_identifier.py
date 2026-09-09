"""Tests fuer die Modell-Auswahl per Identifier (body["model"]).

Deckt ab:
  1) _normalize_identifier
  2) _resolve_model_identifier: Kategoriename, Kategorie+Slot (1-basiert),
     Modellname-Match (fuzzy, alle Kategorien), unbekannt → None
  3) _identifier_for (kanonische ID fuer /v1/models)
  4) Integration in _handle_chat_completion: Identifier schlaegt
     Prompt-Flag-Historie; Prompt-Flag im aktuellen Turn schlaegt den Slot
     des Identifiers; unbekannter Identifier → DEFAULT_CATEGORY
  5) /v1/models liefert Identifier + Backend-Alias
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_proxy_module():
    spec = importlib.util.spec_from_file_location("proxy_ident_test", REPO_ROOT / "proxy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["proxy_ident_test"] = module
    spec.loader.exec_module(module)
    return module


try:
    proxy = _load_proxy_module()
    HAS_PROXY = True
    SKIP_REASON = ""
except Exception as e:  # pragma: no cover
    HAS_PROXY = False
    SKIP_REASON = f"proxy.py konnte nicht importiert werden: {e}"

pytestmark = pytest.mark.skipif(not HAS_PROXY, reason=SKIP_REASON)


# ── Fixture: bekannte Test-Kategorien ──────────────────────────────────────
@pytest.fixture
def fake_categories(monkeypatch):
    cats = {
        "local": {
            "api_url": "http://local.test/v1/chat/completions",
            "model_name": "Qwen/Qwen3-Next-80B",
            "max_tokens": 1024,
        },
        "coworker": {
            "api_url": "http://cw.test/v1/chat/completions",
            "model_name": "Qwen3-Coder-30B",
            "max_tokens": 1024,
        },
        "light": [
            {"api_url": "http://l1.test/v1", "model_name": "gpt-4.1-mini"},
            {"api_url": "http://l2.test/v1", "model_name": "gpt-4.1-nano"},
            {"api_url": "http://l3.test/v1", "model_name": "Qwen3.8-27b-instruct"},
        ],
        "strong": [
            {"api_url": "http://s1.test/v1", "model_name": "claude-sonnet-4"},
            {"api_url": "http://s2.test/v1", "model_name": "claude-opus-4"},
            {"api_url": "", "model_name": ""},  # leer → ignoriert
        ],
        "vision": [
            {"api_url": "http://v1.test/v1", "model_name": "gpt-4o"},
        ],
    }
    monkeypatch.setattr(proxy, "_MODEL_CATEGORIES", cats)
    return cats


# ═══════════════════════════════════════════════════════════════════════════
# 1. Normalisierung
# ═══════════════════════════════════════════════════════════════════════════

def test_normalize_identifier_strips_separators():
    assert proxy._normalize_identifier("Qwen3.8-27B") == "qwen3827b"
    assert proxy._normalize_identifier("Qwen/Qwen3-Next-80B") == "qwenqwen3next80b"
    assert proxy._normalize_identifier("  gpt-4.1_mini ") == "gpt41mini"


def test_normalize_identifier_empty():
    assert proxy._normalize_identifier("") == ""


# ═══════════════════════════════════════════════════════════════════════════
# 2. Resolver
# ═══════════════════════════════════════════════════════════════════════════

def test_resolve_category_name(fake_categories):
    assert proxy._resolve_model_identifier("local") == ("local", 0)
    assert proxy._resolve_model_identifier("light") == ("light", 0)
    assert proxy._resolve_model_identifier("strong") == ("strong", 0)
    assert proxy._resolve_model_identifier("vision") == ("vision", 0)
    assert proxy._resolve_model_identifier("coworker") == ("coworker", 0)


def test_resolve_category_is_case_insensitive(fake_categories):
    assert proxy._resolve_model_identifier("LIGHT") == ("light", 0)
    assert proxy._resolve_model_identifier("Strong") == ("strong", 0)


def test_resolve_category_slot_one_based(fake_categories):
    # 1-basiert: light1 == light == 1. Modell, light2 = 2. Modell
    assert proxy._resolve_model_identifier("light1") == ("light", 0)
    assert proxy._resolve_model_identifier("light2") == ("light", 1)
    assert proxy._resolve_model_identifier("light3") == ("light", 2)
    assert proxy._resolve_model_identifier("vision1") == ("vision", 0)


def test_resolve_slot_out_of_range_falls_back_to_primary(fake_categories):
    # vision hat nur 1 Slot → vision5 → (vision, 0) statt None
    assert proxy._resolve_model_identifier("vision5") == ("vision", 0)


def test_resolve_slot_zero_invalid(fake_categories):
    assert proxy._resolve_model_identifier("light0") is None


def test_resolve_model_name_fuzzy(fake_categories):
    # Teilstring-Match in beide Richtungen, normalisiert
    assert proxy._resolve_model_identifier("Qwen3.8-27b") == ("light", 2)
    assert proxy._resolve_model_identifier("qwen3.8-27b-instruct") == ("light", 2)
    assert proxy._resolve_model_identifier("gpt-4.1-mini") == ("light", 0)
    assert proxy._resolve_model_identifier("claude-sonnet-4") == ("strong", 0)
    assert proxy._resolve_model_identifier("gpt-4o") == ("vision", 0)


def test_resolve_model_name_searches_all_categories(fake_categories):
    # "Qwen3" matcht local (Qwen/Qwen3-Next-80B) UND coworker (Qwen3-Coder-30B)
    # UND light[2] (Qwen3.8-27b-instruct) → Kategorie-Reihenfolge: local zuerst
    assert proxy._resolve_model_identifier("Qwen3") == ("local", 0)


def test_resolve_unknown_returns_none(fake_categories):
    assert proxy._resolve_model_identifier("gpt-5-turbo") is None
    assert proxy._resolve_model_identifier("") is None
    assert proxy._resolve_model_identifier("nonsense-name") is None


# ═══════════════════════════════════════════════════════════════════════════
# 3. Kanonische Identifier
# ═══════════════════════════════════════════════════════════════════════════

def test_identifier_for():
    assert proxy._identifier_for("light", 0) == "light"
    assert proxy._identifier_for("light", 1) == "light2"
    assert proxy._identifier_for("light", 2) == "light3"
    assert proxy._identifier_for("local", 0) == "local"


# ═══════════════════════════════════════════════════════════════════════════
# 4. Integration in _handle_chat_completion
# ═══════════════════════════════════════════════════════════════════════════

class _Captured(Exception):
    """Sentinel: faengt den (category, force_start_idx) am Stream-Eintritt."""


def _install_stream_capture(monkeypatch) -> Dict[str, Any]:
    captured: Dict[str, Any] = {}

    def fake_stream_events(body, category, force_start_idx=None):
        # Bewusst sync: der Handler ruft _stream_events(...) direkt auf, bevor
        # _io_tee den Generator iteriert — so fangen wir (category, idx) sofort.
        captured["body"] = body
        captured["category"] = category
        captured["force_start_idx"] = force_start_idx
        raise _Captured()

    monkeypatch.setattr(proxy, "_stream_events", fake_stream_events)
    return captured


async def _run_handler(body: Dict[str, Any]) -> Dict[str, Any]:
    try:
        await proxy._handle_chat_completion(body)
    except _Captured:
        pass
    return body


def _body(model: str, text: str = "hallo") -> Dict[str, Any]:
    return {"model": model, "stream": True,
            "messages": [{"role": "user", "content": text}]}


def test_identifier_category_wins(fake_categories, monkeypatch):
    captured = _install_stream_capture(monkeypatch)

    import asyncio
    asyncio.run(_run_handler(_body("vision", "beschreibe das bild")))

    assert captured["category"] == "vision"
    assert captured["force_start_idx"] is None


def test_identifier_slot_sets_force_start_idx(fake_categories, monkeypatch):
    captured = _install_stream_capture(monkeypatch)

    import asyncio
    asyncio.run(_run_handler(_body("light2")))

    assert captured["category"] == "light"
    assert captured["force_start_idx"] == 1


def test_identifier_model_match(fake_categories, monkeypatch):
    captured = _install_stream_capture(monkeypatch)

    import asyncio
    asyncio.run(_run_handler(_body("Qwen3.8-27b")))

    assert captured["category"] == "light"
    assert captured["force_start_idx"] == 2


def test_identifier_beats_history_flag(fake_categories, monkeypatch):
    """Identifier im body gewinnt gegen ein --flag in der Historie."""
    captured = _install_stream_capture(monkeypatch)

    body = {
        "model": "strong",
        "stream": True,
        "messages": [
            {"role": "user", "content": "alt --vision"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "neue frage"},
        ],
    }

    import asyncio
    asyncio.run(_run_handler(body))

    assert captured["category"] == "strong"


def test_current_turn_flag_slot_beats_identifier_slot(fake_categories, monkeypatch):
    """--light 3 im aktuellen Turn schlaegt den Slot aus model='light2'."""
    captured = _install_stream_capture(monkeypatch)

    import asyncio
    asyncio.run(_run_handler(_body("light2", "mach was --light 3")))

    assert captured["category"] == "light"
    assert captured["force_start_idx"] == 2


def test_unknown_identifier_falls_back_to_default(fake_categories, monkeypatch):
    captured = _install_stream_capture(monkeypatch)

    import asyncio
    asyncio.run(_run_handler(_body("does-not-exist-xyz")))

    assert captured["category"] == proxy.DEFAULT_CATEGORY


def test_no_identifier_uses_prompt_flag(fake_categories, monkeypatch):
    """Ohne model (leer) bleibt das Flag-System unveraendert nutzbar."""
    captured = _install_stream_capture(monkeypatch)

    import asyncio
    asyncio.run(_run_handler(_body("", "architektur bitte --strong 2")))

    assert captured["category"] == "strong"
    assert captured["force_start_idx"] == 1


def test_flags_stripped_from_messages(fake_categories, monkeypatch):
    captured = _install_stream_capture(monkeypatch)

    import asyncio
    body = _body("light2", "text --light 3")
    asyncio.run(_run_handler(body))

    content = captured["body"]["messages"][-1]["content"]
    assert "--light" not in content


# ═══════════════════════════════════════════════════════════════════════════
# 5. /v1/models
# ═══════════════════════════════════════════════════════════════════════════

def test_list_models_exposes_identifiers(fake_categories, monkeypatch):
    import asyncio

    monkeypatch.setattr(proxy, "PROXY_AUTH_ENABLED", False)

    class _Headers:
        @staticmethod
        def get(key, default=""):
            return default

    class _Url:
        path = "/v1/models"

    class _QueryParams:
        @staticmethod
        def get(key, default=""):
            return default

    class _Req:
        headers = _Headers()
        url = _Url()
        query_params = _QueryParams()

    resp = asyncio.run(proxy.list_models(_Req()))
    import json
    payload = json.loads(resp.body.decode())
    ids = [m["id"] for m in payload["data"]]

    assert "light" in ids
    assert "light2" in ids
    assert "light3" in ids
    assert "gpt-4.1-mini" in ids  # Backend-Alias

    by_id = {m["id"]: m for m in payload["data"]}
    assert by_id["light2"]["owned_by"] == "proxy:light[1]"
    assert by_id["light2"]["backend_model"] == "gpt-4.1-nano"
    assert by_id["gpt-4.1-mini"]["proxy_identifier"] == "light"
