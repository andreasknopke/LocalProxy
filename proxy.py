"""
LocalProxy v3.0 — Single-Model Pass-Through Proxy (OpenAI-kompatibel)

Architektur:
  VS Code Copilot → FastAPI Gateway → Modell (1 von 4 Kategorien)
    ├─ Hindsight Recall (System-Message-Praefix)
    ├─ Transparente Modifikationen:
    │   ├─ Moonshot-Parameter-Patch (nur bei moonshot-ai URL)
    │   ├─ image_url-Sanitizer (wenn is_vision=False)
    │   └─ Tool-Result-Capping (Token-Bombing-Schutz)
    └─ Pass-Through Streaming → VS Code Copilot

Komponenten:
  1. Qdrant-basiertes Hindsight Memory
  2. 4 Modell-Kategorien: local, light, strong, vision
  3. Prompt-Flag-Steuerung: --local, --light, --strong, --vision
  4. WebUI-Konfiguration (webui.py)
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
import secrets
import sys
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Set, Tuple

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from dataclasses import dataclass, field

# ── UTF-8 erzwingen (Docker slim-Images haben oft kein Locale) ─────────
# Muss VOR jeglichem print() passieren.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
try:
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── Logging ────────────────────────────────────────────────────────────────
import datetime as _dt

LOG_FILE = os.getenv("LOG_FILE", str(Path(__file__).parent / "data" / "proxy.log"))


def _log(msg: str) -> None:
    """Schreibt eine Log-Zeile mit Timestamp in Datei + stdout.
    Faengt UnicodeEncodeError ab (z.B. wenn stdout ASCII-only ist in Docker/CI).
    """
    timestamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    # Robustes stdout: fallback auf ASCII-safe print
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _truncate_key(key: str) -> str:
    """Kuerzt einen API-Key: erste 8 Zeichen + '...'"""
    if not key:
        return "(leer)"
    if len(key) <= 8:
        return key
    return key[:8] + "..."


def _safe_str(val: object) -> str:
    """Konvertiert zu str mit ASCII-safe Fallback. Fangt ALLE Exceptions, NIEMALS crashen."""
    try:
        s = str(val)
        try:
            s.encode("ascii")
            return s
        except UnicodeEncodeError:
            # Enthaelt non-ASCII: mit replace konvertieren
            return s.encode("ascii", errors="replace").decode("ascii")
    except Exception:
        # Letzter Ausweg: repr und dann ASCII-safe
        try:
            r = repr(val)
            return r.encode("ascii", errors="replace").decode("ascii")
        except Exception:
            return "(unbeschreiblich)"


# ── Webinterface ───────────────────────────────────────────────────────────
try:
    from webui import mount_webui, _load_config as _webui_load_config
    _WEBUI_AVAILABLE = True
except Exception:
    _WEBUI_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════
# Konfiguration ── Zero-Config-Startwerte, via config.json + Env ueberschreibbar
# ═══════════════════════════════════════════════════════════════════════════

# ── Modell-Kategorien (4 Slots) ────────────────────────────────────────────
# local: Single-Def (keine Fallbacks)
# light/strong/vision: Array mit bis zu 3 Slots [Primary, Fallback 2, Fallback 3]
_MODEL_CATEGORIES: Dict[str, Any] = {
    "local": {
        "api_url": os.getenv("LOCAL_API_URL", "http://localhost:8000/v1/chat/completions"),
        "api_key": os.getenv("LOCAL_API_KEY", ""),
        "model_name": os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen3-Next-80B"),
        "max_tokens": int(os.getenv("LOCAL_MAX_TOKENS", "65536")),
        "is_vision": os.getenv("LOCAL_IS_VISION", "false").lower() in {"1", "true", "yes", "y", "on"},
        "timeout_seconds": float(os.getenv("LOCAL_TIMEOUT_SECONDS", "300")),
        "label": "local primary",
    },
    "light": [
        {
            "label": "light primary",
            "api_url": os.getenv("LIGHT_API_URL", "https://api.openai.com/v1/chat/completions"),
            "api_key": os.getenv("LIGHT_API_KEY", ""),
            "model_name": os.getenv("LIGHT_MODEL_NAME", "gpt-4.1-mini"),
            "max_tokens": int(os.getenv("LIGHT_MAX_TOKENS", "65536")),
            "is_vision": os.getenv("LIGHT_IS_VISION", "false").lower() in {"1", "true", "yes", "y", "on"},
            "timeout_seconds": float(os.getenv("LIGHT_TIMEOUT_SECONDS", "180")),
        },
        {"label": "light fallback 2", "api_url": "", "api_key": "", "model_name": "", "max_tokens": 65536, "is_vision": False, "timeout_seconds": 180},
        {"label": "light fallback 3", "api_url": "", "api_key": "", "model_name": "", "max_tokens": 65536, "is_vision": False, "timeout_seconds": 180},
    ],
    "strong": [
        {
            "label": "strong primary",
            "api_url": os.getenv("STRONG_API_URL", "https://api.anthropic.com/v1/chat/completions"),
            "api_key": os.getenv("STRONG_API_KEY", ""),
            "model_name": os.getenv("STRONG_MODEL_NAME", "claude-sonnet-4-20250514"),
            "max_tokens": int(os.getenv("STRONG_MAX_TOKENS", "65536")),
            "is_vision": os.getenv("STRONG_IS_VISION", "false").lower() in {"1", "true", "yes", "y", "on"},
            "timeout_seconds": float(os.getenv("STRONG_TIMEOUT_SECONDS", "300")),
        },
        {"label": "strong fallback 2", "api_url": "", "api_key": "", "model_name": "", "max_tokens": 65536, "is_vision": False, "timeout_seconds": 300},
        {"label": "strong fallback 3", "api_url": "", "api_key": "", "model_name": "", "max_tokens": 65536, "is_vision": False, "timeout_seconds": 300},
    ],
    "vision": [
        {
            "label": "vision primary",
            "api_url": os.getenv("VISION_API_URL", "https://api.openai.com/v1/chat/completions"),
            "api_key": os.getenv("VISION_API_KEY", ""),
            "model_name": os.getenv("VISION_MODEL_NAME", "gpt-4o"),
            "max_tokens": int(os.getenv("VISION_MAX_TOKENS", "65536")),
            "is_vision": os.getenv("VISION_IS_VISION", "true").lower() in {"1", "true", "yes", "y", "on"},
            "timeout_seconds": float(os.getenv("VISION_TIMEOUT_SECONDS", "180")),
        },
        {"label": "vision fallback 2", "api_url": "", "api_key": "", "model_name": "", "max_tokens": 65536, "is_vision": True, "timeout_seconds": 180},
        {"label": "vision fallback 3", "api_url": "", "api_key": "", "model_name": "", "max_tokens": 65536, "is_vision": True, "timeout_seconds": 180},
    ],
}

DEFAULT_CATEGORY: str = os.getenv("DEFAULT_CATEGORY", "light")

# ── Fallback-System ─────────────────────────────────────────────────────────
# Aktueller aktiver Index pro Kategorie (light/strong/vision)
_CATEGORY_ACTIVE_IDX: Dict[str, int] = {"local": 0, "light": 0, "strong": 0, "vision": 0}

COOLDOWN_FILE: Path = Path(os.getenv("COOLDOWN_FILE", str(Path(__file__).parent / "data" / "cooldowns.json")))
COOLDOWN_DEFAULT_SECONDS: float = float(os.getenv("COOLDOWN_DEFAULT_SECONDS", "300"))


def _model_defs(category: str) -> List[Dict[str, Any]]:
    """Gibt Liste von Modell-Definitionen fuer eine Kategorie zurueck.
    - Bei Array-Struktur (light/strong/vision): gefiltert auf konfigurierte (api_url+model_name nicht leer)
    - Bei Single-Def (local): als 1-elementige Liste
    - Bei nicht-konfiguriert: leere Liste
    """
    cat = _MODEL_CATEGORIES.get(category)
    if isinstance(cat, list):
        result = [d for d in cat if isinstance(d, dict) and d.get("api_url") and d.get("model_name")]
        # Safety-Net: Jede Def nochmal sanitizen
        for d in result:
            _sanitize_def_ascii(d)
        return result
    if isinstance(cat, dict) and cat.get("api_url"):
        _sanitize_def_ascii(cat)
        return [cat]
    return []


def _model_key(category: str, idx: int) -> str:
    """Stabile ID fuer Cooldown-Speicherung: 'category:model_name'."""
    defs = _model_defs(category)
    if 0 <= idx < len(defs):
        return f"{category}:{defs[idx].get('model_name','idx'+str(idx))}"
    return f"{category}:idx{idx}"


def _load_cooldowns() -> Dict[str, float]:
    """Liest Cooldown-Daten aus Datei. Mapping: model_key → expire_timestamp."""
    try:
        with open(COOLDOWN_FILE, "r") as f:
            data = json.load(f)
            return {k: float(v) for k, v in data.items()}
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError):
        return {}


def _save_cooldowns(data: Dict[str, float]) -> None:
    """Schreibt Cooldown-Daten in Datei."""
    try:
        COOLDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(COOLDOWN_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass  # Cooldown-Verlust ≠ Katastrophe


def _is_in_cooldown(category: str, idx: int) -> bool:
    """Prueft, ob eine Modell-Definition im Cooldown ist."""
    expires_at = _load_cooldowns().get(_model_key(category, idx), 0)
    return time.time() < expires_at


def _start_cooldown(category: str, idx: int, duration_override: Optional[float] = None) -> None:
    """Startet Cooldown fuer eine Modell-Definition (Default: 300s)."""
    defs = _model_defs(category)
    if idx >= len(defs):
        return
    duration = duration_override if duration_override is not None else COOLDOWN_DEFAULT_SECONDS
    data = _load_cooldowns()
    key = _model_key(category, idx)
    expires_at = time.time() + max(10.0, float(duration))
    data[key] = expires_at
    _save_cooldowns(data)
    model_name = defs[idx].get("model_name", "?")
    _log(f"Cooldown: {category}[{idx}]={model_name} fuer {max(10.0, float(duration)):.0f}s (bis {time.strftime('%H:%M:%S', time.localtime(expires_at))})")


def _retry_after_seconds(status: int, response_headers: Any) -> Optional[float]:
    """Extrahiert Retry-After Header (nur Sekunden-Format)."""
    if status not in (429, 503):
        return None
    try:
        val = response_headers.get("retry-after") or response_headers.get("Retry-After")
        if val is not None:
            return float(val)
    except (AttributeError, ValueError, TypeError):
        pass
    return None


def _sanitize_def_ascii(d: Dict[str, Any]) -> None:
    """Erzwingt ASCII-only fuer api_key und model_name in-place.
    httpx und Docker-stdout crashen sonst bei non-ASCII in Headers/Payloads.
    """
    for field in ("api_key", "model_name"):
        val = d.get(field)
        if isinstance(val, str):
            try:
                val.encode("ascii")
            except UnicodeEncodeError:
                d[field] = val.encode("ascii", errors="replace").decode("ascii")


# ═══════════════════════════════════════════════════════════════════════════
# Alle initialen Model-Defs ascii-sanitizen
# ═══════════════════════════════════════════════════════════════════════════
for _catkey in ("local", "light", "strong", "vision"):
    _catval = _MODEL_CATEGORIES.get(_catkey)
    if isinstance(_catval, list):
        for _d in _catval:
            if isinstance(_d, dict):
                _sanitize_def_ascii(_d)
    elif isinstance(_catval, dict):
        _sanitize_def_ascii(_catval)

# ── Proxy ──────────────────────────────────────────────────────────────────
PROXY_PORT: int = int(os.getenv("PROXY_PORT", os.getenv("PORT", "9001")))
PROXY_AUTH_ENABLED: bool = os.getenv("PROXY_AUTH_ENABLED", "true").lower() in {"1", "true", "yes", "y", "on"}
PROXY_API_KEY: str = os.getenv("PROXY_API_KEY", "")
if PROXY_AUTH_ENABLED and not PROXY_API_KEY:
    PROXY_API_KEY = "localfox-" + secrets.token_hex(16)
    _log(f"Auto-generated PROXY_API_KEY: {PROXY_API_KEY}")

# ── Hindsight / Qdrant ─────────────────────────────────────────────────────
HINDSIGHT_ENABLED: bool = os.getenv("HINDSIGHT_ENABLED", "true").lower() in {"1", "true", "yes", "y", "on"}
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
HINDSIGHT_COLLECTION: str = os.getenv("HINDSIGHT_COLLECTION", "hindsight_memory")
HINDSIGHT_EMBEDDING_DIM: int = int(os.getenv("HINDSIGHT_EMBEDDING_DIM", "768"))
HINDSIGHT_MAX_MEMORY_TOKENS: int = int(os.getenv("HINDSIGHT_MAX_MEMORY_TOKENS", "4000"))
HINDSIGHT_MIN_SIMILARITY: float = float(os.getenv("HINDSIGHT_MIN_SIMILARITY", "0.18"))
HINDSIGHT_RETAIN_DELAY_SECONDS: float = float(os.getenv("HINDSIGHT_RETAIN_DELAY_SECONDS", "0"))
HINDSIGHT_USE_QDRANT: bool = os.getenv(
    "HINDSIGHT_USE_QDRANT",
    "true" if QDRANT_URL and QDRANT_URL != "http://localhost:6333" else "false",
).lower() in {"1", "true", "yes", "y", "on"}
HINDSIGHT_DIR: Path = Path(os.getenv("HINDSIGHT_DIR", "./data/.hindsight_memory"))

# ── Tool-Result-Capping ────────────────────────────────────────────────────
TOOL_RESULT_CAP: int = int(os.getenv("TOOL_RESULT_CAP", "0"))


# ═══════════════════════════════════════════════════════════════════════════
# Config aus config.json laden (nachdem Globals initialisiert sind)
# ═══════════════════════════════════════════════════════════════════════════

def _apply_config_file() -> None:
    """Uebernimmt Werte aus config.json (wird beim Startup + WebUI-Save aufgerufen).
    Leere Strings aus config.json ueberschreiben NIEMALS gesetzte Env-Var-Werte.
    """
    if not _WEBUI_AVAILABLE:
        return
    try:
        cfg = _webui_load_config()
    except Exception:
        return

    global _MODEL_CATEGORIES, DEFAULT_CATEGORY

    saved_cats = cfg.get("model_categories", {})
    if isinstance(saved_cats, dict):
        for key in ("local", "light", "strong", "vision"):
            if key not in saved_cats:
                continue
            sc = saved_cats[key]

            if isinstance(sc, dict):
                # Legacy / local: Single-Def-Dict
                existing = _MODEL_CATEGORIES.get(key)
                if isinstance(existing, list):
                    # Memory hat Array, config sagt Dict → als 1-elementiges Array behandeln
                    merged: Dict[str, Any] = {}
                    for field in ("label", "api_url", "api_key", "model_name", "max_tokens",
                                   "is_vision", "timeout_seconds"):
                        if field in sc:
                            val = sc[field]
                            if field in ("api_url", "api_key") and isinstance(val, str) and val.strip() == "":
                                continue
                            if field == "max_tokens":
                                val = int(val)
                            elif field == "is_vision":
                                val = bool(val) if not isinstance(val, str) else \
                                    str(val).lower() in {"1", "true", "yes", "y", "on"}
                            elif field == "timeout_seconds":
                                val = float(val)
                            merged[field] = val
                    merged.setdefault("label", key)
                    _MODEL_CATEGORIES[key] = [merged]
                else:
                    cat = _MODEL_CATEGORIES.setdefault(key, {})
                    for field in ("label", "api_url", "api_key", "model_name", "max_tokens",
                                   "is_vision", "timeout_seconds"):
                        if field in sc:
                            val = sc[field]
                            if field in ("api_url", "api_key") and isinstance(val, str) and val.strip() == "":
                                continue
                            if field == "max_tokens":
                                val = int(val)
                            elif field == "is_vision":
                                val = bool(val) if not isinstance(val, str) else \
                                    str(val).lower() in {"1", "true", "yes", "y", "on"}
                            elif field == "timeout_seconds":
                                val = float(val)
                            cat[field] = val

            elif isinstance(sc, list):
                # Neue Array-Struktur (light/strong/vision)
                cleaned_list: List[Dict[str, Any]] = []
                for d in sc:
                    if not isinstance(d, dict):
                        continue
                    element: Dict[str, Any] = {}
                    for field in ("label", "api_url", "api_key", "model_name", "max_tokens",
                                   "is_vision", "timeout_seconds"):
                        if field in d:
                            val = d[field]
                            if field in ("api_url", "api_key") and isinstance(val, str) and val.strip() == "":
                                continue
                            if field == "max_tokens":
                                val = int(val)
                            elif field == "is_vision":
                                val = (bool(val) if not isinstance(val, str) else
                                       str(val).lower() in {"1", "true", "yes", "y", "on"})
                            elif field == "timeout_seconds":
                                val = float(val)
                            element[field] = val
                    cleaned_list.append(element)
                _MODEL_CATEGORIES[key] = cleaned_list

    # Alle geladenen Model-Defs ASCII-sanitizen (non-ASCII API-Keys crashen httpx/Docker-stdout)
    for _k in ("local", "light", "strong", "vision"):
        _v = _MODEL_CATEGORIES.get(_k)
        if isinstance(_v, list):
            for _d in _v:
                if isinstance(_d, dict):
                    _sanitize_def_ascii(_d)
        elif isinstance(_v, dict):
            _sanitize_def_ascii(_v)

    # Active-Indices nach Config-Update validieren
    for key in ("light", "strong", "vision"):
        defs = _model_defs(key)
        if not defs:
            _CATEGORY_ACTIVE_IDX[key] = 0
        elif _CATEGORY_ACTIVE_IDX[key] >= len(defs):
            _CATEGORY_ACTIVE_IDX[key] = 0

    dc = cfg.get("default_category", "")
    if dc in ("local", "light", "strong", "vision"):
        DEFAULT_CATEGORY = str(dc)

    global PROXY_PORT, PROXY_AUTH_ENABLED, PROXY_API_KEY
    PROXY_PORT = int(cfg.get("proxy", {}).get("port", PROXY_PORT))
    PROXY_AUTH_ENABLED = bool(cfg.get("proxy", {}).get("auth_enabled", PROXY_AUTH_ENABLED))
    pk = cfg.get("proxy", {}).get("api_key", "")
    if pk:
        PROXY_API_KEY = pk

    global HINDSIGHT_ENABLED, QDRANT_URL, QDRANT_API_KEY, HINDSIGHT_COLLECTION
    global HINDSIGHT_EMBEDDING_DIM, HINDSIGHT_MAX_MEMORY_TOKENS, HINDSIGHT_MIN_SIMILARITY
    global HINDSIGHT_RETAIN_DELAY_SECONDS, HINDSIGHT_USE_QDRANT, HINDSIGHT_DIR
    hs = cfg.get("hindsight", {})
    if isinstance(hs, dict):
        HINDSIGHT_ENABLED = bool(hs.get("enabled", HINDSIGHT_ENABLED))
        QDRANT_URL = hs.get("qdrant_url", QDRANT_URL)
        if hs.get("qdrant_api_key"):
            QDRANT_API_KEY = hs["qdrant_api_key"]
        HINDSIGHT_COLLECTION = hs.get("collection", HINDSIGHT_COLLECTION)
        HINDSIGHT_EMBEDDING_DIM = int(hs.get("embedding_dim", HINDSIGHT_EMBEDDING_DIM))
        HINDSIGHT_MAX_MEMORY_TOKENS = int(hs.get("max_memory_tokens", HINDSIGHT_MAX_MEMORY_TOKENS))
        HINDSIGHT_MIN_SIMILARITY = float(hs.get("min_similarity", HINDSIGHT_MIN_SIMILARITY))
        HINDSIGHT_RETAIN_DELAY_SECONDS = float(hs.get("retain_delay_seconds", HINDSIGHT_RETAIN_DELAY_SECONDS))
        HINDSIGHT_USE_QDRANT = bool(hs.get("use_qdrant", HINDSIGHT_USE_QDRANT))
        HINDSIGHT_DIR = Path(hs.get("dir", str(HINDSIGHT_DIR)))

    global TOOL_RESULT_CAP
    TOOL_RESULT_CAP = int(cfg.get("tokens", {}).get("tool_result_cap", TOOL_RESULT_CAP))

    _log("Config aus config.json neu geladen")


_apply_config_file()


# ═══════════════════════════════════════════════════════════════════════════
# Prompt-Flag-Extraktion ── Modell-Kategorie-Auswahl via --flag
# ═══════════════════════════════════════════════════════════════════════════

_MODEL_FLAG_PATTERN = re.compile(r'--(local|light|strong|vision)(?:\s+(\d+))?(?:\s|$)', re.IGNORECASE)
_VALID_CATEGORIES: Set[str] = {"local", "light", "strong", "vision"}


def _extract_model_flag(text: str) -> Tuple[str, Optional[str], Optional[int]]:
    """Extrahiert --local/--light/--strong/--vision [1-3] aus dem Text.
    Returns: (bereinigter_text, category_string oder None, slot_number oder None)
    Slot-Nummer: 1=Primary, 2=Fallback 2, 3=Fallback 3. None = kein gültiger Slot angegeben.
    """
    found: Optional[str] = None
    found_slot: Optional[int] = None
    for match in _MODEL_FLAG_PATTERN.finditer(text):
        cat = match.group(1).lower()
        if cat in _VALID_CATEGORIES:
            found = cat
            if match.group(2):
                slot_val = int(match.group(2))
                found_slot = slot_val if 1 <= slot_val <= 3 else None
            else:
                found_slot = None
    cleaned = _MODEL_FLAG_PATTERN.sub("", text)
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()
    return cleaned, found, found_slot


def _strip_model_flags_from_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Entfernt --local/--light/--strong/--vision [1-3] aus allen User-Messages (in-place)."""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                cleaned, _, _ = _extract_model_flag(content)
                msg["content"] = cleaned
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        cleaned, _, _ = _extract_model_flag(block.get("text", ""))
                        block["text"] = cleaned
    return messages


# ═══════════════════════════════════════════════════════════════════════════
# Reset-Flag ── --reset setzt alle Kategorien auf Primary (Idx=0)
# ═══════════════════════════════════════════════════════════════════════════

_RESET_FLAG_PATTERN = re.compile(r'--reset\b', re.IGNORECASE)


def _detect_reset_flag(text: str) -> bool:
    """Prueft, ob --reset im Text vorkommt."""
    return bool(_RESET_FLAG_PATTERN.search(text))


def _do_reset() -> None:
    """Setzt alle Kategorien auf Primary (Idx=0). Loescht Cooldowns."""
    for key in ("light", "strong", "vision"):
        _CATEGORY_ACTIVE_IDX[key] = 0
    # Cooldowns leeren
    try:
        if COOLDOWN_FILE.exists():
            COOLDOWN_FILE.unlink()
    except OSError:
        pass
    _log("Reset: alle Kategorien auf Primary (Idx=0), Cooldowns geleert")


# ═══════════════════════════════════════════════════════════════════════════
# Hindsight Memory Engine (Qdrant + JSONL-Fallback)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryRecord:
    id: str
    text: str
    networks: List[str]
    created_at: float = field(default_factory=time.time)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "networks": self.networks,
            "created_at": self.created_at,
            "score": self.score,
        }


class HindsightMemory:
    """Hindsight Persistent Memory mit Qdrant (primaer) oder JSONL (Fallback)."""

    def __init__(self) -> None:
        self._qdrant: Optional[QdrantClient] = None
        self._use_qdrant = HINDSIGHT_USE_QDRANT and HINDSIGHT_ENABLED
        if self._use_qdrant:
            try:
                self._qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
                self._ensure_collection()
                _log(f"Hindsight: Qdrant connected @ {QDRANT_URL}")
            except Exception as exc:
                _log(f"Qdrant nicht erreichbar ({_safe_str(exc)}), fallback auf JSONL")
                self._use_qdrant = False
                self._qdrant = None

    def _ensure_collection(self) -> None:
        if self._qdrant is None:
            return
        collections = [c.name for c in self._qdrant.get_collections().collections]
        if HINDSIGHT_COLLECTION not in collections:
            self._qdrant.create_collection(
                collection_name=HINDSIGHT_COLLECTION,
                vectors_config=qdrant_models.VectorParams(
                    size=HINDSIGHT_EMBEDDING_DIM,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )

    @staticmethod
    def _embed(text: str) -> List[float]:
        """Erzeugt einen deterministischen Embedding-Vektor via Feature-Hashing."""
        words = re.findall(r"[A-Za-z0-9_+#\-\[\]\.]{2,}", text.lower())
        vec = np.zeros(HINDSIGHT_EMBEDDING_DIM, dtype=np.float32)
        for i, word in enumerate(words):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16) % HINDSIGHT_EMBEDDING_DIM
            vec[h] += 1.0 / (1.0 + i * 0.01)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def recall(
        self,
        query: str,
        min_similarity: float = HINDSIGHT_MIN_SIMILARITY,
        limit: int = 20,
    ) -> List[MemoryRecord]:
        """Recall: Relevante Erinnerungen aus den 4 Netzwerken abrufen."""
        if not HINDSIGHT_ENABLED:
            return []
        if self._use_qdrant and self._qdrant is not None:
            return self._recall_qdrant(query, min_similarity, limit)
        return self._recall_jsonl(query, min_similarity, limit)

    def _recall_qdrant(self, query: str, min_similarity: float, limit: int) -> List[MemoryRecord]:
        if self._qdrant is None:
            return []
        try:
            vector = self._embed(query)
            results = self._qdrant.search(
                collection_name=HINDSIGHT_COLLECTION,
                query_vector=vector,
                limit=limit,
                score_threshold=min_similarity,
            )
            records: List[MemoryRecord] = []
            for point in results:
                payload = point.payload or {}
                records.append(MemoryRecord(
                    id=str(point.id),
                    text=str(payload.get("text", "")),
                    networks=list(payload.get("networks", [])),
                    created_at=float(payload.get("created_at", time.time())),
                    score=float(point.score),
                ))
            return records
        except Exception as exc:
            _log(f"Qdrant recall fehlgeschlagen: {_safe_str(exc)}")
            return []

    def _recall_jsonl(self, query: str, min_similarity: float, _limit: int) -> List[MemoryRecord]:
        records = _load_memory_records()
        scored = []
        ms = float(min_similarity)
        for rec in records:
            score = max(_text_similarity(query, rec.text), rec.score)
            if score >= ms:
                rec.score = score
                scored.append(rec)
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:_limit]

    def retain(self, body: Dict[str, Any], response_text: str) -> None:
        """Retain & Reflect: Anfrage+Antwort speichern (asynchron)."""
        if not HINDSIGHT_ENABLED:
            return
        text = f"{_last_user_text(body.get('messages', []))}\n\n{response_text}"
        query_hash = _simple_hash(text)
        networks = _classify_networks(text)
        if self._use_qdrant and self._qdrant is not None:
            self._retain_qdrant(query_hash, text, networks)
        else:
            self._retain_jsonl(query_hash, text, networks)

    def _retain_qdrant(self, point_id: str, text: str, networks: List[str]) -> None:
        if self._qdrant is None:
            return
        try:
            existing = self._qdrant.retrieve(
                collection_name=HINDSIGHT_COLLECTION,
                ids=[point_id],
            )
            if existing:
                return
            vector = self._embed(text)
            self._qdrant.upsert(
                collection_name=HINDSIGHT_COLLECTION,
                points=[
                    qdrant_models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "text": _token_budget_guard(text, 6000),
                            "networks": list(dict.fromkeys(networks)),
                            "created_at": time.time(),
                        },
                    )
                ],
            )
        except Exception as exc:
            _log(f"Qdrant retain fehlgeschlagen: {_safe_str(exc)}")

    def _retain_jsonl(self, point_id: str, text: str, networks: List[str]) -> None:
        records = _load_memory_records()
        if any(r.id == point_id for r in records):
            return
        records.append(MemoryRecord(
            id=point_id,
            text=_token_budget_guard(text, 6000),
            networks=list(dict.fromkeys(networks)),
            created_at=time.time(),
        ))
        records.sort(key=lambda r: r.created_at, reverse=True)
        _save_memory_records(records[:500])

    def format_context(self, records: Sequence[MemoryRecord]) -> str:
        """Formatiert Memory-Records als Kontext-Block."""
        if not records:
            return ""
        chunks: List[str] = []
        total_chars = 0
        budget = int(HINDSIGHT_MAX_MEMORY_TOKENS) * 4
        for rec in records:
            nets = ", ".join(rec.networks)
            chunk = f"- [{rec.score:.2f}] [{nets}] {rec.text}"
            if total_chars + len(chunk) > budget:
                break
            chunks.append(chunk)
            total_chars += len(chunk)
        return "\n".join(chunks)

    async def retain_async(self, body: Dict[str, Any], response_text: str) -> None:
        """Asynchroner Retain-Task (wird als Background-Task gestartet)."""
        if HINDSIGHT_RETAIN_DELAY_SECONDS > 0:
            await asyncio.sleep(HINDSIGHT_RETAIN_DELAY_SECONDS)
        self.retain(body, response_text)


_hindsight = HindsightMemory()


# ═══════════════════════════════════════════════════════════════════════════
# Hilfsfunktionen
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, str):
        return text
    return json.dumps(text, ensure_ascii=False)


def _message_text(message: Dict[str, Any]) -> str:
    return _normalize_text(message.get("content", ""))


def _last_user_text(messages: Sequence[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return _message_text(msg)
    return ""


def _token_budget_guard(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[HINDSIGHT_TRUNCATED]\n"


def _simple_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_+#\-]{2,}", text.lower())


def _text_similarity(left: str, right: str) -> float:
    left_tokens = set(_word_tokens(left))
    right_tokens = set(_word_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)


def _classify_networks(text: str) -> List[str]:
    networks = ["Agent Experiences"]
    if re.search(r"\b(api|endpoint|route|server|client|build|pipeline|ci|cd)\b", text, re.IGNORECASE):
        networks.append("World Facts")
    if re.search(r"\b(class|function|module|component|service|file|import)\b", text, re.IGNORECASE):
        networks.append("Entity Summaries")
    if re.search(r"\b(refactor|migration|decision|architecture|design|plan)\b", text, re.IGNORECASE):
        networks.append("Evolving Beliefs")
    return networks


def _load_memory_records() -> List[MemoryRecord]:
    if not HINDSIGHT_ENABLED:
        return []
    records: List[MemoryRecord] = []
    for path in HINDSIGHT_DIR.glob("*.jsonl"):
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    records.append(MemoryRecord(
                        id=str(data.get("id", "")),
                        text=str(data.get("text", "")),
                        networks=list(data.get("networks", [])),
                        created_at=float(data.get("created_at", time.time())),
                        score=float(data.get("score", 0.0)),
                    ))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    return records


def _save_memory_records(records: List[MemoryRecord]) -> None:
    HINDSIGHT_DIR.mkdir(parents=True, exist_ok=True)
    path = HINDSIGHT_DIR / "memory.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")


def _extract_choice_message(result: Dict[str, Any]) -> Dict[str, Any]:
    choices = result.get("choices", [])
    if not choices:
        return {}
    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return {}
    if message.get("content") is None and message.get("reasoning"):
        message = dict(message)
        message["content"] = message["reasoning"]
        message["reasoning"] = None
    return message


def _extract_choice_content(result: Dict[str, Any]) -> str:
    message = _extract_choice_message(result)
    content = message.get("content")
    return content or ""


def _is_tool_continuation(messages: Sequence[Dict[str, Any]]) -> bool:
    if not messages:
        return False
    last = messages[-1]
    if isinstance(last, dict) and last.get("role") == "tool":
        return True
    if isinstance(last, dict) and last.get("role") == "assistant" and bool(last.get("tool_calls")):
        return True
    return False


def _contains_tool_calls(text: str) -> bool:
    if not text:
        return False
    return bool(
        re.search(r'</?(?:tool_call|tool_calls|invoke|function_call)', text)
        or re.search(r'<[a-z_]+_tool', text)
        or "callTool" in text
        or "DSML" in text
        or "\uff5c\uff5ctool_calls" in text
        or "\uff5c\uff5cinvoke" in text
        or "<｜｜DSML｜｜tool_calls>" in text
    )


def _normalize_tool_call_arguments(args: Any) -> str:
    if isinstance(args, str):
        s = args.strip()
        if not s:
            return "{}"
        try:
            parsed = json.loads(s)
            return json.dumps(parsed, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            _log(f"Tool-Arg kein gueltiges JSON, wrappe als string: {s[:80]}")
            return json.dumps({"query": s, "text": s}, ensure_ascii=False)
    if isinstance(args, dict):
        return json.dumps(args, ensure_ascii=False)
    if args is None:
        return "{}"
    try:
        return json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


def _normalize_tool_calls(
    tool_calls: Optional[List[Dict[str, Any]]],
    allowed: Optional[set] = None,
) -> Optional[List[Dict[str, Any]]]:
    if not tool_calls or not isinstance(tool_calls, list):
        return None
    normalized = []
    for i, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            continue
        func = tc.get("function") or {}
        if not isinstance(func, dict):
            func = {}
        name = str(func.get("name", "")).strip()
        if not name:
            continue
        if allowed and name not in allowed:
            continue
        normalized.append({
            "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "index": tc.get("index", i),
            "function": {
                "name": name,
                "arguments": _normalize_tool_call_arguments(func.get("arguments")),
            },
        })
    return normalized or None


def _cut_tool_results_inplace(messages: List[Dict[str, Any]], label: str = "Payload",
                              max_chars: Optional[int] = None) -> int:
    limit = max_chars if max_chars is not None else TOOL_RESULT_CAP
    if limit <= 0:
        return 0
    capped_count = 0
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "tool":
            continue
        content = m.get("content")
        if isinstance(content, str) and len(content) > limit:
            cut = len(content) - limit
            m["content"] = content[:limit] + f"\n...[TRUNCATED: {cut} chars cut]"
            capped_count += 1
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "text":
                    continue
                txt = part.get("content") or part.get("text") or ""
                if isinstance(txt, str) and len(txt) > limit:
                    new_txt = txt[:limit] + "\n...[TRUNCATED]"
                    if "content" in part:
                        part["content"] = new_txt
                    else:
                        part["text"] = new_txt
                    capped_count += 1
    if capped_count:
        _log(f"Tool-Results auf {limit} chars gekappt: {capped_count} messages")
    return capped_count


def _sanitize_image_urls_inplace(messages: List[Dict[str, Any]], label: str = "Payload") -> int:
    removed = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        kept_parts: List[Dict[str, Any]] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                removed += 1
                continue
            kept_parts.append(part)
        if removed and len(kept_parts) != len(content):
            if not kept_parts:
                kept_parts = [{"type": "text", "text": "[image content omitted: text-only model]"}]
            msg["content"] = kept_parts
    if removed:
        _log(f"{label}-Payload: {removed} image_url-Part(s) entfernt (text-only Model)")
    return removed


def _patch_moonshot_payload(payload: Dict[str, Any], api_url: str) -> None:
    """Erzwingt Moonshot-kompatible Parameter — NUR wenn api_url moonshot-ai enthaelt."""
    url_lower = api_url.lower()
    if "moonshot-ai" not in url_lower and "moonshot" not in url_lower:
        return
    fixes = []
    if payload.get("temperature") is not None and payload["temperature"] != 1.0:
        fixes.append(f"temp {payload['temperature']}->1.0")
        payload["temperature"] = 1.0
    if payload.get("top_p") is not None and payload["top_p"] != 0.95:
        fixes.append(f"top_p {payload['top_p']}->0.95")
        payload["top_p"] = 0.95
    if "top_k" in payload:
        fixes.append("top_k entfernt")
        del payload["top_k"]
    for key in ("presence_penalty", "frequency_penalty"):
        val = payload.get(key)
        if val is not None and val != 0.0:
            fixes.append(f"{key} {val}->0")
            payload[key] = 0.0
    if fixes:
        _log(f"Moonshot-Fixes: {'; '.join(fixes)}")


def _clean_payload(payload: Dict[str, Any], keep_tools: bool = False) -> Dict[str, Any]:
    if not payload.get("stream") and "stream_options" in payload:
        payload.pop("stream_options")
    strip_keys = ["stop_sequences", "safety_settings", "response_format", "top_k"]
    if not keep_tools:
        strip_keys += ["tool_choice", "tools", "functions", "function_call"]
    for key in strip_keys:
        payload.pop(key, None)
    return payload


def _derive_models_url(api_url: str) -> str:
    base = api_url.rstrip("/")
    if "/chat/completions" in base:
        return base.rsplit("/chat/completions", 1)[0] + "/models"
    if base.endswith("/v1"):
        return base + "/models"
    return base + "/v1/models"


def _api_headers(api_key: str) -> Dict[str, str]:
    if api_key:
        # Sicherstellen dass Key ASCII-only ist (httpx crasht sonst in Docker)
        try:
            api_key.encode("ascii")
        except UnicodeEncodeError:
            _log("WARNUNG: API-Key enthielt non-ASCII Zeichen, wird bereinigt")
            api_key = api_key.encode("ascii", errors="replace").decode("ascii")
        return {"Authorization": f"Bearer {api_key}"}
    return {}


# ═══════════════════════════════════════════════════════════════════════════
# Payload-Builder ── Pass-Through
# ═══════════════════════════════════════════════════════════════════════════

def _build_passthrough_payload(body: Dict[str, Any], category: str, def_idx: int = 0) -> Dict[str, Any]:
    defs = _model_defs(category)
    cat = defs[def_idx] if defs and def_idx < len(defs) else _model_defs("light")[0]
    payload = copy.deepcopy(body)
    payload["model"] = cat["model_name"]
    payload["max_tokens"] = int(cat.get("max_tokens", 65536))
    payload["stream"] = False

    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        messages = []
        payload["messages"] = messages

    _cut_tool_results_inplace(messages, "Passthrough", TOOL_RESULT_CAP)

    if not cat.get("is_vision", False):
        _sanitize_image_urls_inplace(messages, "Passthrough")

    _patch_moonshot_payload(payload, cat.get("api_url", ""))

    return _clean_payload(payload, keep_tools=True)


# ═══════════════════════════════════════════════════════════════════════════
# Hindsight Recall ── System-Message-Praefix
# ═══════════════════════════════════════════════════════════════════════════

def _inject_hindsight_context(messages: List[Dict[str, Any]]) -> None:
    if not HINDSIGHT_ENABLED:
        return
    query = _last_user_text(messages)
    if not query.strip():
        return
    records = _hindsight.recall(query)
    context = _hindsight.format_context(records)
    if context:
        messages.insert(0, {
            "role": "system",
            "content": f"[HINDSIGHT MEMORY CONTEXT]\n{context}\n[/HINDSIGHT]",
        })
        _log(f"Hindsight-Recall: {len(records)} records als System-Message-Praefix")


# ═══════════════════════════════════════════════════════════════════════════
# Modell-Call ── Single Model Request
# ═══════════════════════════════════════════════════════════════════════════

async def _call_single_model(body: Dict[str, Any], category: str, def_idx: int = 0) -> Dict[str, Any]:
    defs = _model_defs(category)
    if not defs or def_idx >= len(defs):
        return {"category": category, "status": "error", "def_idx": def_idx,
                "content": f"Keine gueltige Modell-Definition fuer {category}[{def_idx}]",
                "trigger_fallback": False}
    cat = defs[def_idx]
    started = time.perf_counter()

    payload = _build_passthrough_payload(body, category, def_idx=def_idx)

    messages = payload.get("messages", [])
    if isinstance(messages, list):
        _inject_hindsight_context(messages)

    model = cat["model_name"]
    api_url = cat["api_url"].rstrip("/")
    api_key = cat.get("api_key", "")
    timeout = float(cat.get("timeout_seconds", 300))

    msg_count = len(messages) if isinstance(messages, list) else 0
    total_chars = sum(len(str(m.get("content", ""))) for m in (messages if isinstance(messages, list) else []))
    _log(f"Single-Model call cat={category}[{def_idx}] model={model} "
         f"api_key={_truncate_key(api_key)} "
         f"messages={msg_count} chars={total_chars} timeout={timeout:.0f}s")

    req_id = f"model_{category}_{def_idx}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    _dump_debug_payload(req_id, f"model_{category}_{def_idx}", payload, extra={
        "category": category, "def_idx": def_idx, "model": model,
        "timeout": timeout, "messages_count": msg_count,
    })
    _register_debug_request(req_id, {
        "type": "model_call_start",
        "category": category, "def_idx": def_idx, "model": model,
        "messages_count": msg_count, "chars": total_chars,
        "timeout": timeout,
    })
    _register_active_call(req_id, {
        "agent_key": category, "def_idx": def_idx, "model": model,
        "phase": "passthrough",
    })

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                api_url, json=payload, headers=_api_headers(api_key), timeout=timeout,
            )
        duration = time.perf_counter() - started
        _finish_active_call(req_id, "done", {"duration_seconds": duration})

        if response.status_code == 200:
            result = response.json()
            message = _extract_choice_message(result)
            tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
            reasoning_content = message.get("reasoning_content") if isinstance(message, dict) else None
            content = _extract_choice_content(result)
            if tool_calls:
                _log(f"Model returned structured tool_calls: {len(tool_calls)}")
            if reasoning_content:
                _log(f"Model returned reasoning_content: {len(reasoning_content)} chars")
            _log(f"Model OK cat={category}[{def_idx}] duration={duration:.1f}s content_len={len(content)}")
            return {
                "category": category, "def_idx": def_idx, "status": "ok",
                "content": content, "message": message,
                "tool_calls": tool_calls, "reasoning_content": reasoning_content,
                "duration_seconds": duration, "usage": result.get("usage"),
                "trigger_fallback": False,
            }

        # Fehlerhafter Status-Code
        err_detail = ""
        try:
            err_body = response.json()
            if isinstance(err_body.get("error"), dict):
                err_detail = _safe_str(err_body["error"].get("message", ""))
            elif isinstance(err_body.get("error"), str):
                err_detail = _safe_str(err_body["error"])
        except Exception:
            err_detail = f"HTTP {response.status_code}"
        _log(f"Model STATUS {response.status_code} cat={category}[{def_idx}] "
             f"duration={duration:.1f}s: {err_detail}")

        # Cooldown-Logik
        should_fallback = True
        if response.status_code == 400:
            # 400 ist Payload-Problem → kein Fallback
            should_fallback = False
        elif response.status_code == 429:
            ra = _retry_after_seconds(response.status_code, getattr(response, "headers", {}))
            _start_cooldown(category, def_idx, duration_override=ra)
            _log(f"   → Rate-Limit: Cooldown fuer {category}[{def_idx}]={model}")
        elif response.status_code in (401, 403):
            _log(f"   → Auth-Fehler, kein Cooldown (Config-Fehler)")
        elif response.status_code >= 500:
            _start_cooldown(category, def_idx, duration_override=60.0)
            _log(f"   → Server-Fehler: Cooldown (60s) fuer {category}[{def_idx}]={model}")

        return {
            "category": category, "def_idx": def_idx, "status": "failed",
            "content": _safe_str(f"Model status {response.status_code}: {err_detail}"),
            "duration_seconds": duration, "usage": None,
            "trigger_fallback": should_fallback,
        }

    except Exception as exc:
        duration = time.perf_counter() - started
        exc_type = type(exc).__name__
        exc_msg = _safe_str(exc)
        _finish_active_call(req_id, "error", {"duration_seconds": duration, "error": exc_msg})
        _log(f"Model ERROR cat={category}[{def_idx}] duration={duration:.1f}s type={exc_type}: {exc_msg}")
        should_fallback = True
        # Timeout/ConnectionError: kurzer Cooldown
        if "timeout" in exc_type.lower() or "connect" in exc_type.lower():
            _start_cooldown(category, def_idx, duration_override=30.0)
            _log(f"   → Timeout/ConnError: Cooldown (30s) fuer {category}[{def_idx}]={model}")
        return {
            "category": category, "def_idx": def_idx, "status": "error",
            "content": _safe_str(f"Model error nach {duration:.0f}s ({exc_type}): {exc_msg}"),
            "duration_seconds": duration, "usage": None,
            "trigger_fallback": should_fallback,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Fallback-Orchestrierung ── Mit Fallback-Chain, Cooldown und 2 Runden
# ═══════════════════════════════════════════════════════════════════════════

_ASYNC_CATEGORY_ACTIVE_IDX: Dict[str, int] = {}  # Lese-/Schreib-Cache fuer _CATEGORY_ACTIVE_IDX


async def _call_model_with_fallbacks(body: Dict[str, Any], category: str,
                                         force_start_idx: Optional[int] = None) -> Dict[str, Any]:
    """Versucht Modelle in Kategorie mit Fallback-Chain.
    - Runde 1: Starte mit active_idx (oder force_start_idx), dann alle anderen, überspringe Cooldowns
    - Runde 2: Alle erneut (Cooldowns ignoriert)
    - Max 2 Runden, dann Fehler.
    """
    defs = _model_defs(category)
    if not defs:
        _log(f"Fallback: Kategorie '{category}' hat keine konfigurierten Modelle")
        return {"all_failed": True, "result": {"status": "error", "category": category,
                "content": f"Kategorie '{category}' hat keine konfigurierten Modelle"},
                "used_idx": 0, "used_model": "(none)", "attempts": []}

    start_idx = force_start_idx if force_start_idx is not None else _CATEGORY_ACTIVE_IDX.get(category, 0)
    if start_idx >= len(defs):
        start_idx = 0
        _CATEGORY_ACTIVE_IDX[category] = 0

    attempted: List[Dict[str, Any]] = []
    last_error_result: Optional[Dict[str, Any]] = None
    last_used_idx: Optional[int] = None

    for round_num in (1, 2):
        # Reihenfolge: start_idx zuerst, dann alle anderen in Array-Reihenfolge
        indices_to_try: List[int] = []
        indices_to_try.append(start_idx)
        for i in range(len(defs)):
            if i != start_idx:
                indices_to_try.append(i)

        for curr_idx in indices_to_try:
            # Runde 1: Cooldowns überspringen
            if round_num == 1 and _is_in_cooldown(category, curr_idx):
                _log(f"Fallback skip R1: {category}[{curr_idx}] = cooldown")
                continue

            _log(f"Fallback try R{round_num}: {category}[{curr_idx}] = {defs[curr_idx].get('model_name', '?')}")
            result = await _call_single_model(body, category, def_idx=curr_idx)
            attempted.append({"idx": curr_idx, "model": defs[curr_idx].get("model_name", "?"), "status": result.get("status")})
            last_used_idx = curr_idx

            if result.get("status") == "ok":
                _log(f"Fallback SUCCESS: {category}[{curr_idx}] = {defs[curr_idx].get('model_name', '?')}")
                # Bei Erfolg: active_idx auf dieses Modell setzen (permanent)
                if curr_idx != start_idx:
                    _CATEGORY_ACTIVE_IDX[category] = curr_idx
                    _log(f"   → neues active primary idx={curr_idx} fuer {category}")

                # model für Hindsight korrekt setzen
                body["model"] = defs[curr_idx].get("model_name", body.get("model", ""))

                return {
                    "result": result,
                    "used_idx": curr_idx,
                    "used_model": defs[curr_idx].get("model_name", "?"),
                    "attempts": attempted,
                    "all_failed": False,
                }

            last_error_result = result

            # Wenn trigger_fallback=False (z.B. 400): weitermachen bringt nichts
            if not result.get("trigger_fallback", True):
                _log(f"   → trigger_fallback=False, breche Fallback-Kette ab")
                break

        # Nach Runde 1: Prüfen ob ein Cooldown-Modell jetzt verwendbar wäre
        # (trotzdem Runde 2 machen)

    # Nach 2 Runden: alle fehlgeschlagen
    _log(f"Fallback EXHAUSTED fuer {category}: {len(attempted)} Versuche fehlgeschlagen")
    err_idx = last_used_idx if last_used_idx is not None else start_idx
    err_def = defs[err_idx] if err_idx < len(defs) else {}
    body["model"] = err_def.get("model_name", body.get("model", ""))
    return {
        "result": last_error_result or {"status": "error", "category": category,
                "content": "Alle Fallbacks fehlgeschlagen"},
        "used_idx": err_idx,
        "used_model": err_def.get("model_name", "(none)"),
        "attempts": attempted,
        "all_failed": True,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Streaming-Formatierung (OpenAI SSE)
# ═══════════════════════════════════════════════════════════════════════════

def _format_openai_stream_chunk(
    model: str,
    content: str = "",
    finish_reason: Optional[str] = None,
    include_role: bool = False,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    reasoning_content: Optional[str] = None,
    chunk_id: Optional[str] = None,
) -> str:
    delta: Dict[str, Any] = {}
    if finish_reason and not tool_calls and not content and not include_role and not reasoning_content:
        pass
    elif tool_calls is not None:
        delta["content"] = None
        if reasoning_content is not None:
            delta["reasoning_content"] = reasoning_content
        delta["tool_calls"] = tool_calls
        if include_role:
            delta["role"] = "assistant"
    else:
        if reasoning_content is not None:
            delta["reasoning_content"] = reasoning_content
        delta["content"] = content
        if include_role:
            delta["role"] = "assistant"
    cid = chunk_id or f"chatcmpl-spark-{uuid.uuid4().hex}"
    payload_data = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload_data, ensure_ascii=False)}\n\n"


def _build_response_payload(
    body: Dict[str, Any], combined_text: str, results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    model = body.get("model", "")
    tool_calls = None
    reasoning_content = None
    for r in reversed(results):
        if r.get("tool_calls") and not tool_calls:
            tool_calls = r["tool_calls"]
        if r.get("reasoning_content") and not reasoning_content:
            reasoning_content = r["reasoning_content"]
        if tool_calls and reasoning_content:
            break

    if tool_calls:
        tool_calls = _normalize_tool_calls(tool_calls)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": combined_text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ═══════════════════════════════════════════════════════════════════════════
# Debug-Infrastructure
# ═══════════════════════════════════════════════════════════════════════════

DEBUG_DIR: Path = Path(os.getenv("DEBUG_DIR", "./data/debug"))
DEBUG_MAX_FILES: int = int(os.getenv("DEBUG_MAX_FILES", "200"))
DEBUG_ENABLED: bool = os.getenv("DEBUG_ENABLED", "1").lower() in {"1", "true", "yes", "on"}

_DEBUG_RING: List[Dict[str, Any]] = []
_DEBUG_RING_MAX: int = int(os.getenv("DEBUG_RING_MAX", "50"))
_ACTIVE_CALLS: Dict[str, Dict[str, Any]] = {}


def _dump_debug_payload(req_id: str, phase: str, payload_to_dump: Dict[str, Any],
                         extra: Optional[Dict[str, Any]] = None) -> None:
    if not DEBUG_ENABLED:
        return
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    snapshot: Dict[str, Any] = {"dumped_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        cloned = copy.deepcopy(payload_to_dump)
        msgs = cloned.get("messages") if isinstance(cloned, dict) else None
        if isinstance(msgs, list):
            summarized = []
            for i, m in enumerate(msgs):
                if not isinstance(m, dict):
                    continue
                content = m.get("content", "")
                content_len = len(str(content)) if content is not None else 0
                content_preview = str(content)[:2000] if content_len > 2000 else content
                tcs = m.get("tool_calls")
                summary = {
                    "idx": i, "role": m.get("role"),
                    "content_len": content_len, "content_preview": content_preview,
                    "has_tool_calls": bool(tcs),
                    "tool_calls_count": len(tcs) if isinstance(tcs, list) else 0,
                }
                if isinstance(tcs, list):
                    summary["tool_call_names"] = [
                        (t.get("function", {}).get("name") if isinstance(t, dict) else None)
                        for t in tcs
                    ]
                    tcs_args = []
                    for t in tcs:
                        if not isinstance(t, dict):
                            tcs_args.append(None)
                            continue
                        fn = t.get("function", {}) if isinstance(t.get("function"), dict) else {}
                        a = fn.get("arguments", "")
                        tcs_args.append({
                            "id": str(t.get("id", ""))[:30],
                            "name": fn.get("name", ""),
                            "args_len": len(str(a)) if a is not None else 0,
                            "args_preview": str(a)[:300] if a is not None else None,
                        })
                    summary["tool_call_details"] = tcs_args
                if m.get("reasoning_content"):
                    rc = m.get("reasoning_content")
                    summary["reasoning_content_len"] = len(str(rc))
                    summary["reasoning_content_preview"] = str(rc)[:500]
                summarized.append(summary)
            cloned["messages"] = summarized
        snapshot["payload"] = cloned
        if extra:
            snapshot["extra"] = extra
    except Exception as exc:
        snapshot["_dump_error"] = str(exc)

    safe_id = "".join(c for c in str(req_id) if c.isalnum() or c in "-_")[:20] or "x"
    safe_phase = "".join(c for c in str(phase) if c.isalnum() or c in "-_")[:40] or "phase"
    filename = f"{safe_id}_{safe_phase}.json"
    try:
        (DEBUG_DIR / filename).write_text(
            json.dumps(snapshot, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


def _cleanup_old_debug_files() -> None:
    if not DEBUG_DIR.exists():
        return
    try:
        files = sorted(DEBUG_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if len(files) > DEBUG_MAX_FILES:
            for f in files[DEBUG_MAX_FILES:]:
                try:
                    f.unlink()
                except Exception:
                    pass
    except Exception:
        pass


def _register_debug_request(req_id: str, info: Dict[str, Any]) -> None:
    global _DEBUG_RING
    info = dict(info)
    info["req_id"] = req_id
    info["ts_iso"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _DEBUG_RING.append(info)
    if len(_DEBUG_RING) > _DEBUG_RING_MAX:
        _DEBUG_RING = _DEBUG_RING[-_DEBUG_RING_MAX:]


def _register_active_call(call_id: str, info: Dict[str, Any]) -> None:
    info = dict(info)
    info["started_at"] = time.time()
    info["started_iso"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _ACTIVE_CALLS[call_id] = info
    asyncio.ensure_future(_active_call_heartbeat(call_id))


async def _active_call_heartbeat(call_id: str) -> None:
    while call_id in _ACTIVE_CALLS:
        await asyncio.sleep(30)
        info = _ACTIVE_CALLS.get(call_id)
        if not info:
            return
        elapsed = time.time() - info.get("started_at", time.time())
        _log(f"[{call_id}] aktiver Call seit {elapsed:.0f}s "
             f"(category={info.get('agent_key','?')}, model={info.get('model','?')})")


def _finish_active_call(call_id: str, status: str = "done",
                          extra: Optional[Dict[str, Any]] = None) -> None:
    info = _ACTIVE_CALLS.pop(call_id, None)
    if not info:
        return
    elapsed = time.time() - info.get("started_at", time.time())
    _log(f"[{call_id}] Call beendet nach {elapsed:.0f}s ({status})")


# ═══════════════════════════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="LocalProxy v3.0 — Single-Model Pass-Through",
    version="3.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
)


# ── Request-Logging-Middleware ─────────────────────────────────────────────
@app.middleware("http")
async def _log_all_requests(request: Request, call_next):
    """Loggt JEDEN eingehenden Request — Method, Path, Auth-Status."""
    path = request.url.path
    # Nur API-Routen loggen, nicht Static/WebUI
    if path.startswith("/webui") or path in ("/docs", "/openapi.json", "/favicon.ico"):
        return await call_next(request)

    auth_header = request.headers.get("authorization", "")
    auth_prefix = ""
    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        if PROXY_AUTH_ENABLED and token:
            auth_prefix = "proxy-key" if secrets.compare_digest(token, PROXY_API_KEY) else "fremd-key"
        else:
            auth_prefix = "no-proxy-auth" if not PROXY_AUTH_ENABLED else "kein-key"
    else:
        auth_prefix = "kein-auth-header"

    _log(f"REQ-IN {request.method} {path} auth={auth_prefix}")
    response = await call_next(request)
    _log(f"REQ-OUT {request.method} {path} status={response.status_code}")
    return response


async def _run_startup_health_checks() -> None:
    """Nicht-blockierende Health-Checks (als Background-Task gestartet)."""
    await asyncio.sleep(1)  # Kurz warten bis Server ready ist

    def _parse_error(r):
        try:
            body = r.json()
            if isinstance(body.get("error"), dict):
                return _safe_str(body["error"].get("message", ""))
            if isinstance(body.get("error"), str):
                return _safe_str(body["error"])
            return _safe_str(body.get("message") or body.get("detail") or "")
        except Exception:
            return f"HTTP {r.status_code}"

    for key in ("local", "light", "strong", "vision"):
        defs = _model_defs(key)
        for i, cat in enumerate(defs):
            api_url = str(cat.get("api_url", "")).rstrip("/")
            if not api_url:
                continue
            _log(f"   {key}[{i}] '{cat.get('model_name','?')}' @ {api_url} "
                 f"api_key={_truncate_key(str(cat.get('api_key', '')))} ...")
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=3.0)) as hc:
                    r = await hc.post(
                        api_url,
                        json={"model": cat.get("model_name", ""),
                              "messages": [{"role": "user", "content": "ping"}],
                              "max_tokens": 1},
                        headers=_api_headers(str(cat.get("api_key", ""))),
                    )
                if r.status_code in (200, 201):
                    _log(f"   {key}[{i}]: OK ({cat.get('model_name')})")
                elif r.status_code in (401, 403):
                    err = _parse_error(r) or "AUTH-DENIED"
                    _log(f"   {key}[{i}]: AUTH-FEHLER {r.status_code} - {err}")
                elif r.status_code == 404:
                    err = _parse_error(r) or "Modell nicht gefunden"
                    _log(f"   {key}[{i}]: 404 - {err}")
                else:
                    err = _parse_error(r) or f"HTTP {r.status_code}"
                    _log(f"   {key}[{i}]: {err}")
            except Exception as exc:
                _log(f"   {key}[{i}]: nicht erreichbar - {type(exc).__name__}: {_safe_str(exc)}")

    _log("Health-Checks abgeschlossen")


@app.on_event("startup")
async def _startup_event() -> None:
    _log(f"LocalProxy v3.0 starting on port {PROXY_PORT}")
    _log(f"   Default:  {DEFAULT_CATEGORY}")
    for key in ("local", "light", "strong", "vision"):
        defs = _model_defs(key)
        if defs:
            for i, d in enumerate(defs):
                _log(f"   {key}[{i}]: {d.get('model_name','?')} @ {d.get('api_url','?')}")
    _log(f"   Memory:  {'Qdrant' if _hindsight._use_qdrant else 'JSONL' if HINDSIGHT_ENABLED else 'disabled'}")
    _log(f"   Auth:    {'enabled' if PROXY_AUTH_ENABLED else 'disabled'}")
    _log(f"   Debug:   {'enabled' if DEBUG_ENABLED else 'disabled'}")
    _log(f"   Tool-Cap: {TOOL_RESULT_CAP if TOOL_RESULT_CAP > 0 else 'off'}")

    # Health-Checks nicht-blockierend im Hintergrund starten
    asyncio.ensure_future(_run_startup_health_checks())


@app.on_event("shutdown")
async def _shutdown_event() -> None:
    _log("LocalProxy shutting down.")


# ── Auth ───────────────────────────────────────────────────────────────────
def _get_bearer_token(request: Request) -> str:
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    return ""


def _is_authorized(request: Request) -> bool:
    if not PROXY_AUTH_ENABLED:
        return True
    token = _get_bearer_token(request)
    return bool(token and secrets.compare_digest(token, PROXY_API_KEY))


async def _auth_or_raise(request: Request) -> None:
    if not PROXY_AUTH_ENABLED:
        _log(f"AUTH-SKIP (auth disabled) path={request.url.path}")
        return
    token = _get_bearer_token(request)
    if not token:
        _log(f"AUTH-FAIL kein Token path={request.url.path}")
        raise HTTPException(status_code=401, detail="Unauthorized — kein Bearer-Token",
                           headers={"WWW-Authenticate": "Bearer"})
    if not secrets.compare_digest(token, PROXY_API_KEY):
        _log(f"AUTH-FAIL falscher Token path={request.url.path} "
             f"got={_truncate_key(token)} expected={_truncate_key(PROXY_API_KEY)}")
        raise HTTPException(status_code=401, detail="Unauthorized — falscher Proxy-API-Key",
                           headers={"WWW-Authenticate": "Bearer"})
    _log(f"AUTH-OK path={request.url.path}")


# ── Shared Chat-Completion Handler ─────────────────────────────────────────
async def _handle_chat_completion(body: Dict[str, Any]) -> JSONResponse | StreamingResponse:
    """Gemeinsame Logik fuer /v1/chat/completions und /chat/completions."""
    if "messages" not in body:
        raise HTTPException(status_code=400, detail="Invalid payload: 'messages' required.")

    msgs = body.get("messages", [])
    _log(f"Request: {len(msgs)} messages, stream={body.get('stream')}, "
         f"tool_cont={_is_tool_continuation(msgs)}")

    last_user = _last_user_text(msgs)

    # --reset Flag abfangen
    if _detect_reset_flag(last_user):
        _do_reset()
        return JSONResponse(content={
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "system",
            "choices": [{"index": 0, "message": {"role": "assistant",
                          "content": "[Proxy Reset] Alle Kategorien auf Primary (Idx=0) zurückgesetzt."},
                          "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    cleaned, flag_category, flag_slot = _extract_model_flag(last_user)
    category = flag_category if flag_category else DEFAULT_CATEGORY

    # Slot-Nummer in 0-basierten Index umrechnen (--light 2 → Idx 1)
    force_start_idx: Optional[int] = None
    if flag_slot is not None and category != "local":
        force_start_idx = flag_slot - 1  # Slot 1=Idx 0, Slot 2=Idx 1, Slot 3=Idx 2
        defs_validate = _model_defs(category)
        if force_start_idx >= len(defs_validate) or force_start_idx < 0:
            _log(f"Slot {flag_slot} ungültig für {category} (nur {len(defs_validate)} Slots), verwende active_idx")
            force_start_idx = None

    _strip_model_flags_from_messages(msgs)

    # Log aktiven Index für die Kategorie
    active_idx = force_start_idx if force_start_idx is not None else _CATEGORY_ACTIVE_IDX.get(category, 0)
    defs = _model_defs(category)
    active_model = defs[active_idx]["model_name"] if defs and active_idx < len(defs) else "?"
    _log(f"Kategorie: {category} (Flag={'--'+category if flag_category else 'default'}"
         f"{' Slot='+str(flag_slot) if flag_slot else ''}), "
         f"Idx={active_idx}, Modell={active_model}")

    if body.get("stream"):
        return StreamingResponse(
            _stream_events(body, category, force_start_idx),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                     "X-Accel-Buffering": "no"},
        )

    # Non-Streaming: Fallback-Chain nutzen
    outcome = await _call_model_with_fallbacks(body, category, force_start_idx=force_start_idx)
    result = outcome.get("result", {})
    content = result.get("content", "") or ""

    # Bei totalem Fehlschlag: Fehler-Text mit Prefx
    if outcome.get("all_failed"):
        content = f"[Proxy: ALLE Fallbacks fehlgeschlagen]\n{content}"

    response_payload = _build_response_payload(body, content, [result])
    asyncio.ensure_future(_hindsight.retain_async(body, content))
    return JSONResponse(content=response_payload)


# ── /v1/chat/completions ───────────────────────────────────────────────────
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    await _auth_or_raise(request)
    body = await request.json()
    return await _handle_chat_completion(body)


# ── /chat/completions (ohne /v1/ Prefix, Catch-all) ────────────────────────
@app.post("/chat/completions")
async def chat_completions_no_v1(request: Request):
    _log("Route /chat/completions (ohne /v1/)")
    await _auth_or_raise(request)
    body = await request.json()
    return await _handle_chat_completion(body)


async def _stream_events(body: Dict[str, Any], category: str,
                          force_start_idx: Optional[int] = None) -> AsyncIterator[str]:
    """2-Pass-Streaming: Erst Backend-Call mit Fallback-Loop, dann OpenAI-SSE an Copilot."""
    outcome = await _call_model_with_fallbacks(body, category, force_start_idx=force_start_idx)
    result = outcome.get("result", {})
    content = result.get("content", "") or ""
    used_model = outcome.get("used_model", category)

    asyncio.ensure_future(_hindsight.retain_async(body, content))

    tool_calls = result.get("tool_calls")
    reasoning_content = result.get("reasoning_content")

    if outcome.get("all_failed"):
        content = f"[Proxy: ALLE Fallbacks fehlgeschlagen]\n{content}"

    if tool_calls:
        tool_calls = _normalize_tool_calls(tool_calls) or tool_calls
        stream_id = f"chatcmpl-spark-{uuid.uuid4().hex}"
        first_tcs = []
        for i, tc in enumerate(tool_calls):
            func = tc.get("function", {})
            args = func.get("arguments", "")
            if not isinstance(args, str):
                try:
                    args = json.dumps(args, ensure_ascii=False)
                except Exception:
                    args = str(args)
            first_tcs.append({
                "index": i,
                "id": tc.get("id", f"call_{uuid.uuid4().hex}"),
                "type": "function",
                "function": {"name": func.get("name", ""), "arguments": args},
            })
        yield _format_openai_stream_chunk(
            used_model, include_role=True, tool_calls=first_tcs,
            reasoning_content=reasoning_content, chunk_id=stream_id,
        )
        yield _format_openai_stream_chunk(used_model, finish_reason="tool_calls", chunk_id=stream_id)
    else:
        yield _format_openai_stream_chunk(
            used_model, content=content, include_role=True,
            reasoning_content=reasoning_content,
        )
        yield _format_openai_stream_chunk(used_model, "", finish_reason="stop")


# ── /v1/models ─────────────────────────────────────────────────────────────
@app.get("/v1/models")
async def list_models(request: Request):
    logs_str = request.query_params.get("logs", "")
    if logs_str and logs_str.isdigit() and int(logs_str) > 0:
        return JSONResponse(content=await _get_logs_handler(lines=int(logs_str)))
    await _auth_or_raise(request)
    models = []
    for key in ("local", "light", "strong", "vision"):
        defs = _model_defs(key)
        for i, d in enumerate(defs):
            models.append({
                "id": d.get("model_name", "?"),
                "object": "model",
                "owned_by": f"category:{key}[{i}]",
            })
    return JSONResponse(content={"object": "list", "data": models})


# ── /healthz ───────────────────────────────────────────────────────────────
@app.get("/healthz")
async def healthz(request: Request):
    return JSONResponse(content={
        "status": "ok",
        "version": "3.0.0",
        "default_category": DEFAULT_CATEGORY,
        "categories": {
            key: [{
                "model_name": d.get("model_name", "?"),
                "is_vision": d.get("is_vision", False),
                "api_url": d.get("api_url", ""),
                "active": (i == _CATEGORY_ACTIVE_IDX.get(key, 0)
                           if key in ("light", "strong", "vision") else i == 0),
            } for i, d in enumerate(defs)]
            for key, defs in {
                "local": _model_defs("local"),
                "light": _model_defs("light"),
                "strong": _model_defs("strong"),
                "vision": _model_defs("vision"),
            }.items() if defs
        },
        "proxy_auth_enabled": PROXY_AUTH_ENABLED,
        "hindsight_enabled": HINDSIGHT_ENABLED,
        "hindsight_backend": "qdrant" if _hindsight._use_qdrant else "jsonl",
        "debug_enabled": DEBUG_ENABLED,
        "tool_result_cap": TOOL_RESULT_CAP,
    })


# ── /logs & /v1/logs ────────────────────────────────────────────────────────
async def _get_logs_handler(lines: int = 200):
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
    except (FileNotFoundError, OSError):
        all_lines = []
    last = all_lines[-lines:] if lines > 0 else all_lines
    return {"count": len(last), "total": len(all_lines), "file": LOG_FILE, "lines": last}


@app.get("/logs")
async def get_logs(request: Request, lines: int = 200):
    return JSONResponse(content=await _get_logs_handler(lines))


@app.get("/v1/logs")
async def get_v1_logs(request: Request, lines: int = 200):
    return JSONResponse(content=await _get_logs_handler(lines))


# ── /debug/* ───────────────────────────────────────────────────────────────
@app.get("/debug/files")
async def debug_files(request: Request):
    try:
        files = []
        if DEBUG_DIR.exists():
            for entry in sorted(DEBUG_DIR.glob("*.json"),
                                key=lambda p: p.stat().st_mtime, reverse=True):
                stat = entry.stat()
                files.append({
                    "name": entry.name,
                    "size": stat.st_size,
                    "modified": time.strftime("%Y-%m-%d %H:%M:%S",
                                             time.localtime(stat.st_mtime)),
                })
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    return JSONResponse(content={"count": len(files), "debug_dir": str(DEBUG_DIR),
                                  "files": files})


@app.get("/debug/file/{file_id}")
async def debug_file(file_id: str, request: Request):
    if "/" in file_id or "\\" in file_id or ".." in file_id:
        return JSONResponse(status_code=400, content={"error": "invalid file_id"})
    path = DEBUG_DIR / file_id
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "file not found"})
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    return JSONResponse(content={"name": file_id, "content": content})


@app.get("/debug/ring")
async def debug_ring(request: Request, limit: int = 50):
    items = list(_DEBUG_RING)
    if limit > 0:
        items = items[-limit:]
    return JSONResponse(content={"count": len(items), "max": _DEBUG_RING_MAX,
                                  "entries": items})


@app.get("/debug/active")
async def debug_active(request: Request):
    active = []
    for call_id, info in _ACTIVE_CALLS.items():
        elapsed = time.time() - info.get("started_at", time.time())
        active.append({
            "call_id": call_id,
            "elapsed_seconds": round(elapsed, 1),
            **{k: v for k, v in info.items() if k not in ("started_at",)},
        })
    return JSONResponse(content={"count": len(active), "active": active})


@app.post("/debug/cleanup")
async def debug_cleanup(request: Request):
    _cleanup_old_debug_files()
    count = 0
    if DEBUG_DIR.exists():
        count = len(list(DEBUG_DIR.glob("*.json")))
    return JSONResponse(content={"status": "ok", "remaining_files": count,
                                  "max": DEBUG_MAX_FILES})


# ── Webinterface mounten ───────────────────────────────────────────────────
if _WEBUI_AVAILABLE:
    mount_webui(app)
    _log("Web-Konfigurationsinterface: http://0.0.0.0:" + str(PROXY_PORT) + "/webui/")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    _log("""
╔══════════════════════════════════════════════════════════════╗
║        LocalProxy v3.0  —  Single-Model Pass-Through         ║
║  OpenAI-kompatibel · Hindsight Memory · 4 Kategorien        ║
╚══════════════════════════════════════════════════════════════╝
    """.strip())
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT, log_config=None)
