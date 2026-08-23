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
        "use_max_completion_tokens": os.getenv("LOCAL_USE_MAX_COMPLETION_TOKENS", "false").lower() in {"1", "true", "yes", "y", "on"},
        "is_vision": os.getenv("LOCAL_IS_VISION", "false").lower() in {"1", "true", "yes", "y", "on"},
        "timeout_seconds": float(os.getenv("LOCAL_TIMEOUT_SECONDS", "300")),
        "read_timeout_seconds": float(os.getenv("LOCAL_READ_TIMEOUT_SECONDS", "120")),
        "retry_on_timeout": int(os.getenv("LOCAL_RETRY_ON_TIMEOUT", "2")),
        "retry_delay_seconds": float(os.getenv("LOCAL_RETRY_DELAY_SECONDS", "5")),
        "label": "local primary",
    },
    "light": [
        {
            "label": "light primary",
            "api_url": os.getenv("LIGHT_API_URL", "https://api.openai.com/v1/chat/completions"),
            "api_key": os.getenv("LIGHT_API_KEY", ""),
            "model_name": os.getenv("LIGHT_MODEL_NAME", "gpt-4.1-mini"),
            "max_tokens": int(os.getenv("LIGHT_MAX_TOKENS", "65536")),
            "use_max_completion_tokens": os.getenv("LIGHT_USE_MAX_COMPLETION_TOKENS", "false").lower() in {"1", "true", "yes", "y", "on"},
            "is_vision": os.getenv("LIGHT_IS_VISION", "false").lower() in {"1", "true", "yes", "y", "on"},
            "timeout_seconds": float(os.getenv("LIGHT_TIMEOUT_SECONDS", "180")),
        },
        {"label": "light fallback 2", "api_url": "", "api_key": "", "model_name": "", "max_tokens": 65536, "use_max_completion_tokens": False, "is_vision": False, "timeout_seconds": 180},
        {"label": "light fallback 3", "api_url": "", "api_key": "", "model_name": "", "max_tokens": 65536, "use_max_completion_tokens": False, "is_vision": False, "timeout_seconds": 180},
    ],
    "strong": [
        {
            "label": "strong primary",
            "api_url": os.getenv("STRONG_API_URL", "https://api.anthropic.com/v1/chat/completions"),
            "api_key": os.getenv("STRONG_API_KEY", ""),
            "model_name": os.getenv("STRONG_MODEL_NAME", "claude-sonnet-4-20250514"),
            "max_tokens": int(os.getenv("STRONG_MAX_TOKENS", "65536")),
            "use_max_completion_tokens": os.getenv("STRONG_USE_MAX_COMPLETION_TOKENS", "false").lower() in {"1", "true", "yes", "y", "on"},
            "is_vision": os.getenv("STRONG_IS_VISION", "false").lower() in {"1", "true", "yes", "y", "on"},
            "timeout_seconds": float(os.getenv("STRONG_TIMEOUT_SECONDS", "300")),
        },
        {"label": "strong fallback 2", "api_url": "", "api_key": "", "model_name": "", "max_tokens": 65536, "use_max_completion_tokens": False, "is_vision": False, "timeout_seconds": 300},
        {"label": "strong fallback 3", "api_url": "", "api_key": "", "model_name": "", "max_tokens": 65536, "use_max_completion_tokens": False, "is_vision": False, "timeout_seconds": 300},
    ],
    "vision": [
        {
            "label": "vision primary",
            "api_url": os.getenv("VISION_API_URL", "https://api.openai.com/v1/chat/completions"),
            "api_key": os.getenv("VISION_API_KEY", ""),
            "model_name": os.getenv("VISION_MODEL_NAME", "gpt-4o"),
            "max_tokens": int(os.getenv("VISION_MAX_TOKENS", "65536")),
            "use_max_completion_tokens": os.getenv("VISION_USE_MAX_COMPLETION_TOKENS", "false").lower() in {"1", "true", "yes", "y", "on"},
            "is_vision": os.getenv("VISION_IS_VISION", "true").lower() in {"1", "true", "yes", "y", "on"},
            "timeout_seconds": float(os.getenv("VISION_TIMEOUT_SECONDS", "180")),
        },
        {"label": "vision fallback 2", "api_url": "", "api_key": "", "model_name": "", "max_tokens": 65536, "use_max_completion_tokens": False, "is_vision": True, "timeout_seconds": 180},
        {"label": "vision fallback 3", "api_url": "", "api_key": "", "model_name": "", "max_tokens": 65536, "use_max_completion_tokens": False, "is_vision": True, "timeout_seconds": 180},
    ],
    # coworker: zweites lokales Modell auf separater Hardware (Co-Worker).
    # Single-Def wie "local" (keine Fallback-Slots). Wird NUR bei aktiver
    # Kategorie "local" als ask_coworker-Tool injiziert — und nur wenn der
    # Health-Check das Modell als erreichbar meldet.
    "coworker": {
        "api_url": os.getenv("COWORKER_API_URL", ""),
        "api_key": os.getenv("COWORKER_API_KEY", ""),
        "model_name": os.getenv("COWORKER_MODEL_NAME", ""),
        "max_tokens": int(os.getenv("COWORKER_MAX_TOKENS", "65536")),
        "use_max_completion_tokens": os.getenv("COWORKER_USE_MAX_COMPLETION_TOKENS", "false").lower() in {"1", "true", "yes", "y", "on"},
        "is_vision": os.getenv("COWORKER_IS_VISION", "false").lower() in {"1", "true", "yes", "y", "on"},
        "timeout_seconds": float(os.getenv("COWORKER_TIMEOUT_SECONDS", "300")),
        "read_timeout_seconds": float(os.getenv("COWORKER_READ_TIMEOUT_SECONDS", "120")),
        "retry_on_timeout": int(os.getenv("COWORKER_RETRY_ON_TIMEOUT", "2")),
        "retry_delay_seconds": float(os.getenv("COWORKER_RETRY_DELAY_SECONDS", "5")),
        "label": "coworker primary",
    },
}

DEFAULT_CATEGORY: str = os.getenv("DEFAULT_CATEGORY", "light")

# ── Fallback-System ─────────────────────────────────────────────────────────
# Aktueller aktiver Index pro Kategorie (light/strong/vision)
_CATEGORY_ACTIVE_IDX: Dict[str, int] = {"local": 0, "light": 0, "strong": 0, "vision": 0, "coworker": 0}

COOLDOWN_FILE: Path = Path(os.getenv("COOLDOWN_FILE", str(Path(__file__).parent / "data" / "cooldowns.json")))
COOLDOWN_DEFAULT_SECONDS: float = float(os.getenv("COOLDOWN_DEFAULT_SECONDS", "300"))


def _model_defs(category: str) -> List[Dict[str, Any]]:
    """Gibt Liste von Modell-Definitionen fuer eine Kategorie zurueck."""
    cat = _MODEL_CATEGORIES.get(category)
    if isinstance(cat, list):
        return [d for d in cat if isinstance(d, dict) and d.get("api_url") and d.get("model_name")]
    if isinstance(cat, dict) and cat.get("api_url"):
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


# ═══════════════════════════════════════════════════════════════════════════
# UTF-8 in stdout/stderr für Docker (kein Locale in slim-Images)
# ═══════════════════════════════════════════════════════════════════════════
# Diese Bloecke laufen VOR der Konfiguration und VOR jedem print().
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
try:
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── Alle initialen Model-Defs gegen non-ASCII in api_key/model_name haerten ─
# Nur _api_headers() braucht diesen Schutz (HTTP-Header muessen ASCII sein).
# config.json und _MODEL_CATEGORIES bleiben unberuehrt — Keys duerfen nie
# veraendert werden, sonst werden sie invalide.
# ═══════════════════════════════════════════════════════════════════════════

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

# ── Laguna-S-2.1 Sampling & Anti-Loop ─────────────────────────────────────
# Empfohlene Parameter aus poolside/Laguna-S-2.1 Diskussion #23:
#   temp=0.7, top_p=0.95, top_k=20, min_p=0.0
#   DRY: multiplier=0.8, base=1.75, allowed_length=3, penalty_last_n=-1
#   sequence_breaker: \n,:,",*,;,{,}
#   samplers: top_k;top_p;min_p;temperature;dry
LOCAL_TEMPERATURE: float = float(os.getenv("LOCAL_TEMPERATURE", "0.7"))
LOCAL_TOP_P: float = float(os.getenv("LOCAL_TOP_P", "0.95"))
LOCAL_TOP_K: int = int(os.getenv("LOCAL_TOP_K", "20"))
LOCAL_MIN_P: float = float(os.getenv("LOCAL_MIN_P", "0.0"))
LOCAL_DRY_MULTIPLIER: float = float(os.getenv("LOCAL_DRY_MULTIPLIER", "0.8"))
LOCAL_DRY_BASE: float = float(os.getenv("LOCAL_DRY_BASE", "1.75"))
LOCAL_DRY_ALLOWED_LENGTH: int = int(os.getenv("LOCAL_DRY_ALLOWED_LENGTH", "3"))
LOCAL_DRY_PENALTY_LAST_N: int = int(os.getenv("LOCAL_DRY_PENALTY_LAST_N", "-1"))
LOCAL_DRY_SEQUENCE_BREAKER: str = os.getenv("LOCAL_DRY_SEQUENCE_BREAKER", '\n,:,",*,;,{,}')
LOCAL_ENABLE_THINKING: bool = os.getenv("LOCAL_ENABLE_THINKING", "true").lower() in {"1", "true", "yes", "y", "on"}
LOCAL_PRESERVE_THINKING: bool = os.getenv("LOCAL_PRESERVE_THINKING", "true").lower() in {"1", "true", "yes", "y", "on"}
# Anti-Loop-System-Prompt: wird als zusaetzliche System-Message injiziert
LOCAL_ANTI_LOOP_SYSTEM_PROMPT: str = os.getenv(
    "LOCAL_ANTI_LOOP_SYSTEM_PROMPT",
    "CRITICAL RULES:\n"
    "1. NEVER call the same tool with the same arguments more than once.\n"
    "2. After reading a file or getting search results, IMMEDIATELY use the "
    "information to take the next action (edit, create, run, or respond).\n"
    "3. Do NOT re-read files you already read. Do NOT re-search patterns you "
    "already searched.\n"
    "4. If a tool call fails or returns no results, try a DIFFERENT approach — "
    "do not retry the same call.\n"
    "5. When updating a todo/task list, do it ONCE then move on to actual work.\n"
    "6. Always produce reasoning in <think> tags before taking action."
)

# ── Read-Loop-Detection (NUR Laguna-S-2.1) ─────────────────────────────────
# Erkennt wenn ein Modell dieselbe Datei mit denselben Zeilen >N mal liest
# und injiziert eine Interventions-Message.
READ_LOOP_THRESHOLD: int = int(os.getenv("READ_LOOP_THRESHOLD", "3"))
# File-Crawl-Detection: gleiche Datei >N mal in den letzten M Reads (auch verschiedene Zeilen)
READ_LOOP_FILE_THRESHOLD: int = int(os.getenv("READ_LOOP_FILE_THRESHOLD", "8"))
READ_LOOP_FILE_WINDOW: int = int(os.getenv("READ_LOOP_FILE_WINDOW", "12"))
# Bei File-Crawl: wie viele Reads der Datei behalten (Rest + Folgeloop wird trunkiert)
READ_LOOP_FILE_KEEP: int = int(os.getenv("READ_LOOP_FILE_KEEP", "1"))
READ_LOOP_INTERVENTION: str = os.getenv(
    "READ_LOOP_INTERVENTION",
    "STOP! You are looping. You have already read this file/range {count} times. "
    "You have all the information you need. STOP READING NOW and proceed with the "
    "next step of your task (edit, build, run, or respond). Do NOT issue another "
    "read_file call for this file. Go on."
)
READ_LOOP_FILE_INTERVENTION: str = os.getenv(
    "READ_LOOP_FILE_INTERVENTION",
    "STOP! You are crawling file '{file}' — {count} reads in the last {window} tool calls. "
    "You already have enough content from this file. Do NOT read '{file}' again. "
    "Do NOT re-search symbols you already found. "
    "NOW: Summarize in one sentence what you know, then take the next concrete "
    "action for the user's original request — use replace_string_in_file or "
    "create_file to implement the change, or ask a specific question. Go on."
)

# ── Generic Tool-Loop-Detection (NUR Laguna-S-2.1) ─────────────────────────
# Erkennt wenn ein beliebiges Tool (z.B. manage_todo_list, create_file, etc.)
# mit identischen Argumenten >N mal hintereinander aufgerufen wird.
# Default 0 = deaktiviert (opt-in), damit legitime Wiederholungen nicht blockiert werden.
GENERIC_TOOL_LOOP_THRESHOLD: int = int(os.getenv("GENERIC_TOOL_LOOP_THRESHOLD", "3"))
GENERIC_TOOL_LOOP_INTERVENTION: str = os.getenv(
    "GENERIC_TOOL_LOOP_INTERVENTION",
    "STOP! You are looping. You have already called the tool '{tool}' with the "
    "exact same arguments {count} times. The task/update is already done. "
    "Do NOT call '{tool}' again with these arguments. "
    "NOW: proceed to the next step of your task — implement the change with "
    "replace_string_in_file/create_file, run a test, or respond to the user. Go on."
)

# ── Search-Loop-Detection (NUR Laguna-S-2.1) ────────────────────────────────
# No-Match-Loops: gleiche Suche >N mal ohne Treffer.
SEARCH_LOOP_THRESHOLD: int = int(os.getenv("SEARCH_LOOP_THRESHOLD", "3"))
# Repeat-Loops: gleiche Suche >N mal — AUCH mit Treffern (Spam/Exploration-Loop).
SEARCH_REPEAT_THRESHOLD: int = int(os.getenv("SEARCH_REPEAT_THRESHOLD", "3"))
SEARCH_LOOP_INTERVENTION: str = os.getenv(
    "SEARCH_LOOP_INTERVENTION",
    "STOP! You are looping on a search that returns no results. You have already "
    "searched for '{query}' {count} times with NO MATCHES. The pattern does not "
    "exist in this workspace or is excluded by ignore/exclude settings. "
    "STOP SEARCHING for this pattern. Change your approach: try a different "
    "search term, read the file directly, use a different tool, or proceed with "
    "the information you already have. Do NOT repeat this search. Go on."
)
SEARCH_REPEAT_INTERVENTION: str = os.getenv(
    "SEARCH_REPEAT_INTERVENTION",
    "STOP! You already searched for '{query}' {count} times and got results. "
    "Repeating the same search will not help. Use the results you already have. "
    "Do NOT search for '{query}' again. "
    "NOW: take the next concrete action — edit the file with replace_string_in_file, "
    "create the missing code with create_file, or answer the user. Go on."
)

# Response-Level Enforcement: blockiert Tool-Calls die trotz Historie-Intervention
# erneut denselben Loop fortsetzen. Default aktiv (3) fuer Laguna-S-2.1.
# 0 = deaktiviert.
_RESPONSE_LOOP_THRESHOLD: int = int(os.getenv("RESPONSE_LOOP_THRESHOLD", "3"))
# Hinweis: Bei geblocktem Response-Loop liefert der Proxy KEINE passive
# Abschluss-Meldung mehr. Stattdessen wird eine handlungsorientierte
# Interventions-Nachricht direkt in die Request-Messages eingefuegt und das
# Modell erneut aufgerufen. Dadurch sieht VS Code niemals einen toten
# "I've stopped ..." Text, sondern das Modell arbeitet am urspruenglichen Task.
RESPONSE_LOOP_REDIRECT_TEXT: str = os.getenv(
    "RESPONSE_LOOP_REDIRECT_TEXT",
    "SYSTEM-INTERVENTION: Dein letzter Tool-Call wurde vom Proxy geblockt, weil "
    "er einen Loop fortgesetzt haette ({reasons}). Du hast diese Informationen "
    "bereits. Fuehre jetzt den naechsten sinnvollen Schritt fuer die eigentliche "
    "Aufgabe aus: implementiere die Aenderung (replace_string_in_file/create_file), "
    "oder formuliere eine konkrete Antwort. Wiederhole nicht die geblockten "
    "Read/Search-Calls. Go on."
)


# ── Co-Worker-Delegation (ask_coworker) ────────────────────────────────────
# Zweites lokales Modell auf separater Hardware. Wird NUR bei aktiver
# Kategorie "local" als ask_coworker-Tool in den Payload injiziert — und nur
# wenn der Health-Check das Modell als erreichbar meldet.
COWORKER_ENABLED: bool = os.getenv("COWORKER_ENABLED", "true").lower() in {"1", "true", "yes", "y", "on"}
COWORKER_MAX_DELEGATIONS: int = int(os.getenv("COWORKER_MAX_DELEGATIONS", "2"))
COWORKER_TASK_CAP: int = int(os.getenv("COWORKER_TASK_CAP", "8000"))
COWORKER_RESULT_CAP: int = int(os.getenv("COWORKER_RESULT_CAP", "12000"))
# Automatisch angehaengter Datei-Kontext (VS-Code-Attachments + Tool-Ergebnisse)
# fuer ask_coworker-Calls: Budget in Zeichen, 0 = deaktiviert.
COWORKER_FILES_CAP: int = int(os.getenv("COWORKER_FILES_CAP", "60000"))
# Health-Check: wie oft der Co-Worker geprobt wird / wie lange ein Probe-Timeout dauern darf
COWORKER_HEALTH_INTERVAL: float = float(os.getenv("COWORKER_HEALTH_INTERVAL", "60"))
COWORKER_PROBE_TIMEOUT: float = float(os.getenv("COWORKER_PROBE_TIMEOUT", "5"))
# Fork-Join Fabric (v3.2): dispatch_coworker/collect_coworker Hintergrund-Tasks.
# Der Co-Worker (gleiche Modell-Familie, separate Hardware) rechnet parallel
# im Hintergrund weiter, waehrend das Hauptmodell eigene Arbeit erledigt.
COWORKER_FORK_JOIN: bool = os.getenv("COWORKER_FORK_JOIN", "true").lower() in {"1", "true", "yes", "y", "on"}
# Wie viele Hintergrund-Tasks GLEICHZEITIG auf dem Co-Worker laufen duerfen
COWORKER_MAX_PARALLEL: int = int(os.getenv("COWORKER_MAX_PARALLEL", "8"))
# Obergrenze dispatch_coworker-Calls pro Request (Schutz gegen Dispatch-Loops)
COWORKER_DISPATCH_CAP: int = int(os.getenv("COWORKER_DISPATCH_CAP", "12"))
# TTL der Hintergrund-Tasks in Sekunden (0 = unbegrenzt); laufende Tasks werden
# bei Ablauf abgebrochen (Status=expired), abgelieferte nach 60s entfernt.
COWORKER_BG_TTL: float = float(os.getenv("COWORKER_BG_TTL", "1800"))
# Bootstrap-Anleitung: lehrt das Hauptmodell als system-Message, WANN und WIE
# es an den Co-Worker delegiert. Ohne diesen Hinweis delegiert das Modell in
# der Praxis nie — Tool-Beschreibungen allein aendern das Verhalten nicht.
COWORKER_TEACH_DELEGATION: bool = os.getenv("COWORKER_TEACH_DELEGATION", "true").lower() in {"1", "true", "yes", "y", "on"}
COWORKER_SYSTEM_PROMPT: str = os.getenv(
    "COWORKER_SYSTEM_PROMPT",
    "You are a co-worker coding model acting as a subagent for planning, code "
    "review, brainstorming and parallel implementation tasks. You receive a "
    "self-contained task and optional context. Answer with concrete, actionable "
    "output: for planning give a step-by-step plan; for review list issues with "
    "file/line references and concrete fixes; for coding give complete, idiomatic "
    "code. You have NO access to tools, the conversation history or the workspace "
    "— work only from what is given to you.",
)


# ── Laguna-S-2.1 Modell-Erkennung ─────────────────────────────────────────
# Loop-Schutz (Read/Search/Generic/Response) und Sampling-Patches gelten
# ausschliesslich fuer Laguna-S-2.1 Modelle — egal ob local oder cloud,
# egal welche Quantisierung (NVFP4, GGUF, etc.).
_LAGUNA_MODEL_RE = re.compile(r"laguna", re.IGNORECASE)


def _is_laguna_model(model_name: str) -> bool:
    """True wenn der Modellname ein Laguna-S-2.1 Modell ist (case-insensitive)."""
    return bool(model_name and _LAGUNA_MODEL_RE.search(model_name))


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
        for key in ("local", "coworker", "light", "strong", "vision"):
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
                                   "use_max_completion_tokens", "is_vision", "timeout_seconds",
                                   "read_timeout_seconds", "retry_on_timeout", "retry_delay_seconds"):
                        if field in sc:
                            val = sc[field]
                            if field in ("api_url", "api_key") and isinstance(val, str) and val.strip() == "":
                                continue
                            if field == "max_tokens":
                                val = int(val)
                            elif field == "use_max_completion_tokens":
                                val = bool(val) if not isinstance(val, str) else \
                                    str(val).lower() in {"1", "true", "yes", "y", "on"}
                            elif field == "is_vision":
                                val = bool(val) if not isinstance(val, str) else \
                                    str(val).lower() in {"1", "true", "yes", "y", "on"}
                            elif field in ("timeout_seconds", "read_timeout_seconds", "retry_delay_seconds"):
                                val = float(val)
                            elif field == "retry_on_timeout":
                                val = int(val)
                            merged[field] = val
                    merged.setdefault("label", key)
                    _MODEL_CATEGORIES[key] = [merged]
                else:
                    cat = _MODEL_CATEGORIES.setdefault(key, {})
                    for field in ("label", "api_url", "api_key", "model_name", "max_tokens",
                                   "use_max_completion_tokens", "is_vision", "timeout_seconds",
                                   "read_timeout_seconds", "retry_on_timeout", "retry_delay_seconds"):
                        if field in sc:
                            val = sc[field]
                            if field in ("api_url", "api_key") and isinstance(val, str) and val.strip() == "":
                                continue
                            if field == "max_tokens":
                                val = int(val)
                            elif field == "use_max_completion_tokens":
                                val = bool(val) if not isinstance(val, str) else \
                                    str(val).lower() in {"1", "true", "yes", "y", "on"}
                            elif field == "is_vision":
                                val = bool(val) if not isinstance(val, str) else \
                                    str(val).lower() in {"1", "true", "yes", "y", "on"}
                            elif field in ("timeout_seconds", "read_timeout_seconds", "retry_delay_seconds"):
                                val = float(val)
                            elif field == "retry_on_timeout":
                                val = int(val)
                            cat[field] = val

            elif isinstance(sc, list):
                # Neue Array-Struktur (light/strong/vision)
                cleaned_list: List[Dict[str, Any]] = []
                for d in sc:
                    if not isinstance(d, dict):
                        continue
                    element: Dict[str, Any] = {}
                    for field in ("label", "api_url", "api_key", "model_name", "max_tokens",
                                   "use_max_completion_tokens", "is_vision", "timeout_seconds"):
                        if field in d:
                            val = d[field]
                            if field in ("api_url", "api_key") and isinstance(val, str) and val.strip() == "":
                                continue
                            if field == "max_tokens":
                                val = int(val)
                            elif field == "use_max_completion_tokens":
                                val = (bool(val) if not isinstance(val, str) else
                                       str(val).lower() in {"1", "true", "yes", "y", "on"})
                            elif field == "is_vision":
                                val = (bool(val) if not isinstance(val, str) else
                                       str(val).lower() in {"1", "true", "yes", "y", "on"})
                            elif field == "timeout_seconds":
                                val = float(val)
                            element[field] = val
                    cleaned_list.append(element)
                _MODEL_CATEGORIES[key] = cleaned_list

    # Active-Indices nach Config-Update validieren
    for key in ("light", "strong", "vision"):
        defs = _model_defs(key)
        if not defs:
            _CATEGORY_ACTIVE_IDX[key] = 0
        elif _CATEGORY_ACTIVE_IDX[key] >= len(defs):
            _CATEGORY_ACTIVE_IDX[key] = 0

    _log("Config aus config.json neu geladen")

    dc = cfg.get("default_category", "")
    if dc in ("local", "coworker", "light", "strong", "vision"):
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

    global READ_LOOP_THRESHOLD, READ_LOOP_INTERVENTION
    global READ_LOOP_FILE_THRESHOLD, READ_LOOP_FILE_WINDOW, READ_LOOP_FILE_KEEP
    global READ_LOOP_FILE_INTERVENTION
    tokens_cfg = cfg.get("tokens", {})
    READ_LOOP_THRESHOLD = int(tokens_cfg.get("read_loop_threshold", READ_LOOP_THRESHOLD))
    READ_LOOP_FILE_THRESHOLD = int(tokens_cfg.get("read_loop_file_threshold", READ_LOOP_FILE_THRESHOLD))
    READ_LOOP_FILE_WINDOW = int(tokens_cfg.get("read_loop_file_window", READ_LOOP_FILE_WINDOW))
    READ_LOOP_FILE_KEEP = int(tokens_cfg.get("read_loop_file_keep", READ_LOOP_FILE_KEEP))
    rl_intervention = tokens_cfg.get("read_loop_intervention", "")
    if rl_intervention:
        READ_LOOP_INTERVENTION = str(rl_intervention)
    rl_file_intervention = tokens_cfg.get("read_loop_file_intervention", "")
    if rl_file_intervention:
        READ_LOOP_FILE_INTERVENTION = str(rl_file_intervention)

    global SEARCH_LOOP_THRESHOLD, SEARCH_LOOP_INTERVENTION
    global SEARCH_REPEAT_THRESHOLD, SEARCH_REPEAT_INTERVENTION
    global _RESPONSE_LOOP_THRESHOLD
    global GENERIC_TOOL_LOOP_THRESHOLD, GENERIC_TOOL_LOOP_INTERVENTION
    SEARCH_LOOP_THRESHOLD = int(tokens_cfg.get("search_loop_threshold", SEARCH_LOOP_THRESHOLD))
    SEARCH_REPEAT_THRESHOLD = int(tokens_cfg.get("search_repeat_threshold", SEARCH_REPEAT_THRESHOLD))
    sl_intervention = tokens_cfg.get("search_loop_intervention", "")
    if sl_intervention:
        SEARCH_LOOP_INTERVENTION = str(sl_intervention)
    sr_intervention = tokens_cfg.get("search_repeat_intervention", "")
    if sr_intervention:
        SEARCH_REPEAT_INTERVENTION = str(sr_intervention)
    _RESPONSE_LOOP_THRESHOLD = int(tokens_cfg.get(
        "response_loop_threshold", _RESPONSE_LOOP_THRESHOLD))
    GENERIC_TOOL_LOOP_THRESHOLD = int(tokens_cfg.get(
        "generic_tool_loop_threshold", GENERIC_TOOL_LOOP_THRESHOLD))
    gt_intervention = tokens_cfg.get("generic_tool_loop_intervention", "")
    if gt_intervention:
        GENERIC_TOOL_LOOP_INTERVENTION = str(gt_intervention)

    global COWORKER_ENABLED, COWORKER_MAX_DELEGATIONS
    global COWORKER_TASK_CAP, COWORKER_RESULT_CAP
    global COWORKER_FILES_CAP
    global COWORKER_HEALTH_INTERVAL, COWORKER_PROBE_TIMEOUT
    global COWORKER_FORK_JOIN, COWORKER_MAX_PARALLEL
    global COWORKER_DISPATCH_CAP, COWORKER_BG_TTL
    global COWORKER_SYSTEM_PROMPT, COWORKER_TEACH_DELEGATION
    cw = tokens_cfg.get("coworker", {})
    if isinstance(cw, dict) and cw:
        COWORKER_ENABLED = bool(cw.get("enabled", COWORKER_ENABLED))
        COWORKER_MAX_DELEGATIONS = int(cw.get("max_delegations_per_request", COWORKER_MAX_DELEGATIONS))
        COWORKER_TASK_CAP = int(cw.get("task_cap_chars", COWORKER_TASK_CAP))
        COWORKER_RESULT_CAP = int(cw.get("result_cap_chars", COWORKER_RESULT_CAP))
        COWORKER_FILES_CAP = int(cw.get("files_cap_chars", COWORKER_FILES_CAP))
        COWORKER_HEALTH_INTERVAL = float(cw.get("health_interval_seconds", COWORKER_HEALTH_INTERVAL))
        COWORKER_PROBE_TIMEOUT = float(cw.get("probe_timeout_seconds", COWORKER_PROBE_TIMEOUT))
        COWORKER_FORK_JOIN = bool(cw.get("enable_fork_join", COWORKER_FORK_JOIN))
        COWORKER_MAX_PARALLEL = int(cw.get("max_parallel", COWORKER_MAX_PARALLEL))
        COWORKER_DISPATCH_CAP = int(cw.get("dispatch_cap_per_request", COWORKER_DISPATCH_CAP))
        COWORKER_BG_TTL = float(cw.get("bg_ttl_seconds", COWORKER_BG_TTL))
        COWORKER_TEACH_DELEGATION = bool(cw.get("teach_delegation", COWORKER_TEACH_DELEGATION))
        cw_prompt = cw.get("system_prompt", "")
        if cw_prompt:
            COWORKER_SYSTEM_PROMPT = str(cw_prompt)

    global LOCAL_TEMPERATURE, LOCAL_TOP_P, LOCAL_TOP_K, LOCAL_MIN_P
    global LOCAL_DRY_MULTIPLIER, LOCAL_DRY_BASE, LOCAL_DRY_ALLOWED_LENGTH
    global LOCAL_DRY_PENALTY_LAST_N, LOCAL_DRY_SEQUENCE_BREAKER
    global LOCAL_ENABLE_THINKING, LOCAL_PRESERVE_THINKING
    global LOCAL_ANTI_LOOP_SYSTEM_PROMPT
    local_cfg = tokens_cfg.get("local_sampling", {})
    if isinstance(local_cfg, dict) and local_cfg:
        LOCAL_TEMPERATURE = float(local_cfg.get("temperature", LOCAL_TEMPERATURE))
        LOCAL_TOP_P = float(local_cfg.get("top_p", LOCAL_TOP_P))
        LOCAL_TOP_K = int(local_cfg.get("top_k", LOCAL_TOP_K))
        LOCAL_MIN_P = float(local_cfg.get("min_p", LOCAL_MIN_P))
        LOCAL_DRY_MULTIPLIER = float(local_cfg.get("dry_multiplier", LOCAL_DRY_MULTIPLIER))
        LOCAL_DRY_BASE = float(local_cfg.get("dry_base", LOCAL_DRY_BASE))
        LOCAL_DRY_ALLOWED_LENGTH = int(local_cfg.get("dry_allowed_length", LOCAL_DRY_ALLOWED_LENGTH))
        LOCAL_DRY_PENALTY_LAST_N = int(local_cfg.get("dry_penalty_last_n", LOCAL_DRY_PENALTY_LAST_N))
        LOCAL_DRY_SEQUENCE_BREAKER = str(local_cfg.get("dry_sequence_breaker", LOCAL_DRY_SEQUENCE_BREAKER))
        LOCAL_ENABLE_THINKING = bool(local_cfg.get("enable_thinking", LOCAL_ENABLE_THINKING))
        LOCAL_PRESERVE_THINKING = bool(local_cfg.get("preserve_thinking", LOCAL_PRESERVE_THINKING))
        al_prompt = local_cfg.get("anti_loop_system_prompt", "")
        if al_prompt:
            LOCAL_ANTI_LOOP_SYSTEM_PROMPT = str(al_prompt)

    _log("Config aus config.json neu geladen")


_apply_config_file()


# ═══════════════════════════════════════════════════════════════════════════
# Prompt-Flag-Extraktion ── Modell-Kategorie-Auswahl via --flag
# ═══════════════════════════════════════════════════════════════════════════

_MODEL_FLAG_PATTERN = re.compile(r'--(local|light|strong|vision|coworker)(?:\s+(\d+))?\s*$', re.IGNORECASE | re.MULTILINE)
_VALID_CATEGORIES: Set[str] = {"local", "light", "strong", "vision", "coworker"}
_FLAG_PROXIMITY_MAX_CHARS = 300  # Flag muss in den letzten 300 Zeichen des Texts stehen


def _extract_model_flag(text: str) -> Tuple[str, Optional[str], Optional[int]]:
    """Extrahiert --local/--light/--strong/--vision [1-3] nahe am ENDE des Texts.
    MULTILINE: $ matcht am Ende jeder Zeile, damit XML-Wrapping (z.B. </userRequest>)
    nach dem Flag die Erkennung nicht verhindert.
    Proximity-Check: Nur Matches innerhalb der letzten 300 Zeichen werden akzeptiert,
    um False-Positives in Code-Blöcken zu vermeiden.
    Returns: (bereinigter_text, category_string oder None, slot_number oder None)
    Slot-Nummer: 1=Primary, 2=Fallback 2, 3=Fallback 3. None = kein gültiger Slot angegeben.
    """
    found: Optional[str] = None
    found_slot: Optional[int] = None
    # Alle Matches finden, letzten gueltigen nahe am Text-Ende nehmen
    best_match = None
    for m in _MODEL_FLAG_PATTERN.finditer(text):
        cat = m.group(1).lower()
        if cat in _VALID_CATEGORIES and (len(text) - m.end()) <= _FLAG_PROXIMITY_MAX_CHARS:
            best_match = m
    if best_match:
        m = best_match
        found = m.group(1).lower()
        if m.group(2):
            slot_val = int(m.group(2))
            found_slot = slot_val if 1 <= slot_val <= 3 else None
        # Flag nur am Ende entfernen (vorherigen Whitespace mitnehmen)
        text = text[:m.start()].rstrip()
    cleaned = re.sub(r' {2,}', ' ', text)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()
    return cleaned, found, found_slot


def _strip_model_flags_from_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Entfernt --local/--light/--strong/--vision [1-3] NUR am ENDE jeder User-Message (in-place).
    Flags mitten im Text oder am Anfang werden ignoriert.
    """
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
    for key in ("light", "strong", "vision", "coworker"):
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
    content = message.get("content", "")
    if isinstance(content, list):
        # Array-Content (z.B. von VS Code Copilot mit Attachments): extrahiere Text-Blöcke
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts)
    return _normalize_text(content)


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


def _normalize_reasoning_value(val: Any) -> Optional[str]:
    """Normalisiert einen Reasoning-Wert aus verschiedenen Backend-Strukturen:
      - str: direkt
      - list: Content-Parts (z.B. {"type":"text","text":"..."}) -> konkateniert
      - dict: {"text": ...} / {"content": ...} / {"reasoning_content": ...}
    Returns str oder None."""
    if val is None:
        return None
    if isinstance(val, str):
        return val if val.strip() else None
    if isinstance(val, list):
        parts: List[str] = []
        for p in val:
            if isinstance(p, dict):
                for k in ("text", "content", "reasoning", "thinking", "reasoning_content"):
                    v = p.get(k)
                    if v is not None:
                        parts.append(str(v))
                        break
            elif p is not None:
                parts.append(str(p))
        joined = "".join(parts).strip()
        return joined or None
    if isinstance(val, dict):
        for k in ("text", "content", "reasoning_content", "reasoning", "thinking"):
            v = val.get(k)
            if v is not None and str(v).strip():
                return str(v)
        return None
    s = str(val)
    return s if s.strip() else None


def _extract_reasoning_from_delta(delta: Dict[str, Any]) -> Optional[str]:
    """Extrahiert Reasoning aus einem Stream-Delta bzw. einer Message.

    Mappt die verschiedenen Backend-Strukturen auf einheitliches
    reasoning_content:
      - reasoning_content (DeepSeek / vLLM Qwen3-Parser)
      - reasoning (Ollama / LM Studio, auch als Liste/Objekt)
      - thinking (manche Server)
    """
    if not isinstance(delta, dict):
        return None
    for key in ("reasoning_content", "reasoning", "thinking"):
        val = delta.get(key)
        if val is not None:
            norm = _normalize_reasoning_value(val)
            if norm:
                return norm
    return None


def _find_trailing_tag_prefix(s: str, tag: str) -> Optional[str]:
    """Findet das laengste Suffix von s, das ein ECHTES Praefix (kuerzer als
    tag) von tag ist. Returns None wenn keins existiert (dann ist kein
    angebrochenes Tag am Ende)."""
    if not s:
        return None
    max_len = min(len(s), len(tag) - 1)
    for cut in range(max_len, 0, -1):
        suffix = s[-cut:]
        if tag[:len(suffix)].lower() == suffix.lower():
            return suffix
    return None


def _split_think_chunk(chunk: str, state: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Streaming-Variante: verarbeitet einen Content-Chunk und liefert
    (reasoning_part, content_part) fuer <think>...</think>-Bloecke.

    <think>-Bloecke koennen ueber Chunk-Grenzen verteilt sein; `state` haelt
    (in_think, pending) — pending ist ein am Chunk-Ende angebrochenes Tag,
    das auf den naechsten Chunk wartet."""
    in_think = bool(state.get("in_think", False))
    s = str(state.get("pending", "") or "") + chunk
    state["pending"] = ""
    reasoning_out: List[str] = []
    content_out: List[str] = []
    while s:
        if not in_think:
            m = re.search(r"<think\s*>", s, re.IGNORECASE)
            if m:
                content_out.append(s[:m.start()])
                in_think = True
                s = s[m.end():]
                continue
            prefix = _find_trailing_tag_prefix(s, "<think>")
            if prefix is not None:
                content_out.append(s[:-len(prefix)])
                s = prefix
                break
            content_out.append(s)
            s = ""
        else:
            m = re.search(r"</think\s*>", s, re.IGNORECASE)
            if m:
                reasoning_out.append(s[:m.start()])
                in_think = False
                s = s[m.end():]
                continue
            prefix = _find_trailing_tag_prefix(s, "</think>")
            if prefix is not None:
                reasoning_out.append(s[:-len(prefix)])
                s = prefix
                break
            reasoning_out.append(s)
            s = ""
    state["in_think"] = in_think
    state["pending"] = s
    reasoning = "".join(reasoning_out) or None
    content = "".join(content_out) or None
    return reasoning, content


def _split_think_content(content: str) -> Tuple[str, Optional[str]]:
    """Trennt <think>...</think>-Bloecke aus einem KOMPLETTEN Content-String.
    Returns (clean_content, reasoning) — reasoning=None wenn kein Block."""
    if not content:
        return content, None
    reasoning_parts: List[str] = []
    clean_parts: List[str] = []
    pos = 0
    in_think = False
    while pos < len(content):
        if not in_think:
            m = re.search(r"<think\s*>", content[pos:], re.IGNORECASE)
            if not m:
                clean_parts.append(content[pos:])
                break
            clean_parts.append(content[pos:pos + m.start()])
            in_think = True
            pos += m.end()
        else:
            m = re.search(r"</think\s*>", content[pos:], re.IGNORECASE)
            if not m:
                reasoning_parts.append(content[pos:])
                break
            reasoning_parts.append(content[pos:pos + m.start()])
            in_think = False
            pos += m.end()
    if not reasoning_parts:
        return content, None
    reasoning = "".join(reasoning_parts).strip()
    clean = "".join(clean_parts)
    return clean, reasoning or None


def _extract_message_parts(result: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[List[Dict[str, Any]]]]:
    """Extrahiert (content, reasoning_content, tool_calls) aus einer Chat-Response.

    Mappt verschiedene Reasoning-Strukturen in einheitliches reasoning_content:
      - message.reasoning_content / .reasoning / .thinking
      - <think>...</think> im content (vLLM Qwen3 preserve_thinking) -> wird
        aus dem content herausgetrennt und als reasoning_content gemappt.
    """
    message = _extract_choice_message(result)
    tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
    reasoning_content = _extract_reasoning_from_delta(message) if isinstance(message, dict) else None
    content = _extract_choice_content(result)
    if not reasoning_content and content:
        clean, think_r = _split_think_content(content)
        if think_r:
            reasoning_content = think_r
            content = clean
    return content, reasoning_content, tool_calls


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


# ── Read-Loop-Detection ─────────────────────────────────────────────────────

def _extract_read_signature(args_raw: Any) -> Optional[str]:
    """Extrahiert eine stabile Signatur (filePath|startLine|endLine) aus read_file Arguments.
    Gibt None zurueck wenn keine gueltige read_file-Signatur erkennbar ist.
    """
    try:
        if isinstance(args_raw, str):
            args = json.loads(args_raw)
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            return None
    except (json.JSONDecodeError, ValueError, TypeError):
        return None

    if not isinstance(args, dict):
        return None

    file_path = str(args.get("filePath") or args.get("file_path") or args.get("path") or "").strip()
    start_line = args.get("startLine") or args.get("start_line") or args.get("start")
    end_line = args.get("endLine") or args.get("end_line") or args.get("end")

    if not file_path:
        return None

    start_str = str(start_line).strip() if start_line is not None else ""
    end_str = str(end_line).strip() if end_line is not None else ""

    return f"{file_path}|{start_str}|{end_str}"


_READ_FILE_TOOL_NAMES = {"read_file", "readFile", "read_lines", "readLines"}

# Bekannte Search-Tool-Namen (fuer Signatur-Extraktion / Response-Filter).
# Die Detection selbst ist zusaetzlich tool-namen-unabhaengig via No-Match-Results.
_SEARCH_TOOL_NAMES = {"grep_search", "grepSearch", "file_search", "fileSearch",
                      "codebase_search", "search", "grep", "find_files"}

_NO_MATCHES_INDICATORS = (
    "No matches found",
    "no matches found",
    "0 results",
    "No files found",
    "no results found",
    "No results",
)


def _truncate_messages_from(messages: List[Dict[str, Any]], msg_index: int,
                            intervention_text: str) -> None:
    """Trunciert die messages-Liste ab msg_index (inklusiv) und appended eine
    Intervention-User-Message. Stellt sicher, dass keine dangling tool_calls / tool
    results zurueckbleiben (d.h. assistant mit tool_calls ohne result ist ungueltig).

    WICHTIG: Schneidet bis zum ENDE ab (nicht nur ein Segment). Das ist noetig,
    weil VS Code die volle Historie mitschickt — wenn nur die Mitte entfernt
    wuerde, bliebe der Loop-Tail erhalten und die Detection feuert endlos
    mit demselben Cut-Punkt, waehrend das Modell weiter loopt.
    """
    # Rueckwaerts vom Cut-Punkt pruefen: die letzte verbleibende Message darf KEINE
    # assistant mit tool_calls ohne nachfolgendes tool-result sein.
    # Da wir ab msg_index alles abschneiden, duerfen davor keine unpaarigen tool_calls stehen.
    # Falls die letzte verbleibende Message assistant+tool_calls ist, muessen wir diese
    # ebenfalls entfernen (sonst ist die Konversation ungueltig).
    while msg_index > 0:
        prev = messages[msg_index - 1]
        if isinstance(prev, dict) and prev.get("role") == "assistant" and prev.get("tool_calls"):
            msg_index -= 1
        else:
            break

    del messages[msg_index:]
    messages.append({"role": "user", "content": intervention_text})


def _is_search_tool_name(name: str) -> bool:
    n = (name or "").strip()
    if not n:
        return False
    if n in _SEARCH_TOOL_NAMES:
        return True
    low = n.lower()
    return "search" in low or "grep" in low


def _collect_read_entries(messages: List[Dict[str, Any]]) -> List[Tuple[int, str, str]]:
    """(msg_idx, full_sig, file_path) fuer alle read_file tool_calls."""
    entries: List[Tuple[int, str, str]] = []
    for msg_idx, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        tool_calls = m.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function") or {}
            if not isinstance(func, dict):
                continue
            name = str(func.get("name", "")).strip()
            if name not in _READ_FILE_TOOL_NAMES:
                continue
            sig = _extract_read_signature(func.get("arguments", ""))
            if not sig:
                continue
            file_path = sig.split("|", 1)[0]
            entries.append((msg_idx, sig, file_path))
    return entries


def _collect_search_entries(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Tuple[int, str, str, bool]], Dict[str, Any]]:
    """Sammelt Search-Tool-Calls: (msg_idx, tc_id, sig, is_no_match).

    Returns (entries, diag_info).
    """
    tool_results: Dict[str, str] = {}
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "")
        if role == "tool":
            tc_id = str(m.get("tool_call_id", ""))
            if tc_id:
                tool_results[tc_id] = _extract_tool_result_text(m)
        elif role == "function":
            tc_id = str(m.get("tool_call_id", "") or m.get("name", ""))
            if tc_id:
                tool_results[tc_id] = _extract_tool_result_text(m)

    entries: List[Tuple[int, str, str, bool]] = []
    diag_names: Set[str] = set()
    diag_total = 0
    diag_no_match = 0
    diag_id_miss = 0

    for msg_idx, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        tool_calls = m.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function") or {}
            if not isinstance(func, dict):
                continue
            name = str(func.get("name", "")).strip()
            diag_total += 1
            diag_names.add(name)
            # Search-Detection: tool-namen-unabhaengig via Result ODER Search-Name
            tc_id = str(tc.get("id", ""))
            result_text = tool_results.get(tc_id, "")
            if not result_text:
                # Positionales Fallback: naechste tool-message
                next_idx = msg_idx + 1
                if next_idx < len(messages):
                    next_m = messages[next_idx]
                    if isinstance(next_m, dict) and next_m.get("role") in ("tool", "function"):
                        result_text = _extract_tool_result_text(next_m)
                if not result_text:
                    diag_id_miss += 1
            is_no_match = _is_no_match_result(result_text)
            if is_no_match:
                diag_no_match += 1
            # Nur Search-Tools ODER No-Match-Results als Search-Entries
            if not (_is_search_tool_name(name) or is_no_match):
                continue
            sig = _extract_search_signature(func.get("arguments", ""))
            if sig:
                entries.append((msg_idx, tc_id, sig, is_no_match))

    diag = {
        "total_tcs": diag_total,
        "names": sorted(diag_names),
        "tool_results_mapped": len(tool_results),
        "no_match_results": diag_no_match,
        "id_misses": diag_id_miss,
        "search_entries": len(entries),
    }
    return entries, diag


def _collect_generic_tool_entries(
    messages: List[Dict[str, Any]],
) -> List[Tuple[int, str, str]]:
    """Sammelt ALLE tool_calls: (msg_idx, tool_name, sig).

    Sig = name + universelle Argument-Signatur (sortierte key=value Paare).
    Read/Search-Tools werden ausgeschlossen (haben eigene Detection).
    """
    entries: List[Tuple[int, str, str]] = []
    for msg_idx, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        tool_calls = m.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function") or {}
            if not isinstance(func, dict):
                continue
            name = str(func.get("name", "")).strip()
            if not name:
                continue
            # Read/Search-Tools haben eigene Detection
            if name in _READ_FILE_TOOL_NAMES or _is_search_tool_name(name):
                continue
            sig = _extract_tool_call_signature(func.get("arguments", ""))
            if not sig:
                continue
            entries.append((msg_idx, name, sig))
    return entries


def _detect_generic_tool_loop_inplace(
    messages: List[Dict[str, Any]], label: str = "Payload",
    category: str = "", model_name: str = "",
) -> bool:
    """Erkennt wenn ein beliebiges Tool (z.B. manage_todo_list) mit identischen
    Argumenten >GENERIC_TOOL_LOOP_THRESHOLD mal hintereinander aufgerufen wird.

    Trunciert ab dem 2. Vorkommen bis zum Ende + Intervention.
    Nur fuer Laguna-S-2.1 Modelle aktiv.
    """
    if not _is_laguna_model(model_name):
        return False
    if GENERIC_TOOL_LOOP_THRESHOLD <= 0:
        return False

    entries = _collect_generic_tool_entries(messages)
    if not entries:
        return False

    # Trailing-Sequenz: gleiche (name, sig) am Ende
    _last_idx, last_name, last_sig = entries[-1]
    trailing_indices: List[int] = []
    for msg_idx, name, sig in reversed(entries):
        if name == last_name and sig == last_sig:
            trailing_indices.append(msg_idx)
        else:
            break
    trailing_indices.reverse()

    if len(trailing_indices) > GENERIC_TOOL_LOOP_THRESHOLD:
        cut = trailing_indices[1] if len(trailing_indices) > 1 else trailing_indices[0]
        intervention_text = GENERIC_TOOL_LOOP_INTERVENTION.format(
            tool=last_name, count=len(trailing_indices)
        )
        _truncate_messages_from(messages, cut, intervention_text)
        _log(f"{label}: GENERIC-TOOL-LOOP erkannt ({len(trailing_indices)}x "
             f"'{last_name}' mit identischen Argumenten), Konversation trunkiert "
             f"bei idx={cut}")
        return True

    return False


def _detect_read_loop_inplace(messages: List[Dict[str, Any]], label: str = "Payload",
                              category: str = "", model_name: str = "") -> bool:
    """Erkennt Read-Loops: gleiche Datei + gleiche Zeilen >N mal hintereinander.
    NUR fuer Laguna-S-2.1 Modelle (local oder cloud). Alle anderen Modelle
    werden nicht beeinflusst.

    AKTIVE INTERVENTION durch Truncation: Ab dem Loop-Start wird ALLES bis zum
    Ende entfernt und durch eine Intervention ersetzt. Das verhindert, dass
    VS Code den Loop-Tail mitschickt und die Detection endlos denselben alten
    Cut-Punkt trifft, waehrend das Modell weiter loopt.

    Zwei Detection-Modi:
      1. Exact-Loop: identische Signaturen (file+lines) >READ_LOOP_THRESHOLD konsekutiv
      2. File-Crawl: gleiche Datei (verschiedene Zeilen) >READ_LOOP_FILE_THRESHOLD mal
         in den letzten READ_LOOP_FILE_WINDOW Reads
    """
    if not _is_laguna_model(model_name):
        return False
    if READ_LOOP_THRESHOLD <= 0:
        return False

    read_entries = _collect_read_entries(messages)
    if not read_entries:
        return False

    # ── Modus 1: Exact-Loop (gleiche Datei + gleiche Zeilen konsekutiv) ──
    # Nur die TRAILING-Sequenz am Ende zaehlt (sonst greift alte Historie ewig).
    trailing_sig = read_entries[-1][1]
    trailing_indices: List[int] = []
    for msg_idx, sig, _fp in reversed(read_entries):
        if sig == trailing_sig:
            trailing_indices.append(msg_idx)
        else:
            break
    trailing_indices.reverse()
    trailing_count = len(trailing_indices)

    if trailing_count > READ_LOOP_THRESHOLD:
        # Behalte den ersten Read der trailing-Sequenz, schneide ab dem 2.
        truncate_idx = (
            trailing_indices[1] if len(trailing_indices) > 1 else trailing_indices[0]
        )
        intervention_text = READ_LOOP_INTERVENTION.format(count=trailing_count)
        _truncate_messages_from(messages, truncate_idx, intervention_text)
        _log(f"{label}: READ-LOOP (exact) erkannt ({trailing_count}x trailing "
             f"gleiche read_file-Parameter), Konversation bei idx={truncate_idx} "
             f"trunkiert + Intervention")
        return True

    # ── Modus 2: File-Crawl (gleiche Datei, verschiedene Zeilen, gehaeuft) ──
    if READ_LOOP_FILE_THRESHOLD > 0 and READ_LOOP_FILE_WINDOW > 0:
        window = read_entries[-READ_LOOP_FILE_WINDOW:]
        file_counts: Dict[str, int] = {}
        for _idx, _sig, file_part in window:
            file_counts[file_part] = file_counts.get(file_part, 0) + 1

        # Preferiere die Datei mit den meisten Reads im Fenster
        worst_file = ""
        worst_count = 0
        for file_path, count in file_counts.items():
            if count > worst_count:
                worst_file = file_path
                worst_count = count

        if worst_file and worst_count > READ_LOOP_FILE_THRESHOLD:
            keep_n = max(0, READ_LOOP_FILE_KEEP)
            occurrences = [e_idx for e_idx, _sig, fp in window if fp == worst_file]
            if len(occurrences) > READ_LOOP_FILE_THRESHOLD:
                # Behalte die ersten keep_n Reads der Datei, schneide ab dem naechsten
                cut_pos = min(keep_n, len(occurrences) - 1)
                truncate_idx = occurrences[cut_pos]
                intervention_text = READ_LOOP_FILE_INTERVENTION.format(
                    file=worst_file, count=worst_count, window=READ_LOOP_FILE_WINDOW
                )
                _truncate_messages_from(messages, truncate_idx, intervention_text)
                _log(f"{label}: READ-LOOP (file-crawl) erkannt ({worst_count}x '{worst_file}' "
                     f"in letzten {len(window)} reads), Konversation trunkiert bei "
                     f"idx={truncate_idx} (keep={keep_n})")
                return True

    return False


# ── Search-Loop-Detection ──────────────────────────────────────────────────

def _extract_tool_call_signature(args_raw: Any) -> Optional[str]:
    """Extrahiert eine stabile Signatur aus Tool-Call-Arguments.
    Universell: funktioniert fuer search, read, und beliebige andere Tools.
    Signatur = sortierte key=value Paare der Arguments.
    """
    try:
        if isinstance(args_raw, str):
            args = json.loads(args_raw)
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            return None
    except (json.JSONDecodeError, ValueError, TypeError):
        return None

    if not isinstance(args, dict) or not args:
        return None

    # Stabile Signatur: sortierte key=value Paare
    parts = []
    for k in sorted(args.keys()):
        v = args[k]
        parts.append(f"{k}={v}")
    return "|".join(parts)


def _extract_search_signature(args_raw: Any) -> Optional[str]:
    """Extrahiert eine Search-Signatur (query|includePattern) aus Arguments.
    Fallback auf _extract_tool_call_signature wenn keine query gefunden wird.
    """
    try:
        if isinstance(args_raw, str):
            args = json.loads(args_raw)
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            return None
    except (json.JSONDecodeError, ValueError, TypeError):
        return None

    if not isinstance(args, dict):
        return None

    query = str(args.get("query") or args.get("pattern") or args.get("search") or "").strip()
    if not query:
        # Kein query-Feld: universelle Signatur verwenden
        return _extract_tool_call_signature(args_raw)

    include = str(args.get("includePattern") or args.get("include_pattern")
                  or args.get("path") or args.get("glob") or "").strip()
    return f"{query}|{include}"


def _extract_tool_result_text(msg: Dict[str, Any]) -> str:
    """Extrahiert den Text-Inhalt einer Tool-Result-Message."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "\n".join(parts)
    return ""


def _is_no_match_result(text: str) -> bool:
    """Prueft ob ein Tool-Ergebnis 'kein Ergebnis' signalisiert."""
    if not text:
        return False
    return any(indicator in text for indicator in _NO_MATCHES_INDICATORS)


def _detect_search_loop_inplace(messages: List[Dict[str, Any]], label: str = "Payload",
                                category: str = "", model_name: str = "") -> bool:
    """Erkennt Search-Loops fuer Laguna-S-2.1 Modelle (local oder cloud).

    Zwei Modi:
      1. No-Match-Loop: gleiche Signatur >SEARCH_LOOP_THRESHOLD mit No-Match
      2. Repeat-Loop: gleiche Signatur >SEARCH_REPEAT_THRESHOLD (auch MIT Treffern)

    Nur TRAILING-Sequenzen am Ende der Historie zaehlen. Bei Erkennung:
    Truncation ab dem 2. Vorkommen bis zum Ende + Intervention.
    """
    if not _is_laguna_model(model_name):
        return False
    if SEARCH_LOOP_THRESHOLD <= 0 and SEARCH_REPEAT_THRESHOLD <= 0:
        return False

    all_entries, diag = _collect_search_entries(messages)

    if diag["total_tcs"] > 0 and diag["search_entries"] == 0:
        _log(f"{label}: SEARCH-LOOP-DIAG: {diag['total_tcs']} tool_calls gefunden, "
             f"names={diag['names']}, tool_results_mapped={diag['tool_results_mapped']}, "
             f"no_match_results={diag['no_match_results']}, id_misses={diag['id_misses']}")

    if not all_entries:
        return False

    # ── Modus 1: Trailing No-Match-Loop ──
    if SEARCH_LOOP_THRESHOLD > 0:
        nm_indices: List[int] = []
        nm_sig = ""
        for msg_idx, _tc_id, sig, is_no_match in reversed(all_entries):
            if not is_no_match:
                break
            if not nm_sig:
                nm_sig = sig
            if sig != nm_sig:
                break
            nm_indices.append(msg_idx)
        nm_indices.reverse()
        if nm_sig and len(nm_indices) > SEARCH_LOOP_THRESHOLD:
            cut = nm_indices[1] if len(nm_indices) > 1 else nm_indices[0]
            query = nm_sig.split("|", 1)[0]
            intervention_text = SEARCH_LOOP_INTERVENTION.format(
                query=query[:120], count=len(nm_indices)
            )
            _truncate_messages_from(messages, cut, intervention_text)
            _log(f"{label}: SEARCH-LOOP (no-match) erkannt ({len(nm_indices)}x "
                 f"'{query[:80]}'), Konversation trunkiert bei idx={cut}")
            return True

    # ── Modus 2: Trailing Repeat-Loop (auch mit Treffern) ──
    if SEARCH_REPEAT_THRESHOLD > 0:
        rep_indices: List[int] = []
        rep_sig = all_entries[-1][2]
        for msg_idx, _tc_id, sig, _nm in reversed(all_entries):
            if sig != rep_sig:
                break
            rep_indices.append(msg_idx)
        rep_indices.reverse()
        if len(rep_indices) > SEARCH_REPEAT_THRESHOLD:
            cut = rep_indices[1] if len(rep_indices) > 1 else rep_indices[0]
            query = rep_sig.split("|", 1)[0]
            intervention_text = SEARCH_REPEAT_INTERVENTION.format(
                query=query[:120], count=len(rep_indices)
            )
            _truncate_messages_from(messages, cut, intervention_text)
            _log(f"{label}: SEARCH-LOOP (repeat) erkannt ({len(rep_indices)}x "
                 f"'{query[:80]}' auch mit Treffern), Konversation trunkiert bei "
                 f"idx={cut}")
            return True

    return False


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


# ── Response-Level Loop-Enforcement ────────────────────────────────────────
# Letzter Ausweg: Wenn das Modell trotz History-Intervention denselben
# Read/Search-Loop fortsetzt, werden die offending tool_calls aus der Response
# entfernt. Defaults/Thresholds stehen oben bei den Loop-Globals.


def _detect_response_loop(body: Dict[str, Any], tool_calls: List[Dict[str, Any]],
                          category: str = "", model_name: str = "") -> Tuple[List[str], List[str]]:
    """Detect-only Loop-Check fuer die Modell-Response.

    Gibt (loop_reasons, blocked_tool_names) zurueck. Blockiert nichts.
    Nur fuer Laguna-S-2.1 Modelle aktiv.
    """
    if not _is_laguna_model(model_name) or _RESPONSE_LOOP_THRESHOLD <= 0 or not tool_calls:
        return [], []

    messages = body.get("messages", [])
    if not isinstance(messages, list):
        return [], []

    read_entries = _collect_read_entries(messages)
    search_entries, _diag = _collect_search_entries(messages)

    trailing_read_sig = read_entries[-1][1] if read_entries else ""
    trailing_read_count = 0
    trailing_read_file = ""
    if trailing_read_sig:
        trailing_read_file = trailing_read_sig.split("|", 1)[0]
        for _idx, sig, _fp in reversed(read_entries):
            if sig == trailing_read_sig:
                trailing_read_count += 1
            else:
                break

    file_crawl_counts: Dict[str, int] = {}
    if read_entries and READ_LOOP_FILE_WINDOW > 0:
        window = read_entries[-READ_LOOP_FILE_WINDOW:]
        for _idx, _sig, fp in window:
            file_crawl_counts[fp] = file_crawl_counts.get(fp, 0) + 1

    trailing_search_sig = search_entries[-1][2] if search_entries else ""
    trailing_search_count = 0
    trailing_search_no_match_count = 0
    if trailing_search_sig:
        for _idx, _tcid, sig, is_nm in reversed(search_entries):
            if sig != trailing_search_sig:
                break
            trailing_search_count += 1
        for _idx, _tcid, sig, is_nm in reversed(search_entries):
            if not is_nm or sig != trailing_search_sig:
                break
            trailing_search_no_match_count += 1

    # Generic tool entries (alle non-read/search Tools)
    generic_entries = _collect_generic_tool_entries(messages)
    trailing_gen_name = generic_entries[-1][1] if generic_entries else ""
    trailing_gen_sig = generic_entries[-1][2] if generic_entries else ""
    trailing_gen_count = 0
    if trailing_gen_name:
        for _idx, gname, gsig in reversed(generic_entries):
            if gname == trailing_gen_name and gsig == trailing_gen_sig:
                trailing_gen_count += 1
            else:
                break

    thr = _RESPONSE_LOOP_THRESHOLD
    reasons: List[str] = []
    blocked_names: Set[str] = set()

    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function") or {}
        if not isinstance(func, dict):
            continue
        name = str(func.get("name", "")).strip()
        args_raw = func.get("arguments", "")
        reason = ""

        if name in _READ_FILE_TOOL_NAMES:
            rsig = _extract_read_signature(args_raw)
            if rsig:
                rfile = rsig.split("|", 1)[0]
                if trailing_read_count >= thr and rsig == trailing_read_sig:
                    reason = f"exact-read {trailing_read_count}x"
                elif (READ_LOOP_FILE_THRESHOLD > 0
                      and file_crawl_counts.get(rfile, 0) > READ_LOOP_FILE_THRESHOLD):
                    reason = f"file-crawl {file_crawl_counts[rfile]}x '{rfile}'"
                elif (trailing_read_file and rfile == trailing_read_file
                      and file_crawl_counts.get(rfile, 0) >= thr
                      and trailing_read_count >= thr):
                    reason = f"trailing-file {trailing_read_count}x '{rfile}'"
        elif _is_search_tool_name(name):
            ssig = _extract_search_signature(args_raw)
            if ssig and ssig == trailing_search_sig:
                if trailing_search_no_match_count >= thr:
                    reason = f"search-no-match {trailing_search_no_match_count}x"
                elif trailing_search_count >= thr:
                    reason = f"search-repeat {trailing_search_count}x"
        else:
            # Generische Tools (z.B. manage_todo_list): identische Argumente wiederholt
            gsig = _extract_tool_call_signature(args_raw)
            if (gsig and trailing_gen_name == name and gsig == trailing_gen_sig
                    and GENERIC_TOOL_LOOP_THRESHOLD > 0
                    and trailing_gen_count > GENERIC_TOOL_LOOP_THRESHOLD):
                reason = f"generic-tool {trailing_gen_count}x '{name}'"

        if reason:
            reasons.append(f"{name}: {reason}")
            if name:
                blocked_names.add(name)

    return reasons, sorted(blocked_names)


def _filter_looping_response_tool_calls(
    body: Dict[str, Any],
    tool_calls: List[Dict[str, Any]],
    category: str = "",
    model_name: str = "",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Filtert Tool-Calls aus der Modell-Response die einen aktiven Loop fortsetzen.
    Nur fuer Laguna-S-2.1 Modelle aktiv.

    Blockiert:
      - read_file derselben Datei wenn trailing file-crawl / exact-read-loop
      - search mit gleicher Signatur wenn trailing search-repeat / no-match-loop

    Gibt (kept_tool_calls, removed_tool_calls) zurueck.
    """
    if not _is_laguna_model(model_name) or _RESPONSE_LOOP_THRESHOLD <= 0 or not tool_calls:
        return list(tool_calls), []

    messages = body.get("messages", [])
    if not isinstance(messages, list):
        return list(tool_calls), []

    read_entries = _collect_read_entries(messages)
    search_entries, _diag = _collect_search_entries(messages)

    # Trailing exact-read counts: sig -> count
    trailing_read_sig = read_entries[-1][1] if read_entries else ""
    trailing_read_count = 0
    trailing_read_file = ""
    if trailing_read_sig:
        trailing_read_file = trailing_read_sig.split("|", 1)[0]
        for _idx, sig, _fp in reversed(read_entries):
            if sig == trailing_read_sig:
                trailing_read_count += 1
            else:
                break

    # File-crawl counts im Fenster
    file_crawl_counts: Dict[str, int] = {}
    if read_entries and READ_LOOP_FILE_WINDOW > 0:
        window = read_entries[-READ_LOOP_FILE_WINDOW:]
        for _idx, _sig, fp in window:
            file_crawl_counts[fp] = file_crawl_counts.get(fp, 0) + 1

    # Trailing search counts
    trailing_search_sig = search_entries[-1][2] if search_entries else ""
    trailing_search_count = 0
    trailing_search_no_match_count = 0
    if trailing_search_sig:
        for _idx, _tcid, sig, is_nm in reversed(search_entries):
            if sig != trailing_search_sig:
                break
            trailing_search_count += 1
            if is_nm:
                trailing_search_no_match_count += 1
            else:
                # gemischte Results: no-match-streak bricht, repeat bleibt
                pass
        # reine no-match trailing streak separat
        trailing_search_no_match_count = 0
        for _idx, _tcid, sig, is_nm in reversed(search_entries):
            if not is_nm or sig != trailing_search_sig:
                break
            trailing_search_no_match_count += 1

    thr = _RESPONSE_LOOP_THRESHOLD
    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []

    for tc in tool_calls:
        if not isinstance(tc, dict):
            kept.append(tc)
            continue
        func = tc.get("function") or {}
        if not isinstance(func, dict):
            kept.append(tc)
            continue
        name = str(func.get("name", "")).strip()
        args_raw = func.get("arguments", "")

        block = False
        reason = ""

        if name in _READ_FILE_TOOL_NAMES:
            rsig = _extract_read_signature(args_raw)
            if rsig:
                rfile = rsig.split("|", 1)[0]
                # Exact trailing read loop fortsetzen
                if (trailing_read_count >= thr and rsig == trailing_read_sig):
                    block = True
                    reason = f"exact-read {trailing_read_count}x"
                # File-crawl fortsetzen
                elif (READ_LOOP_FILE_THRESHOLD > 0
                      and file_crawl_counts.get(rfile, 0) > READ_LOOP_FILE_THRESHOLD):
                    block = True
                    reason = (f"file-crawl {file_crawl_counts[rfile]}x "
                              f"'{rfile}'")
                # Auch wenn trailing file matches and count high enough vs thr
                elif (trailing_read_file and rfile == trailing_read_file
                      and file_crawl_counts.get(rfile, 0) >= thr
                      and trailing_read_count >= thr):
                    block = True
                    reason = f"trailing-file {trailing_read_count}x '{rfile}'"

        elif _is_search_tool_name(name):
            ssig = _extract_search_signature(args_raw)
            if ssig and ssig == trailing_search_sig:
                if trailing_search_no_match_count >= thr:
                    block = True
                    reason = f"search-no-match {trailing_search_no_match_count}x"
                elif trailing_search_count >= thr:
                    block = True
                    reason = f"search-repeat {trailing_search_count}x"

        if block:
            _log(f"RESPONSE-LOOP: Tool '{name}' geblockt ({reason}, "
                 f"threshold={thr})")
            removed.append(tc)
        else:
            kept.append(tc)

    return kept, removed


def _check_response_loop(body: Dict[str, Any], tool_calls: List[Dict[str, Any]],
                         category: str = "", model_name: str = "") -> bool:
    """True wenn ALLE tool_calls als Loop gefiltert wuerden."""
    kept, removed = _filter_looping_response_tool_calls(body, tool_calls, category, model_name=model_name)
    return bool(removed) and not kept


# ── Regex fuer Model-Namen, die max_completion_tokens statt max_tokens benoetigen ────
_MODEL_NEEDS_MAX_COMPLETION_TOKENS_RE = re.compile(
    r'^(o[1349]|o[1349]-|o-series|gpt-4\.(?:1|5|o|.5)|gpt-5)', re.IGNORECASE
)


def _needs_max_completion_tokens(model_name: str) -> bool:
    """Ermittelt ob ein Model max_completion_tokens statt max_tokens benoetigt.
    Erkennung via Modell-Namen-Praefix (o1-, o3-, o4-, o-series, gpt-4.1, gpt-4.5, etc.).
    """
    if not model_name:
        return False
    return bool(_MODEL_NEEDS_MAX_COMPLETION_TOKENS_RE.match(model_name))


def _patch_max_tokens_payload(payload: Dict[str, Any], cat: Dict[str, Any]) -> None:
    """Wandelt max_tokens in max_completion_tokens um, wenn das Modell es erfordert.
    Entscheidungsreihenfolge:
      1. cat['use_max_completion_tokens'] == True  → explizite Config
      2. _needs_max_completion_tokens(model_name)   → Auto-Detect via Modell-Praefix
    """
    model_name = str(cat.get("model_name", payload.get("model", "")))
    use_mct = bool(cat.get("use_max_completion_tokens", False))
    if not use_mct and not _needs_max_completion_tokens(model_name):
        return
    # max_tokens steht bereits im payload, umbenennen
    if "max_tokens" in payload:
        val = payload.pop("max_tokens")
        payload["max_completion_tokens"] = val
        _log(f"max_tokens → max_completion_tokens fuer Model '{model_name}' (config={use_mct}, auto={not use_mct})")


# Regex fuer Modelle, bei denen reasoning_effort NICHT mit function tools zusammen verwendet werden darf
_MODEL_REASONING_EFFORT_STRIP_RE = re.compile(
    r'^(gpt-5)', re.IGNORECASE
)


def _needs_reasoning_effort_strip(model_name: str, payload: Dict[str, Any]) -> bool:
    """Ermittelt, ob reasoning_effort aus dem Payload entfernt werden muss.
    Das ist der Fall, wenn das Modell reasoning_effort nicht in Kombination
    mit function tools im /v1/chat/completions-Endpunkt unterstuetzt.
    """
    if not model_name:
        return False
    # Nur relevant wenn reasoning_effort UND tools vorhanden sind
    if "reasoning_effort" not in payload:
        return False
    has_tools = bool(payload.get("tools")) or bool(payload.get("tool_choice"))
    if not has_tools:
        return False
    return bool(_MODEL_REASONING_EFFORT_STRIP_RE.match(model_name))


def _patch_reasoning_effort_payload(payload: Dict[str, Any], cat: Dict[str, Any]) -> None:
    """Entfernt reasoning_effort aus dem Payload, wenn das Modell es nicht
    zusammen mit function tools unterstuetzt.
    """
    model_name = str(cat.get("model_name", payload.get("model", "")))
    if not _needs_reasoning_effort_strip(model_name, payload):
        return
    if "reasoning_effort" in payload:
        old_val = payload.pop("reasoning_effort")
        _log(f"reasoning_effort='{old_val}' entfernt fuer Model '{model_name}' (tools + reasoning_effort inkompatibel)")


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


def _clean_payload(payload: Dict[str, Any], keep_tools: bool = False,
                   keep_top_k: bool = False) -> Dict[str, Any]:
    if not payload.get("stream") and "stream_options" in payload:
        payload.pop("stream_options")
    strip_keys = ["stop_sequences", "safety_settings", "response_format"]
    if not keep_top_k:
        strip_keys.append("top_k")
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

def _patch_local_sampling_payload(payload: Dict[str, Any]) -> None:
    """Setzt empfohlene Sampling-Parameter fuer Laguna-S-2.1 Modelle (local oder cloud).

    Empfohlen aus poolside/Laguna-S-2.1 Diskussion #23:
      temp=0.7, top_p=0.95, top_k=20, min_p=0.0
      DRY: multiplier=0.8, base=1.75, allowed_length=3, penalty_last_n=-1
      chat_template_kwargs: enable_thinking=true, preserve_thinking=true
    """
    payload["temperature"] = LOCAL_TEMPERATURE
    payload["top_p"] = LOCAL_TOP_P
    payload["top_k"] = LOCAL_TOP_K
    payload["min_p"] = LOCAL_MIN_P

    # DRY-Parameter (llama.cpp / vLLM kompatibel)
    if LOCAL_DRY_MULTIPLIER > 0:
        payload["dry_multiplier"] = LOCAL_DRY_MULTIPLIER
        payload["dry_base"] = LOCAL_DRY_BASE
        payload["dry_allowed_length"] = LOCAL_DRY_ALLOWED_LENGTH
        payload["dry_penalty_last_n"] = LOCAL_DRY_PENALTY_LAST_N
        payload["dry_sequence_breaker"] = LOCAL_DRY_SEQUENCE_BREAKER

    # chat_template_kwargs (vLLM / llama.cpp --jinja)
    payload["chat_template_kwargs"] = {
        "enable_thinking": LOCAL_ENABLE_THINKING,
        "preserve_thinking": LOCAL_PRESERVE_THINKING,
    }

    _log(f"Local-Sampling: temp={LOCAL_TEMPERATURE}, top_p={LOCAL_TOP_P}, "
         f"top_k={LOCAL_TOP_K}, min_p={LOCAL_MIN_P}, "
         f"dry_mult={LOCAL_DRY_MULTIPLIER}, dry_base={LOCAL_DRY_BASE}, "
         f"thinking={LOCAL_ENABLE_THINKING}, preserve={LOCAL_PRESERVE_THINKING}")


def _inject_local_anti_loop_system(messages: List[Dict[str, Any]]) -> None:
    """Injiziert Anti-Loop-System-Prompt fuer Laguna-S-2.1 Modelle.

    Wird NACH der ersten System-Message (falls vorhanden) eingefuegt,
    damit der Modell-eigene System-Prompt Vorrang hat.
    """
    if not LOCAL_ANTI_LOOP_SYSTEM_PROMPT.strip():
        return
    insert_idx = 0
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        insert_idx = 1
    messages.insert(insert_idx, {
        "role": "system",
        "content": LOCAL_ANTI_LOOP_SYSTEM_PROMPT,
    })
    _log("Local-Anti-Loop-System-Prompt injiziert")


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

    # Laguna-S-2.1 Loop-Detection: derzeit deaktiviert (Laguna nicht mehr im Einsatz)
    # _detect_read_loop_inplace(messages, "Passthrough", category=category, model_name=cat["model_name"])
    # _detect_search_loop_inplace(messages, "Passthrough", category=category, model_name=cat["model_name"])

    if not cat.get("is_vision", False):
        _sanitize_image_urls_inplace(messages, "Passthrough")

    # Laguna-S-2.1 Sampling + Anti-Loop: derzeit deaktiviert (Laguna nicht mehr im Einsatz)
    # if _is_laguna_model(cat["model_name"]):
    #     _patch_local_sampling_payload(payload)
    #     _inject_local_anti_loop_system(messages)

    _patch_moonshot_payload(payload, cat.get("api_url", ""))
    _patch_max_tokens_payload(payload, cat)
    _patch_reasoning_effort_payload(payload, cat)

    # Co-Worker-Delegation: ask_coworker-Tool nur bei Kategorie=local + Health-OK
    if category == "local":
        _inject_coworker_tool(payload)

    return _clean_payload(payload, keep_tools=True, keep_top_k=False)


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

async def _call_single_model(body: Dict[str, Any], category: str, def_idx: int = 0,
                             inject_hindsight: bool = True) -> Dict[str, Any]:
    defs = _model_defs(category)
    if not defs or def_idx >= len(defs):
        return {"category": category, "status": "error", "def_idx": def_idx,
                "content": f"Keine gueltige Modell-Definition fuer {category}[{def_idx}]",
                "trigger_fallback": False}
    cat = defs[def_idx]
    started = time.perf_counter()

    payload = _build_passthrough_payload(body, category, def_idx=def_idx)

    messages = payload.get("messages", [])
    if isinstance(messages, list) and inject_hindsight:
        _inject_hindsight_context(messages)

    model = cat["model_name"]
    api_url = cat["api_url"].rstrip("/")
    api_key = cat.get("api_key", "")
    timeout = float(cat.get("timeout_seconds", 300))
    read_timeout = float(cat.get("read_timeout_seconds", timeout))

    msg_count = len(messages) if isinstance(messages, list) else 0
    total_chars = sum(len(str(m.get("content", ""))) for m in (messages if isinstance(messages, list) else []))
    _log(f"Single-Model call cat={category}[{def_idx}] model={model} "
         f"api_key={_truncate_key(api_key)} "
         f"messages={msg_count} chars={total_chars} timeout={timeout:.0f}s read_timeout={read_timeout:.0f}s")

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

    # Retry-Loop fuer Timeout/ConnectionError (lokales Modell haengt manchmal)
    max_retries = int(cat.get("retry_on_timeout", 0))
    retry_delay = float(cat.get("retry_delay_seconds", 5))
    _http_timeout = httpx.Timeout(timeout, read=read_timeout)

    response = None
    last_exc: Optional[Exception] = None
    try:
        for attempt in range(1 + max_retries):
            try:
                async with httpx.AsyncClient(timeout=_http_timeout) as client:
                    response = await client.post(
                        api_url, json=payload, headers=_api_headers(api_key),
                    )
                last_exc = None
                break  # Erfolg — kein Retry noetig
            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError, OSError) as exc:
                last_exc = exc
                duration_so_far = time.perf_counter() - started
                exc_type = type(exc).__name__
                if attempt < max_retries:
                    _log(f"Model TIMEOUT/CONN cat={category}[{def_idx}] attempt={attempt+1}/{1+max_retries} "
                         f"duration={duration_so_far:.1f}s type={exc_type}: {_safe_str(exc)}")
                    _log(f"   → Auto-Retry in {retry_delay:.0f}s...")
                    await asyncio.sleep(retry_delay)
                else:
                    _log(f"Model TIMEOUT/CONN cat={category}[{def_idx}] attempt={attempt+1}/{1+max_retries} "
                         f"duration={duration_so_far:.1f}s type={exc_type}: {_safe_str(exc)} — KEINE Retries mehr")
    except asyncio.CancelledError:
        # Client-Disconnect (VS Code bricht Request ab). CancelledError ist eine
        # BaseException — wuerde vom except Exception weiter unten NICHT gefangen.
        # Daher explizit abfangen, aufraeumen, dann weiterpropagieren.
        duration = time.perf_counter() - started
        _finish_active_call(req_id, "cancelled", {"duration_seconds": duration})
        _log(f"Model CANCELLED cat={category}[{def_idx}] duration={duration:.1f}s "
             f"(Client-Disconnect / Task-Abbruch)")
        raise

    if last_exc is not None:
        # Alle Retries erschoepft — Timeout/ConnectionError
        duration = time.perf_counter() - started
        exc_type = type(last_exc).__name__
        exc_msg = _safe_str(last_exc)
        _finish_active_call(req_id, "error", {"duration_seconds": duration, "error": exc_msg,
                                               "attempts": 1 + max_retries})
        _log(f"Model ERROR cat={category}[{def_idx}] duration={duration:.1f}s type={exc_type}: {exc_msg} "
             f"(nach {1+max_retries} Versuchen)")
        _start_cooldown(category, def_idx, duration_override=30.0)
        return {
            "category": category, "def_idx": def_idx, "status": "error",
            "content": _safe_str(f"Model error nach {duration:.0f}s ({exc_type}, {1+max_retries} attempts): {exc_msg}"),
            "duration_seconds": duration, "usage": None,
            "trigger_fallback": True,
        }

    try:
        duration = time.perf_counter() - started
        _finish_active_call(req_id, "done", {"duration_seconds": duration})

        if response.status_code == 200:
            result = response.json()
            message = _extract_choice_message(result)
            content, reasoning_content, tool_calls = _extract_message_parts(result)
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
            # Pruefen ob max_tokens → max_completion_tokens noetig ist
            if "max_completion_tokens" in err_detail.lower() or "max_tokens" in err_detail.lower():
                _log(f"   → 400 deutet auf max_tokens-Problem hin: {err_detail}")
                if "max_tokens" in payload and "max_completion_tokens" not in payload:
                    # Wir versuchen es nochmal mit max_completion_tokens
                    model_name = cat.get("model_name", "")
                    if not _needs_max_completion_tokens(model_name):
                        _log(f"   → Model '{model_name}' wurde nicht als max_completion_tokens-Kandidat erkannt, aber wir versuchen es trotzdem")
                    _log("   → Retry mit max_completion_tokens...")
                    payload_retry = copy.deepcopy(payload)
                    val = payload_retry.pop("max_tokens", None)
                    if val is not None:
                        payload_retry["max_completion_tokens"] = val
                    try:
                        async with httpx.AsyncClient(timeout=_http_timeout) as client:
                            response2 = await client.post(
                                api_url, json=payload_retry, headers=_api_headers(api_key),
                            )
                        if response2.status_code == 200:
                            result = response2.json()
                            message = _extract_choice_message(result)
                            content, reasoning_content, tool_calls = _extract_message_parts(result)
                            _log(f"   → Retry mit max_completion_tokens ERFOLGREICH! Model={model_name}")
                            return {
                                "category": category, "def_idx": def_idx, "status": "ok",
                                "content": content, "message": message,
                                "tool_calls": tool_calls, "reasoning_content": reasoning_content,
                                "duration_seconds": time.perf_counter() - started, "usage": result.get("usage"),
                                "trigger_fallback": False,
                            }
                        else:
                            _log(f"   → Retry mit max_completion_tokens auch fehlgeschlagen: HTTP {response2.status_code}")
                    except Exception as retry_exc:
                        _log(f"   → Retry mit max_completion_tokens Exception: {_safe_str(retry_exc)}")
            # Pruefen ob reasoning_effort mit tools inkompatibel ist (gpt-5 Serie)
            if "reasoning_effort" in err_detail.lower() and "tools" in err_detail.lower():
                _log(f"   → 400 deutet auf reasoning_effort-Tools-Inkompatibilitaet hin: {err_detail}")
                # Kein Check auf "reasoning_effort in payload" noetig — dieses Feld
                # wurde ggf. bereits durch _patch_reasoning_effort_payload entfernt,
                # aber der Fehler zeigt, dass das Backend es noch im Request sieht
                # (z.B. wenn ein anderer Teil des Systems es wieder hinzufuegt oder
                # die Erkennung nicht griff). Einfach sicherheitshalber entfernen + Retry.
                _log("   → Retry ohne reasoning_effort...")
                payload_retry = copy.deepcopy(payload)
                payload_retry.pop("reasoning_effort", None)
                try:
                    async with httpx.AsyncClient(timeout=_http_timeout) as client:
                        response2 = await client.post(
                            api_url, json=payload_retry, headers=_api_headers(api_key),
                        )
                    if response2.status_code == 200:
                        result = response2.json()
                        message = _extract_choice_message(result)
                        content, reasoning_content, tool_calls = _extract_message_parts(result)
                        _log(f"   → Retry ohne reasoning_effort ERFOLGREICH!")
                        return {
                            "category": category, "def_idx": def_idx, "status": "ok",
                            "content": content, "message": message,
                            "tool_calls": tool_calls, "reasoning_content": reasoning_content,
                            "duration_seconds": time.perf_counter() - started, "usage": result.get("usage"),
                            "trigger_fallback": False,
                        }
                    else:
                        _log(f"   → Retry ohne reasoning_effort auch fehlgeschlagen: HTTP {response2.status_code}")
                except Exception as retry_exc2:
                    _log(f"   → Retry ohne reasoning_effort Exception: {_safe_str(retry_exc2)}")
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
        return {
            "category": category, "def_idx": def_idx, "status": "error",
            "content": _safe_str(f"Model error nach {duration:.0f}s ({exc_type}): {exc_msg}"),
            "duration_seconds": duration, "usage": None,
            "trigger_fallback": True,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Co-Worker-Delegation ── ask_coworker Tool-Injection + interner Subagent-Call
# ═══════════════════════════════════════════════════════════════════════════
# Das Hauptmodell (Kategorie "local") bekommt ein synthetisches Tool
# "ask_coworker" angeboten. Ruft es das Tool auf, arbeitet der Proxy den Call
# intern an das Co-Worker-Modell ab (frische, minimale Session — KEINE
# VS-Code-History, KEIN reasoning_content), fuegt das Ergebnis als tool-Message
# ein und ruft das Hauptmodell erneut auf. Fuer VS Code bleibt alles unsichtbar.
#
# Aktivierung NUR bei Kategorie=local UND wenn der Health-Check den Co-Worker
# als erreichbar meldet (Hauptrechner an).

_COWORKER_TOOL_NAME = "ask_coworker"

# System-Guidance, die das Hauptmodell bootstrapt, die Co-Worker-Tools
# ueberhaupt zu nutzen. Wird als system-Message NACH der Client-System-Message
# injiziert (nur wenn Tools injiziert wurden). Lehrt die Parallelitaets-
# Regeln aus dem Fork-Join-Design (Skalier-Lektionen):
#   triviales → selbst, 1 Task pro Datei/Aspekt im selben Turn, 4-8 fuer
#   grosse Aufgahaben, dispatch → eigene Arbeit → collect.
_COWORKER_GUIDANCE_SYSTEM: str = (
    "[PROXY DELEGATION GUIDANCE]\n"
    "You lead a two-machine team: you run on machine A; a Co-Worker model runs "
    "on a SEPARATE machine B (own hardware, ~6-10 parallel sessions). It is "
    "reachable ONLY via the ask_coworker / dispatch_coworker / collect_coworker "
    "tools. Using it well makes the team much faster — this is expected "
    "behavior, not an exception.\n"
    "WHEN to delegate:\n"
    "- Multi-file or multi-aspect work: dispatch ONE task per "
    "file/aspect — all in the SAME turn, not sequentially.\n"
    "- Large explore/review/refactor jobs: fan out 4-8 tasks, then integrate.\n"
    "- Trivial single-file questions: do them yourself.\n"
    "- A long read/search list: hand the files to the Co-Worker and keep only "
    "a small set for yourself.\n"
    "HOW:\n"
    "- Independent sub-tasks → dispatch_coworker (non-blocking, returns "
    "task_id immediately), then do your OWN work, then collect_coworker.\n"
    "- Need the answer before continuing → ask_coworker (blocking).\n"
    "- Patterns to avoid: dispatching one task per turn sequentially; "
    "doing everything yourself while machine B idles; delegating "
    "trivia.\n"
    "The proxy automatically attaches the files from this conversation to "
    "every co-worker call — task/context can stay concise."
)

_COWORKER_TOOL_DEF: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": _COWORKER_TOOL_NAME,
        "description": (
            "Delegate a self-contained blocking sub-task (planning, code "
            "review, brainstorming) to the co-worker model on a SEPARATE "
            "server (DGX Spark, separate hardware, reached ONLY through this "
            "function call). This call BLOCKS until the answer arrives — use "
            "it when you need the result immediately. For fire-and-forget "
            "parallelism use dispatch_coworker instead. The proxy "
            "AUTOMATICALLY appends the file contents from this conversation "
            "(attached files and read/search tool results) to the co-worker's "
            "context — task/context can stay concise and does NOT need to "
            "repeat the files. Returns the co-worker's text answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The self-contained task or question for the co-worker.",
                },
                "context": {
                    "type": "string",
                    "description": "Optional context: code snippets, file excerpts, constraints — everything the co-worker needs.",
                },
            },
            "required": ["task"],
        },
    },
}

# ── Fork-Join Fabric (v3.2): dispatch/collect ──────────────────────────────
_COWORKER_DISPATCH_TOOL_NAME = "dispatch_coworker"
_COWORKER_COLLECT_TOOL_NAME = "collect_coworker"

_COWORKER_DISPATCH_TOOL_DEF: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": _COWORKER_DISPATCH_TOOL_NAME,
        "description": (
            "Fire-and-forget: dispatch a background task to the co-worker "
            "model on the SEPARATE DGX Spark server. Returns IMMEDIATELY with "
            "a task_id (e.g. \"cw_ab12cd\") — the co-worker keeps computing "
            "in the background while you CONTINUE YOUR OWN WORK (call VS-Code "
            "tools, edit files, think). The next client turn will remind you "
            "of running/done tasks until you collect them. Fan-out scaling: "
            "trivial sub-task → just do it yourself; one task per file for "
            "multi-file work; 4-8 parallel tasks for large refactors. Pattern: "
            "dispatch ALL independent sub-tasks in ONE turn (multiple "
            "dispatch_coworker calls), do your own work, then collect_coworker. "
            "The proxy AUTOMATICALLY appends conversation file contents to "
            "each dispatched task. Task must be fully self-contained — the "
            "co-worker has NO access to tools or this conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The self-contained task for the background co-worker.",
                },
                "context": {
                    "type": "string",
                    "description": "Optional context: code snippets, file excerpts, constraints.",
                },
            },
            "required": ["task"],
        },
    },
}

_COWORKER_COLLECT_TOOL_DEF: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": _COWORKER_COLLECT_TOOL_NAME,
        "description": (
            "Join point: collect the results of previously dispatched "
            "background tasks. BLOCKS until the requested tasks are done (or "
            "timeout). Call with no arguments to collect ALL "
            "running/finished tasks. Returns a list of "
            "{task_id, status, result} entries. Typical flow: "
            "dispatch_coworker (several) → do your own work → "
            "collect_coworker → integrate the results into your answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of task_ids (e.g. [\"cw_ab12cd\"]) to collect. Omit to collect ALL tasks.",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Optional max wait in seconds (default 600). Tasks still running after the timeout are reported with status=running.",
                },
            },
            "required": [],
        },
    },
}

# Health-Cache: wird vom periodischen Health-Check + Startup-Probe befuellt
_COWORKER_HEALTH_CACHE: Dict[str, Any] = {
    "reachable": False,
    "checked_at": 0.0,
    "last_error": "noch nicht geprueft",
}


def _coworker_configured() -> bool:
    """True wenn die coworker-Kategorie eine gueltige Definition hat."""
    return bool(_model_defs("coworker"))


async def _probe_coworker() -> bool:
    """Minimaler Ping ans Co-Worker-Modell. Erreichbar = HTTP < 500 (auch 4xx:
    die Maschine lebt; ein 4xx liefert spaeter einen nutzbaren Fehlertext im
    tool-result). Timeout/ConnectError = Maschine aus → unreachable.
    Wird nur aufgerufen, wenn COWORKER_ENABLED aktiv ist."""
    if not COWORKER_ENABLED:
        _COWORKER_HEALTH_CACHE.update({
            "reachable": False,
            "checked_at": time.time(),
            "last_error": "Co-Worker deaktiviert",
        })
        return False
    defs = _model_defs("coworker")
    if not defs:
        _COWORKER_HEALTH_CACHE.update({
            "reachable": False,
            "checked_at": time.time(),
            "last_error": "nicht konfiguriert",
        })
        return False
    cat = defs[0]
    api_url = cat["api_url"].rstrip("/")
    api_key = cat.get("api_key", "")
    timeout = min(float(cat.get("timeout_seconds", 300)), COWORKER_PROBE_TIMEOUT)
    probe_payload = {
        "model": cat["model_name"],
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, read=timeout)) as client:
            resp = await client.post(api_url, json=probe_payload, headers=_api_headers(api_key))
        reachable = resp.status_code < 500
        _COWORKER_HEALTH_CACHE.update({
            "reachable": reachable,
            "checked_at": time.time(),
            "last_error": "" if reachable else f"HTTP {resp.status_code}",
        })
        return reachable
    except Exception as exc:
        _COWORKER_HEALTH_CACHE.update({
            "reachable": False,
            "checked_at": time.time(),
            "last_error": _safe_str(exc),
        })
        return False


async def _coworker_health_loop() -> None:
    """Periodischer Health-Check fuer den Co-Worker (Startup-Probe + Intervall).
    Laeuft NUR wenn COWORKER_ENABLED aktiv ist — bei deaktiviertem Co-Worker
    wird kein Ping ans andere System geschickt."""
    if not COWORKER_ENABLED:
        _COWORKER_HEALTH_CACHE.update({"reachable": False, "last_error": "Co-Worker deaktiviert"})
        return
    try:
        if _coworker_configured():
            await _probe_coworker()
            state = ("erreichbar" if _COWORKER_HEALTH_CACHE.get("reachable")
                     else f"UNREACHABLE ({_COWORKER_HEALTH_CACHE.get('last_error', '?')})")
            _log(f"Co-Worker Health-Check (Startup): {state}")
    except asyncio.CancelledError:
        return
    except Exception:
        pass
    while True:
        try:
            await asyncio.sleep(COWORKER_HEALTH_INTERVAL)
            if not COWORKER_ENABLED:
                # Co-Worker wurde waehrend der Laufzeit deaktiviert → stoppen
                _COWORKER_HEALTH_CACHE.update({"reachable": False, "last_error": "Co-Worker deaktiviert"})
                return
            if not _coworker_configured():
                _COWORKER_HEALTH_CACHE.update({"reachable": False, "last_error": "nicht konfiguriert"})
                continue
            await _probe_coworker()
            state = ("erreichbar" if _COWORKER_HEALTH_CACHE.get("reachable")
                     else f"UNREACHABLE ({_COWORKER_HEALTH_CACHE.get('last_error', '?')})")
            _log(f"Co-Worker Health-Check: {state}")
            # Fork-Join: TTL-Cleanup der Hintergrund-Tasks mit inline ziehen
            if COWORKER_FORK_JOIN and _COWORKER_BG_TASKS:
                _cleanup_bg_tasks()
        except asyncio.CancelledError:
            return
        except Exception:
            pass


def _inject_coworker_tool(payload: Dict[str, Any]) -> bool:
    """Injiziert die Co-Worker-Tools (ask_coworker; bei aktivem Fork-Join
    zusaetzlich dispatch_coworker + collect_coworker) in den Payload — NUR
    wenn aktiviert, konfiguriert und laut Health-Check erreichbar.
    Returns True bei Injection."""
    if not COWORKER_ENABLED:
        return False
    if not _coworker_configured():
        return False
    if not _COWORKER_HEALTH_CACHE.get("reachable", False):
        _log("Co-Worker-Tool nicht injiziert: Health-Check nicht bestanden "
             f"({_COWORKER_HEALTH_CACHE.get('last_error', '?')})")
        return False
    tools = payload.get("tools")
    if not isinstance(tools, list):
        tools = []
        payload["tools"] = tools
    existing = {str((t.get("function") or {}).get("name", "")) for t in tools if isinstance(t, dict)}
    if _COWORKER_TOOL_NAME in existing:
        return True  # bereits vorhanden
    tools.append(copy.deepcopy(_COWORKER_TOOL_DEF))
    if COWORKER_FORK_JOIN:
        if _COWORKER_DISPATCH_TOOL_NAME not in existing:
            tools.append(copy.deepcopy(_COWORKER_DISPATCH_TOOL_DEF))
        if _COWORKER_COLLECT_TOOL_NAME not in existing:
            tools.append(copy.deepcopy(_COWORKER_COLLECT_TOOL_DEF))
    if "tool_choice" not in payload:
        payload["tool_choice"] = "auto"
    # Bootstrap-Guidance: system-Message direkt nach der Client-System-Message.
    # Ohne diesen Hinweis nutzen lokale Modelle die Co-Worker-Tools in der
    # Praxis nicht (Tool-Beschreibungen allein reichen nicht) — getestet in
    # test_inject_coworker_tool_ok / test_inject_coworker_tool_fork_join.
    if COWORKER_TEACH_DELEGATION:
        messages = payload.get("messages")
        if isinstance(messages, list):
            has_guidance = any(
                isinstance(m, dict) and m.get("role") == "system"
                and "[PROXY DELEGATION GUIDANCE]" in str(m.get("content", ""))
                for m in messages)
            if not has_guidance:
                insert_idx = 1 if (messages and isinstance(messages[0], dict)
                                   and messages[0].get("role") == "system") else 0
                messages.insert(insert_idx, {
                    "role": "system",
                    "content": _COWORKER_GUIDANCE_SYSTEM,
                })
    _log(f"Co-Worker-Tools injiziert (Health-OK"
         f"{', Fork-Join' if COWORKER_FORK_JOIN else ''}"
         f"{', Guidance' if COWORKER_TEACH_DELEGATION else ''})")
    return True


def _partition_tool_calls(tool_calls: Optional[List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Teilt tool_calls in 4 Buckets auf:
    (dispatch_calls, collect_calls, ask_calls, other_calls)."""
    dispatch: List[Dict[str, Any]] = []
    collect: List[Dict[str, Any]] = []
    ask: List[Dict[str, Any]] = []
    others: List[Dict[str, Any]] = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function") or {}
        name = str(func.get("name", "")).strip()
        if name == _COWORKER_TOOL_NAME:
            ask.append(tc)
        elif name == _COWORKER_DISPATCH_TOOL_NAME:
            dispatch.append(tc)
        elif name == _COWORKER_COLLECT_TOOL_NAME:
            collect.append(tc)
        else:
            others.append(tc)
    return dispatch, collect, ask, others


def _extract_conversation_files(messages: Optional[List[Dict[str, Any]]],
                                max_chars: int = 0) -> str:
    """Extrahiert Dateiinhalte aus der Chat-History fuer den Co-Worker:
    - VS-Code-Attachments: content-Array-Parts mit type=="file"
      (path + content, auch verschachtelt unter part["file"])
    - Tool-Ergebnisse (read_file/grep_search etc.): role=="tool" messages
    Dedupliziert nach Dateipfad bzw. identischem Text. Returns einen
    formatierten Block oder '' wenn nichts relevantes gefunden wurde.
    max_chars<=0 => unbegrenzt."""
    budget = max_chars if max_chars and max_chars > 0 else 10 ** 9
    blocks: List[str] = []
    used = 0
    truncated = False
    seen_paths: Set[str] = set()
    seen_texts: Set[str] = set()

    def add_block(label: str, text: str) -> None:
        nonlocal used, truncated
        text = (text or "").strip()
        if not text:
            return
        block = f"### {label}\n{text}"
        if used + len(block) > budget:
            truncated = True
            return
        blocks.append(block)
        used += len(block)

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "file":
                    continue
                f = part.get("file")
                if isinstance(f, dict):
                    path = str(f.get("path") or f.get("name") or "").strip()
                    text = str(f.get("content") or f.get("text") or "").strip()
                else:
                    path = str(part.get("path") or part.get("name") or "").strip()
                    text = str(part.get("content") or "").strip()
                if not path or not text:
                    continue
                key = path.lower()
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                add_block(f"Datei: {path}", text)
        elif msg.get("role") == "tool":
            name = str(msg.get("name") or msg.get("tool_name") or "tool").strip()
            text = content if isinstance(content, str) else _normalize_text(content)
            text = text.strip()
            if not text:
                continue
            h = _simple_hash(text)
            if h in seen_texts:
                continue
            seen_texts.add(h)
            add_block(f"Tool-Ergebnis ({name})", text)
    if truncated:
        blocks.append("…[Datei-Kontext gekappt: Budget "
                      f"{budget if budget < 10 ** 9 else 'unbegrenzt'} Zeichen]")
    return "\n\n".join(blocks)


def _build_coworker_body(task: str, context: str,
                         extra_context: Optional[str] = None) -> Dict[str, Any]:
    """Baut eine frische, minimale Session fuer den Co-Worker.
    KEINE VS-Code-History, KEINE tool_calls, KEIN reasoning_content,
    KEIN Hindsight. Nur System-Prompt + eine User-Message (task+context).

    extra_context: automatisch angehaengte Dateiinhalte aus dem Chat
    (VS-Code-Attachments + Tool-Ergebnisse) — damit der Co-Worker auch bei
    komplexen Fragen IMMER alle relevanten Dateiinhalte bekommt, selbst wenn
    das Hauptmodell sie nicht in task/context uebernommen hat."""
    task = (task or "").strip()
    context = (context or "").strip()
    extra = (extra_context or "").strip()
    user_content = f"{task}\n\n## Context\n{context}" if context else task
    if COWORKER_TASK_CAP > 0 and len(user_content) > COWORKER_TASK_CAP:
        user_content = user_content[:COWORKER_TASK_CAP] + "\n…[gekappt]"
    if extra:
        user_content += (
            "\n\n## Dateiinhalte aus dem Chat (vom Proxy automatisch "
            "angehaengt — gehoeren zum aktuellen Kontext)\n" + extra
        )
    messages: List[Dict[str, Any]] = []
    if COWORKER_SYSTEM_PROMPT.strip():
        messages.append({"role": "system", "content": COWORKER_SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_content})
    return {
        "model": "",
        "messages": messages,
        "stream": False,
    }


async def _run_coworker_call(tool_call: Dict[str, Any],
                             extra_context: Optional[str] = None) -> Dict[str, Any]:
    """Fuehrt EINEN ask_coworker-Call intern aus und liefert eine tool-result
    Message. Fehler werden NICHT geworfen, sondern als tool-content zurueck-
    gegeben, damit das Hauptmodell weiterarbeiten kann."""
    tool_call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:12]}"
    args_raw = (tool_call.get("function") or {}).get("arguments", "{}")
    try:
        args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        if not isinstance(args, dict):
            args = {}
    except (json.JSONDecodeError, ValueError, TypeError):
        args = {}
    task = str(args.get("task", "") or "")
    context = str(args.get("context", "") or "")

    started = time.perf_counter()
    body = _build_coworker_body(task, context, extra_context=extra_context)
    # inject_hindsight=False: kein Hindsight-Recall fuer Co-Worker-Calls
    # (vermeidet Kontamination des Co-Worker-Sessions mit Haupt-History)
    result = await _call_single_model(body, "coworker", 0, inject_hindsight=False)
    duration = time.perf_counter() - started

    if result.get("status") == "ok":
        content = result.get("content", "") or ""
        if COWORKER_RESULT_CAP > 0 and len(content) > COWORKER_RESULT_CAP:
            content = content[:COWORKER_RESULT_CAP] + "\n…[gekappt]"
        _log(f"Co-Worker OK duration={duration:.1f}s len={len(content)}")
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": _COWORKER_TOOL_NAME,
            "content": content,
            "reasoning_content": result.get("reasoning_content"),
        }

    # Fehler: Health-Cache invalidieren (naechste Requests injizieren das Tool
    # nicht mehr) + Fehlertext als tool-result.
    _COWORKER_HEALTH_CACHE["reachable"] = False
    err = result.get("content") or "unbekannter Fehler"
    _log(f"Co-Worker FEHLER duration={duration:.1f}s: {_safe_str(err)[:200]}")
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": _COWORKER_TOOL_NAME,
        "content": f"[Co-Worker nicht verfuegbar]\n{err}",
        "reasoning_content": None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Fork-Join Fabric (v3.2) ── Hintergrund-Task-Store + dispatch/collect
# ═══════════════════════════════════════════════════════════════════════════
# dispatch_coworker startet einen Hintergrund-Sub-Session auf dem Co-Worker
# (DGX Spark) und kehrt SOFORT mit einer task_id zurueck. Das Hauptmodell
# kann meanwhile eigene Arbeit erledigen (VS-Code-Tools, Thinking), bis es
# per collect_coworker die Ergebnisse einsammelt (Join). Der Status offener
# Tasks wird jedem neuen Request als kompakte user-Notiz Praefix-artig
# mitgegeben, bis er abgeliefert wurden (delivered=True).

@dataclass
class CoworkerTask:
    task_id: str
    preview: str                       # 60-Zeichen-Task-Vorschau (Status-Zeile)
    status: str = "running"            # running | done | error | expired
    result: Optional[str] = None       # Co-Worker-Antwort (bei done)
    error: Optional[str] = None        # Fehlertext (bei error/expired)
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    delivered: bool = False            # True sobald per collect abgeliefert
    file_context: Optional[str] = None # Datei-Kontext zum Dispatch-Zeitpunkt
    aio_task: Optional[asyncio.Task] = None  # der laufende Hintergrund-Task

    def summary(self) -> Dict[str, Any]:
        """Kompakte JSON-Repraesentation fuer tool-results / Status-Zeilen."""
        d: Dict[str, Any] = {"task_id": self.task_id, "status": self.status}
        if self.status == "done" and self.result is not None:
            d["result"] = self.result
        elif self.status in ("error", "expired"):
            d["error"] = self.error or "unbekannter Fehler"
        return d


# Globaler Task-Store: task_id -> CoworkerTask. Prozess-global, ueberlebt
# einzelne Requests (das ist der Sinn der Sache — Hintergrund-Arbeit ist
# groesser als ein Chat-Turn).
_COWORKER_BG_TASKS: Dict[str, CoworkerTask] = {}
# Semaphore begrenzt die gleichzeitigen Co-Worker-Calls (Schutz der Spark-
# Hardware). Lazy initialisiert, weil asyncio.Semaphore sich an das laufende
# Event-Loop bindet.
_COWORKER_SEMAPHORE: Optional[asyncio.Semaphore] = None


def _coworker_semaphore() -> asyncio.Semaphore:
    global _COWORKER_SEMAPHORE
    if _COWORKER_SEMAPHORE is None:
        _COWORKER_SEMAPHORE = asyncio.Semaphore(max(1, COWORKER_MAX_PARALLEL))
    return _COWORKER_SEMAPHORE


def _coworker_bg_snapshot(undelivered_only: bool = True,
                          max_entries: int = 12) -> List[Dict[str, Any]]:
    """Kompakter Snapshot des Task-Stores fuer Status-Injection / collect."""
    out: List[Dict[str, Any]] = []
    for t in _COWORKER_BG_TASKS.values():
        if undelivered_only and t.delivered:
            continue
        out.append(t.summary())
        if len(out) >= max_entries:
            break
    return out


def _coworker_fit(task: CoworkerTask, result: str) -> str:
    """Kuerzt ein collect-Ergebnis auf COWORKER_RESULT_CAP."""
    if COWORKER_RESULT_CAP > 0 and len(result) > COWORKER_RESULT_CAP:
        return result[:COWORKER_RESULT_CAP] + "\n…[gekappt]"
    return result


async def _run_bg_coworker_task(task: CoworkerTask, tool_call_args: Dict[str, Any]) -> None:
    """Coroutine eines Hintergrund-Tasks: baut die minimale Sub-Session und
    ruft den Co-Worker NON-streaming auf (kein Client-Queue vorhanden — das
    Live-Streaming der content entfaellt, Reasoning faellt weg). Ergebnis
    landet in task.result; Fehler in task.error mit status=error."""
    started = time.perf_counter()
    task_text = str(tool_call_args.get("task", "") or "")
    context_text = str(tool_call_args.get("context", "") or "")
    try:
        body = _build_coworker_body(task_text, context_text, extra_context=task.file_context)
        # Begrenzung der parallelen Co-Worker-Calls passiert HIER, vor dem Call
        async with _coworker_semaphore():
            result = await _call_single_model(body, "coworker", 0, inject_hindsight=False)
        duration = time.perf_counter() - started
        if result.get("status") == "ok":
            content = result.get("content", "") or ""
            if COWORKER_RESULT_CAP > 0 and len(content) > COWORKER_RESULT_CAP:
                content = content[:COWORKER_RESULT_CAP] + "\n…[gekappt]"
            task.result = content
            task.status = "done"
            _log(f"BG-Task {task.task_id} OK duration={duration:.1f}s len={len(content)}")
        else:
            err = result.get("content") or "unbekannter Fehler"
            task.error = _safe_str(err)
            task.status = "error"
            _COWORKER_HEALTH_CACHE["reachable"] = False
            _log(f"BG-Task {task.task_id} FEHLER duration={duration:.1f}s: {task.error[:200]}")
    except asyncio.CancelledError:
        task.status = "expired"
        task.error = "abgebrochen (TTL/Shutdown)"
        raise
    except Exception as exc:
        task.error = _safe_str(exc)
        task.status = "error"
        _log(f"BG-Task {task.task_id} EXCEPTION: {task.error[:200]}")


def _task_preview_from_args(args: Dict[str, Any], max_chars: int = 60) -> str:
    task_text = str(args.get("task", "") or "").strip()
    if len(task_text) <= max_chars:
        return task_text
    return task_text[:max_chars] + "…"


def _register_bg_dispatch(tool_call: Dict[str, Any],
                          files_context: str) -> CoworkerTask:
    """Legt einen neuen Hintergrund-Task an und startet die Coroutine
    (fire-and-forget, non-blocking). Caller prueft cap/limits VOR dem Aufruf."""
    tool_call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:12]}"
    args_raw = (tool_call.get("function") or {}).get("arguments", "{}")
    try:
        args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        if not isinstance(args, dict):
            args = {}
    except (json.JSONDecodeError, ValueError, TypeError):
        args = {}
    task_id = f"cw_{uuid.uuid4().hex[:8]}"
    ct = CoworkerTask(
        task_id=task_id,
        preview=_task_preview_from_args(args),
        file_context=files_context or None,
    )
    aio = asyncio.ensure_future(_run_bg_coworker_task(ct, args))
    ct.aio_task = aio
    _COWORKER_BG_TASKS[task_id] = ct
    _log(f"BG-Dispatch {task_id} (tool_call={tool_call_id}): {ct.preview}")
    return ct


async def _await_bg_tasks(task_ids: Optional[List[str]],
                          timeout_seconds: float = 600.0) -> List[Dict[str, Any]]:
    """Join: wartet (max. timeout) auf die angegebenen Tasks (oder alle
    undelivered), liefert deren summaries. Laeuft-after-timeout wird als
    status=running gemeldet — kein Hard-Abort."""
    # Cleanup zuerst: abgelaufene/abgelieferte raus (auch async gesteuert)
    _cleanup_bg_tasks()
    if task_ids:
        wanted = [str(t).strip() for t in task_ids if str(t).strip()]
        selected: List[CoworkerTask] = []
        missing: List[str] = []
        for tid in wanted:
            ct = _COWORKER_BG_TASKS.get(tid)
            if ct is not None:
                selected.append(ct)
            else:
                missing.append(tid)
    else:
        selected = [t for t in _COWORKER_BG_TASKS.values() if not t.delivered]
        missing = []

    deadline = time.monotonic() + max(0.0, timeout_seconds)
    pending: List[CoworkerTask] = []
    for t in selected:
        if t.status == "running" and t.aio_task is not None and not t.aio_task.done():
            pending.append(t)

    if pending:
        try:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                await asyncio.wait(
                    {t.aio_task for t in pending},   # type: ignore[arg-type]
                    timeout=remaining,
                )
        except Exception as exc:
            _log(f"collect_coworker wait-Fehler: {_safe_str(exc)[:120]}")

    out: List[Dict[str, Any]] = []
    for tid in missing:
        out.append({"task_id": tid, "status": "unknown",
                    "error": "Task-ID nicht gefunden (bereits abgeliefert oder abgelaufen)"})
    for t in selected:
        out.append(t.summary())
        if t.status in ("done", "error", "expired"):
            t.delivered = True
            t.finished_at = t.finished_at or time.time()
        # running Tasks bleiben undelivered → Status-Injection erinnert daran
    return out


def _cleanup_bg_tasks() -> None:
    """Raumt den Task-Store auf:
    - laufende Tasks aelter als COWORKER_BG_TTL → canceln (Status=expired)
    - abgelieferte Tasks mit finished_at aelter als 60s → entfernen
    - Store auf MAX_PARALLEL*4 Einheiten begrenzen (aeltete zuerst raus)"""
    now = time.time()
    if COWORKER_BG_TTL > 0:
        for t in list(_COWORKER_BG_TASKS.values()):
            if t.status == "running" and (now - t.created_at) > COWORKER_BG_TTL:
                _log(f"BG-Task {t.task_id} TTL abgelaufen → cancel")
                t.status = "expired"
                t.error = f"Task nach {int(now - t.created_at)}s abgelaufen (TTL)"
                t.finished_at = now
                if t.aio_task is not None and not t.aio_task.done():
                    t.aio_task.cancel()
    # Abgelieferte nach 60s entfernen
    for t in list(_COWORKER_BG_TASKS.values()):
        if t.delivered and t.finished_at is not None and (now - t.finished_at) > 60:
            del _COWORKER_BG_TASKS[t.task_id]
    # Harte Grenze: aelteste raus
    max_entries = max(8, COWORKER_MAX_PARALLEL * 4)
    if len(_COWORKER_BG_TASKS) > max_entries:
        by_age = sorted(_COWORKER_BG_TASKS.values(), key=lambda t: t.created_at)
        for t in by_age[: len(_COWORKER_BG_TASKS) - max_entries]:
            if t.status != "running":
                _COWORKER_BG_TASKS.pop(t.task_id, None)


def _coworker_status_line() -> Optional[str]:
    """Baut die kompakte Status-Notiz fuer undelivered Tasks ( None = keine
    offenen Tasks). Format:
    [Proxy] 2 Co-Worker Hintergrund-Tasks offen:
    - ✅ cw_ab12cd: review proxy.py lines 100-200…
    - ⏳ cw_ef34gh: refactor tools/auth.py…
    """
    entries = [t for t in _COWORKER_BG_TASKS.values() if not t.delivered]
    if not entries:
        return None
    lines = [f"[Proxy] {len(entries)} Co-Worker Hintergrund-Task(s) aktiv:"]
    for t in entries[:8]:
        icon = {"done": "✅", "error": "❌", "expired": "⏱️"}.get(t.status, "⏳")
        detail = ""
        if t.status in ("error", "expired"):
            detail = f" — {t.error[:80]}" if t.error else ""
        lines.append(f"- {icon} {t.task_id}: {t.preview}{detail}")
    if len(entries) > 8:
        lines.append(f"- … und {len(entries) - 8} weitere")
    lines.append("Rufe collect_coworker auf, um Ergebnisse abzuholen.")
    return "\n".join(lines)


async def _delegation_loop(body: Dict[str, Any], category: str,
                           force_start_idx: Optional[int] = None) -> Dict[str, Any]:
    """Hauptmodell aufrufen, Co-Worker-Calls intern abarbeiten, erneut
    aufrufen — bis keine Delegation mehr gewuenscht wird oder das
    max_delegations-Limit erreicht ist. Returns das finale outcome (Format
    wie _call_model_with_fallbacks).

    Wichtig: ask_coworker wird NUR intern abgearbeitet, wenn es der EINZIGE
    Tool-Typ im Turn ist. Gemischte Turns (ask_coworker + VS-Code-Tools) werden
    mit einem Hinweis beantwortet und neu aufgerufen — die History-
    Rekonstruktion fuer gemischte Turns ist zu fragil.

    Fork-Join (v3.2): dispatch_coworker ist NON-blocking — der Call kehrt
    sofort mit einer task_id zurueck und darf auch gemischt mit VS-Code-Tools
    auftreten (das Ergebnis ist history-unabhaengig, der Store ist die
    Truth). collect_coworker ist der Join und blockt, bis die Ergebnisse
    da sind (oder Timeout)."""
    rounds = 0
    dispatch_count = 0
    # Datei-Kontext aus der Chat-History einmalig extrahieren — der Co-Worker
    # bekommt IMMER die relevanten Dateiinhalte, auch wenn das Hauptmodell sie
    # nicht in task/context uebernommen hat.
    files_context = _extract_conversation_files(body.get("messages"), COWORKER_FILES_CAP)
    if files_context:
        _log(f"Co-Worker-Delegation: {len(files_context)} chars Datei-Kontext "
             f"automatisch angehaengt")
    while True:
        outcome = await _call_model_with_fallbacks(body, category, force_start_idx=force_start_idx)
        result = outcome.get("result", {})
        tool_calls = result.get("tool_calls")
        if not tool_calls:
            return outcome  # fertig: keine Tool-Calls
        dispatch_calls, collect_calls, ask_calls, other_calls = _partition_tool_calls(tool_calls)
        coworker_calls = ask_calls
        if not coworker_calls and not dispatch_calls and not collect_calls:
            return outcome  # nur VS-Code-Tools → normal durchreichen

        # ── Fork: dispatches ausfuehren (non-blocking), mini-results sofort ──
        if dispatch_calls:
            if dispatch_count + len(dispatch_calls) > COWORKER_DISPATCH_CAP:
                _log(f"Dispatch-Cap erreicht ({COWORKER_DISPATCH_CAP}/Request) — "
                     f"weitere dispatches blockt")
                msgs = body.get("messages", [])
                msgs.append({"role": "user", "content":
                    f"[Proxy] Dispatch-Limit erreicht ({COWORKER_DISPATCH_CAP} pro Request). "
                    "Sammle zuerst mit collect_coworker oder arbeite selbst weiter."})
                outcome = await _call_model_with_fallbacks(body, category, force_start_idx=force_start_idx)
                result = outcome.get("result", {})
                dispatch_calls, collect_calls, ask_calls, other_calls = _partition_tool_calls(
                    result.get("tool_calls") or [])
                coworker_calls = ask_calls
                if not dispatch_calls and not collect_calls and not coworker_calls:
                    # Keine Co-Worker-Aktion mehr → Payload normal zurueck
                    return outcome
                if dispatch_calls:
                    # will trotzdem weiter dispatchen → hart stoppen, Rest durch
                    final_tcs = [tc for tc in (result.get("tool_calls") or [])
                                 if (tc.get("function") or {}).get("name") not in
                                 (_COWORKER_TOOL_NAME, _COWORKER_DISPATCH_TOOL_NAME,
                                  _COWORKER_COLLECT_TOOL_NAME)]
                    result["tool_calls"] = final_tcs or None
                    if not final_tcs:
                        result["content"] = ((result.get("content") or "") +
                                             "\n\n[Proxy] Dispatch-Limit erreicht.")
                    return outcome
                dispatch_count = COWORKER_DISPATCH_CAP  # Limit hart erreicht
            else:
                dispatch_count += len(dispatch_calls)
                msgs = body.get("messages", [])
                dispatch_norm = _normalize_tool_calls(dispatch_calls) or dispatch_calls
                msgs.append({"role": "assistant", "content": None,
                             "tool_calls": dispatch_norm})
                mini_results: List[Dict[str, Any]] = []
                for tc in dispatch_norm:
                    ct = _register_bg_dispatch(tc, files_context)
                    mini_results.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                        "name": _COWORKER_DISPATCH_TOOL_NAME,
                        "content": json.dumps(
                            {"task_id": ct.task_id, "status": "dispatched"},
                            ensure_ascii=False),
                    })
                msgs.extend(mini_results)
                # Dispatch ist non-blocking: VS-Code-Tools im selben Turn
                # werden JETZT durchgereicht (mixed-turn unlock) — nur die
                # dispatches sind intern beantwortet. collect_coworker wird
                # NIE durchgereicht (Client kennt das Tool nicht) — es faellt
                # in den Join-Block unten.
                if other_calls:
                    result["tool_calls"] = other_calls
                    return outcome
                if collect_calls:
                    pass  # Join-Block unten uebernimmt (nach dem Fork)
                else:
                    continue  # nur dispatches → nächste Runde

        rounds += 1
        if rounds > COWORKER_MAX_DELEGATIONS:
            _log(f"Co-Worker-Delegation-Limit erreicht ({COWORKER_MAX_DELEGATIONS} Runden)")
            msgs = body.get("messages", [])
            msgs.append({"role": "user", "content":
                "[Proxy] Co-Worker-Delegation-Limit erreicht. Beantworte die Aufgabe "
                "jetzt direkt, ohne ask_coworker erneut aufzurufen."})
            final_outcome = await _call_model_with_fallbacks(body, category, force_start_idx=force_start_idx)
            final_result = final_outcome.get("result", {})
            final_tcs = final_result.get("tool_calls")
            if final_tcs:
                d2, c2, cw2, other2 = _partition_tool_calls(final_tcs)
                if (cw2 or c2) and not other2:
                    # Modell will trotzdem weiter delegieren/sammeln → hart stoppen
                    final_result["tool_calls"] = None
                    final_result["content"] = (
                        (final_result.get("content") or "") + "\n\n[Proxy] "
                        "Co-Worker-Delegation-Limit erreicht — Aufgabe ohne "
                        "Co-Worker-Unterstuetzung beantwortet."
                    )
                elif (cw2 or c2) and other2:
                    # Nur coworker-Calls entfernen, VS-Code-Tools durchreichen
                    final_result["tool_calls"] = other2
            return final_outcome

        # ── Join: collect_coworker blockt bis Ergebnisse da sind ──
        if collect_calls:
            msgs = body.get("messages", [])
            msgs.append({"role": "assistant", "content": None,
                         "tool_calls": _normalize_tool_calls(collect_calls) or collect_calls})
            collect_results: List[Dict[str, Any]] = []
            for tc in collect_calls:
                args_raw = (tc.get("function") or {}).get("arguments", "{}")
                try:
                    cargs = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                    if not isinstance(cargs, dict):
                        cargs = {}
                except (json.JSONDecodeError, ValueError, TypeError):
                    cargs = {}
                summaries = await _await_bg_tasks(
                    cargs.get("task_ids"),
                    timeout_seconds=float(cargs.get("timeout_seconds", 600) or 600),
                )
                collect_results.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                    "name": _COWORKER_COLLECT_TOOL_NAME,
                    "content": json.dumps(summaries, ensure_ascii=False, indent=2),
                })
            msgs.extend(collect_results)
            if ask_calls and not other_calls:
                # ask + collect im selben Turn: erst sammeln, dann asks parallel —
                # pragmatisch: asks ebenfalls abarbeiten (intern) und zusammen-
                # fassen. History bleibt konsistent (assistant mit beiden calls).
                msgs.append({"role": "user", "content":
                    "[Proxy-Hinweis] ask_coworker und collect_coworker im selben Turn "
                    "werden sequenziell abgearbeitet (erst collect, dann ask)."})
                # Achtung: assistant-turn für ask_calls fehlt hier bewusst NICHT —
                # der folgende Code hängt ihn an (siehe unten, gemeinsamer Pfad).
                pass
            else:
                continue  # collect done → nächste Runde (Modell verarbeitet)

        if other_calls and (coworker_calls or collect_calls):
            # Gemischter Turn: Hinweis injizieren + neu aufrufen. Die bisherigen
            # assistant-tool_calls bleiben in der History ohne Ergebnisse — das
            # Modell formuliert den naechsten Schritt neu (gleiches Muster wie
            # der fruehere Loop-Retry).
            _log(f"Gemischter Turn (ask_coworker + {len(other_calls)} andere Tools) "
                 f"— Hinweis injizieren")
            msgs = body.get("messages", [])
            msgs.append({"role": "user", "content":
                "[Proxy-Hinweis] Rufe ask_coworker nicht zusammen mit anderen Tools "
                "auf. Fuehre entweder die anderen Tools aus ODER rufe ask_coworker "
                "in einem separaten Turn allein auf."})
            continue

        # Reiner Co-Worker-Turn: intern abarbeiten (parallel, Hardware getrennt)
        if coworker_calls:
            coworker_calls_norm = _normalize_tool_calls(coworker_calls) or coworker_calls
            msgs = body.get("messages", [])
            msgs.append({"role": "assistant", "content": None, "tool_calls": coworker_calls_norm})
            results = await asyncio.gather(
                *[_run_coworker_call(tc, extra_context=files_context)
                  for tc in coworker_calls_norm])
            # reasoning_content nicht an das Hauptmodell schicken (nur Reporting)
            for r in results:
                r.pop("reasoning_content", None)
            msgs.extend(results)
            _log(f"Co-Worker-Delegation Runde {rounds}: {len(coworker_calls_norm)} Call(s) abgearbeitet")

        # Erinnerung injizieren: bei langem Context vergisst das Modell,
        # dass es ask_coworker hat. Als user-Message (NICHT system — lokale
        # Modelle erwarten system nur am Anfang der Konversation und geraten
        # bei system mitten im Tool-Loop in Rollen-Verwirrung).
        if rounds > 1:
            reminder = (
                "[Proxy-Hinweis] Erinnerung: Du (das Hauptmodell) hast Zugriff auf "
                "die Co-Worker-Tools 'ask_coworker' (blockierend) sowie "
                "'dispatch_coworker' + 'collect_coworker' (nicht-blockierend, "
                "fuer Parallelitaet). Der Co-Worker ist ein VÖLLIG ANDERES "
                "Modell auf einem ANDEREN Server (eigene Base-URL, separate "
                "Hardware) — er ist ausschließlich über diese Funktionsaufrufe "
                "erreichbar. Wenn du Teilaufgaben an einen Subagenten delegieren "
                "willst (Planung, Code-Review, Parallelisierung), rufe die Tools "
                "mit task/context auf — mehrere dispatch_coworker gerne im "
                "gleichen Turn. Simuliere keinen Subagenten selbst und "
                "beantworte delegierbare Teilaufgaben nicht als imaginären "
                "Subagenten in deinem Text."
            )
            msgs.append({"role": "user", "content": reminder})


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
    # Modelle, die trigger_fallback=False lieferten (z.B. Gemini bei Tool-Continuations):
    # in Runde 2 ueberspringen — kennen das Format grundsaetzlich nicht.
    _skip_r2: Set[int] = set()

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
            # Runde 2: Modelle mit known incompatibility überspringen
            if round_num == 2 and curr_idx in _skip_r2:
                _log(f"Fallback skip R2: {category}[{curr_idx}] = known-incompatible (trigger_fallback=False)")
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

            # trigger_fallback=False (z.B. 400): kein Cooldown, aber trotzdem
            # naechsten Fallback probieren — ein anderes Modell kann andere
            # Parameter-Erwartungen haben und erfolgreich sein.
            if not result.get("trigger_fallback", True):
                _log(f"   → trigger_fallback=False, probiere trotzdem naechsten Fallback")
                _skip_r2.add(curr_idx)  # Runde 2 ueberspringen — Modell kann Format nicht
                continue

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

    message: Dict[str, Any] = {"role": "assistant", "content": combined_text}
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
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


# Maximale Lebensdauer eines aktiven Calls (Sekunden). Darueber wird der
# Call als 'stale' (verwaist) markiert und automatisch bereinigt.
# Default: 15 Minuten — praeventiert, dass abgebrochene Tasks (Client-Disconnect)
# den Heartbeat ewig laufen lassen.
_ACTIVE_CALL_HARD_TIMEOUT: int = int(os.getenv("ACTIVE_CALL_HARD_TIMEOUT", "900"))


def _register_active_call(call_id: str, info: Dict[str, Any]) -> None:
    info = dict(info)
    info["started_at"] = time.time()
    info["started_iso"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _ACTIVE_CALLS[call_id] = info
    asyncio.ensure_future(_active_call_heartbeat(call_id))


async def _active_call_heartbeat(call_id: str) -> None:
    """Heartbeat fuer aktive Calls. Logt alle 30s die verstrichene Zeit.
    Self-Cleaning: Calls die laenger als _ACTIVE_CALL_HARD_TIMEOUT leben
    (z.B. weil der Client abgebrochen ist und CancelledError nicht richtig
    aufgeraeumt hat) werden automatisch als stale entfernt.
    """
    while call_id in _ACTIVE_CALLS:
        await asyncio.sleep(30)
        info = _ACTIVE_CALLS.get(call_id)
        if not info:
            return
        elapsed = time.time() - info.get("started_at", time.time())
        if elapsed > _ACTIVE_CALL_HARD_TIMEOUT:
            _log(f"[{call_id}] STALE CALL: {elapsed:.0f}s alt — automatisch bereinigt "
                 f"(category={info.get('agent_key','?')}, model={info.get('model','?')}). "
                 f"Client-Disconnect oder fehlende Bereinigung vermutet.")
            _ACTIVE_CALLS.pop(call_id, None)
            return
        _log(f"[{call_id}] aktiver Call seit {elapsed:.0f}s "
             f"(category={info.get('agent_key','?')}, model={info.get('model','?')})")


def _finish_active_call(call_id: str, status: str = "done",
                          extra: Optional[Dict[str, Any]] = None) -> None:
    info = _ACTIVE_CALLS.pop(call_id, None)
    if not info:
        return
    elapsed = time.time() - info.get("started_at", time.time())
    _log(f"[{call_id}] Call beendet nach {elapsed:.0f}s ({status})")


def _purge_stale_active_calls() -> int:
    """Entfernt alle aktiven Calls, die das Hard-Timeout ueberschritten haben.
    Wird beim Startup (Persistenz ueber Restart) und periodisch aufgerufen.
    Gibt die Anzahl der entfernten Calls zurueck.
    """
    now = time.time()
    stale_ids: List[str] = []
    for cid, info in _ACTIVE_CALLS.items():
        elapsed = now - info.get("started_at", now)
        if elapsed > _ACTIVE_CALL_HARD_TIMEOUT:
            stale_ids.append(cid)
    for cid in stale_ids:
        info = _ACTIVE_CALLS.pop(cid, None)
        if info:
            elapsed = now - info.get("started_at", now)
            _log(f"[{cid}] STALE CALL purge: {elapsed:.0f}s alt "
                 f"(category={info.get('agent_key','?')}, model={info.get('model','?')})")
    if stale_ids:
        _log(f"Startup-Sweep: {len(stale_ids)} stale active call(s) bereinigt")
    return len(stale_ids)


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

    for key in ("local", "coworker", "light", "strong", "vision"):
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
                # Bei 400: max_tokens ggf. durch max_completion_tokens ersetzen
                if r.status_code == 400:
                    try:
                        err_body = r.json()
                        err_msg = str(err_body.get("error", {}).get("message", "")) if isinstance(err_body.get("error"), dict) else str(err_body.get("error", ""))
                    except Exception:
                        err_msg = ""
                    if "max_completion_tokens" in err_msg.lower() or "max_tokens" in err_msg.lower():
                        _log(f"   {key}[{i}]: max_tokens-Problem erkannt, versuche max_completion_tokens...")
                        try:
                            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=3.0)) as hc:
                                r2 = await hc.post(
                                    api_url,
                                    json={"model": cat.get("model_name", ""),
                                          "messages": [{"role": "user", "content": "ping"}],
                                          "max_completion_tokens": 1},
                                    headers=_api_headers(str(cat.get("api_key", ""))),
                                )
                            if r2.status_code in (200, 201):
                                _log(f"   {key}[{i}]: OK (mit max_completion_tokens) ({cat.get('model_name')})")
                            else:
                                err = _parse_error(r2) or f"HTTP {r2.status_code}"
                                _log(f"   {key}[{i}]: auch mit max_completion_tokens fehlgeschlagen: {err}")
                        except Exception as exc2:
                            _log(f"   {key}[{i}]: Retry-Fehler - {type(exc2).__name__}: {_safe_str(exc2)}")
            except Exception as exc:
                _log(f"   {key}[{i}]: nicht erreichbar - {type(exc).__name__}: {_safe_str(exc)}")

    _log("Health-Checks abgeschlossen")


@app.on_event("startup")
async def _startup_event() -> None:
    _log(f"LocalProxy v3.0 starting on port {PROXY_PORT}")
    _log(f"   Default:  {DEFAULT_CATEGORY}")
    for key in ("local", "coworker", "light", "strong", "vision"):
        defs = _model_defs(key)
        if defs:
            for i, d in enumerate(defs):
                _log(f"   {key}[{i}]: {d.get('model_name','?')} @ {d.get('api_url','?')}")
    _log(f"   Memory:  {'Qdrant' if _hindsight._use_qdrant else 'JSONL' if HINDSIGHT_ENABLED else 'disabled'}")
    _log(f"   Auth:    {'enabled' if PROXY_AUTH_ENABLED else 'disabled'}")
    _log(f"   Debug:   {'enabled' if DEBUG_ENABLED else 'disabled'}")
    _log(f"   Tool-Cap: {TOOL_RESULT_CAP if TOOL_RESULT_CAP > 0 else 'off'}")

    # Stale active calls bereinigen (Vorfahre aus vorherigem Prozess-Lauf,
    # falls _ACTIVE_CALLS persistiert wurde oder der Prozess neu startete)
    _purge_stale_active_calls()

    # Co-Worker-Health-Loop starten (Startup-Probe + periodischer Check)
    asyncio.ensure_future(_coworker_health_loop())

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


def _find_category_in_messages(messages: Sequence[Dict[str, Any]], default: str) -> str:
    """Durchsucht ALLE User-Nachrichten (von hinten nach vorne) nach --flag.
    Session-spezifisch: Jeder Request traegt seine Historie selbst → kein globaler State.
    """
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        text = _message_text(msg)
        _, flag_cat, _ = _extract_model_flag(text)
        if flag_cat:
            return flag_cat
    return default


def _find_best_idx(category: str, preferred_idx: int) -> int:
    """Findet den besten Start-Index: preferred_idx wenn nicht im Cooldown,
    sonst der erste nicht-cooldown Index. Updated _CATEGORY_ACTIVE_IDX."""
    defs = _model_defs(category)
    if not defs:
        return 0
    if preferred_idx < len(defs) and not _is_in_cooldown(category, preferred_idx):
        return preferred_idx
    for i in range(len(defs)):
        if not _is_in_cooldown(category, i):
            if i != preferred_idx:
                _log(f"   → preferred idx {preferred_idx} im Cooldown, weiche zu idx {i} aus")
            _CATEGORY_ACTIVE_IDX[category] = i
            return i
    # Alle im Cooldown → preferred nehmen (Fallback-Chain probiert dann trotzdem alle)
    return preferred_idx


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

    if flag_category:
        # Flag in der aktuellen Message → explizite Wahl
        category = flag_category
    elif _is_tool_continuation(msgs):
        # Tool-Continuation: Flag in der Nachrichten-Historie suchen
        category = _find_category_in_messages(msgs, DEFAULT_CATEGORY)
    else:
        # Neuer Request ohne Flag: Default
        category = DEFAULT_CATEGORY

    # Slot-Nummer in 0-basierten Index umrechnen (--light 2 → Idx 1)
    force_start_idx: Optional[int] = None
    if flag_slot is not None and category not in ("local", "coworker"):
        force_start_idx = flag_slot - 1  # Slot 1=Idx 0, Slot 2=Idx 1, Slot 3=Idx 2
        defs_validate = _model_defs(category)
        if force_start_idx >= len(defs_validate) or force_start_idx < 0:
            _log(f"Slot {flag_slot} ungültig für {category} (nur {len(defs_validate)} Slots), verwende active_idx")
            force_start_idx = None

    # Aktiven Index ermitteln + Cooldown-Check (nach Rate-Limit automatisch naechsten Slot)
    if force_start_idx is not None:
        # Expliziter Slot per --flag → keinen Cooldown-Skip (User-Wunsch respektieren)
        active_idx = force_start_idx
    else:
        preferred_idx = _CATEGORY_ACTIVE_IDX.get(category, 0)
        active_idx = _find_best_idx(category, preferred_idx)
        if active_idx != preferred_idx:
            _CATEGORY_ACTIVE_IDX[category] = active_idx
            _log(f"   → active_idx auf {active_idx} aktualisiert (Cooldown-Umgehung)")

    _strip_model_flags_from_messages(msgs)

    # Log aktiven Index für die Kategorie
    defs = _model_defs(category)
    active_model = defs[active_idx]["model_name"] if defs and active_idx < len(defs) else "?"

    # ── Loop-Detection + Intervention auf REQUEST-Ebene ──
    # Laguna-S-2.1: derzeit auskommentiert (Laguna nicht mehr im Einsatz).
    # loop_intervened = False
    # loop_reasons: List[str] = []
    # if _is_laguna_model(active_model) and isinstance(msgs, list):
    #     read_hit = _detect_read_loop_inplace(msgs, "Passthrough", category=category, model_name=active_model)
    #     search_hit = _detect_search_loop_inplace(msgs, "Passthrough", category=category, model_name=active_model)
    #     generic_hit = _detect_generic_tool_loop_inplace(msgs, "Passthrough", category=category, model_name=active_model)
    #     loop_intervened = bool(read_hit or search_hit or generic_hit)
    #     if loop_intervened:
    #         last_iv = msgs[-1].get("content", "") if isinstance(msgs[-1], dict) else ""
    #         loop_reasons.append(str(last_iv)[:200])
    #         _log(f"Loop-Intervention: history truncated + appended "
    #              f"(read={read_hit}, search={search_hit}, generic={generic_hit})")

    # Co-Worker-Health-Cache frisch halten (nicht-blockierend): Wenn der Cache
    # aelter als das Health-Intervall ist, Probe im Hintergrund anstossen, damit
    # der naechste Request den aktuellen Zustand sieht.
    if category == "local" and COWORKER_ENABLED and _coworker_configured():
        if time.time() - float(_COWORKER_HEALTH_CACHE.get("checked_at", 0.0)) > COWORKER_HEALTH_INTERVAL:
            asyncio.ensure_future(_probe_coworker())

    # Fork-Join: Status offener Hintergrund-Tasks als kompakte user-Notiz
    # ans Ende der History haengen (nach Kategorie-Detection — beeinflusst
    # weder Flag-Extraktion noch Tool-Continuation-Erkennung).
    if (category == "local" and COWORKER_ENABLED and COWORKER_FORK_JOIN
            and _COWORKER_BG_TASKS):
        status = _coworker_status_line()
        if status:
            msgs.append({"role": "user", "content": status})
            _log("Fork-Join: Status-Notiz fuer offene BG-Tasks injiziert")

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

    # Non-Streaming: Fallback-Chain nutzen (mit Co-Worker-Delegation bei local)
    if category == "local":
        outcome = await _delegation_loop(body, category, force_start_idx=force_start_idx)
    else:
        outcome = await _call_model_with_fallbacks(body, category, force_start_idx=force_start_idx)
    result = outcome.get("result", {})
    content = result.get("content", "") or ""
    used_model = outcome.get("used_model", active_model)

    # Bei totalem Fehlschlag: Fehler-Text mit Prefx
    if outcome.get("all_failed"):
        content = f"[Proxy: ALLE Fallbacks fehlgeschlagen]\n{content}"

    tool_calls = result.get("tool_calls")
    # Response-Level Loop-Retry (non-streaming): Laguna-spezifisch, derzeit
    # auskommentiert (Laguna nicht mehr im Einsatz).
    # if tool_calls and _RESPONSE_LOOP_THRESHOLD > 0 and _is_laguna_model(used_model):
    #     resp_reasons, blocked_names = _detect_response_loop(body, tool_calls, category=category, model_name=used_model)
    #     if resp_reasons:
    #         ... (siehe Git-History)

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
    """2-Pass-Streaming: Erst Backend-Call mit Fallback-Loop, dann OpenAI-SSE an Copilot.

    Waehrend der Backend-Calls werden regelmaessig SSE-Keepalive-Kommentare
    gesendet, damit der Client (VS Code Copilot, 300s HTTP-Timeout) die
    Verbindung nicht als tot betrachtet und abbricht.
    """
    _KEEPALIVE_INTERVAL = 15  # Sekunden zwischen Keepalive-Kommentaren

    # Kategorie=local: Live-Streaming mit Co-Worker-Stream-Inject (kein 2-Pass).
    # reasoning_content (Thinking) und content fliessen live an VS Code durch,
    # damit der User sofort sieht, dass das Modell arbeitet (kein Timeout-Gefuehl).
    # Delegation + Co-Worker-Antwort werden per Stream-Inject sichtbar gemacht.
    if category == "local":
        async for sse in _stream_local_events(body, category, force_start_idx):
            yield sse
        return

    # Backend-Call als Task starten, damit wir nebenbei Keepalives senden koennen.
    # (Nur fuer Nicht-Local-Kategorien; local nutzt _stream_local_events oben.)
    backend_task = asyncio.ensure_future(
        _call_model_with_fallbacks(body, category, force_start_idx=force_start_idx)
    )

    try:
        # Waehrend der Backend laeuft: periodisch SSE-Keepalive senden
        while not backend_task.done():
            await asyncio.sleep(_KEEPALIVE_INTERVAL)
            if not backend_task.done():
                # SSE-Kommentar (wird von Clients ignoriert, haelt Verbindung alive)
                yield ": keepalive\n\n"
        # Task-Ergebnis abholen (propagiert ggf. CancelledError/Exception)
        outcome = backend_task.result()
    except (asyncio.CancelledError, GeneratorExit):
        backend_task.cancel()
        raise
    except Exception:
        if not backend_task.done():
            backend_task.cancel()
        raise

    result = outcome.get("result", {})
    content = result.get("content", "") or ""
    used_model = outcome.get("used_model", category)

    asyncio.ensure_future(_hindsight.retain_async(body, content))

    tool_calls = result.get("tool_calls")
    reasoning_content = result.get("reasoning_content")

    if outcome.get("all_failed"):
        content = f"[Proxy: ALLE Fallbacks fehlgeschlagen]\n{content}"

    # Response-Level Loop-Retry (streaming): Laguna-spezifisch, derzeit
    # auskommentiert (Laguna nicht mehr im Einsatz).
    if tool_calls:
        tool_calls = _normalize_tool_calls(tool_calls) or tool_calls

        # if _RESPONSE_LOOP_THRESHOLD > 0 and _is_laguna_model(used_model):
        #     resp_reasons, blocked_names = _detect_response_loop(body, tool_calls, category=category, model_name=used_model)
        #     if resp_reasons:
        #         ... (siehe Git-History — Laguna-Retry deaktiviert)

    if tool_calls:
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


# ═══════════════════════════════════════════════════════════════════════════
# Live-Streaming fuer Kategorie=local ── inkl. Co-Worker-Stream-Inject
# ═══════════════════════════════════════════════════════════════════════════
# Lokale Modelle sind oft langsam. Statt (wie bei Cloud-Kategorien) erst auf
# die komplette Antwort zu warten (2-Pass) wird hier das Backend mit
# stream=True aufgerufen:
#
#   * reasoning_content (Thinking) und content fliessen LIVE an VS Code durch —
#     der User sieht sofort, dass das Modell arbeitet, und es entstehen keine
#     leeren Phasen, die in Timeouts laufen wuerden.
#   * Keepalive-Kommentare halten die Verbindung auch dann am Leben, wenn das
#     lokale Modell zwischen zwei Tokens lange braucht oder der Co-Worker
#     arbeitet.
#   * ask_coworker-Calls werden intern abgearbeitet; per Stream-Inject wird
#     sichtbar gemacht: (1) dass delegiert wird, (2) die Co-Worker-Antwort,
#     (3) danach, wie das Hauptmodell die Antwort verarbeitet (live gestreamt).

_STREAM_KEEPALIVE_INTERVAL = 15  # Sekunden zwischen Keepalive-Kommentaren


def _build_forward_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Baut die tool_calls-Chunks fuer VS Code (OpenAI-Stream-Format).

    Erwartet tool_calls im normalisierten Format (id/function.name/arguments)
    und liefert die Liste, die als delta.tool_calls an Copilot geht.
    """
    first_tcs: List[Dict[str, Any]] = []
    for i, tc in enumerate(tool_calls):
        func = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
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
    return first_tcs


def _accumulate_stream_tool_calls(state: Dict[str, Any], deltas: List[Dict[str, Any]]) -> None:
    """Akkumuliert streamed Tool-Call-Deltas (OpenAI-Format) in state['tool_calls'].

    Waehrend des Streams ist state['tool_calls'] ein Dict {index: call}, weil
    Argumente ueber mehrere Chunks hinweg zusammengesetzt werden. Am Turn-Ende
    konvertiert _finalize_stream_tool_calls das Dict in eine Liste.
    """
    acc = state.get("tool_calls")
    if not isinstance(acc, dict):
        acc = {}
        state["tool_calls"] = acc
    for tc in deltas:
        if not isinstance(tc, dict):
            continue
        idx = tc.get("index", 0)
        entry = acc.get(idx)
        if not isinstance(entry, dict):
            entry = {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
            acc[idx] = entry
        if tc.get("id"):
            entry["id"] = tc["id"]
        fn = tc.get("function")
        if isinstance(fn, dict):
            if fn.get("name"):
                entry["function"]["name"] = str(fn["name"])
            args = fn.get("arguments")
            if args is not None:
                entry["function"]["arguments"] = (entry["function"]["arguments"] or "") + str(args)


def _finalize_stream_tool_calls(state: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Konvertiert den Tool-Call-Akkumulator in eine Liste im erwarteten Format.
    Returns None wenn keine (vollstaendigen) Tool-Calls vorhanden."""
    acc = state.get("tool_calls")
    if not isinstance(acc, dict) or not acc:
        state["tool_calls"] = []
        return None
    calls: List[Dict[str, Any]] = []
    for idx in sorted(acc.keys()):
        e = acc[idx]
        if not isinstance(e, dict):
            continue
        fn = e.get("function") if isinstance(e.get("function"), dict) else {}
        name = str(fn.get("name", "")).strip()
        if not name:
            continue
        args = fn.get("arguments", "")
        calls.append({
            "index": idx,
            "id": e.get("id") or f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {"name": name, "arguments": args or "{}"},
        })
    state["tool_calls"] = calls
    return calls or None


def _coworker_task_preview(coworker_calls: List[Dict[str, Any]], max_chars: int = 200) -> str:
    """Kurze Vorschau der task-Arguments fuer den Stream-Inject-Hinweis."""
    for tc in coworker_calls:
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        args_raw = fn.get("arguments", "{}")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            if not isinstance(args, dict):
                args = {}
        except (json.JSONDecodeError, ValueError, TypeError):
            args = {}
        task = str(args.get("task", "") or "").strip()
        if task:
            return task if len(task) <= max_chars else task[:max_chars] + "…"
    return "Sub-Task"


async def _read_stream_error(response: httpx.Response) -> str:
    """Liest den Fehlertext aus einer Nicht-200-Stream-Response."""
    try:
        body = await response.aread()
        err_body = json.loads(body.decode("utf-8", errors="replace"))
        if isinstance(err_body.get("error"), dict):
            return _safe_str(err_body["error"].get("message", ""))
        if isinstance(err_body.get("error"), str):
            return _safe_str(err_body["error"])
        return _safe_str(err_body.get("message") or err_body.get("detail") or err_body)
    except Exception:
        return f"HTTP {response.status_code}"


async def _stream_single_model_events(body: Dict[str, Any], category: str, def_idx: int = 0,
                                      inject_hindsight: bool = True) -> AsyncIterator[Dict[str, Any]]:
    """Streaming-Backend-Call (OpenAI-SSE) fuer EIN Modell. Yields Events:
      {"type": "chunk", "choice": <choices[0]>}   — pro SSE-Chunk
      {"type": "done"}                            — Stream sauber beendet
      {"type": "error", "status_code": int|None, "content": str, "trigger_fallback": bool}

    Gleiche Retry-/Cooldown-Logik wie _call_single_model (non-streaming), aber
    mit stream=True. Timeout/Connect-Fehler werden beim Oeffnen des Streams
    mit retry_on_timeout behandelt; Mid-Stream-Abbruche liefern ein error-Event
    (die Chunks davor sind bereits an den Client gegangen).
    """
    defs = _model_defs(category)
    if not defs or def_idx >= len(defs):
        yield {"type": "error", "status_code": None,
               "content": f"Keine gueltige Modell-Definition fuer {category}[{def_idx}]",
               "trigger_fallback": False}
        return
    cat = defs[def_idx]
    started = time.perf_counter()

    payload = _build_passthrough_payload(body, category, def_idx=def_idx)
    payload["stream"] = True

    messages = payload.get("messages", [])
    if isinstance(messages, list) and inject_hindsight:
        _inject_hindsight_context(messages)

    model = cat["model_name"]
    api_url = cat["api_url"].rstrip("/")
    api_key = cat.get("api_key", "")
    timeout = float(cat.get("timeout_seconds", 300))
    read_timeout = float(cat.get("read_timeout_seconds", timeout))
    # Streaming: Read-Timeout grosszuegig ansetzen — langsame lokale Modelle
    # brauchen manchmal lange bis zum ersten Token bzw. zwischen Tokens.
    if read_timeout < timeout:
        read_timeout = timeout

    msg_count = len(messages) if isinstance(messages, list) else 0
    total_chars = sum(len(str(m.get("content", ""))) for m in (messages if isinstance(messages, list) else []))
    _log(f"Stream call cat={category}[{def_idx}] model={model} "
         f"messages={msg_count} chars={total_chars} timeout={timeout:.0f}s read_timeout={read_timeout:.0f}s")

    req_id = f"stream_{category}_{def_idx}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    _dump_debug_payload(req_id, f"stream_{category}_{def_idx}", payload, extra={
        "category": category, "def_idx": def_idx, "model": model,
        "timeout": timeout, "messages_count": msg_count, "streaming": True,
    })
    _register_debug_request(req_id, {
        "type": "model_call_start", "streaming": True,
        "category": category, "def_idx": def_idx, "model": model,
        "messages_count": msg_count, "chars": total_chars,
        "timeout": timeout,
    })
    _register_active_call(req_id, {
        "agent_key": category, "def_idx": def_idx, "model": model,
        "phase": "passthrough-stream",
    })

    max_retries = int(cat.get("retry_on_timeout", 0))
    retry_delay = float(cat.get("retry_delay_seconds", 5))
    _http_timeout = httpx.Timeout(timeout, read=read_timeout)

    try:
        async with httpx.AsyncClient(timeout=_http_timeout) as client:
            # ── Stream oeffnen (Connect/Timeout-Retry wie non-streaming) ──
            stream_ctx = None
            entered = False
            last_exc: Optional[Exception] = None
            response: Optional[httpx.Response] = None
            for attempt in range(1 + max_retries):
                try:
                    stream_ctx = client.stream("POST", api_url, json=payload,
                                               headers=_api_headers(api_key))
                    response = await stream_ctx.__aenter__()
                    entered = True
                    last_exc = None
                    break
                except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError, OSError) as exc:
                    last_exc = exc
                    if stream_ctx is not None and entered:
                        try:
                            await stream_ctx.__aexit__(None, None, None)
                        except Exception:
                            pass
                        entered = False
                        stream_ctx = None
                    duration_so_far = time.perf_counter() - started
                    if attempt < max_retries:
                        _log(f"Stream TIMEOUT/CONN cat={category}[{def_idx}] attempt={attempt+1}/{1+max_retries} "
                             f"duration={duration_so_far:.1f}s type={type(exc).__name__}: {_safe_str(exc)}")
                        _log(f"   → Auto-Retry in {retry_delay:.0f}s...")
                        await asyncio.sleep(retry_delay)
                    else:
                        _log(f"Stream TIMEOUT/CONN cat={category}[{def_idx}] attempt={attempt+1}/{1+max_retries} "
                             f"duration={duration_so_far:.1f}s type={type(exc).__name__}: {_safe_str(exc)} — KEINE Retries mehr")

            if last_exc is not None:
                # Alle Retries erschoepft — Timeout/ConnectionError vor Stream-Beginn
                duration = time.perf_counter() - started
                exc_type = type(last_exc).__name__
                exc_msg = _safe_str(last_exc)
                _finish_active_call(req_id, "error", {"duration_seconds": duration, "error": exc_msg,
                                                       "attempts": 1 + max_retries})
                _log(f"Stream ERROR cat={category}[{def_idx}] duration={duration:.1f}s type={exc_type}: {exc_msg} "
                     f"(nach {1+max_retries} Versuchen)")
                _start_cooldown(category, def_idx, duration_override=30.0)
                yield {"type": "error", "status_code": None,
                       "content": _safe_str(f"Stream error nach {duration:.0f}s ({exc_type}, {1+max_retries} attempts): {exc_msg}"),
                       "trigger_fallback": True}
                return

            try:
                if response.status_code != 200:
                    err_detail = await _read_stream_error(response)
                    duration = time.perf_counter() - started
                    _finish_active_call(req_id, "error", {"duration_seconds": duration, "error": err_detail})
                    _log(f"Stream STATUS {response.status_code} cat={category}[{def_idx}] "
                         f"duration={duration:.1f}s: {err_detail}")
                    should_fallback = True
                    if response.status_code == 400:
                        should_fallback = False
                        _log(f"   → 400 im Stream (Payload-Problem), kein Fallback: {err_detail}")
                    elif response.status_code == 429:
                        ra = _retry_after_seconds(response.status_code, getattr(response, "headers", {}))
                        _start_cooldown(category, def_idx, duration_override=ra)
                        _log(f"   → Rate-Limit: Cooldown fuer {category}[{def_idx}]={model}")
                    elif response.status_code in (401, 403):
                        _log(f"   → Auth-Fehler, kein Cooldown (Config-Fehler)")
                    elif response.status_code >= 500:
                        _start_cooldown(category, def_idx, duration_override=60.0)
                        _log(f"   → Server-Fehler: Cooldown (60s) fuer {category}[{def_idx}]={model}")
                    yield {"type": "error", "status_code": response.status_code,
                           "content": _safe_str(f"Model status {response.status_code}: {err_detail}"),
                           "trigger_fallback": should_fallback}
                    return

                # ── SSE-Stream lesen und Events weiterreichen ──
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    choices = chunk.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    yield {"type": "chunk", "choice": choices[0]}

                duration = time.perf_counter() - started
                _finish_active_call(req_id, "done", {"duration_seconds": duration})
                _log(f"Stream OK cat={category}[{def_idx}] duration={duration:.1f}s")
                yield {"type": "done"}
            except (httpx.TimeoutException, httpx.ReadError, httpx.ConnectError, OSError) as exc:
                # Mid-Stream-Abbruch (nachdem ggf. schon Chunks geliefert wurden)
                duration = time.perf_counter() - started
                exc_type = type(exc).__name__
                _finish_active_call(req_id, "error", {"duration_seconds": duration, "error": _safe_str(exc)})
                _log(f"Stream ABBRUCH cat={category}[{def_idx}] duration={duration:.1f}s "
                     f"type={exc_type}: {_safe_str(exc)}")
                yield {"type": "error", "status_code": None,
                       "content": _safe_str(f"Stream abgebrochen nach {duration:.0f}s ({exc_type}): {exc}"),
                       "trigger_fallback": True}
            except asyncio.CancelledError:
                # Client-Disconnect / Task-Abbruch
                duration = time.perf_counter() - started
                _finish_active_call(req_id, "cancelled", {"duration_seconds": duration})
                _log(f"Stream CANCELLED cat={category}[{def_idx}] duration={duration:.1f}s (Client-Disconnect)")
                raise
            finally:
                if entered and stream_ctx is not None:
                    try:
                        await stream_ctx.__aexit__(None, None, None)
                    except Exception:
                        pass
    except asyncio.CancelledError:
        # Aufraeumen falls der Abbruch VOR dem inneren try kam (z.B. im Retry-Sleep)
        _finish_active_call(req_id, "cancelled", {"duration_seconds": time.perf_counter() - started})
        raise


async def _stream_backend_turn(body: Dict[str, Any], category: str,
                               force_start_idx: Optional[int],
                               state: Dict[str, Any]) -> AsyncIterator[str]:
    """Fuehrt EINEN Streaming-Backend-Turn aus (mit Fallback ueber die defs).

    Yields SSE-Strings an VS Code — inkl. Keepalive-Kommentaren, wenn das
    lokale Modell zwischen zwei Tokens laenger braucht als der Keepalive-
    Intervall. Mutiert `state` (per Turn zurueckgesetzt):
      content / reasoning / tool_calls (Akkumulator) / finish_reason /
      model / all_failed / mid_stream_error / error_content / role_sent (bleibt)
    """
    state.update({
        "content": "", "reasoning": "", "tool_calls": {},
        "finish_reason": None, "all_failed": False,
        "mid_stream_error": None, "error_content": None,
        "def_idx": 0, "model": category,
        "has_explicit_reasoning": False,
    })
    think_state: Dict[str, Any] = {"in_think": False, "pending": ""}

    defs = _model_defs(category)
    if not defs:
        state["all_failed"] = True
        state["error_content"] = f"Kategorie '{category}' hat keine konfigurierten Modelle"
        return

    start_idx = force_start_idx if force_start_idx is not None else _CATEGORY_ACTIVE_IDX.get(category, 0)
    if start_idx >= len(defs):
        start_idx = 0
    indices: List[int] = [start_idx] + [i for i in range(len(defs)) if i != start_idx]

    queue: asyncio.Queue = asyncio.Queue(maxsize=256)

    async def worker() -> None:
        """Liest Events vom Backend und legt sie in die Queue.
        Fallback: Pre-Stream-Fehler → naechstes def probieren."""
        last_err: Optional[Dict[str, Any]] = None
        for idx in indices:
            if idx != start_idx and _is_in_cooldown(category, idx):
                continue
            forwarded = False
            state["model"] = defs[idx].get("model_name", "?")
            state["def_idx"] = idx
            try:
                async for ev in _stream_single_model_events(body, category, idx):
                    ev_type = ev.get("type") if isinstance(ev, dict) else None
                    if ev_type == "chunk":
                        forwarded = True
                        await queue.put(ev)
                    elif ev_type == "done":
                        # Erfolg: aktiven Index merken (wie non-streaming Fallback)
                        _CATEGORY_ACTIVE_IDX[category] = idx
                        await queue.put(ev)
                        return
                    elif ev_type == "error":
                        last_err = ev
                        if forwarded:
                            state["mid_stream_error"] = ev.get("content", "Stream-Fehler")
                            await queue.put({"type": "__end__"})
                            return
                        break  # Pre-Stream-Fehler → naechstes def
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _log(f"Stream-Turn EXCEPTION cat={category} idx={idx}: {_safe_str(exc)}")
                last_err = {"type": "error", "status_code": None,
                            "content": _safe_str(exc), "trigger_fallback": True}
                break
        # Alle defs fehlgeschlagen (vor Stream-Beginn)
        state["all_failed"] = True
        state["error_content"] = (last_err or {}).get("content", "Unbekannter Stream-Fehler")
        await queue.put({"type": "__end__"})

    task = asyncio.ensure_future(worker())
    try:
        while True:
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=_STREAM_KEEPALIVE_INTERVAL)
            except asyncio.TimeoutError:
                # Modell arbeitet noch, aber es kommt gerade kein Token —
                # SSE-Kommentar haelt die VS-Code-Verbindung am Leben.
                yield ": keepalive\n\n"
                continue

            ev_type = ev.get("type") if isinstance(ev, dict) else None
            if ev_type in ("done", "__end__"):
                break
            if ev_type != "chunk":
                continue

            choice = ev.get("choice") or {}
            delta = choice.get("delta") or {}
            if choice.get("finish_reason"):
                state["finish_reason"] = choice["finish_reason"]

            rc = _extract_reasoning_from_delta(delta)
            if rc:
                state["has_explicit_reasoning"] = True
                state["reasoning"] = (state.get("reasoning") or "") + rc
                yield _format_openai_stream_chunk(
                    state.get("model", category), reasoning_content=rc,
                    include_role=not state.get("role_sent"),
                    chunk_id=state.get("stream_id"))
                state["role_sent"] = True

            c = delta.get("content")
            if isinstance(c, str) and c:
                if state.get("has_explicit_reasoning"):
                    # Backend liefert Reasoning in eigenem Feld → content unveraendert
                    state["content"] = (state.get("content") or "") + c
                    yield _format_openai_stream_chunk(
                        state.get("model", category), content=c,
                        include_role=not state.get("role_sent"),
                        chunk_id=state.get("stream_id"))
                    state["role_sent"] = True
                else:
                    # <think>...</think> im content (vLLM Qwen3 preserve_thinking)
                    # → als eigenen Reasoning-Context mappen, Rest als content
                    reasoning_part, content_part = _split_think_chunk(c, think_state)
                    if reasoning_part:
                        state["reasoning"] = (state.get("reasoning") or "") + reasoning_part
                        yield _format_openai_stream_chunk(
                            state.get("model", category), reasoning_content=reasoning_part,
                            include_role=not state.get("role_sent"),
                            chunk_id=state.get("stream_id"))
                        state["role_sent"] = True
                    if content_part:
                        state["content"] = (state.get("content") or "") + content_part
                        yield _format_openai_stream_chunk(
                            state.get("model", category), content=content_part,
                            include_role=not state.get("role_sent"),
                            chunk_id=state.get("stream_id"))
                        state["role_sent"] = True

            tcs = delta.get("tool_calls")
            if tcs:
                _accumulate_stream_tool_calls(state, tcs)

        # Angebrochene <think>-Tags am Turn-Ende flushen (Modell stoppt selten
        # mitten in einem Tag, aber falls doch: in reasoning/content nachladen)
        if think_state.get("pending"):
            pending = think_state.pop("pending", "")
            if think_state.get("in_think"):
                state["reasoning"] = (state.get("reasoning") or "") + pending
            else:
                state["content"] = (state.get("content") or "") + pending
    finally:
        if not task.done():
            task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


async def _stream_coworker_call(tool_call: Dict[str, Any], queue: asyncio.Queue,
                                model: str, stream_id: str,
                                extra_context: Optional[str] = None) -> Dict[str, Any]:
    """Fuehrt EINEN ask_coworker-Call intern aus und streamt dabei dessen
    reasoning_content LIVE als eigenen Reasoning-Context an VS Code durch
    (delta.reasoning_content) — der User sieht, worueber der Co-Worker
    nachdenkt, waehrend er arbeitet.

    Die Antwort-Content streamt LIVE token-fuer-token an VS Code durch
    (einmal "[Proxy] Co-Worker-Antwort:"-Header, dann die Text-Chunks) —
    so gibt es waehrend der Co-Worker-Arbeit keine toten Phasen und kein
    Timeout-Gefuehl. Returns die tool-result Message (inkl.
    'reasoning_content' fuer Reporting/Logs)."""
    tool_call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:12]}"
    args_raw = (tool_call.get("function") or {}).get("arguments", "{}")
    try:
        args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        if not isinstance(args, dict):
            args = {}
    except (json.JSONDecodeError, ValueError, TypeError):
        args = {}
    task = str(args.get("task", "") or "")
    context = str(args.get("context", "") or "")

    started = time.perf_counter()
    body = _build_coworker_body(task, context, extra_context=extra_context)
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    think_state: Dict[str, Any] = {"in_think": False, "pending": ""}
    has_explicit_reasoning = False
    status = "failed"
    err_text = "unbekannter Fehler"
    answer_header_sent = False

    async def push_reasoning(rc: str) -> None:
        reasoning_parts.append(rc)
        await queue.put(_format_openai_stream_chunk(
            model, reasoning_content=rc, include_role=False, chunk_id=stream_id))

    async def push_content(c: str) -> None:
        """Streamt die Antwort-Content LIVE an VS Code (token-fuer-token).
        Einmal pro Call wird der '[Proxy] Co-Worker-Antwort:'-Header davor
        gesetzt — so sieht der User den Co-Worker-Text waehrend er entsteht."""
        nonlocal answer_header_sent
        if not answer_header_sent:
            await queue.put(_format_openai_stream_chunk(
                model, content="\n\n[Proxy] Co-Worker-Antwort:\n",
                include_role=False, chunk_id=stream_id))
            answer_header_sent = True
        await queue.put(_format_openai_stream_chunk(
            model, content=c, include_role=False, chunk_id=stream_id))

    try:
        async for ev in _stream_single_model_events(body, "coworker", 0, inject_hindsight=False):
            ev_type = ev.get("type") if isinstance(ev, dict) else None
            if ev_type == "chunk":
                choice = ev.get("choice") or {}
                delta = choice.get("delta") or {}
                rc = _extract_reasoning_from_delta(delta)
                if rc:
                    has_explicit_reasoning = True
                    await push_reasoning(rc)
                c = delta.get("content")
                if isinstance(c, str) and c:
                    if has_explicit_reasoning:
                        content_parts.append(c)
                        await push_content(c)
                    else:
                        reasoning_part, content_part = _split_think_chunk(c, think_state)
                        if reasoning_part:
                            await push_reasoning(reasoning_part)
                        if content_part:
                            content_parts.append(content_part)
                            await push_content(content_part)
            elif ev_type == "done":
                status = "ok"
            elif ev_type == "error":
                err_text = ev.get("content") or err_text
                status = "failed"
        # Angebrochene <think>-Tags am Ende flushen
        if think_state.get("pending"):
            pending = think_state.pop("pending", "")
            if think_state.get("in_think"):
                await push_reasoning(pending)
            else:
                content_parts.append(pending)
                await push_content(pending)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        err_text = _safe_str(exc)
        status = "failed"
        _log(f"Co-Worker-Stream FEHLER: {err_text[:200]}")

    duration = time.perf_counter() - started
    content = "".join(content_parts).strip()
    reasoning = "".join(reasoning_parts).strip()

    if status == "ok":
        if COWORKER_RESULT_CAP > 0 and len(content) > COWORKER_RESULT_CAP:
            content = content[:COWORKER_RESULT_CAP] + "\n…[gekappt]"
        _log(f"Co-Worker-Stream OK duration={duration:.1f}s len={len(content)} "
             f"reasoning={len(reasoning)}")
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": _COWORKER_TOOL_NAME,
            "content": content,
            "reasoning_content": reasoning or None,
        }

    # Fehlertext ebenfalls sichtbar streamen (sonst wuerde der User nichts sehen)
    _COWORKER_HEALTH_CACHE["reachable"] = False
    _log(f"Co-Worker-Stream FEHLER duration={duration:.1f}s: {err_text[:200]}")
    err_display = f"[Co-Worker nicht verfuegbar]\n{err_text}"
    try:
        if not answer_header_sent:
            await queue.put(_format_openai_stream_chunk(
                model, content="\n\n[Proxy] Co-Worker-Antwort:\n",
                include_role=False, chunk_id=stream_id))
        await queue.put(_format_openai_stream_chunk(
            model, content=err_display, include_role=False, chunk_id=stream_id))
    except Exception:
        pass
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": _COWORKER_TOOL_NAME,
        "content": err_display,
        "reasoning_content": reasoning or None,
    }


async def _stream_coworker_phase(coworker_calls_norm: List[Dict[str, Any]],
                                 coworker_state: Dict[str, Any]) -> AsyncIterator[str]:
    """Fuehrt die Co-Worker-Calls (parallel, separate Hardware) intern aus.

    Das reasoning_content der Co-Worker wird LIVE als eigener Reasoning-Context
    (delta.reasoning_content) an VS Code gestreamt; die Antwort-Content streamt
    LIVE token-fuer-token mit "[Proxy] Co-Worker-Antwort:"-Header.

    Yields Keepalive-SSE-Kommentare, damit VS Code waehrend der (langen)
    Co-Worker-Arbeit nicht timeoutet. Die tool-result Messages landen in
    coworker_state['coworker_results'] (inkl. 'reasoning_content').
    """
    model = coworker_state.get("model", "local")
    stream_id = coworker_state.get("stream_id") or f"chatcmpl-spark-{uuid.uuid4().hex}"
    files_context = coworker_state.get("files_context") or ""
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    results: List[Optional[Dict[str, Any]]] = [None] * len(coworker_calls_norm)
    done_count = 0

    async def run_one(idx: int, tc: Dict[str, Any]) -> None:
        nonlocal done_count
        try:
            results[idx] = await _stream_coworker_call(
                tc, queue, model, stream_id, extra_context=files_context)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log(f"Co-Worker-Stream EXCEPTION call#{idx}: {_safe_str(exc)}")
            err_text = f"[Co-Worker nicht verfuegbar]\n{_safe_str(exc)}"
            results[idx] = {
                "role": "tool",
                "tool_call_id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                "name": _COWORKER_TOOL_NAME,
                "content": err_text,
            }
            try:
                await queue.put(_format_openai_stream_chunk(
                    model, content=f"\n\n[Proxy] Co-Worker-Antwort:\n{err_text}",
                    include_role=False, chunk_id=stream_id))
            except Exception:
                pass
        done_count += 1

    tasks = [asyncio.ensure_future(run_one(i, tc)) for i, tc in enumerate(coworker_calls_norm)]
    try:
        while done_count < len(tasks):
            try:
                sse = await asyncio.wait_for(queue.get(), timeout=_STREAM_KEEPALIVE_INTERVAL)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            yield sse
        # Queue-Reste nach allen Calls drainen
        while True:
            try:
                yield queue.get_nowait()
            except asyncio.QueueEmpty:
                break
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    coworker_state["coworker_results"] = [r for r in results if r is not None]
    _log(f"Co-Worker-Phase beendet: {len(coworker_state['coworker_results'])} Ergebnis(se)")


async def _stream_local_events(body: Dict[str, Any], category: str,
                               force_start_idx: Optional[int] = None) -> AsyncIterator[str]:
    """Live-Streaming fuer Kategorie=local inkl. Co-Worker-Stream-Inject.

    Ablauf pro Runde:
      * _stream_backend_turn streamt das Hauptmodell live (Thinking + content).
      * Endet der Turn ohne Tool-Calls → finish_chunk, fertig.
      * ask_coworker-Calls werden intern abgearbeitet. Dabei werden per
        Stream-Inject sichtbar gemacht:
          1. "[Proxy] Delegation an Co-Worker: <task>…"
          2. das reasoning_content des Co-Workers streamt LIVE als eigener
             Reasoning-Context an VS Code durch (Issue: Reasoning des
             Co-Workers sichtbar machen)
          3. die Co-Worker-Antwort streamt LIVE token-fuer-token (mit
             "[Proxy] Co-Worker-Antwort:"-Header)
          4. danach streamt das Hauptmodell live weiter — der User sieht,
             was es mit der Antwort macht.
      * dispatch_coworker (Fork-Join v3.2): non-blocking — Task im Store
        registrieren, mini-result task_id in die History, sichtbarer
        "[Proxy] Co-Worker dispatched"-Chunk, dann direkt weiter (gemischt
        mit VS-Code-Tools erlaubt — die werden durchgereicht).
      * collect_coworker (Join): blockt bis Ergebnisse da sind oder Timeout;
        Ergebnisse als tool-Message in die History, sichtbarer
        "[Proxy] Sammle Co-Worker-Ergebnisse"-Chunk.
      * VS-Code-Tools (read_file etc.) werden unveraendert an Copilot
        durchgereicht.
    """
    stream_id = f"chatcmpl-spark-{uuid.uuid4().hex}"
    state: Dict[str, Any] = {"stream_id": stream_id, "role_sent": False}
    # Datei-Kontext aus der Chat-History einmalig extrahieren — der Co-Worker
    # bekommt IMMER die relevanten Dateiinhalte, auch wenn das Hauptmodell sie
    # nicht in task/context uebernommen hat.
    files_context = _extract_conversation_files(body.get("messages"), COWORKER_FILES_CAP)
    if files_context:
        _log(f"Co-Worker-Stream: {len(files_context)} chars Datei-Kontext "
             f"automatisch angehaengt")
    rounds = 0
    dispatch_count = 0
    hard_limit = False

    while True:
        async for sse in _stream_backend_turn(body, category, force_start_idx, state):
            yield sse

        model = state.get("model", category)

        # ── Alle Modelle fehlgeschlagen (vor Stream-Beginn) ──
        if state.get("all_failed"):
            _log(f"Stream: ALLE Fallbacks fehlgeschlagen cat={category}: {state.get('error_content','?')}")
            yield _format_openai_stream_chunk(
                model,
                content=f"[Proxy: Stream-Fehler] {state.get('error_content','')}",
                include_role=not state.get("role_sent"), chunk_id=stream_id)
            yield _format_openai_stream_chunk(model, "", finish_reason="stop", chunk_id=stream_id)
            return

        # ── Mid-Stream-Abbruch (Chunks kamen, dann Fehler) ──
        if state.get("mid_stream_error"):
            yield _format_openai_stream_chunk(
                model,
                content=f"\n\n[Proxy: Stream abgebrochen] {state.get('mid_stream_error','')}",
                include_role=not state.get("role_sent"), chunk_id=stream_id)
            yield _format_openai_stream_chunk(model, "", finish_reason="stop", chunk_id=stream_id)
            return

        tool_calls = _finalize_stream_tool_calls(state)

        # ── Keine Tool-Calls: Turn normal beenden ──
        if not tool_calls:
            content = state.get("content", "") or ""
            fr = state.get("finish_reason") or "stop"
            yield _format_openai_stream_chunk(model, "", finish_reason=fr, chunk_id=stream_id)
            if content.strip():
                asyncio.ensure_future(_hindsight.retain_async(body, content))
            return

        dispatch_calls, collect_calls, ask_calls, other_calls = _partition_tool_calls(tool_calls)
        coworker_calls = ask_calls

        # ── Delegations-Limit: hart stoppen falls weiter delegiert wird ──
        if hard_limit and (coworker_calls or collect_calls):
            if not other_calls:
                _log("Co-Worker-Delegation-Limit: hart gestoppt (Modell wollte weiter delegieren)")
                yield _format_openai_stream_chunk(
                    model,
                    content="\n\n[Proxy] Co-Worker-Delegation-Limit erreicht — Aufgabe ohne "
                            "Co-Worker-Unterstuetzung beantwortet.",
                    include_role=not state.get("role_sent"), chunk_id=stream_id)
                yield _format_openai_stream_chunk(model, "", finish_reason="stop", chunk_id=stream_id)
                return
            # Nur coworker-Calls entfernen, VS-Code-Tools durchreichen
            yield _format_openai_stream_chunk(
                model, include_role=True,
                tool_calls=_build_forward_tool_calls(other_calls), chunk_id=stream_id)
            yield _format_openai_stream_chunk(model, finish_reason="tool_calls", chunk_id=stream_id)
            return

        # ── Nur VS-Code-Tools: an Copilot durchreichen ──
        if not coworker_calls and not dispatch_calls and not collect_calls:
            yield _format_openai_stream_chunk(
                model, include_role=True,
                tool_calls=_build_forward_tool_calls(tool_calls), chunk_id=stream_id)
            yield _format_openai_stream_chunk(model, finish_reason="tool_calls", chunk_id=stream_id)
            return

        # ── Fork: dispatches registrieren (non-blocking, Store = Truth) ──
        if dispatch_calls:
            if dispatch_count + len(dispatch_calls) > COWORKER_DISPATCH_CAP:
                _log(f"Dispatch-Cap erreicht ({COWORKER_DISPATCH_CAP}/Request) — Hinweis")
                msgs = body.get("messages", [])
                msgs.append({"role": "user", "content":
                    f"[Proxy] Dispatch-Limit erreicht ({COWORKER_DISPATCH_CAP} pro Request). "
                    "Sammle zuerst mit collect_coworker oder arbeite selbst weiter."})
                yield _format_openai_stream_chunk(
                    model,
                    content=f"\n\n[Proxy] Dispatch-Limit erreicht ({COWORKER_DISPATCH_CAP} pro Request) — "
                            "sammle zuerst mit collect_coworker.",
                    include_role=not state.get("role_sent"), chunk_id=stream_id)
                state["role_sent"] = True
                rounds += 1  # Loop-Schutz: Cap-Zustand zaehlt als Runde
                if rounds > COWORKER_MAX_DELEGATIONS:
                    hard_limit = True
                continue
            dispatch_count += len(dispatch_calls)
            msgs = body.get("messages", [])
            dispatch_norm = _normalize_tool_calls(dispatch_calls) or dispatch_calls
            msgs.append({"role": "assistant", "content": None,
                         "tool_calls": dispatch_norm})
            dispatched_note = ["\n\n[Proxy] Co-Worker dispatched:"]
            for tc in dispatch_norm:
                ct = _register_bg_dispatch(tc, files_context)
                msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                    "name": _COWORKER_DISPATCH_TOOL_NAME,
                    "content": json.dumps(
                        {"task_id": ct.task_id, "status": "dispatched"},
                        ensure_ascii=False),
                })
                dispatched_note.append(f" {ct.task_id} ({ct.preview})")
            # Live-Inject: User sieht sofort, was im Hintergrund laeuft
            yield _format_openai_stream_chunk(
                model,
                content=" ".join(dispatched_note),
                include_role=not state.get("role_sent"), chunk_id=stream_id)
            state["role_sent"] = True
            # VS-Code-Tools im selben Turn: JETZT durchreichen (mixed-turn
            # unlock) — dispatches sind intern beantwortet, collect nie.
            if other_calls:
                yield _format_openai_stream_chunk(
                    model, include_role=True,
                    tool_calls=_build_forward_tool_calls(other_calls), chunk_id=stream_id)
                yield _format_openai_stream_chunk(model, finish_reason="tool_calls", chunk_id=stream_id)
                return
            continue  # nur dispatches → nächste Runde (Modell arbeitet weiter)

        rounds += 1

        # ── Delegations-Limit erreicht: finale Runde ohne weitere Delegation ──
        if rounds > COWORKER_MAX_DELEGATIONS:
            _log(f"Co-Worker-Delegation-Limit erreicht ({COWORKER_MAX_DELEGATIONS} Runden)")
            msgs = body.get("messages", [])
            msgs.append({"role": "user", "content":
                "[Proxy] Co-Worker-Delegation-Limit erreicht. Beantworte die Aufgabe "
                "jetzt direkt, ohne ask_coworker erneut aufzurufen."})
            hard_limit = True
            continue

        # ── Join: collect_coworker blockt bis Ergebnisse da sind ──
        if collect_calls:
            collect_norm = _normalize_tool_calls(collect_calls) or collect_calls
            msgs = body.get("messages", [])
            msgs.append({"role": "assistant", "content": None,
                         "tool_calls": collect_norm})
            yield _format_openai_stream_chunk(
                model,
                content="\n\n[Proxy] Sammle Co-Worker-Ergebnisse …",
                include_role=not state.get("role_sent"), chunk_id=stream_id)
            state["role_sent"] = True
            for tc in collect_norm:
                args_raw = (tc.get("function") or {}).get("arguments", "{}")
                try:
                    cargs = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                    if not isinstance(cargs, dict):
                        cargs = {}
                except (json.JSONDecodeError, ValueError, TypeError):
                    cargs = {}
                summaries = await _await_bg_tasks(
                    cargs.get("task_ids"),
                    timeout_seconds=float(cargs.get("timeout_seconds", 600) or 600),
                )
                msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                    "name": _COWORKER_COLLECT_TOOL_NAME,
                    "content": json.dumps(summaries, ensure_ascii=False, indent=2),
                })
            if other_calls:
                # Andere Tools im selben Turn wie collect: wurden NICHT
                # ausgefuehrt — Modell muss sie naechste Runde erneut rufen.
                msgs.append({"role": "user", "content":
                    "[Proxy-Hinweis] Die anderen Tools in diesem Turn wurden nicht "
                    "ausgefuehrt (collect_coworker blockt). Rufe sie jetzt erneut auf."})
            if ask_calls and not other_calls:
                # ask + collect im selben Turn: erst gesammelt, jetzt asks —
                # faellt durch in den ask-Pfad unten (gemeinsame History).
                pass
            else:
                continue  # collect done → nächste Runde (Modell verarbeitet)

        # ── Gemischter Turn: Hinweis injizieren + neu aufrufen ──
        if other_calls:
            _log(f"Gemischter Turn (ask_coworker + {len(other_calls)} andere Tools) "
                 f"— Hinweis injizieren")
            msgs = body.get("messages", [])
            msgs.append({"role": "user", "content":
                "[Proxy-Hinweis] Rufe ask_coworker nicht zusammen mit anderen Tools "
                "auf. Fuehre entweder die anderen Tools aus ODER rufe ask_coworker "
                "in einem separaten Turn allein auf."})
            continue

        # ── Reiner Co-Worker-Turn: intern abarbeiten + Stream-Inject ──
        coworker_calls_norm = _normalize_tool_calls(coworker_calls) or coworker_calls
        msgs = body.get("messages", [])
        msgs.append({"role": "assistant", "content": None, "tool_calls": coworker_calls_norm})

        # 1) Stream-Inject: Delegations-Hinweis (User sieht, dass uebergeben wird)
        preview = _coworker_task_preview(coworker_calls_norm)
        yield _format_openai_stream_chunk(
            model,
            content=f"\n\n[Proxy] Delegation an Co-Worker: {preview}…",
            include_role=not state.get("role_sent"), chunk_id=stream_id)
        state["role_sent"] = True

        # 2) Co-Worker-Calls ausfuehren (Reasoning streamt live als eigener
        #    Reasoning-Context; Keepalives halten VS Code am Leben)
        coworker_state: Dict[str, Any] = {"stream_id": stream_id, "model": model,
                                          "files_context": files_context}
        async for sse in _stream_coworker_phase(coworker_calls_norm, coworker_state):
            yield sse
        results = coworker_state.get("coworker_results", [])
        # reasoning_content nicht in die History schreiben (nur Reporting —
        # der User sieht es bereits live als Reasoning-Context in VS Code)
        for r in results:
            r.pop("reasoning_content", None)
        msgs.extend(results)

        # 3) Die Co-Worker-Antwort wurde bereits waehrend der Phase LIVE
        #    token-fuer-token gestreamt (siehe _stream_coworker_call/push_content)
        #    — kein One-Shot-Inject mehr noetig.

        # 4) Naechste Runde: Hauptmodell verarbeitet die Antwort (live gestreamt)
        continue


# ── /v1/models ─────────────────────────────────────────────────────────────
@app.get("/v1/models")
async def list_models(request: Request):
    logs_str = request.query_params.get("logs", "")
    if logs_str and logs_str.isdigit() and int(logs_str) > 0:
        return JSONResponse(content=await _get_logs_handler(lines=int(logs_str)))
    await _auth_or_raise(request)
    models = []
    for key in ("local", "coworker", "light", "strong", "vision"):
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
                "coworker": _model_defs("coworker"),
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
