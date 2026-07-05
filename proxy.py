"""
LocalProxy — Hybrider Agentischer Routing-Proxy (OpenAI-kompatibel)

Architektur (laut Gemini-Plan):
  VS Code (Continue/Cline/Roo) → FastAPI Gateway → Intent Classifier
    ├─ Direkt: Lokal/Free (Worker & Fast)
    └─ Agentisch: Hindsight Recall → Cloud-Planer (Caveman) → Worker (80B) → Verify

Komponenten:
  1. Qdrant-basiertes Hindsight Memory (4 Netzwerke)
  2. LiteLLM Cloud-Routing (DeepSeek/Claude/GPT via OpenRouter)
  3. 3‑Phasen‑Agenten‑Workflow: Plan → Execute → Verify
  4. MCP‑Server für VS‑Code‑Tool‑Zugriff
  5. Caveman‑Ultra‑Prompt‑Kompression
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import textwrap
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

# ── Logging ────────────────────────────────────────────────────────────────
import datetime

LOG_FILE = os.getenv("LOG_FILE", str(Path(__file__).parent / "proxy.log"))


def _log(msg: str) -> None:
    """Schreibt eine Log-Zeile mit Timestamp in Datei + stdout."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass  # kein Log-File? egal


# ── Webinterface ───────────────────────────────────────────────────────────
try:
    from webui import mount_webui, _load_config as _webui_load_config

    _WEBUI_AVAILABLE = True
except Exception:
    _WEBUI_AVAILABLE = False

# ── Optional: LiteLLM ──────────────────────────────────────────────────────
try:
    import litellm  # type: ignore[import-untyped]

    _LITELLM_AVAILABLE = True
except Exception:
    litellm = None  # type: ignore[assignment]
    _LITELLM_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════
# Zero-Config‑Startwerte (alles via env überschreibbar)
# ═══════════════════════════════════════════════════════════════════════════

VLLM_API_URL: str = os.getenv("VLLM_API_URL", "http://localhost:8000/v1/chat/completions")
VLLM_API_KEY: str = os.getenv("VLLM_API_KEY", "")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen3-Next-80B-Chat-mxfp4")
FAST_MODEL_NAME: str = os.getenv("FAST_MODEL_NAME", "Qwen/Qwen3.6-27B-Chat-FP8")
PROXY_PORT: int = int(os.getenv("PROXY_PORT", "9001"))
MCP_PORT: int = int(os.getenv("MCP_PORT", "9002"))
MCP_ENABLED: bool = os.getenv("MCP_ENABLED", "true").lower() in {"1", "true", "yes", "y", "on"}

# ── Auth ───────────────────────────────────────────────────────────────────
PROXY_AUTH_ENABLED: bool = os.getenv("PROXY_AUTH_ENABLED", "true").lower() in {"1", "true", "yes", "y", "on"}
PROXY_API_KEY: str = os.getenv("PROXY_API_KEY", "")
if PROXY_AUTH_ENABLED and not PROXY_API_KEY:
    PROXY_API_KEY = "localfox-" + secrets.token_hex(16)
    _log(f"⚡ Auto-generated PROXY_API_KEY: {PROXY_API_KEY}")

# ── Chatty / Progress ──────────────────────────────────────────────────────
CHATTY_MODE: bool = os.getenv("CHATTY_MODE", "true").lower() in {"1", "true", "yes", "y", "on"}
CHATTY_HEARTBEAT_SECONDS: float = float(os.getenv("CHATTY_HEARTBEAT_SECONDS", "15"))

# ── Cloud Review / Planning ────────────────────────────────────────────────
CLOUD_REVIEW_ENABLED: bool = os.getenv("CLOUD_REVIEW_ENABLED", "false").lower() in {"1", "true", "yes", "y", "on"}
CLOUD_REVIEW_API_URL: str = os.getenv("CLOUD_REVIEW_API_URL", "https://api.openai.com/v1/chat/completions")
CLOUD_REVIEW_API_KEY: str = os.getenv("CLOUD_REVIEW_API_KEY", "")
CLOUD_REVIEW_MODEL: str = os.getenv("CLOUD_REVIEW_MODEL", "gpt-4.1-mini")
CLOUD_REVIEW_MAX_TOKENS: int = int(os.getenv("CLOUD_REVIEW_MAX_TOKENS", "128000"))
CLOUD_REVIEW_TIMEOUT_SECONDS: float = float(os.getenv("CLOUD_REVIEW_TIMEOUT_SECONDS", "180"))

# LiteLLM Cloud-Provider (DeepSeek, Claude, GPT via OpenRouter o.ä.)
LITELLM_CLOUD_MODEL: str = os.getenv("LITELLM_CLOUD_MODEL", "")
LITELLM_CLOUD_API_KEY: str = os.getenv("LITELLM_CLOUD_API_KEY", "")
LITELLM_CLOUD_API_URL: str = os.getenv("LITELLM_CLOUD_API_URL", "")
LITELLM_CLOUD_MAX_TOKENS: int = int(os.getenv("LITELLM_CLOUD_MAX_TOKENS", "16384"))
LITELLM_CLOUD_TIMEOUT_SECONDS: float = float(os.getenv("LITELLM_CLOUD_TIMEOUT_SECONDS", "180"))

# ── Caveman ────────────────────────────────────────────────────────────────
CAVEMAN_ENABLED: bool = os.getenv("CAVEMAN_ENABLED", "true").lower() in {"1", "true", "yes", "y", "on"}
CAVEMAN_SYSTEM_PROMPT: str = (
    "CAVEMAN ULTRA MODE. Only compact symbols, arrows, terse keywords. "
    "No filler, no grammar, no prose. Use ->, !, ?, FIX, RISK, TODO. "
    "Return executable plan only."
)
CAVEMAN_MAX_TOKENS: int = int(os.getenv("CAVEMAN_MAX_TOKENS", "8192"))

# ── Tool-Result-Capping ────────────────────────────────────────────────────
# Verhindert Token-Bombing, wenn VS Code riesige Tool-Results (z.B. 111KB grep
# auf level_data.json) an den Proxy returniert. Tool-Messages im Payload werden
# nach TOOL_RESULT_CAP chars hart abgeschnitten (+ TRUNCATED marker).
TOOL_RESULT_CAP: int = int(os.getenv("TOOL_RESULT_CAP", "0"))  # 0 = deaktiviert

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
HINDSIGHT_DIR: Path = Path(os.getenv("HINDSIGHT_DIR", "./.hindsight_memory"))

# ── Plans-Verzeichnis (Codespace-Copilot-Stil) ───────────────────────────
# Pläne des Cloud-Planners werden als .md-Files persistiert:
#   data/plans/Plan_<session_hash>.md
# Vorteile:
#   - Worker kann Plan via File-Inhalt lesen (stabil, kein JSON-Wrapping-Bruch)
#   - Persistente Debug-Historie im Volume Mount
#   - WebUI/API kann Plans showcase/downladen
#   - Robust gegen "thinking content could not be passed"
PLANS_DIR: Path = Path(os.getenv("PLANS_DIR", "./data/plans"))

# ── Trigger-Wörter ─────────────────────────────────────────────────────────
AGENT_TRIGGER_WORDS: Tuple[str, ...] = (
    "refactor", "refactore", "refactoring", "architektur", "architecture",
    "bug", "fehler", "debug", "multithreading", "race condition", "racecondition",
    "performance", "optimierung", "test", "lint", "migration",
    "security", "sicherheit", "explain", "erkläre",
    "implement", "implementiere", "create", "erstelle",
    "add", "hinzufügen", "hinzufuegen", "umfassend", "komplex",
    "rewrite", "umschreiben", "design", "entwurf",
)
DIRECT_TRIGGER_WORDS: Tuple[str, ...] = (
    "autocomplete", "kurz", "klein", "trivial",
    "inline", "fix typo", "rename", "format", "linter",
    "kommentar", "comment", "variable umbenennen",
    "syntax", "indent", "whitespace", "formatting",
)

# ── Token-Budgets & Timeouts ───────────────────────────────────────────────
DEFAULT_DIRECT_MAX_TOKENS: int = int(os.getenv("DIRECT_MAX_TOKENS", "65536"))
DEFAULT_AGENT_MAX_TOKENS: int = int(os.getenv("SUB_AGENT_MAX_TOKENS", "65536"))
SUB_AGENT_TIMEOUT_SECONDS: float = float(os.getenv("SUB_AGENT_TIMEOUT_SECONDS", "300"))
VERIFY_TIMEOUT_SECONDS: float = float(os.getenv("VERIFY_TIMEOUT_SECONDS", "120"))

# ── Phase-3 Verification ───────────────────────────────────────────────────
VERIFY_ENABLED: bool = os.getenv("VERIFY_ENABLED", "true").lower() in {"1", "true", "yes", "y", "on"}
VERIFY_LINT_COMMAND: str = os.getenv("VERIFY_LINT_COMMAND", "")
VERIFY_TEST_COMMAND: str = os.getenv("VERIFY_TEST_COMMAND", "")

# ── Planner Session Tracking ──────────────────────────────────────────────
# Wenn -force planning aktiv: Cloud-Planner (Kimi K2.7) wird zum vollwertigen
# VS Code Sub-Agent mit Tools. Er exploriert den Workspace und erstellt einen
# detaillierten Caveman-Plan, den der Worker dann ausführt.
_PLANNER_SESSIONS: Dict[str, Dict[str, Any]] = {}
# Session-Struktur: {
#   "state": "active"|"done",
#   "iterations": int,                       # Anzahl bisheriger Tool-Runden
#   "tool_signatures": List[str],            # ["grep_search|<hash>", ...] in Reihenfolge
#   "distinct_files": Set[str],              # Dateien, die read/grep'd wurden → Fortschritts-Metrik
#   "plan": "...", "pflags": {...}, "ts": float
# }

# ── Planner-Loop-Detection (Replace für arbitrary Cap) ──────────────
# FRÜHER: Hard-Cap bei 8 Iterationen → hat legittime Exploration großer
#         Workspaces abgebrochen (jeder tool_cont zählte als volle Iteration).
# JETZT: Signature-basierte Loop-Erkennung. Ein echter Endlos-Loop liegt nur
#         vor, wenn Kimi dasselbe Tool mit denselben Argumenten WIEDERHOLT.
#
# Strategie (3-stufig):
#   1) REPEAT-Detection: Tool-Signaturen (name, arg_hash) werden in der
#      Session gespeichert. Echter Loop = identische Signatur 2x hintereinander.
#   2) EXACT-REPEAT Hard-Stop: 2x identischer `(name, args)` → Plan erzwingen.
#      Verhindert die klassische "No matches found" → gleiches grep nochmal Loop.
#   3) SOFT-CAP (PLANNER_SOFT_CAP): Sehr hoher Sicherheits-Tripwire, nur
#      wenn Kimi völlig außer Kontrolle gerät.
#
# Detector-Historie wird beim Start jeder neuen Planner-Session zurückgesetzt.
MAX_PLANNER_ITERATIONS = 15       # Aider-Äquivalent: Architect hat begrenzten Kontext.
                                  # Mit dem <exploration_budget> von 5–8 Reads + ein paar
                                  # Clarifying/Grep-Calls sind 15 Runden großzügig.
                                  # Danach: harte Plan-Ausgabe (kein weiteres Tool-Ping-Pong).
PLANNER_REPEAT_HARD_STOP = 3      # Bei N identischen (name,args)-Wiederholungen in Folge → Plan erzwingen
                                  # (3 statt 2: read(x)→read(x) kann legitim sein; 3x = echter Loop)
PLANNER_WARN_AFTER = 12           # Ab Iteration M: sanfter System-Hinweis "bald plan ausgeben"
PLANNER_DISTINCT_WARN_AFTER = 18  # Ab Iteration M mit kaum Fortschritt: stärkere Warnung
# Bug-Malformed-Loop: Wenn Kimi gebuggte Tool-Calls ohne Args rausschickt
# (z.B. parallele read_file-Batches ohne filePath) und das in N aufeinander-
# folgenden Runden passiert, ist der Planner offiziell im Loop — Every tool
# call ergibt einen VS Code Error, aber die Loop-Detection hat die bisher
# explizit ignoriert → Kimi kreiste ewig ohne zum Plan zu kommen.
# Threshold = 3: 3x Malformed in Folge = eindeutig verklemmt, nicht Glitch.
PLANNER_MALFORMED_HARD_STOP = 3

def _get_planner_session_hash(messages: Sequence[Dict[str, Any]]) -> str:
    """Hash aus der ersten User-Message (ohne Flags) für Session-Tracking."""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            text, _ = _extract_pipeline_flags(_message_text(msg))
            return _simple_hash(text[:300])
    return "unknown"

def _cleanup_old_planner_sessions(max_age_seconds: float = 600) -> None:
    """Entfernt Planner-Sessions älter als max_age_seconds."""
    now = time.time()
    stale = [h for h, s in _PLANNER_SESSIONS.items() if now - s.get("ts", 0) > max_age_seconds]
    for h in stale:
        del _PLANNER_SESSIONS[h]

PLAN_MARKERS = ("## Plan:", "## plan:", "# Plan:", "# plan:", "## CLOUD EXECUTION PLAN")

def _content_contains_plan(content: str) -> bool:
    """Prüft ob ein Text einen Plan enthält (Kimi hat fertig geplant)."""
    if not content or not isinstance(content, str):
        return False
    return any(m in content for m in PLAN_MARKERS)


# ── Loop-Detection-Helpers (Signature-basiert) ───────────────────────────
def _tool_signature(tool_call: Dict[str, Any]) -> str:
    """Erzeugt eine stabile Signatur für einen Tool-Call: 'name|arg_hash'.

   VERSCHIEDENE Tool-Namen, gleiche Args → verschiedene Signatur.
    GLEICHE Tool-Name, verschiedene Args → verschiedene Signatur.
    Nur exakt identisch (name+args) → gleiche Signatur → echter Loop.
    """
    func = tool_call.get("function") or {} if isinstance(tool_call, dict) else {}
    name = str(func.get("name", "?"))
    args_raw = func.get("arguments", "")
    try:
        # JSON-String → dict → kanonischer sortierter String → Hash
        if isinstance(args_raw, str):
            args_dict = json.loads(args_raw) if args_raw.strip() else {}
        elif isinstance(args_raw, dict):
            args_dict = args_raw
        else:
            args_dict = {"_raw": str(args_raw)}
        canonical = json.dumps(args_dict, sort_keys=True, ensure_ascii=False)
    except Exception:
        canonical = str(args_raw)[:200]
    arg_hash = _simple_hash(canonical)
    return f"{name}|{arg_hash}"


def _tool_call_has_args(tool_call: Dict[str, Any]) -> bool:
    """Prüft ob ein Tool-Call sinnvolle Argumente hat.

    Kimi generiert manchmal Tool-Calls mit leeren/ungültigen Argumenten
    (z.B. read_file ohne filePath, grep_search ohne query). Das sind
    Model-Glitches, KEINE echten Loop-Indikatoren. Diese Funktion
    erkennt solche invaliden Calls, damit sie aus der Loop-Erkennung
    ausgeschlossen werden können.
    """
    func = tool_call.get("function") or {} if isinstance(tool_call, dict) else {}
    args_raw = func.get("arguments", "")
    try:
        if isinstance(args_raw, str):
            if not args_raw.strip():
                return False
            args_dict = json.loads(args_raw)
        elif isinstance(args_raw, dict):
            args_dict = args_raw
        else:
            return False
    except Exception:
        return False
    # Leeres Dict nach JSON-Parsing → kein sinnvolles Argument
    return bool(args_dict)


def _extract_tool_file_refs(tool_calls: Optional[List[Dict[str, Any]]]) -> List[str]:
    """Extrahiert Datei-/Pfad-Referenzen aus Tool-Args als Fortschritts-Metrik.

    Gibt Liste normalisierter Pfade/Queries zurück (z.B. für read_file den
    filePath, für grep_search den includePattern/query-String).
    Verwendet für die 'distinct_files'-Menge → zeigt, ob Kimi NEUE
    Information aufdeckt oder nur im Kreis läuft.
    """
    refs: List[str] = []
    if not tool_calls:
        return refs
    interesting_keys = ("filepath", "filename", "path", "includepattern",
                        "query", "symbol", "uri", "filepath_relative")
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function") or {}
        name = str(func.get("name", "")).lower()
        args_raw = func.get("arguments", "")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) and args_raw.strip() else (args_raw if isinstance(args_raw, dict) else {})
        except Exception:
            args = {}
        for k, v in args.items():
            kl = k.lower()
            if kl in interesting_keys and v:
                refs.append(f"{name}:{str(v).lower()[:120]}")
    return refs


def _detect_planner_loop(session: Dict[str, Any]) -> Optional[str]:
    """Erkennt echte Loops im Planner anhand der Tool-Signaturen.

    Returns:
        None  → kein Loop, normal weitermachen
        str   → Loop-Grund (für Log + Force-Plan-Decision)
    """
    sigs = session.get("tool_signatures") or []
    if not sigs:
        # Sonderfall: keine validen Signaturen, dafür viele malformed Rounds
        # → Kimi versendet Tool-Calls ohne Args und VS Code blockt alles ab.
        malformed_runs = int(session.get("consecutive_malformed_rounds", 0))
        if malformed_runs >= PLANNER_MALFORMED_HARD_STOP:
            return f"malformed-loop: {malformed_runs} rounds/all-tool-calls-args-missing"

    # ── Malformed-Loop: N aufeinanderfolgende Runden in denen ALLE Tool-Calls
    #    invalide Args hatten (Kimi-Parallel-Batch-Bug). Das ist KEINE
    #    legitime Exploration - jeder Call erzeugt einen VS Code Error und
    #    Kimi wiederholt im nächsten Round dieselben immaculat Calls.
    malformed_runs = int(session.get("consecutive_malformed_rounds", 0))
    if malformed_runs >= PLANNER_MALFORMED_HARD_STOP:
        return f"malformed-loop: {malformed_runs} consecutive rounds with bad args"

    # ── Hard-Stop: 'PLANNER_REPEAT_HARD_STOP' identische Signaturen am Stück
    #    Klassischer Loop: 'No matches' → gleiches grep nochmal → ...
    from collections import Counter
    # Letzte N Signaturen prüfen (rolling window)
    last_n = sigs[-(PLANNER_REPEAT_HARD_STOP + 2):]  # etwas Kontext
    if len(last_n) >= PLANNER_REPEAT_HARD_STOP:
        # Zähle gleiche aufeinanderfolgende am Ende
        last_sig = last_n[-1]
        run = 1
        for s in reversed(last_n[:-1]):
            if s == last_sig:
                run += 1
                if run >= PLANNER_REPEAT_HARD_STOP:
                    return f"exact-repeat: {last_sig} x{run}"
            else:
                break

    # ── Stagnation: viele Runden, fast keine neuen Files entdeckt
    distinct_files = session.get("distinct_files") or set()
    iterations = int(session.get("iterations", 0))
    if iterations >= PLANNER_DISTINCT_WARN_AFTER:
        # In den letzten 6 Signaturen < 2 neue Files → stagniert
        recent_sigs = set(sigs[-6:])
        if len(recent_sigs) <= 2:
            return f"stagnation: {len(recent_sigs)} distinct tools in last 6 rounds"

    # ── Exploration-only-Loop: viele Runden, aber keine neuen Files/Queries
    #    mehr entdeckt. Das ist das klassische Kimi-Kreisen: immer tiefer
    #    in denselben Dateien graben, ohne zur Konklusion zu kommen.
    #    Forciere Plan-Output NUR wenn die Exploration stagniert.
    #    Threshold = PLANNER_DISTINCT_WARN_AFTER (18) + Stagnation,
    #    denn davor greift _should_warn_planner (12) mit sanftem Nudge.
    if iterations >= PLANNER_DISTINCT_WARN_AFTER and len(distinct_files) >= 5:
        # Stagnation: In den letzten Runden keine neuen Dateien/Queries
        # entdeckt. Wenn distinct_files ≈ iterations, ist Kimi noch
        # produktiv und sollte weiter explorieren dürfen.
        if len(distinct_files) < iterations - 3:  # 3+ Runden ohne neuen Content
            contents = session.get("assistant_contents") or []
            if not any("## plan:" in str(c).lower()[:200] for c in contents[-4:]):
                return (f"exploration-only-loop: {iterations} rounds, "
                        f"{len(distinct_files)} distinct files, no plan yet")

    # ── Panik-Tripwire: ARBITRARY HARD-CAP nur noch reiner Safety-Net
    if iterations >= MAX_PLANNER_ITERATIONS:
        return f"hard-tripwire: {iterations}/{MAX_PLANNER_ITERATIONS}"

    return None


def _should_warn_planner(session: Dict[str, Any]) -> Optional[str]:
    """Sanftes Nudge statt Hard-Stop. Gibt Warn-Text zurück oder None.

    Wird NICHT in _call_cloud_planner_agent angewendet (dort sind tools ohnehin
    aktiv), sondern nur als Injekt am System-Prompt im Force-Plan-Branch,
    damit Kimi nach angemessener Exploration selbst den Plan ausspuckt.
    """
    iterations = int(session.get("iterations", 0))
    if iterations < PLANNER_WARN_AFTER:
        return None
    distinct_files = session.get("distinct_files") or set()
    sigs = session.get("tool_signatures") or []
    distinct_sigs = len(set(sigs))
    if iterations >= PLANNER_WARN_AFTER and len(sigs) >= PLANNER_WARN_AFTER:
        return (f"[SYSTEM NUDGE: Du hast {iterations} Runden Tools ausgeführt "
                f"({distinct_sigs} distinct, {len(distinct_files)} Dateien/Queries). "
                "Wenn du genug Kontext hast, gib JETZT den finalen Plan aus. "
                "Ansonsten explorieren - aber zügig.]")
    return None


def _extract_last_assistant_tool_calls(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extrahiert die tool_calls des letzten assistant-Message-Paars.

    Planner-Sessions empfangen Tool-Resultate via `tool`-Rollen in der History.
    Vor jedem Tool-Resultat-Block steht eine assistant-Message mit tool_calls.
    Diese Funktion findet die zuletzt eingetroffenen tool_calls (also die aus
    der VORIGEN Planner-Runde, deren Resultate gerade eingetroffen sind).
    """
    if not messages:
        return []
    # Von hinten nach dem letzten assistant mit tool_calls suchen
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tcs = msg.get("tool_calls")
            return tcs if isinstance(tcs, list) else []
        # Sobald wir ein 'tool'-Ergebnis passiert haben und eine User-Nachricht
        # sehen, sind wir aus dem Tool-Continuation-Block raus → abbrechen.
        if msg.get("role") == "user":
            break
    return []


# ── Display-Namen ──────────────────────────────────────────────────────────
DISPLAY_NAMES: Dict[str, str] = {
    "architect": "Cloud-Planer",
    "worker": f"Lokaler Worker ({MODEL_NAME})",
    "fast_worker": f"Lokaler Fast Worker ({FAST_MODEL_NAME})",
    "direct": "Direkt Lokal",
    "memory": "Hindsight Memory",
    "caveman": "Caveman Plan",
    "cloud_reviewer": "Cloud Reviewer",
    "verifier": "Verifier",
}


# ═══════════════════════════════════════════════════════════════════════════
# Config-Datei-Loader — übernimmt Werte aus config.json (via webui._load_config)
# ═══════════════════════════════════════════════════════════════════════════

def _apply_config_file() -> None:
    """Übernimmt Werte aus config.json.
    
    webui._load_config() merged: Hardcoded-Defaults ← config.json ← Env-Vars (nur beim 1. Start).
    Ab dem ersten WebUI-Save hat config.json Vorrang – auch vor Coolify-Env-Vars.
    """
    if not _WEBUI_AVAILABLE:
        return
    try:
        cfg = _webui_load_config()
    except Exception:
        return

    # Models
    global VLLM_API_URL, VLLM_API_KEY, MODEL_NAME, FAST_MODEL_NAME
    VLLM_API_URL = cfg.get("models", {}).get("vllm_api_url", VLLM_API_URL)
    ak = cfg.get("models", {}).get("vllm_api_key", "")
    if ak:
        VLLM_API_KEY = ak
    MODEL_NAME = cfg.get("models", {}).get("model_name", MODEL_NAME)
    FAST_MODEL_NAME = cfg.get("models", {}).get("fast_model_name", FAST_MODEL_NAME)

    # Proxy
    global PROXY_PORT, PROXY_AUTH_ENABLED, PROXY_API_KEY, CHATTY_MODE
    PROXY_PORT = int(cfg.get("proxy", {}).get("port", PROXY_PORT))
    PROXY_AUTH_ENABLED = bool(cfg.get("proxy", {}).get("auth_enabled", PROXY_AUTH_ENABLED))
    pk = cfg.get("proxy", {}).get("api_key", "")
    if pk:
        PROXY_API_KEY = pk
    CHATTY_MODE = bool(cfg.get("proxy", {}).get("chatty_mode", CHATTY_MODE))

    # Cloud
    global CLOUD_REVIEW_ENABLED, CLOUD_REVIEW_API_URL, CLOUD_REVIEW_API_KEY
    global CLOUD_REVIEW_MODEL, CLOUD_REVIEW_MAX_TOKENS, CLOUD_REVIEW_TIMEOUT_SECONDS
    CLOUD_REVIEW_ENABLED = bool(cfg.get("cloud", {}).get("enabled", CLOUD_REVIEW_ENABLED))
    CLOUD_REVIEW_API_URL = cfg.get("cloud", {}).get("api_url", CLOUD_REVIEW_API_URL)
    ck = cfg.get("cloud", {}).get("api_key", "")
    if ck:
        CLOUD_REVIEW_API_KEY = ck
    CLOUD_REVIEW_MODEL = cfg.get("cloud", {}).get("model", CLOUD_REVIEW_MODEL)
    CLOUD_REVIEW_MAX_TOKENS = int(cfg.get("cloud", {}).get("max_tokens", CLOUD_REVIEW_MAX_TOKENS))
    CLOUD_REVIEW_TIMEOUT_SECONDS = float(cfg.get("cloud", {}).get("timeout_seconds", CLOUD_REVIEW_TIMEOUT_SECONDS))

    # LiteLLM
    global LITELLM_CLOUD_MODEL, LITELLM_CLOUD_API_KEY, LITELLM_CLOUD_API_URL
    global LITELLM_CLOUD_MAX_TOKENS, LITELLM_CLOUD_TIMEOUT_SECONDS
    LITELLM_CLOUD_MODEL = cfg.get("litellm", {}).get("model", LITELLM_CLOUD_MODEL)
    lk = cfg.get("litellm", {}).get("api_key", "")
    if lk:
        LITELLM_CLOUD_API_KEY = lk
    LITELLM_CLOUD_API_URL = cfg.get("litellm", {}).get("api_url", LITELLM_CLOUD_API_URL)
    LITELLM_CLOUD_MAX_TOKENS = int(cfg.get("litellm", {}).get("max_tokens", LITELLM_CLOUD_MAX_TOKENS))
    LITELLM_CLOUD_TIMEOUT_SECONDS = float(cfg.get("litellm", {}).get("timeout_seconds", LITELLM_CLOUD_TIMEOUT_SECONDS))

    # Tokens
    global DEFAULT_DIRECT_MAX_TOKENS, DEFAULT_AGENT_MAX_TOKENS
    global SUB_AGENT_TIMEOUT_SECONDS, VERIFY_TIMEOUT_SECONDS
    DEFAULT_DIRECT_MAX_TOKENS = int(cfg.get("tokens", {}).get("direct_max_tokens", DEFAULT_DIRECT_MAX_TOKENS))
    DEFAULT_AGENT_MAX_TOKENS = int(cfg.get("tokens", {}).get("agent_max_tokens", DEFAULT_AGENT_MAX_TOKENS))
    raw = cfg.get("tokens", {}).get("sub_agent_timeout_seconds", SUB_AGENT_TIMEOUT_SECONDS)
    SUB_AGENT_TIMEOUT_SECONDS = max(float(raw), 120.0)
    _log(f"  ⏱ SUB_AGENT_TIMEOUT_SECONDS = {SUB_AGENT_TIMEOUT_SECONDS:.0f}s (cfg raw={raw})")
    VERIFY_TIMEOUT_SECONDS = float(cfg.get("tokens", {}).get("verify_timeout_seconds", VERIFY_TIMEOUT_SECONDS))

    # Caveman
    global CAVEMAN_ENABLED, CAVEMAN_MAX_TOKENS
    CAVEMAN_ENABLED = bool(cfg.get("caveman", {}).get("enabled", CAVEMAN_ENABLED))
    CAVEMAN_MAX_TOKENS = int(cfg.get("tokens", {}).get("caveman_max_tokens", CAVEMAN_MAX_TOKENS))

    # Tool-Result-Cap (verhindert Token-Bombing)
    global TOOL_RESULT_CAP
    TOOL_RESULT_CAP = int(cfg.get("tokens", {}).get("tool_result_cap", TOOL_RESULT_CAP))
    _log(f"  ✂ TOOL_RESULT_CAP = {TOOL_RESULT_CAP} chars (pro tool-result-message)")

    # Hindsight
    global HINDSIGHT_ENABLED, QDRANT_URL, QDRANT_API_KEY, HINDSIGHT_COLLECTION
    global HINDSIGHT_EMBEDDING_DIM, HINDSIGHT_MAX_MEMORY_TOKENS, HINDSIGHT_MIN_SIMILARITY
    global HINDSIGHT_RETAIN_DELAY_SECONDS, HINDSIGHT_USE_QDRANT, HINDSIGHT_DIR
    HINDSIGHT_ENABLED = bool(cfg.get("hindsight", {}).get("enabled", HINDSIGHT_ENABLED))
    QDRANT_URL = cfg.get("hindsight", {}).get("qdrant_url", QDRANT_URL)
    qk = cfg.get("hindsight", {}).get("qdrant_api_key", "")
    if qk:
        QDRANT_API_KEY = qk
    HINDSIGHT_COLLECTION = cfg.get("hindsight", {}).get("collection", HINDSIGHT_COLLECTION)
    val_ed = cfg.get("hindsight", {}).get("embedding_dim", HINDSIGHT_EMBEDDING_DIM)
    HINDSIGHT_EMBEDDING_DIM = int(val_ed) if not isinstance(val_ed, int) else val_ed
    val_mt = cfg.get("hindsight", {}).get("max_memory_tokens", HINDSIGHT_MAX_MEMORY_TOKENS)
    HINDSIGHT_MAX_MEMORY_TOKENS = int(val_mt) if not isinstance(val_mt, int) else val_mt
    val_ms = cfg.get("hindsight", {}).get("min_similarity", HINDSIGHT_MIN_SIMILARITY)
    HINDSIGHT_MIN_SIMILARITY = float(val_ms) if not isinstance(val_ms, (int, float)) else val_ms
    val_rd = cfg.get("hindsight", {}).get("retain_delay_seconds", HINDSIGHT_RETAIN_DELAY_SECONDS)
    HINDSIGHT_RETAIN_DELAY_SECONDS = float(val_rd) if not isinstance(val_rd, (int, float)) else val_rd
    HINDSIGHT_USE_QDRANT = bool(cfg.get("hindsight", {}).get("use_qdrant", HINDSIGHT_USE_QDRANT))
    HINDSIGHT_DIR = Path(cfg.get("hindsight", {}).get("dir", str(HINDSIGHT_DIR)))

    # Verify
    global VERIFY_ENABLED, VERIFY_LINT_COMMAND, VERIFY_TEST_COMMAND
    VERIFY_ENABLED = bool(cfg.get("verify", {}).get("enabled", VERIFY_ENABLED))
    VERIFY_LINT_COMMAND = cfg.get("verify", {}).get("lint_command", VERIFY_LINT_COMMAND)
    VERIFY_TEST_COMMAND = cfg.get("verify", {}).get("test_command", VERIFY_TEST_COMMAND)

    # MCP
    global MCP_ENABLED
    MCP_ENABLED = bool(cfg.get("mcp", {}).get("enabled", MCP_ENABLED))


_apply_config_file()

# ── URL-Normalisierung ──────────────────────────────────────────────────
_api_base_clean = VLLM_API_URL.rstrip("/")
if not _api_base_clean.endswith("/chat/completions"):
    if _api_base_clean.endswith("/v1"):
        # /v1 ist bereits ein gültiger OpenAI-kompatibler Endpunkt — nichts anhängen
        _log(f"ℹ️  URL-Norm: VLLM_API_URL endet auf /v1 – wird als vollständiger Endpunkt verwendet")
    else:
        VLLM_API_URL = _api_base_clean + "/chat/completions"
        _log(f"🔧 URL-Norm: VLLM_API_URL → {VLLM_API_URL}")


def _derive_models_url(api_url: str) -> str:
    """Leitet die /models-URL aus der Chat-API-URL ab."""
    base = api_url.rstrip("/")
    if "/chat/completions" in base:
        return base.rsplit("/chat/completions", 1)[0] + "/models"
    if base.endswith("/v1"):
        return base + "/models"
    return base + "/v1/models"

# ── Cloud-URL-Normalisierung ────────────────────────────────────────────
# Moonshot: /v1 → /v1/chat/completions, DeepSeek: direkt /chat/completions
_cloud_clean = CLOUD_REVIEW_API_URL.rstrip("/")
if not _cloud_clean.endswith("/chat/completions"):
    if _cloud_clean.endswith("/v1"):
        CLOUD_REVIEW_API_URL = _cloud_clean + "/chat/completions"
    else:
        _log(f"ℹ️  Cloud-URL '{CLOUD_REVIEW_API_URL}' endet nicht auf /chat/completions – wird trotzdem verwendet")
    _log(f"🔧 URL-Norm: CLOUD_REVIEW_API_URL → {CLOUD_REVIEW_API_URL}")

# ═══════════════════════════════════════════════════════════════════════════
# Reasoning-Content Cache (für Tool-Continuations)
# ═══════════════════════════════════════════════════════════════════════════
# DeepSeek erfordert, dass reasoning_content bei Folgerequests mit tool_calls
# im Assistant-Message erhalten bleibt. VS Code kennt dieses Feld nicht und
# sendet es nicht zurück — also cachen wir es und injizieren es automatisch.

_reasoning_cache: Dict[str, str] = {}  # tool_call_id → reasoning_content

def _cache_reasoning(tool_calls: Any, reasoning_content: Optional[str]) -> None:
    """Speichert reasoning_content, keyed by tool_call_id."""
    if not reasoning_content or not tool_calls:
        return
    for tc in tool_calls:
        tc_id = tc.get("id") or tc.get("function", {}).get("name", "")
        if tc_id:
            _reasoning_cache[tc_id] = reasoning_content

def _inject_reasoning_from_cache(messages: List[Dict[str, Any]]) -> None:
    """Inject missing reasoning_content into assistant messages with tool_calls."""
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        if not msg.get("tool_calls"):
            continue
        if msg.get("reasoning_content"):
            continue  # schon vorhanden
        # reasoning_content fehlt → aus Cache holen
        for tc in msg["tool_calls"]:
            tc_id = tc.get("id") or tc.get("function", {}).get("name", "")
            cached = _reasoning_cache.get(tc_id)
            if cached:
                msg["reasoning_content"] = cached
                break


# ═══════════════════════════════════════════════════════════════════════════
# Plan-Persistenz (Codespace-Copilot-Stil)
# ═══════════════════════════════════════════════════════════════════════════
# Pläne werden unter data/plans/Plan_<hash>.md gespeichert.
# Der Worker-Payload referenziert die Plan-Datei zusätzlich (Pointer + Inhalt),
# sodass DeepSeek robust an den Plan kommt - egal, ob die JSON-Wrapping-Ebene
# der Conversations-History aus Kimi-Tool-Runden das Wandern übersteht.

def _save_plan_to_file(session_hash: str, plan: str, query: str = "") -> Path:
    """Persistiert einen Plan als Markdown-Datei in PLANS_DIR.

    Returns: Pfad zur geschriebenen Datei.
    """
    try:
        PLANS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        _log(f"  ⚠ PLANS_DIR.mkdir fehlgeschlagen ({exc}) – retour in-memory")
        return None
    safe_hash = "".join(c for c in str(session_hash) if c.isalnum() or c in "-_")[:20] or "unknown"
    plan_path = PLANS_DIR / f"Plan_{safe_hash}.md"
    # Header mit Task-Kurzbeschreibung für Debugging
    task_short = (query or "").strip().split("\n")[0][:120]
    header = (
        f"# Plan\n\n"
        f"- **Session**: `{safe_hash}`\n"
        f"- **Task**: {task_short or '(unbenannt)'}\n"
        f"- **Created**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- **Planner**: Cloud-Planner (Kimi K2.7)\n\n"
        f"---\n\n"
    )
    try:
        plan_path.write_text(header + plan, encoding="utf-8")
        _log(f"  📄 Plan persistiert: {plan_path} ({len(plan)} chars)")
        return plan_path
    except Exception as exc:
        _log(f"  ⚠ Plan-File-Write fehlgeschlagen ({exc}) – In-Monly")
        return None


def _load_plan_from_file(session_hash: str) -> Optional[str]:
    """Liest den gespeicherten Plan für eine Session, falls vorhanden."""
    safe_hash = "".join(c for c in str(session_hash) if c.isalnum() or c in "-_")[:20] or "unknown"
    plan_path = PLANS_DIR / f"Plan_{safe_hash}.md"
    if not plan_path.exists():
        return None
    try:
        return plan_path.read_text(encoding="utf-8")
    except Exception as exc:
        _log(f"  ⚠ Plan-File-Read fehlgeschlagen: {exc}")
        return None


def _strip_kimi_reasoning(messages: List[Dict[str, Any]]) -> int:
    """Entfernt reasoning_content aus ALLEN Messages für Worker-Calls.

    Hintergrund: Kimi (Cloud-Planner) liefert `reasoning_content`-Thinking.
    VS Code gibt dieses Feld jedoch nicht zurück. DeepSeek (Worker) empfindet
    fremdes reasoning_content in der History als Kontaminierung – was zu
    'thinking content could not be passed' Style-Fehlern führt.

    Strategie:
      - Assistant-Messages MIT tool_calls → reasoning_content DRINLASSEN
        (DeepSeek braucht das für Tool-Continuations via _inject_reasoning).
      - Alle anderen Messages → reasoning_content entfernen
        (insbesondere reine Plan-Finalisierung ohne tool_calls von Kimi).

    Returns: Anzahl entfernter reasoning_content-Felder.
    """
    removed = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        if msg.get("tool_calls"):
            continue  # Tool-aktive Assistant → reasoning drinlassen
        if msg.get("reasoning_content"):
            msg.pop("reasoning_content", None)
            removed += 1
    return removed


# ═══════════════════════════════════════════════════════════════════════════
# DEBUG-Infrastructure — Punchline: VS Code hosts and we need full visibility
# ═══════════════════════════════════════════════════════════════════════════
# Problem: In VS Code-Copilot-Tests sieht's lokal toll aus, in VSCode läuft
# alles schief (lang, hängend, falsche Prompts). _log() zeigt nur Summaries,
# aber die PAYLOADS und Tool-Continuations sind ein Blackbox.
#
# Lösung 1: Request-Dumps in data/debug/<req_id>__<phase>.json
#   Pro Request/Phase wird der OUTBOUND-Payload (an Kimi/DeepSeek) als JSON
#   gespeichert. So siehst du GENAU, was der Proxy rausschickt.
#
# Lösung 2: In-Memory Request-Buffer (letzte N Requests mit High-Level-Daten).
#
# Lösung 3: REST-Endpoint /debug/* für Remote-Inspection aus VSCode.
#
# Lösung 4: Active-call-tracking + Heartbeat ("Kimi hängt seit X").

DEBUG_DIR: Path = Path(os.getenv("DEBUG_DIR", "./data/debug"))
DEBUG_MAX_FILES: int = int(os.getenv("DEBUG_MAX_FILES", "200"))
DEBUG_ENABLED: bool = os.getenv("DEBUG_ENABLED", "1").lower() in {"1", "true", "yes", "on"}

# In-Memory Ring-Buffer der letzten N Requests
_DEBUG_RING: List[Dict[str, Any]] = []
_DEBUG_RING_MAX: int = int(os.getenv("DEBUG_RING_MAX", "50"))

# Aktuell laufende Planner-Calls (für "Kimi hängt seit X" Visibility)
_ACTIVE_CALLS: Dict[str, Dict[str, Any]] = {}  # call_id -> metadata


def _dump_debug_payload(req_id: str, phase: str, payload_to_dump: Dict[str, Any],
                         extra: Optional[Dict[str, Any]] = None) -> None:
    """Schreibt einen Snapshot des Outbound-Payloads als JSON ins DEBUG_DIR.

    Phasen-Beispiele: planner_r1, planner_r2, worker, fallback, feedback_r1.
    Dateinamen sind sortierbar nach Req-ID + Phase.
    + Groß-Inhalte kappen, sonst explodiert das File.
    """
    if not DEBUG_ENABLED:
        return
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    import copy as _copy
    snapshot: Dict[str, Any] = {"dumped_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        cloned = _copy.deepcopy(payload_to_dump)
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
                    "idx": i,
                    "role": m.get("role"),
                    "content_len": content_len,
                    "content_preview": content_preview,
                    "has_tool_calls": bool(tcs),
                    "tool_calls_count": len(tcs) if isinstance(tcs, list) else 0,
                }
                if isinstance(tcs, list):
                    summary["tool_call_names"] = [
                        (t.get("function", {}).get("name") if isinstance(t, dict) else None)
                        for t in tcs
                    ]
                    # NEU: tool_call-args mit speichern (cap bei 300 chars/call)
                    # OHNE das würde der Root-Cause von Args-Verlust im Stream-
                    # Pipeline unsichtbar bleiben.
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
    """Räumt alte Debug-Files auf, wenn DEBUG_MAX_FILES überschritten."""
    if not DEBUG_DIR.exists():
        return
    try:
        files = sorted(DEBUG_DIR.glob("*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if len(files) > DEBUG_MAX_FILES:
            for f in files[DEBUG_MAX_FILES:]:
                try:
                    f.unlink()
                except Exception:
                    pass
    except Exception:
        pass


def _register_debug_request(req_id: str, info: Dict[str, Any]) -> None:
    """Fügt einen Request in den In-Memory Ring-Buffer ein."""
    global _DEBUG_RING
    info = dict(info)
    info["req_id"] = req_id
    info["ts_iso"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _DEBUG_RING.append(info)
    if len(_DEBUG_RING) > _DEBUG_RING_MAX:
        _DEBUG_RING = _DEBUG_RING[-_DEBUG_RING_MAX:]


def _register_active_call(call_id: str, info: Dict[str, Any]) -> None:
    """Trackt aktive Langläufer (Cloud-Planner-Calls) für Heartbeat/Visibility."""
    info = dict(info)
    info["started_at"] = time.time()
    info["started_iso"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _ACTIVE_CALLS[call_id] = info
    asyncio.ensure_future(_active_call_heartbeat(call_id))


async def _active_call_heartbeat(call_id: str) -> None:
    """Loggt alle 30s, dass ein aktiver Call noch läuft."""
    while call_id in _ACTIVE_CALLS:
        await asyncio.sleep(30)
        info = _ACTIVE_CALLS.get(call_id)
        if not info:
            return
        elapsed = time.time() - info.get("started_at", time.time())
        _log(f"  ⏳[{call_id}] aktiver Call seit {elapsed:.0f}s "
             f"(agent_key={info.get('agent_key','?')}, "
             f"model={info.get('model','?')}, "
             f"phase={info.get('phase','?')})")


def _finish_active_call(call_id: str, status: str = "done",
                          extra: Optional[Dict[str, Any]] = None) -> None:
    """Beendet einen aktiven Call (entfernt aus _ACTIVE_CALLS)."""
    info = _ACTIVE_CALLS.pop(call_id, None)
    if not info:
        return
    elapsed = time.time() - info.get("started_at", time.time())
    _log(f"  ✓[{call_id}] Call beendet nach {elapsed:.0f}s ({status})")
    summary = {"call_id": call_id, "elapsed_seconds": elapsed, "status": status, **info}
    if extra:
        summary.update(extra)


# ═══════════════════════════════════════════════════════════════════════════
# Hindsight Memory Netzwerke (4 logische Ebenen)
# ═══════════════════════════════════════════════════════════════════════════

HINDSIGHT_NETWORKS = [
    "World Facts",
    "Agent Experiences",
    "Entity Summaries",
    "Evolving Beliefs",
]


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


# ═══════════════════════════════════════════════════════════════════════════
# Hindsight Memory Engine (Qdrant + JSONL-Fallback)
# ═══════════════════════════════════════════════════════════════════════════

class HindsightMemory:
    """Hindsight Persistent Memory mit Qdrant (primär) oder JSONL (Fallback)."""

    def __init__(self) -> None:
        self._qdrant: Optional[QdrantClient] = None
        self._use_qdrant = HINDSIGHT_USE_QDRANT and HINDSIGHT_ENABLED
        if self._use_qdrant:
            try:
                self._qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
                self._ensure_collection()
                _log(f"🧠 Hindsight: Qdrant connected @ {QDRANT_URL}")
            except Exception as exc:
                _log(f"⚠️  Qdrant nicht erreichbar ({exc}), fallback auf JSONL")
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
            _log(f"⚠️  Qdrant recall fehlgeschlagen: {exc}")
            return []

    def _recall_jsonl(self, query: str, min_similarity: float, _limit: int) -> List[MemoryRecord]:
        records = _load_memory_records()
        scored = []
        ms = float(min_similarity)  # config.json speichert manchmal strings
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
            _log(f"⚠️  Qdrant retain fehlgeschlagen: {exc}")

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


# ── Globale Hindsight-Instanz ──────────────────────────────────────────────
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


# ── Pipeline-Steuerflags ─────────────────────────────────────────────────
# Prompt-Zusätze, die die Pipeline manuell steuern:
#   -force planning   → Cloud-Planung erzwingen (Phase 1)
#   -force review     → Cloud-Review/Verifikation erzwingen (Phase 3)
#   -bypass worker    → Lokalen Worker (80B) überspringen, Cloud antwortet direkt

_PIPELINE_FLAG_PATTERN = re.compile(
    r'(-\s*(?:force|bypass)\s*(?:planning|review|worker))',
    re.IGNORECASE,
)

_FLAG_ALIASES = {
    "force-planning": "force_planning",
    "force planning": "force_planning",
    "force-review": "force_review",
    "force review": "force_review",
    "bypass-worker": "bypass_worker",
    "bypass worker": "bypass_worker",
}


def _extract_pipeline_flags(text: str) -> Tuple[str, Dict[str, bool]]:
    """Erkennt Pipeline-Steuerflags im Text und gibt bereinigten Text + Flags zurück."""
    flags: Dict[str, bool] = {
        "force_planning": False,
        "force_review": False,
        "bypass_worker": False,
    }
    cleaned = text
    for match in _PIPELINE_FLAG_PATTERN.finditer(text):
        raw = match.group(1).strip().lower().replace(" ", "-").replace("--", "-")
        # Normalisieren: "-force-planning" → "force_planning"
        key = _FLAG_ALIASES.get(raw.replace("-", " ").strip(), raw.replace("-", "_"))
        if key in flags:
            flags[key] = True
    # Entferne alle Flags aus dem Text
    for flag_text in _FLAG_ALIASES:
        # Case-insensitive Replacement
        cleaned = re.sub(re.escape(flag_text), "", cleaned, flags=re.IGNORECASE)
    # Cleanup: mehrfache Leerzeichen & leere Zeilen
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()
    return cleaned, flags


def _strip_pipeline_flags_from_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Entfernt Pipeline-Steuerflags aus allen User-Messages (in-place)."""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                cleaned, _ = _extract_pipeline_flags(content)
                msg["content"] = cleaned
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        cleaned, _ = _extract_pipeline_flags(block.get("text", ""))
                        block["text"] = cleaned
    return messages


def _has_pipeline_flag(messages: Sequence[Dict[str, Any]], flag: str) -> bool:
    """Prüft ob ein bestimmtes Pipeline-Flag in den Messages gesetzt ist."""
    text = _last_user_text(messages)
    _, flags = _extract_pipeline_flags(text)
    return flags.get(flag, False)


def _last_user_text(messages: Sequence[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return _message_text(msg)
    return ""


def _debug_log_request(body: Dict[str, Any]) -> None:
    """Loggt detailliert, was Copilot an den Proxy sendet."""
    msgs = body.get("messages", [])
    tools = body.get("tools", [])
    tool_choice = body.get("tool_choice")

    # Tool-Defs
    tool_names = [t.get("function", {}).get("name", "?") for t in tools] if tools else []
    _log(f"🔍 DEBUG REQUEST: {len(msgs)} messages, {len(tool_names)} tools={tool_names}, "
         f"tool_choice={tool_choice}, stream={body.get('stream')}")

    # Letzte 5 Messages (Rolle + Länge + erste 300 Zeichen)
    for i, m in enumerate(msgs[-5:], len(msgs)-4):
        role = m.get("role", "?")
        content = _message_text(m)
        name = m.get("name", "")
        tool_call_id = m.get("tool_call_id", "")
        extra = f" name={name}" if name else ""
        extra += f" tc_id={tool_call_id[:20]}" if tool_call_id else ""
        content_preview = content[:300].replace("\n", "\\n")
        _log(f"  [{i}] {role}{extra}: len={len(content)} | {content_preview}")

    # System-Prompt: Tool-Defs aus System-Message extrahieren
    for m in msgs:
        if m.get("role") == "system":
            sys_text = _message_text(m)
            # Zeilen mit function/tool patterns finden
            tool_lines = [l for l in sys_text.split("\n") if "function" in l.lower() or "tool" in l.lower()][:20]
            if tool_lines:
                _log(f"  🛠️ System-Prompt Tool-Lines ({len(tool_lines)}):")
                for tl in tool_lines[:10]:
                    _log(f"    {tl[:200]}")
            break


def _debug_log_thinking(result: Dict[str, Any], agent_key: str) -> None:
    """Loggt DeepSeeks Thinking/Reasoning aus der Response."""
    message = result.get("message") or {}
    reasoning = message.get("reasoning_content", "")
    content = result.get("content", "")
    tool_calls = result.get("tool_calls")

    if reasoning:
        _log(f"  🧠 DEEPSEEK THINKING ({agent_key}, {len(reasoning)} chars):")
        # Letzte 2000 Zeichen (da kommt meist die Schlussfolgerung)
        tail = reasoning[-2000:] if len(reasoning) > 2000 else reasoning
        for line in tail.replace("\r", "").split("\n")[-30:]:
            _log(f"    {line[:250]}")

    if tool_calls:
        for i, tc in enumerate(tool_calls):
            func = tc.get("function", {})
            _log(f"  🔧 TOOL-CALL[{i}]: {func.get('name')}({func.get('arguments','')[:300]})")

    if not reasoning and not tool_calls and content:
        _log(f"  📝 RESPONSE ({agent_key}, {len(content)} chars): {content[:500].replace(chr(10),' ')}")



def _build_rich_context(messages: Sequence[Dict[str, Any]], max_chars: int = 14000) -> str:
    """Baut umfassenden Caveman-Kontext aus ALLEN Messages für Cloud-Planner/Reviewer.

    Extrahiert: System-Prompt (Agent-Instruktionen, Tool-Defs, Coding-Rules),
    Tool-Resultate (ENTHALTEN DIE WORKSPACE-DATEIEN aus read_file!),
    Konversations-Historie und den aktuellen User-Task.
    Alles in Caveman-Format komprimiert.
    """
    sections: List[str] = []

    # 1. System-Prompt (Agent-Instruktionen, Tool-Defs, Coding-Guidelines)
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            sys_text = _message_text(msg)
            if sys_text:
                sections.append(f"[AGENT INSTRUCTIONS & TOOLS]\n{_token_budget_guard(sys_text, 5000)}")
            break

    # 2. Tool-Resultate (ENTHALTEN WORKSPACE-DATEIINHALTE!)
    tool_results: List[str] = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            content = _message_text(msg)
            if content:
                tool_name = str(msg.get("name", "") or msg.get("tool_call_id", "tool"))[:40]
                tool_results.append(f"[TOOL: {tool_name}]\n{_token_budget_guard(content, 3000)}")
    if tool_results:
        sections.append(f"[WORKSPACE FILES ({len(tool_results)} tool results, newest last)]\n" + "\n---\n".join(tool_results[-10:]))

    # 3. Konversations-Historie (User/Assistant)
    history: List[str] = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") in ("user", "assistant"):
            content = _message_text(msg)
            if content:
                role = str(msg["role"]).upper()
                history.append(f"[{role}]\n{_token_budget_guard(content, 2000)}")
    if history:
        sections.append(f"[CONVERSATION HISTORY ({len(history)} messages)]\n" + "\n---\n".join(history[-8:]))

    combined = "\n\n".join(sections)
    return _token_budget_guard(combined, max_chars)


def _compact_messages(messages: Sequence[Dict[str, Any]], max_messages: int = 12) -> List[Dict[str, Any]]:
    return list(messages[-max_messages:])


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


def _extract_choice_message(result: Dict[str, Any]) -> Dict[str, Any]:
    choices = result.get("choices", [])
    if not choices:
        return {}
    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return {}
    # Qwen3/Aeon: wenn content null ist, aber reasoning existiert → reasoning als content verwenden
    # (passiert bei --reasoning-parser qwen3 ohne reasoning_effort=none)
    if message.get("content") is None and message.get("reasoning"):
        message = dict(message)
        message["content"] = message["reasoning"]
        message["reasoning"] = None
    return message


def _extract_choice_content(result: Dict[str, Any]) -> str:
    message = _extract_choice_message(result)
    content = message.get("content")
    return content or ""


def _sum_usage(results: List[Dict[str, Any]]) -> Dict[str, int]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    for r in results:
        usage = r.get("usage") or {}
        try:
            prompt_tokens += int(usage.get("prompt_tokens", 0))
            completion_tokens += int(usage.get("completion_tokens", 0))
            total_tokens += int(usage.get("total_tokens", 0))
        except (TypeError, ValueError):
            pass
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens}


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


# ═══════════════════════════════════════════════════════════════════════════
# Intent-Klassifizierung (deterministisch + optional Fast Model)
# ═══════════════════════════════════════════════════════════════════════════

def _classify_intent_deterministic(text: str) -> Optional[str]:
    """Deterministische Intent-Erkennung via Trigger-Wörter."""
    t = text.lower()

    # AGENT zuerst prüfen — Agent-Intent hat Vorrang
    if any(trigger in t for trigger in AGENT_TRIGGER_WORDS):
        return "agent"

    # Lange Texte (>500 Zeichen) immer als Agent behandeln
    if len(t) > 500:
        return "agent"

    # DIRECT nur wenn KEIN Agent-Wort gefunden wurde
    if any(trigger in t for trigger in DIRECT_TRIGGER_WORDS):
        return "direct"

    # Kurze Texte ohne klare Trigger → direct
    if len(t) < 240:
        return "direct"

    # Mittellange Texte ohne Trigger → Agent (vorsichtshalber)
    return "agent"


def _is_tool_continuation(messages: Sequence[Dict[str, Any]]) -> bool:
    """True nur wenn Modell aktiv Tool-Calls verarbeitet:
    - Letzte Message ist assistant mit tool_calls (Modell hat Tools requested)
    - Letzte Message ist tool (Client hat soeben Tool-Resultate zurückgeschickt)
    NICHT: wenn irgendwo in der History mal ein Tool vorkam."""
    if not messages:
        return False
    last = messages[-1]
    if isinstance(last, dict) and last.get("role") == "tool":
        return True
    if isinstance(last, dict) and last.get("role") == "assistant" and bool(last.get("tool_calls")):
        return True
    return False


def _contains_tool_calls(text: str) -> bool:
    """Erkennt Tool-Call-Markup inkl. VS Code/Copilot DSML-Format."""
    if not text:
        return False
    return bool(
        re.search(r'</?(?:tool_call|tool_calls|invoke|function_call)', text)
        or re.search(r'<[a-z_]+_tool', text)
        or "callTool" in text
        or "DSML" in text
        or "｜｜tool_calls" in text
        or "｜｜invoke" in text
        or "<｜｜DSML｜｜tool_calls>" in text
    )


# ── Tool-Call-Normalisierung (kritisch für Cloud-Planner mit VS Code Tools) ──
# Bug-B-Fix: Manche Cloud-Modelle (insb. Moonshot/Kimi) liefern tool_calls
# mit Arguments die KEIN gültiger JSON-String sind (dict, leer, malformed).
# VS Code Copilot (OpenAI-Standard) verlangt strictly-typed JSON-String-Arguments.
# Symptom: "Cannot read properties of undefined (reading 'match')" fuer grep_search
#          "Cannot read properties of undefined (reading 'startsWith')" fuer list_dir
# Folge: VS Code return ERROR Tool-Results → Kimi probiert es wieder → Endlos-Loop.


# Sichere Subset von VS Code Tools, die der Cloud-Planner verwenden darf.
# Grund: Volle 47 Tools ueberfordern Kimi + riskieren Side-Effects.
# Der Planner darf nur LESEN/EXPLORIEREN, nichts mutieren.
_PLANNER_ALLOWED_TOOLS = {
    # Read-only exploration (matching Copilot Plan Mode)
    "read_file", "grep_search", "list_dir", "file_search",
    "view_image", "get_errors", "copilot_getNotebookSummary",
    "read_notebook_cell_output", "terminal_last_command", "terminal_selection",
    "get_task_output", "get_terminal_output", "testFailure",
    "fetch_webpage", "github_repo", "github_text_search",
    "session_store_sql", "vscode_listCodeUsages",
    # Interaction + persistence (matching Copilot Plan Mode)
    "memory", "resolve_memory_file_uri",
    "vscode_askQuestions",
}


def _filter_planner_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Filtert die vom Client uebergebenen Tool-Definitionen auf das sichere
    READ-ONLY Subset, das der Cloud-Planner verwenden darf.
    
    Das reduziert die Tool-Liste von 47 auf ~15 - Kimi wird stabiler beim
    Tool-Argument-Mashing und die Komplexitaet des Tool-Auswahl-Logits sinkt.
    Im Idealfall reduziert das auch die malformed-tool-args-Rate massiv.
    """
    if not tools:
        return tools
    filtered = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("function", {}).get("name") or tool.get("name", "")
        if name in _PLANNER_ALLOWED_TOOLS:
            filtered.append(tool)
    return filtered or None


def _normalize_tool_call_arguments(args: Any) -> str:
    """Konvertiert beliebige tool_call arguments in einen GUELTIGEN JSON-String.
    
    Bug-B: VS Code Copilot ruft JSON.parse(arguments) auf und erwartet danach ein
    Objekt mit den Parametern. Wenn arguments = '' (leer) oder ungueltiges JSON
    ist, schlaegt parse fehl → 'undefined.match()' Crash.
    
    Diese Funktion garantiert dass ein JSON-Objekt-String returned wird.
    """
    # Fall 1: args ist bereits ein String
    if isinstance(args, str):
        s = args.strip()
        if not s:
            return "{}"
        # Versuchen als JSON-String zu validieren
        try:
            parsed = json.loads(s)
            # Re-serialisieren fuer konsistente Formatierung
            return json.dumps(parsed, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            # Ungueltiges JSON - versuchen als Plain-Text in "query" zu wrappen
            _log(f"  ⚠ Tool-Arg kein gueltiges JSON, wrappe als string: {s[:80]}")
            return json.dumps({"query": s, "text": s}, ensure_ascii=False)
    
    # Fall 2: args ist ein dict → als JSON-String serialisieren
    if isinstance(args, dict):
        return json.dumps(args, ensure_ascii=False)
    
    # Fall 3: args ist None → leeres Objekt
    if args is None:
        return "{}"
    
    # Fall 4: alles andere → als JSON serialisieren
    try:
        return json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


def _normalize_tool_calls(
    tool_calls: Optional[List[Dict[str, Any]]],
    allowed: Optional[set] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Normalisiert eine Liste von tool_calls in OpenAI-kompatibles Format.
    
    - arguments werden IMMER zu gueltigem JSON-String konvertiert
    - function.name wird auf einen String gesichert
    - id wird bei Bedarf generiert
    - tool_calls mit leeren/None Namen werden weggeworfen
    - Filterung auf 'allowed' Set (optional)
    """
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
            _log(f"  🔧 Tool '{name}' nicht erlaubt fuer Planner → filtere")
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


def _classify_intent(messages: Sequence[Dict[str, Any]]) -> str:
    """Intent-Klassifizierung: deterministisch, bei Mehrdeutigkeit 'agent'."""
    text = _last_user_text(messages)

    # ── Pipeline-Steuerflags MÜSSEN ZUERST geprüft werden ────────────
    _, pflags = _extract_pipeline_flags(text)
    if pflags.get("force_planning"):
        _log("→ Intent: agent (Flag: -force planning)")
        return "agent"
    if pflags.get("force_review"):
        _log("→ Intent: agent (Flag: -force review)")
        return "agent"
    if pflags.get("bypass_worker"):
        _log("→ Intent: agent (Flag: -bypass worker)")
        return "agent"

    # Tool-Continuation: Check ob wir in einer Planner-Session sind
    if _is_tool_continuation(messages):
        session_hash = _get_planner_session_hash(messages)
        session = _PLANNER_SESSIONS.get(session_hash)
        if session and session.get("state") == "active":
            _log("→ Intent: agent (Planner-Session aktiv, Tool-Fortsetzung → Cloud-Planner)")
            return "agent"
        _log("→ Intent: direct (tool-continuation, Pipeline übersprungen)")
        return "direct"

    # Prüfen ob Cloud/LiteLLM konfiguriert ist → weniger aggressive Bypässe
    _cloud_available = bool(
        (CLOUD_REVIEW_ENABLED and CLOUD_REVIEW_API_KEY) or
        (LITELLM_CLOUD_MODEL and LITELLM_CLOUD_API_KEY)
    )

    if _cloud_available:
        # Cloud verfügbar: Ausgewogener – Cloud nur für komplexe Tasks,
        # triviale Kurzanfragen bleiben direkt.
        t = text.lower()
        has_agent = any(w in t for w in AGENT_TRIGGER_WORDS)
        has_direct = any(w in t for w in DIRECT_TRIGGER_WORDS)
        is_short = len(t) < 60
        is_medium = 60 <= len(t) < 400

        if has_agent:
            # Agent-Trigger → Cloud-Planer sinnvoll
            _log(f"→ Intent: agent (Cloud verfügbar, Agent-Trigger)")
            return "agent"
        if is_short and not has_agent:
            # Sehr kurze Anfragen OHNE Agent-Trigger → direkt (auch mit Cloud)
            _log(f"→ Intent: direct (Cloud verfügbar, Kurzanfrage)")
            return "direct"
        if has_direct:
            # Direkt-Trigger → direkt
            _log(f"→ Intent: direct (Cloud verfügbar, Direct-Trigger)")
            return "direct"
        if is_medium:
            # Mittellang → Cloud-Planer (lohnt sich)
            _log(f"→ Intent: agent (Cloud verfügbar, mittellang)")
            return "agent"
        # Lang (>400) → Cloud-Planer
        _log(f"→ Intent: agent (Cloud verfügbar, langer Text)")
        return "agent"

    # Kein Cloud → normale deterministische Klassifikation
    result = _classify_intent_deterministic(text)
    trigger_info = ""
    t = text.lower()
    if result == "agent":
        matches = [w for w in AGENT_TRIGGER_WORDS if w in t]
        trigger_info = f" triggers={matches[:3]}" if matches else " length"
    elif result == "direct":
        matches = [w for w in DIRECT_TRIGGER_WORDS if w in t]
        trigger_info = f" triggers={matches[:3]}" if matches else " length"
    _log(f"→ Intent: {result} (deterministisch, text_len={len(text)}{trigger_info})")
    return result


async def _classify_intent_with_fast_model(
    client: httpx.AsyncClient,
    messages: Sequence[Dict[str, Any]],
) -> str:
    """Nutzt das schnelle 27B-Modell zur Intent-Klassifikation (für mehrdeutige Fälle)."""
    deterministic = _classify_intent_deterministic(_last_user_text(messages))
    if deterministic is not None:
        return deterministic

    try:
        classify_payload = {
            "model": FAST_MODEL_NAME,
            "messages": [
                {"role": "system", "content": (
                    "Classify the user request into exactly one word: "
                    "'direct' (simple autocomplete, inline fix, rename, format, trivial) or "
                    "'agent' (refactor, architecture, bug, debug, test, security, complex multi-step). "
                    "Answer with ONLY the word 'direct' or 'agent'."
                )},
                *_compact_messages(list(messages), 6),
            ],
            "max_tokens": 4,
            "temperature": 0.0,
            "stream": False,
        }
        response = await client.post(VLLM_API_URL, json=classify_payload, headers=_vllm_headers(), timeout=15.0)
        if response.status_code == 200:
            result = response.json()
            content = _extract_choice_content(result).strip().lower()
            if "agent" in content:
                return "agent"
            return "direct"
    except Exception:
        pass
    return "agent"


# ═══════════════════════════════════════════════════════════════════════════
# Prompt-Builder (Caveman + Hindsight + Worker)
# ═══════════════════════════════════════════════════════════════════════════

def _clean_payload(payload: Dict[str, Any], keep_tools: bool = False) -> Dict[str, Any]:
    """Entfernt Client-spezifische Felder, die Provider verwirren."""
    # stream_options darf nur bei stream=true existieren
    if not payload.get("stream") and "stream_options" in payload:
        payload.pop("stream_options")
    # Weitere problematische Felder auf Top-Level
    strip_keys = ["stop_sequences", "safety_settings", "response_format", "top_k"]
    if not keep_tools:
        strip_keys += ["tool_choice", "tools", "functions", "function_call"]
    for key in strip_keys:
        payload.pop(key, None)
    return payload


def _build_direct_payload(
    body: Dict[str, Any],
    model_name: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    payload = copy.deepcopy(body)
    payload["model"] = model_name or body.get("model") or MODEL_NAME
    payload["max_tokens"] = int(max_tokens or payload.get("max_tokens", DEFAULT_DIRECT_MAX_TOKENS))
    payload["stream"] = False
    # ALLE Messages behalten – kein Compact!
    # Die System-Message (mit Tool-Definitionen) muss immer erhalten bleiben.
    payload["messages"] = list(payload.get("messages", []))
    # Tool-Result-Truncation + image_url-Sanitizer für text-only Models.
    # Spiegelung von _build_worker_payload — der Fallback-Pfad nach
    # Planner-Failure nutzt _build_direct_payload statt _build_worker_payload.
    messages = payload["messages"]
    _cap_tool_results_inplace(messages, "DirectPayload", max_chars=0)
    effective_model = model_name or body.get("model") or MODEL_NAME
    if _is_text_only_model(effective_model):
        _sanitize_image_urls_inplace(messages, "DirectPayload")
    # reasoning_content von ALLEN Messages entfernen (DeepSeek crasht sonst)
    removed_rc = _strip_kimi_reasoning(messages)
    for m in messages:
        if isinstance(m, dict) and "reasoning_content" in m:
            del m["reasoning_content"]
            removed_rc += 1
    if removed_rc:
        _log(f"  🧹 DirectPayload: {removed_rc} reasoning_content-Felder entfernt")
    return _clean_payload(payload, keep_tools=True)


def _cap_tool_results_inplace(messages: List[Dict[str, Any]], label: str = "Payload",
                              max_chars: Optional[int] = None) -> int:
    """Kappt Tool-Result-Messages in-place auf max_chars (oder TOOL_RESULT_CAP).

    Verhindert Token-Bombing. Für Planner-Calls: max_chars=0 → kein Cap.
    """
    limit = max_chars if max_chars is not None else TOOL_RESULT_CAP
    if limit <= 0:
        return 0  # Kein Cap gewünscht
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
        _log(f"  ✂ {label}-Payload: {capped_count} Tool-Results auf {limit} chars gekappt")
    return capped_count


# Models die als text-only gelten (kein multimodal Support → image_url 400).
# Heuristisch: jeder Name, der 'deepseek' enthält. Falls Kimi o.ä. gemischt
# wird, muss Liste erweitert werden.
_TEXT_ONLY_MODEL_MARKERS = ("deepseek",)


def _is_text_only_model(model_name: str) -> bool:
    """True, wenn Model-Name auf einen text-only-Worker schließen lässt."""
    if not model_name:
        return False
    name_lower = str(model_name).lower()
    return any(marker in name_lower for marker in _TEXT_ONLY_MODEL_MARKERS)


def _sanitize_image_urls_inplace(messages: List[Dict[str, Any]], label: str = "Payload") -> int:
    """Entfernt 'image_url'-Parts aus multimodalem Content, weil text-only-
    Models (wie DeepSeek V4) sonst HTTP 400 werfen ('unknown variant image_url').

    Strategie:
      - String-content: bleibt unverändert (kann nicht image_url enthalten).
      - List-of-parts:
          * 'image_url'-Parts werden GESTRIPPED.
          * Wenn NUR image_url-Parts vorhanden → ersetze durch Fallback-Text,
            damit die Message nicht inhaltslos wird.

    Returns: Anzahl entfernte image_url-Parts.
    """
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
                # Content komplett leer nach Sanitize → Platzhalter setzen,
                # sonst klagt die API ("content required").
                kept_parts = [{"type": "text", "text": "[image content omitted: text-only model]"}]
            msg["content"] = kept_parts
    if removed:
        _log(f"  🧼 {label}-Payload: {removed} image_url-Part(s) entfernt (text-only Model)")
    return removed


def _extract_planner_file_contents(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    """Aider-Äquivalent zu get_chat_files_messages(): Scannt die Planner-
    Conversation nach read_file-Ergebnissen und extrahiert sie als Datei-
    Inhalt-Map. Diese werden dem Worker als Pre-Loaded-Kontext injiziert,
    sodass er NICHT selbst read_file aufrufen muss (verhindert die
    klassische "Worker liest immer wieder dieselben Files"-Schleife).
    
    Returns: {filePath: content, ...}
    """
    # 1. Sammle alle tool_call_ids von assistant-Nachrichten mit read_file
    read_tc_ids: Dict[str, str] = {}  # tc_id → filePath
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = (tc.get("function") or {}).get("name", "")
            if fn not in ("read_file",):
                continue
            tc_id = tc.get("id", "")
            if not tc_id:
                continue
            try:
                args = json.loads((tc.get("function") or {}).get("arguments", "{}") or "{}")
            except Exception:
                continue
            fp = args.get("filePath") or args.get("path") or ""
            if fp:
                read_tc_ids[tc_id] = fp

    if not read_tc_ids:
        return {}

    # 2. Finde die tool-result messages mit passenden tool_call_ids
    file_contents: Dict[str, str] = {}
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "tool":
            continue
        tc_id = m.get("tool_call_id", "")
        if tc_id not in read_tc_ids:
            continue
        fp = read_tc_ids[tc_id]
        content = m.get("content", "")
        if isinstance(content, str) and content.strip():
            # Nimm den neuesten Read pro File (letzter gewinnt)
            file_contents[fp] = content

    return file_contents


def _build_worker_payload(
    body: Dict[str, Any],
    plan: str,
    memory_context: str,
    model_name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    plan_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Baut den Payload fuer den WORKER - ein FULLY TOOL-CAPABLE Sub-Agent.
    Der Worker bekommt die KOMPLETTE VS Code Umgebung:
      - Originale System-Message (mit Tool-Definitionen & Calling-Format)
      - Alle Messages (kein Compact!)
      - tools / tool_choice aus dem Original-Body
      - Plan + Memory als ZUSAETZLICHER Kontext in der letzten User-Message
    
    USER-VISION "Plan-bindend":
      Der Worker wird EXPLIZIT angewiesen, den Plan strikt zu befolgen.
      Er hat ALLE VS Code Tools, aber die System-Instruktion ist der Vertrag:
      "Implementiere den Plan. Keine(). Neue(). Ideen. Wenn der Plan Fehler
      hat, mache最小-korrektur und kommentiere. Keine Refactors, keine
      Zusatz-Features, keine neuen Dateien es sei denn der Plan verlangt es."

    PLAN-FILE (Codespace-Copilot-Stil):
      Neben der In-Prompt-Injection wird der Plan zusaetzlich als .md-File
      persistiert (data/plans/Plan_<hash>.md). Der File-Pfad wird als
      Pointer im System-Prompt veroeffentlicht - so kann der Worker auch
      nach Runden von Tool-Calls den Plan nachlesen.

    REASONING-CONTENT HYGIENE:
      Kimi-urspruengliche Assistant-Messages mit `reasoning_content` (Thinking)
      werden entfernt, ausser wenn tool_calls dranhaengen (dann braucht
      DeepSeek das Thinking fuer Tool-Continuations). Vermeidet
      'thinking content could not be passed'-Fehler beim Hand-off.
    """
    payload = copy.deepcopy(body)
    payload["model"] = model_name or body.get("model") or MODEL_NAME
    payload["max_tokens"] = int(max_tokens or payload.get("max_tokens", DEFAULT_AGENT_MAX_TOKENS))
    payload["stream"] = False

    messages = list(payload.get("messages", []))

    # ══ PLAN-ISOLAT MESSAGE TRIAGE ═══════════════════════════════════
    # Konzept: "Senior-Planner delegiert an Junior-Worker" – der manuelle
    # Workflow des Users: Frontier-Modell plant mit vollem Kontext, dann
    # delegiert an anderes Modell NUR den Plan (kein Kontext).
    #
    # Der Worker (DeepSeek) bekommt NUR:
    #   1. System-Message (Tool-Definitionen, Calling-Format) – original
    #   2. Die ORIGINAL User-Aufgabenbeschreibung (erste substantielle User-Msg)
    #   3. Den fertigen Plan (binding contract + Kontext, unten injectiert)
    #   4. Bei Continuation: NUR den letzten Worker-Tool-Zyklus
    #
    # Der GESAMTE Planner-Verlauf (Kimi Tool-Calls, read_file-Ergebnisse,
    # Reasoning-Content,compactierte Caveman-Kontexte) wird ENTFERNT.
    # Der Worker MUSS eigene read_file/grep_call machen, um Code zu sehen –
    # exakt wie im manuellen Workflow. So wird DeepSeek nicht durch fremden
    # Tool-Cruft verwirrt und hat einen sauberen, winzigen Kontext.
    system_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "system"]

    # Original-User-Task isolieren: ERSTE User-Message mit substantiellem Content.
    # Überspringt VSCode-injected System-User-Msgs ("[SYSTEM:", Error-Injections).
    original_user_msg = None
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content", "")
        if isinstance(content, str):
            stripped = content.strip()
            if not stripped:
                continue
            # Skip re-injected system/error messages
            if stripped.startswith("[SYSTEM:") or stripped.startswith("[HINDSIGHT"):
                continue
            original_user_msg = copy.deepcopy(m)
            break
        elif isinstance(content, list) and content:
            original_user_msg = copy.deepcopy(m)
            break

    has_tool_msgs = any(isinstance(m, dict) and m.get("role") == "tool" for m in messages)

    # Baue Payload: System + Original-User
    new_msgs = list(system_msgs)
    if original_user_msg is not None:
        new_msgs.append(original_user_msg)

    # ══ PRE-LOADED FILES (Aider's get_chat_files_messages) ═══════
    # JEDEN Round — nicht nur First-Call. Der Worker verliert sonst
    # den Code-Kontext nach Runde 1 und fängt an, memory/read_file
    # aufzurufen → klassische Re-Read-Loop.
    # Extrahiere Kimi's read_file-Ergebnisse aus der ORIGINALEN
    # Conversation (die 175+ msgs enthalten sie immer).
    pre_loaded_files = {}
    if plan:
        pre_loaded_files = _extract_planner_file_contents(messages)
        if pre_loaded_files:
            injected = 0
            for fp, content in pre_loaded_files.items():
                block = (
                    f"[PRE-LOADED FILE: {fp} — FULL CONTENT, TRUST THIS]\n"
                    f"Do NOT re-read this file. Its complete contents are below.\n"
                    f"---FILE-START---\n"
                    f"{content}\n"
                    f"---FILE-END---"
                )
                new_msgs.append({"role": "user", "content": block})
                injected += 1
            _log(f"  📎 Preloaded: {injected} Dateien "
                 f"({sum(len(v) for v in pre_loaded_files.values())} chars total)")

    if has_tool_msgs:
        # Continuation: NUR den letzten assistant(tool_calls) → tool results
        # Zyklus behalten (NACH den Pre-Loaded Files, die schon in new_msgs sind)
        last_worker_asst = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], dict) and messages[i].get("role") == "assistant" and messages[i].get("tool_calls"):
                last_worker_asst = i
                break
        if last_worker_asst is not None:
            for j in range(last_worker_asst, len(messages)):
                msg = messages[j]
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") == "system":
                    continue
                new_msgs.append(copy.deepcopy(msg))
        _log(f"  ✂ Worker-Payload (Plan-Isolat, Continuation): {len(new_msgs)} Messages "
             f"(System+Task+{len(pre_loaded_files)}Files+Worker-Cycle, "
             f"Original {len(messages)} → {len(new_msgs)})")
    else:
        _log(f"  ✂ Worker-Payload (Plan-Isolat, First-Call): {len(new_msgs)} Messages "
             f"(Original {len(messages)} → {len(new_msgs)}, Planner-Verlauf entfernt)")

    messages = new_msgs

    # ══ Tool-Error-Loop-Detection ═══════════════════════════════════
    # Wenn in der Continuation ALLE Tool-Results Fehler sind, ersetze den
    # Zyklus durch eine User-Instruktion. Sonst dreht DeepSeek sich im Kreis.
    tool_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
    error_tool_msgs = [m for m in tool_msgs if "error" in _message_text(m).lower()]
    if tool_msgs and len(error_tool_msgs) == len(tool_msgs):
        _log(f"  ⚠ Alle {len(tool_msgs)} Tool-Results sind Fehler → Error-Loop-Prävention")
        # Schneide NACH der letzten user-msg ab, ersetze Tool-Zyklus durch Instruktion
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], dict) and messages[i].get("role") == "user":
                last_user_idx = i
                break
        error_summary = "; ".join(set(_message_text(m)[:120] for m in tool_msgs))
        messages = messages[:last_user_idx + 1] if last_user_idx is not None else messages[:1]
        messages.append({
            "role": "user",
            "content": (
                f"[SYSTEM: Your last {len(tool_msgs)} tool calls all failed validation: {error_summary}]\n"
                "[SYSTEM: Re-read the tool definitions in the system prompt. "
                "Required parameters MUST be valid JSON. Try a different tool or simpler arguments.]"
            ),
        })
        _log(f"  🛡️ Tool-Error-Loop gebrochen: {len(tool_msgs)} Fehler durch Instruktion ersetzt")

    # ══ Tool-Result-Truncation: verhindert Token-Bombing ══════════════
    # Wenn VS Code riesige Tool-Results (z.B. 111KB grep_hits) zurückschickt,
    # explodiert der Payload. Cap hart auf TOOL_RESULT_CAP chars pro Tool-Msg.
    _cap_tool_results_inplace(messages, "Worker", max_chars=0)

    # ══ image_url-Sanitizer für text-only Models ═════════════════════
    # DeepSeek V4 wirft sonst 400 ('unknown variant image_url, expected text').
    # Bedingte Anwendung: nur wenn das Ziel-Model text-only ist.
    effective_model = model_name or body.get("model") or MODEL_NAME
    if _is_text_only_model(effective_model):
        _sanitize_image_urls_inplace(messages, "Worker")

    # ══ reasoning_content von Kimi-Seite entfernen ═══════              
    removed = _strip_kimi_reasoning(messages)
    # Auch reasoning_content auf Assistants MIT tool_calls strippen
    for m in messages:
        if isinstance(m, dict) and "reasoning_content" in m:
            del m["reasoning_content"]
            removed += 1
    if removed:
        _log(f"  🧹 Worker-Payload: {removed} reasoning_content-Felder entfernt")

    # ══ Loop-Detection: Wiederholte Reads desselben Files? ════════════
    # Bei Tool-Continuation prüfen, ob der Worker gerade immer wieder
    # dieselben Files liest (read_file/grep_call mit identischen Args).
    # Das ist das Symptom der "Plan-Schleife": DeepSeek startet in jeder
    # neuen Runde von vorn, weil es denkt, der Plan sei neu.
    is_continuation = has_tool_msgs
    read_repeat_count = 0
    if is_continuation:
        read_call_files = []
        for m in messages:
            if not isinstance(m, dict) or m.get("role") != "assistant":
                continue
            for tc in (m.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                fn = (tc.get("function") or {}).get("name", "")
                if fn not in ("read_file", "grep_search", "file_search", "list_dir"):
                    continue
                try:
                    args = json.loads((tc.get("function") or {}).get("arguments", "{}") or "{}")
                except Exception:
                    args = {}
                sig_key = (fn, str(args.get("filePath") or args.get("path") or
                                   args.get("query") or args.get("pattern") or ""))
                read_call_files.append(sig_key)
        if read_call_files:
            uniq = set(read_call_files)
            read_repeat_count = len(read_call_files) - len(uniq)
            if read_repeat_count >= 2:
                _log(f"  ⚠ Worker-Read-Loop erkannt: {read_repeat_count} wiederholte Reads "
                     f"(von {len(read_call_files)} Read-Calls) → Kompakt-Instruktion injizieren")

    # ══ Plan-binding System-Prompt injecten (JEDEN Round!) ══════
    # VS Code sendet bei tool_continuation seinen EIGENEN System-Prompt,
    # nicht unseren modifizierten. Deshalb MUSS der Plan in JEDER Runde
    # neu in den System-Prompt injiziert werden.
    # Früherer Kommentar "nicht erneut injecten → Worker denkt neuer
    # Auftrag" war falsch — der CONTINUATION REMINDER sagt "mid-execution".
    if plan:
        plan_binding = (
            "\n\n[PLAN-BINDING CONTRACT — READ CAREFULLY]\n"
            "A senior planner has prepared a strategic plan for you. Your job: IMPLEMENT IT.\n"
            "IMPORTANT: Code context is PRE-LOADED in [PRE-LOADED FILE] messages above.\n"
            "You do NOT need to read files — the code is already in your context.\n"
            "Rules:\n"
            "1. Edit files directly — their contents are in [PRE-LOADED FILE] blocks above.\n"
            "   Only read a file if it was NOT pre-loaded AND the plan references it.\n"
            "2. Execute each plan step IN ORDER.\n"
            "3. DO NOT refactor, add features, or 'improve' anything not in the plan.\n"
            "4. If a step references the wrong file/line: read to find the real location, then proceed.\n"
            "5. Do NOT create new files unless the plan explicitly says 'CREATE <path>'.\n"
            "6. After finishing, output '## Implementation Summary' with each step ✓/⚠/✗.\n"
            "\n"
            "═══ THE PLAN (always visible in every round) ═══\n"
            f"{plan}\n"
            "═══ END PLAN ═══"
        )
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            existing = str(messages[0].get("content", ""))
            # Do NOT re-inject if the plan is already in the system prompt
            if "═══ THE PLAN" not in existing:
                messages[0]["content"] = existing + plan_binding
        else:
            messages.insert(0, {"role": "system", "content": plan_binding.strip()})

    # ══ Plan + Memory als User-Messages ══════════════════════════════
    # BEIM FIRST-CALL: voller Plan als abschließende User-Message → Worker
    #   weiss, was er tun soll, und startet mit read_file.
    # BEI CONTINUATION: KEINE Re-Injection des vollen Plans! Sonst Schleife.
    #   Nur ein kurzer Reminder, wo der Plan liegt, und ein Loop-Breaker
    #   wenn der Worker wiederholt dieselben Files liest.
    context_blocks = []
    if plan and not is_continuation:
        context_blocks.append(
            "[CLOUD EXECUTION PLAN — SEE SYSTEM PROMPT ABOVE]\n"
            "The full plan is embedded in the system prompt (see '═══ THE PLAN ═══').\n"
            "The files you need are PRE-LOADED above as [PRE-LOADED FILE] blocks.\n"
            "Start with Step 1 and edit directly."
        )
        if memory_context:
            context_blocks.append(
                "---\n"
                "⚠️ BACKGROUND KNOWLEDGE — NOT YOUR CURRENT TASK ⚠️\n"
                "The following is LEARNED CONTEXT from PRIOR coding sessions.\n"
                "It may contain patterns, fixes, and conventions relevant to the\n"
                "current task. Use it for REFERENCE only.\n"
                "YOUR CURRENT TASK is described in the user message ABOVE this one.\n"
                "The PLAN you must execute is above this section.\n"
                "---\n"
                f"{memory_context}"
            )
    elif is_continuation:
        plan_hint = (
            "[CONTINUATION REMINDER — THE PLAN IS IN YOUR SYSTEM PROMPT]\n"
            "You are mid-execution. The complete plan is at '═══ THE PLAN ═══'\n"
            "in your system prompt above. Read it there — do NOT use the memory tool.\n"
            "The code you need is in [PRE-LOADED FILE] blocks above.\n"
            "Pick the NEXT unedited step and execute it NOW using replace_string_in_file."
        )
        if read_repeat_count >= 2:
            plan_hint += (
                "\n\n⚠ LOOP ALERT: You have read the same files multiple times. "
                "STOP reading files. STOP reading memory. "
                "The file contents are ALREADY in your context (see [PRE-LOADED FILE] blocks).\n"
                "Pick the NEXT plan step you have NOT started yet and EXECUTE "
                "the edit now using replace_string_in_file / multi_replace_string_in_file."
            )
        else:
            plan_hint += (
                " DO NOT use the memory tool — the plan is in your system prompt. "
                "DO NOT re-read files — their contents are in [PRE-LOADED FILE] blocks above."
            )
        context_blocks.append(plan_hint)
    context_str = "\n\n".join(context_blocks)

    if context_str:
        messages.append({"role": "user", "content": context_str})

    # ALLE Messages behalten - kein Compact! Das System-Prompt mit Tool-Defs bleibt erhalten.
    payload["messages"] = messages
    return _clean_payload(payload, keep_tools=True)


def _build_cloud_plan_payload(body: Dict[str, Any], memory_context: str) -> Dict[str, Any]:
    """Baut den Payload für den Cloud-Planer (Caveman Ultra Modus).
    
    Sendet JETZT den VOLLEN Workspace-Kontext: Agent-Instruktionen, Tool-Resultate
    (Dateiinhalte aus read_file!), Konversations-Historie und Task.
    """
    rich_context = _build_rich_context(body.get("messages", []))
    user_text = _last_user_text(body.get("messages", []))
    
    prompt_parts = [
        "=== FULL WORKSPACE CONTEXT ===",
        rich_context,
        "",
        "=== YOUR TASK ===",
        user_text,
        "",
        "CONSTRAINTS:",
        "- You NOW HAVE the actual codebase context above. Use it to plan precisely.",
        "- Return a DETAILED execution plan (8-15 steps). Reference specific files/lines.",
        "- Each step: WHAT to do, WHICH file (exact path), WHY (based on the code you see).",
        "- DO NOT write code. DO NOT use tool_calls. Just plan from the context provided.",
        "- DO NOT say 'future' or 'later'. This plan will be executed NOW.",
        "- Use symbols: -> ! ? FIX RISK TODO",
    ]
    if CAVEMAN_ENABLED:
        prompt_parts.append(f"- {CAVEMAN_SYSTEM_PROMPT}")

    prompt = "\n".join(prompt_parts)

    if memory_context:
        prompt = f"[HINDSIGHT MEMORY]\n{memory_context}\n\n" + prompt

    payload = _clean_payload({
        "model": CLOUD_REVIEW_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "STRATEGIC PLANNER WITH FULL CODEBASE CONTEXT. "
                    "You receive: agent instructions, workspace files (from tool results), "
                    "conversation history, and the task. "
                    "Your ONLY output: a detailed execution plan (8-15 steps). "
                    "Reference SPECIFIC files and lines from the context. "
                    "DO NOT write code. DO NOT use tool_calls. "
                    "Format: numbered list, each line < 100 chars."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": CAVEMAN_MAX_TOKENS,
        "temperature": 0.2,
        "stream": False,
    })
    _patch_moonshot_payload(payload)
    return payload


def _build_verify_payload(
    code: str,
    test_results: str,
    original_task: str,
) -> Dict[str, Any]:
    """Baut einen Payload für die Verifikations-Phase."""
    return {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a code verifier. Review the code against the original task "
                    "and test/lint results. If there are issues, fix them. "
                    "Return the corrected code or confirm it is correct."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"ORIGINAL TASK:\n{original_task}\n\n"
                    f"CODE:\n{code}\n\n"
                    f"TEST/LINT RESULTS:\n{test_results}\n\n"
                    "Is the code correct? If not, provide the fixed version."
                ),
            },
        ],
        "max_tokens": DEFAULT_AGENT_MAX_TOKENS,
        "stream": False,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Progress/Status Formatierung
# ═══════════════════════════════════════════════════════════════════════════

def _format_status_event(stage: str, message: str, data: Optional[Dict[str, Any]] = None) -> str:
    payload = {
        "type": "localproxy.status",
        "stage": stage,
        "message": message,
        "timestamp": time.time(),
    }
    if data is not None:
        payload["data"] = data
    return json.dumps(payload, ensure_ascii=False)


def _format_chat_progress_message(stage: str, message: str, data: Optional[Dict[str, Any]] = None) -> str:
    if not CHATTY_MODE:
        return ""
    data_text = f"  \n`{json.dumps(data, ensure_ascii=False)}`" if data is not None else ""
    return f"**▸ STATUS [{stage}]:** {message}{data_text}\n\n"


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
        # Reiner Finish-Chunk: delta MUSS leer sein ({}) für OpenAI-Kompatibilität
        pass
    elif tool_calls is not None:
        # Tool-Calls-Modus: content explizit None setzen
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
    # Konsistente Chunk-ID für den gesamten Stream
    cid = chunk_id or f"chatcmpl-spark-{uuid.uuid4().hex}"
    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _build_response_payload(
    body: Dict[str, Any],
    combined_response_text: str,
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    # Pipeline-Summary (nur bei Text, nicht bei Tool-Calls)
    is_tool_cont = any(r.get("tool_calls") for r in results)
    has_dsml = any(_contains_tool_calls(str(r.get("content", ""))) for r in results)
    if is_tool_cont or has_dsml:
        intent = "direct"
        if not is_tool_cont and has_dsml:
            _log(f"  ⚠ Non-Streaming: DSML erkannt, Summary unterdrückt")
    else:
        has_cloud = any(r.get("agent_key") == "cloud_planner" for r in results)
        has_worker = any(r.get("agent_key") == "worker" for r in results)
        intent = "agent" if (has_cloud or has_worker) else "direct"
        summary = _build_pipeline_summary(results, intent)
        combined_response_text = combined_response_text.rstrip() + summary

    message: Dict[str, Any] = {"role": "assistant", "content": combined_response_text}
    # Falls Provider echte OpenAI tool_calls zurückgibt: strukturiert durchreichen.
    # reasoning_content aus dem ersten passenden Result übernehmen.
    reasoning_found = None
    for r in reversed(results):
        if r.get("reasoning_content") and reasoning_found is None:
            reasoning_found = r.get("reasoning_content")
        if r.get("tool_calls"):
            message = {
                "role": "assistant",
                "content": r.get("content") or None,
                "tool_calls": r.get("tool_calls"),
            }
            if reasoning_found:
                message["reasoning_content"] = reasoning_found
            break
    else:
        # Keine tool_calls gefunden → reasoning_content an die Message anhängen
        if reasoning_found:
            message["reasoning_content"] = reasoning_found

    return {
        "id": f"chatcmpl-spark-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model") or MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
        }],
        "usage": _sum_usage(results),
    }


def _build_pipeline_summary(
    results: List[Dict[str, Any]],
    intent: str,
) -> str:
    """Erzeugt eine kompakte Pipeline-Zusammenfassung."""
    parts = []
    total_duration = 0.0

    # Durchsicht der Results nach agent_keys
    cloud_dur = None
    worker_dur = None
    cloud_model = None
    verify_ok = None

    for r in results:
        dur = r.get("duration_seconds", 0) or 0
        total_duration += dur
        ak = r.get("agent_key", "")

        if ak == "cloud_planner" and r.get("status") == "ok":
            cloud_dur = dur
            cloud_model = CLOUD_REVIEW_MODEL if CLOUD_REVIEW_ENABLED else LITELLM_CLOUD_MODEL
        elif ak == "worker" and r.get("status") == "ok":
            worker_dur = dur
        elif ak == "verify":
            stage = r.get("stage", "")
            if "passed" in stage or "ok" in stage:
                verify_ok = True
            elif "failed" in stage:
                verify_ok = False

    # Fallback: Wenn kein worker result mit agent_key, nimm das letzte "ok" result
    if worker_dur is None:
        for r in reversed(results):
            if r.get("status") == "ok" and r.get("duration_seconds"):
                worker_dur = r["duration_seconds"]
                break

    if intent == "direct":
        parts.append(f"📋 Pipeline: direkt")
    else:
        parts.append(f"📋 Pipeline: agent")
        if cloud_dur is not None:
            parts.append(f"Cloud: {cloud_model or '?'} ({cloud_dur:.1f}s)")
        elif any(r.get("agent_key") == "cloud_planner" and r.get("status") in ("failed", "error", "skipped") for r in results):
            parts.append(f"Cloud: ⛔")

    if worker_dur is not None:
        parts.append(f"Worker: {worker_dur:.1f}s")
    if verify_ok is True:
        parts.append(f"Verify: ✅")
    elif verify_ok is False:
        parts.append(f"Verify: ❌")

    total = total_duration or time.perf_counter()
    parts.append(f"∑ {total:.1f}s")
    return "  \n" + " · ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Cloud API Calls (HTTPX + LiteLLM)
# ═══════════════════════════════════════════════════════════════════════════

async def _call_cloud_planner(
    client: httpx.AsyncClient,
    body: Dict[str, Any],
    memory_context: str,
    force: bool = False,
) -> Dict[str, Any]:
    """Cloud-Planer via HTTPX (OpenAI-kompatibel) oder LiteLLM.
    Wenn force=True, wird CLOUD_REVIEW_ENABLED ignoriert (aber Key wird noch benötigt)."""
    if (not CLOUD_REVIEW_ENABLED and not force) or not CLOUD_REVIEW_API_KEY:
        return {
            "agent_key": "cloud_planner",
            "status": "skipped",
            "content": "",
            "duration_seconds": 0.0,
            "usage": None,
        }

    # LiteLLM-Route (OpenRouter / DeepSeek / Claude)
    # - Wenn die litellm-Library installiert ist: Nutze sie (unterstützt Provider-Prefix)
    # - Wenn LITELLM_CLOUD_API_URL gesetzt ist: HTTPX-Direktaufruf (auch ohne litellm-Library)
    _lite_available = _LITELLM_AVAILABLE or bool(LITELLM_CLOUD_API_URL)
    if LITELLM_CLOUD_MODEL and LITELLM_CLOUD_API_KEY and _lite_available:
        _log(f"  ☁️ Cloud-Planner via LiteLLM: model={LITELLM_CLOUD_MODEL}")
        return await _call_cloud_via_litellm(body, memory_context)

    # Standard OpenAI-kompatible Route
    _log(f"  ☁️ Cloud-Planner: model={CLOUD_REVIEW_MODEL} url={CLOUD_REVIEW_API_URL}")
    payload = _build_cloud_plan_payload(body, memory_context)
    started = time.perf_counter()
    headers = {"Authorization": f"Bearer {CLOUD_REVIEW_API_KEY}"}

    try:
        response = await client.post(
            CLOUD_REVIEW_API_URL,
            json=payload,
            headers=headers,
            timeout=CLOUD_REVIEW_TIMEOUT_SECONDS,
        )
        duration = time.perf_counter() - started
        if response.status_code == 200:
            result = response.json()
            _log(f"  ✓ Cloud-Planner OK: duration={duration:.1f}s")
            return {
                "agent_key": "cloud_planner",
                "status": "ok",
                "content": _extract_choice_content(result),
                "duration_seconds": duration,
                "usage": result.get("usage"),
            }
        _log(f"  ⚠ Cloud-Planner STATUS {response.status_code}: duration={duration:.1f}s")
        return {
            "agent_key": "cloud_planner",
            "status": "failed",
            "content": f"Cloud status {response.status_code}: {response.text[:500]}",
            "duration_seconds": duration,
            "usage": None,
        }
    except Exception as exc:
        _log(f"  ✗ Cloud-Planner ERROR: model={CLOUD_REVIEW_MODEL} – {exc}")
        return {
            "agent_key": "cloud_planner",
            "status": "error",
            "content": f"Cloud error: {exc}",
            "duration_seconds": time.perf_counter() - started,
            "usage": None,
        }


async def _call_cloud_via_litellm(body: Dict[str, Any], memory_context: str) -> Dict[str, Any]:
    """Cloud-Planer via LiteLLM — JETZT mit vollem Workspace-Kontext."""
    rich_context = _build_rich_context(body.get("messages", []))
    user_text = _last_user_text(body.get("messages", []))
    prompt = (
        f"=== FULL WORKSPACE CONTEXT ===\n{rich_context}\n\n"
        f"=== YOUR TASK ===\n{user_text}\n\n"
        "Create a DETAILED execution plan (8-15 steps). "
        "Reference specific files and lines from the context above. "
        "WHAT to do, WHICH file, WHY. "
        "NO code. NO tool calls. This will be executed NOW."
    )
    if CAVEMAN_ENABLED:
        prompt = f"{CAVEMAN_SYSTEM_PROMPT}\n\n{prompt}"
    if memory_context:
        prompt = f"[HINDSIGHT]\n{memory_context}\n\n{prompt}"

    payload = {
        "model": LITELLM_CLOUD_MODEL,
        "messages": [
            {"role": "system", "content": (
                "STRATEGIC PLANNER WITH FULL CODEBASE CONTEXT. "
                "You receive agent instructions, workspace files, history, and the task. "
                "Output: detailed execution plan (8-15 steps) referencing specific files/lines. "
                "No code. No tool calls."
            )},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": min(max(1, int(LITELLM_CLOUD_MAX_TOKENS or CAVEMAN_MAX_TOKENS or 65536)), 131072),
        "temperature": 0.2,
        "stream": False,
    }
    _patch_moonshot_payload(payload)
    payload = _clean_payload(payload)
    _log(f"  🔍 LiteLLM payload: max_tokens={payload['max_tokens']} model={payload['model']} temp={payload.get('temperature')}")
    headers = {"Authorization": f"Bearer {LITELLM_CLOUD_API_KEY}"}

    started = time.perf_counter()
    try:
        # Wenn eine API-URL gesetzt ist, direkt per httpx (funktioniert für opencode.ai etc.)
        if LITELLM_CLOUD_API_URL:
            # LiteLLM-Provider-Prefix entfernen (z.B. "deepseek/deepseek-v4-pro" → "deepseek-v4-pro")
            # Die LiteLLM-Library braucht den Prefix, die direkte API versteht ihn nicht.
            model_name = LITELLM_CLOUD_MODEL
            if "/" in model_name and not model_name.startswith("http"):
                parts = model_name.split("/", 1)
                known_prefixes = {"deepseek", "openai", "openrouter", "anthropic", "google", "groq", "together", "mistral", "perplexity", "claude"}
                if parts[0].lower() in known_prefixes:
                    _log(f"  ℹ️  Stripped LiteLLM prefix '{parts[0]}' from model name: '{model_name}' → '{parts[1]}'")
                    model_name = parts[1]
            payload["model"] = model_name
            async with httpx.AsyncClient(timeout=LITELLM_CLOUD_TIMEOUT_SECONDS) as lc:
                r = await lc.post(LITELLM_CLOUD_API_URL, json=payload, headers=headers)
            duration = time.perf_counter() - started
            if r.status_code == 200:
                result = r.json()
                content = _extract_choice_content(result)
                _log(f"  ✓ LiteLLM OK: model={LITELLM_CLOUD_MODEL} duration={duration:.1f}s")
                return {
                    "agent_key": "cloud_planner", "status": "ok",
                    "content": content or "", "duration_seconds": duration,
                    "usage": result.get("usage"),
                }
            _log(f"  ⚠ LiteLLM STATUS {r.status_code}: model={LITELLM_CLOUD_MODEL} duration={duration:.1f}s")
            err_detail = ""
            try:
                err_body = r.json()
                err_detail = err_body.get("error", {}).get("message", "") if isinstance(err_body.get("error"), dict) else str(err_body.get("error",""))
            except Exception:
                err_detail = r.text[:200]
            if err_detail:
                _log(f"  ⚠ LiteLLM Fehlerdetails: {err_detail}")
            return {
                "agent_key": "cloud_planner", "status": "failed",
                "content": f"LiteLLM status {r.status_code}: {r.text[:500]}",
                "duration_seconds": duration, "usage": None,
            }

        # Fallback: litellm-Library (braucht Provider-Prefix wie openai/gpt-4)
        if litellm is None:
            return {"agent_key": "cloud_planner", "status": "error", "content": "LiteLLM not available", "duration_seconds": 0.0, "usage": None}
        kwargs = dict(
            model=LITELLM_CLOUD_MODEL,
            messages=payload["messages"],
            max_tokens=payload["max_tokens"],
            temperature=payload["temperature"],
            api_key=LITELLM_CLOUD_API_KEY,
            timeout=LITELLM_CLOUD_TIMEOUT_SECONDS,
        )
        result = await asyncio.to_thread(litellm.completion, **kwargs)
        duration = time.perf_counter() - started
        content = result.choices[0].message.content if result.choices else ""
        _log(f"  ✓ LiteLLM OK: model={LITELLM_CLOUD_MODEL} duration={duration:.1f}s")
        return {
            "agent_key": "cloud_planner",
            "status": "ok",
            "content": content or "",
            "duration_seconds": duration,
            "usage": result.usage.dict() if hasattr(result, 'usage') and result.usage else None,
        }
    except Exception as exc:
        _log(f"  ✗ LiteLLM ERROR: model={LITELLM_CLOUD_MODEL} – {exc}")
        return {
            "agent_key": "cloud_planner",
            "status": "error",
            "content": f"LiteLLM error: {exc}",
            "duration_seconds": time.perf_counter() - started,
            "usage": None,
        }


async def _call_cloud_as_responder(
    client: httpx.AsyncClient,
    body: Dict[str, Any],
    plan: str,
    memory_context: str,
) -> Dict[str, Any]:
    """Cloud (LiteLLM/ReviewLLM) als direkter Responder — Worker-Bypass.
    Sendet den Task + Caveman-Plan + Workspace-Kontext an das Cloud-Modell."""
    rich_context = _build_rich_context(body.get("messages", []))
    user_text = _last_user_text(body.get("messages", []))
    prompt = (
        f"=== FULL WORKSPACE CONTEXT ===\n{rich_context}\n\n"
        f"=== TASK ===\n{user_text}\n\n"
        f"=== EXECUTION PLAN ===\n{plan}\n\n"
        "You are the executor. Implement the plan using the context above. "
        "Provide complete code, detailed explanation, and working solution. "
        "No tool calls — write code directly."
    )
    if memory_context:
        prompt = f"[HINDSIGHT]\n{memory_context}\n\n{prompt}"

    # Prefer LiteLLM, fallback to Cloud Reviewer
    if LITELLM_CLOUD_MODEL and LITELLM_CLOUD_API_KEY:
        model = LITELLM_CLOUD_MODEL
        api_key = LITELLM_CLOUD_API_KEY
        api_url = LITELLM_CLOUD_API_URL
        max_tok = LITELLM_CLOUD_MAX_TOKENS
        timeout = LITELLM_CLOUD_TIMEOUT_SECONDS
        _log(f"  ☁️ Cloud-Responder via LiteLLM: model={model}")
    elif CLOUD_REVIEW_ENABLED and CLOUD_REVIEW_API_KEY:
        model = CLOUD_REVIEW_MODEL
        api_key = CLOUD_REVIEW_API_KEY
        api_url = CLOUD_REVIEW_API_URL
        max_tok = CLOUD_REVIEW_MAX_TOKENS
        timeout = CLOUD_REVIEW_TIMEOUT_SECONDS
        _log(f"  ☁️ Cloud-Responder via ReviewLLM: model={model}")
    else:
        _log("  ✗ Cloud-Responder: Kein Cloud-Modell verfügbar")
        return {
            "agent_key": "cloud_responder",
            "status": "skipped",
            "content": "[Cloud-Responder nicht verfügbar — kein API-Key konfiguriert]",
            "duration_seconds": 0.0,
            "usage": None,
        }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": (
                "You are a senior software engineer. Execute the plan precisely. "
                "Provide complete, working code. Be thorough and detailed."
            )},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": min(max(1, int(max_tok or 32768)), 131072),
        "temperature": 0.3,
        "stream": False,
    }
    _patch_moonshot_payload(payload)
    payload = _clean_payload(payload)
    headers = {"Authorization": f"Bearer {api_key}"}

    started = time.perf_counter()
    try:
        target_url = api_url or "https://api.openai.com/v1/chat/completions"
        async with httpx.AsyncClient(timeout=timeout) as lc:
            r = await lc.post(target_url, json=payload, headers=headers)
        duration = time.perf_counter() - started
        if r.status_code == 200:
            result = r.json()
            content = _extract_choice_content(result)
            _log(f"  ✓ Cloud-Responder OK: model={model} duration={duration:.1f}s")
            return {
                "agent_key": "cloud_responder",
                "status": "ok",
                "content": content or "",
                "duration_seconds": duration,
                "usage": result.get("usage"),
            }
        _log(f"  ⚠ Cloud-Responder STATUS {r.status_code}: duration={duration:.1f}s")
        return {
            "agent_key": "cloud_responder",
            "status": "failed",
            "content": f"Cloud-Responder status {r.status_code}: {r.text[:500]}",
            "duration_seconds": duration,
            "usage": None,
        }
    except Exception as exc:
        _log(f"  ✗ Cloud-Responder ERROR: model={model} – {exc}")
        return {
            "agent_key": "cloud_responder",
            "status": "error",
            "content": f"Cloud-Responder error: {exc}",
            "duration_seconds": time.perf_counter() - started,
            "usage": None,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Review mit Feedback-Loop (User-Vision: Cloud prüft gegen Plan, ggf Re-Run)
# ═══════════════════════════════════════════════════════════════════════════
# -force review: Kimi/LiteLLM prüft die Worker-Antwort GEGEN DEN PLAN.
# - Bei PASS: ok, Done.
# - Bei NEEDS FIX: Worker wird mit dem Feedback NOCHMAL gerufen,
#   max READY_MAX_FEEDBACK_ROUNDES mal. Danach: best-effort rausgeben.
MAX_FEEDBACK_ROUNDS = 2

# ═══════════════════════════════════════════════════════════════════════════
# Deterministisches Review-Layout für Provider-Caching (DeepSeek KV-Cache,
# Moonshot 60s-Context-Cache). Der STABILE Teil (Format-Template, Task, Plan)
# steht IMMER identisch vorne, nur der variable Teil (Worker-Antwort)
# variiert zwischen Review-Runden. Prov-Cache-Hit-Rate steigt ~70-90 %
# in Feedback-Loops.  Siehe /memories/repo/localproxy-plan-execute-verify.md
# ═══════════════════════════════════════════════════════════════════════════
REVIEWER_SYSTEM_PROMPT = (
    "You are a STRICT code reviewer enforcing a plan. Be CONCISE.\n"
    "You will receive, in this exact order:\n"
    "  USER message 1: the ORIGINAL TASK (stable across rounds).\n"
    "  USER message 2: the EXECUTION PLAN (stable across rounds).\n"
    "  USER message 3: the RESPONSE TO REVIEW (changes each round).\n"
    "Check the response AGAINST THE PLAN for: correctness, completeness, "
    "bugs, edge cases, security issues. Did the worker implement the plan "
    "correctly?\n"
    "OUTPUT FORMAT (exact, no preface):\n"
    "## Review Result: [PASS / NEEDS FIX]\n"
    "## Issues Found:\n- ...\n"
    "## Suggestions:\n- ...\n"
    "If PASS, output ONLY:\n"
    "## Review Result: PASS\n"
)


def _parse_review_verdict(review_text: str) -> str:
    """Extrahiert 'PASS' / 'NEEDS FIX' aus Cloud-Review-Output."""
    if not review_text:
        return "UNKNOWN"
    r = review_text.strip().upper()
    # Pattern 1: "## Review Result: PASS" / "NEEDS FIX"
    m = re.search(r"REVIEW\s*RESULT\s*:?\s*(PASS|NEEDS?\s*FIX|FAIL)", r)
    if m:
        v = m.group(1).replace(" ", "")
        if v.startswith("NEED"):
            return "NEEDS_FIX"
        return v if v in {"PASS", "FAIL"} else "UNKNOWN"
    # Pattern 2: alleine das Wort am Anfang
    if re.match(r"^\s*PASS\b", r):
        return "PASS"
    if re.search(r"\b(NEEDS?\s*FIX|FAIL)\b", r):
        return "NEEDS_FIX"
    return "UNKNOWN"


async def _review_with_feedback_loop(
    client: httpx.AsyncClient,
    body: Dict[str, Any],
    plan: str,
    task: str,
    worker_response: str,
    force_review: bool,
    progress: List[str],
    results: List[Dict[str, Any]],
    session_hash: Optional[str] = None,
) -> str:
    """Fuehrt Cloud-Review durch + Feedback-Loop bei NEEDS FIX.

    Gibt die finale (ggf re-implementierte) Worker-Antwort zurueck.
    force_review=False → nur lokale Verifikation, kein Cloud-Loop.
    """
    if not force_review:
        verified, _ = await _verify_and_correct(client, task, worker_response)
        return verified

    rounds = 0
    current_response = worker_response
    while rounds <= MAX_FEEDBACK_ROUNDS:
        rounds += 1
        _log(f"  🔍 Review-Runde {rounds}/{MAX_FEEDBACK_ROUNDS+1}")
        progress.append(_format_chat_progress_message(
            f"phase3_review_round_{rounds}",
            f"🔍 Cloud-Review (Runde {rounds}) prüft gegen Plan.",
            {"round": rounds},
        ))
        review_r = await _call_cloud_reviewer(client, task, current_response, plan)
        results.append(review_r)
        if review_r.get("status") != "ok":
            _log("  ⚠ Review fehlgeschlagen → nehme Worker-Output ungeprueft")
            return current_response
        rtext = review_r.get("content", "").strip()
        verdict = _parse_review_verdict(rtext)
        _log(f"  🔍 Review-Verdict: {verdict}")

        if verdict == "PASS" or verdict == "UNKNOWN":
            # Pass - Review-Text an Worker-Output anhaengen, fertig
            if rtext:
                current_response = f"{current_response}\n\n---\n## 🔍 Cloud Review (PASS, Runde {rounds})\n{rtext}"
            return current_response

        # NEEDS_FIX (oder FAIL): Worker mit Feedback neu starten
        current_response = f"{current_response}\n\n---\n## 🔍 Cloud Review (NEEDS FIX, Runde {rounds})\n{rtext}"
        if rounds > MAX_FEEDBACK_ROUNDS:
            _log(f"  ⚠ Max Feedback-Runden erreicht ({MAX_FEEDBACK_ROUNDS}) → rausgeben wie es ist")
            return current_response
        progress.append(_format_chat_progress_message(
            f"phase3b_feedback_{rounds}",
            f"♻️ Review fordert Fixes → Worker-Neuausführung.",
            {"round": rounds, "verdict": verdict},
        ))
        # Plan-Path fuer Worker-Continue lookupen (Codespace-Stil)
        _plan_path = None
        if session_hash:
            _plan_path = _PLANNER_SESSIONS.get(session_hash, {}).get("plan_path")
        feedback_payload = _build_worker_payload(body, plan, "", plan_path=_plan_path)
        # CACHING-FREUNDLICHER Feedback-Payload:
        # Bisher wurde das REVIEW-FEEDBACK ans Ende der originalen User-Message geklebt
        # → das zerstört den Prefix-Cache (Original-Message-Ende ist zwischen
        # Worker-Call 1 und 2 nicht mehr identisch).
        #
        # Neu: Wir APPENDEN das Feedback als SEPARATE user-Message. Damit bleibt
        # der gesamte Prefix (system + tool-defs + plan + plan-binding + history +
        # originale User-Nachricht + Memory) byteweise identisch zwischen
        # initial Worker-Call und Fix-Runden. DeepSeek/Moonshot cachen diesen
        # Prefix, nur die appendede Feedback-Message wird neu berechnet.
        feedback_text = (
            "[REVIEW FEEDBACK - FIX THESE ISSUES]\n"
            f"{rtext}\n\n"
            "Re-apply ONLY the fixes from the review. Keep everything else intact."
        )
        feedback_payload.setdefault("messages", []).append(
            {"role": "user", "content": feedback_text}
        )
        re_result = await _call_vllm_with_fallback(client, feedback_payload, "worker_feedback")
        results.append(re_result)
        if re_result.get("status") == "ok":
            current_response = re_result.get("content", "")
            _log(f"  ♻️ Worker-Neuausführung fertig ({len(current_response)} chars)")
        else:
            _log("  ⚠ Worker-Neuausführung fehlgeschlagen → letzter Stand")
            return current_response
    # Fallback (sicherheitshalber)
    return current_response


# ═══════════════════════════════════════════════════════════════════════════
# Phase 0: Caveman Compression VOR dem Cloud-Planner
# ═══════════════════════════════════════════════════════════════════════════
# User-Vision: Bevor der teure Cloud-Planner (Kimi) die User-Prompt verarbeitet,
# wird ein CHEAP Modell (FAST_MODEL = deepseek-v4-flash) die User-Prompt ins
# Caveman-Format komprimieren. Das spart Cloud-Tokens UND zwingt den Planner
# auf das Wesentliche zu fokussieren.

PHASE0_COMPRESS_SYSTEM = (
    "You are a CAVEMAN PROMPT COMPRESSOR. Input: a user request (long or short). "
    "Output: the SAME intent in 30-200 chars max, using only terse keywords + arrows. "
    "Strip politeness, filler, context-noise. Preserve: file names, function names, "
    "intent verbs (add/fix/refactor). Format example:\n"
    "  Input: 'Could you please add a function validateEmail to the User model?'\n"
    "  Output: 'ADD fn validateEmail -> User model. validate RFC email. fail→throw Error.'\n"
    "Output ONLY the compressed prompt, no preface."
)


async def _phase0_compress_prompt(
    client: httpx.AsyncClient,
    user_text: str,
) -> str:
    """Komprimiert die User-Prompt mittels FAST_MODEL (deepseek-v4-flash) in
    Caveman-Keywords. Gibt bei Fehler das Original zurueck (fail-open).
    """
    if not user_text or len(user_text) < 80 or not FAST_MODEL_NAME:
        # Zu kurz oder kein Fast-Model → Compression skipped, Original zurueck
        return user_text

    payload = {
        "model": FAST_MODEL_NAME,
        "messages": [
            {"role": "system", "content": PHASE0_COMPRESS_SYSTEM},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": 512,
        "temperature": 0.0,
        "stream": False,
    }
    payload = _clean_payload(payload)
    _patch_moonshot_payload(payload)

    started = time.perf_counter()
    try:
        # Zustatz-Header: der Fast-Worker laeuft ueber eigenem Endpoint
        headers = _vllm_headers()
        _clean = VLLM_API_URL.rstrip("/")
        if _clean.endswith("/chat/completions") or _clean.endswith("/v1"):
            url = VLLM_API_URL
        else:
            url = _clean + "/chat/completions"
        response = await client.post(url, json=payload, headers=headers, timeout=20.0)
        duration = time.perf_counter() - started
        if response.status_code == 200:
            rj = response.json()
            # DeepSeek V4 legt Output bei temperature=0 häufig in reasoning_content,
            # NICHT in content (BUG: _extract_choice_content hier nicht verwenden).
            # Deshalb hier direkt content ODER reasoning_content ziehen.
            msg = _extract_choice_message(rj)
            compressed = (msg.get("content") or msg.get("reasoning_content") or "").strip()
            if compressed and 10 < len(compressed) < len(user_text):
                _log(f"  🗜 Phase-0 Compression: {len(user_text)} → {len(compressed)} chars ({duration:.1f}s)")
                return compressed
            _log(f"  ⚠ Phase-0 Compression unnuetz (len {len(compressed)}) → Original")
            return user_text
        _log(f"  ⚠ Phase-0 Compression STATUS {response.status_code} → Original")
        return user_text
    except Exception as exc:
        _log(f"  ⚠ Phase-0 Compression ERROR ({exc}) → Original")
        return user_text


# ═══════════════════════════════════════════════════════════════════════════
# Planner-Payload-Konstruktion (Task-zentriert, Recap-basiert ab Runde 2)
# ═══════════════════════════════════════════════════════════════════════════
# Warum eigene Logik statt einfach body weiterzureichen?
#
# In VSCode-Copilot-Chatsessions wächst die Historie. In Runde 15+ sieht
# die "letzte" User-Message für Kimi so aus:
#   [tool] grep_search_98: 100 matches, 111039 chars
# Dazwischenliegen 14 tool-results. Der ursprüngliche Task ("Mach einen
# Plan für Feature X") ist im Verlauf begraben. Kimi hat keinen Anker,
# was es eigentlich tun soll → sucht weiter.
#
# Lösung: Aktive Tool-Continuation-Runden kriegen einen EIGENEN Planner-
# Payload, der aus 4 Teilen besteht:
#   1. Planner-System ("You are a planner, EXPLORE then PLAN")
#   2. USER-TASK als eigene user-message an oberster Stelle
#   3. EXPLORATION RECAP: "You have already done 17 rounds. Tools you ran: …"
#   4. Tool-Continuation: Die 2 letzten tool-results UND der von Kimi
#      gewünschte nächste Tool-Call (als assistant-message mit tool_calls)
#
# So weiß Kimi in Runde N: was ist mein Auftrag, was habe ich schon,
# was ist der nächste Schritt. Kein plan-begraben-in-Rauschen mehr.


def _extract_original_task(messages: Sequence[Dict[str, Any]]) -> str:
    """Findet die ursprüngliche User-Query (task) in der History.

    Strategie: die ERSTE User-Message, die nicht leer ist und nicht
    reiner Pipeline-Flag-Text ist. Falls VS Code zusätzlichen System-
    Prime führt, wird dieser übersprungen.
    """
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        text = _message_text(msg)
        # Pipeline-Flags strippen
        cleaned, flags = _extract_pipeline_flags(text)
        # Avatars/system-prime-text überspringen: mins 15 chars nach Flag-clean
        if len(cleaned.strip()) < 12:
            continue
        return cleaned.strip()
    return ""


def _summarize_exploration(messages: Sequence[Dict[str, Any]], max_items: int = 25) -> str:
    """Baut ein kompaktes Recap aller tool-Aktivitäten in der Historie.

    Beispiel-Output:
      EXPLORATION RECAP (so far): 17 rounds, 18 distinct tools, 21 files inspected.
      Tools executed: read_file (12×), grep_search (4×), list_dir (2×)
      Files seen:
        - d:/foo/bar.py
        - d:/baz/qux.py
        ...
      Last tool result: <truncated 400 chars>
    """
    tool_results: List[Tuple[str, str]] = []
    tool_call_names: List[str] = []
    files_seen: List[str] = []
    last_tool_result_preview: str = ""

    tool_pattern_file_keys = ("filepath", "filename", "filepath_relative",
                               "path", "includepattern", "uri")

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            tcs = msg.get("tool_calls")
            if isinstance(tcs, list):
                for tc in tcs:
                    if not isinstance(tc, dict):
                        continue
                    func = tc.get("function") or {}
                    name = str(func.get("name", "?")).lower()
                    tool_call_names.append(name)
                    # File-Ref extrahieren
                    args_raw = func.get("arguments", "")
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw if isinstance(args_raw, dict) else {})
                    except Exception:
                        args = {}
                    for k, v in args.items():
                        if k.lower() in tool_pattern_file_keys and v:
                            ref = f"{name}:{str(v)[:120]}"
                            if ref not in files_seen:
                                files_seen.append(ref)
        elif role == "tool":
            tn = str(msg.get("name", "") or "tool")[:40]
            content = _message_text(msg)
            tool_results.append((tn, content))
            if content:
                last_tool_result_preview = content

    if not tool_call_names:
        return ""

    # Tool-Nutzung aggregieren
    from collections import Counter
    counter = Counter(tool_call_names)
    tool_summary = ", ".join(f"{name} ({n}×)" for name, n in counter.most_common())

    lines = [
        f"EXPLORATION RECAP — you have already done {len(tool_call_names)} tool-calls "
        f"across {len(files_seen)} distinct files/queries.",
        f"Tools executed so far: {tool_summary}",
    ]
    if files_seen:
        shown = files_seen[:max_items]
        lines.append("Files/queries inspected (most recent first):")
        for ref in shown:
            lines.append(f"  - {ref}")
        if len(files_seen) > max_items:
            lines.append(f"  ... and {len(files_seen) - max_items} more.")
    if last_tool_result_preview:
        preview = last_tool_result_preview[:400].replace("\n", " ")
        lines.append(f"LAST tool result (truncated, 400 chars): {preview}")
    return "\n".join(lines)


def _build_planner_tool_continuation_context(
    body: Dict[str, Any],
    session: Dict[str, Any],
    original_task: str,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Baut einen KOMPAKTEN Planner-Payload für Runde N+1 (Tool-Cont.)

    Anstatt dem Kimi die volle 200+ Message-History (inklusive 111KB
    grep-Treffern) vorzuwerfen, kriegt es einen fokussierten Payload:

      [system] Originaler VS-Code-Systemprompt + Planner-Instruktionen
      [user]   ORIGINAL_TASK
      [user]   EXPLORATION RECAP (was bereits untersucht wurde) + nudge

    Wichtig: KEIN tool_calls/tool_results-Durchreich mehr!
    Früher wurde die letzte assistant-tool_calls + passende tool_results
    kopiert, was zu 'tool_call_id not found' (400) und invaliden tool_calls
    (filePath fehlt) führte. Jetzt startet Kimi in jeder Recap-Runde frisch
    — es kann entweder weiter explorieren (valide tool_calls) oder den Plan
    ausgeben.

    'tools'-Schemas werden mitgegeben (filtered, read-only).
    """
    messages = body.get("messages", [])
    iterations = int(session.get("iterations", 0))
    distinct_files_count = len(session.get("distinct_files") or [])

    # ══ System-Prompt: originalen Kontext BEWAHREN + Planner-Instruktionen anhängen ══
    # WICHTIG: Der originale System-Prompt enthält Tool-Schemas, JSON-Format-Vorgaben
    # und <toolUseInstructions>. Ohne diese generiert Kimi invalide tool_calls
    # (z.B. read_file ohne filePath → 400 von VS Code).
    # Gleiche Strategie wie in Runde 1 (dort wird planner_system auch angehängt).
    original_system = ""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            original_system = str(msg.get("content", ""))
            break
    planner_instructions = (
        "\n\n[PLANNER AGENT MODE — appended by proxy]\n"
        "You are now acting as a STRATEGIC PLANNING AGENT. You have access to VS Code tools. "
        "Your job: EXPLORE the workspace, UNDERSTAND the user's task, then "
        "PRODUCE AN EXECUTION PLAN in Markdown.\n\n"
        "Rules:\n"
        "1. EXPLORE while you need more info (read_file, grep_search, list_dir).\n"
        "2. STOP exploring as soon as you have enough context.\n"
        "3. Then OUTPUT a Plan: format `## Plan: <title>` + numbered steps.\n"
        "   - 10-20 concrete steps, each with file path + WHAT to change + WHY.\n"
        "   - Max 4000 chars total. Use terse caveman compression.\n"
        "   - DO NOT write the code. The worker implements it.\n"
        "4. If a tool result is unhelpful (e.g. empty matches), DO NOT retry "
        "the same tool with the same args — change strategy or move on.\n"
    )
    if original_system:
        planner_system = original_system + planner_instructions
    else:
        planner_system = planner_instructions

    new_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": planner_system},
        {"role": "user", "content": f"USER TASK (do not lose sight of this):\n{original_task}"},
    ]

    # Exploration-Recap als user-message, mit explorations-bedingtem nudge
    recap = _summarize_exploration(messages, max_items=25)
    if recap:
        # Nudge nach ausreichender Exploration, Härte steigt mit iterations
        nudge = ""
        if iterations >= 8 and distinct_files_count >= 5:
            nudge = (
                "\n\n⚠️ You have done many rounds of exploration. The workspace is sufficiently "
                "understood now. STOP calling tools. OUTPUT THE PLAN NOW."
            )
        elif iterations >= 4:
            nudge = (
                "\n\nℹ️ You have explored enough. If you have enough context, output the plan. "
                "Only call another tool if truly necessary."
            )
        new_messages.append({"role": "user", "content": recap + nudge})

    # ══ KEIN tool_calls/tool_results-Durchreich mehr ═══════════════════
    # Früher wurde die letzte assistant-tool_calls-Message + passende
    # tool_results in den Recap-Payload kopiert. Das hat in der Praxis
    # zu zwei Fehlern geführt:
    #   1) tool_call_id mismatch (Kimi 400 'tool_call_id not found') weil
    #      tool_results aus vorherigen Sessions die IDs überschreiben.
    #   2) Kimi generiert im Recap-Modus invalide tool_calls (ohne filePath,
    #      path, command) weil der kontextuelle Bezug fehlt.
    #
    # Neuer Ansatz: Recap enthält NUR System + Task + Exploration-Summary.
    # Kimi startet in jeder Recap-Runde frisch: entweder es exploriert
    # weiter (mit validen tool_calls) oder es gibt den Plan aus.
    # Der _summarize_exploration()-Text sagt Kimi was es schon gesehen hat.

    new_payload = copy.deepcopy(body)
    new_payload["messages"] = new_messages
    # tools beibehalten (read-only subset) — caller muss schon filtern
    if tools is not None:
        new_payload["tools"] = tools
    new_payload["stream"] = False
    new_payload["max_tokens"] = min(CAVEMAN_MAX_TOKENS, 65536)
    new_payload["temperature"] = 0.2
    return new_payload


async def _call_cloud_planner_agent(
    client: httpx.AsyncClient,
    body: Dict[str, Any],
    memory_context: str = "",
    parent_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Cloud-Planner im Copilot Plan-Mode Stil.

    EXAKT wie Copilot: Discovery mit read-only Tools → Alignment → Design → Plan schreiben.
    Tools: 20 read-only (wie Copilot Plan Mode). Keine Edit-Tools.
    Workflow: siehe planner_mode_instructions.
    """
    if not (CLOUD_REVIEW_ENABLED and CLOUD_REVIEW_API_KEY) and not (LITELLM_CLOUD_MODEL and LITELLM_CLOUD_API_KEY):
        return {
            "agent_key": "cloud_planner_agent",
            "status": "skipped", "content": "",
            "duration_seconds": 0.0, "usage": None,
        }

    _log(f"  🧠 Cloud-Planner-Agent: model={CLOUD_REVIEW_MODEL} url={CLOUD_REVIEW_API_URL}")

    body_messages = body.get("messages", [])
    body_sh = _get_planner_session_hash(body_messages)
    body_session = _PLANNER_SESSIONS.get(body_sh)
    is_tool_continuation = (
        body_session is not None
        and body_session.get("state") == "active"
        and int(body_session.get("iterations", 0)) >= 2
    )

    # ══ COPER-STYLE plan mode instructions ════════════════════════════
    planner_mode_instructions = (
        "<modeInstructions>\n"
        "You are currently running in \"Plan\" mode. "
        "Below are your instructions for this mode, they must take precedence "
        "over any instructions above.\n\n"
        "You are a PLANNING AGENT, pairing with the user to create a detailed, "
        "actionable plan.\n\n"
        "You research the codebase → clarify with the user → capture findings "
        "into a comprehensive plan. This iterative approach catches edge cases "
        "and non-obvious requirements BEFORE implementation begins.\n\n"
        "Your SOLE responsibility is planning. NEVER start implementation.\n\n"
        "<exploration_budget>\n"
        "DISCIPLINE: Aim for AT MOST 5–8 read tool calls before producing the plan.\n"
        "The editor model (DeepSeek) will read the relevant files ITSELF before "
        "editing — you do NOT need to dump every line of every file into your context.\n"
        "DO NOT read the same file twice. DO NOT chase every reference; capture the\n"
        "top-level shape of the change (entry points, signatures, where they live)\n"
        "and move to the plan. Reference files by path + symbol name, not by quoted\n"
        "source. If you find yourself reading a 6th+ file, STOP and write the plan\n"
        "with what you already know.\n"
        "</exploration_budget>\n\n"
        "<rules>\n"
        "- STOP if you consider running file editing tools — plans are for "
        "others to execute. The only write tool you have is 'memory' for persisting plans.\n"
        "- Use 'vscode_askQuestions' freely to clarify requirements — don't make large assumptions\n"
        "- Present a well-researched plan BEFORE implementation\n"
        "</rules>\n\n"
        "<workflow>\n"
        "Cycle through these phases. This is iterative, not linear.\n\n"
        "## 1. Discovery\n"
        "Use read_file, grep_search, file_search, list_dir to gather context. "
        "Explore analogous existing features to use as implementation templates. "
        "When the task spans multiple independent areas, launch tools in parallel "
        "to speed up discovery.\n"
        "KEEP IT TIGHT: the goal is to identify the touch points and the shape of\n"
        "the change, not to read every related line. 3–6 well-chosen reads usually\n"
        "suffice. Update your findings via 'memory'.\n\n"
        "## 2. Alignment\n"
        "If research reveals major ambiguities or you need to validate assumptions:\n"
        "- Use 'vscode_askQuestions' to clarify intent with the user.\n"
        "- Surface discovered technical constraints or alternative approaches\n"
        "- If answers significantly change scope, loop back to **Discovery**\n\n"
        "## 3. Design\n"
        "Draft a comprehensive implementation plan:\n"
        "- Structured, concise, scannable, detailed enough for effective execution\n"
        "- Step-by-step with explicit dependencies — mark parallel vs. blocking steps\n"
        "- Group into named phases, each independently verifiable\n"
        "- Verification steps for validating — both automated and manual\n"
        "- Reference specific functions, types, patterns, not just file names\n"
        "- Critical files to modify (with full paths)\n"
        "- Explicit scope boundaries — included AND excluded\n"
        "- Leave no ambiguity\n\n"
        "Save the plan to the plan file via 'memory', then show it to the user.\n\n"
        "## 4. Refinement\n"
        "On user input after showing the plan:\n"
        "- Changes → revise and present updated plan\n"
        "- Questions → clarify, or use 'vscode_askQuestions'\n"
        "- Alternatives → loop back to **Discovery**\n"
        "- Approval → acknowledge completion\n\n"
        "Keep iterating until explicit approval.\n"
        "</workflow>\n\n"
        "<plan_style_guide>\n"
        "## Plan: {Title (2-10 words)}\n"
        "{TL;DR - what, why, and how (your recommended approach).}\n\n"
        "**Steps**\n"
        "1. {Step — note dependency or parallelism}\n"
        "2. {Step}\n\n"
        "**Relevant files**\n"
        "- `{full/path/to/file}` — {what to modify or reuse}\n\n"
        "**Verification**\n"
        "1. {Specific verification step}\n\n"
        "**Decisions** (if applicable)\n"
        "- {Decision and rationale}\n\n"
        "**Further Considerations** (if applicable, 1-3 items)\n"
        "1. {Clarifying question with recommendation}\n\n"
        "Rules:\n"
        "- NO code blocks — describe changes, reference specific symbols\n"
        "- The plan MUST be presented to the user\n"
        "</plan_style_guide>\n"
        "</modeInstructions>"
    )

    if is_tool_continuation:
        _log(f"  🎯 Planner-Cont-Modus (round {body_session.get('iterations')}): "
             f"volle History, {len(body_messages)} messages")
        payload = copy.deepcopy(body)
        payload["model"] = CLOUD_REVIEW_MODEL
        payload["max_tokens"] = min(CAVEMAN_MAX_TOKENS, 65536)
        payload["temperature"] = 0.2
        payload["stream"] = False

        messages = list(payload.get("messages", []))

        # ══ Planner-Cont Window: System + Task + letzte N Messages ══
        # Kein Sliding Window mehr — das zerstörte Tool-Integrität
        # (verwaiste tc_ids → 400 'tool_call_id not found').
        # ══ Window mit Tool-Cluster-Atomicität ═════════════════════════
        # Vorheriger Bug: keep = keep[-(MAX_PLANNER_WINDOW - 2):] schnitt
        # mitten durch einen assistant(tool_calls) → tool1..toolN Cluster.
        # Folge: isolierten tool-Nachrichten ohne Elter, die zu user gemacht
        # wurden → 4-5 user-Msgs am Stück im Payload → Moonshot hat sich bei
        # >100KB mit Strukturfehlern自己和 (90s+ timeout, keine Antwort).
        #
        # NEU: Cluster-aware Cut. Ein 'Cluster' = assistant(tool_calls) +
        # all seine tool-Ergebnisse. Wir schneiden NUR an Cluster-Grenzen
        # und NIE mittendrin. Zusätzlich: User-Fragmente zwischen Clustern
        # VERWERFEN (das sind Proxy/VS Code-Injektionen, keine echte User-
        # Eingaben - die echte User-Aufgabe steht in first_user_msg).
        system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
        first_user_msg = None
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                if isinstance(m.get("content", ""), str) and m["content"].strip():
                    first_user_msg = m
                    break

        # Aider-Äquivalent: Architect bekommt eine begrenzte Kontextgröße
        # (dort via repo-map), nicht 30+ Tool-Resultate. Bewiesenermaßen
        # lief Runde 11 (15 msgs, 11.4s) noch, Runde 12 (45 msgs, 160KB)
        # nicht mehr → Moonshot-Timeout. 22 = System+Task+ca. 9 Cluster.
        MAX_PLANNER_WINDOW = 22

        def _build_clusters(msgs):
            """Gruppiert Messages in Cluster: assistant(tool_calls)+alle tools
            danach bis zur nächsten nicht-tool-Message. Andere Messages
            (assistant-ohne-tool_calls, user, system) sind eigenständige
            Cluster."""
            clusters = []
            i = 0
            while i < len(msgs):
                m = msgs[i]
                if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls"):
                    cluster = [m]
                    j = i + 1
                    while j < len(msgs) and msgs[j].get("role") == "tool":
                        cluster.append(msgs[j])
                        j += 1
                    clusters.append(cluster)
                    i = j
                else:
                    clusters.append([m])
                    i += 1
            return clusters

        if len(messages) > MAX_PLANNER_WINDOW:
            # Cluster bilden, system/first_user extrahieren
            clusters_raw = _build_clusters(messages)

            # Behalte nur Cluster, die NICHT system_msg oder first_user_msg sind
            keep_clusters = []
            for c in clusters_raw:
                # 1-elementiger Cluster mit system/first_user → skip
                if len(c) == 1 and c[0] is system_msg:
                    continue
                if len(c) == 1 and c[0] is first_user_msg:
                    continue
                keep_clusters.append(c)

            # Solange die Gesamt-Message-Zahl zu groß ist: älteste Cluster
            # wegwerfen (FIFO), bis unter Limit ODER nur noch wenige übrig.
            def _cluster_total(clist):
                return sum(len(c) for c in clist)

            # System + first_user brauchen 2 Slots
            available = MAX_PLANNER_WINDOW - 2
            while _cluster_total(keep_clusters) > available and len(keep_clusters) > 2:
                dropped = keep_clusters.pop(0)
                _log(f"  🗜 Planner-Cont: Cluster gedropped "
                     f"({len(dropped)} msgs, started role={dropped[0].get('role')})")

            # Re- zusammensetzen in korrekter Reihenfolge
            result = []
            if system_msg:
                result.append({"role": "system",
                               "content": str(system_msg.get("content", "")) + "\n\n" + planner_mode_instructions})
            if first_user_msg:
                result.append(first_user_msg)
            for c in keep_clusters:
                result.extend(c)
            _log(f"  🗜 Planner-Cont: {len(messages)} → {len(result)} "
                 f"(System + Task + {len(keep_clusters)} Tool-Cluster, atomar geschnitten)")
            messages = result
        else:
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = str(messages[0].get("content", "")) + "\n\n" + planner_mode_instructions

        # reasoning_content strippen — Kimi's Thinking crasht DeepSeek
        removed = 0
        for m in messages:
            if isinstance(m, dict) and "reasoning_content" in m:
                del m["reasoning_content"]
                removed += 1
        if removed:
            _log(f"  🧹 Planner-Cont: {removed} reasoning_content entfernt")

        # Orphaned-Tool-Cleanup: tool→user wenn kein passender assistant(tool_calls)
        valid_tc_ids = set()
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "assistant":
                tcs = m.get("tool_calls")
                if isinstance(tcs, list):
                    for tc in tcs:
                        if isinstance(tc, dict) and tc.get("id"):
                            valid_tc_ids.add(tc["id"])
        orphaned = 0
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "tool":
                tc_id = m.get("tool_call_id", "")
                if tc_id and tc_id not in valid_tc_ids:
                    m["role"] = "user"
                    m.pop("tool_call_id", None)
                    orphaned += 1
        if orphaned:
            _log(f"  🧹 Planner-Cont: {orphaned} verwaiste tool→user konvertiert")

        # Tool-Result-Cap bewusst ENTFERNT — siehe Aider Architect/Editor.
        # Der Architect braucht vollen Code-Sicht, um valide Pläne zu
        # schreiben. Statt Capping wird die Window-Size (MAX_PLANNER_WINDOW)
        # verkleinert — älteste Cluster fallen ganz raus, die verbleibenden
        # Tools liefern unverfälschte Dateiinhalte.
        payload["messages"] = messages
    else:
        # ══ RUNDE 1: Copilot-style plan mode ══
        payload = copy.deepcopy(body)
        payload["model"] = CLOUD_REVIEW_MODEL
        payload["max_tokens"] = min(CAVEMAN_MAX_TOKENS, 65536)
        payload["temperature"] = 0.2
        payload["stream"] = False

        messages = list(payload.get("messages", []))
        _cap_tool_results_inplace(messages, "Planner-R1", max_chars=0)
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = str(messages[0].get("content", "")) + "\n\n" + planner_mode_instructions
        else:
            messages.insert(0, {"role": "system", "content": planner_mode_instructions})
        payload["messages"] = messages

        if memory_context:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    msg["content"] = str(msg.get("content", "")) + f"\n\n[HINDSIGHT]\n{memory_context}"
                    break

    # ══ Tools: genau wie Copilot — 20 read-only von Runde 1 an ══
    payload = _clean_payload(payload, keep_tools=True)
    _patch_moonshot_payload(payload)

    if "tools" in payload:
        filtered = _filter_planner_tools(payload.get("tools"))
        if filtered and len(filtered) < len(payload["tools"]):
            _log(f"  🔧 Planner-Tools: {len(payload['tools'])} → {len(filtered)} (Copilot Plan-Mode read-only)")
            payload["tools"] = filtered
        elif not filtered:
            _log("  ⚠ Keine passenden Planner-Tools → entfernt")
            payload.pop("tools", None)
            payload.pop("tool_choice", None)

    # ══ Loop-Detection (nur Safety-Net, nicht zu aggressiv) ══
    sh = _get_planner_session_hash(payload.get("messages", []))
    existing_session = _PLANNER_SESSIONS.get(sh) or body_session
    cur_messages = payload.get("messages", [])
    if existing_session:
        loop_reason = _detect_planner_loop(existing_session)
        if loop_reason:
            iteration = int(existing_session.get("iterations", 0))
            _log(f"  🛑 Planner LOOP erkannt ({loop_reason}, iter={iteration}) → erzwinge Plan-Output")
            payload.pop("tools", None)
            payload.pop("tool_choice", None)
            # Alle tool_calls aus assistant-Messages entfernen — ohne tools
            # im payload darf Kimi keine tool_calls generieren, und die API
            # lehnt Requests ab wenn assistant mit tool_calls nicht von
            # tool_results gefolgt wird (400 'tool_call_id not found').
            # FRÜHER: nur bei tc_count > following_results entfernt — aber
            # auch wenn genug results folgen, crasht die API ohne tools.
            for msg in cur_messages:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    if msg.get("tool_calls"):
                        _log(f"  🧹 Loop-Detection: {len(msg['tool_calls'])} tool_calls auf assistant entfernt")
                        msg.pop("tool_calls", None)
                    # Reasoning-Content entfernen, damit Kimi nicht denkt
                    # es sei noch im Tool-Modus.
                    msg.pop("reasoning_content", None)
                    # KRITISCH: Moonshot/Mimi wirft 400 "assistant must not be empty"
                    # wenn content==""/None UND tool_calls entfernt wurden. Daher
                    # synthetischen Kontext-Platzhalter einsetzen. Dies ist nur
                    # string-fill um die API zu validieren - die_MESSAGE hat keine
                    # Information weil das echte Workspace-Wissen in den
                    # tool-Resultaten steckt, die wir gerade zu user gemacht haben.
                    c = msg.get("content")
                    if not (isinstance(c, str) and c.strip()):
                        msg["content"] = (
                            "[FRÜHERE TOOL-AUFRUFE ENTFERNT — Lies die folgenden "
                            "Tool-Ergebnisse in den user-Nachrichten als Kontext.]"
                        )
            # Nachdem ALLE tool_calls aus assistants entfernt wurden, haben
            # die tool-Rollen-Nachrichten verwaiste tool_call_ids, die von
            # der API mit 400 'tool_call_id not found' abgelehnt werden.
            # Lösung: tool → user konvertieren (Content bleibt erhalten,
            # tc_id wird entfernt, keine Abhängigkeit mehr).
            orphaned = 0
            for msg in cur_messages:
                if isinstance(msg, dict) and msg.get("role") == "tool":
                    msg["role"] = "user"
                    msg.pop("tool_call_id", None)
                    orphaned += 1
            if orphaned:
                _log(f"  🧹 Loop-Detection: {orphaned} tool-Nachrichten → user konvertiert (tc_id entfernt)")
            if cur_messages and isinstance(cur_messages[-1], dict) and cur_messages[-1].get("role") == "user":
                force_msg = (
                    "\n\n[SYSTEM: Du bist über die Explorationsschwelle gekommen "
                    f"({loop_reason}). Du hast GENUG Code-Kontext. Gib JETZT deinen "
                    "finalen Ausfuehrungs-Plan aus.\n"
                    "KEINE weiteren tool_calls. Der Editor liest die Dateien SELBST.\n"
                    "Pflicht-Format:\n"
                    "  ## Plan: <2-6 word title>\n"
                    "  <TL;DR: was + warum, 2 Sätze>\n"
                    "  **Steps**\n"
                    "  1. < konkrete Anweisung mit Dateipfad + Symbolname >\n"
                    "  2. ...\n"
                    "  **Relevant files**\n"
                    "  - `<full/path>` — <was ändern>\n"
                    "  **Verification**\n"
                    "  1. <wie prüfen>\n"
                    "Schreibe nur den Plan, nichts davor/danach.]"
                )
                user_content = cur_messages[-1].get("content", "")
                if isinstance(user_content, str):
                    cur_messages[-1]["content"] = user_content + force_msg
        else:
            # Sanfter Nudge nach ausreichender Exploration (kein Hard-Stop)
            warn = _should_warn_planner(existing_session)
            if warn:
                if cur_messages and isinstance(cur_messages[-1], dict) and cur_messages[-1].get("role") == "user":
                    user_content = cur_messages[-1].get("content", "")
                    if isinstance(user_content, str) and warn not in user_content:
                        cur_messages[-1]["content"] = user_content + f"\n\n{warn}"

    # ══ FALLBACK-KETTE (Kimi → DeepSeek V4 Pro) ═════════════════════
    # DEF-Strategie: Probiere primär Kimi (CLOUD_REVIEW_*). Bei non-200,
    # Exception, oder leerem Plan → versuche DeepSeek V4 Pro via LiteLLM.
    # Da _patch_moonshot_payload und Stage-spezifische Felder MUTIEREND sind,
    # muss pro Stuge eine FRISCHE Payload-Kopie erzeugt werden.
    planner_chain = []
    if CLOUD_REVIEW_ENABLED and CLOUD_REVIEW_API_KEY:
        planner_chain.append({
            "name": "KIMI",
            "model": CLOUD_REVIEW_MODEL,
            "api_key": CLOUD_REVIEW_API_KEY,
            "api_url": CLOUD_REVIEW_API_URL,
            "timeout": CLOUD_REVIEW_TIMEOUT_SECONDS,
        })
    if LITELLM_CLOUD_MODEL and LITELLM_CLOUD_API_KEY:
        planner_chain.append({
            "name": "DEEPSEEK-V4-PRO",
            "model": LITELLM_CLOUD_MODEL,
            "api_key": LITELLM_CLOUD_API_KEY,
            "api_url": LITELLM_CLOUD_API_URL or "https://api.openai.com/v1/chat/completions",
            "timeout": LITELLM_CLOUD_TIMEOUT_SECONDS,
        })

    started_total = time.perf_counter()
    last_error = ""

    # DEBUG: Snapshot des Outbound-Payloads speichern (vor Stufen-spezifischen Patches)
    planner_call_id = f"planner_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    _dump_debug_payload(planner_call_id, "planner_pre_stage",
                         payload,
                         extra={"planner_chain": [s["name"] for s in planner_chain],
                                "session_hash": sh,
                                "messages_count": len(payload.get("messages", []))})
    _register_debug_request(planner_call_id, {
        "type": "planner_call_start",
        "chain": [s["name"] for s in planner_chain],
        "session_hash": sh,
        "messages_count": len(payload.get("messages", [])),
        "tools_count": len(payload.get("tools", [])),
    })

    for stage_idx, stage in enumerate(planner_chain):
        stage_name = stage["name"]
        stage_label = f"[{stage_idx+1}/{len(planner_chain)} {stage_name}]"
        _log(f"  🧠 Cloud-Planner-Agent {stage_label}: model={stage['model']}")

        # Frisches Payload pro Stufe (mutierende Patches sonst zerstören Loop-Iter)
        stage_payload = copy.deepcopy(payload)
        stage_payload["model"] = stage["model"]
        _patch_moonshot_payload(stage_payload)  # nur wirksam wenn model/url Moonshot ist
        # image_url-Sanitizer für text-only Models (DeepSeek V4).
        # Moonshot/Kimi akzeptiert image_url, DeepSeek wirft 400.
        if _is_text_only_model(stage["model"]):
            _sanitize_image_urls_inplace(stage_payload.get("messages", []), f"Planner-{stage_name}")

        # DEBUG: Per-Stage-Payload dumpen
        stage_call_id = f"{planner_call_id}_s{stage_idx+1}_{stage_name}"
        _dump_debug_payload(stage_call_id, f"planner_stage_{stage_name}",
                             stage_payload,
                             extra={"stage": stage_name, "model": stage["model"],
                                    "timeout": stage["timeout"]})

        headers = {"Authorization": f"Bearer {stage['api_key']}"}
        started = time.perf_counter()

        # Active-Call-Tracking + Heartbeat (für "Kimi hängt" Visibility)
        _register_active_call(stage_call_id, {
            "agent_key": "cloud_planner_agent",
            "model": stage["model"],
            "phase": f"planner_stage_{stage_name}",
            "stage_label": stage_label,
            "url": stage["api_url"],
            "timeout": stage["timeout"],
        })

        try:
            response = await client.post(
                stage["api_url"],
                json=stage_payload,
                headers=headers,
                timeout=stage["timeout"],
            )
            duration = time.perf_counter() - started
            _finish_active_call(stage_call_id, status="response", extra={
                "duration_seconds": duration,
                "http_status": response.status_code,
            })
            if response.status_code == 200:
                result = response.json()
                message = _extract_choice_message(result)
                raw_tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
                content = _extract_choice_content(result)

                # ══ Bug-B-Fix: Tool-Calls normalisieren ══════════════════
                tool_calls = _normalize_tool_calls(raw_tool_calls, allowed=_PLANNER_ALLOWED_TOOLS) if raw_tool_calls else None

                if tool_calls:
                    _log(f"  🔧 Cloud-Planner {stage_label} returned {len(raw_tool_calls)} tool_calls, "
                         f"{len(tool_calls)} nach Normalisierung ({duration:.1f}s)")
                    content_out = content or ""
                elif content and content.strip():
                    _log(f"  ✓ Cloud-Planner {stage_label} Plan: {len(content)} chars ({duration:.1f}s)")
                    content_out = content
                else:
                    _log(f"  ⚠ Cloud-Planner {stage_label}: 200 aber kein tool_calls UND leerer Plan → nächste Stufe")
                    last_error = f"empty plan ({stage_name})"
                    continue

                return {
                    "agent_key": "cloud_planner_agent",
                    "status": "ok",
                    "content": content_out,
                    "tool_calls": tool_calls,
                    "duration_seconds": time.perf_counter() - started_total,
                    "usage": result.get("usage"),
                }
            _log(f"  ⚠ Cloud-Planner {stage_label} STATUS {response.status_code}: {response.text[:300]}")
            last_error = f"HTTP {response.status_code} ({stage_name})"
        except Exception as exc:
            _log(f"  ✗ Cloud-Planner {stage_label} ERROR: {type(exc).__name__}: {exc}")
            last_error = f"{type(exc).__name__}: {exc} ({stage_name})"
            _finish_active_call(stage_call_id, status="error", extra={
                "error": str(exc), "error_type": type(exc).__name__,
            })

    _log(f"  ✗ Cloud-Planner: Alle Stufen fehlgeschlagen – letzter Fehler: {last_error}")
    return {
        "agent_key": "cloud_planner_agent",
        "status": "failed",
        "content": f"Planner all-stages failed: {last_error}",
        "duration_seconds": time.perf_counter() - started_total, "usage": None,
    }


async def _call_cloud_reviewer(
    client: httpx.AsyncClient,
    task: str,
    response: str,
    plan: str = "",
) -> Dict[str, Any]:
    """Cloud-Review: Sendet Worker-Antwort + Plan zur Qualitätsprüfung an die Cloud.
    
    CACHING-OPTIMIERTES LAYOUT (DeepSeek KV-Cache / Moonshot 60s-Cache):
    Die drei Action-Inputs (task, plan, response) werden in SEPARATE user-Messages
    zerlegt statt in einen einzigen User-Prompt gemischt. So bleiben der System-Prompt,
    die Task und der Plan als IDENTISCHER PREFIX über alle Review-Runden erhalten.
    Nur die letzte user-Nachricht (response) ändert sich pro Runde → Cache-Hit für
    System+Task+Plan, nur der Diff wird neu berechnet.
    """
    # Stable Marker-Konstanten, die zwischen allen Feedback-Runden identisch sind.
    # Niemals f-String mit variablen Inhalten hier bauen - damit Prefix stabil bleibt.
    task_msg = {"role": "user", "content": f"ORIGINAL TASK:\n{task}"}
    plan_msg = {"role": "user", "content": f"EXECUTION PLAN:\n{plan or '(no plan provided)'}"}
    response_msg = {"role": "user", "content": f"RESPONSE TO REVIEW (contains the actual changes/code):\n{response}\n\nReview now."}

    # ══ FALLBACK-KETTE (Kimi → DeepSeek V4 Pro) ═════════════════════
    # DEF-Konstante der 3-stufigen Chain (Variable muss VOR Loop gebaut sein,
    # damit sie im except-Fall noch referenzierbar ist): Jede Stufe ist ein eigener
    # Endpoint; bei fehlender Konfiguration (=Skip), non-200, Exception, oder
    # leerer Content wird die nächste Stufe versucht.
    review_chain = []
    if CLOUD_REVIEW_ENABLED and CLOUD_REVIEW_API_KEY:
        review_chain.append({
            "name": "KIMI",
            "model": CLOUD_REVIEW_MODEL,
            "api_key": CLOUD_REVIEW_API_KEY,
            "api_url": CLOUD_REVIEW_API_URL,
            "timeout": CLOUD_REVIEW_TIMEOUT_SECONDS,
        })
    if LITELLM_CLOUD_MODEL and LITELLM_CLOUD_API_KEY:
        review_chain.append({
            "name": "DEEPSEEK-V4-PRO",
            "model": LITELLM_CLOUD_MODEL,
            "api_key": LITELLM_CLOUD_API_KEY,
            "api_url": LITELLM_CLOUD_API_URL or "https://api.openai.com/v1/chat/completions",
            "timeout": LITELLM_CLOUD_TIMEOUT_SECONDS,
        })

    if not review_chain:
        _log("  ✗ Cloud-Reviewer: Kein Cloud-Modell verfügbar")
        return {
            "agent_key": "cloud_reviewer",
            "status": "skipped",
            "content": "",
            "duration_seconds": 0.0,
            "usage": None,
        }

    started_total = time.perf_counter()
    last_error = ""
    for stage_idx, stage in enumerate(review_chain):
        stage_name = stage["name"]
        model = stage["model"]
        api_key = stage["api_key"]
        api_url = stage["api_url"]
        timeout = stage["timeout"]
        stage_label = f"[{stage_idx+1}/{len(review_chain)} {stage_name}]"
        _log(f"  🔍 Cloud-Reviewer {stage_label}: model={model}")

        # Pro Stufe ein FRISCHES Payload-Objekt (loops müssenzu nicht mutiert werden).
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                task_msg,
                plan_msg,
                response_msg,  # ← einzige variable Message zwischen Feedback-Runden
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
            "stream": False,
        }
        _patch_moonshot_payload(payload)
        payload = _clean_payload(payload)
        headers = {"Authorization": f"Bearer {api_key}"}

        started = time.perf_counter()
        try:
            target_url = api_url or "https://api.openai.com/v1/chat/completions"
            async with httpx.AsyncClient(timeout=timeout) as lc:
                r = await lc.post(target_url, json=payload, headers=headers)
            duration = time.perf_counter() - started
            if r.status_code == 200:
                result = r.json()
                content = _extract_choice_content(result)
                if content and content.strip():
                    _log(f"  ✓ Cloud-Reviewer {stage_label} OK: model={model} duration={duration:.1f}s")
                    return {
                        "agent_key": "cloud_reviewer",
                        "status": "ok",
                        "content": content,
                        "duration_seconds": time.perf_counter() - started_total,
                        "usage": result.get("usage"),
                    }
                _log(f"  ⚠ Cloud-Reviewer {stage_label}: 200 aber content leer → nächste Stufe")
                last_error = f"empty content ({stage_name})"
            else:
                _log(f"  ⚠ Cloud-Reviewer {stage_label} STATUS {r.status_code} nach {duration:.1f}s → nächste Stufe")
                last_error = f"HTTP {r.status_code} ({stage_name}): {r.text[:200]}"
        except Exception as exc:
            duration = time.perf_counter() - started
            _log(f"  ✗ Cloud-Reviewer {stage_label} ERROR nach {duration:.1f}s: {exc}")
            last_error = f"{type(exc).__name__}: {exc} ({stage_name})"

    # Alle Stufen fehlgeschlagen
    _log(f"  ✗ Cloud-Reviewer: Alle Stufen fehlgeschlagen – letzter Fehler: {last_error}")
    return {
        "agent_key": "cloud_reviewer",
        "status": "failed",
        "content": "",
        "duration_seconds": time.perf_counter() - started_total,
        "usage": None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Lokal/Free Calls
# ═══════════════════════════════════════════════════════════════════════════

def _vllm_headers() -> Dict[str, str]:
    """Gibt Auth-Header für Lokal/Free-Anfragen zurück (leer falls kein Key gesetzt)."""
    if VLLM_API_KEY:
        return {"Authorization": f"Bearer {VLLM_API_KEY}"}
    return {}


def _patch_moonshot_payload(payload: Dict[str, Any]) -> None:
    """Erzwingt Moonshot-kompatible Parameter (nur spezifische Werte erlaubt).
    
    Prüft NUR das Model und VLLM_API_URL — nicht die Cloud/LiteLLM URLs,
    da sonst Moonshot-Fixes fälschlich auf DeepSeek-Calls angewendet werden."""
    model = str(payload.get("model", ""))
    model_lower = model.lower()
    vllm_url_lower = VLLM_API_URL.lower()
    
    is_moonshot = (
        "kimi" in model_lower or "moonshot" in model_lower or
        "moonshot" in vllm_url_lower or "kimi" in vllm_url_lower
    )
    
    if not is_moonshot:
        return
    
    fixes = []
    
    # Temperature: only 1.0 allowed
    if payload.get("temperature") is not None and payload["temperature"] != 1.0:
        fixes.append(f"temp {payload['temperature']}→1.0")
        payload["temperature"] = 1.0
    
    # top_p: only 0.95 allowed
    if payload.get("top_p") is not None and payload["top_p"] != 0.95:
        fixes.append(f"top_p {payload['top_p']}→0.95")
        payload["top_p"] = 0.95
    
    # top_k: Moonshot lehnt manchmal bestimmte Werte ab → entfernen
    if "top_k" in payload:
        fixes.append("top_k entfernt")
        del payload["top_k"]
    
    # presence_penalty / frequency_penalty: Moonshot lehnt != 0 ab
    for key in ("presence_penalty", "frequency_penalty"):
        val = payload.get(key)
        if val is not None and val != 0.0:
            fixes.append(f"{key} {val}→0")
            payload[key] = 0.0
    
    if fixes:
        _log(f"  🔧 Moonshot-Fixes: {'; '.join(fixes)}")


# ═══════════════════════════════════════════════════════════════════════════
# Fallback: Cloud als Worker-Ersatz bei Lokal/Free-Timeout
# ═══════════════════════════════════════════════════════════════════════════

async def _call_cloud_as_worker(
    payload: Dict[str, Any],
    agent_key: str,
) -> Dict[str, Any]:
    """
    Ruft die Cloud (LiteLLM/DeepSeek oder Cloud Reviewer) mit den ORIGINALEN
    User-Messages als Worker-Ersatz auf. Wird verwendet, wenn Lokal/Free
    timeoutet oder fehlschlägt.
    """
    started = time.perf_counter()
    model = payload.get("model", "?")
    _log(f"  ☁️☁️ CLOUD-FALLBACK agent_key={agent_key} original_model={model}")

    # Payload für Cloud vorbereiten: nur Messages + max_tokens, keine Tools
    cloud_payload = {
        "model": "",
        "messages": list(payload.get("messages", [])),
        "max_tokens": int(payload.get("max_tokens", DEFAULT_DIRECT_MAX_TOKENS)),
        "temperature": 0.3,
        "stream": False,
    }
    # reasoning_content komplett strippen (Kimi-Thinking crasht DeepSeek)
    removed_rc = _strip_kimi_reasoning(cloud_payload.get("messages", []))
    for m in cloud_payload.get("messages", []):
        if isinstance(m, dict) and "reasoning_content" in m:
            del m["reasoning_content"]
            removed_rc += 1
    if removed_rc:
        _log(f"  🧹 Cloud-Fallback: {removed_rc} reasoning_content-Felder entfernt")
    cloud_payload = _clean_payload(cloud_payload, keep_tools=False)

    # ── Versuch 1: LiteLLM (DeepSeek) ──────────────────────────────────
    if LITELLM_CLOUD_API_URL and LITELLM_CLOUD_API_KEY:
        cloud_payload["model"] = LITELLM_CLOUD_MODEL
        # image_url entfernen (DeepSeek text-only)
        if _is_text_only_model(LITELLM_CLOUD_MODEL):
            _sanitize_image_urls_inplace(cloud_payload.get("messages", []), "Cloud-Fallback-LiteLLM")
        _patch_moonshot_payload(cloud_payload)
        headers = {"Authorization": f"Bearer {LITELLM_CLOUD_API_KEY}"}
        _log(f"  ☁️ Fallback-Versuch 1: LiteLLM model={LITELLM_CLOUD_MODEL}")
        try:
            async with httpx.AsyncClient(timeout=LITELLM_CLOUD_TIMEOUT_SECONDS) as lc:
                r = await lc.post(LITELLM_CLOUD_API_URL, json=cloud_payload, headers=headers)
            duration = time.perf_counter() - started
            if r.status_code == 200:
                result = r.json()
                content = _extract_choice_content(result)
                _log(f"  ✓ Cloud-Fallback OK via LiteLLM duration={duration:.1f}s")
                return {
                    "agent_key": agent_key,
                    "status": "ok",
                    "content": content or "",
                    "duration_seconds": duration,
                    "usage": result.get("usage"),
                    "fallback": "liteilm",
                }
            _log(f"  ⚠ Cloud-Fallback LiteLLM STATUS {r.status_code}: duration={duration:.1f}s")
        except Exception as exc:
            _log(f"  ✗ Cloud-Fallback LiteLLM ERROR: {exc}")

    # ── Versuch 2: Cloud Reviewer (Moonshot) ───────────────────────────
    if CLOUD_REVIEW_ENABLED and CLOUD_REVIEW_API_KEY and CLOUD_REVIEW_API_URL:
        cloud_payload["model"] = CLOUD_REVIEW_MODEL
        _patch_moonshot_payload(cloud_payload)
        headers = {"Authorization": f"Bearer {CLOUD_REVIEW_API_KEY}"}
        _log(f"  ☁️ Fallback-Versuch 2: Cloud Reviewer model={CLOUD_REVIEW_MODEL}")
        try:
            async with httpx.AsyncClient(timeout=CLOUD_REVIEW_TIMEOUT_SECONDS) as cc:
                r = await cc.post(CLOUD_REVIEW_API_URL, json=cloud_payload, headers=headers)
            duration = time.perf_counter() - started
            if r.status_code == 200:
                result = r.json()
                content = _extract_choice_content(result)
                _log(f"  ✓ Cloud-Fallback OK via Cloud Reviewer duration={duration:.1f}s")
                return {
                    "agent_key": agent_key,
                    "status": "ok",
                    "content": content or "",
                    "duration_seconds": duration,
                    "usage": result.get("usage"),
                    "fallback": "cloud_reviewer",
                }
            _log(f"  ⚠ Cloud-Fallback Cloud Reviewer STATUS {r.status_code}: duration={duration:.1f}s")
        except Exception as exc:
            _log(f"  ✗ Cloud-Fallback Cloud Reviewer ERROR: {exc}")

    duration = time.perf_counter() - started
    _log(f"  ✗ Cloud-Fallback KOMPLETT FEHLGESCHLAGEN duration={duration:.1f}s")
    return {
        "agent_key": agent_key,
        "status": "error",
        "content": f"Lokal/Free + Cloud-Fallback fehlgeschlagen nach {duration:.0f}s",
        "duration_seconds": duration,
        "usage": None,
        "fallback": "failed",
    }


async def _call_vllm_with_fallback(
    client: httpx.AsyncClient,
    payload: Dict[str, Any],
    agent_key: str,
    timeout_seconds: float = SUB_AGENT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """
    Ruft Lokal/Free (vLLM/Spark) auf. Bei Timeout/Error → automatischer
    Fallback auf Cloud (LiteLLM → Cloud Reviewer).
    """
    result = await _call_vllm(client, payload, agent_key, timeout_seconds)

    if result.get("status") == "ok":
        return result

    # Timeout oder Error → Cloud-Fallback
    duration = result.get("duration_seconds", 0)
    status = result.get("status", "?")
    _log(f"  ⚠ Lokal/Free {status} nach {duration:.1f}s → Cloud-Fallback wird gestartet")
    _log(f"  └─ Original-Fehler: {result.get('content', '')[:200]}")

    cloud_result = await _call_cloud_as_worker(payload, agent_key)

    # Cloud-Fallback-Ergebnis mit einem deutlichen Hinweis versehen
    if cloud_result.get("status") == "ok":
        fb_type = cloud_result.get("fallback", "cloud")
        cloud_result["content"] = (
            f"[⚠️ Lokal/Free nach {duration:.0f}s nicht verfügbar – "
            f"Antwort via {fb_type}-Cloud-Fallback]\n\n"
            f"{cloud_result['content']}"
        )
        _log(f"  ✓ Fallback erfolgreich via {fb_type}")
    else:
        _log(f"  ✗ Auch Cloud-Fallback fehlgeschlagen")

    return cloud_result


async def _call_vllm(
    client: httpx.AsyncClient,
    payload: Dict[str, Any],
    agent_key: str,
    timeout_seconds: float = SUB_AGENT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    started = time.perf_counter()
    model = payload.get("model", "?")
    msg_count = len(payload.get("messages", []))
    total_chars = sum(len(str(m.get("content", ""))) for m in payload.get("messages", []))
    _log(f"  → Lokal/Free call agent_key={agent_key} model={model} "
         f"messages={msg_count} chars={total_chars} timeout={timeout_seconds:.0f}s")

    # DEBUG: Worker-Payload dumpen (für Remote-Inspection)
    worker_call_id = f"worker_{agent_key}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    _dump_debug_payload(worker_call_id, f"worker_{agent_key}", payload, extra={
        "agent_key": agent_key, "model": model,
        "timeout": timeout_seconds, "messages_count": msg_count,
    })
    _register_debug_request(worker_call_id, {
        "type": "worker_call_start",
        "agent_key": agent_key, "model": model,
        "messages_count": msg_count, "chars": total_chars,
        "timeout": timeout_seconds,
    })

    # Moonshot/Kimi-Temperatur-Korrektur (Moonshot erlaubt nur 1.0)
    _patch_moonshot_payload(payload)

    # Nur für Kimi-Continuation reasoning injecten, nicht für Worker
    messages = payload.get("messages", [])
    if isinstance(messages, list) and "kimi" in str(payload.get("model", "")).lower():
        _inject_reasoning_from_cache(messages)

    # Heartbeat-Task: alle 30s loggen, dass wir noch warten
    hb_task: Optional[asyncio.Task] = None

    async def _heartbeat():
        while True:
            await asyncio.sleep(30)
            elapsed = time.perf_counter() - started
            _log(f"  … ⏳ Lokal/Free noch am Warten agent_key={agent_key} "
                 f"elapsed={elapsed:.0f}s timeout={timeout_seconds:.0f}s")

    try:
        hb_task = asyncio.ensure_future(_heartbeat())
        # Kurz warten bis Heartbeat-Task wirklich läuft
        await asyncio.sleep(0.2)
    except Exception:
        pass  # Heartbeat ist nice-to-have

    try:
        response = await client.post(VLLM_API_URL, json=payload, headers=_vllm_headers(), timeout=timeout_seconds)
        duration = time.perf_counter() - started
        if response.status_code == 200:
            result = response.json()
            message = _extract_choice_message(result)
            tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
            reasoning_content = message.get("reasoning_content") if isinstance(message, dict) else None
            if tool_calls:
                _log(f"  🔧 Lokal/Free returned structured tool_calls: {len(tool_calls)}")
            if reasoning_content:
                _log(f"  🧠 Lokal/Free returned reasoning_content: {len(reasoning_content)} chars")
            # Debug: Zeige DeepSeeks Thinking + Tool-Calls
            _debug_log_thinking({
                "message": message, "content": _extract_choice_content(result),
                "tool_calls": tool_calls,
            }, agent_key)
            # reasoning_content für Folge-Requests cachen
            _cache_reasoning(tool_calls, reasoning_content)
            _log(f"  ✓ Lokal/Free OK agent_key={agent_key} duration={duration:.1f}s")
            return {
                "agent_key": agent_key,
                "status": "ok",
                "content": _extract_choice_content(result),
                "message": message,
                "tool_calls": tool_calls,
                "reasoning_content": reasoning_content,
                "duration_seconds": duration,
                "usage": result.get("usage"),
            }
        _log(f"  ⚠ Lokal/Free STATUS {response.status_code} agent_key={agent_key} duration={duration:.1f}s model={model}")
        # Extrahiere Fehlerdetails aus Response-Body
        err_detail = ""
        try:
            err_body = response.json()
            if isinstance(err_body.get("error"), dict):
                err_detail = err_body["error"].get("message", "")
            elif isinstance(err_body.get("error"), str):
                err_detail = err_body["error"]
            else:
                err_detail = str(err_body.get("message") or err_body.get("detail") or "")
        except Exception:
            err_detail = response.text[:200]
        if err_detail:
            _log(f"  ⚠ Fehlerdetails: {err_detail}")
        return {
            "agent_key": agent_key,
            "status": "failed",
            "content": f"Lokal/Free status {response.status_code}: {response.text[:500]}",
            "duration_seconds": duration,
            "usage": None,
        }
    except asyncio.CancelledError:
        duration = time.perf_counter() - started
        _log(f"  ✗ Lokal/Free CANCELLED agent_key={agent_key} duration={duration:.1f}s")
        return {
            "agent_key": agent_key,
            "status": "error",
            "content": f"Lokal/Free cancelled after {duration:.0f}s",
            "duration_seconds": duration,
            "usage": None,
        }
    except Exception as exc:
        duration = time.perf_counter() - started
        # Timeout erkennen und detailliert loggen
        exc_type = type(exc).__name__
        if hasattr(exc, "__cause__") and exc.__cause__:
            exc_type += f" → {type(exc.__cause__).__name__}"
        _log(f"  ✗ Lokal/Free ERROR agent_key={agent_key} duration={duration:.1f}s "
             f"type={exc_type}: {exc}")
        if "timeout" in str(exc).lower() or "timed out" in str(exc).lower():
            _log(f"  ⏰ TIMEOUT nach {duration:.0f}s (Limit: {timeout_seconds:.0f}s). "
                 f"vLLM/Spark braucht für den ersten Request oft 60-90s (DFlash JIT).")
        return {
            "agent_key": agent_key,
            "status": "error",
            "content": f"Lokal/Free error nach {duration:.0f}s ({exc_type}): {exc}",
            "duration_seconds": duration,
            "usage": None,
        }
    finally:
        # Heartbeat sauber beenden
        if hb_task is not None:
            hb_task.cancel()


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Verification & Self-Correction
# ═══════════════════════════════════════════════════════════════════════════

async def _run_lint_check(code: str) -> str:
    """Führt lokalen Linter aus (falls konfiguriert)."""
    if not VERIFY_LINT_COMMAND:
        return "[no lint command configured]"
    try:
        proc = await asyncio.create_subprocess_shell(
            VERIFY_LINT_COMMAND,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        return (stdout.decode() + stderr.decode())[:2000]
    except Exception as exc:
        return f"Lint error: {exc}"


async def _run_test_check(code: str) -> str:
    """Führt lokale Tests aus (falls konfiguriert)."""
    if not VERIFY_TEST_COMMAND:
        return "[no test command configured]"
    try:
        proc = await asyncio.create_subprocess_shell(
            VERIFY_TEST_COMMAND,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60.0)
        return (stdout.decode() + stderr.decode())[:2000]
    except Exception as exc:
        return f"Test error: {exc}"


async def _verify_and_correct(
    client: httpx.AsyncClient,
    original_task: str,
    worker_response: str,
) -> Tuple[str, Dict[str, Any]]:
    """Phase 3: Verifikation mit lokalem Linter/Test + optionalem Cloud-Sanity-Check."""
    if not VERIFY_ENABLED:
        return worker_response, {"stage": "verify_skipped", "status": "disabled"}

    lint_result = await _run_lint_check(worker_response)
    test_result = await _run_test_check(worker_response)
    combined_results = f"LINT:\n{lint_result}\n\nTEST:\n{test_result}"

    # Nur verifizieren wenn es tatsächlich Fehler gab
    has_errors = any(kw in combined_results.lower() for kw in ("error", "failed", "fail", "traceback"))
    if not has_errors or ("[no " in combined_results):
        return worker_response, {
            "stage": "verify_passed",
            "status": "ok",
            "lint": lint_result[:500],
            "test": test_result[:500],
        }

    # Fehler gefunden → Lokales Modell korrigiert
    verify_payload = _build_verify_payload(worker_response, combined_results, original_task)
    verify_result = await _call_vllm(client, verify_payload, "verifier", VERIFY_TIMEOUT_SECONDS)

    corrected = verify_result.get("content", worker_response) if verify_result.get("status") == "ok" else worker_response
    return corrected, {
        "stage": "verify_corrected",
        "status": verify_result.get("status"),
        "lint": lint_result[:500],
        "test": test_result[:500],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Agenten-Workflows
# ═══════════════════════════════════════════════════════════════════════════

async def _run_direct_local(
    body: Dict[str, Any],
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    """Direkter lokaler Request (kein Agent, kein Cloud-Planer)."""
    start_time = time.perf_counter()
    model = body.get("model") or MODEL_NAME
    messages_raw = body.get("messages", [])
    is_tool = _is_tool_continuation(messages_raw)
    _log(f"▶ DIREKT: model={model} messages={len(messages_raw)} tool_cont={is_tool}")
    progress: List[str] = [
        _format_chat_progress_message("intent", "Direkte lokale Anfrage. Keine Cloud-Planung.", {
            "intent": "direct", "model": model,
        }),
    ]

    payload = _build_direct_payload(body)
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient()

    try:
        assert client is not None
        result = await _call_vllm_with_fallback(client, payload, "direct")
    finally:
        if own_client and client is not None:
            await client.aclose()

    duration = time.perf_counter() - start_time
    status = result.get("status", "?")
    content = result.get("content", "")
    if status != "ok":
        _log(f"⚠ DIREKT FEHLER: model={model} status={status} duration={duration:.1f}s "
             f"content={content[:300]}")
        return {
            "combined_response_text": content,
            "results": [result],
            "duration_seconds": duration,
        }

    _log(f"✓ DIREKT OK: model={model} duration={duration:.1f}s "
         f"content_len={len(content)}")

    # Tool-Continuation + Model returned structured/DSML tool_calls → raw durchreichen
    if is_tool or result.get("tool_calls") or _contains_tool_calls(content):
        if _contains_tool_calls(content):
            _log(f"  🔧 DIREKT erkannte DSML-Tool-Calls in content ({len(content)} chars)")
        return {
            "combined_response_text": content,
            "results": [result],
            "duration_seconds": duration,
        }

    progress.append(_format_chat_progress_message("completed", "Direkte lokale Antwort fertig.", {
        "duration_seconds": duration,
        "status": status,
    }))

    return {
        "combined_response_text": "".join(progress) + result.get("content", ""),
        "results": [result],
        "duration_seconds": duration,
    }


async def _run_agent_workflow(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    3-Phasen-Agenten-Workflow mit Planner-First-Architektur:

    Bei -force planning:
      Phase 1: Cloud-Planner (Kimi K2.7) mit VOLLEN Tools exploriert Workspace
      Phase 2: Worker (DeepSeek) führt den Plan aus
      Phase 3: Cloud-Review (Kimi) prüft Ergebnis gegen Plan

    Bei normalem Agent-Intent (ohne -force planning):
      Phase 1: Hindsight → Cloud-Planer (Caveman, ohne Tools)
      Phase 2: Worker führt Plan aus
      Phase 3: Verifikation

    Pipeline-Steuerflags:
      -force planning  → Cloud-Planner als ersten Agent mit Tools
      -force review    → Cloud-Review nach Worker-Ausführung
      -bypass worker   → Worker überspringen, Cloud antwortet direkt
    """
    start_time = time.perf_counter()
    progress: List[str] = []
    results: List[Dict[str, Any]] = []

    query = _last_user_text(body.get("messages", []))
    _, pflags = _extract_pipeline_flags(query)
    force_planning = pflags.get("force_planning", False)
    force_review = pflags.get("force_review", False)
    bypass_worker = pflags.get("bypass_worker", False)

    # Flags aus Messages strippen
    _strip_pipeline_flags_from_messages(body.get("messages", []))
    query_clean = _last_user_text(body.get("messages", []))

    flag_summary = []
    if force_planning: flag_summary.append("force_planning")
    if force_review: flag_summary.append("force_review")
    if bypass_worker: flag_summary.append("bypass_worker")
    flag_str = ", ".join(flag_summary) if flag_summary else "none"

    session_hash = _get_planner_session_hash(body.get("messages", []))
    session = _PLANNER_SESSIONS.get(session_hash)
    _cleanup_old_planner_sessions()

    _log(f"▶ AGENT: worker={MODEL_NAME} cloud={CLOUD_REVIEW_MODEL if CLOUD_REVIEW_ENABLED else '–'} "
         f"litellm={LITELLM_CLOUD_MODEL or '–'} messages={len(body.get('messages',[]))} "
         f"flags=[{flag_str}] session_state={session.get('state') if session else 'none'}")
    client = httpx.AsyncClient()

    # ═══════════════════════════════════════════════════════════════════
    # PLANNER-FIRST: Session-Tracking für Cloud-Planner als Tool-Agent
    # ═══════════════════════════════════════════════════════════════════
    if session and session.get("state") == "active":
        # ── Tool-Continuation in aktiver Planner-Session ──────────────
        planner_flags = session.get("pflags", {})
        force_review = force_review or planner_flags.get("force_review", False)
        bypass_worker = bypass_worker or planner_flags.get("bypass_worker", False)

        # Loop-Detection: statt blinden Counter, letzte Tool-Requests tracken.
        # Die Tool-Resultate (vorige Runde) sind in der History (letzter 'tool'-Eintrag).
        # Wir extrahieren die vorigen tool_calls aus der Message-History.
        #
        # BUGFIX (Runde-2-Fehlalarm): _extract_last_assistant_tool_calls liefert
        # bei jeder Tool-Continuation DIESELBEN assistant-Tool-Calls zurück, die
        # schon bei Session-Creation als first_sigs gespeichert wurden. Ohne ID-
        # Deduplikation würde Runde 2 den Runde-1-Call ein zweites Mal appenden
        # → 'exact-repeat x2' → sofortiger Fehl-Loop-Alarm.
        # Lösung: seen_tool_call_ids pro Session tracken, nur NEUE Calls aufnehmen.
        prev_tool_calls = _extract_last_assistant_tool_calls(body.get("messages", []))
        seen_ids = session.setdefault("seen_tool_call_ids", set())
        if isinstance(seen_ids, list):  # defensive (JSON-Restore aus set)
            seen_ids = set(seen_ids)
            session["seen_tool_call_ids"] = seen_ids
        new_calls: List[Dict[str, Any]] = []
        for tc in prev_tool_calls:
            tc_id = (tc.get("id") if isinstance(tc, dict) else "") or ""
            if tc_id and tc_id in seen_ids:
                continue
            if tc_id:
                seen_ids.add(tc_id)
            new_calls.append(tc)
        # Bug-Malformed-Loop-Detection:
        # VORHER wurden invalide Tool-Calls (leere Args) komplett ignoriert.
        # Wenn Kimi aber Runde für Runde nur noch buggy Calls produziert (z.B.
        # parallele read_file-Batches ohne filePath), hatte jeder Team-Call einen
        # VS Code Error zur Folge → Kimi hat das ignoriert und neue invalide
        # Calls erzeugt → endloses Kreisen ohne jemals den Plan zu erzeugen.
        # LÖSUNG: Zähle consecutive_malformed_rounds. Eine Runde gilt als
        # "all-malformed" wenn ALLE neuen Tool-Calls in dieser Runde keine
        # sinnvollen Args haben. Sobald PLANNER_MALFORMED_HARD_STOP (3) erreicht
        # sind, triggert _detect_planner_loop und erzwingt Plan-Output.
        if not new_calls:
            # Keine neuen Tool-Calls in diesem Round (sehr selten - i.d.R.
            # wenn Kimi ohne tool_calls antwortet). Counter unverändert.
            malformed_round_counter = int(session.get("consecutive_malformed_rounds", 0))
        else:
            malformed_count = sum(1 for tc in new_calls if not _tool_call_has_args(tc))
            if malformed_count == len(new_calls):
                # Alle Calls in dieser Runde invalide → Counter hochzählen
                malformed_round_counter = int(session.get("consecutive_malformed_rounds", 0)) + 1
                session["consecutive_malformed_rounds"] = malformed_round_counter
                _log(f"  ⚠ Runde {session.get('iterations',0)+1}: {malformed_count}/{len(new_calls)} "
                     f"Tool-Calls ohne Args (consecutive_malformed={malformed_round_counter}/"
                     f"{PLANNER_MALFORMED_HARD_STOP})")
            else:
                # Zumindest ein Call hatte Args → Loop gebrochen
                if int(session.get("consecutive_malformed_rounds", 0)) > 0:
                    _log(f"  ✓ Malformed-Streak gebrochen "
                         f"({session.get('consecutive_malformed_rounds',0)} → 0)")
                session["consecutive_malformed_rounds"] = 0
                malformed_round_counter = 0

        if new_calls:
            sigs = session.setdefault("tool_signatures", [])
            for tc in new_calls:
                # Invalide Tool-Calls nicht in Signaturen tracken — sie haben
                # keine Args und erzeugen alle dieselbe Signatur, was die
                # 'exact-repeat'-Detection verfälschen würde. Stattdessen
                # oben über consecutive_malformed_rounds getrackt.
                if not _tool_call_has_args(tc):
                    continue
                sigs.append(_tool_signature(tc))
            # Fortschritts-Metrik: neue File-Refs hinzufügen
            refs = _extract_tool_file_refs(new_calls)
            files_set = session.setdefault("distinct_files", set())
            if isinstance(files_set, list):  # defensive (json can't serialize set)
                files_set = set(files_set)
                session["distinct_files"] = files_set
            for r in refs:
                files_set.add(r)

        # Exploration-only-Loop-Detection: Assistent-Content (von Kimi's
        # voriger Runde) mitschreiben, damit _detect_planner_loop '## plan'
        # erkennen kann. Wir nehmen den Content aus der LETZEN assistant-msg.
        last_assistant_content = ""
        for m in reversed(body.get("messages", [])):
            if isinstance(m, dict) and m.get("role") == "assistant":
                last_assistant_content = _message_text(m) or ""
                break
        if last_assistant_content:
            contents_list = session.setdefault("assistant_contents", [])
            contents_list.append(last_assistant_content[:400])
            # nur letzte 6 Inhalte halten
            if len(contents_list) > 6:
                del contents_list[:len(contents_list) - 6]

        session["iterations"] = int(session.get("iterations", 0)) + 1
        sig_count = len(session.get("tool_signatures") or [])
        distinct_files = len(session.get("distinct_files") or [])
        _log(f"  🔧 Planner-Runde {session['iterations']} (distinct tools={sig_count}, "
             f"explored files/queries={distinct_files})")

        planner_result = await _call_cloud_planner_agent(client, body)
        results.append(planner_result)

        # ══ Plan-Erkennung: Kimi hat fertig, auch wenn noch tool_calls (memory) anstehen ══
        pc = planner_result.get("content", "")
        if _content_contains_plan(pc) and len(pc) > 200:
            _log(f"  🎉 Plan im Content erkannt ({len(pc)} chars) → Session done, Plan persistiert")
            plan_path = _save_plan_to_file(session_hash, pc, query=body.get("messages", [{}])[0].get("content", ""))
            _PLANNER_SESSIONS[session_hash] = {
                "state": "done", "ts": time.time(),
                "plan": pc, "pflags": planner_flags,
                "plan_path": str(plan_path) if plan_path else None,
            }

        if planner_result.get("tool_calls"):
            # Kimi will weitere Tools → Tool-Calls an VS Code weiterleiten
            _log("  🔧 Cloud-Planner-Agent fordert weitere Tools → Durchreiche an VS Code")
            await client.aclose()
            return {
                "combined_response_text": planner_result.get("content", ""),
                "results": [planner_result],
                "duration_seconds": time.perf_counter() - start_time,
            }

        # Kimi hat fertig → Plan extrahieren
        plan = pc if planner_result.get("status") == "ok" else ""
        if plan:
            _log(f"  📝 Cloud-Planner hat Plan erstellt ({len(plan)} chars)")
            # Plan in dak-dat File persistieren (Codespace-Copilot-Stil)
            plan_path = _save_plan_to_file(session_hash, plan, query=body.get("messages", [{}])[0].get("content", ""))
            _PLANNER_SESSIONS[session_hash] = {
                "state": "done", "ts": time.time(),
                "plan": plan, "pflags": planner_flags,
                "plan_path": str(plan_path) if plan_path else None,
            }
            progress.append(_format_chat_progress_message(
                "phase1_plan_ready",
                f"🧠 Cloud-Planner (Kimi): Plan erstellt ({len(plan)} chars).",
                {"duration_seconds": planner_result.get("duration_seconds")},
            ))

            # Jetzt Worker mit Plan ausführen
            progress.append(_format_chat_progress_message(
                "phase2_execute",
                f"🛠️ Worker ({MODEL_NAME}) führt Plan aus.",
                {"model": MODEL_NAME},
            ))

            worker_payload = _build_worker_payload(
                body, plan, "",
                plan_path=_PLANNER_SESSIONS.get(session_hash, {}).get("plan_path"),
            )
            worker_result = await _call_vllm_with_fallback(client, worker_payload, "worker")
            results.append(worker_result)
            worker_response = worker_result.get("content", "")

            if worker_result.get("tool_calls") or _contains_tool_calls(worker_response):
                _log("  🔧 Worker enthält Tool-Calls → Durchreiche an Client")
                await client.aclose()
                # Session NICHT löschen – "done" bleibt erhalten, damit der
                # nächste Request (tool_cont) den Plan-Kontext wiederfindet
                # und der Worker weiterarbeiten kann. Sonst startet VS Code
                # eine brandneue Planner-Session ohne Plan-Kontext.
                _hindsight.retain(body, worker_response)
                return {
                    "combined_response_text": worker_response,
                    "results": results,
                    "duration_seconds": time.perf_counter() - start_time,
                }

            progress.append(_format_chat_progress_message(
                "phase2_done", "Worker-Ausführung abgeschlossen.",
                {"duration_seconds": worker_result.get("duration_seconds")},
            ))

            # Phase 3: Review mit Feedback-Loop (nur im Planner-Session-Pfad)
            verified_response = await _review_with_feedback_loop(
                client, body, plan, query_clean, worker_response,
                force_review, progress, results,
                session_hash=session_hash,
            )

            del _PLANNER_SESSIONS[session_hash]
            await client.aclose()

            final = verified_response
            if CHATTY_MODE:
                plan_preview = plan[:2000] + ("\n...[truncated]" if len(plan) > 2000 else "")
                final = (
                    f"## 🧠 Cloud-Planner Plan (Kimi K2.7)\n\n{plan_preview}\n\n"
                    f"---\n\n## 🛠️ Worker Ausführung ({MODEL_NAME})\n\n{verified_response}"
                )
            combined = "".join(progress) + final
            _hindsight.retain(body, combined)
            return {
                "combined_response_text": combined,
                "results": results,
                "duration_seconds": time.perf_counter() - start_time,
            }
        else:
            # Planner fehlgeschlagen → Fallback
            _log("  ⚠ Cloud-Planner-Agent fehlgeschlagen → Fallback auf Worker direkt")
            del _PLANNER_SESSIONS[session_hash]
            worker_payload = _build_direct_payload(body)
            worker_result = await _call_vllm_with_fallback(client, worker_payload, "worker")
            results.append(worker_result)
            await client.aclose()
            combined = "".join(progress) + worker_result.get("content", "")
            _hindsight.retain(body, combined)
            return {
                "combined_response_text": combined,
                "results": results,
                "duration_seconds": time.perf_counter() - start_time,
            }

    elif session and session.get("state") == "done":
        # ── Plan ready, Worker soll ausführen (nächste User-Nachricht ohne Tool-Cont) ──
        plan = session.get("plan", "")
        planner_flags = session.get("pflags", {})
        force_review = force_review or planner_flags.get("force_review", False)

        if plan:
            # Falls kein Plan-Path in der alten Session stand, jetzt speichern
            plan_path = session.get("plan_path")
            if plan and not plan_path:
                plan_path_obj = _save_plan_to_file(
                    session_hash, plan,
                    query=body.get("messages", [{}])[0].get("content", ""),
                )
                plan_path = str(plan_path_obj) if plan_path_obj else None
                session["plan_path"] = plan_path
            worker_payload = _build_worker_payload(body, plan, "", plan_path=plan_path)
            worker_result = await _call_vllm_with_fallback(client, worker_payload, "worker")
        else:
            worker_payload = _build_direct_payload(body)
            worker_result = await _call_vllm_with_fallback(client, worker_payload, "worker")

        results.append(worker_result)
        worker_response = worker_result.get("content", "")

        # Wenn der Worker Tool-Calls zurückgibt, Session behalten, damit
        # der nächste Request (tool_cont) den Plan-Kontext wiederfindet.
        if not (worker_result.get("tool_calls") or _contains_tool_calls(worker_response)):
            del _PLANNER_SESSIONS[session_hash]
        await client.aclose()
        _hindsight.retain(body, worker_response)
        return {
            "combined_response_text": worker_response,
            "results": results,
            "duration_seconds": time.perf_counter() - start_time,
        }

    # ═══════════════════════════════════════════════════════════════════
    # STANDARD / FORCE-PLANNING: Erster Request
    # ═══════════════════════════════════════════════════════════════════

    # ── Hindsight Recall ──────────────────────────────────────────────
    memory_records = _hindsight.recall(query_clean)
    memory_context = _hindsight.format_context(memory_records)
    progress.append(_format_chat_progress_message(
        "phase1_recall",
        f"Hindsight Recall: {len(memory_records)} relevante Erinnerungen geladen.",
        {"memory_records": len(memory_records), "networks": list(set(n for r in memory_records for n in r.networks))},
    ))

    if force_planning and CLOUD_REVIEW_ENABLED and CLOUD_REVIEW_API_KEY:
        # ── PLANNER-FIRST: Kimi als Tool-Agent, dann Worker ──────────
        _log("  🧠 Cloud-Planner-Agent (Kimi K2.7 mit Tools) exploriert Workspace...")
        progress.append(_format_chat_progress_message(
            "phase1_planner_start",
            "🧠 Cloud-Planner (Kimi K2.7) exploriert Workspace und erstellt Plan.",
            {"model": CLOUD_REVIEW_MODEL},
        ))

        # ══ PHASE 0: User-Prompt mit Fast-Modell komprimieren ══════════
        # User-Vision: CHEAP model compresses prompt BEFORE teure Cloud.
        if CAVEMAN_ENABLED and len(query_clean) > 80:
            progress.append(_format_chat_progress_message(
                "phase0_compress", "🗜️ Phase 0: User-Prompt wird komprimiert.",
                {"fast_model": FAST_MODEL_NAME, "original_chars": len(query_clean)},
            ))
            compressed = await _phase0_compress_prompt(client, query_clean)
            # Komprimierte Version an letzte User-Message in body anhaengen
            if compressed != query_clean:
                msgs = body.get("messages", [])
                for msg in reversed(msgs):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        orig = msg.get("content", "")
                        if isinstance(orig, str):
                            msg["content"] = f"{orig}\n\n[CAVEMAN-PROMPT-COMPRESSED]\n{compressed}"
                        break

        planner_result = await _call_cloud_planner_agent(client, body, "", results)  # Kein Hindsight im Planner — sauberer Plan-Kontext
        results.append(planner_result)

        # ══ Plan-Erkennung: Kimi hat fertig, auch wenn noch tool_calls (memory) anstehen ══
        pc = planner_result.get("content", "")
        plan_detected = _content_contains_plan(pc) and len(pc) > 200
        if plan_detected:
            _log(f"  🎉 Plan im Content erkannt ({len(pc)} chars) → Session done, Plan persistiert")
            plan_path = _save_plan_to_file(session_hash, pc, query=body.get("messages", [{}])[0].get("content", ""))
            _PLANNER_SESSIONS[session_hash] = {
                "state": "done", "ts": time.time(),
                "plan": pc, "pflags": {"force_review": force_review, "bypass_worker": bypass_worker},
                "plan_path": str(plan_path) if plan_path else None,
            }

        if planner_result.get("tool_calls") and not plan_detected:
            # Kimi will Tools ausführen → an VS Code weiterleiten, Session speichern.
            # ABER NUR wenn KEIN Plan erkannt wurde (sonst ist session schon "done")
            # Initialisiere Loop-Detection-State für diese Session.
            first_calls = planner_result.get("tool_calls", []) or []
            first_sigs = [_tool_signature(tc) for tc in first_calls]
            first_refs = _extract_tool_file_refs(first_calls)
            first_ids = {tc.get("id", "") for tc in first_calls
                         if isinstance(tc, dict) and tc.get("id")}
            _log(f"  🔧 Cloud-Planner will {len(first_calls)} Tools → Session starten "
                 f"(initial signatures: {len(first_sigs)}, ids: {len(first_ids)})")
            _PLANNER_SESSIONS[session_hash] = {
                "state": "active", "ts": time.time(), "iterations": 1,
                "tool_signatures": first_sigs,
                "distinct_files": set(first_refs),
                "seen_tool_call_ids": first_ids,
                "pflags": {"force_review": force_review, "bypass_worker": bypass_worker},
            }
            await client.aclose()
            return {
                "combined_response_text": planner_result.get("content", ""),
                "results": [planner_result],
                "duration_seconds": time.perf_counter() - start_time,
            }

        # Kimi hat direkt einen Plan geliefert (keine Tools nötig)
        plan = planner_result.get("content", "") if planner_result.get("status") == "ok" else ""
        if plan:
            _log(f"  📝 Cloud-Planner direkt: Plan ({len(plan)} chars)")
            # Plan als Datei persistieren
            plan_path = _save_plan_to_file(session_hash, plan, query=body.get("messages", [{}])[0].get("content", ""))
            _PLANNER_SESSIONS[session_hash] = {
                "state": "done", "ts": time.time(),
                "plan": plan, "pflags": planner_flags,
                "plan_path": str(plan_path) if plan_path else None,
            }
            progress.append(_format_chat_progress_message(
                "phase1_plan_ready",
                f"🧠 Cloud-Planner: Plan in {planner_result.get('duration_seconds',0):.1f}s erstellt.",
                {"duration_seconds": planner_result.get("duration_seconds")},
            ))
        else:
            _log("  ⚠ Cloud-Planner lieferte keinen Plan → Fallback")
            worker_payload = _build_direct_payload(body)
            worker_result = await _call_vllm_with_fallback(client, worker_payload, "worker")
            results.append(worker_result)
            await client.aclose()
            combined = "".join(progress) + worker_result.get("content", "")
            _hindsight.retain(body, combined)
            return {
                "combined_response_text": combined,
                "results": results,
                "duration_seconds": time.perf_counter() - start_time,
            }
    else:
        # ── Klassischer Cloud-Planer (Caveman, ohne Tools) ───────────
        planner_result = await _call_cloud_planner(client, body, memory_context)
        results.append(planner_result)
        plan_status = planner_result.get("status")
        plan = planner_result.get("content", "") if plan_status == "ok" else ""

        if plan_status == "ok":
            plan_path = _save_plan_to_file(session_hash, plan, query=body.get("messages", [{}])[0].get("content", ""))
            _PLANNER_SESSIONS[session_hash] = {
                "state": "done", "ts": time.time(),
                "plan": plan,
                "plan_path": str(plan_path) if plan_path else None,
            }
            progress.append(_format_chat_progress_message(
                "phase1_plan_ready",
                f"Cloud-Planer: Caveman-Plan erstellt.",
                {"duration_seconds": planner_result.get("duration_seconds")},
            ))
        else:
            progress.append(_format_chat_progress_message(
                "phase1_plan_fallback",
                f"Cloud-Planer nicht verfügbar ({plan_status}). Fallback auf lokale Ausführung.",
                {"status": plan_status},
            ))
            worker_payload = _build_direct_payload(body)
            worker_result = await _call_vllm_with_fallback(client, worker_payload, "worker")
            results.append(worker_result)
            await client.aclose()
            combined = "".join(progress) + worker_result.get("content", "")
            _hindsight.retain(body, combined)
            return {
                "combined_response_text": combined,
                "results": results,
                "duration_seconds": time.perf_counter() - start_time,
            }

    # ── Phase 2: Worker (oder Bypass) ────────────────────────────────
    if bypass_worker:
        _log("  ⚡ -bypass worker: Worker übersprungen, Cloud antwortet direkt")
        progress.append(_format_chat_progress_message(
            "phase2_bypass",
            f"⚡ Worker übersprungen. Cloud ({LITELLM_CLOUD_MODEL or CLOUD_REVIEW_MODEL}) antwortet direkt.",
            {"bypass": True},
        ))
        cloud_response = await _call_cloud_as_responder(client, body, plan, memory_context)
        results.append(cloud_response)
        worker_response = cloud_response.get("content", "")
        progress.append(_format_chat_progress_message(
            "phase2_done", "Cloud-Direktantwort abgeschlossen.",
            {"duration_seconds": cloud_response.get("duration_seconds")},
        ))
    else:
        progress.append(_format_chat_progress_message(
            "phase2_execute",
            f"Worker ({MODEL_NAME}) führt Plan aus.",
            {"model": MODEL_NAME},
        ))
        worker_payload = _build_worker_payload(
            body, plan, memory_context,
            plan_path=_PLANNER_SESSIONS.get(session_hash, {}).get("plan_path"),
        )
        worker_result = await _call_vllm_with_fallback(client, worker_payload, "worker")
        results.append(worker_result)
        worker_response = worker_result.get("content", "")

        if worker_result.get("tool_calls") or _contains_tool_calls(worker_response):
            _log("  🔧 Worker enthält Tool-Calls → Durchreiche an Client")
            await client.aclose()
            _hindsight.retain(body, worker_response)
            return {
                "combined_response_text": worker_response,
                "results": results,
                "duration_seconds": time.perf_counter() - start_time,
            }

        progress.append(_format_chat_progress_message(
            "phase2_done", "Worker-Ausführung abgeschlossen.",
            {"duration_seconds": worker_result.get("duration_seconds")},
        ))

    # ── Phase 3: Review mit Feedback-Loop ────────────────────────────
    if force_review and not bypass_worker:
        verified_response = await _review_with_feedback_loop(
            client, body, plan, query_clean, worker_response,
            force_review, progress, results,
            session_hash=session_hash,
        )
    else:
        verified_response, _ = await _verify_and_correct(client, query_clean, worker_response)

    await client.aclose()
    _hindsight.retain(body, verified_response)

    combined = "".join(progress) + verified_response
    return {
        "combined_response_text": combined,
        "results": results,
        "duration_seconds": time.perf_counter() - start_time,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Streaming Helpers
# ═══════════════════════════════════════════════════════════════════════════

async def _stream_chat_completion(body: Dict[str, Any]) -> Dict[str, Any]:
    """Non-streaming: Führt Workflow aus und gibt kombiniertes Ergebnis zurück."""
    intent = _classify_intent(body.get("messages", []))
    if intent == "agent":
        return await _run_agent_workflow(body)
    return await _run_direct_local(body)


async def _stream_events(request: Request, body: Dict[str, Any]) -> AsyncIterator[str]:
    """
    SSE-Streaming: Sendet OpenAI-kompatible Chunks.
    - Wenn der Worker/Direct structured tool_calls returned: korrektes
      Streaming-Format mit tool_calls-Deltas + finish_reason='tool_calls'.
    - Sonst: normaler Text-Content + finish_reason='stop'.
    """
    model = body.get("model", MODEL_NAME)
    intent = _classify_intent(body.get("messages", []))

    if intent == "agent":
        streamed = await _run_agent_workflow(body)
    else:
        streamed = await _run_direct_local(body)

    # Pipeline-Summary bauen
    has_cloud = any(r.get("agent_key") == "cloud_planner" for r in streamed.get("results", []))
    has_worker = any(r.get("agent_key") == "worker" for r in streamed.get("results", []))
    is_tool_cont = any(r.get("tool_calls") for r in streamed.get("results", []))
    if is_tool_cont:
        s_intent = "direct"
    elif has_cloud or has_worker:
        s_intent = "agent"
    else:
        s_intent = "direct"
    pipeline_summary = _build_pipeline_summary(streamed.get("results", []), s_intent)

    # Strukturierte Tool-Calls + Reasoning aus den Ergebnissen extrahieren
    tool_calls = None
    reasoning_content = None
    for r in reversed(streamed.get("results", [])):
        tc = r.get("tool_calls")
        if tc:
            tool_calls = tc
        rc = r.get("reasoning_content")
        if rc and not reasoning_content:
            reasoning_content = rc
        if tool_calls and reasoning_content:
            break

    # Tool-Calls normalisieren (arguments als JSON-String sicherstellen)
    # DeepSeek liefert arguments manchmal als Dict statt JSON-String,
    # was VS Code beim Parsen der SSE-Chunks nicht verarbeiten kann
    # → "must have required property 'filePath'" obwohl filePath da ist.
    if tool_calls:
        normalized = _normalize_tool_calls(tool_calls)
        if normalized:
            tool_calls = normalized
            _log(f"  🔄 Worker-Tool-Calls normalisiert ({len(tool_calls)} calls)")

    # Prüfen auf DSML-Tool-Calls im Text (auch ohne structured tool_calls)
    text_content = streamed.get("combined_response_text", "")
    has_dsml = _contains_tool_calls(text_content) and not tool_calls
    
    if has_dsml:
        _log(f"  ⚠ Streaming: DSML in content erkannt, ohne structured tool_calls – "
             f"Content wird pur an Client durchgereicht (keine Summary)")
    
    if tool_calls:
        # ── STRUKTURIERTE TOOL_CALLS STREAMING (OpenAI-Format) ──
        # Bei Tool-Calls KEINE Summary anhängen (würde Format brechen)
        stream_id = f"chatcmpl-spark-{uuid.uuid4().hex}"

        # ROBUST STREAMING PATTERN (FIX für VS Code MCP-Client Bug):
        # VS Code's OpenAI-compat-Client akkumuliert arguments-deltas nicht
        # zuverlässig über separate Chunks hinweg. Argumente gingen in R2
        # verloren → VS Code executed tools mit leeren args →
        # "must have required property 'filePath'" Errors → infinite Loop.
        #
        # FIX: Kompletten tool_call (ID + Name + Args) in EINEM Chunk
        # senden. Header-empty-args + separate args-delta vermeiden.

        first_tcs = []
        for i, tc in enumerate(tool_calls):
            func = tc.get("function", {})
            args = func.get("arguments", "")
            # Args sicher als String (defensive)
            if not isinstance(args, str):
                try:
                    args = json.dumps(args, ensure_ascii=False)
                except Exception:
                    args = str(args)
            first_tcs.append({
                "index": i,
                "id": tc.get("id", f"call_{uuid.uuid4().hex}"),
                "type": "function",
                "function": {
                    "name": func.get("name", ""),
                    "arguments": args,   # VOLLSTÄNDIGE Args direkt im Header!
                },
            })
        # 1. (einziger) Tool-Call-Chunk: role + content=null + tool_calls[full]
        #    reasoning_content im selben Chunk (falls vorhanden)
        yield _format_openai_stream_chunk(
            model, include_role=True, tool_calls=first_tcs,
            reasoning_content=reasoning_content, chunk_id=stream_id,
        )

        # 2. Finaler Chunk mit finish_reason='tool_calls' (leeres delta)
        yield _format_openai_stream_chunk(model, finish_reason="tool_calls", chunk_id=stream_id)
    else:
        # ── NORMALES TEXT-STREAMING ──
        # Pipeline-Summary anhängen (nur wenn weder tool_calls noch DSML)
        if has_dsml:
            text = streamed["combined_response_text"]
        else:
            text = streamed["combined_response_text"].rstrip() + pipeline_summary
        # Reasoning-Content im ersten Chunk mitsenden
        yield _format_openai_stream_chunk(model, content=text, include_role=True, reasoning_content=reasoning_content)
        yield _format_openai_stream_chunk(model, "", finish_reason="stop")


# ═══════════════════════════════════════════════════════════════════════════
# MCP-Server (Model Context Protocol) für VS Code Tool-Zugriff
# ═══════════════════════════════════════════════════════════════════════════

MCP_TOOLS = [
    {
        "name": "localproxy_read_file",
        "description": "Liest eine Datei aus dem lokalen Workspace.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relativer oder absoluter Pfad zur Datei."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "localproxy_list_files",
        "description": "Listet Dateien in einem Verzeichnis auf.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Verzeichnispfad."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "localproxy_run_terminal",
        "description": "Führt einen Terminal-Befehl aus und gibt stdout/stderr zurück.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Auszuführender Shell-Befehl."},
                "cwd": {"type": "string", "description": "Arbeitsverzeichnis (optional)."},
            },
            "required": ["command"],
        },
    },
    {
        "name": "localproxy_search_code",
        "description": "Durchsucht den Workspace nach einem Regex-Pattern.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex-Pattern."},
                "path": {"type": "string", "description": "Suchpfad (optional, default: Workspace-Root)."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "localproxy_write_file",
        "description": "Schreibt Inhalt in eine Datei im Workspace.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Zieldatei-Pfad."},
                "content": {"type": "string", "description": "Dateiinhalt."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "localproxy_hindsight_recall",
        "description": "Ruft relevante Erinnerungen aus dem Hindsight-Gedächtnis ab.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Suchanfrage für Memory-Recall."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "localproxy_get_status",
        "description": "Gibt Status-Informationen des Proxys zurück (Modelle, Speicher, Konfiguration).",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


async def _handle_mcp_tool_call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Führt einen MCP-Tool-Call aus."""
    try:
        if tool_name == "localproxy_read_file":
            path = Path(arguments["path"])
            if not path.is_absolute():
                path = Path.cwd() / path
            if not path.exists():
                return {"content": [{"type": "text", "text": f"File not found: {path}"}], "isError": True}
            content = path.read_text(encoding="utf-8")
            return {"content": [{"type": "text", "text": content}]}

        elif tool_name == "localproxy_list_files":
            path = Path(arguments["path"])
            if not path.is_absolute():
                path = Path.cwd() / path
            if not path.is_dir():
                return {"content": [{"type": "text", "text": f"Not a directory: {path}"}], "isError": True}
            items = []
            for item in sorted(path.iterdir()):
                suffix = "/" if item.is_dir() else ""
                items.append(f"{item.name}{suffix}")
            return {"content": [{"type": "text", "text": "\n".join(items)}]}

        elif tool_name == "localproxy_run_terminal":
            cmd = arguments["command"]
            cwd = arguments.get("cwd", str(Path.cwd()))
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            except asyncio.TimeoutError:
                proc.kill()
                return {"content": [{"type": "text", "text": "Command timed out after 30s"}], "isError": True}
            output = stdout.decode() + stderr.decode()
            return {"content": [{"type": "text", "text": output[:10000]}]}

        elif tool_name == "localproxy_search_code":
            pattern = arguments["pattern"]
            search_path = Path(arguments.get("path", "."))
            if not search_path.is_absolute():
                search_path = Path.cwd() / search_path
            results = []
            try:
                for file_path in search_path.rglob("*.py"):
                    if file_path.is_file():
                        content = file_path.read_text(encoding="utf-8")
                        for i, line in enumerate(content.splitlines(), 1):
                            if re.search(pattern, line):
                                results.append(f"{file_path}:{i}: {line.strip()[:200]}")
                return {"content": [{"type": "text", "text": "\n".join(results[:50]) or "No matches"}]}
            except Exception as exc:
                return {"content": [{"type": "text", "text": f"Search error: {exc}"}], "isError": True}

        elif tool_name == "localproxy_write_file":
            path = Path(arguments["path"])
            if not path.is_absolute():
                path = Path.cwd() / path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(arguments["content"], encoding="utf-8")
            return {"content": [{"type": "text", "text": f"Written to {path}"}]}

        elif tool_name == "localproxy_hindsight_recall":
            query = arguments["query"]
            records = _hindsight.recall(query)
            context = _hindsight.format_context(records)
            return {"content": [{"type": "text", "text": context or "No relevant memories found."}]}

        elif tool_name == "localproxy_get_status":
            status = {
                "models": {"fast": FAST_MODEL_NAME, "worker": MODEL_NAME},
                "vllm_url": VLLM_API_URL,
                "cloud_enabled": CLOUD_REVIEW_ENABLED,
                "cloud_model": CLOUD_REVIEW_MODEL,
                "caveman_enabled": CAVEMAN_ENABLED,
                "hindsight_enabled": HINDSIGHT_ENABLED,
                "verify_enabled": VERIFY_ENABLED,
            }
            return {"content": [{"type": "text", "text": json.dumps(status, indent=2)}]}

        else:
            return {"content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}], "isError": True}

    except Exception as exc:
        return {"content": [{"type": "text", "text": f"Tool error: {exc}"}], "isError": True}


# ═══════════════════════════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════════════════════════


app = FastAPI(
    title="DX Spark Hybrid Agentic Proxy",
    version="2.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
)


@app.on_event("startup")
async def _startup_event() -> None:
    """Log startup configuration und führt Health-Checks für alle Modelle aus."""
    _log(f"🚀 LocalProxy starting on port {PROXY_PORT}")
    _log(f"   Worker:  {MODEL_NAME}")
    _log(f"   Fast:    {FAST_MODEL_NAME}")
    _log(f"   Lokal/Free: {VLLM_API_URL}")
    _log(f"   Lokal/Free Key: {'✓ gesetzt' if VLLM_API_KEY else '–'}")
    _log(f"   Cloud:   {'enabled' if CLOUD_REVIEW_ENABLED else 'disabled'}")
    _log(f"   LiteLLM: {'enabled' if LITELLM_CLOUD_MODEL else 'disabled'} ({LITELLM_CLOUD_MODEL or '–'})")
    _log(f"   Caveman: {'enabled' if CAVEMAN_ENABLED else 'disabled'}")
    _log(f"   Memory:  {'Qdrant' if _hindsight._use_qdrant else 'JSONL' if HINDSIGHT_ENABLED else 'disabled'}")
    _log(f"   Verify:  {'enabled' if VERIFY_ENABLED else 'disabled'}")
    _log(f"   MCP:     {'enabled' if MCP_ENABLED else 'disabled'}")

    # ── Health-Checks für alle Modell-Endpoints ──────────────────────────
    _log("🔍 Health-Checks werden gestartet...")
    async with httpx.AsyncClient(timeout=10.0) as hc:

        def _parse_error(r):
            """Versucht aus einer Error-Response JSON den message-String zu extrahieren."""
            try:
                body = r.json()
                # OpenAI-kompatibel
                if isinstance(body.get("error"), dict):
                    return body["error"].get("message", "")
                # Anthropic-Format
                if isinstance(body.get("error"), str):
                    return body["error"]
                # Zencoder / andere
                return str(body.get("message") or body.get("detail") or "")
            except Exception:
                return r.text[:200] if r.text else ""

        # 1. Worker-Modell – /v1/models testen (robuster, vermeidet JIT-Timeout)
        _models_url = _derive_models_url(VLLM_API_URL)
        _log(f"   🔍 Worker '{MODEL_NAME}' @ {VLLM_API_URL} ...")
        _log(f"      Models-URL: {_models_url}")
        try:
            r = await hc.get(
                _models_url,
                headers=_vllm_headers(),
                timeout=httpx.Timeout(30.0, connect=5.0),
            )
            if r.status_code == 200:
                data = r.json()
                ids = [m.get("id","?") for m in data.get("data",[])]
                if MODEL_NAME in ids:
                    _log(f"   ✅ Worker: OK ({MODEL_NAME} gelistet, {len(ids)} total)")
                else:
                    _log(f"   ⚠️  Worker: Models geladen ({len(ids)}), aber '{MODEL_NAME}' nicht in Liste – {', '.join(ids[:8])}")
            elif r.status_code == 404:
                _log(f"   ❌ Worker: 404 – {_models_url} nicht erreichbar")
            else:
                _log(f"   ❌ Worker: HTTP {r.status_code}")
        except Exception as exc:
            _log(f"   ❌ Worker: NICHT ERREICHBAR – {type(exc).__name__}: {exc}")

        # 1b. Fast-Modell separat testen falls abweichend
        if FAST_MODEL_NAME != MODEL_NAME:
            _log(f"   🔍 Fast '{FAST_MODEL_NAME}' @ {VLLM_API_URL} ...")
            try:
                r = await hc.post(
                    VLLM_API_URL,
                    json={"model": FAST_MODEL_NAME, "messages": [{"role":"user","content":"ping"}], "max_tokens":1},
                    headers=_vllm_headers(),
                )
                if r.status_code == 200:
                    _log(f"   ✅ Fast-Modell: OK ({FAST_MODEL_NAME})")
                elif r.status_code in (401,403):
                    err = _parse_error(r) or "AUTH-DENIED"
                    _log(f"   ❌ Fast-Modell: AUTH-FEHLER {r.status_code} – {err}")
                elif r.status_code == 404:
                    err = _parse_error(r) or "Modell nicht gefunden"
                    _log(f"   ❌ Fast-Modell: 404 – {err} ({FAST_MODEL_NAME})")
                else:
                    err = _parse_error(r) or f"HTTP {r.status_code}"
                    _log(f"   ❌ Fast-Modell: {err} ({FAST_MODEL_NAME})")
            except Exception as exc:
                _log(f"   ❌ Fast-Modell: NICHT ERREICHBAR – {type(exc).__name__}: {exc}")

        # 1c. Models-Liste (nice-to-have, viele Cloud-Proxys haben keinen /models-Endpoint)
        _list_url = _derive_models_url(VLLM_API_URL)
        _log(f"   🔍 Models-Liste via {_list_url} ...")
        try:
            r = await hc.get(_list_url, headers=_vllm_headers())
            if r.status_code == 200:
                data = r.json()
                ids = [m.get("id","?") for m in data.get("data",[])]
                _log(f"   ✅ Models-Liste: OK ({len(ids)} Modelle: {', '.join(ids[:5])}{'...' if len(ids)>5 else ''})")
            else:
                _log(f"   ℹ️  Models-Liste: STATUS {r.status_code} (viele Cloud-Proxys haben keinen /models-Endpoint)")
        except Exception as exc:
            _log(f"   ℹ️  Models-Liste: nicht erreichbar – {exc} (viele Cloud-Proxys haben keinen /models-Endpoint)")

        # 2. Cloud Reviewer
        if CLOUD_REVIEW_ENABLED and CLOUD_REVIEW_API_KEY:
            _log(f"   🔍 Cloud '{CLOUD_REVIEW_MODEL}' @ {CLOUD_REVIEW_API_URL} ...")
            try:
                r = await hc.post(
                    CLOUD_REVIEW_API_URL,
                    json={"model": CLOUD_REVIEW_MODEL, "messages": [{"role":"user","content":"ping"}], "max_tokens":1},
                    headers={"Authorization": f"Bearer {CLOUD_REVIEW_API_KEY}"},
                )
                if r.status_code in (200,201):
                    _log(f"   ✅ Cloud Reviewer: OK ({CLOUD_REVIEW_MODEL})")
                elif r.status_code in (401,403):
                    err = _parse_error(r) or "AUTH-DENIED"
                    _log(f"   ❌ Cloud Reviewer: AUTH-FEHLER {r.status_code} – {err}")
                elif r.status_code == 404:
                    err = _parse_error(r) or "Modell nicht gefunden"
                    _log(f"   ❌ Cloud Reviewer: 404 – {err} ({CLOUD_REVIEW_MODEL})")
                else:
                    err = _parse_error(r) or f"HTTP {r.status_code}"
                    _log(f"   ❌ Cloud Reviewer: {err} ({CLOUD_REVIEW_MODEL})")
            except Exception as exc:
                _log(f"   ❌ Cloud Reviewer: NICHT ERREICHBAR – {exc}")
        elif CLOUD_REVIEW_ENABLED:
            _log(f"   ⚠️  Cloud Reviewer: aktiviert aber KEIN API-KEY")

        # 3. LiteLLM
        if LITELLM_CLOUD_MODEL and LITELLM_CLOUD_API_KEY:
            lite_url = LITELLM_CLOUD_API_URL or "https://api.openai.com/v1/chat/completions"
            _log(f"   🔍 LiteLLM '{LITELLM_CLOUD_MODEL}' @ {lite_url} ...")
            try:
                r = await hc.post(
                    lite_url,
                    json={"model": LITELLM_CLOUD_MODEL, "messages": [{"role":"user","content":"ping"}], "max_tokens":1},
                    headers={"Authorization": f"Bearer {LITELLM_CLOUD_API_KEY}"},
                )
                if r.status_code in (200,201):
                    _log(f"   ✅ LiteLLM: OK ({LITELLM_CLOUD_MODEL})")
                elif r.status_code in (401,403):
                    err = _parse_error(r) or "AUTH-DENIED"
                    _log(f"   ❌ LiteLLM: AUTH-FEHLER {r.status_code} – {err}")
                elif r.status_code == 404:
                    err = _parse_error(r) or "Modell nicht gefunden"
                    _log(f"   ❌ LiteLLM: 404 – {err} ({LITELLM_CLOUD_MODEL})")
                else:
                    err = _parse_error(r) or f"HTTP {r.status_code}"
                    _log(f"   ❌ LiteLLM: {err} ({LITELLM_CLOUD_MODEL})")
            except Exception as exc:
                _log(f"   ❌ LiteLLM: NICHT ERREICHBAR – {exc}")
        elif LITELLM_CLOUD_MODEL:
            _log(f"   ⚠️  LiteLLM: Modell gesetzt aber KEIN API-KEY")
    _log("🔍 Health-Checks abgeschlossen")


@app.on_event("shutdown")
async def _shutdown_event() -> None:
    _log("👋 LocalProxy shutting down.")


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
    if not _is_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Bearer"})


# ── /v1/chat/completions ──────────────────────────────────────────────────

_COPILOT_REQUEST_COUNTER = 0
_COPILOT_DUMP_MAX = 5


def _dump_copilot_request(request: Request, body: Dict[str, Any]) -> None:
    """Speichert die ersten N Copilot-Requests als JSON für Analyse.

    Nutzt data/debug/ (existiert schon, gitignored).
    Damit analysieren wir was Copilot im Plan-Mode wirklich sendet.
    """
    global _COPILOT_REQUEST_COUNTER
    _COPILOT_REQUEST_COUNTER += 1
    if _COPILOT_REQUEST_COUNTER > _COPILOT_DUMP_MAX:
        return
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    filename = f"copilot_request_{_COPILOT_REQUEST_COUNTER:03d}.json"
    snapshot = {
        "dumped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "request_number": _COPILOT_REQUEST_COUNTER,
        "url": str(request.url),
        "method": request.method,
        "headers": dict(request.headers),
        "query_params": dict(request.query_params),
        "body": body,
    }
    try:
        (DEBUG_DIR / filename).write_text(
            json.dumps(snapshot, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        _log(f"📸 Copilot-Request #{_COPILOT_REQUEST_COUNTER} gespeichert → data/debug/{filename}")
    except Exception as e:
        _log(f"⚠️ Copilot-Request-Dump fehlgeschlagen: {e}")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    await _auth_or_raise(request)
    body = await request.json()

    if "messages" not in body:
        raise HTTPException(status_code=400, detail="Invalid payload: 'messages' required.")

    # Dump für Plan-Mode-Analyse
    _dump_copilot_request(request, body)

    # Ignoriere Client-Modell – der Proxy verwendet sein konfiguriertes Modell
    body["model"] = MODEL_NAME

    msgs = body.get("messages", [])
    _log(f"📨 Request: {len(msgs)} messages, stream={body.get('stream')}, tool_cont={_is_tool_continuation(msgs)}")
    _debug_log_request(body)

    if body.get("stream"):
        return StreamingResponse(
            _stream_events(request, body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    streamed = await _stream_chat_completion(body)
    response_payload = _build_response_payload(body, streamed["combined_response_text"], streamed["results"])
    return JSONResponse(content=response_payload)


# ── /v1/models ─────────────────────────────────────────────────────────────
@app.get("/v1/models")
async def list_models(request: Request):
    # /v1/models?logs=N → Logs zurückgeben (ohne Auth, für Debugging)
    logs_str = request.query_params.get("logs", "")
    if logs_str and logs_str.isdigit() and int(logs_str) > 0:
        return JSONResponse(content=await _get_logs_handler(lines=int(logs_str)))
    await _auth_or_raise(request)
    models = [
        {"id": MODEL_NAME, "object": "model", "owned_by": "local-free"},
        {"id": FAST_MODEL_NAME, "object": "model", "owned_by": "local-free"},
    ]
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(_derive_models_url(VLLM_API_URL), headers=_vllm_headers())
            if response.status_code == 200:
                data = response.json()
                for m in data.get("data", []):
                    if m.get("id") not in {x.get("id") for x in models}:
                        models.append(m)
    except Exception:
        pass
    return JSONResponse(content={"object": "list", "data": models})


# ── /healthz ───────────────────────────────────────────────────────────────
@app.get("/healthz")
async def healthz(request: Request):
    return JSONResponse(content={
        "status": "ok",
        "version": "2.0.0",
        "routes": ["direct-local", "hybrid-agent"],
        "models": {"fast": FAST_MODEL_NAME, "worker": MODEL_NAME},
        "local_free_api_key_configured": bool(VLLM_API_KEY),
        "proxy_auth_enabled": PROXY_AUTH_ENABLED,
        "cloud_review_enabled": CLOUD_REVIEW_ENABLED,
        "cloud_review_model": CLOUD_REVIEW_MODEL if CLOUD_REVIEW_ENABLED else None,
        "litellm_model": LITELLM_CLOUD_MODEL or None,
        "litellm_api_url": LITELLM_CLOUD_API_URL or None,
        "litellm_max_tokens": LITELLM_CLOUD_MAX_TOKENS,
        "hindsight_enabled": HINDSIGHT_ENABLED,
        "hindsight_backend": "qdrant" if _hindsight._use_qdrant else "jsonl",
        "caveman_enabled": CAVEMAN_ENABLED,
        "verify_enabled": VERIFY_ENABLED,
        "mcp_enabled": MCP_ENABLED,
        "plans_dir": str(PLANS_DIR),
    })


# ── /plans ─────────────────────────────────────────────────────────────────
# Codespace-Copilot-Stil: Pläne liegen als .md-Files vor und können hier
# gelistet und abgerufen werden (für Debugging, WebUI-Anzeige, Downloads).

@app.get("/plans")
async def list_plans(request: Request):
    """Listet alle gespeicherten Plan-Dateien auf."""
    try:
        files = []
        if PLANS_DIR.exists():
            for entry in sorted(PLANS_DIR.glob("Plan_*.md"), reverse=True):
                stat = entry.stat()
                files.append({
                    "name": entry.name,
                    "path": str(entry),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                })
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    return JSONResponse(content={"count": len(files), "plans_dir": str(PLANS_DIR), "plans": files})


@app.get("/plans/{plan_name}")
async def get_plan(plan_name: str, request: Request):
    """Liefert den Inhalt einer konkreten Plan-Datei."""
    # Sicherheits-Check: nur Dateinamen, keine Pfad-Traversale
    if "/" in plan_name or "\\" in plan_name or ".." in plan_name:
        return JSONResponse(status_code=400, content={"error": "invalid plan_name"})
    if not plan_name.startswith("Plan_") or not plan_name.endswith(".md"):
        return JSONResponse(status_code=400, content={"error": "invalid plan_name pattern"})
    plan_path = PLANS_DIR / plan_name
    if not plan_path.exists():
        return JSONResponse(status_code=404, content={"error": "plan not found"})
    try:
        content = plan_path.read_text(encoding="utf-8")
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    return JSONResponse(content={"name": plan_name, "path": str(plan_path), "content": content})


# ── /logs (auch als /v1/logs für Coolify nginx) ────────────────────────────
async def _get_logs_handler(lines: int = 200):
    """Shared handler: Gibt die letzten Log-Zeilen zurück."""
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
    except (FileNotFoundError, OSError):
        all_lines = []
    last = all_lines[-lines:] if lines > 0 else all_lines
    return {
        "count": len(last),
        "total": len(all_lines),
        "file": LOG_FILE,
        "lines": last,
    }

@app.get("/logs")
async def get_logs(request: Request, lines: int = 200):
    """Ohne Auth (WebUI-Aufruf ohne Bearer-Token)."""
    return JSONResponse(content=await _get_logs_handler(lines))

@app.get("/v1/logs")
async def get_v1_logs(request: Request, lines: int = 200):
    """Coolify-nginx-kompatibler /v1/logs Endpoint."""
    return JSONResponse(content=await _get_logs_handler(lines))


# ── /debug/* ───────────────────────────────────────────────────────────────
# Drei Rexource-Kategorien:
#   * /debug/sessions  → aktive + letzte _PLANNER_SESSIONS (Iterations, hashes, last_tool_calls)
#   * /debug/active    → aktuell laufende _ACTIVE_CALLS (für "hängt seit X"-Diagnose)
#   * /debug/ring      → in-memory ring buffer der letzten N Requests
#   * /debug/files     → listet alle dump-JSON-Files in data/debug/
#   * /debug/file/<id> → konkreten Dump abrufen
#   * /debug/cleanup   → räumt alte Files auf
@app.get("/debug/sessions")
async def debug_sessions(request: Request, limit: int = 20):
    """In-Memory Planner-Sessions (iterations, hashes, seen_tool_calls)."""
    sessions = []
    # _PLANNER_SESSIONS ist globales dict aus dem agent block
    try:
        items = list(_PLANNER_SESSIONS.items())[-limit:] if limit > 0 else list(_PLANNER_SESSIONS.items())
        for sh, data in items:
            entry = {"session_hash": sh}
            if isinstance(data, dict):
                entry["iterations"] = data.get("iterations", 0)
                entry["last_tool_calls"] = list(data.get("seen_tool_call_ids", []))[-10:]
                entry["plan_path"] = data.get("plan_path")
                entry["last_request_len"] = len(str(data.get("last_request_hash", "")))
            sessions.append(entry)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    return JSONResponse(content={
        "count": len(sessions),
        "total_sessions": len(_PLANNER_SESSIONS),
        "sessions": sessions,
    })


@app.get("/debug/active")
async def debug_active(request: Request):
    """Aktuell laufende Calls (cloud planner, worker). 'Kimi hängt'-Diagnose."""
    now = time.time()
    active = []
    for call_id, info in _ACTIVE_CALLS.items():
        entry = {"call_id": call_id, "elapsed_seconds": now - info.get("started_at", now)}
        for k in ("agent_key", "model", "phase", "stage_label", "url", "timeout", "started_iso"):
            if k in info:
                entry[k] = info[k]
        active.append(entry)
    return JSONResponse(content={
        "count": len(active),
        "active_calls": active,
        "ring_size": len(_ACTIVE_CALLS),
    })


@app.get("/debug/ring")
async def debug_ring(request: Request, limit: int = 20):
    """Letzte N In-Memory Request-Events (compact summary)."""
    items = _DEBUG_RING[-limit:] if limit > 0 else list(_DEBUG_RING)
    return JSONResponse(content={
        "count": len(items),
        "ring_capacity": _DEBUG_RING_MAX,
        "ring_total": len(_DEBUG_RING),
        "items": items,
    })


@app.get("/debug/files")
async def debug_files(request: Request, limit: int = 50):
    """Listet Debug-Dump-Files in data/debug/ auf."""
    try:
        files = []
        if DEBUG_DIR.exists():
            for entry in sorted(DEBUG_DIR.glob("*.json"), reverse=True)[:limit]:
                stat = entry.stat()
                files.append({
                    "name": entry.name,
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "modified_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                })
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    return JSONResponse(content={
        "count": len(files),
        "debug_dir": str(DEBUG_DIR),
        "config": {
            "DEBUG_ENABLED": DEBUG_ENABLED,
            "DEBUG_MAX_FILES": DEBUG_MAX_FILES,
            "DEBUG_RING_MAX": _DEBUG_RING_MAX,
        },
        "files": files,
    })


@app.get("/debug/file/{file_id}")
async def debug_file(file_id: str, request: Request):
    """Liefert den Inhalt eines Debug-Dumps (file_id ohne .json-Ext)."""
    safe = "".join(c for c in file_id if c.isalnum() or c in "-_")[:100]
    if not safe or ".." in file_id or "/" in file_id or "\\" in file_id:
        return JSONResponse(status_code=400, content={"error": "invalid file_id"})
    target = DEBUG_DIR / f"{safe}.json"
    if not target.exists():
        return JSONResponse(status_code=404, content={"error": "file not found", "path": str(target)})
    try:
        content = target.read_text(encoding="utf-8")
        data = json.loads(content) if content.strip().startswith("{") else {"raw": content}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    return JSONResponse(content={"file_id": safe, "path": str(target), "data": data})


@app.post("/debug/cleanup")
async def debug_cleanup(request: Request):
    """Räumt alte Debug-Files auf (behält DEBUG_MAX_FILES)."""
    before = len(list(DEBUG_DIR.glob("*.json"))) if DEBUG_DIR.exists() else 0
    _cleanup_old_debug_files()
    after = len(list(DEBUG_DIR.glob("*.json"))) if DEBUG_DIR.exists() else 0
    return JSONResponse(content={
        "before": before, "after": after, "removed": before - after,
        "DEBUG_MAX_FILES": DEBUG_MAX_FILES,
    })


@app.post("/debug/copilot-dump-reset")
async def debug_copilot_dump_reset(request: Request):
    """Setzt den Copilot-Request-Dump-Counter zurück (neue 5 Dumps)."""
    global _COPILOT_REQUEST_COUNTER
    _COPILOT_REQUEST_COUNTER = 0
    _log("📸 Copilot-Request-Dump-Counter zurückgesetzt")
    return JSONResponse(content={"status": "ok", "message": "Counter reset — next 5 requests werden gedumpt"})


# ── MCP Endpoint ───────────────────────────────────────────────────────────
@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """Model Context Protocol JSON-RPC Endpoint für VS Code."""
    if not MCP_ENABLED:
        raise HTTPException(status_code=404, detail="MCP not enabled")

    body = await request.json()
    method = body.get("method", "")
    req_id = body.get("id")

    # initialize
    if method == "initialize":
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "localproxy-mcp", "version": "2.0.0"},
            },
        })

    # tools/list
    if method == "tools/list":
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": MCP_TOOLS},
        })

    # tools/call
    if method == "tools/call":
        params = body.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        result = await _handle_mcp_tool_call(tool_name, arguments)
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result,
        })

    # notifications/initialized
    if method == "notifications/initialized":
        return JSONResponse(content={"jsonrpc": "2.0", "id": req_id, "result": {}})

    return JSONResponse(content={
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    })


# ── Webinterface mounten ───────────────────────────────────────────────────
if _WEBUI_AVAILABLE:
    mount_webui(app)
    _log("🌐 Web-Konfigurationsinterface: http://0.0.0.0:" + str(PROXY_PORT) + "/webui/")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    _log(f"""
╔══════════════════════════════════════════════════════════════╗
║          DX Spark Hybrid Agentic Proxy  v2.0.0               ║
║  OpenAI-kompatibel · Hindsight · Caveman · 3-Phasen-Agent   ║
╚══════════════════════════════════════════════════════════════╝
    """.strip())
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)
