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
VLLM_MODELS_URL: str = os.getenv("VLLM_MODELS_URL", "http://localhost:8000/v1/models")
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
# Config-Datei-Loader (überschreibt Env-Variablen, falls config.json existiert)
# ═══════════════════════════════════════════════════════════════════════════

def _apply_config_file() -> None:
    """Lädt config.json und überschreibt Modul-Variablen (niedrigere Priorität als Env)."""
    if not _WEBUI_AVAILABLE:
        return
    try:
        cfg = _webui_load_config()
    except Exception:
        return

    def _env_or(key: str, cfg_val: Any) -> Any:
        """Env hat Vorrang vor config.json."""
        return cfg_val  # Env wurde bereits in den Modul-Konstanten gelesen

    # Models
    global VLLM_API_URL, VLLM_MODELS_URL, VLLM_API_KEY, MODEL_NAME, FAST_MODEL_NAME
    if not os.getenv("VLLM_API_URL"):
        VLLM_API_URL = cfg.get("models", {}).get("vllm_api_url", VLLM_API_URL)
    if not os.getenv("VLLM_MODELS_URL"):
        VLLM_MODELS_URL = cfg.get("models", {}).get("vllm_models_url", VLLM_MODELS_URL)
    # API-Keys: config.json überschreibt Env — WebUI-Speicherung soll immer wirken
    ak = cfg.get("models", {}).get("vllm_api_key", "")
    if ak:
        VLLM_API_KEY = ak
    if not os.getenv("MODEL_NAME"):
        MODEL_NAME = cfg.get("models", {}).get("model_name", MODEL_NAME)
    if not os.getenv("FAST_MODEL_NAME"):
        FAST_MODEL_NAME = cfg.get("models", {}).get("fast_model_name", FAST_MODEL_NAME)

    # Proxy
    global PROXY_PORT, PROXY_AUTH_ENABLED, PROXY_API_KEY, CHATTY_MODE
    if not os.getenv("PROXY_PORT"):
        PROXY_PORT = cfg.get("proxy", {}).get("port", PROXY_PORT)
    if not os.getenv("PROXY_AUTH_ENABLED"):
        PROXY_AUTH_ENABLED = cfg.get("proxy", {}).get("auth_enabled", PROXY_AUTH_ENABLED)
    # Proxy-Auth: config.json überschreibt Env (WebUI-Save wirkt)
    pk = cfg.get("proxy", {}).get("api_key", "")
    if pk:
        PROXY_API_KEY = pk
    if not os.getenv("CHATTY_MODE"):
        CHATTY_MODE = cfg.get("proxy", {}).get("chatty_mode", CHATTY_MODE)

    # Cloud
    global CLOUD_REVIEW_ENABLED, CLOUD_REVIEW_API_URL, CLOUD_REVIEW_API_KEY
    global CLOUD_REVIEW_MODEL, CLOUD_REVIEW_MAX_TOKENS, CLOUD_REVIEW_TIMEOUT_SECONDS
    if not os.getenv("CLOUD_REVIEW_ENABLED"):
        CLOUD_REVIEW_ENABLED = cfg.get("cloud", {}).get("enabled", CLOUD_REVIEW_ENABLED)
    if not os.getenv("CLOUD_REVIEW_API_URL"):
        CLOUD_REVIEW_API_URL = cfg.get("cloud", {}).get("api_url", CLOUD_REVIEW_API_URL)
    # Cloud API-Key: config.json überschreibt Env (WebUI-Save wirkt)
    ck = cfg.get("cloud", {}).get("api_key", "")
    if ck:
        CLOUD_REVIEW_API_KEY = ck
    if not os.getenv("CLOUD_REVIEW_MODEL"):
        CLOUD_REVIEW_MODEL = cfg.get("cloud", {}).get("model", CLOUD_REVIEW_MODEL)
    if not os.getenv("CLOUD_REVIEW_MAX_TOKENS"):
        CLOUD_REVIEW_MAX_TOKENS = cfg.get("cloud", {}).get("max_tokens", CLOUD_REVIEW_MAX_TOKENS)
    if not os.getenv("CLOUD_REVIEW_TIMEOUT_SECONDS"):
        CLOUD_REVIEW_TIMEOUT_SECONDS = cfg.get("cloud", {}).get("timeout_seconds", CLOUD_REVIEW_TIMEOUT_SECONDS)

    # LiteLLM
    global LITELLM_CLOUD_MODEL, LITELLM_CLOUD_API_KEY, LITELLM_CLOUD_API_URL
    global LITELLM_CLOUD_MAX_TOKENS, LITELLM_CLOUD_TIMEOUT_SECONDS
    if not os.getenv("LITELLM_CLOUD_MODEL"):
        LITELLM_CLOUD_MODEL = cfg.get("litellm", {}).get("model", LITELLM_CLOUD_MODEL)
    # LiteLLM API-Key: config.json überschreibt Env (WebUI-Save wirkt)
    lk = cfg.get("litellm", {}).get("api_key", "")
    if lk:
        LITELLM_CLOUD_API_KEY = lk
    if not os.getenv("LITELLM_CLOUD_API_URL"):
        LITELLM_CLOUD_API_URL = cfg.get("litellm", {}).get("api_url", LITELLM_CLOUD_API_URL)
    if not os.getenv("LITELLM_CLOUD_MAX_TOKENS"):
        LITELLM_CLOUD_MAX_TOKENS = cfg.get("litellm", {}).get("max_tokens", LITELLM_CLOUD_MAX_TOKENS)
    if not os.getenv("LITELLM_CLOUD_TIMEOUT_SECONDS"):
        LITELLM_CLOUD_TIMEOUT_SECONDS = cfg.get("litellm", {}).get("timeout_seconds", LITELLM_CLOUD_TIMEOUT_SECONDS)

    # Tokens
    global DEFAULT_DIRECT_MAX_TOKENS, DEFAULT_AGENT_MAX_TOKENS
    global SUB_AGENT_TIMEOUT_SECONDS, VERIFY_TIMEOUT_SECONDS
    if not os.getenv("DIRECT_MAX_TOKENS"):
        DEFAULT_DIRECT_MAX_TOKENS = cfg.get("tokens", {}).get("direct_max_tokens", DEFAULT_DIRECT_MAX_TOKENS)
    if not os.getenv("SUB_AGENT_MAX_TOKENS"):
        DEFAULT_AGENT_MAX_TOKENS = cfg.get("tokens", {}).get("agent_max_tokens", DEFAULT_AGENT_MAX_TOKENS)
    if not os.getenv("SUB_AGENT_TIMEOUT_SECONDS"):
        raw = cfg.get("tokens", {}).get("sub_agent_timeout_seconds", SUB_AGENT_TIMEOUT_SECONDS)
        SUB_AGENT_TIMEOUT_SECONDS = max(float(raw), 120.0)  # mind. 120s wg. DFlash JIT
        _log(f"  ⏱ SUB_AGENT_TIMEOUT_SECONDS = {SUB_AGENT_TIMEOUT_SECONDS:.0f}s (cfg raw={raw})")
    if not os.getenv("VERIFY_TIMEOUT_SECONDS"):
        VERIFY_TIMEOUT_SECONDS = cfg.get("tokens", {}).get("verify_timeout_seconds", VERIFY_TIMEOUT_SECONDS)

    # Caveman
    global CAVEMAN_ENABLED, CAVEMAN_MAX_TOKENS
    if not os.getenv("CAVEMAN_ENABLED"):
        CAVEMAN_ENABLED = cfg.get("caveman", {}).get("enabled", CAVEMAN_ENABLED)
    if not os.getenv("CAVEMAN_MAX_TOKENS"):
        CAVEMAN_MAX_TOKENS = cfg.get("tokens", {}).get("caveman_max_tokens", CAVEMAN_MAX_TOKENS)

    # Hindsight
    global HINDSIGHT_ENABLED, QDRANT_URL, QDRANT_API_KEY, HINDSIGHT_COLLECTION
    global HINDSIGHT_EMBEDDING_DIM, HINDSIGHT_MAX_MEMORY_TOKENS, HINDSIGHT_MIN_SIMILARITY
    global HINDSIGHT_RETAIN_DELAY_SECONDS, HINDSIGHT_USE_QDRANT, HINDSIGHT_DIR
    if not os.getenv("HINDSIGHT_ENABLED"):
        HINDSIGHT_ENABLED = cfg.get("hindsight", {}).get("enabled", HINDSIGHT_ENABLED)
    if not os.getenv("QDRANT_URL"):
        QDRANT_URL = cfg.get("hindsight", {}).get("qdrant_url", QDRANT_URL)
    # Qdrant API-Key: config.json überschreibt Env (WebUI-Save wirkt)
    qk = cfg.get("hindsight", {}).get("qdrant_api_key", "")
    if qk:
        QDRANT_API_KEY = qk
    if not os.getenv("HINDSIGHT_COLLECTION"):
        HINDSIGHT_COLLECTION = cfg.get("hindsight", {}).get("collection", HINDSIGHT_COLLECTION)
    if not os.getenv("HINDSIGHT_EMBEDDING_DIM"):
        val = cfg.get("hindsight", {}).get("embedding_dim", HINDSIGHT_EMBEDDING_DIM)
        HINDSIGHT_EMBEDDING_DIM = int(val) if not isinstance(val, int) else val
    if not os.getenv("HINDSIGHT_MAX_MEMORY_TOKENS"):
        val = cfg.get("hindsight", {}).get("max_memory_tokens", HINDSIGHT_MAX_MEMORY_TOKENS)
        HINDSIGHT_MAX_MEMORY_TOKENS = int(val) if not isinstance(val, int) else val
    if not os.getenv("HINDSIGHT_MIN_SIMILARITY"):
        val = cfg.get("hindsight", {}).get("min_similarity", HINDSIGHT_MIN_SIMILARITY)
        HINDSIGHT_MIN_SIMILARITY = float(val) if not isinstance(val, (int, float)) else val
    if not os.getenv("HINDSIGHT_RETAIN_DELAY_SECONDS"):
        val = cfg.get("hindsight", {}).get("retain_delay_seconds", HINDSIGHT_RETAIN_DELAY_SECONDS)
        HINDSIGHT_RETAIN_DELAY_SECONDS = float(val) if not isinstance(val, (int, float)) else val
    if not os.getenv("HINDSIGHT_USE_QDRANT"):
        HINDSIGHT_USE_QDRANT = cfg.get("hindsight", {}).get("use_qdrant", HINDSIGHT_USE_QDRANT)
    if not os.getenv("HINDSIGHT_DIR"):
        HINDSIGHT_DIR = Path(cfg.get("hindsight", {}).get("dir", str(HINDSIGHT_DIR)))

    # Verify
    global VERIFY_ENABLED, VERIFY_LINT_COMMAND, VERIFY_TEST_COMMAND
    if not os.getenv("VERIFY_ENABLED"):
        VERIFY_ENABLED = cfg.get("verify", {}).get("enabled", VERIFY_ENABLED)
    if not os.getenv("VERIFY_LINT_COMMAND"):
        VERIFY_LINT_COMMAND = cfg.get("verify", {}).get("lint_command", VERIFY_LINT_COMMAND)
    if not os.getenv("VERIFY_TEST_COMMAND"):
        VERIFY_TEST_COMMAND = cfg.get("verify", {}).get("test_command", VERIFY_TEST_COMMAND)

    # MCP
    global MCP_ENABLED
    if not os.getenv("MCP_ENABLED"):
        MCP_ENABLED = cfg.get("mcp", {}).get("enabled", MCP_ENABLED)


_apply_config_file()

# ── URL-Normalisierung ──────────────────────────────────────────────────
_api_base_clean = VLLM_API_URL.rstrip("/")
if not _api_base_clean.endswith("/chat/completions"):
    if _api_base_clean.endswith("/v1"):
        VLLM_API_URL = _api_base_clean + "/chat/completions"
    else:
        VLLM_API_URL = _api_base_clean + "/chat/completions"
    _log(f"🔧 URL-Norm: VLLM_API_URL → {VLLM_API_URL}")

_models_base_clean = VLLM_MODELS_URL.rstrip("/")
# Models-URL muss ein gültiges Protokoll haben — sonst aus API-URL ableiten
if not _models_base_clean.startswith(("http://", "https://")):
    _api_for_models = VLLM_API_URL.rstrip("/")
    if "/chat/completions" in _api_for_models:
        VLLM_MODELS_URL = _api_for_models.rsplit("/chat/completions", 1)[0] + "/models"
    elif _api_for_models.endswith("/v1"):
        VLLM_MODELS_URL = _api_for_models + "/models"
    else:
        VLLM_MODELS_URL = _api_for_models + "/v1/models"
    _log(f"🔧 URL-Norm: VLLM_MODELS_URL aus API-URL → {VLLM_MODELS_URL}")
elif not _models_base_clean.endswith("/models") and not _models_base_clean.endswith("/v1/models"):
    VLLM_MODELS_URL = _models_base_clean + "/models"
    _log(f"🔧 URL-Norm: VLLM_MODELS_URL → {VLLM_MODELS_URL}")

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
    return messages


def _last_user_text(messages: Sequence[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return _message_text(msg)
    return ""


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
    return message if isinstance(message, dict) else {}


def _extract_choice_content(result: Dict[str, Any]) -> str:
    message = _extract_choice_message(result)
    content = message.get("content", "")
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
    """Erkennt ob dieser Request eine Tool-Ausführungs-Fortsetzung ist."""
    for msg in messages:
        if msg.get("role") in ("tool", "function"):
            return True
        # Cline/Roo senden tool_result als eigenen role-Typ oder als Content
        if msg.get("role") == "user" and isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
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


def _classify_intent(messages: Sequence[Dict[str, Any]]) -> str:
    """Intent-Klassifizierung: deterministisch, bei Mehrdeutigkeit 'agent'."""
    # Tool-Continuation → immer direkt (kein neuer Plan!)
    if _is_tool_continuation(messages):
        _log("→ Intent: direct (tool-continuation, Pipeline übersprungen)")
        return "direct"

    text = _last_user_text(messages)

    # ── Pipeline-Steuerflags prüfen ──────────────────────────────────
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
    return _clean_payload(payload, keep_tools=True)


def _build_worker_payload(
    body: Dict[str, Any],
    plan: str,
    memory_context: str,
    model_name: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Baut den Payload für den WORKER – ein FULLY TOOL-CAPABLE Sub-Agent.
    Der Worker bekommt die KOMPLETTE VS Code Umgebung:
      - Originale System-Message (mit Tool-Definitionen & Calling-Format)
      - Alle Messages (kein Compact!)
      - tools / tool_choice aus dem Original-Body
      - Plan + Memory als ZUSÄTZLICHER Kontext in der letzten User-Message
    Keine redundante Worker-Instruktion – die VS Code System-Prompt steuert das Verhalten.
    """
    payload = copy.deepcopy(body)
    payload["model"] = model_name or body.get("model") or MODEL_NAME
    payload["max_tokens"] = int(max_tokens or payload.get("max_tokens", DEFAULT_AGENT_MAX_TOKENS))
    payload["stream"] = False

    messages = list(payload.get("messages", []))

    # Nur Kontext ANHÄNGEN, ohne die originale Struktur zu beschädigen
    context_blocks = []
    if memory_context:
        context_blocks.append(f"[HINDSIGHT MEMORY]\n{memory_context}")
    if plan:
        context_blocks.append(f"[CLOUD EXECUTION PLAN]\n{plan}")
    context_str = "\n\n".join(context_blocks)

    # Plan + Memory an die letzte User-Message anhängen
    if context_str:
        if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "user":
            user_content = messages[-1]["content"]
            if isinstance(user_content, str):
                messages[-1]["content"] = f"{user_content}\n\n{context_str}"
            elif isinstance(user_content, list):
                messages[-1]["content"] = user_content + [{"type": "text", "text": context_str}]
        else:
            messages.append({"role": "user", "content": context_str})

    # ALLE Messages behalten – kein Compact! Das System-Prompt mit Tool-Defs bleibt erhalten.
    payload["messages"] = messages
    return _clean_payload(payload, keep_tools=True)


def _build_cloud_plan_payload(body: Dict[str, Any], memory_context: str) -> Dict[str, Any]:
    """Baut den Payload für den Cloud-Planer (Caveman Ultra Modus)."""
    user_text = _last_user_text(body.get("messages", []))
    prompt_parts = [
        "TASK:",
        user_text,
        "",
        "CONSTRAINTS:",
        "- Return ONLY abstract execution plan (no code, no tool calls).",
        "- Each step: WHAT to do, WHICH file, WHY.",
        "- DO NOT write code. DO NOT explore. Just plan.",
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
                    "STRATEGIC PLANNER ONLY. You have NO tools, NO terminal, NO file access. "
                    "Your ONLY output: a terse abstract execution plan (5-8 bullet points). "
                    "DO NOT produce code. DO NOT use tool_calls, invoke, or function calls. "
                    "DO NOT explore the workspace — just plan from the task description. "
                    "Format: numbered list, each line < 80 chars, symbols only (-> ! ? TODO FIX)."
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
    """Cloud-Planer via LiteLLM (direkter httpx-Call wenn URL gesetzt, sonst via litellm-Library)."""
    user_text = _last_user_text(body.get("messages", []))
    prompt = (
        f"TASK:\n{user_text}\n\n"
        "Create an abstract execution plan (5-8 steps). WHAT to do, WHICH file. "
        "NO code. NO tool calls. NO exploration. This will be executed NOW."
    )
    if CAVEMAN_ENABLED:
        prompt = f"{CAVEMAN_SYSTEM_PROMPT}\n\n{prompt}"
    if memory_context:
        prompt = f"[HINDSIGHT]\n{memory_context}\n\n{prompt}"

    payload = {
        "model": LITELLM_CLOUD_MODEL,
        "messages": [
            {"role": "system", "content": (
                "STRATEGIC PLANNER ONLY. NO tools. NO code execution. NO file access. "
                "Output ONLY: 5-8 bullet point abstract plan. Terse symbols. No code. No tool calls."
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
    Sendet den Task + Caveman-Plan an das Cloud-Modell, das die Antwort generiert."""
    user_text = _last_user_text(body.get("messages", []))
    prompt = (
        f"TASK:\n{user_text}\n\n"
        f"EXECUTE THIS PLAN:\n{plan}\n\n"
        "You are the executor. Implement the plan. Provide complete code, "
        "detailed explanation, and working solution. No tool calls — write code directly."
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


async def _call_cloud_reviewer(
    client: httpx.AsyncClient,
    task: str,
    response: str,
) -> Dict[str, Any]:
    """Cloud-Review: Sendet Worker-Antwort zur Qualitätsprüfung an die Cloud."""
    prompt = (
        f"ORIGINAL TASK:\n{task}\n\n"
        f"RESPONSE TO REVIEW:\n{response}\n\n"
        "Review the response. Check for: correctness, completeness, bugs, edge cases, "
        "security issues. Be concise. Output format:\n"
        "## Review Result: [PASS / NEEDS FIX]\n"
        "## Issues Found:\n- ...\n"
        "## Suggestions:\n- ...\n"
        "If PASS, just say 'PASS — no issues found.'"
    )

    # Prefer LiteLLM, fallback to Cloud Reviewer
    if LITELLM_CLOUD_MODEL and LITELLM_CLOUD_API_KEY:
        model = LITELLM_CLOUD_MODEL
        api_key = LITELLM_CLOUD_API_KEY
        api_url = LITELLM_CLOUD_API_URL
        timeout = LITELLM_CLOUD_TIMEOUT_SECONDS
        _log(f"  🔍 Cloud-Reviewer via LiteLLM: model={model}")
    elif CLOUD_REVIEW_ENABLED and CLOUD_REVIEW_API_KEY:
        model = CLOUD_REVIEW_MODEL
        api_key = CLOUD_REVIEW_API_KEY
        api_url = CLOUD_REVIEW_API_URL
        timeout = CLOUD_REVIEW_TIMEOUT_SECONDS
        _log(f"  🔍 Cloud-Reviewer via ReviewLLM: model={model}")
    else:
        _log("  ✗ Cloud-Reviewer: Kein Cloud-Modell verfügbar")
        return {
            "agent_key": "cloud_reviewer",
            "status": "skipped",
            "content": "",
            "duration_seconds": 0.0,
            "usage": None,
        }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict code reviewer. Be concise."},
            {"role": "user", "content": prompt},
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
            _log(f"  ✓ Cloud-Reviewer OK: model={model} duration={duration:.1f}s")
            return {
                "agent_key": "cloud_reviewer",
                "status": "ok",
                "content": content or "",
                "duration_seconds": duration,
                "usage": result.get("usage"),
            }
        _log(f"  ⚠ Cloud-Reviewer STATUS {r.status_code}: duration={duration:.1f}s")
        return {
            "agent_key": "cloud_reviewer",
            "status": "failed",
            "content": "",
            "duration_seconds": duration,
            "usage": None,
        }
    except Exception as exc:
        _log(f"  ✗ Cloud-Reviewer ERROR: model={model} – {exc}")
        return {
            "agent_key": "cloud_reviewer",
            "status": "error",
            "content": "",
            "duration_seconds": time.perf_counter() - started,
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
    """Erzwingt Moonshot-kompatible Parameter (nur spezifische Werte erlaubt)."""
    model = str(payload.get("model", ""))
    
    model_lower = model.lower()
    vllm_url_lower = VLLM_API_URL.lower()
    cloud_url_lower = CLOUD_REVIEW_API_URL.lower()
    lite_url_lower = LITELLM_CLOUD_API_URL.lower()
    
    is_moonshot = (
        "kimi" in model_lower or "moonshot" in model_lower or
        "moonshot" in vllm_url_lower or "kimi" in vllm_url_lower or
        "moonshot" in cloud_url_lower or "kimi" in cloud_url_lower or
        "moonshot" in lite_url_lower or "kimi" in lite_url_lower
    )
    
    _log(f"  🔍 Moonshot-check: model={model} VLLM_URL={VLLM_API_URL[:50]} "
         f"CLOUD_URL={CLOUD_REVIEW_API_URL[:50]} "
         f"LITELLM_URL={LITELLM_CLOUD_API_URL[:50]} is_moonshot={is_moonshot}")
    
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
    cloud_payload = _clean_payload(cloud_payload, keep_tools=False)

    # ── Versuch 1: LiteLLM (DeepSeek) ──────────────────────────────────
    if LITELLM_CLOUD_API_URL and LITELLM_CLOUD_API_KEY:
        cloud_payload["model"] = LITELLM_CLOUD_MODEL
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

    # Moonshot/Kimi-Temperatur-Korrektur (Moonshot erlaubt nur 1.0)
    _patch_moonshot_payload(payload)

    # Reasoning-Content aus Cache in Assistant-Messages injizieren
    # (DeepSeek braucht reasoning_content beim Folgerequest mit tool_calls)
    messages = payload.get("messages", [])
    if isinstance(messages, list):
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
    3-Phasen-Agenten-Workflow:
      Phase 1: Hindsight Recall → Cloud-Planer (Caveman)
      Phase 2: Worker führt Caveman-Plan aus
      Phase 3: Verifikation & Self-Correction (Linter/Tests + lokales Modell)

    Pipeline-Steuerflags (via Prompt-Zusätze):
      -force planning  → Cloud-Planung erzwingen (auch wenn deaktiviert)
      -force review    → Cloud-Review nach Worker-Ausführung erzwingen
      -bypass worker   → Worker (80B) überspringen, Cloud antwortet direkt
    """
    start_time = time.perf_counter()
    progress: List[str] = []
    results: List[Dict[str, Any]] = []

    query = _last_user_text(body.get("messages", []))
    _, pflags = _extract_pipeline_flags(query)
    force_planning = pflags.get("force_planning", False)
    force_review = pflags.get("force_review", False)
    bypass_worker = pflags.get("bypass_worker", False)

    # Flags aus Messages strippen bevor sie an Modelle gehen
    _strip_pipeline_flags_from_messages(body.get("messages", []))
    query_clean = _last_user_text(body.get("messages", []))

    flag_summary = []
    if force_planning: flag_summary.append("force_planning")
    if force_review: flag_summary.append("force_review")
    if bypass_worker: flag_summary.append("bypass_worker")
    flag_str = ", ".join(flag_summary) if flag_summary else "none"

    _log(f"▶ AGENT: worker={MODEL_NAME} cloud={CLOUD_REVIEW_MODEL if CLOUD_REVIEW_ENABLED else '–'} "
         f"litellm={LITELLM_CLOUD_MODEL or '–'} messages={len(body.get('messages',[]))} "
         f"flags=[{flag_str}]")
    client = httpx.AsyncClient()

    # ── Phase 1: Hindsight Recall + Cloud-Planung ──────────────────────
    memory_records = _hindsight.recall(query_clean)
    memory_context = _hindsight.format_context(memory_records)
    progress.append(_format_chat_progress_message(
        "phase1_recall",
        f"Hindsight Recall: {len(memory_records)} relevante Erinnerungen geladen.",
        {"memory_records": len(memory_records), "networks": list(set(n for r in memory_records for n in r.networks))},
    ))

    planner_result = await _call_cloud_planner(client, body, memory_context, force=force_planning)
    results.append(planner_result)
    plan_status = planner_result.get("status")
    plan = planner_result.get("content", "") if plan_status == "ok" else ""

    if plan_status == "ok":
        progress.append(_format_chat_progress_message(
            "phase1_plan_ready",
            f"Cloud-Planer: Caveman-Plan erstellt{' (forced)' if force_planning else ''}.",
            {"duration_seconds": planner_result.get("duration_seconds")},
        ))
    else:
        # Planer fehlgeschlagen → Fallback
        fallback_reason = f"Cloud-Planer nicht verfügbar ({plan_status})."
        if force_planning:
            fallback_reason += " -force planning ignoriert, Fallback auf lokale Ausführung."

        progress.append(_format_chat_progress_message(
            "phase1_plan_fallback",
            f"{fallback_reason}",
            {"status": plan_status},
        ))
        # Fallback: direkt lokal ausführen (mit Cloud-Fallback)
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

    # ── Phase 2: Worker (oder Cloud-Bypass) ─────────────────────────────
    if bypass_worker:
        # ── -bypass worker: Cloud (LiteLLM/ReviewLLM) antwortet direkt ──
        _log("  ⚡ -bypass worker: Worker übersprungen, Cloud antwortet direkt")
        progress.append(_format_chat_progress_message(
            "phase2_bypass",
            f"⚡ Worker übersprungen. Cloud ({LITELLM_CLOUD_MODEL or CLOUD_REVIEW_MODEL}) antwortet direkt.",
            {"bypass": True, "plan_len": len(plan)},
        ))

        cloud_response = await _call_cloud_as_responder(client, body, plan, memory_context)
        results.append(cloud_response)
        worker_response = cloud_response.get("content", "")

        progress.append(_format_chat_progress_message(
            "phase2_done",
            "Cloud-Direktantwort abgeschlossen.",
            {"status": cloud_response.get("status"), "duration_seconds": cloud_response.get("duration_seconds")},
        ))
    else:
        # ── Normaler Worker-Pfad ──────────────────────────────────────────
        progress.append(_format_chat_progress_message(
            "phase2_execute",
            f"Worker ({MODEL_NAME}) führt Caveman-Plan aus.",
            {"model": body.get("model", MODEL_NAME)},
        ))

        worker_payload = _build_worker_payload(body, plan, memory_context)
        worker_result = await _call_vllm_with_fallback(client, worker_payload, "worker")
        results.append(worker_result)
        worker_response = worker_result.get("content", "")

        # Wenn der Worker Tool-Calls generiert: SOFORT raw an VS Code zurückgeben.
        if worker_result.get("tool_calls") or _contains_tool_calls(worker_response):
            _log("  🔧 Worker enthält Tool-Calls/DSML → sofort Durchreiche an Client")
            await client.aclose()
            _hindsight.retain(body, worker_response)
            return {
                "combined_response_text": worker_response,
                "results": results,
                "duration_seconds": time.perf_counter() - start_time,
            }

        progress.append(_format_chat_progress_message(
            "phase2_done",
            "Worker-Ausführung abgeschlossen.",
            {"status": worker_result.get("status"), "duration_seconds": worker_result.get("duration_seconds")},
        ))

    # ── Phase 3: Verifikation & Self-Correction ────────────────────────
    # -force review: Cloud-Review nach Worker-Ausführung
    if force_review and not bypass_worker:
        # Cloud-Review der Worker-Antwort
        _log("  🔍 -force review: Cloud-Review der Worker-Antwort")
        progress.append(_format_chat_progress_message(
            "phase3_cloud_review",
            "🔍 Cloud-Review: Worker-Antwort wird von Cloud geprüft.",
            {},
        ))
        cloud_review_result = await _call_cloud_reviewer(client, query_clean, worker_response)
        if cloud_review_result.get("status") == "ok":
            review_text = cloud_review_result.get("content", "")
            if review_text:
                worker_response = f"{worker_response}\n\n---\n## 🔍 Cloud Review\n{review_text}"
            progress.append(_format_chat_progress_message(
                "phase3_review_done",
                "Cloud-Review abgeschlossen.",
                {"status": "ok"},
            ))
        else:
            progress.append(_format_chat_progress_message(
                "phase3_review_failed",
                f"Cloud-Review fehlgeschlagen: {cloud_review_result.get('status')}",
                {"status": cloud_review_result.get("status")},
            ))
        verified_response = worker_response
        verify_info = {"stage": "cloud_review", "status": cloud_review_result.get("status")}
    else:
        # Standard-Verifikation (Linter/Tests + lokales Modell)
        progress.append(_format_chat_progress_message(
            "phase3_verify",
            "Phase 3: Linter/Tests + Self-Correction.",
            {},
        ))

        verified_response, verify_info = await _verify_and_correct(client, query_clean, worker_response)
        progress.append(_format_chat_progress_message(
            "phase3_done",
            f"Verifikation: {verify_info.get('stage', 'unknown')}",
            verify_info,
        ))

    await client.aclose()

    # ── Tool-Calls aus Cloud-Plan filtern (Planer hat keine Tools) ────
    if _contains_tool_calls(plan):
        _log("  ⚠ Cloud-Plan enthält Tool-Calls → wird gefiltert")
        plan = re.sub(r'<[^>]+>.*?</[^>]+>', '', plan, flags=re.DOTALL).strip()

    # ── Antwort zusammenbauen ──────────────────────────────────────────
    has_tools = _contains_tool_calls(worker_response)
    if has_tools:
        _log("  🔧 Worker/Cloud enthält Tool-Calls → raw Durchreiche an Client")
        combined = worker_response
    elif CHATTY_MODE and planner_result.get("status") == "ok":
        source_label = LITELLM_CLOUD_MODEL or CLOUD_REVIEW_MODEL if bypass_worker else MODEL_NAME
        phase_label = "☁️ Cloud-Direktantwort" if bypass_worker else "🛠️ Lokale Umsetzung"
        final_response = (
            f"## 🧭 Caveman Cloud Plan\n\n{plan}\n\n"
            f"---\n\n"
            f"## {phase_label} ({source_label})\n\n{verified_response}"
        )
        combined = "".join(progress) + final_response
    else:
        final_response = verified_response
        combined = "".join(progress) + final_response
    _hindsight.retain(body, combined)

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
        
        # 1. Role-Chunk: assistant + content=null + reasoning_content
        #    reasoning_content MUSS im ersten Delta sein, damit VS Code es in der
        #    History speichert – sonst fehlt es beim nächsten DeepSeek-Request → 400.
        yield _format_openai_stream_chunk(model, include_role=True, reasoning_content=reasoning_content, chunk_id=stream_id)
        
        # 2. Tool-Call Header-Chunk: id + type + function.name (leere arguments)
        first_tcs = []
        for i, tc in enumerate(tool_calls):
            func = tc.get("function", {})
            first_tcs.append({
                "index": i,
                "id": tc.get("id", f"call_{uuid.uuid4().hex}"),
                "type": "function",
                "function": {
                    "name": func.get("name", ""),
                    "arguments": "",
                },
            })
        yield _format_openai_stream_chunk(model, tool_calls=first_tcs, chunk_id=stream_id)

        # 3. Arguments für jeden Tool-Call als separater Chunk
        for i, tc in enumerate(tool_calls):
            args = tc.get("function", {}).get("arguments", "")
            if args:
                yield _format_openai_stream_chunk(model, tool_calls=[
                    {"index": i, "function": {"arguments": args}},
                ], chunk_id=stream_id)

        # 4. Finaler Chunk mit finish_reason='tool_calls' (leeres delta)
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
        _models_url = VLLM_MODELS_URL
        if not _models_url.startswith(("http://", "https://")):
            _models_url = VLLM_API_URL.rstrip("/").rsplit("/chat/completions", 1)[0] + "/models"
        elif not _models_url.endswith("/models") and not _models_url.endswith("/v1/models"):
            _models_url = VLLM_API_URL.rstrip("/") + "/models"
        _log(f"   🔍 Worker '{MODEL_NAME}' @ {VLLM_API_URL} ...")
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
            _log(f"   ❌ Worker: NICHT ERREICHBAR – {exc}")

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
                _log(f"   ❌ Fast-Modell: NICHT ERREICHBAR – {exc}")

        # 1c. Models-Liste (nice-to-have, viele Cloud-Proxys haben keinen /models-Endpoint)
        _log(f"   🔍 Models-Liste via {VLLM_MODELS_URL} ...")
        try:
            r = await hc.get(VLLM_MODELS_URL, headers=_vllm_headers())
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
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    await _auth_or_raise(request)
    body = await request.json()

    if "messages" not in body:
        raise HTTPException(status_code=400, detail="Invalid payload: 'messages' required.")

    # Ignoriere Client-Modell – der Proxy verwendet sein konfiguriertes Modell
    body["model"] = MODEL_NAME

    msgs = body.get("messages", [])
    _log(f"📨 Request: {len(msgs)} messages, stream={body.get('stream')}, tool_cont={_is_tool_continuation(msgs)}")

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
    await _auth_or_raise(request)
    models = [
        {"id": MODEL_NAME, "object": "model", "owned_by": "local-free"},
        {"id": FAST_MODEL_NAME, "object": "model", "owned_by": "local-free"},
    ]
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(VLLM_MODELS_URL, headers=_vllm_headers())
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
    })


# ── /logs ──────────────────────────────────────────────────────────────────
@app.get("/logs")
async def get_logs(request: Request, lines: int = 200):
    """Gibt die letzten Log-Zeilen zurück. Ohne Auth (WebUI-Aufruf ohne Bearer-Token)."""
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
    except (FileNotFoundError, OSError):
        all_lines = []
    last = all_lines[-lines:] if lines > 0 else all_lines
    return JSONResponse(content={
        "count": len(last),
        "total": len(all_lines),
        "file": LOG_FILE,
        "lines": last,
    })


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
