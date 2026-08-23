"""
Web-Konfigurationsinterface fuer LocalProxy v3.0
Modernes Single-Page-Dashboard: 4 Modell-Kategorien, Hindsight, Proxy.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import signal
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

CONFIG_PATH = Path(os.getenv("LOCALPROXY_CONFIG", "data/config.json"))
PROFILES_DIR = CONFIG_PATH.parent / "profiles"
LOG_FILE = os.getenv("LOG_FILE", str(Path(__file__).parent / "data" / "proxy.log"))


def _log(msg: str) -> None:
    import datetime as _dt
    timestamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [webui] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _safe_str(val: object) -> str:
    """Konvertiert zu str mit ASCII-safe Fallback. Faengt ALLE Exceptions."""
    try:
        s = str(val)
        s.encode("ascii")
        return s
    except Exception:
        try:
            return str(val).encode("ascii", errors="replace").decode("ascii")
        except Exception:
            return "(unreadable)"


# ═══════════════════════════════════════════════════════════════════════════
# WebUI Auth
# ═══════════════════════════════════════════════════════════════════════════

WEBUI_USERNAME: str = os.getenv("SPARK_AUTH_USERNAME", "admin")
WEBUI_PASSWORD: str = os.getenv("SPARK_AUTH_PASSWORD", "")
if not WEBUI_PASSWORD:
    WEBUI_PASSWORD = "localfox-" + secrets.token_hex(16)
    _log(f"WebUI Auto-Passwort (kein SPARK_AUTH_PASSWORD gesetzt): {WEBUI_PASSWORD}")
else:
    _log("WebUI Login via SPARK_AUTH_USERNAME / SPARK_AUTH_PASSWORD")

_active_tokens: Set[str] = set()
COOKIE_NAME = "webui_token"


def _generate_token() -> str:
    token = uuid.uuid4().hex + secrets.token_hex(16)
    _active_tokens.add(token)
    return token


def _validate_token(token: str) -> bool:
    return token in _active_tokens


def _remove_token(token: str) -> None:
    _active_tokens.discard(token)


LOGIN_HTML = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>LocalProxy Login</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Oxygen,Ubuntu,sans-serif;
    background: #0d1117; color: #e6edf3;
    display: flex; align-items: center; justify-content: center;
    min-height: 100vh;
  }
  .login-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 40px; width: 380px; max-width: 90vw;
    box-shadow: 0 8px 32px rgba(0,0,0,.4);
  }
  .login-card h1 { font-size: 1.6rem; margin-bottom: 4px; }
  .login-card p { color: #8b949e; font-size: 0.85rem; margin-bottom: 24px; }
  .form-group { margin-bottom: 16px; }
  .form-group label { display: block; font-size: 0.8rem; margin-bottom: 6px; color: #8b949e; }
  .form-group input {
    width: 100%; padding: 10px 12px; background: #0d1117; border: 1px solid #30363d;
    border-radius: 6px; color: #e6edf3; font-size: 0.9rem; outline: none;
    transition: border-color .15s;
  }
  .form-group input:focus { border-color: #58a6ff; }
  .btn {
    width: 100%; padding: 10px; background: #238636; color: #fff; border: none;
    border-radius: 6px; font-size: 0.9rem; font-weight: 500; cursor: pointer;
    transition: background .15s;
  }
  .btn:hover { background: #2ea043; }
  .error { color: #f85149; font-size: 0.8rem; margin-top: 12px; text-align: center; display: none; }
  .badge { font-size: 0.6rem; background: #1f6feb33; color: #58a6ff; padding: 2px 8px; border-radius: 10px; vertical-align: middle; }
</style>
</head>
<body>
<div class="login-card">
  <h1>LocalProxy <span class="badge">v3.0</span></h1>
  <p>Bitte anmelden um auf das Dashboard zuzugreifen</p>
  <form id="loginForm">
    <div class="form-group">
      <label for="username">Benutzername</label>
      <input type="text" id="username" name="username" placeholder="admin" autocomplete="username" autofocus>
    </div>
    <div class="form-group">
      <label for="password">Passwort</label>
      <input type="password" id="password" name="password" placeholder="..." autocomplete="current-password">
    </div>
    <div class="error" id="loginError">Falscher Benutzername oder Passwort</div>
    <button type="submit" class="btn">Anmelden</button>
  </form>
</div>
<script>
document.getElementById('loginForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const username = document.getElementById('username').value;
  const password = document.getElementById('password').value;
  const errorEl = document.getElementById('loginError');
  try {
    const r = await fetch('/webui/api/login', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({username, password})
    });
    if (!r.ok) { errorEl.style.display = 'block'; return; }
    const data = await r.json();
    document.cookie = 'webui_token=' + data.token + '; path=/webui; max-age=86400; SameSite=Lax';
    window.location.href = '/webui/?token=' + data.token;
  } catch(e) {
    errorEl.textContent = 'Netzwerkfehler: ' + e.message;
    errorEl.style.display = 'block';
  }
});
</script>
</body>
</html>"""


DEFAULT_CONFIG: Dict[str, Any] = {
    "model_categories": {
        "local": {
            "api_url": "http://localhost:8000/v1/chat/completions",
            "api_key": "",
            "model_name": "Qwen/Qwen3-Next-80B",
            "max_tokens": 65536,
            "use_max_completion_tokens": False,
            "is_vision": False,
            "timeout_seconds": 300,
            "read_timeout_seconds": 120,
            "retry_on_timeout": 2,
            "retry_delay_seconds": 5,
        },
        "coworker": {
            "api_url": "",
            "api_key": "",
            "model_name": "",
            "max_tokens": 65536,
            "use_max_completion_tokens": False,
            "is_vision": False,
            "timeout_seconds": 300,
            "read_timeout_seconds": 120,
            "retry_on_timeout": 2,
            "retry_delay_seconds": 5,
        },
        "light": [
            {
                "label": "light primary",
                "api_url": "https://api.openai.com/v1/chat/completions",
                "api_key": "",
                "model_name": "gpt-4.1-mini",
                "max_tokens": 65536,
                "use_max_completion_tokens": False,
                "is_vision": False,
                "timeout_seconds": 180,
            },
            {"label": "light fallback 2", "api_url": "", "api_key": "", "model_name": "", "max_tokens": 65536, "use_max_completion_tokens": False, "is_vision": False, "timeout_seconds": 180},
            {"label": "light fallback 3", "api_url": "", "api_key": "", "model_name": "", "max_tokens": 65536, "use_max_completion_tokens": False, "is_vision": False, "timeout_seconds": 180},
        ],
        "strong": [
            {
                "label": "strong primary",
                "api_url": "https://api.anthropic.com/v1/chat/completions",
                "api_key": "",
                "model_name": "claude-sonnet-4-20250514",
                "max_tokens": 65536,
                "use_max_completion_tokens": False,
                "is_vision": False,
                "timeout_seconds": 300,
            },
            {"label": "strong fallback 2", "api_url": "", "api_key": "", "model_name": "", "max_tokens": 65536, "use_max_completion_tokens": False, "is_vision": False, "timeout_seconds": 300},
            {"label": "strong fallback 3", "api_url": "", "api_key": "", "model_name": "", "max_tokens": 65536, "use_max_completion_tokens": False, "is_vision": False, "timeout_seconds": 300},
        ],
        "vision": [
            {
                "label": "vision primary",
                "api_url": "https://api.openai.com/v1/chat/completions",
                "api_key": "",
                "model_name": "gpt-4o",
                "max_tokens": 65536,
                "use_max_completion_tokens": False,
                "is_vision": True,
                "timeout_seconds": 180,
            },
            {"label": "vision fallback 2", "api_url": "", "api_key": "", "model_name": "", "max_tokens": 65536, "use_max_completion_tokens": False, "is_vision": True, "timeout_seconds": 180},
            {"label": "vision fallback 3", "api_url": "", "api_key": "", "model_name": "", "max_tokens": 65536, "use_max_completion_tokens": False, "is_vision": True, "timeout_seconds": 180},
        ],
    },
    "default_category": "light",
    "proxy": {
        "port": 9001,
        "auth_enabled": True,
        "api_key": "",
    },
    "hindsight": {
        "enabled": True,
        "qdrant_url": "http://localhost:6333",
        "qdrant_api_key": "",
        "use_qdrant": False,
        "collection": "hindsight_memory",
        "embedding_dim": 768,
        "max_memory_tokens": 4000,
        "min_similarity": 0.18,
        "retain_delay_seconds": 0,
        "dir": "./.hindsight_memory",
    },
    "tokens": {
        "tool_result_cap": 0,
        "read_loop_threshold": 3,
        "read_loop_intervention": "",
        "read_loop_file_threshold": 8,
        "read_loop_file_window": 12,
        "read_loop_file_keep": 1,
        "read_loop_file_intervention": "",
        "search_loop_threshold": 3,
        "search_loop_intervention": "",
        "search_repeat_threshold": 3,
        "search_repeat_intervention": "",
        "response_loop_threshold": 3,
        "generic_tool_loop_threshold": 3,
        "generic_tool_loop_intervention": "",
        "coworker": {
            "enabled": True,
            "max_delegations_per_request": 2,
            "task_cap_chars": 8000,
            "result_cap_chars": 12000,
            "files_cap_chars": 60000,
            "health_interval_seconds": 60,
            "probe_timeout_seconds": 5,
            "enable_fork_join": True,
            "max_parallel": 8,
            "dispatch_cap_per_request": 12,
            "bg_ttl_seconds": 1800,
            "teach_delegation": True,
            "system_prompt": "You are a co-worker coding model acting as a subagent for planning, code review, brainstorming and parallel implementation tasks. You receive a self-contained task and optional context. Answer with concrete, actionable output: for planning give a step-by-step plan; for review list issues with file/line references and concrete fixes; for coding give complete, idiomatic code. You have NO access to tools, the conversation history or the workspace — work only from what is given to you.",
        },
        "local_sampling": {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "dry_multiplier": 0.8,
            "dry_base": 1.75,
            "dry_allowed_length": 3,
            "dry_penalty_last_n": -1,
            "dry_sequence_breaker": "\n,:,\",*,;,{,}",
            "enable_thinking": True,
            "preserve_thinking": True,
            "anti_loop_system_prompt": "",
        },
    },
}

_ENV_TO_CONFIG: Dict[str, Tuple[str, str]] = {
    "PROXY_PORT": ("proxy", "port"),
    "PROXY_AUTH_ENABLED": ("proxy", "auth_enabled"),
    "PROXY_API_KEY": ("proxy", "api_key"),
    "HINDSIGHT_ENABLED": ("hindsight", "enabled"),
    "QDRANT_URL": ("hindsight", "qdrant_url"),
    "QDRANT_API_KEY": ("hindsight", "qdrant_api_key"),
    "HINDSIGHT_USE_QDRANT": ("hindsight", "use_qdrant"),
    "HINDSIGHT_COLLECTION": ("hindsight", "collection"),
    "HINDSIGHT_EMBEDDING_DIM": ("hindsight", "embedding_dim"),
    "HINDSIGHT_MAX_MEMORY_TOKENS": ("hindsight", "max_memory_tokens"),
    "HINDSIGHT_MIN_SIMILARITY": ("hindsight", "min_similarity"),
    "HINDSIGHT_RETAIN_DELAY_SECONDS": ("hindsight", "retain_delay_seconds"),
    "HINDSIGHT_DIR": ("hindsight", "dir"),
    "TOOL_RESULT_CAP": ("tokens", "tool_result_cap"),
    "READ_LOOP_THRESHOLD": ("tokens", "read_loop_threshold"),
    "READ_LOOP_INTERVENTION": ("tokens", "read_loop_intervention"),
    "SEARCH_LOOP_THRESHOLD": ("tokens", "search_loop_threshold"),
    "SEARCH_LOOP_INTERVENTION": ("tokens", "search_loop_intervention"),
    "SEARCH_REPEAT_THRESHOLD": ("tokens", "search_repeat_threshold"),
    "SEARCH_REPEAT_INTERVENTION": ("tokens", "search_repeat_intervention"),
    "RESPONSE_LOOP_THRESHOLD": ("tokens", "response_loop_threshold"),
    "GENERIC_TOOL_LOOP_THRESHOLD": ("tokens", "generic_tool_loop_threshold"),
    "GENERIC_TOOL_LOOP_INTERVENTION": ("tokens", "generic_tool_loop_intervention"),
}


def _env_to_config_val(env_val: str, default_val: Any) -> Any:
    if isinstance(default_val, bool):
        return env_val.lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(default_val, int):
        try:
            return int(env_val)
        except (ValueError, TypeError):
            return default_val
    if isinstance(default_val, float):
        try:
            return float(env_val)
        except (ValueError, TypeError):
            return default_val
    return env_val


def _normalize_model_categories(cfg: Dict[str, Any]) -> None:
    """Stellt sicher, dass light/strong/vision immer Arrays sind (auch bei Legacy-Dict-Format).
    Fehlende Felder werden mit Defaults aus DEFAULT_CONFIG aufgefuellt."""
    cats = cfg.get("model_categories")
    if not isinstance(cats, dict):
        return
    default_cats = DEFAULT_CONFIG.get("model_categories", {})
    for key in ("light", "strong", "vision"):
        cat = cats.get(key)
        if isinstance(cat, dict):
            # Legacy single-dict → in Array wandeln
            cats[key] = [cat]
        elif not isinstance(cat, list) or len(cat) == 0:
            # Kein Array oder leeres Array → Default-Array aus DEFAULT_CONFIG
            default_arr = default_cats.get(key, [])
            cats[key] = json.loads(json.dumps(default_arr))
        else:
            # Array vorhanden: fehlende Felder pro Slot mit Defaults auffuellen
            default_arr = default_cats.get(key, [])
            for i, slot in enumerate(cat):
                if not isinstance(slot, dict):
                    cat[i] = json.loads(json.dumps(default_arr[0] if default_arr else {}))
                    continue
                defaults_for_slot = default_arr[i] if i < len(default_arr) else (default_arr[0] if default_arr else {})
                for field in ("label", "api_url", "api_key", "model_name", "max_tokens",
                               "use_max_completion_tokens", "is_vision", "timeout_seconds"):
                    if field not in slot:
                        slot[field] = defaults_for_slot.get(field)
                # Typ-Garantien
                slot["max_tokens"] = int(slot.get("max_tokens", 65536))
                slot["timeout_seconds"] = float(slot.get("timeout_seconds", 180))
                slot["use_max_completion_tokens"] = bool(slot.get("use_max_completion_tokens", False))
                slot["is_vision"] = bool(slot.get("is_vision", False))
    # local bleibt Single-Dict (kann aber auch schon Array sein → normalisieren)
    local_cat = cats.get("local")
    if isinstance(local_cat, list):
        cats["local"] = local_cat[0] if local_cat and isinstance(local_cat[0], dict) else cats.get("local")
    if isinstance(cats.get("local"), dict):
        loc = cats["local"]
        loc.setdefault("label", "local")
        loc.setdefault("api_url", "")
        loc.setdefault("api_key", "")
        loc.setdefault("model_name", "")
        loc.setdefault("max_tokens", 65536)
        loc.setdefault("timeout_seconds", 300)
        loc.setdefault("read_timeout_seconds", 120)
        loc.setdefault("retry_on_timeout", 2)
        loc.setdefault("retry_delay_seconds", 5)
        loc.setdefault("use_max_completion_tokens", False)
        loc.setdefault("is_vision", False)
    # coworker bleibt ebenfalls Single-Dict (wie local)
    coworker_cat = cats.get("coworker")
    if isinstance(coworker_cat, list):
        cats["coworker"] = coworker_cat[0] if coworker_cat and isinstance(coworker_cat[0], dict) else cats.get("coworker")
    if isinstance(cats.get("coworker"), dict):
        cw = cats["coworker"]
        cw.setdefault("label", "coworker")
        cw.setdefault("api_url", "")
        cw.setdefault("api_key", "")
        cw.setdefault("model_name", "")
        cw.setdefault("max_tokens", 65536)
        cw.setdefault("timeout_seconds", 300)
        cw.setdefault("read_timeout_seconds", 120)
        cw.setdefault("retry_on_timeout", 2)
        cw.setdefault("retry_delay_seconds", 5)
        cw.setdefault("use_max_completion_tokens", False)
        cw.setdefault("is_vision", False)


def _load_config() -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            _deep_merge(cfg, saved)
        except (json.JSONDecodeError, OSError):
            pass
        _normalize_model_categories(cfg)
        return cfg

    for env_name, (section, key) in _ENV_TO_CONFIG.items():
        val = os.environ.get(env_name)
        if val is not None:
            default_val = DEFAULT_CONFIG.get(section, {}).get(key)
            cfg[section][key] = _env_to_config_val(val, default_val)

    _save_config(cfg)
    _log(f"config.json aus Env-Vars erzeugt: {CONFIG_PATH}")
    return cfg


def _save_config(cfg: Dict[str, Any]) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def _deep_merge(base: Dict, override: Dict) -> None:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _mask_key(key: str) -> str:
    if os.getenv("PROXY_MASK_KEYS", "true").lower() in {"0", "false", "no", "off", "n"}:
        return key  # Debug-Mode: Keys im Klartext zeigen
    if not key or len(key) < 8:
        return key
    return key[:4] + "\u2022" * (len(key) - 8) + key[-4:]


# ═══════════════════════════════════════════════════════════════════════════
# HTML Dashboard
# ═══════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LocalProxy v3.0 — Konfiguration</title>
<style>
:root {
  --bg: #0d1117; --surface: #161b22; --surface2: #21262d; --border: #30363d;
  --text: #e6edf3; --text2: #8b949e; --accent: #58a6ff; --accent2: #238636;
  --danger: #f85149; --warn: #d2991d; --radius: 8px;
  --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: var(--font); background: var(--bg); color: var(--text); line-height: 1.5; }
header {
  background: var(--surface); border-bottom: 1px solid var(--border);
  padding: 16px 24px; display: flex; align-items: center; justify-content: space-between;
  position: sticky; top: 0; z-index: 100;
}
header h1 { font-size: 1.3rem; font-weight: 600; }
header .badge { font-size: 0.75rem; background: var(--accent2); color: #fff; padding: 3px 10px; border-radius: 12px; }
header .status { display: flex; gap: 16px; align-items: center; }
nav {
  background: var(--surface); border-bottom: 1px solid var(--border);
  display: flex; gap: 0; padding: 0 24px; overflow-x: auto;
}
nav button {
  background: none; border: none; color: var(--text2); cursor: pointer;
  padding: 12px 20px; font-size: 0.9rem; border-bottom: 2px solid transparent;
  transition: .15s; white-space: nowrap; font-family: var(--font);
}
nav button:hover { color: var(--text); }
nav button.active { color: var(--accent); border-bottom-color: var(--accent); }
main { max-width: 900px; margin: 24px auto; padding: 0 24px; }
section { display: none; }
section.active { display: block; }
.card {
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 20px; margin-bottom: 16px;
}
.card h3 { font-size: 1rem; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
.form-group { margin-bottom: 14px; position: relative; }
.pw-wrapper { position: relative; }
.pw-wrapper input { width: 100%; padding-right: 36px; }
.pw-toggle { position: absolute; right: 6px; top: 50%; transform: translateY(-50%); background: none; border: none; color: var(--text2); cursor: pointer; font-size: 0.9rem; padding: 4px; line-height: 1; }
.pw-toggle:hover { color: var(--text); }
.form-group label { display: block; font-size: 0.82rem; color: var(--text2); margin-bottom: 4px; font-weight: 500; }
.form-group .hint { font-size: 0.72rem; color: var(--text2); opacity: 0.7; margin-top: 2px; }
input[type="text"], input[type="url"], input[type="number"], input[type="password"], select, textarea {
  width: 100%; padding: 8px 12px; background: var(--surface2); border: 1px solid var(--border);
  border-radius: 6px; color: var(--text); font-size: 0.9rem; font-family: var(--font);
  transition: border-color .15s;
}
input:focus, select:focus, textarea:focus { outline: none; border-color: var(--accent); }
.row { display: flex; gap: 12px; }
.row > * { flex: 1; }
.toggle-row { display: flex; align-items: center; justify-content: space-between; padding: 8px 0; }
.toggle-row span { font-size: 0.9rem; }
.toggle {
  position: relative; width: 44px; height: 24px; flex-shrink: 0;
}
.toggle input { opacity: 0; width: 0; height: 0; }
.toggle .slider {
  position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
  background: var(--surface2); border-radius: 24px; border: 1px solid var(--border);
  transition: .2s;
}
.toggle .slider:before {
  content: ""; position: absolute; height: 18px; width: 18px;
  left: 2px; bottom: 2px; background: var(--text2); border-radius: 50%;
  transition: .2s;
}
.toggle input:checked + .slider { background: var(--accent2); border-color: var(--accent2); }
.toggle input:checked + .slider:before { transform: translateX(20px); background: #fff; }
.actions { display: flex; gap: 12px; margin-top: 20px; }
.btn {
  padding: 10px 20px; border: none; border-radius: var(--radius); font-size: 0.9rem;
  cursor: pointer; font-family: var(--font); font-weight: 500;
  transition: background .15s;
}
.btn-primary { background: var(--accent2); color: #fff; }
.btn-primary:hover { background: #2ea043; }
.btn-danger { background: var(--danger); color: #fff; }
.btn-danger:hover { background: #e5534b; }
.btn-small { padding: 6px 12px; border: 1px solid var(--border); border-radius: var(--radius); font-size: 0.78rem; cursor: pointer; font-family: var(--font); background: var(--bg2); color: var(--text); transition: background .15s; }
.btn-small:hover { background: var(--bg3); }
.toast {
  position: fixed; bottom: 24px; right: 24px; padding: 12px 20px;
  border-radius: var(--radius); font-size: 0.85rem; color: #fff;
  opacity: 0; transform: translateY(20px); transition: .3s;
  z-index: 200;
}
.toast.show { opacity: 1; transform: translateY(0); }
.toast.success { background: var(--accent2); }
.toast.error { background: var(--danger); }
.model-tabs { display: flex; gap: 0; margin-bottom: 16px; border-bottom: 1px solid var(--border); }
.model-tabs button {
  background: none; border: none; color: var(--text2); cursor: pointer;
  padding: 8px 16px; font-size: 0.85rem; font-family: var(--font);
  border-bottom: 2px solid transparent;
}
.model-tabs button:hover { color: var(--text); }
.model-tabs button.active { color: var(--accent); border-bottom-color: var(--accent); }
.model-card { display: none; }
.model-card.active { display: block; }
.model-slot { padding: 4px 0; }
.status-line { font-size: 0.8rem; color: var(--text2); margin-top: 4px; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }
.status-dot.ok { background: #3fb950; }
.status-dot.err { background: var(--danger); }
/* Log-Viewer Farbcodierung */
.log-error { color: #f85149; }
.log-warn { color: #d2991d; }
.log-ok { color: #3fb950; }
.log-webui { color: #a371f7; }
.log-req { color: #79c0ff; }
</style>
</head>
<body>
<header>
  <h1>LocalProxy <span class="badge">v3.0</span></h1>
  <div class="status">
    <span id="statusIndicator"><span class="status-dot" id="statusDot"></span><span id="statusText"></span></span>
    <small style="color:var(--text2)">Port <span id="proxyPort">9001</span></small>
  </div>
</header>
<nav>
  <button data-section="models" class="active">Modelle</button>
  <button data-section="hindsight">Hindsight</button>
  <button data-section="proxy">Proxy</button>
  <button data-section="logs">Logs</button>
</nav>
<main>

  <!-- ====== SEKTION: MODELLE ====== -->
  <section id="section-models" class="active">
    <div class="card">
      <h3>Default-Kategorie</h3>
      <div class="form-group">
        <label>Standard-Kategorie (wenn kein --flag im Prompt)</label>
        <select id="defaultCategory">
          <option value="local">local — lokales Modell</option>
          <option value="coworker">coworker — Co-Worker (2. lokales Modell)</option>
          <option value="light">light — schnelles Cloud-Modell</option>
          <option value="strong">strong — leistungsstarkes Modell</option>
          <option value="vision">vision — multimodales Modell</option>
        </select>
      </div>
    </div>

    <div class="card">
      <h3>Modell-Kategorien</h3>
      <div class="model-tabs">
        <button data-model="local" class="active">local</button>
        <button data-model="coworker">coworker</button>
        <button data-model="light">light</button>
        <button data-model="strong">strong</button>
        <button data-model="vision">vision</button>
      </div>

      <div id="modelCard_local" class="model-card active">
        <div class="form-group">
          <label>API URL (OpenAI-kompatibel)</label>
          <input type="url" id="local_api_url" placeholder="http://localhost:8000/v1/chat/completions">
        </div>
        <div class="form-group">
          <label>API Key</label>
          <input type="password" id="local_api_key" placeholder="sk-...">
        </div>
        <div class="form-group">
          <label>Model Name</label>
          <input type="text" id="local_model_name" placeholder="Qwen/Qwen3-Next-80B">
        </div>
        <div class="row">
          <div class="form-group">
            <label>Max Tokens</label>
            <input type="number" id="local_max_tokens" min="1" max="256000">
          </div>
          <div class="form-group">
            <label>Timeout (Sekunden)</label>
            <input type="number" id="local_timeout_seconds" min="10" max="3600">
          </div>
        </div>
        <div class="row">
          <div class="form-group">
            <label>Read-Timeout (Sekunden)</label>
            <input type="number" id="local_read_timeout_seconds" min="10" max="3600">
            <div class="hint">Max. Wartezeit auf Antwort-Daten. Bei Haengern: hier niedriger setzen.</div>
          </div>
          <div class="form-group">
            <label>Auto-Retries bei Timeout</label>
            <input type="number" id="local_retry_on_timeout" min="0" max="10">
            <div class="hint">Wie oft automatisch erneut versucht wird (0=aus).</div>
          </div>
          <div class="form-group">
            <label>Retry-Verzoegerung (s)</label>
            <input type="number" id="local_retry_delay_seconds" min="1" max="120">
          </div>
        </div>
        <div class="toggle-row">
          <span>max_completion_tokens (statt max_tokens)</span>
          <label class="toggle"><input type="checkbox" id="local_use_max_completion_tokens"><span class="slider"></span></label>
        </div>
        <div class="toggle-row">
          <span>Vision (image_url Support)</span>
          <label class="toggle"><input type="checkbox" id="local_is_vision"><span class="slider"></span></label>
        </div>
        <button class="btn btn-primary" onclick="testCategory('local')" style="margin-top:8px">Test-Endpunkt</button>
        <div class="status-line" id="local_test_status"></div>
      </div>

      <div id="modelCard_coworker" class="model-card">
        <div class="form-group">
          <label>API URL (OpenAI-kompatibel)</label>
          <input type="url" id="coworker_api_url" placeholder="http://192.168.x.x:8000/v1/chat/completions">
          <div class="hint">Zweites lokales Modell auf separater Hardware. Wird als ask_coworker-Tool bei Kategorie 'local' injiziert (nur wenn Health-Check OK).</div>
        </div>
        <div class="form-group">
          <label>API Key</label>
          <input type="password" id="coworker_api_key" placeholder="sk-...">
        </div>
        <div class="form-group">
          <label>Model Name</label>
          <input type="text" id="coworker_model_name" placeholder="z.B. Qwen3-Coder">
        </div>
        <div class="row">
          <div class="form-group">
            <label>Max Tokens</label>
            <input type="number" id="coworker_max_tokens" min="1" max="256000">
          </div>
          <div class="form-group">
            <label>Timeout (Sekunden)</label>
            <input type="number" id="coworker_timeout_seconds" min="10" max="3600">
          </div>
        </div>
        <div class="row">
          <div class="form-group">
            <label>Read-Timeout (Sekunden)</label>
            <input type="number" id="coworker_read_timeout_seconds" min="10" max="3600">
          </div>
          <div class="form-group">
            <label>Auto-Retries bei Timeout</label>
            <input type="number" id="coworker_retry_on_timeout" min="0" max="10">
          </div>
          <div class="form-group">
            <label>Retry-Verzoegerung (s)</label>
            <input type="number" id="coworker_retry_delay_seconds" min="1" max="120">
          </div>
        </div>
        <div class="toggle-row">
          <span>max_completion_tokens (statt max_tokens)</span>
          <label class="toggle"><input type="checkbox" id="coworker_use_max_completion_tokens"><span class="slider"></span></label>
        </div>
        <div class="toggle-row">
          <span>Vision (image_url Support)</span>
          <label class="toggle"><input type="checkbox" id="coworker_is_vision"><span class="slider"></span></label>
        </div>
        <button class="btn btn-primary" onclick="testCategory('coworker')" style="margin-top:8px">Test-Endpunkt</button>
        <div class="status-line" id="coworker_test_status"></div>

        <h4 style="margin:16px 0 8px;color:#58a6ff;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.5px">Delegation (ask_coworker)</h4>
        <div class="toggle-row">
          <span>Co-Worker-Delegation aktiviert</span>
          <label class="toggle"><input type="checkbox" id="coworker_enabled" checked><span class="slider"></span></label>
        </div>
        <div class="row">
          <div class="form-group">
            <label>Max. Delegationen pro Request</label>
            <input type="number" id="coworker_max_delegations" min="0" max="20">
            <div class="hint">Schutz gegen Delegation-Loops (0=sofort blocken).</div>
          </div>
          <div class="form-group">
            <label>Task-Cap (Zeichen)</label>
            <input type="number" id="coworker_task_cap" min="100" max="100000">
          </div>
          <div class="form-group">
            <label>Result-Cap (Zeichen)</label>
            <input type="number" id="coworker_result_cap" min="100" max="100000">
          </div>
          <div class="form-group">
            <label>Datei-Kontext-Cap (Zeichen)</label>
            <input type="number" id="coworker_files_cap" min="0" max="500000">
            <div class="hint">Automatisch angehängte Dateiinhalte (Attachments + Tool-Ergebnisse) aus dem Chat — 0 = deaktiviert.</div>
          </div>
        </div>
        <div class="row">
          <div class="form-group">
            <label>Health-Check-Intervall (s)</label>
            <input type="number" id="coworker_health_interval" min="5" max="3600">
          </div>
          <div class="form-group">
            <label>Probe-Timeout (s)</label>
            <input type="number" id="coworker_probe_timeout" min="1" max="60">
          </div>
        </div>
        <h4 style="margin:16px 0 8px;color:#58a6ff;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.5px">Fork-Join (dispatch/collect)</h4>
        <div class="toggle-row">
          <span>Fork-Join aktiviert (dispatch_coworker + collect_coworker)</span>
          <label class="toggle"><input type="checkbox" id="coworker_fork_join" checked><span class="slider"></span></label>
        </div>
        <div class="toggle-row">
          <span>Delegations-Anleitung injizieren (system-Message: wann/wie Coworker-Tools nutzen)</span>
          <label class="toggle"><input type="checkbox" id="coworker_teach_delegation" checked><span class="slider"></span></label>
        </div>
        <div class="row">
          <div class="form-group">
            <label>Max. parallele BG-Tasks (Semaphore)</label>
            <input type="number" id="coworker_max_parallel" min="1" max="32">
            <div class="hint">Gleichzeitige Co-Worker-Calls im Hintergrund (DGX Spark: 8 = Sweet Spot).</div>
          </div>
          <div class="form-group">
            <label>Dispatch-Cap pro Request</label>
            <input type="number" id="coworker_dispatch_cap" min="1" max="50">
            <div class="hint">Schutz gegen Dispatch-Flooding (12 = Standard).</div>
          </div>
          <div class="form-group">
            <label>BG-TTL (s)</label>
            <input type="number" id="coworker_bg_ttl" min="60" max="86400">
            <div class="hint">Hintergrund-Tasks laufen maximal so lange, dann Ablauf (1800 = 30 min).</div>
          </div>
        </div>
        <div class="form-group">
          <label>Co-Worker System-Prompt (Rolle als Subagent)</label>
          <textarea id="coworker_system_prompt" rows="5" style="width:100%;font-family:monospace;font-size:0.8rem"></textarea>
        </div>
      </div>

      <div id="modelCard_light" class="model-card">
        <!-- ── Slot 1: Primary ── -->
        <div class="model-slot">
          <h4 style="margin:12px 0 8px;color:#58a6ff;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.5px">Primary</h4>
          <div class="form-group">
            <label>Label</label>
            <input type="text" id="light_label_1" placeholder="z.B. OpenAI GPT-4.1-mini">
          </div>
          <div class="form-group">
            <label>API URL (OpenAI-kompatibel)</label>
            <input type="url" id="light_api_url" placeholder="https://api.openai.com/v1/chat/completions">
          </div>
          <div class="form-group">
            <label>API Key</label>
            <input type="password" id="light_api_key" placeholder="sk-...">
          </div>
          <div class="form-group">
            <label>Model Name</label>
            <input type="text" id="light_model_name" placeholder="gpt-4.1-mini">
          </div>
          <div class="row">
            <div class="form-group">
              <label>Max Tokens</label>
              <input type="number" id="light_max_tokens" min="1" max="256000">
            </div>
            <div class="form-group">
              <label>Timeout (Sekunden)</label>
              <input type="number" id="light_timeout_seconds" min="10" max="3600">
            </div>
          </div>
          <div class="toggle-row">
            <span>max_completion_tokens (statt max_tokens)</span>
            <label class="toggle"><input type="checkbox" id="light_use_max_completion_tokens"><span class="slider"></span></label>
          </div>
          <div class="toggle-row">
            <span>Vision (image_url Support)</span>
            <label class="toggle"><input type="checkbox" id="light_is_vision"><span class="slider"></span></label>
          </div>
          <button class="btn btn-small" onclick="testCategory('light',1)" style="margin-top:8px">Test Slot 1</button>
          <div class="status-line" id="light_test_status"></div>
        </div>

        <!-- ── Slot 2: Fallback ── -->
        <hr style="border-color:#30363d;margin:16px 0">
        <div class="model-slot">
          <h4 style="margin:0 0 8px;color:#f0883e;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.5px">Fallback 2</h4>
          <div class="form-group">
            <label>Label</label>
            <input type="text" id="light_label_2" placeholder="z.B. Gemini-2.5-flash">
          </div>
          <div class="form-group">
            <label>API URL</label>
            <input type="url" id="light_api_url_2" placeholder="https://api.openai.com/v1/chat/completions">
          </div>
          <div class="form-group">
            <label>API Key</label>
            <input type="password" id="light_api_key_2" placeholder="sk-...">
          </div>
          <div class="form-group">
            <label>Model Name</label>
            <input type="text" id="light_model_name_2" placeholder="gpt-4.1-mini">
          </div>
          <div class="row">
            <div class="form-group">
              <label>Max Tokens</label>
              <input type="number" id="light_max_tokens_2" min="1" max="256000">
            </div>
            <div class="form-group">
              <label>Timeout (Sekunden)</label>
              <input type="number" id="light_timeout_seconds_2" min="10" max="3600">
            </div>
          </div>
          <div class="toggle-row">
            <span>max_completion_tokens (statt max_tokens)</span>
            <label class="toggle"><input type="checkbox" id="light_use_max_completion_tokens_2"><span class="slider"></span></label>
          </div>
          <div class="toggle-row">
            <span>Vision (image_url Support)</span>
            <label class="toggle"><input type="checkbox" id="light_is_vision_2"><span class="slider"></span></label>
          </div>
          <button class="btn btn-small" onclick="testCategory('light',2)" style="margin-top:8px">Test Slot 2</button>
          <div class="status-line" id="light_test_status_2"></div>
        </div>

        <!-- ── Slot 3: Fallback ── -->
        <hr style="border-color:#30363d;margin:16px 0">
        <div class="model-slot">
          <h4 style="margin:0 0 8px;color:#f0883e;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.5px">Fallback 3</h4>
          <div class="form-group">
            <label>Label</label>
            <input type="text" id="light_label_3" placeholder="z.B. DeepSeek-Chat">
          </div>
          <div class="form-group">
            <label>API URL</label>
            <input type="url" id="light_api_url_3" placeholder="https://api.openai.com/v1/chat/completions">
          </div>
          <div class="form-group">
            <label>API Key</label>
            <input type="password" id="light_api_key_3" placeholder="sk-...">
          </div>
          <div class="form-group">
            <label>Model Name</label>
            <input type="text" id="light_model_name_3" placeholder="gpt-4.1-mini">
          </div>
          <div class="row">
            <div class="form-group">
              <label>Max Tokens</label>
              <input type="number" id="light_max_tokens_3" min="1" max="256000">
            </div>
            <div class="form-group">
              <label>Timeout (Sekunden)</label>
              <input type="number" id="light_timeout_seconds_3" min="10" max="3600">
            </div>
          </div>
          <div class="toggle-row">
            <span>max_completion_tokens (statt max_tokens)</span>
            <label class="toggle"><input type="checkbox" id="light_use_max_completion_tokens_3"><span class="slider"></span></label>
          </div>
          <div class="toggle-row">
            <span>Vision (image_url Support)</span>
            <label class="toggle"><input type="checkbox" id="light_is_vision_3"><span class="slider"></span></label>
          </div>
          <button class="btn btn-small" onclick="testCategory('light',3)" style="margin-top:8px">Test Slot 3</button>
          <div class="status-line" id="light_test_status_3"></div>
        </div>
      </div>

      <div id="modelCard_strong" class="model-card">
        <!-- ── Slot 1: Primary ── -->
        <div class="model-slot">
          <h4 style="margin:12px 0 8px;color:#58a6ff;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.5px">Primary</h4>
          <div class="form-group">
            <label>Label</label>
            <input type="text" id="strong_label_1" placeholder="z.B. Claude Sonnet">
          </div>
          <div class="form-group">
            <label>API URL (OpenAI-kompatibel)</label>
            <input type="url" id="strong_api_url" placeholder="https://api.anthropic.com/v1/chat/completions">
          </div>
          <div class="form-group">
            <label>API Key</label>
            <input type="password" id="strong_api_key" placeholder="sk-ant-...">
          </div>
          <div class="form-group">
            <label>Model Name</label>
            <input type="text" id="strong_model_name" placeholder="claude-sonnet-4-20250514">
          </div>
          <div class="row">
            <div class="form-group">
              <label>Max Tokens</label>
              <input type="number" id="strong_max_tokens" min="1" max="256000">
            </div>
            <div class="form-group">
              <label>Timeout (Sekunden)</label>
              <input type="number" id="strong_timeout_seconds" min="10" max="3600">
            </div>
          </div>
          <div class="toggle-row">
            <span>max_completion_tokens (statt max_tokens)</span>
            <label class="toggle"><input type="checkbox" id="strong_use_max_completion_tokens"><span class="slider"></span></label>
          </div>
          <div class="toggle-row">
            <span>Vision (image_url Support)</span>
            <label class="toggle"><input type="checkbox" id="strong_is_vision"><span class="slider"></span></label>
          </div>
          <button class="btn btn-small" onclick="testCategory('strong',1)" style="margin-top:8px">Test Slot 1</button>
          <div class="status-line" id="strong_test_status"></div>
        </div>

        <!-- ── Slot 2: Fallback ── -->
        <hr style="border-color:#30363d;margin:16px 0">
        <div class="model-slot">
          <h4 style="margin:0 0 8px;color:#f0883e;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.5px">Fallback 2</h4>
          <div class="form-group">
            <label>Label</label>
            <input type="text" id="strong_label_2" placeholder="z.B. Gemini 2.5 Pro">
          </div>
          <div class="form-group">
            <label>API URL</label>
            <input type="url" id="strong_api_url_2" placeholder="https://api.anthropic.com/v1/chat/completions">
          </div>
          <div class="form-group">
            <label>API Key</label>
            <input type="password" id="strong_api_key_2" placeholder="sk-ant-...">
          </div>
          <div class="form-group">
            <label>Model Name</label>
            <input type="text" id="strong_model_name_2" placeholder="claude-sonnet-4-20250514">
          </div>
          <div class="row">
            <div class="form-group">
              <label>Max Tokens</label>
              <input type="number" id="strong_max_tokens_2" min="1" max="256000">
            </div>
            <div class="form-group">
              <label>Timeout (Sekunden)</label>
              <input type="number" id="strong_timeout_seconds_2" min="10" max="3600">
            </div>
          </div>
          <div class="toggle-row">
            <span>max_completion_tokens (statt max_tokens)</span>
            <label class="toggle"><input type="checkbox" id="strong_use_max_completion_tokens_2"><span class="slider"></span></label>
          </div>
          <div class="toggle-row">
            <span>Vision (image_url Support)</span>
            <label class="toggle"><input type="checkbox" id="strong_is_vision_2"><span class="slider"></span></label>
          </div>
          <button class="btn btn-small" onclick="testCategory('strong',2)" style="margin-top:8px">Test Slot 2</button>
          <div class="status-line" id="strong_test_status_2"></div>
        </div>

        <!-- ── Slot 3: Fallback ── -->
        <hr style="border-color:#30363d;margin:16px 0">
        <div class="model-slot">
          <h4 style="margin:0 0 8px;color:#f0883e;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.5px">Fallback 3</h4>
          <div class="form-group">
            <label>Label</label>
            <input type="text" id="strong_label_3" placeholder="z.B. DeepSeek-V3">
          </div>
          <div class="form-group">
            <label>API URL</label>
            <input type="url" id="strong_api_url_3" placeholder="https://api.anthropic.com/v1/chat/completions">
          </div>
          <div class="form-group">
            <label>API Key</label>
            <input type="password" id="strong_api_key_3" placeholder="sk-ant-...">
          </div>
          <div class="form-group">
            <label>Model Name</label>
            <input type="text" id="strong_model_name_3" placeholder="claude-sonnet-4-20250514">
          </div>
          <div class="row">
            <div class="form-group">
              <label>Max Tokens</label>
              <input type="number" id="strong_max_tokens_3" min="1" max="256000">
            </div>
            <div class="form-group">
              <label>Timeout (Sekunden)</label>
              <input type="number" id="strong_timeout_seconds_3" min="10" max="3600">
            </div>
          </div>
          <div class="toggle-row">
            <span>max_completion_tokens (statt max_tokens)</span>
            <label class="toggle"><input type="checkbox" id="strong_use_max_completion_tokens_3"><span class="slider"></span></label>
          </div>
          <div class="toggle-row">
            <span>Vision (image_url Support)</span>
            <label class="toggle"><input type="checkbox" id="strong_is_vision_3"><span class="slider"></span></label>
          </div>
          <button class="btn btn-small" onclick="testCategory('strong',3)" style="margin-top:8px">Test Slot 3</button>
          <div class="status-line" id="strong_test_status_3"></div>
        </div>
      </div>

      <div id="modelCard_vision" class="model-card">
        <!-- ── Slot 1: Primary ── -->
        <div class="model-slot">
          <h4 style="margin:12px 0 8px;color:#58a6ff;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.5px">Primary</h4>
          <div class="form-group">
            <label>Label</label>
            <input type="text" id="vision_label_1" placeholder="z.B. GPT-4o">
          </div>
          <div class="form-group">
            <label>API URL (OpenAI-kompatibel)</label>
            <input type="url" id="vision_api_url" placeholder="https://api.openai.com/v1/chat/completions">
          </div>
          <div class="form-group">
            <label>API Key</label>
            <input type="password" id="vision_api_key" placeholder="sk-...">
          </div>
          <div class="form-group">
            <label>Model Name</label>
            <input type="text" id="vision_model_name" placeholder="gpt-4o">
          </div>
          <div class="row">
            <div class="form-group">
              <label>Max Tokens</label>
              <input type="number" id="vision_max_tokens" min="1" max="256000">
            </div>
            <div class="form-group">
              <label>Timeout (Sekunden)</label>
              <input type="number" id="vision_timeout_seconds" min="10" max="3600">
            </div>
          </div>
          <div class="toggle-row">
            <span>max_completion_tokens (statt max_tokens)</span>
            <label class="toggle"><input type="checkbox" id="vision_use_max_completion_tokens"><span class="slider"></span></label>
          </div>
          <div class="toggle-row">
            <span>Vision (image_url Support)</span>
            <label class="toggle"><input type="checkbox" id="vision_is_vision"><span class="slider"></span></label>
          </div>
          <button class="btn btn-small" onclick="testCategory('vision',1)" style="margin-top:8px">Test Slot 1</button>
          <div class="status-line" id="vision_test_status"></div>
        </div>

        <!-- ── Slot 2: Fallback ── -->
        <hr style="border-color:#30363d;margin:16px 0">
        <div class="model-slot">
          <h4 style="margin:0 0 8px;color:#f0883e;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.5px">Fallback 2</h4>
          <div class="form-group">
            <label>Label</label>
            <input type="text" id="vision_label_2" placeholder="z.B. Gemini Flash Vision">
          </div>
          <div class="form-group">
            <label>API URL</label>
            <input type="url" id="vision_api_url_2" placeholder="https://api.openai.com/v1/chat/completions">
          </div>
          <div class="form-group">
            <label>API Key</label>
            <input type="password" id="vision_api_key_2" placeholder="sk-...">
          </div>
          <div class="form-group">
            <label>Model Name</label>
            <input type="text" id="vision_model_name_2" placeholder="gpt-4o">
          </div>
          <div class="row">
            <div class="form-group">
              <label>Max Tokens</label>
              <input type="number" id="vision_max_tokens_2" min="1" max="256000">
            </div>
            <div class="form-group">
              <label>Timeout (Sekunden)</label>
              <input type="number" id="vision_timeout_seconds_2" min="10" max="3600">
            </div>
          </div>
          <div class="toggle-row">
            <span>max_completion_tokens (statt max_tokens)</span>
            <label class="toggle"><input type="checkbox" id="vision_use_max_completion_tokens_2"><span class="slider"></span></label>
          </div>
          <div class="toggle-row">
            <span>Vision (image_url Support)</span>
            <label class="toggle"><input type="checkbox" id="vision_is_vision_2"><span class="slider"></span></label>
          </div>
          <button class="btn btn-small" onclick="testCategory('vision',2)" style="margin-top:8px">Test Slot 2</button>
          <div class="status-line" id="vision_test_status_2"></div>
        </div>

        <!-- ── Slot 3: Fallback ── -->
        <hr style="border-color:#30363d;margin:16px 0">
        <div class="model-slot">
          <h4 style="margin:0 0 8px;color:#f0883e;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.5px">Fallback 3</h4>
          <div class="form-group">
            <label>Label</label>
            <input type="text" id="vision_label_3" placeholder="z.B. Claude Vision">
          </div>
          <div class="form-group">
            <label>API URL</label>
            <input type="url" id="vision_api_url_3" placeholder="https://api.openai.com/v1/chat/completions">
          </div>
          <div class="form-group">
            <label>API Key</label>
            <input type="password" id="vision_api_key_3" placeholder="sk-...">
          </div>
          <div class="form-group">
            <label>Model Name</label>
            <input type="text" id="vision_model_name_3" placeholder="gpt-4o">
          </div>
          <div class="row">
            <div class="form-group">
              <label>Max Tokens</label>
              <input type="number" id="vision_max_tokens_3" min="1" max="256000">
            </div>
            <div class="form-group">
              <label>Timeout (Sekunden)</label>
              <input type="number" id="vision_timeout_seconds_3" min="10" max="3600">
            </div>
          </div>
          <div class="toggle-row">
            <span>max_completion_tokens (statt max_tokens)</span>
            <label class="toggle"><input type="checkbox" id="vision_use_max_completion_tokens_3"><span class="slider"></span></label>
          </div>
          <div class="toggle-row">
            <span>Vision (image_url Support)</span>
            <label class="toggle"><input type="checkbox" id="vision_is_vision_3"><span class="slider"></span></label>
          </div>
          <button class="btn btn-small" onclick="testCategory('vision',3)" style="margin-top:8px">Test Slot 3</button>
          <div class="status-line" id="vision_test_status_3"></div>
        </div>
      </div>
    </div>
  </section>

  <!-- ====== SEKTION: HINDSIGHT ====== -->
  <section id="section-hindsight">
    <div class="card">
      <h3>Hindsight Memory</h3>
      <div class="toggle-row">
        <span>Hindsight aktiviert</span>
        <label class="toggle"><input type="checkbox" id="hindsight_enabled"><span class="slider"></span></label>
      </div>
      <div class="form-group">
        <label>Qdrant URL</label>
        <input type="text" id="qdrant_url" placeholder="http://localhost:6333">
        <div class="hint">Leer lassen fuer JSONL-Fallback</div>
      </div>
      <div class="form-group">
        <label>Qdrant API Key</label>
        <input type="password" id="qdrant_api_key" placeholder="(optional)">
      </div>
      <div class="toggle-row">
        <span>Qdrant verwenden</span>
        <label class="toggle"><input type="checkbox" id="use_qdrant"><span class="slider"></span></label>
      </div>
      <div class="form-group">
        <label>Collection Name</label>
        <input type="text" id="hindsight_collection" placeholder="hindsight_memory">
      </div>
      <div class="row">
        <div class="form-group">
          <label>Embedding Dimension</label>
          <input type="number" id="embedding_dim" min="64" max="4096">
        </div>
        <div class="form-group">
          <label>Max Memory Tokens</label>
          <input type="number" id="max_memory_tokens" min="100" max="32000">
        </div>
      </div>
      <div class="row">
        <div class="form-group">
          <label>Min Similarity</label>
          <input type="number" id="min_similarity" min="0" max="1" step="0.01">
        </div>
        <div class="form-group">
          <label>Retain Delay (s)</label>
          <input type="number" id="retain_delay_seconds" min="0" max="60" step="0.1">
        </div>
      </div>
      <div class="form-group">
        <label>Speicherverzeichnis</label>
        <input type="text" id="hindsight_dir" placeholder="./.hindsight_memory">
      </div>
    </div>
  </section>

  <!-- ====== SEKTION: PROXY ====== -->
  <section id="section-proxy">
    <div class="card">
      <h3>Proxy-Einstellungen</h3>
      <div class="form-group">
        <label>Port</label>
        <input type="number" id="proxy_port" min="1" max="65535">
      </div>
      <div class="toggle-row">
        <span>Auth aktiviert</span>
        <label class="toggle"><input type="checkbox" id="proxy_auth_enabled"><span class="slider"></span></label>
      </div>
      <div class="form-group">
        <label>API Key</label>
        <input type="text" id="proxy_api_key" placeholder="localfox-... (auto-generiert falls leer)">
      </div>
    </div>
    <div class="card">
      <h3>Token-Schutz</h3>
      <div class="form-group">
        <label>Tool-Result-Cap (Zeichen, 0=aus)</label>
        <input type="number" id="tool_result_cap" min="0" max="1000000">
        <div class="hint">Kappt grosse grep/read-Ergebnisse auf dieses Limit. 0 = deaktiviert.</div>
      </div>
      <div class="form-group">
        <label>Read-Loop-Schwelle (0=aus)</label>
        <input type="number" id="read_loop_threshold" min="0" max="100">
        <div class="hint">Erkennt wiederholtes Lesen derselben Datei/Zeilen. Bei &gt;N Wiederholungen wird eine Intervention injiziert. 0 = deaktiviert.</div>
      </div>
      <div class="form-group">
        <label>Read-Loop-Interventionstext (optional)</label>
        <textarea id="read_loop_intervention" rows="3" placeholder="Leer = Standardtext verwenden. {count} wird durch Anzahl ersetzt."></textarea>
        <div class="hint">Eigener Interventionstext. Platzhalter {count} = Anzahl der Wiederholungen.</div>
      </div>
      <div class="form-group">
        <label>Search-Loop-Schwelle No-Match (0=aus)</label>
        <input type="number" id="search_loop_threshold" min="0" max="100">
        <div class="hint">Erkennt wiederholte erfolglose Suchen (No matches found). Bei &gt;N Wiederholungen wird eine Intervention injiziert. 0 = deaktiviert.</div>
      </div>
      <div class="form-group">
        <label>Search-Loop-Interventionstext (optional)</label>
        <textarea id="search_loop_intervention" rows="3" placeholder="Leer = Standardtext verwenden. {query} und {count} werden ersetzt."></textarea>
        <div class="hint">Eigener Interventionstext. Platzhalter {query} = Suchbegriff, {count} = Anzahl der Wiederholungen.</div>
      </div>
      <div class="form-group">
        <label>Search-Repeat-Schwelle auch mit Treffern (0=aus)</label>
        <input type="number" id="search_repeat_threshold" min="0" max="100">
        <div class="hint">Erkennt wiederholte identische Suchen AUCH wenn Treffer kommen (z.B. currentWeekShifts 10x). 0 = deaktiviert.</div>
      </div>
      <div class="form-group">
        <label>Search-Repeat-Interventionstext (optional)</label>
        <textarea id="search_repeat_intervention" rows="3" placeholder="Leer = Standardtext. {query}/{count}"></textarea>
      </div>
      <div class="form-group">
        <label>Response-Loop-Enforcement (0=aus)</label>
        <input type="number" id="response_loop_threshold" min="0" max="100">
        <div class="hint">Blockiert Tool-Calls in der Modell-Antwort, die denselben Read/Search-Loop fortsetzen. Default 3.</div>
      </div>
      <div class="form-group">
        <label>Generic-Tool-Loop-Schwelle (0=aus)</label>
        <input type="number" id="generic_tool_loop_threshold" min="0" max="100">
        <div class="hint">Erkennt wiederholte identische Aufrufe beliebiger Tools (z.B. manage_todo_list, create_file) mit denselben Argumenten. Default 3. 0 = deaktiviert.</div>
      </div>
      <div class="form-group">
        <label>Generic-Tool-Loop-Interventionstext (optional)</label>
        <textarea id="generic_tool_loop_intervention" rows="3" placeholder="Leer = Standardtext. {tool}/{count}"></textarea>
        <div class="hint">Eigener Interventionstext. Platzhalter {tool} = Tool-Name, {count} = Anzahl Wiederholungen.</div>
      </div>

      <h4 style="margin-top:24px;border-top:1px solid var(--border);padding-top:12px">🧠 Local-Model Sampling (Laguna-S-2.1)</h4>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px">
        <div class="form-group">
          <label>Temperature</label>
          <input type="number" id="local_temperature" step="0.05" min="0" max="2">
        </div>
        <div class="form-group">
          <label>Top-P</label>
          <input type="number" id="local_top_p" step="0.01" min="0" max="1">
        </div>
        <div class="form-group">
          <label>Top-K</label>
          <input type="number" id="local_top_k" min="0" max="200">
        </div>
        <div class="form-group">
          <label>Min-P</label>
          <input type="number" id="local_min_p" step="0.01" min="0" max="1">
        </div>
        <div class="form-group">
          <label>DRY Multiplier</label>
          <input type="number" id="local_dry_multiplier" step="0.1" min="0" max="5">
          <div class="hint">0 = DRY deaktiviert</div>
        </div>
        <div class="form-group">
          <label>DRY Base</label>
          <input type="number" id="local_dry_base" step="0.05" min="1" max="3">
        </div>
        <div class="form-group">
          <label>DRY Allowed Length</label>
          <input type="number" id="local_dry_allowed_length" min="0" max="20">
        </div>
        <div class="form-group">
          <label>DRY Penalty Last N</label>
          <input type="number" id="local_dry_penalty_last_n" min="-1" max="1000">
          <div class="hint">-1 = gesamter Kontext</div>
        </div>
        <div class="form-group">
          <label>DRY Sequence Breaker</label>
          <input type="text" id="local_dry_sequence_breaker">
          <div class="hint">Kommagetrennte Zeichen</div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
        <div class="form-group">
          <label><input type="checkbox" id="local_enable_thinking" checked> Enable Thinking</label>
          <div class="hint">Reasoning in &lt;think&gt;-Tags aktivieren</div>
        </div>
        <div class="form-group">
          <label><input type="checkbox" id="local_preserve_thinking" checked> Preserve Thinking</label>
          <div class="hint">Vorherige &lt;think&gt;-Blöcke in Historie behalten</div>
        </div>
      </div>
      <div class="form-group">
        <label>Anti-Loop-System-Prompt (optional)</label>
        <textarea id="local_anti_loop_system_prompt" rows="4" placeholder="Leer = Standard-Anti-Loop-Prompt"></textarea>
        <div class="hint">Zusätzlicher System-Prompt der dem lokalen Modell Anti-Loop-Regeln vorgibt.</div>
      </div>
    </div>
  </section>

  <!-- ====== SEKTION: LOGS ====== -->
  <section id="section-logs">
    <div class="card" style="padding:12px">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:12px">
        <h3 style="margin:0">Proxy-Logs <span style="font-size:0.75rem;color:var(--text2);font-weight:400" id="logFileLabel"></span></h3>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
          <input type="text" id="logSearch" placeholder="Filter..." style="width:180px;padding:5px 10px;font-size:0.8rem" oninput="refreshLogs()">
          <select id="logLines" style="width:90px;padding:5px 8px;font-size:0.8rem" onchange="refreshLogs()">
            <option value="100">100</option>
            <option value="200" selected>200</option>
            <option value="500">500</option>
            <option value="1000">1000</option>
            <option value="5000">5000</option>
          </select>
          <button class="btn-small" id="logPauseBtn" onclick="toggleLogPause()" title="Auto-Refresh pausieren">⏸</button>
          <button class="btn-small" onclick="refreshLogs()" title="Manuell aktualisieren">🔄</button>
          <button class="btn-small" onclick="copyLogs()" title="In Zwischenablage kopieren">📋</button>
          <span style="font-size:0.75rem;color:var(--text2)" id="logStatus"></span>
        </div>
      </div>
      <pre id="logViewer" style="
        background:var(--bg);border:1px solid var(--border);border-radius:6px;
        padding:10px 14px;margin:0;overflow:auto;max-height:70vh;
        font-family:'Cascadia Code','Fira Code','JetBrains Mono','Consolas',monospace;
        font-size:0.78rem;line-height:1.55;white-space:pre-wrap;word-break:break-all;
        color:var(--text2);
      ">Lade Logs...</pre>
    </div>
  </section>

  <div class="actions">
    <button class="btn btn-primary" onclick="saveConfig()">Konfiguration speichern</button>
    <button class="btn btn-danger" onclick="restartProxy()">Proxy neustarten</button>
  </div>
</main>

<div class="toast" id="toast"></div>

<script>
// ============ NAVIGATION ============
document.querySelectorAll('nav button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const section = btn.dataset.section;
    document.querySelectorAll('main > section').forEach(s => s.classList.remove('active'));
    document.getElementById('section-' + section).classList.add('active');
  });
});

// ============ MODEL-TABS ============
document.querySelectorAll('.model-tabs button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.model-tabs button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const model = btn.dataset.model;
    document.querySelectorAll('.model-card').forEach(c => c.classList.remove('active'));
    document.getElementById('modelCard_' + model).classList.add('active');
  });
});

// ============ TOAST ============
function showToast(msg, type) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast ' + type + ' show';
  setTimeout(() => { t.classList.remove('show'); }, 3000);
}

// ============ LOAD CONFIG ============
async function loadConfig() {
  try {
    const r = await fetch('/webui/api/config');
    const cfg = await r.json();

    // Default category
    document.getElementById('defaultCategory').value = cfg.default_category || 'light';

    // Model categories
    const cats = cfg.model_categories || {};
    for (const key of ['local','coworker','light','strong','vision']) {
      const c = cats[key];
      if (Array.isArray(c)) {
        // Array-Struktur (light/strong/vision): 3 Slots
        for (let i = 0; i < 3; i++) {
          const def = c[i] || {};
          const suffix = i === 0 ? '' : '_' + (i+1);
          const urlEl = document.getElementById(key + '_api_url' + suffix);
          if (urlEl) {
            urlEl.value = def.api_url || '';
            document.getElementById(key + '_api_key' + suffix).value = def.api_key || '';
            document.getElementById(key + '_model_name' + suffix).value = def.model_name || '';
            document.getElementById(key + '_max_tokens' + suffix).value = def.max_tokens || 65536;
            document.getElementById(key + '_timeout_seconds' + suffix).value = def.timeout_seconds || 180;
            document.getElementById(key + '_use_max_completion_tokens' + suffix).checked = !!def.use_max_completion_tokens;
            document.getElementById(key + '_is_vision' + suffix).checked = !!def.is_vision;
            // Label ist optional
            const labelEl = document.getElementById(key + '_label_' + (i+1));
            if (labelEl) labelEl.value = def.label || '';
          }
        }
      } else if (c) {
        // Single-Def (local) oder legacy
        document.getElementById(key + '_api_url').value = c.api_url || '';
        document.getElementById(key + '_api_key').value = c.api_key || '';
        document.getElementById(key + '_model_name').value = c.model_name || '';
        document.getElementById(key + '_max_tokens').value = c.max_tokens || 65536;
        document.getElementById(key + '_timeout_seconds').value = c.timeout_seconds || 180;
        const rtEl = document.getElementById(key + '_read_timeout_seconds');
        if (rtEl) rtEl.value = c.read_timeout_seconds || c.timeout_seconds || 120;
        const retryEl = document.getElementById(key + '_retry_on_timeout');
        if (retryEl) retryEl.value = c.retry_on_timeout != null ? c.retry_on_timeout : 2;
        const retryDelayEl = document.getElementById(key + '_retry_delay_seconds');
        if (retryDelayEl) retryDelayEl.value = c.retry_delay_seconds || 5;
        document.getElementById(key + '_use_max_completion_tokens').checked = !!c.use_max_completion_tokens;
        document.getElementById(key + '_is_vision').checked = !!c.is_vision;
        const labelEl = document.getElementById(key + '_label_1');
        if (labelEl) labelEl.value = c.label || key;
      }
    }

    // Hindsight
    const hs = cfg.hindsight || {};
    document.getElementById('hindsight_enabled').checked = hs.enabled !== false;
    document.getElementById('qdrant_url').value = hs.qdrant_url || '';
    document.getElementById('qdrant_api_key').value = hs.qdrant_api_key || '';
    document.getElementById('use_qdrant').checked = !!hs.use_qdrant;
    document.getElementById('hindsight_collection').value = hs.collection || 'hindsight_memory';
    document.getElementById('embedding_dim').value = hs.embedding_dim || 768;
    document.getElementById('max_memory_tokens').value = hs.max_memory_tokens || 4000;
    document.getElementById('min_similarity').value = hs.min_similarity || 0.18;
    document.getElementById('retain_delay_seconds').value = hs.retain_delay_seconds || 0;
    document.getElementById('hindsight_dir').value = hs.dir || './.hindsight_memory';

    // Proxy
    const px = cfg.proxy || {};
    document.getElementById('proxy_port').value = px.port || 9001;
    document.getElementById('proxy_auth_enabled').checked = px.auth_enabled !== false;
    document.getElementById('proxy_api_key').value = px.api_key || '';

    // Tokens
    const tk = cfg.tokens || {};
    document.getElementById('tool_result_cap').value = tk.tool_result_cap || 0;
    document.getElementById('read_loop_threshold').value = tk.read_loop_threshold != null ? tk.read_loop_threshold : 3;
    document.getElementById('read_loop_intervention').value = tk.read_loop_intervention || '';
    document.getElementById('search_loop_threshold').value = tk.search_loop_threshold != null ? tk.search_loop_threshold : 3;
    document.getElementById('search_loop_intervention').value = tk.search_loop_intervention || '';
    document.getElementById('search_repeat_threshold').value = tk.search_repeat_threshold != null ? tk.search_repeat_threshold : 3;
    document.getElementById('search_repeat_intervention').value = tk.search_repeat_intervention || '';
    document.getElementById('response_loop_threshold').value = tk.response_loop_threshold != null ? tk.response_loop_threshold : 3;
    document.getElementById('generic_tool_loop_threshold').value = tk.generic_tool_loop_threshold != null ? tk.generic_tool_loop_threshold : 3;
    document.getElementById('generic_tool_loop_intervention').value = tk.generic_tool_loop_intervention || '';
    const ls = tk.local_sampling || {};
    document.getElementById('local_temperature').value = ls.temperature != null ? ls.temperature : 0.7;
    document.getElementById('local_top_p').value = ls.top_p != null ? ls.top_p : 0.95;
    document.getElementById('local_top_k').value = ls.top_k != null ? ls.top_k : 20;
    document.getElementById('local_min_p').value = ls.min_p != null ? ls.min_p : 0.0;
    document.getElementById('local_dry_multiplier').value = ls.dry_multiplier != null ? ls.dry_multiplier : 0.8;
    document.getElementById('local_dry_base').value = ls.dry_base != null ? ls.dry_base : 1.75;
    document.getElementById('local_dry_allowed_length').value = ls.dry_allowed_length != null ? ls.dry_allowed_length : 3;
    document.getElementById('local_dry_penalty_last_n').value = ls.dry_penalty_last_n != null ? ls.dry_penalty_last_n : -1;
    document.getElementById('local_dry_sequence_breaker').value = ls.dry_sequence_breaker || '\\n,:,",*,;,{,}';
    document.getElementById('local_enable_thinking').checked = ls.enable_thinking !== false;
    document.getElementById('local_preserve_thinking').checked = ls.preserve_thinking !== false;
    document.getElementById('local_anti_loop_system_prompt').value = ls.anti_loop_system_prompt || '';

    // Co-Worker-Delegation Settings
    const cw = tk.coworker || {};
    const cwEnabledEl = document.getElementById('coworker_enabled');
    if (cwEnabledEl) cwEnabledEl.checked = cw.enabled !== false;
    const cwMaxEl = document.getElementById('coworker_max_delegations');
    if (cwMaxEl) cwMaxEl.value = cw.max_delegations_per_request != null ? cw.max_delegations_per_request : 2;
    const cwTaskEl = document.getElementById('coworker_task_cap');
    if (cwTaskEl) cwTaskEl.value = cw.task_cap_chars != null ? cw.task_cap_chars : 8000;
    const cwResultEl = document.getElementById('coworker_result_cap');
    if (cwResultEl) cwResultEl.value = cw.result_cap_chars != null ? cw.result_cap_chars : 12000;
    const cwFilesEl = document.getElementById('coworker_files_cap');
    if (cwFilesEl) cwFilesEl.value = cw.files_cap_chars != null ? cw.files_cap_chars : 60000;
    const cwHIntEl = document.getElementById('coworker_health_interval');
    if (cwHIntEl) cwHIntEl.value = cw.health_interval_seconds != null ? cw.health_interval_seconds : 60;
    const cwPTOEl = document.getElementById('coworker_probe_timeout');
    if (cwPTOEl) cwPTOEl.value = cw.probe_timeout_seconds != null ? cw.probe_timeout_seconds : 5;
    const cwFJEl = document.getElementById('coworker_fork_join');
    if (cwFJEl) cwFJEl.checked = cw.enable_fork_join !== false;
    const cwMPEl = document.getElementById('coworker_max_parallel');
    if (cwMPEl) cwMPEl.value = cw.max_parallel != null ? cw.max_parallel : 8;
    const cwDCEl = document.getElementById('coworker_dispatch_cap');
    if (cwDCEl) cwDCEl.value = cw.dispatch_cap_per_request != null ? cw.dispatch_cap_per_request : 12;
    const cwTTLEl = document.getElementById('coworker_bg_ttl');
    if (cwTTLEl) cwTTLEl.value = cw.bg_ttl_seconds != null ? cw.bg_ttl_seconds : 1800;
    const cwTDEl = document.getElementById('coworker_teach_delegation');
    if (cwTDEl) cwTDEl.checked = cw.teach_delegation !== false;
    const cwSPEl = document.getElementById('coworker_system_prompt');
    if (cwSPEl) cwSPEl.value = cw.system_prompt || '';

  } catch(e) {
    showToast('Fehler beim Laden: ' + e.message, 'error');
  }
}

// ============ SAVE CONFIG ============
async function saveConfig() {
  const cfg = {};

  cfg.default_category = document.getElementById('defaultCategory').value;

  cfg.model_categories = {};
  for (const key of ['local','coworker','light','strong','vision']) {
    if (key === 'local' || key === 'coworker') {
      // local & coworker bleiben Single-Def Dict
      cfg.model_categories[key] = {
        label: document.getElementById(key + '_label_1')?.value || key,
        api_url: document.getElementById(key + '_api_url').value,
        api_key: document.getElementById(key + '_api_key').value,
        retry_on_timeout: parseInt(document.getElementById(key + '_retry_on_timeout')?.value) || 0,
        retry_delay_seconds: parseFloat(document.getElementById(key + '_retry_delay_seconds')?.value) || 5,
        model_name: document.getElementById(key + '_model_name').value,
        max_tokens: parseInt(document.getElementById(key + '_max_tokens').value) || 65536,
        timeout_seconds: parseFloat(document.getElementById(key + '_timeout_seconds').value) || 180,
        read_timeout_seconds: parseFloat(document.getElementById(key + '_read_timeout_seconds')?.value) || 120,
        use_max_completion_tokens: document.getElementById(key + '_use_max_completion_tokens').checked,
        is_vision: document.getElementById(key + '_is_vision').checked,
      };
    } else {
      // Array mit 3 Slots
      const arr = [];
      for (let i = 0; i < 3; i++) {
        const suffix = i === 0 ? '' : '_' + (i+1);
        arr.push({
          label: document.getElementById(key + '_label_' + (i+1))?.value || (i === 0 ? key + ' primary' : key + ' fallback ' + (i+1)),
          api_url: document.getElementById(key + '_api_url' + suffix).value,
          api_key: document.getElementById(key + '_api_key' + suffix).value,
          model_name: document.getElementById(key + '_model_name' + suffix).value,
          max_tokens: parseInt(document.getElementById(key + '_max_tokens' + suffix).value) || 65536,
          timeout_seconds: parseFloat(document.getElementById(key + '_timeout_seconds' + suffix).value) || 180,
          use_max_completion_tokens: document.getElementById(key + '_use_max_completion_tokens' + suffix).checked,
          is_vision: document.getElementById(key + '_is_vision' + suffix).checked,
        });
      }
      cfg.model_categories[key] = arr;
    }
  }

  cfg.hindsight = {
    enabled: document.getElementById('hindsight_enabled').checked,
    qdrant_url: document.getElementById('qdrant_url').value,
    qdrant_api_key: document.getElementById('qdrant_api_key').value,
    use_qdrant: document.getElementById('use_qdrant').checked,
    collection: document.getElementById('hindsight_collection').value,
    embedding_dim: parseInt(document.getElementById('embedding_dim').value) || 768,
    max_memory_tokens: parseInt(document.getElementById('max_memory_tokens').value) || 4000,
    min_similarity: parseFloat(document.getElementById('min_similarity').value) || 0.18,
    retain_delay_seconds: parseFloat(document.getElementById('retain_delay_seconds').value) || 0,
    dir: document.getElementById('hindsight_dir').value,
  };

  cfg.proxy = {
    port: parseInt(document.getElementById('proxy_port').value) || 9001,
    auth_enabled: document.getElementById('proxy_auth_enabled').checked,
    api_key: document.getElementById('proxy_api_key').value,
  };

  cfg.tokens = {
    tool_result_cap: parseInt(document.getElementById('tool_result_cap').value) || 0,
    read_loop_threshold: parseInt(document.getElementById('read_loop_threshold').value) || 0,
    read_loop_intervention: document.getElementById('read_loop_intervention').value || '',
    search_loop_threshold: parseInt(document.getElementById('search_loop_threshold').value) || 0,
    search_loop_intervention: document.getElementById('search_loop_intervention').value || '',
    search_repeat_threshold: parseInt(document.getElementById('search_repeat_threshold').value) || 0,
    search_repeat_intervention: document.getElementById('search_repeat_intervention').value || '',
    response_loop_threshold: parseInt(document.getElementById('response_loop_threshold').value) || 0,
    generic_tool_loop_threshold: parseInt(document.getElementById('generic_tool_loop_threshold').value) || 0,
    generic_tool_loop_intervention: document.getElementById('generic_tool_loop_intervention').value || '',
    coworker: {
      enabled: document.getElementById('coworker_enabled')?.checked ?? true,
      max_delegations_per_request: parseInt(document.getElementById('coworker_max_delegations')?.value) || 2,
      task_cap_chars: parseInt(document.getElementById('coworker_task_cap')?.value) || 8000,
      result_cap_chars: parseInt(document.getElementById('coworker_result_cap')?.value) || 12000,
      files_cap_chars: parseInt(document.getElementById('coworker_files_cap')?.value) ?? 60000,
      health_interval_seconds: parseFloat(document.getElementById('coworker_health_interval')?.value) || 60,
      probe_timeout_seconds: parseFloat(document.getElementById('coworker_probe_timeout')?.value) || 5,
      enable_fork_join: document.getElementById('coworker_fork_join')?.checked ?? true,
      max_parallel: parseInt(document.getElementById('coworker_max_parallel')?.value) || 8,
      dispatch_cap_per_request: parseInt(document.getElementById('coworker_dispatch_cap')?.value) || 12,
      bg_ttl_seconds: parseFloat(document.getElementById('coworker_bg_ttl')?.value) || 1800,
      teach_delegation: document.getElementById('coworker_teach_delegation')?.checked ?? true,
      system_prompt: document.getElementById('coworker_system_prompt')?.value || '',
    },
    local_sampling: {
      temperature: parseFloat(document.getElementById('local_temperature').value) || 0.7,
      top_p: parseFloat(document.getElementById('local_top_p').value) || 0.95,
      top_k: parseInt(document.getElementById('local_top_k').value) || 20,
      min_p: parseFloat(document.getElementById('local_min_p').value) || 0.0,
      dry_multiplier: parseFloat(document.getElementById('local_dry_multiplier').value) || 0,
      dry_base: parseFloat(document.getElementById('local_dry_base').value) || 1.75,
      dry_allowed_length: parseInt(document.getElementById('local_dry_allowed_length').value) || 3,
      dry_penalty_last_n: parseInt(document.getElementById('local_dry_penalty_last_n').value),
      dry_sequence_breaker: document.getElementById('local_dry_sequence_breaker').value || '',
      enable_thinking: document.getElementById('local_enable_thinking').checked,
      preserve_thinking: document.getElementById('local_preserve_thinking').checked,
      anti_loop_system_prompt: document.getElementById('local_anti_loop_system_prompt').value || '',
    },
  };

  try {
    const r = await fetch('/webui/api/config', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(cfg),
    });
    if (!r.ok) { const err = await r.json(); throw new Error(err.detail || 'Fehler'); }
    showToast('Konfiguration gespeichert', 'success');
  } catch(e) {
    showToast('Fehler: ' + e.message, 'error');
  }
}

// ============ RESTART PROXY ============
async function restartProxy() {
  if (!confirm('Proxy wirklich neustarten? Laufende Requests werden abgebrochen.')) return;
  try {
    await fetch('/webui/api/restart', {method:'POST'});
    showToast('Neustart eingeleitet...', 'success');
  } catch(e) {
    showToast('Neustart-Fehler: ' + e.message, 'error');
  }
}

// ============ TEST ENDPOINT ============
async function testCategory(key, slot) {
  slot = slot || 1;
  const suffix = slot === 1 ? '' : '_' + slot;
  const apiUrl = document.getElementById(key + '_api_url' + suffix).value;
  const apiKey = document.getElementById(key + '_api_key' + suffix).value;
  const modelName = document.getElementById(key + '_model_name' + suffix).value;
  const statusEl = document.getElementById(key + '_test_status' + suffix);
  if (!statusEl) return;

  if (!apiUrl || !modelName) {
    statusEl.innerHTML = '<span class="status-dot err"></span> URL und Model-Name erforderlich';
    return;
  }

  statusEl.innerHTML = '<span class="status-dot" style="background:var(--warn)"></span> Teste...';

  try {
    const r = await fetch('/webui/api/test-endpoint', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({api_url: apiUrl, api_key: apiKey, model_name: modelName}),
    });
    const data = await r.json();
    if (data.ok) {
      statusEl.innerHTML = '<span class="status-dot ok"></span> OK (' + data.duration + ')';
    } else {
      statusEl.innerHTML = '<span class="status-dot err"></span> ' + (data.error || 'Unbekannter Fehler');
    }
  } catch(e) {
    statusEl.innerHTML = '<span class="status-dot err"></span> ' + e.message;
  }
}

// ============ LOG VIEWER ============
let _logAutoRefresh = true;
let _logTimer = null;
let _logScrolledToBottom = true;

function toggleLogPause() {
  _logAutoRefresh = !_logAutoRefresh;
  const btn = document.getElementById('logPauseBtn');
  btn.textContent = _logAutoRefresh ? '\u23F8' : '\u25B6';
  btn.title = _logAutoRefresh ? 'Auto-Refresh pausieren' : 'Auto-Refresh fortsetzen';
  if (_logAutoRefresh) { refreshLogs(); }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function colorizeLogLine(line) {
  // Farbcodierung fuer verschiedene Log-Level und Keywords
  let cls = '';
  const lower = line.toLowerCase();
  if (/\b(error|fail|exception|traceback|fehlgeschlagen|ungültig)\b/i.test(line)) {
    cls = 'log-error';
  } else if (/\b(warn|warning|achtung|warnung)\b/i.test(line)) {
    cls = 'log-warn';
  } else if (/\b(success|ok\b|erfolg|bestanden)\b/i.test(line)) {
    cls = 'log-ok';
  } else if (/\b(model ok|fallback success|auth-ok)\b/i.test(line)) {
    cls = 'log-ok';
  } else if (/\b(auth-fail|401|403|429|5\d\d)\b/i.test(line)) {
    cls = 'log-error';
  } else if (/\b\[webui\]/i.test(line)) {
    cls = 'log-webui';
  } else if (/\b(req-in|req-out)\b/i.test(line)) {
    cls = 'log-req';
  }
  return cls;
}

async function refreshLogs() {
  const viewer = document.getElementById('logViewer');
  const statusEl = document.getElementById('logStatus');
  const lines = document.getElementById('logLines').value;
  const search = document.getElementById('logSearch').value;

  // Scroll-Position merken
  const wasAtBottom = viewer.scrollHeight - viewer.scrollTop - viewer.clientHeight < 40;
  _logScrolledToBottom = wasAtBottom;

  try {
    statusEl.textContent = 'Lade...';
    const params = new URLSearchParams({lines: lines});
    if (search) params.set('search', search);
    const r = await fetch('/webui/api/logs?' + params.toString());
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();

    if (data.error) {
      viewer.innerHTML = '<span style="color:var(--danger)">Fehler: ' + escapeHtml(data.error) + '</span>';
      statusEl.textContent = 'Fehler';
      return;
    }

    document.getElementById('logFileLabel').textContent = data.file ? ' — ' + data.file.split('/').pop().split('\\').pop() : '';

    if (data.lines.length === 0) {
      viewer.innerHTML = '<span style="color:var(--text2);font-style:italic">(keine Logs' + (search ? ' fuer Filter "' + escapeHtml(search) + '"' : '') + ')</span>';
      statusEl.textContent = '0 Zeilen';
      return;
    }

    // Zeilen mit Farbcodierung rendern
    let html = '';
    for (const line of data.lines) {
      const cls = colorizeLogLine(line);
      const escaped = escapeHtml(line);
      if (cls) {
        html += '<span class="' + cls + '">' + escaped + '</span>\n';
      } else {
        html += escaped + '\n';
      }
    }
    viewer.innerHTML = html;

    // Auto-scroll wenn vorher am Ende
    if (_logScrolledToBottom) {
      viewer.scrollTop = viewer.scrollHeight;
    }

    const now = new Date();
    const timeStr = now.toLocaleTimeString('de-DE', {hour:'2-digit', minute:'2-digit', second:'2-digit'});
    statusEl.textContent = data.total + ' Zeilen @ ' + timeStr;
  } catch(e) {
    viewer.innerHTML = '<span style="color:var(--danger)">Fehler: ' + escapeHtml(e.message) + '</span>';
    statusEl.textContent = 'Fehler';
  }

  // Auto-Refresh Timer
  if (_logTimer) clearTimeout(_logTimer);
  if (_logAutoRefresh) {
    _logTimer = setTimeout(refreshLogs, 3000);
  }
}

async function copyLogs() {
  const viewer = document.getElementById('logViewer');
  try {
    await navigator.clipboard.writeText(viewer.textContent);
    showToast('Logs in Zwischenablage kopiert', 'success');
  } catch(e) {
    // Fallback: Selection
    const range = document.createRange();
    range.selectNodeContents(viewer);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
    showToast('Logs selektiert — bitte Strg+C druecken', 'success');
  }
}

// Tab-Klick triggern
document.querySelectorAll('nav button').forEach(btn => {
  btn.addEventListener('click', () => {
    setTimeout(() => {
      const logsSection = document.getElementById('section-logs');
      if (logsSection && logsSection.classList.contains('active')) {
        if (!_logTimer) refreshLogs();
      } else {
        if (_logTimer) { clearTimeout(_logTimer); _logTimer = null; }
      }
    }, 50);
  });
});

// ============ PASSWORD TOGGLE ============
function togglePw(btn) {
  const input = btn.parentElement.querySelector('input');
  const show = input.type === 'password';
  input.type = show ? 'text' : 'password';
  btn.textContent = show ? '🙈' : '👁';
}

// ============ INIT ============
loadConfig();
// Fuege Show/Hide-Buttons zu allen Passwortfeldern hinzu
document.querySelectorAll('input[type=password]').forEach(function(inp) {
  var wrapper = document.createElement('span');
  wrapper.className = 'pw-wrapper';
  inp.parentNode.insertBefore(wrapper, inp);
  wrapper.appendChild(inp);
  var btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'pw-toggle';
  btn.textContent = '👁';
  btn.onclick = function() { togglePw(btn); };
  wrapper.appendChild(btn);
});
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════════════════

def _reload_proxy_config() -> None:
    try:
        from proxy import _apply_config_file
        _apply_config_file()
        _log("Proxy-Config aus config.json neu geladen (in-process)")
    except Exception as e:
        _log(f"Proxy-Config-Reload fehlgeschlagen: {e}")


def _trigger_restart() -> None:
    _log("Neustart angefordert via WebUI")
    import threading as _th
    import subprocess as _sp

    def _do_restart():
        time.sleep(0.5)
        try:
            result = _sp.run(["systemctl", "restart", "localproxy"], capture_output=True, timeout=5)
            if result.returncode == 0:
                _log("systemctl restart localproxy ausgefuehrt")
                return
        except Exception as e:
            _log(f"systemctl nicht verfuegbar ({e}), versuche SIGTERM...")
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception as e:
            _log(f"SIGTERM fehlgeschlagen: {e}")

    _th.Thread(target=_do_restart, daemon=True).start()


def create_webui_app() -> FastAPI:
    webapp = FastAPI(docs_url=None, openapi_url=None, redoc_url=None)

    @webapp.middleware("http")
    async def _auth_middleware(request: Request, call_next):
        if request.url.path in ("/webui/login", "/webui/api/login", "/webui/api/logout"):
            return await call_next(request)
        token = request.cookies.get(COOKIE_NAME) or request.query_params.get("token", "")
        if not token or not _validate_token(token):
            return RedirectResponse(url="/webui/login", status_code=302)
        return await call_next(request)

    @webapp.get("/login", response_class=HTMLResponse)
    async def login_page():
        return LOGIN_HTML

    @webapp.post("/api/login")
    async def api_login(request: Request, response: Response):
        try:
            data = await request.json()
        except Exception:
            raise HTTPException(status_code=400)
        if data.get("username") != WEBUI_USERNAME or data.get("password") != WEBUI_PASSWORD:
            raise HTTPException(status_code=401, detail="Ungueltige Anmeldedaten")
        token = _generate_token()
        return {"token": token, "redirect": "/webui/"}

    @webapp.post("/api/logout")
    async def api_logout(request: Request):
        token = request.cookies.get(COOKIE_NAME) or ""
        if token:
            _remove_token(token)
        return {"status": "ok"}

    @webapp.get("/", response_class=HTMLResponse)
    async def dashboard():
        return DASHBOARD_HTML

    @webapp.get("/api/config")
    async def get_config():
        cfg = _load_config()
        # API keys maskieren
        for key in ("local", "coworker", "light", "strong", "vision"):
            cat = cfg.get("model_categories", {}).get(key)
            if isinstance(cat, list):
                for d in cat:
                    if isinstance(d, dict) and d.get("api_key"):
                        d["api_key"] = _mask_key(d["api_key"])
            elif isinstance(cat, dict) and cat.get("api_key"):
                cat["api_key"] = _mask_key(cat["api_key"])
        return JSONResponse(content=cfg)

    @webapp.put("/api/config")
    async def put_config(request: Request):
        try:
            new_cfg = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Ungueltiges JSON")

        current = _load_config()

        # API-Keys: wenn maskiert (enthaelt bullet), alten Wert behalten
        for key in ("local", "coworker", "light", "strong", "vision"):
            nc = new_cfg.get("model_categories", {}).get(key)
            cc = current.get("model_categories", {}).get(key)
            if nc is None or cc is None:
                continue
            if isinstance(nc, list) and isinstance(cc, list):
                for i in range(min(len(nc), len(cc))):
                    nd = nc[i] if isinstance(nc[i], dict) else {}
                    cd = cc[i] if isinstance(cc[i], dict) else {}
                    if nd and cd and "\u2022" in str(nd.get("api_key", "")):
                        nd["api_key"] = cd.get("api_key", "")
            elif isinstance(nc, dict) and isinstance(cc, dict):
                if "\u2022" in str(nc.get("api_key", "")):
                    nc["api_key"] = cc.get("api_key", "")

        _deep_merge(current, new_cfg)
        _save_config(current)
        _reload_proxy_config()
        return JSONResponse(content={"status": "ok", "message": "Config gespeichert + Proxy-Config neu geladen"})

    @webapp.post("/api/restart")
    async def api_restart():
        _trigger_restart()
        return JSONResponse(content={"status": "ok", "message": "Neustart eingeleitet"})

    @webapp.post("/api/test-endpoint")
    async def api_test_endpoint(request: Request):
        try:
            data = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Ungueltiges JSON")

        api_url = str(data.get("api_url", "")).strip()
        api_key = str(data.get("api_key", "")).strip()
        model_name = str(data.get("model_name", "")).strip()

        if not api_url or not model_name:
            return JSONResponse(content={"ok": False, "error": "api_url und model_name erforderlich"})

        # Maskierten Key erkennen: wenn Bullets (U+2022) enthalten sind,
        # wurde der Key nicht geaendert → echten Key aus Config nachladen
        if "\u2022" in api_key and len(api_key) > 8:
            cfg = _load_config()
            for cat_key in ("local", "coworker", "light", "strong", "vision"):
                cat = cfg.get("model_categories", {}).get(cat_key)
                if isinstance(cat, list):
                    for d in cat:
                        if isinstance(d, dict) and d.get("api_url", "").strip() == api_url:
                            api_key = d.get("api_key", "")
                            break
                elif isinstance(cat, dict) and cat.get("api_url", "").strip() == api_url:
                    api_key = cat.get("api_key", "")
                    break

        # Key fuer Logging trunktieren — NUR ASCII-Safe
        try:
            _trunc = api_key[:8] + "..."
            _trunc.encode("ascii")
        except (UnicodeEncodeError, UnicodeDecodeError):
            _trunc = "sk-****..."

        _log(f"Test-Endpoint: model={model_name} api_key={_trunc} url={api_url}")

        # Auto-Erkennung ob Model max_completion_tokens braucht (o1-, o3-, o4-, gpt-4.1, gpt-4.5, etc.)
        _needs_mct = bool(re.match(r'^(o[1349]|o[1349]-|o-series|gpt-4\.(?:1|5|o|.5)|gpt-5)', model_name, re.IGNORECASE))

        import httpx
        started = time.perf_counter()
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            # Versuche zuerst den korrekten Parameter
            payload: Dict[str, Any] = {
                "model": model_name,
                "messages": [{"role": "user", "content": "ping"}],
            }
            if _needs_mct:
                payload["max_completion_tokens"] = 1
            else:
                payload["max_tokens"] = 1

            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as hc:
                r = await hc.post(api_url, json=payload, headers=headers)
            duration = f"{time.perf_counter() - started:.1f}s"

            # Bei 400: ggf. mit alternativem Parameter wiederholen
            if r.status_code == 400:
                err_body_text = ""
                try:
                    err_body = r.json()
                    if isinstance(err_body.get("error"), dict):
                        err_body_text = str(err_body["error"].get("message", ""))
                    elif isinstance(err_body.get("error"), str):
                        err_body_text = str(err_body["error"])
                except Exception:
                    pass
                # Wenn Fehler auf max_tokens hindeutet, mit max_completion_tokens versuchen
                if "max_completion_tokens" in err_body_text.lower() or "max_tokens" in err_body_text.lower():
                    _log(f"Test-Endpoint 400: {err_body_text[:200]}, retry mit max_completion_tokens")
                    payload2: Dict[str, Any] = {
                        "model": model_name,
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_completion_tokens": 1,
                    }
                    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as hc2:
                        r2 = await hc2.post(api_url, json=payload2, headers=headers)
                    if r2.status_code in (200, 201):
                        duration2 = f"{time.perf_counter() - started:.1f}s"
                        return JSONResponse(content={"ok": True, "duration": duration2,
                                                       "note": "max_completion_tokens statt max_tokens"})
                    else:
                        try:
                            err_body2 = r2.json()
                            if isinstance(err_body2.get("error"), dict):
                                err = _safe_str(err_body2["error"].get("message", ""))
                            elif isinstance(err_body2.get("error"), str):
                                err = _safe_str(err_body2["error"])
                            else:
                                err = f"HTTP {r2.status_code}"
                        except Exception:
                            err = f"HTTP {r2.status_code}"
                        return JSONResponse(content={"ok": False, "duration": f"{time.perf_counter() - started:.1f}s",
                                                       "error": err or f"HTTP {r2.status_code}"})

            if r.status_code in (200, 201):
                return JSONResponse(content={"ok": True, "duration": duration})
            err = ""
            try:
                body = r.json()
                if isinstance(body.get("error"), dict):
                    err = _safe_str(body["error"].get("message", ""))
                elif isinstance(body.get("error"), str):
                    err = _safe_str(body["error"])
            except Exception:
                err = f"HTTP {r.status_code}"
            return JSONResponse(content={"ok": False, "duration": duration, "error": err or f"HTTP {r.status_code}"})
        except Exception as exc:
            duration = f"{time.perf_counter() - started:.1f}s"
            return JSONResponse(content={"ok": False, "duration": duration, "error": _safe_str(exc)})

    @webapp.get("/api/logs")
    async def api_logs(request: Request, lines: int = 200, search: str = ""):
        """Gibt die letzten N Zeilen aus der Logdatei zurueck.
        Optional: search filtert Zeilen (case-insensitive substring).
        """
        max_lines = min(max(lines, 10), 5000)
        try:
            log_path = Path(LOG_FILE)
            if not log_path.exists():
                return JSONResponse(content={"lines": [], "total": 0, "file": str(log_path), "error": "Logdatei nicht gefunden"})

            # Dateigroesse checken, nur das Ende lesen
            file_size = log_path.stat().st_size
            # Rough estimate: ~200 bytes per line, read enough to cover max_lines * 2
            read_size = min(file_size, max_lines * 400 + 8192)
            with open(log_path, "rb") as f:
                if file_size > read_size:
                    f.seek(file_size - read_size)
                raw = f.read()

            # Decode mit UTF-8 Fallback
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("utf-8", errors="replace")

            all_lines = text.splitlines()
            # Nur die letzten max_lines behalten
            if len(all_lines) > max_lines:
                all_lines = all_lines[-max_lines:]

            # Optional: Suche/Filter
            if search:
                search_lower = search.lower()
                all_lines = [l for l in all_lines if search_lower in l.lower()]

            return JSONResponse(content={
                "lines": all_lines,
                "total": len(all_lines),
                "file": str(log_path),
            })
        except Exception as e:
            return JSONResponse(content={"lines": [], "total": 0, "file": str(LOG_FILE), "error": _safe_str(e)})

    return webapp


def mount_webui(parent_app: FastAPI) -> None:
    webapp = create_webui_app()
    parent_app.mount("/webui", webapp)
