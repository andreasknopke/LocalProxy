"""
Web-Konfigurationsinterface für LocalProxy v2.0
────────────────────────────────────────────────
Modernes Single-Page-Dashboard zum Konfigurieren aller Proxy-Einstellungen:
Modelle, Cloud-APIs, Tokens, Memory, Verifikation.
"""

from __future__ import annotations

import json
import os
import secrets
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

# ═══════════════════════════════════════════════════════════════════════════
# Config-Datei
# ═══════════════════════════════════════════════════════════════════════════

CONFIG_PATH = Path(os.getenv("LOCALPROXY_CONFIG", "data/config.json"))
LOG_FILE = os.getenv("LOG_FILE", str(Path(__file__).parent / "proxy.log"))


def _log(msg: str) -> None:
    """Schreibt eine Log-Zeile mit Timestamp ins selbe Log-File wie proxy.py."""
    import datetime as _dt
    timestamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [webui] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════════════════
# WebUI Auth — Zwangs-Login via SPARK_AUTH_USERNAME / SPARK_AUTH_PASSWORD
# ═══════════════════════════════════════════════════════════════════════════

WEBUI_USERNAME: str = os.getenv("SPARK_AUTH_USERNAME", "admin")
WEBUI_PASSWORD: str = os.getenv("SPARK_AUTH_PASSWORD", "")
if not WEBUI_PASSWORD:
    WEBUI_PASSWORD = "localfox-" + secrets.token_hex(16)
    _log(f"⚡ WebUI Auto-Passwort (kein SPARK_AUTH_PASSWORD gesetzt): {WEBUI_PASSWORD}")
else:
    _log("🔐 WebUI Login via SPARK_AUTH_USERNAME / SPARK_AUTH_PASSWORD")

# In-Memory Session-Tokens
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


# ── Login-HTML ────────────────────────────────────────────────────────────

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
  <h1>🦊 LocalProxy <span class="badge">v2.0</span></h1>
  <p>Bitte anmelden um auf das Dashboard zuzugreifen</p>
  <form id="loginForm">
    <div class="form-group">
      <label for="username">Benutzername</label>
      <input type="text" id="username" name="username" placeholder="admin" autocomplete="username" autofocus>
    </div>
    <div class="form-group">
      <label for="password">Passwort</label>
      <input type="password" id="password" name="password" placeholder="••••••••" autocomplete="current-password">
    </div>
    <div class="error" id="loginError">Falscher Benutzername oder Passwort</div>
    <button type="submit" class="btn">🔐 Anmelden</button>
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
    // Token im Cookie speichern + als Query-Parameter (Fallback)
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
    "models": {
        "vllm_api_url": "http://localhost:8000/v1/chat/completions",
        "vllm_models_url": "http://localhost:8000/v1/models",
        "vllm_api_key": "",
        "model_name": "Qwen/Qwen3-Next-80B-Chat-mxfp4",
        "fast_model_name": "Qwen/Qwen3.6-27B-Chat-FP8",
    },
    "cloud": {
        "enabled": False,
        "api_url": "https://api.openai.com/v1/chat/completions",
        "api_key": "",
        "model": "gpt-4.1-mini",
        "max_tokens": 128000,
        "timeout_seconds": 180,
    },
    "litellm": {
        "model": "",
        "api_key": "",
        "api_url": "",
        "max_tokens": 16384,
        "timeout_seconds": 180,
    },
    "proxy": {
        "port": 9001,
        "auth_enabled": True,
        "api_key": "",
        "chatty_mode": True,
    },
    "tokens": {
        "direct_max_tokens": 32768,
        "agent_max_tokens": 65536,
        "caveman_max_tokens": 8192,
        "sub_agent_timeout_seconds": 120,
        "verify_timeout_seconds": 120,
    },
    "caveman": {
        "enabled": True,
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
    "verify": {
        "enabled": True,
        "lint_command": "",
        "test_command": "",
    },
    "mcp": {
        "enabled": True,
    },
}


# Mapping: Env-Variable → (config-Sektion, config-Key)
# Wird NUR beim erstmaligen Erzeugen von config.json verwendet.
_ENV_TO_CONFIG: Dict[str, Tuple[str, str]] = {
    "VLLM_API_URL": ("models", "vllm_api_url"),
    "VLLM_MODELS_URL": ("models", "vllm_models_url"),
    "VLLM_API_KEY": ("models", "vllm_api_key"),
    "MODEL_NAME": ("models", "model_name"),
    "FAST_MODEL_NAME": ("models", "fast_model_name"),
    "PROXY_PORT": ("proxy", "port"),
    "PROXY_AUTH_ENABLED": ("proxy", "auth_enabled"),
    "PROXY_API_KEY": ("proxy", "api_key"),
    "CHATTY_MODE": ("proxy", "chatty_mode"),
    "CLOUD_REVIEW_ENABLED": ("cloud", "enabled"),
    "CLOUD_REVIEW_API_URL": ("cloud", "api_url"),
    "CLOUD_REVIEW_API_KEY": ("cloud", "api_key"),
    "CLOUD_REVIEW_MODEL": ("cloud", "model"),
    "CLOUD_REVIEW_MAX_TOKENS": ("cloud", "max_tokens"),
    "CLOUD_REVIEW_TIMEOUT_SECONDS": ("cloud", "timeout_seconds"),
    "LITELLM_CLOUD_MODEL": ("litellm", "model"),
    "LITELLM_CLOUD_API_KEY": ("litellm", "api_key"),
    "LITELLM_CLOUD_API_URL": ("litellm", "api_url"),
    "LITELLM_CLOUD_MAX_TOKENS": ("litellm", "max_tokens"),
    "LITELLM_CLOUD_TIMEOUT_SECONDS": ("litellm", "timeout_seconds"),
    "DIRECT_MAX_TOKENS": ("tokens", "direct_max_tokens"),
    "SUB_AGENT_MAX_TOKENS": ("tokens", "agent_max_tokens"),
    "SUB_AGENT_TIMEOUT_SECONDS": ("tokens", "sub_agent_timeout_seconds"),
    "VERIFY_TIMEOUT_SECONDS": ("tokens", "verify_timeout_seconds"),
    "CAVEMAN_ENABLED": ("caveman", "enabled"),
    "CAVEMAN_MAX_TOKENS": ("tokens", "caveman_max_tokens"),
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
    "VERIFY_ENABLED": ("verify", "enabled"),
    "VERIFY_LINT_COMMAND": ("verify", "lint_command"),
    "VERIFY_TEST_COMMAND": ("verify", "test_command"),
    "MCP_ENABLED": ("mcp", "enabled"),
}


def _env_to_config_val(env_val: str, default_val: Any) -> Any:
    """Konvertiert einen Env-Var-String in den passenden Typ."""
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


def _load_config() -> Dict[str, Any]:
    """Lädt Config aus JSON-Datei. Falls nicht vorhanden, aus Env-Vars erzeugen."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            _deep_merge(cfg, saved)
        except (json.JSONDecodeError, OSError):
            pass
        return cfg

    # config.json existiert nicht → aus Env-Vars + Defaults erzeugen
    for env_name, (section, key) in _ENV_TO_CONFIG.items():
        val = os.environ.get(env_name)
        if val is not None:
            default_val = DEFAULT_CONFIG.get(section, {}).get(key)
            cfg[section][key] = _env_to_config_val(val, default_val)

    # Neu erzeugte Config sofort speichern (überschreibt sich beim nächsten WebUI-Save)
    _save_config(cfg)
    _log(f"📝 config.json aus Env-Vars erzeugt: {CONFIG_PATH}")
    return cfg


def _save_config(cfg: Dict[str, Any]) -> None:
    """Speichert Config als JSON."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def _deep_merge(base: Dict, override: Dict) -> None:
    """Merge override dict into base dict recursively (modifiziert base in-place)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _mask_key(key: str) -> str:
    """Maskiert API-Keys für die Anzeige."""
    if not key or len(key) < 8:
        return key
    return key[:4] + "•" * (len(key) - 8) + key[-4:]


# ═══════════════════════════════════════════════════════════════════════════
# HTML Dashboard (Single Page App, Vanilla JS, Dark Theme)
# ═══════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LocalProxy v2.0 — Konfiguration</title>
<style>
:root {
  --bg: #0d1117;
  --surface: #161b22;
  --surface2: #21262d;
  --border: #30363d;
  --text: #e6edf3;
  --text2: #8b949e;
  --accent: #58a6ff;
  --accent2: #238636;
  --danger: #f85149;
  --warn: #d2991d;
  --radius: 8px;
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
.status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.status-dot.ok { background: #3fb950; }
.status-dot.err { background: var(--danger); }
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
.card h3 .icon { font-size: 1.2rem; }
.form-group { margin-bottom: 14px; }
.form-group label { display: block; font-size: 0.82rem; color: var(--text2); margin-bottom: 4px; font-weight: 500; }
.form-group .hint { font-size: 0.72rem; color: var(--text2); opacity: 0.7; margin-top: 2px; }
input[type="text"], input[type="url"], input[type="number"], input[type="password"], select, textarea {
  width: 100%; padding: 8px 12px; background: var(--surface2); border: 1px solid var(--border);
  border-radius: 6px; color: var(--text); font-size: 0.9rem; font-family: var(--font);
  transition: border-color .15s;
}
input:focus, select:focus, textarea:focus { outline: none; border-color: var(--accent); }
input[type="range"] { width: 100%; accent-color: var(--accent); }
.row { display: flex; gap: 12px; }
.row > * { flex: 1; }
.toggle-row { display: flex; align-items: center; justify-content: space-between; padding: 8px 0; }
.toggle-row span { font-size: 0.9rem; }
.toggle {
  position: relative; width: 44px; height: 24px; cursor: pointer;
}
.toggle input { display: none; }
.toggle .slider {
  position: absolute; inset: 0; background: var(--surface2); border: 1px solid var(--border);
  border-radius: 12px; transition: .2s;
}
.toggle .slider::after {
  content: ''; position: absolute; width: 18px; height: 18px; border-radius: 50%;
  background: var(--text2); top: 2px; left: 2px; transition: .2s;
}
.toggle input:checked + .slider { background: var(--accent2); border-color: var(--accent2); }
.toggle input:checked + .slider::after { background: #fff; transform: translateX(20px); }
.range-value { font-size: 0.8rem; color: var(--accent); margin-left: 8px; font-weight: 600; }
.btn {
  padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer;
  font-size: 0.9rem; font-weight: 500; transition: .15s; font-family: var(--font);
}
.btn-primary { background: var(--accent2); color: #fff; }
.btn-primary:hover { background: #2ea043; }
.btn-secondary { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }
.btn-secondary:hover { background: var(--border); }
.btn-danger { background: var(--danger); color: #fff; }
.actions { display: flex; gap: 8px; margin-top: 20px; justify-content: flex-end; }
.toast {
  position: fixed; bottom: 24px; right: 24px; padding: 12px 20px;
  border-radius: var(--radius); font-size: 0.9rem; z-index: 200;
  animation: slideUp .3s ease; box-shadow: 0 4px 20px rgba(0,0,0,.4);
}
.toast.success { background: var(--accent2); color: #fff; }
.toast.error { background: var(--danger); color: #fff; }
@keyframes slideUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }

code { background: var(--surface2); padding: 2px 6px; border-radius: 4px; font-size: 0.85em; }
pre { background: var(--surface2); padding: 12px; border-radius: var(--radius); overflow-x: auto; font-size: 0.82rem; }
</style>
</head>
<body>
<header>
  <div>
    <h1>🦊 LocalProxy <span class="badge">v2.0</span></h1>
    <span style="font-size:0.75rem;color:var(--text2)">Konfiguration &amp; Dashboard</span>
  </div>
  <div style="display:flex;align-items:center;gap:12px">
    <span id="userDisplay" style="font-size:0.8rem;color:var(--text2)"></span>
    <button class="btn btn-secondary" onclick="logout()" style="font-size:0.75rem;padding:4px 12px">🚪 Abmelden</button>
    <div class="status">
      <span id="proxyStatus"><span class="status-dot err"></span> Proxy</span>
      <span id="lokalFreeStatus"><span class="status-dot err"></span> Lokal/Free</span>
      <span id="cloudStatus"><span class="status-dot err"></span> Cloud</span>
    </div>
  </div>
</header>
<nav>
  <button class="active" data-tab="models">🤖 Modelle</button>
  <button data-tab="cloud">☁️ Cloud-APIs</button>
  <button data-tab="tokens">🎯 Tokens &amp; Timeouts</button>
  <button data-tab="features">⚙️ Features</button>
  <button data-tab="hindsight">🧠 Hindsight</button>
  <button data-tab="verify">✅ Verifikation</button>
  <button data-tab="logs">📋 Log</button>
</nav>
<main>
  <!-- Models -->
  <section id="tab-models" class="active">
    <div class="card">
      <h3><span class="icon">🖥️</span> Lokal/Free (Worker &amp; Fast)</h3>
      <div class="form-group">
        <label>Lokal/Free API URL</label>
        <input type="url" id="cfg-models-vllm_api_url" placeholder="http://localhost:8000/v1/chat/completions">
        <div class="hint">Endpoint für Lokal/Free (lokal oder Cloud-Free-Tier)</div>
      </div>
      <div class="form-group">
        <label>Lokal/Free Models URL</label>
        <input type="url" id="cfg-models-vllm_models_url" placeholder="http://localhost:8000/v1/models">
      </div>
      <div class="form-group">
        <label>Lokal/Free API Key (optional)</label>
        <input type="password" id="cfg-models-vllm_api_key" placeholder="sk-... für Cloud-Free-Tier">
        <div class="hint">Leer lassen für lokalen Endpoint ohne Auth</div>
      </div>
      <div class="row">
        <div class="form-group">
          <label>Hauptmodell (Worker 80B)</label>
          <input type="text" id="cfg-models-model_name" placeholder="Qwen/Qwen3-Next-80B-Chat-mxfp4">
        </div>
        <div class="form-group">
          <label>Schnelles Modell (27B)</label>
          <input type="text" id="cfg-models-fast_model_name" placeholder="Qwen/Qwen3.6-27B-Chat-FP8">
        </div>
      </div>
    </div>
    <div class="card">
      <h3><span class="icon">🔌</span> Proxy-Einstellungen</h3>
      <div class="row">
        <div class="form-group">
          <label>Proxy Port</label>
          <input type="number" id="cfg-proxy-port" min="1" max="65535">
        </div>
        <div class="form-group">
          <label>API-Key <span style="font-size:0.7rem;color:var(--warn)">(auto-generiert wenn leer)</span></label>
          <input type="text" id="cfg-proxy-api_key" placeholder="localfox-...">
        </div>
      </div>
      <div class="toggle-row">
        <span>🔐 Authentifizierung</span>
        <label class="toggle"><input type="checkbox" id="cfg-proxy-auth_enabled"><span class="slider"></span></label>
      </div>
      <div class="toggle-row">
        <span>💬 Chatty Mode (Status-Infos in Antworten)</span>
        <label class="toggle"><input type="checkbox" id="cfg-proxy-chatty_mode"><span class="slider"></span></label>
      </div>
    </div>
    <div class="actions">
      <button class="btn btn-primary" onclick="saveAndRestart()">💾 Speichern &amp; 🔄 Neustart</button>
    </div>
  </section>

  <!-- Cloud APIs -->
  <section id="tab-cloud">
    <div class="card">
      <h3><span class="icon">☁️</span> Cloud-Planer (OpenAI-kompatibel)</h3>
      <div class="toggle-row">
        <span>Cloud-Planung aktivieren</span>
        <label class="toggle"><input type="checkbox" id="cfg-cloud-enabled"><span class="slider"></span></label>
      </div>
      <div class="form-group">
        <label>API URL</label>
        <input type="url" id="cfg-cloud-api_url" placeholder="https://api.openai.com/v1/chat/completions">
      </div>
      <div class="form-group">
        <label>API Key</label>
        <input type="password" id="cfg-cloud-api_key" placeholder="sk-...">
      </div>
      <div class="row">
        <div class="form-group">
          <label>Modell</label>
          <input type="text" id="cfg-cloud-model" placeholder="gpt-4.1-mini">
        </div>
        <div class="form-group">
          <label>Max Tokens</label>
          <input type="number" id="cfg-cloud-max_tokens" min="64" max="1048576">
        </div>
      </div>
      <div class="form-group">
        <label>Timeout (Sekunden)</label>
        <input type="number" id="cfg-cloud-timeout_seconds" min="5" max="600">
      </div>
    </div>
    <div class="card">
      <h3><span class="icon">🔄</span> LiteLLM (OpenRouter / DeepSeek / Claude)</h3>
      <div class="form-group">
        <label>LiteLLM Modell</label>
        <input type="text" id="cfg-litellm-model" placeholder="openrouter/deepseek/deepseek-r1">
        <div class="hint">z.B. <code>openrouter/anthropic/claude-3.7-sonnet</code> oder <code>deepseek/deepseek-r1</code></div>
      </div>
      <div class="form-group">
        <label>LiteLLM API Key</label>
        <input type="password" id="cfg-litellm-api_key" placeholder="sk-or-...">
      </div>
      <div class="form-group">
        <label>API URL (optional, z.B. OpenRouter Base-URL)</label>
        <input type="url" id="cfg-litellm-api_url" placeholder="https://openrouter.ai/api/v1">
        <div class="hint">Leer lassen für LiteLLM-Standard-Routing (anhand des Modellnamens)</div>
      </div>
      <div class="row">
        <div class="form-group">
          <label>Max Tokens</label>
          <input type="number" id="cfg-litellm-max_tokens" min="64" max="1048576" value="16384">
        </div>
        <div class="form-group">
          <label>Timeout (Sekunden)</label>
          <input type="number" id="cfg-litellm-timeout_seconds" min="5" max="600" value="180">
        </div>
      </div>
    </div>
    <div class="actions">
      <button class="btn btn-primary" onclick="saveAndRestart()">💾 Speichern &amp; 🔄 Neustart</button>
    </div>
  </section>

  <!-- Tokens & Timeouts -->
  <section id="tab-tokens">
    <div class="card">
      <h3><span class="icon">🎯</span> Token-Budgets</h3>
      <div class="form-group">
        <label>Direkte Requests: Max Tokens <span class="range-value" id="val-direct_max_tokens">32768</span></label>
        <input type="range" id="cfg-tokens-direct_max_tokens" min="256" max="131072" step="256" oninput="document.getElementById('val-direct_max_tokens').textContent=this.value">
      </div>
      <div class="form-group">
        <label>Agent-Worker: Max Tokens <span class="range-value" id="val-agent_max_tokens">65536</span></label>
        <input type="range" id="cfg-tokens-agent_max_tokens" min="256" max="262144" step="256" oninput="document.getElementById('val-agent_max_tokens').textContent=this.value">
      </div>
      <div class="form-group">
        <label>Caveman-Plan: Max Tokens <span class="range-value" id="val-caveman_max_tokens">8192</span></label>
        <input type="range" id="cfg-tokens-caveman_max_tokens" min="64" max="65536" step="64" oninput="document.getElementById('val-caveman_max_tokens').textContent=this.value">
      </div>
    </div>
    <div class="card">
      <h3><span class="icon">⏱️</span> Timeouts</h3>
      <div class="form-group">
        <label>Sub-Agent Timeout (Sekunden) <span class="range-value" id="val-sub_agent_timeout">60</span></label>
        <input type="range" id="cfg-tokens-sub_agent_timeout_seconds" min="5" max="300" step="5" oninput="document.getElementById('val-sub_agent_timeout').textContent=this.value">
      </div>
      <div class="form-group">
        <label>Verify Timeout (Sekunden) <span class="range-value" id="val-verify_timeout">45</span></label>
        <input type="range" id="cfg-tokens-verify_timeout_seconds" min="5" max="300" step="5" oninput="document.getElementById('val-verify_timeout').textContent=this.value">
      </div>
    </div>
    <div class="actions">
      <button class="btn btn-primary" onclick="saveAndRestart()">💾 Speichern &amp; 🔄 Neustart</button>
    </div>
  </section>

  <!-- Features -->
  <section id="tab-features">
    <div class="card">
      <h3><span class="icon">⚙️</span> Feature-Toggles</h3>
      <div class="toggle-row">
        <span>🗿 Caveman Ultra (Token-Kompression)</span>
        <label class="toggle"><input type="checkbox" id="cfg-caveman-enabled"><span class="slider"></span></label>
      </div>
      <div class="toggle-row">
        <span>🧠 Hindsight Memory</span>
        <label class="toggle"><input type="checkbox" id="cfg-hindsight-enabled"><span class="slider"></span></label>
      </div>
      <div class="toggle-row">
        <span>✅ Phase-3-Verifikation (Linter/Tests)</span>
        <label class="toggle"><input type="checkbox" id="cfg-verify-enabled"><span class="slider"></span></label>
      </div>
      <div class="toggle-row">
        <span>🔧 MCP-Server (VS Code Tool-Zugriff)</span>
        <label class="toggle"><input type="checkbox" id="cfg-mcp-enabled"><span class="slider"></span></label>
      </div>
    </div>
    <div class="actions">
      <button class="btn btn-primary" onclick="saveAndRestart()">💾 Speichern &amp; 🔄 Neustart</button>
    </div>
  </section>

  <!-- Hindsight -->
  <section id="tab-hindsight">
    <div class="card">
      <h3><span class="icon">🧠</span> Hindsight Memory — Qdrant</h3>
      <div class="toggle-row">
        <span>Qdrant verwenden (statt JSONL)</span>
        <label class="toggle"><input type="checkbox" id="cfg-hindsight-use_qdrant"><span class="slider"></span></label>
      </div>
      <div class="form-group">
        <label>Qdrant URL</label>
        <input type="url" id="cfg-hindsight-qdrant_url" placeholder="http://localhost:6333">
      </div>
      <div class="form-group">
        <label>Qdrant API Key</label>
        <input type="password" id="cfg-hindsight-qdrant_api_key" placeholder="(optional)">
      </div>
      <div class="row">
        <div class="form-group">
          <label>Collection</label>
          <input type="text" id="cfg-hindsight-collection" placeholder="hindsight_memory">
        </div>
        <div class="form-group">
          <label>Embedding-Dimension</label>
          <input type="number" id="cfg-hindsight-embedding_dim" min="64" max="4096">
        </div>
      </div>
    </div>
    <div class="card">
      <h3><span class="icon">📐</span> Recall &amp; Retain</h3>
      <div class="form-group">
        <label>Max Memory Tokens <span class="range-value" id="val-max_memory_tokens">4000</span></label>
        <input type="range" id="cfg-hindsight-max_memory_tokens" min="256" max="16384" step="256" oninput="document.getElementById('val-max_memory_tokens').textContent=this.value">
      </div>
      <div class="form-group">
        <label>Min Similarity <span class="range-value" id="val-min_similarity">0.18</span></label>
        <input type="range" id="cfg-hindsight-min_similarity" min="0.05" max="0.95" step="0.01" oninput="document.getElementById('val-min_similarity').textContent=this.value">
      </div>
      <div class="form-group">
        <label>Retain Delay (Sekunden)</label>
        <input type="number" id="cfg-hindsight-retain_delay_seconds" min="0" max="60" step="0.1">
      </div>
      <div class="form-group">
        <label>JSONL-Speicherverzeichnis</label>
        <input type="text" id="cfg-hindsight-dir" placeholder="./.hindsight_memory">
      </div>
    </div>
    <div class="actions">
      <button class="btn btn-secondary" onclick="clearMemory()">🗑️ Memory löschen</button>
      <button class="btn btn-primary" onclick="saveAndRestart()">💾 Speichern &amp; 🔄 Neustart</button>
    </div>
  </section>

  <!-- Verify -->
  <section id="tab-verify">
    <div class="card">
      <h3><span class="icon">✅</span> Phase 3 — Verifikation</h3>
      <div class="form-group">
        <label>Lint-Befehl</label>
        <input type="text" id="cfg-verify-lint_command" placeholder="ruff check">
        <div class="hint">Wird im Shell ausgeführt, Output wird analysiert</div>
      </div>
      <div class="form-group">
        <label>Test-Befehl</label>
        <input type="text" id="cfg-verify-test_command" placeholder="pytest -x --tb=short">
        <div class="hint">Wird im Shell ausgeführt, Output wird analysiert</div>
      </div>
    </div>
    <div class="actions">
      <button class="btn btn-primary" onclick="saveAndRestart()">💾 Speichern &amp; 🔄 Neustart</button>
    </div>
  </section>
  <!-- Logs -->
  <section id="tab-logs">
    <div class="card">
      <h3><span class="icon">📋</span> Live-Log <span style="font-weight:400;font-size:0.75rem;color:var(--text2)">(letzte 200 Zeilen, auto-refresh)</span></h3>
      <div style="display:flex;gap:8px;margin-bottom:12px;align-items:center">
        <button class="btn btn-secondary" onclick="refreshLogs()" style="font-size:0.8rem;padding:4px 12px">🔄 Jetzt laden</button>
        <label style="display:flex;align-items:center;gap:6px;font-size:0.8rem;color:var(--text2);cursor:pointer">
          <input type="checkbox" id="log-autorefresh" checked onchange="toggleLogAutoRefresh()"> Auto-Refresh (3s)
        </label>
        <span id="log-info" style="font-size:0.75rem;color:var(--text2);margin-left:auto"></span>
      </div>
      <pre id="log-viewer" style="background:var(--bg);border:1px solid var(--border);max-height:60vh;overflow-y:auto;font-size:0.75rem;line-height:1.6;white-space:pre-wrap;word-break:break-all">Wird geladen...</pre>
    </div>
  </section>
</main>

<script>
// ── State ──────────────────────────────────────────────────────────────
let currentConfig = {};

const ID_MAP = {
  'cfg-models-vllm_api_url': ['models','vllm_api_url'],
  'cfg-models-vllm_models_url': ['models','vllm_models_url'],
  'cfg-models-vllm_api_key': ['models','vllm_api_key'],
  'cfg-models-model_name': ['models','model_name'],
  'cfg-models-fast_model_name': ['models','fast_model_name'],
  'cfg-proxy-port': ['proxy','port'],
  'cfg-proxy-auth_enabled': ['proxy','auth_enabled'],
  'cfg-proxy-api_key': ['proxy','api_key'],
  'cfg-proxy-chatty_mode': ['proxy','chatty_mode'],
  'cfg-cloud-enabled': ['cloud','enabled'],
  'cfg-cloud-api_url': ['cloud','api_url'],
  'cfg-cloud-api_key': ['cloud','api_key'],
  'cfg-cloud-model': ['cloud','model'],
  'cfg-cloud-max_tokens': ['cloud','max_tokens'],
  'cfg-cloud-timeout_seconds': ['cloud','timeout_seconds'],
  'cfg-litellm-model': ['litellm','model'],
  'cfg-litellm-api_key': ['litellm','api_key'],
  'cfg-litellm-api_url': ['litellm','api_url'],
  'cfg-litellm-max_tokens': ['litellm','max_tokens'],
  'cfg-litellm-timeout_seconds': ['litellm','timeout_seconds'],
  'cfg-tokens-direct_max_tokens': ['tokens','direct_max_tokens'],
  'cfg-tokens-agent_max_tokens': ['tokens','agent_max_tokens'],
  'cfg-tokens-caveman_max_tokens': ['tokens','caveman_max_tokens'],
  'cfg-tokens-sub_agent_timeout_seconds': ['tokens','sub_agent_timeout_seconds'],
  'cfg-tokens-verify_timeout_seconds': ['tokens','verify_timeout_seconds'],
  'cfg-caveman-enabled': ['caveman','enabled'],
  'cfg-hindsight-enabled': ['hindsight','enabled'],
  'cfg-hindsight-use_qdrant': ['hindsight','use_qdrant'],
  'cfg-hindsight-qdrant_url': ['hindsight','qdrant_url'],
  'cfg-hindsight-qdrant_api_key': ['hindsight','qdrant_api_key'],
  'cfg-hindsight-collection': ['hindsight','collection'],
  'cfg-hindsight-embedding_dim': ['hindsight','embedding_dim'],
  'cfg-hindsight-max_memory_tokens': ['hindsight','max_memory_tokens'],
  'cfg-hindsight-min_similarity': ['hindsight','min_similarity'],
  'cfg-hindsight-retain_delay_seconds': ['hindsight','retain_delay_seconds'],
  'cfg-hindsight-dir': ['hindsight','dir'],
  'cfg-verify-enabled': ['verify','enabled'],
  'cfg-verify-lint_command': ['verify','lint_command'],
  'cfg-verify-test_command': ['verify','test_command'],
  'cfg-mcp-enabled': ['mcp','enabled'],
};

function getNested(obj, path) { return path.reduce((o,k) => (o||{})[k], obj); }

function setField(id, value) {
  const path = ID_MAP[id]; if (!path) return;
  let obj = currentConfig;
  for (let i=0; i<path.length-1; i++) { if (!obj[path[i]]) obj[path[i]]={}; obj=obj[path[i]]; }
  obj[path[path.length-1]] = value;
}

function populateForm() {
  Object.entries(ID_MAP).forEach(([id, path]) => {
    const el = document.getElementById(id); if (!el) return;
    const val = getNested(currentConfig, path);
    if (el.type === 'checkbox') el.checked = !!val;
    else el.value = val ?? '';
  });
  // Sync range displays
  ['direct_max_tokens','agent_max_tokens','caveman_max_tokens','max_memory_tokens'].forEach(k => {
    const el = document.getElementById('cfg-tokens-'+k) || document.getElementById('cfg-hindsight-'+k);
    const disp = document.getElementById('val-'+k);
    if (el && disp) disp.textContent = el.value;
  });
  ['sub_agent_timeout_seconds','verify_timeout_seconds'].forEach(k => {
    const el = document.getElementById('cfg-tokens-'+k);
    const disp = document.getElementById('val-'+k.replace('_seconds',''));
    if (el && disp) disp.textContent = el.value;
  });
  const minSim = document.getElementById('cfg-hindsight-min_similarity');
  if (minSim) document.getElementById('val-min_similarity').textContent = minSim.value;
}

function collectForm() {
  Object.entries(ID_MAP).forEach(([id,_]) => {
    const el = document.getElementById(id); if (!el) return;
    const val = el.type === 'checkbox' ? el.checked :
                (el.type === 'number' || el.type === 'range') ? (parseFloat(el.value) || 0) : el.value;
    setField(id, val);
  });
}

// ── Tab Switching ──────────────────────────────────────────────────────
document.querySelectorAll('nav button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('section').forEach(s => s.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  });
});

// ── Toast ──────────────────────────────────────────────────────────────
function toast(msg, type='success') {
  const t = document.createElement('div');
  t.className = 'toast ' + type; t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3000);
}

// ── API Calls ──────────────────────────────────────────────────────────
async function loadConfig() {
  try {
    const r = await apiFetch('/webui/api/config');
    currentConfig = await r.json();
    populateForm();
    toast('Konfiguration geladen', 'success');
  } catch(e) { toast('Fehler beim Laden: '+e.message, 'error'); }
}

async function saveConfig() {
  collectForm();
  try {
    const r = await apiFetch('/webui/api/config', {
      method: 'PUT', headers: {'Content-Type':'application/json'},
      body: JSON.stringify(currentConfig)
    });
    if (r.ok) { toast('✅ Gespeichert — Proxy muss neu gestartet werden für Änderungen', 'success'); return true; }
    else { const e = await r.json(); toast('Fehler: '+e.detail, 'error'); return false; }
  } catch(e) { toast('Fehler beim Speichern: '+e.message, 'error'); return false; }
}

async function saveAndRestart() {
  const saved = await saveConfig();
  if (!saved) return;
  toast('🔄 Starte Proxy neu...', 'success');
  try {
    const r = await apiFetch('/webui/api/restart', {method:'POST'});
    if (r.ok) {
      toast('✅ Neustart läuft — Seite lädt in 4s neu', 'success');
      // Warten bis Proxy wieder da ist, dann neuladen
      setTimeout(async () => {
        for (let i=0; i<20; i++) {
          try {
            const rr = await fetch('/healthz');
            if (rr.ok) { location.reload(); return; }
          } catch(e) {}
          await new Promise(r => setTimeout(r, 1000));
        }
        location.reload();
      }, 4000);
    } else {
      toast('⚠️ Config gespeichert, aber Neustart fehlgeschlagen', 'error');
    }
  } catch(e) {
    // Proxy ist schon tot (erwartet), Seite lädt neu
    toast('🔄 Neustart läuft...', 'success');
    setTimeout(() => {
      (async () => {
        for (let i=0; i<20; i++) {
          try { const rr = await fetch('/healthz'); if (rr.ok) { location.reload(); return; } } catch(e) {}
          await new Promise(r => setTimeout(r, 1000));
        }
        location.reload();
      })();
    }, 3000);
  }
}

async function clearMemory() {
  if (!confirm('Hindsight Memory wirklich löschen?')) return;
  try {
    const r = await apiFetch('/webui/api/memory/clear', {method:'POST'});
    if (r.ok) toast('Memory gelöscht', 'success');
    else toast('Fehler beim Löschen', 'error');
  } catch(e) { toast('Fehler: '+e.message, 'error'); }
}

async function refreshStatus() {
  try {
    const r = await fetch('/healthz');
    const h = await r.json();
    document.querySelector('#proxyStatus .status-dot').className = 'status-dot ' + (h.status==='ok'?'ok':'err');
    document.querySelector('#proxyStatus').childNodes[1].textContent = ' Proxy';
  } catch(e) {}
  try {
    const r = await apiFetch('/webui/api/status');
    const s = await r.json();
    document.querySelector('#lokalFreeStatus .status-dot').className = 'status-dot ' + (s.vllm_ok?'ok':'err');
    document.querySelector('#cloudStatus .status-dot').className = 'status-dot ' + (s.cloud_configured?'ok':'err');
    if (s.user) document.getElementById('userDisplay').textContent = '👤 ' + s.user;
  } catch(e) {}
}

let logAutoRefresh = true;
let logTimer = null;

function toggleLogAutoRefresh() {
  logAutoRefresh = document.getElementById('log-autorefresh').checked;
  if (logAutoRefresh) { startLogPolling(); }
  else { if (logTimer) { clearInterval(logTimer); logTimer = null; } }
}

async function refreshLogs() {
  try {
    const r = await fetch('/logs?lines=200');
    if (!r.ok) throw new Error(r.status);
    const d = await r.json();
    const el = document.getElementById('log-viewer');
    if (el) {
      el.textContent = d.lines.join('');
      el.scrollTop = el.scrollHeight;
    }
    const info = document.getElementById('log-info');
    if (info) info.textContent = d.count + '/' + d.total + ' Zeilen';
  } catch(e) {
    const el = document.getElementById('log-viewer');
    if (el) el.textContent = 'Fehler beim Laden: ' + e.message;
  }
}

function startLogPolling() {
  if (logTimer) clearInterval(logTimer);
  logTimer = setInterval(refreshLogs, 3000);
}

// ── Init ───────────────────────────────────────────────────────────────

// Benutzer anzeigen
const params = new URLSearchParams(window.location.search);
const tokenParam = params.get('token');
if (tokenParam) {
  document.cookie = 'webui_token=' + tokenParam + '; path=/webui; max-age=86400; SameSite=Lax';
}
document.getElementById('userDisplay').textContent = '👤 ' + ('WEBUI_USER'); // wird von refreshStatus() überschrieben

async function logout() {
  const r = await fetch('/webui/api/logout', {method:'POST'});
  if (r.ok) {
    document.cookie = 'webui_token=; path=/webui; max-age=0; SameSite=Lax';
    window.location.href = '/webui/login';
  }
}

// 401-Handler für API-Fetch
async function apiFetch(url, options = {}) {
  const r = await fetch(url, options);
  if (r.status === 401) {
    document.cookie = 'webui_token=; path=/webui; max-age=0; SameSite=Lax';
    window.location.href = '/webui/login';
    throw new Error('Unauthorized');
  }
  return r;
}

loadConfig();
refreshStatus();
refreshLogs();
setInterval(refreshStatus, 15000);
startLogPolling();
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════════════════

def create_webui_app() -> FastAPI:
    """Erstellt eine FastAPI-Sub-App mit dem Webinterface."""

    webapp = FastAPI(docs_url=None, openapi_url=None, redoc_url=None)

    # ── Auth-Middleware ──────────────────────────────────────────────────
    @webapp.middleware("http")
    async def _auth_middleware(request: Request, call_next):
        # Unprotected paths
        if request.url.path in ("/webui/login", "/webui/api/login", "/webui/api/logout"):
            return await call_next(request)

        # OPTIONS (CORS preflight) immer erlauben
        if request.method == "OPTIONS":
            return await call_next(request)

        # Token aus Cookie oder Query-Parameter
        token = request.cookies.get(COOKIE_NAME, "")
        if not token:
            token = request.query_params.get("token", "")

        if not _validate_token(token):
            # API-Calls → 401, sonst Redirect zum Login
            if request.url.path.startswith("/webui/api/"):
                return JSONResponse(status_code=401, content={"error": "Unauthorized", "login_url": "/webui/login"})
            return RedirectResponse(url="/webui/login")

        return await call_next(request)

    # ── Login-Seite ──────────────────────────────────────────────────────
    @webapp.get("/login", response_class=HTMLResponse)
    async def login_page():
        return LOGIN_HTML

    # ── Login-API ────────────────────────────────────────────────────────
    @webapp.post("/api/login")
    async def api_login(request: Request, response: Response):
        try:
            data = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        username = data.get("username", "")
        password = data.get("password", "")

        if username == WEBUI_USERNAME and password == WEBUI_PASSWORD:
            token = _generate_token()
            _log(f"✅ WebUI Login erfolgreich: {username}")
            return {"token": token, "status": "ok"}
        _log(f"⚠ WebUI Login fehlgeschlagen: {username}")
        raise HTTPException(status_code=401, detail="Invalid credentials")

    @webapp.post("/api/logout")
    async def api_logout(request: Request):
        token = request.cookies.get(COOKIE_NAME, "") or request.query_params.get("token", "")
        if token:
            _remove_token(token)
        return {"status": "ok"}

    @webapp.get("/", response_class=HTMLResponse)
    async def dashboard():
        return DASHBOARD_HTML

    @webapp.get("/api/config")
    async def get_config():
        cfg = _load_config()
        # API-Keys maskieren
        if cfg["cloud"]["api_key"]:
            cfg["cloud"]["api_key"] = _mask_key(cfg["cloud"]["api_key"])
        if cfg["litellm"]["api_key"]:
            cfg["litellm"]["api_key"] = _mask_key(cfg["litellm"]["api_key"])
        if cfg["models"].get("vllm_api_key"):
            cfg["models"]["vllm_api_key"] = _mask_key(cfg["models"]["vllm_api_key"])
        if cfg["hindsight"]["qdrant_api_key"]:
            cfg["hindsight"]["qdrant_api_key"] = _mask_key(cfg["hindsight"]["qdrant_api_key"])
        return JSONResponse(content=cfg)

    @webapp.put("/api/config")
    async def put_config(request: Request):
        try:
            new_cfg = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        # Merge mit bestehender Config (um API-Keys zu erhalten die nicht mitgesendet wurden)
        current = _load_config()
        _deep_merge(current, new_cfg)

        # Wenn API-Keys maskiert sind, originalen Wert behalten
        for section, key_name in [
            ("cloud", "api_key"),
            ("litellm", "api_key"),
            ("models", "vllm_api_key"),
            ("hindsight", "qdrant_api_key"),
        ]:
            val = current.get(section, {}).get(key_name, "")
            if "•" in val:
                old_val = _load_config().get(section, {}).get(key_name, "")
                if old_val:
                    current[section][key_name] = old_val

        _save_config(current)
        _log("💾 Config gespeichert")
        return JSONResponse(content={"status": "ok", "message": "Config saved"})

    @webapp.get("/api/status")
    async def get_status():
        """Einfacher Status-Check für Lokal/Free und Cloud."""
        import httpx

        vllm_ok = False
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(os.getenv("VLLM_MODELS_URL", "http://localhost:8000/v1/models"))
                vllm_ok = r.status_code == 200
        except Exception:
            pass

        cfg = _load_config()
        cloud_configured = bool(
            cfg["cloud"]["enabled"] and cfg["cloud"]["api_key"]
        ) or bool(cfg["litellm"]["api_key"]) or bool(cfg["litellm"]["api_url"])

        return JSONResponse(content={
            "vllm_ok": vllm_ok,
            "cloud_configured": cloud_configured,
            "config_exists": CONFIG_PATH.exists(),
            "user": WEBUI_USERNAME,
        })

    @webapp.post("/api/restart")
    async def restart_proxy():
        """Startet den Proxy-Service neu (systemctl restart localproxy)."""
        _log("🔄 Neustart angefordert via WebUI")
        # Hintergrund-Restart mit kurzer Verzögerung, damit die Antwort noch rausgeht
        import threading as _th
        def _do_restart():
            import time as _t
            _t.sleep(0.5)
            import subprocess as _sp
            _sp.run(["systemctl", "restart", "localproxy"], capture_output=True)
        _th.Thread(target=_do_restart, daemon=True).start()
        return JSONResponse(content={"status": "ok", "message": "Restarting..."})

    @webapp.post("/api/memory/clear")
    async def clear_memory():
        """Löscht die Hindsight-Memory-Dateien."""
        cfg = _load_config()
        mem_dir = Path(cfg.get("hindsight", {}).get("dir", "./.hindsight_memory"))
        deleted = 0
        if mem_dir.exists():
            for f in mem_dir.glob("*.jsonl"):
                try:
                    f.unlink()
                    deleted += 1
                except OSError:
                    pass
        return JSONResponse(content={"status": "ok", "deleted_files": deleted})

    return webapp


def mount_webui(parent_app: FastAPI, prefix: str = "/webui") -> None:
    """Mountet das Webinterface in die Haupt-App."""
    webui = create_webui_app()
    parent_app.mount(prefix, webui)
