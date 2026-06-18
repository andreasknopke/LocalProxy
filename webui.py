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
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

# ═══════════════════════════════════════════════════════════════════════════
# Config-Datei
# ═══════════════════════════════════════════════════════════════════════════

CONFIG_PATH = Path(os.getenv("LOCALPROXY_CONFIG", "config.json"))

DEFAULT_CONFIG: Dict[str, Any] = {
    "models": {
        "vllm_api_url": "http://localhost:8000/v1/chat/completions",
        "vllm_models_url": "http://localhost:8000/v1/models",
        "model_name": "Qwen/Qwen3-Next-80B-Chat-mxfp4",
        "fast_model_name": "Qwen/Qwen3.6-27B-Chat-FP8",
    },
    "cloud": {
        "enabled": False,
        "api_url": "https://api.openai.com/v1/chat/completions",
        "api_key": "",
        "model": "gpt-4.1-mini",
        "max_tokens": 2048,
        "timeout_seconds": 90,
    },
    "litellm": {
        "model": "",
        "api_key": "",
    },
    "proxy": {
        "port": 9001,
        "auth_enabled": True,
        "api_key": "",
        "chatty_mode": True,
    },
    "tokens": {
        "direct_max_tokens": 2048,
        "agent_max_tokens": 4096,
        "caveman_max_tokens": 1024,
        "sub_agent_timeout_seconds": 60,
        "verify_timeout_seconds": 45,
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


def _load_config() -> Dict[str, Any]:
    """Lädt Config aus JSON-Datei, merged mit Defaults."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            _deep_merge(cfg, saved)
        except (json.JSONDecodeError, OSError):
            pass
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
  <div class="status">
    <span id="proxyStatus"><span class="status-dot err"></span> Proxy</span>
    <span id="vllmStatus"><span class="status-dot err"></span> vLLM</span>
    <span id="cloudStatus"><span class="status-dot err"></span> Cloud</span>
  </div>
</header>
<nav>
  <button class="active" data-tab="models">🤖 Modelle</button>
  <button data-tab="cloud">☁️ Cloud-APIs</button>
  <button data-tab="tokens">🎯 Tokens &amp; Timeouts</button>
  <button data-tab="features">⚙️ Features</button>
  <button data-tab="hindsight">🧠 Hindsight</button>
  <button data-tab="verify">✅ Verifikation</button>
</nav>
<main>
  <!-- Models -->
  <section id="tab-models" class="active">
    <div class="card">
      <h3><span class="icon">🖥️</span> Lokale vLLM-Modelle</h3>
      <div class="form-group">
        <label>vLLM API URL</label>
        <input type="url" id="cfg-models-vllm_api_url" placeholder="http://localhost:8000/v1/chat/completions">
        <div class="hint">Endpoint des lokalen vLLM-Servers</div>
      </div>
      <div class="form-group">
        <label>vLLM Models URL</label>
        <input type="url" id="cfg-models-vllm_models_url" placeholder="http://localhost:8000/v1/models">
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
      <button class="btn btn-primary" onclick="saveConfig()">💾 Speichern</button>
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
          <input type="number" id="cfg-cloud-max_tokens" min="64" max="32768">
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
    </div>
    <div class="actions">
      <button class="btn btn-primary" onclick="saveConfig()">💾 Speichern</button>
    </div>
  </section>

  <!-- Tokens & Timeouts -->
  <section id="tab-tokens">
    <div class="card">
      <h3><span class="icon">🎯</span> Token-Budgets</h3>
      <div class="form-group">
        <label>Direkte Requests: Max Tokens <span class="range-value" id="val-direct_max_tokens">2048</span></label>
        <input type="range" id="cfg-tokens-direct_max_tokens" min="256" max="16384" step="256" oninput="document.getElementById('val-direct_max_tokens').textContent=this.value">
      </div>
      <div class="form-group">
        <label>Agent-Worker: Max Tokens <span class="range-value" id="val-agent_max_tokens">4096</span></label>
        <input type="range" id="cfg-tokens-agent_max_tokens" min="256" max="32768" step="256" oninput="document.getElementById('val-agent_max_tokens').textContent=this.value">
      </div>
      <div class="form-group">
        <label>Caveman-Plan: Max Tokens <span class="range-value" id="val-caveman_max_tokens">1024</span></label>
        <input type="range" id="cfg-tokens-caveman_max_tokens" min="64" max="8192" step="64" oninput="document.getElementById('val-caveman_max_tokens').textContent=this.value">
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
      <button class="btn btn-primary" onclick="saveConfig()">💾 Speichern</button>
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
      <button class="btn btn-primary" onclick="saveConfig()">💾 Speichern</button>
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
      <button class="btn btn-primary" onclick="saveConfig()">💾 Speichern</button>
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
      <button class="btn btn-primary" onclick="saveConfig()">💾 Speichern</button>
    </div>
  </section>
</main>

<script>
// ── State ──────────────────────────────────────────────────────────────
let currentConfig = {};

const ID_MAP = {
  'cfg-models-vllm_api_url': ['models','vllm_api_url'],
  'cfg-models-vllm_models_url': ['models','vllm_models_url'],
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
                el.type === 'number' ? (parseFloat(el.value) || 0) : el.value;
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
    const r = await fetch('/webui/api/config');
    currentConfig = await r.json();
    populateForm();
    toast('Konfiguration geladen', 'success');
  } catch(e) { toast('Fehler beim Laden: '+e.message, 'error'); }
}

async function saveConfig() {
  collectForm();
  try {
    const r = await fetch('/webui/api/config', {
      method: 'PUT', headers: {'Content-Type':'application/json'},
      body: JSON.stringify(currentConfig)
    });
    if (r.ok) { toast('✅ Gespeichert — Proxy muss neu gestartet werden für Änderungen', 'success'); }
    else { const e = await r.json(); toast('Fehler: '+e.detail, 'error'); }
  } catch(e) { toast('Fehler beim Speichern: '+e.message, 'error'); }
}

async function clearMemory() {
  if (!confirm('Hindsight Memory wirklich löschen?')) return;
  try {
    const r = await fetch('/webui/api/memory/clear', {method:'POST'});
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
    const r = await fetch('/webui/api/status');
    const s = await r.json();
    document.querySelector('#vllmStatus .status-dot').className = 'status-dot ' + (s.vllm_ok?'ok':'err');
    document.querySelector('#cloudStatus .status-dot').className = 'status-dot ' + (s.cloud_configured?'ok':'err');
  } catch(e) {}
}

// ── Init ───────────────────────────────────────────────────────────────
loadConfig();
refreshStatus();
setInterval(refreshStatus, 15000);
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════════════════

def create_webui_app() -> FastAPI:
    """Erstellt eine FastAPI-Sub-App mit dem Webinterface."""

    webapp = FastAPI(docs_url=None, openapi_url=None, redoc_url=None)

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
            ("hindsight", "qdrant_api_key"),
        ]:
            val = current.get(section, {}).get(key_name, "")
            if "•" in val:
                old_val = _load_config().get(section, {}).get(key_name, "")
                if old_val:
                    current[section][key_name] = old_val

        _save_config(current)
        return JSONResponse(content={"status": "ok", "message": "Config saved"})

    @webapp.get("/api/status")
    async def get_status():
        """Einfacher Status-Check für vLLM und Cloud."""
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
        ) or bool(cfg["litellm"]["api_key"])

        return JSONResponse(content={
            "vllm_ok": vllm_ok,
            "cloud_configured": cloud_configured,
            "config_exists": CONFIG_PATH.exists(),
        })

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
