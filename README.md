# LocalProxy v2.0 — Hybrider Agentischer Routing-Proxy

Ein OpenAI-kompatibler FastAPI-Proxy für lokale vLLM-/Qwen-Coder-Modelle auf dem **DX Spark GB10 (128 GB VRAM)**.

Der Proxy entscheidet pro Request zwischen zwei Routen:

| Route | Auslöser | Ablauf |
| --- | --- | --- |
| **Direkt lokal** | Kurze Prompts, Autocomplete, Inline-Fix, Rename, Format | Prompt → Qwen 80B → Antwort |
| **Hybrid agentisch** | Refactor, Bug, Architektur, Tests, Security, komplexe Tasks | Hindsight Recall → Cloud-Planer (Caveman) → Worker (80B) → **Verify** |

## Architektur

```
+------------------------------------------------------------+
|                       VS Code                              |
|          (Extension: Continue / Cline / Roo-Code)          |
+------------------------------------------------------------+
                             |
                             v  (OpenAI REST API)
+------------------------------------------------------------+
|                FastAPI Proxy Gateway v2.0                  |
|  - Intent Classifier (deterministisch + Fast Model 27B)    |
|  - LiteLLM Cloud-Router                                    |
|  - MCP-Server (/mcp)                                       |
+------------------------------------------------------------+
         |                                           |
         v (Komplexe Tasks)                          v (Standard)
+------------------------------+       +-------------------------------+
|   3-Phasen Agenten-Harnisch  |       |     Lokales vLLM Backend      |
|  1. Hindsight Recall         |       |     - Qwen 3.6 - 27B (Fast)  |
|  2. Cloud-Planer (Caveman)   |       |     - Qwen 3 Next-Coder 80B  |
|  3. Worker 80B + Verify      |       +-------------------------------+
+------------------------------+
         |                     |
         v (Cloud-Planung)     v (Lokale Ausführung)
+-------------------+       +-----------------------------+
|    Cloud-API      |       |       Spark GB10 (128GB)    |
| (DeepSeek-R1 /    |       |  - Chunked Prefill          |
|  Claude 3.7 /     |       |  - KV-Prompt Caching        |
|  GPT-4.1)         |       |  - Tensor-Parallelismus     |
+-------------------+       +-----------------------------+
```

## Komponenten

### 1. Intent-Klassifizierung
- **Deterministisch:** Regex/Trigger-Wörter für klare Fälle
- **Fast Model (27B):** Bei mehrdeutigen Prompts klassifiziert das schnelle 27B-Modell

### 2. Hindsight Memory (Qdrant + JSONL-Fallback)
Vier logische Netzwerke:
- **World Facts:** API-Endpunkte, Build-/Test-Kontext, Repository-Fakten
- **Agent Experiences:** Erfolgreiche & fehlgeschlagene Lösungsansätze
- **Entity Summaries:** Module, Klassen, Funktionen, Komponenten
- **Evolving Beliefs:** Aktuelle Refactoring- und Architektur-Entscheidungen

Primär wird **Qdrant** als Vektordatenbank genutzt. Falls nicht erreichbar, fallback auf JSONL-Dateien.

### 3. Caveman Ultra (Token-Kompression)
Der Cloud-Planer antwortet im Caveman-Stil: nur Symbole, Pfeile, Keywords. 
Senkt Token-Verbrauch um 60–75 %.

### 4. 3-Phasen-Agenten-Workflow
1. **Phase 1 – Plan:** Hindsight Recall → Cloud-Planer erstellt Caveman-Plan
2. **Phase 2 – Execute:** Qwen 80B Worker führt den Plan strikt aus
3. **Phase 3 – Verify:** Linter/Tests + Self-Correction durch lokales Modell

### 5. LiteLLM Cloud-Routing
Unterstützt OpenAI, Anthropic, DeepSeek, OpenRouter und alle LiteLLM-kompatiblen Provider.

### 6. MCP-Server (Model Context Protocol)
原生 MCP-Integration für VS Code (Continue/Cline):
- `localproxy_read_file` / `localproxy_write_file`
- `localproxy_list_files` / `localproxy_search_code`
- `localproxy_run_terminal`
- `localproxy_hindsight_recall` / `localproxy_get_status`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Qdrant (optional, für Vektor-Memory)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Starten

### Minimal (nur lokales vLLM)

```bash
VLLM_API_URL=http://localhost:8000/v1/chat/completions \
MODEL_NAME="Qwen/Qwen3-Next-80B-Chat-mxfp4" \
PROXY_AUTH_ENABLED=false \
python proxy.py
```

### Hybrid-Modus mit Cloud-Planung (OpenAI)

```bash
VLLM_API_URL=http://localhost:8000/v1/chat/completions \
MODEL_NAME="Qwen/Qwen3-Next-80B-Chat-mxfp4" \
CLOUD_REVIEW_ENABLED=true \
CLOUD_REVIEW_API_KEY="sk-..." \
CLOUD_REVIEW_MODEL="gpt-4.1-mini" \
python proxy.py
```

### Hybrid-Modus mit LiteLLM (OpenRouter/DeepSeek/Claude)

```bash
VLLM_API_URL=http://localhost:8000/v1/chat/completions \
MODEL_NAME="Qwen/Qwen3-Next-80B-Chat-mxfp4" \
CLOUD_REVIEW_ENABLED=true \
LITELLM_CLOUD_MODEL="openrouter/deepseek/deepseek-r1" \
LITELLM_CLOUD_API_KEY="sk-or-..." \
python proxy.py
```

### Mit Qdrant Memory

```bash
QDRANT_URL=http://localhost:6333 \
HINDSIGHT_USE_QDRANT=true \
VLLM_API_URL=http://localhost:8000/v1/chat/completions \
python proxy.py
```

Der Proxy läuft standardmäßig auf **Port 9001**:
```text
http://0.0.0.0:9001
```

MCP-Endpoint: `http://0.0.0.0:9001/mcp`

## Konfiguration

| Variable | Standard | Beschreibung |
| --- | --- | --- |
| `VLLM_API_URL` | `http://localhost:8000/v1/chat/completions` | Ziel-API des lokalen vLLM-Servers |
| `VLLM_MODELS_URL` | `http://localhost:8000/v1/models` | Modelle-Endpoint |
| `MODEL_NAME` | `Qwen/Qwen3-Next-80B-Chat-mxfp4` | Hauptmodell (Worker) |
| `FAST_MODEL_NAME` | `Qwen/Qwen3.6-27B-Chat-FP8` | Schnelles Modell (Klassifikation/Autocomplete) |
| `PROXY_PORT` | `9001` | Proxy-Port |
| `PROXY_AUTH_ENABLED` | `true` | API-Key-Authentifizierung |
| `PROXY_API_KEY` | auto-generiert | API-Key (bei Auth) |
| `CHATTY_MODE` | `true` | Status-Updates im Chat-Output |
| `CLOUD_REVIEW_ENABLED` | `false` | Cloud-Planer aktivieren |
| `CLOUD_REVIEW_API_URL` | `https://api.openai.com/v1/chat/completions` | Cloud-API-URL |
| `CLOUD_REVIEW_API_KEY` | leer | Cloud-API-Key |
| `CLOUD_REVIEW_MODEL` | `gpt-4.1-mini` | Cloud-Modell |
| `CLOUD_REVIEW_MAX_TOKENS` | `2048` | Max. Tokens Cloud |
| `CLOUD_REVIEW_TIMEOUT_SECONDS` | `90` | Timeout Cloud |
| `LITELLM_CLOUD_MODEL` | leer | LiteLLM-Modell (z.B. `openrouter/deepseek/deepseek-r1`) |
| `LITELLM_CLOUD_API_KEY` | leer | LiteLLM API-Key |
| `CAVEMAN_ENABLED` | `true` | Caveman-Prompt-Injektion |
| `CAVEMAN_MAX_TOKENS` | `1024` | Max. Tokens Caveman-Plan |
| `HINDSIGHT_ENABLED` | `true` | Hindsight Memory |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant-URL |
| `QDRANT_API_KEY` | leer | Qdrant-API-Key |
| `HINDSIGHT_USE_QDRANT` | `false` (auto bei non-localhost QDRANT_URL) | Qdrant statt JSONL |
| `HINDSIGHT_DIR` | `./.hindsight_memory` | JSONL-Speicherort |
| `HINDSIGHT_EMBEDDING_DIM` | `768` | Embedding-Dimension |
| `HINDSIGHT_MAX_MEMORY_TOKENS` | `4000` | Token-Budget Recall-Kontext |
| `HINDSIGHT_MIN_SIMILARITY` | `0.18` | Min. Ähnlichkeit Recall |
| `VERIFY_ENABLED` | `true` | Phase-3-Verifikation |
| `VERIFY_LINT_COMMAND` | leer | Lint-Befehl (z.B. `ruff check`) |
| `VERIFY_TEST_COMMAND` | leer | Test-Befehl (z.B. `pytest -x`) |
| `VERIFY_TIMEOUT_SECONDS` | `45` | Timeout Verifikation |
| `MCP_ENABLED` | `true` | MCP-Server aktiv |
| `DIRECT_MAX_TOKENS` | `2048` | Tokens für direkte Requests |
| `SUB_AGENT_MAX_TOKENS` | `4096` | Tokens für Agent-Worker |
| `SUB_AGENT_TIMEOUT_SECONDS` | `60` | Timeout pro lokaler Anfrage |

## Testaufrufe

### Direkter Request (trivial)
```bash
curl http://localhost:9001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "Qwen/Qwen3-Next-80B-Chat-mxfp4",
    "messages": [{"role": "user", "content": "Fix typo in this inline function."}]
  }'
```

### Komplexer Request (triggert Agent-Workflow)
```bash
curl http://localhost:9001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "Qwen/Qwen3-Next-80B-Chat-mxfp4",
    "messages": [{"role": "user", "content": "Refactor this class to reduce coupling and add tests."}]
  }'
```

### MCP-Request (VS Code Tool-Zugriff)
```bash
curl http://localhost:9001/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

## Nutzung mit VS Code

Konfiguriere **Continue**, **Cline** oder **Roo-Code**:

```json
{
  "model": "Qwen/Qwen3-Next-80B-Chat-mxfp4",
  "apiBase": "http://localhost:9001/v1",
  "apiKey": "YOUR_PROXY_API_KEY"
}
```

Für MCP-Toolzugriff in `.vscode/mcp.json`:
```json
{
  "servers": {
    "localproxy": {
      "type": "http",
      "url": "http://localhost:9001/mcp"
    }
  }
}
```

## Web-Konfigurationsinterface 🌐

Der Proxy enthält ein eingebautes Web-Dashboard unter:

```text
http://0.0.0.0:9001/webui/
```

Das Dashboard ermöglicht die grafische Konfiguration aller Proxy-Einstellungen:

| Tab | Inhalt |
| --- | --- |
| 🤖 **Modelle** | vLLM URLs, Hauptmodell (80B), Fast-Modell (27B), Proxy-Port, API-Key, Auth/Chatty-Toggle |
| ☁️ **Cloud-APIs** | Cloud-Planer aktivieren, API-URL/Key, Modell, Max Tokens, LiteLLM (OpenRouter/DeepSeek/Claude) |
| 🎯 **Tokens & Timeouts** | Slider für alle Token-Budgets und Timeouts |
| ⚙️ **Features** | Caveman, Hindsight, Verifikation, MCP — alle als Toggle |
| 🧠 **Hindsight** | Qdrant/JSONL, Embedding-Dim, Recall-Parameter, Memory löschen |
| ✅ **Verifikation** | Lint- und Test-Befehle für Phase 3 |

Die Konfiguration wird in `config.json` gespeichert und beim nächsten Neustart geladen (Env-Variablen haben Vorrang).
