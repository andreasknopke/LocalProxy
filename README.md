# LocalProxy v2.1 — Hybrider Agentischer Routing-Proxy

Ein OpenAI-kompatibler FastAPI-Proxy für lokale vLLM-/Qwen-Coder-Modelle auf dem **DX Spark GB10 (128 GB VRAM)**.

Der Proxy entscheidet pro Request zwischen zwei Routen – mit automatischem Cloud-Fallback bei lokalen Timeouts:

| Route | Auslöser | Ablauf |
| --- | --- | --- |
| **Direkt lokal** | Kurze Prompts, Autocomplete, Inline-Fix, Rename, Tool-Continuations | Prompt → Qwen 80B → Antwort (bei Timeout: Cloud-Fallback) |
| **Hybrid agentisch** | Refactor, Bug, Architektur, Tests, Security, komplexe Tasks | Hindsight Recall → Cloud-Planer (Caveman) → Worker (80B) → **Verify** → Cloud-Fallback |

## Architektur

```
+----------------------------------------------------------------+
|                         VS Code                                |
|  (Continue / Cline / Roo-Code / GitHub Copilot Chat)           |
+----------------------------------------------------------------+
                               |
                               v  (OpenAI REST API / SSE Streaming)
+----------------------------------------------------------------+
|                 FastAPI Proxy Gateway v2.1                      |
|                                                                 |
|  ┌─────────────────────────────────────────────────────────┐   |
|  │  URL-Normalisierung  (/v1 → /v1/chat/completions etc.)  │   |
|  │  Auth (Bearer Token / API-Key)                           │   |
|  └─────────────────────────────────────────────────────────┘   |
|                                                                 |
|  ┌─────────────── Intent Classifier ───────────────────────┐   |
|  │  Deterministisch (Trigger-Wörter + Textlänge)            │   |
|  │  ⇅ 27B Fast Model (bei Mehrdeutigkeit)                  │   |
|  │  ⇅ Tool-Continuation Detection (Bypass aller Pipelines) │   |
|  └─────────────────────────────────────────────────────────┘   |
|         │                                          │            |
|   »agent« (komplex)                           »direct« (kurz)  |
|         │                                          │            |
|  ┌──────┴──────────────────────┐    ┌──────────────┴────────┐  |
|  │  3-Phasen Agent-Workflow    │    │  Direkt Lokal          │  |
|  │                             │    │                        │  |
|  │  Phase 1: Hindsight Recall  │    │  reasoning_content     │  |
|  │     ↓                       │    │  Cache-Re-Injection    │  |
|  │  Cloud-Planer (Caveman)     │    │     ↓                  │  |
|  │   · Moonshot/Kimi Patch     │    │  DSML <tool_call>      │  |
|  │   · LiteLLM / HTTPX direkt  │    │  Detection             │  |
|  │     ↓                       │    │     ↓                  │  |
|  │  Phase 2: Worker (80B)      │    │  vLLM Call             │  |
|  │   · VS Code Tools erhalten  │    │  (Qwen3-Next-80B)      │  |
|  │   · Plan+Memory als Kontext │    │     │                  │  |
|  │     ↓                       │    │     │                  │  |
|  │  Phase 3: Verify            │    │     ▼                  │  |
|  │   · Linter + Tests          │    │  ┌──────────────┐     │  |
|  │   · Self-Correction (80B)   │    │  │ Timeout/Error?│     │  |
|  │                             │    │  └──┬───────┬───┘     │  |
|  └──────────┬──────────────────┘    │   Ja│      │Nein     │  |
|             │                       │     ▼      ▼          │  |
|             │                       │ Cloud-    Antwort     │  |
|             │                       │ Fallback  (Streaming  │  |
|             │                       │   │       / Non-      │  |
|             │                       │   │       Stream)     │  |
|             └───────────┬───────────┴───┘                   │  |
|                         │                                    │  |
|  ┌──────────────────────┴──────────────────────────────────┐ │  |
|  │  Antwort-Rendering                                      │ │  |
|  │  · Pipeline-Summary (Chatty Mode)                        │ │  |
|  │  · Caveman-Plan + Worker-Ergebnis (Agent Mode)           │ │  |
|  │  · Hindsight Retain (asynchron)                          │ │  |
|  │  · reasoning_content Cache (für Tool-Continuations)      │ │  |
|  └──────────────────────┬──────────────────────────────────┘ │  |
|                         │                                     |
+----------------------------------------------------------------+
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
+──────────────+  +──────────────+  +──────────────────+
| Lokales vLLM |  |  Cloud-API   |  |  Cloud-Fallback  |
| (Spark GB10) |  |  (Planung)   |  |  (bei Timeout)   |
|              |  |              |  |                  |
| Qwen3-Next   |  | Moonshot     |  | 1. LiteLLM       |
| 80B-Chat     |  | Kimi K2.7    |  |    (DeepSeek)    |
| mxfp4        |  | ODER         |  | 2. Cloud Review  |
|              |  | LiteLLM:     |  |    (Moonshot)    |
| Qwen3.6-27B  |  | · DeepSeek   |  |                  |
| Chat-FP8     |  | · Claude     |  |                  |
|              |  | · OpenRouter |  |                  |
+──────────────+  +──────────────+  +──────────────────+
```

## Komponenten

### 1. Intent-Klassifizierung
- **Deterministisch:** Regex/Trigger-Wörter + Textlänge (< 60 Zeichen → direct, > 500 → agent)
- **Fast Model (27B):** Bei mehrdeutigen Prompts klassifiziert das schnelle 27B-Modell
- **Tool-Continuation Detection:** Erkennt `tool`/`function`-Roles sowie Cline/Roo `tool_result`-Blöcke → bypassed sofort den Agent-Workflow

### 2. Hindsight Memory (Qdrant + JSONL-Fallback)
Vier logische Netzwerke:
- **World Facts:** API-Endpunkte, Build-/Test-Kontext, Repository-Fakten
- **Agent Experiences:** Erfolgreiche & fehlgeschlagene Lösungsansätze
- **Entity Summaries:** Module, Klassen, Funktionen, Komponenten
- **Evolving Beliefs:** Aktuelle Refactoring- und Architektur-Entscheidungen

Primär wird **Qdrant** als Vektordatenbank genutzt. Falls nicht erreichbar, fallback auf JSONL-Dateien. Embeddings via deterministisches Feature-Hashing (kein externes Modell nötig).

### 3. Caveman Ultra (Prompt-Kompression)
Der Cloud-Planer antwortet im Caveman-Stil: nur Symbole, Pfeile, Keywords. System-Prompt: *"Only compact symbols, arrows, terse keywords. No filler, no grammar, no prose."* Senkt Token-Verbrauch um 60–75 %.

### 4. 3-Phasen-Agenten-Workflow
1. **Phase 1 – Plan:** Hindsight Recall → Cloud-Planer (Moonshot Kimi K2.7 / LiteLLM) erstellt Caveman-Plan
2. **Phase 2 – Execute:** Qwen 80B Worker führt den Plan aus – mit VOLLER VS Code Tool-Umgebung (System-Prompt, Tools, Tool-Choice bleiben erhalten; Plan + Memory werden nur an die User-Message angehängt)
3. **Phase 3 – Verify:** Linter/Tests + Self-Correction durch das 80B-Modell. Übersprungen bei Tool-Calls oder DSML-Output.

### 5. LiteLLM & HTTPX Cloud-Routing
- **LiteLLM Library:** Optional (`pip install litellm`), unterstützt Provider-Präfixe (`openrouter/`, `deepseek/`, etc.)
- **HTTPX direkt:** Funktioniert OHNE LiteLLM-Package – setze einfach `LITELLM_CLOUD_API_URL` auf den OpenAI-kompatiblen Endpoint
- Unterstützt OpenAI, Anthropic, DeepSeek, OpenRouter, Moonshot/Kimi und alle OpenAI-kompatiblen Provider

### 6. MCP-Server (Model Context Protocol)
原生 MCP-Integration für VS Code (Continue/Cline/Copilot):
- `localproxy_read_file` / `localproxy_write_file`
- `localproxy_list_files` / `localproxy_search_code`
- `localproxy_run_terminal`
- `localproxy_hindsight_recall` / `localproxy_get_status`

### 7. Cloud-Fallback (automatisch)
Wenn der lokale vLLM-Server timeoutet oder einen Fehler zurückgibt, schaltet der Proxy automatisch auf eine Cloud-Fallback-Kaskade:
1. **LiteLLM** (DeepSeek o.ä.) – falls konfiguriert
2. **Cloud Reviewer** (Moonshot Kimi K2.7) – falls konfiguriert

Die Antwort wird mit `[⚠️ Lokal/Free nach Xs nicht verfügbar – Antwort via ...Cloud-Fallback]` markiert. Der Fallback respektiert alle Timeout-Konfigurationen.

### 8. DeepSeek reasoning_content Handling
DeepSeek-Modelle erfordern, dass `reasoning_content` aus Assistant-Messages bei Folge-Requests mit Tool-Calls erhalten bleibt. VS Code speichert dieses Feld nicht – daher:
- **Cache:** `reasoning_content` wird pro `tool_call_id` gespeichert
- **Re-Injektion:** Vor jedem vLLM-Call werden fehlende `reasoning_content`-Felder automatisch aus dem Cache ergänzt
- **Eviction:** Bei >1000 Einträgen werden die ältesten 500 entfernt

### 9. DSML / Tool-Call Detection
Erkennt Tool-Calls in verschiedenen Formaten:
- **Structured:** OpenAI `tool_calls` Array in der Response
- **DSML:** `<tool_call>`, `<｜｜DSML｜｜tool_calls>`, `<invoke>` XML-Tags
- **Copilot:** `｜｜tool_calls`, `｜｜invoke` Marker
- **Patterns:** `callTool`, Funktionsaufruf-Muster

Bei erkannten Tool-Calls wird die Pipeline-Summary unterdrückt und der Output raw an den Client durchgereicht (Streaming wie Non-Streaming).

### 10. Provider-Kompatibilität
- **Moonshot/Kimi Patch:** Erzwingt `temperature=1.0`, `top_p=0.95`, entfernt `top_k`, setzt Penalties auf 0
- **URL-Normalisierung:** Auto-Append von `/chat/completions`, `/v1/models` an beliebige Base-URLs
- **Heartbeat-Logging:** Alle 30s Status-Update bei langen vLLM-Calls (via `CHATTY_HEARTBEAT_SECONDS` konfigurierbar)

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
LITELLM_CLOUD_MODEL="deepseek/deepseek-chat" \
LITELLM_CLOUD_API_KEY="sk-..." \
python proxy.py
```

### Hybrid-Modus mit LiteLLM API-URL (ohne litellm-Package)

```bash
VLLM_API_URL=http://localhost:8000/v1/chat/completions \
MODEL_NAME="Qwen/Qwen3-Next-80B-Chat-mxfp4" \
CLOUD_REVIEW_ENABLED=true \
LITELLM_CLOUD_MODEL="deepseek-v4" \
LITELLM_CLOUD_API_KEY="sk-..." \
LITELLM_CLOUD_API_URL="https://api.deepseek.com/v1/chat/completions" \
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
| `VLLM_API_KEY` | leer | API-Key für lokales vLLM (falls benötigt) |
| `MODEL_NAME` | `Qwen/Qwen3-Next-80B-Chat-mxfp4` | Hauptmodell (Worker) |
| `FAST_MODEL_NAME` | `Qwen/Qwen3.6-27B-Chat-FP8` | Schnelles Modell (Klassifikation) |
| `PROXY_PORT` | `9001` | Proxy-Port |
| `PROXY_AUTH_ENABLED` | `true` | API-Key-Authentifizierung |
| `PROXY_API_KEY` | auto-generiert (`localfox-...`) | API-Key (bei Auth) |
| `CHATTY_MODE` | `true` | Status-Updates im Chat-Output |
| `CHATTY_HEARTBEAT_SECONDS` | `15` | Intervall für Heartbeat-Logs bei langen Calls |
| `LOG_FILE` | `./proxy.log` | Pfad zur Log-Datei |
| `CLOUD_REVIEW_ENABLED` | `false` | Cloud-Planer aktivieren |
| `CLOUD_REVIEW_API_URL` | `https://api.openai.com/v1/chat/completions` | Cloud-API-URL |
| `CLOUD_REVIEW_API_KEY` | leer | Cloud-API-Key |
| `CLOUD_REVIEW_MODEL` | `gpt-4.1-mini` | Cloud-Modell |
| `CLOUD_REVIEW_MAX_TOKENS` | `128000` | Max. Tokens Cloud |
| `CLOUD_REVIEW_TIMEOUT_SECONDS` | `180` | Timeout Cloud (Sekunden) |
| `LITELLM_CLOUD_MODEL` | leer | LiteLLM-Modell (z.B. `deepseek/deepseek-chat`) |
| `LITELLM_CLOUD_API_KEY` | leer | LiteLLM API-Key |
| `LITELLM_CLOUD_API_URL` | leer | LiteLLM Endpoint-URL (für HTTPX-Direktaufruf) |
| `LITELLM_CLOUD_MAX_TOKENS` | `16384` | Max. Tokens LiteLLM |
| `LITELLM_CLOUD_TIMEOUT_SECONDS` | `180` | Timeout LiteLLM (Sekunden) |
| `CAVEMAN_ENABLED` | `true` | Caveman-Prompt-Injektion |
| `CAVEMAN_MAX_TOKENS` | `8192` | Max. Tokens Caveman-Plan |
| `HINDSIGHT_ENABLED` | `true` | Hindsight Memory |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant-URL |
| `QDRANT_API_KEY` | leer | Qdrant-API-Key |
| `HINDSIGHT_USE_QDRANT` | `false` (auto bei non-localhost QDRANT_URL) | Qdrant statt JSONL |
| `HINDSIGHT_DIR` | `./.hindsight_memory` | JSONL-Speicherort |
| `HINDSIGHT_EMBEDDING_DIM` | `768` | Embedding-Dimension |
| `HINDSIGHT_MAX_MEMORY_TOKENS` | `4000` | Token-Budget Recall-Kontext |
| `HINDSIGHT_MIN_SIMILARITY` | `0.18` | Min. Ähnlichkeit Recall |
| `HINDSIGHT_RETAIN_DELAY_SECONDS` | `0` | Verzögerung vor Memory-Speicherung |
| `VERIFY_ENABLED` | `true` | Phase-3-Verifikation |
| `VERIFY_LINT_COMMAND` | leer | Lint-Befehl (z.B. `ruff check`) |
| `VERIFY_TEST_COMMAND` | leer | Test-Befehl (z.B. `pytest -x`) |
| `VERIFY_TIMEOUT_SECONDS` | `120` | Timeout Verifikation (Sekunden) |
| `MCP_ENABLED` | `true` | MCP-Server aktiv |
| `DIRECT_MAX_TOKENS` | `65536` | Tokens für direkte Requests |
| `SUB_AGENT_MAX_TOKENS` | `65536` | Tokens für Agent-Worker |
| `SUB_AGENT_TIMEOUT_SECONDS` | `300` | Timeout pro lokaler Anfrage (min. 120s für DFlash JIT) |

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

Konfiguriere **Continue**, **Cline**, **Roo-Code** oder **GitHub Copilot Chat**:

```json
{
  "model": "Qwen/Qwen3-Next-80B-Chat-mxfp4",
  "apiBase": "http://localhost:9001/v1",
  "apiKey": "YOUR_PROXY_API_KEY"
}
```

Für schnelle Responses (Autocomplete, Inline-Fixes) kann auch das 27B-Modell direkt angewählt werden:
```json
{
  "model": "Qwen/Qwen3.6-27B-Chat-FP8",
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

### Konfigurations-Priorität

Der Proxy lädt Einstellungen aus drei Quellen – Priorität (niedrig → hoch):

1. **Code-Defaults** (in `proxy.py` hartkodiert)
2. **`config.json`** (via WebUI gespeichert, über `LOCALPROXY_CONFIG` konfigurierbar)
3. **`.env` / Environment** (höchste Priorität, überschreibt alles)

API-Keys und Secrets werden in `config.json` gespeichert (niedrigere Priorität als Env), URLs und Token-Budgets dagegen nur per Env überschreibbar.

## Web-Konfigurationsinterface 🌐

Der Proxy enthält ein eingebautes Web-Dashboard unter:

```text
http://0.0.0.0:9001/webui/
```

Das Dashboard ermöglicht die grafische Konfiguration aller Proxy-Einstellungen:

| Tab | Inhalt |
| --- | --- |
| 🤖 **Modelle** | vLLM URLs, Hauptmodell (80B), Fast-Modell (27B), Proxy-Port, API-Key, Auth/Chatty-Toggle |
| ☁️ **Cloud-APIs** | Cloud-Planer aktivieren, API-URL/Key, Modell, Max Tokens, LiteLLM (Modell, Key, API-URL) |
| 🎯 **Tokens & Timeouts** | Slider für alle Token-Budgets (bis 65536) und Timeouts (bis 300s) |
| ⚙️ **Features** | Caveman, Hindsight, Verifikation, MCP — alle als Toggle |
| 🧠 **Hindsight** | Qdrant/JSONL, Embedding-Dim, Recall-Parameter, Retain-Delay, Memory löschen |
| ✅ **Verifikation** | Lint- und Test-Befehle für Phase 3 |

Die Konfiguration wird in `config.json` gespeichert und beim nächsten Neustart geladen (Env-Variablen haben Vorrang).

## Docker-Deployment (Coolify)

Der Proxy kann einfach als Docker-Container auf **Coolify** (oder jedem anderen Docker-Host) deployed werden.

### Schnellstart

```bash
docker build -t localproxy .
docker run -d --name localproxy \
  -p 9001:9001 \
  -e PROXY_AUTH_ENABLED=false \
  -e VLLM_API_URL=http://192.168.1.100:8000/v1/chat/completions \
  -e MODEL_NAME="Qwen/Qwen3-Next-80B-Chat-mxfp4" \
  localproxy
```

### Coolify-Konfiguration

| Einstellung | Wert |
|-------------|------|
| **Build Pack** | `Dockerfile` |
| **Port** | `9001` |
| **Healthcheck** | `/healthz` → wird automatisch via `HEALTHCHECK`-Instruction geprüft |

Alle Umgebungsvariablen aus der [Konfigurationstabelle](#konfiguration) können in Coolify unter **Environment Variables** gesetzt werden.

> **⚠️ Wichtig — Env-Vars vs. WebUI:**
> **Beim ersten Start** werden die Env-Vars aus Coolify in `config.json` übernommen
> und im WebUI angezeigt. **Speicherst du danach via WebUI**, hat `config.json` Vorrang
> — auch wenn die Coolify-Env-Vars noch gesetzt sind.
>
> **Reset:** Wenn du wieder die Coolify-Env-Vars anwenden willst, lösche `config.json`
> im Volume und starte den Container neu. Dann wird es neu aus den Env-Vars erzeugt.
>
> **💾 Storage-Mount für Coolify:**
> Damit WebUI-Änderungen dauerhaft sind, in Coolify unter **Storage** → **New mount**:
> - **Source Path:** z.B. `/var/lib/coolify/proxy-data` (ein Verzeichnis auf dem Docker-Host)
> - **Destination Path:** `/app/data`
>
> Nach dem ersten Speichern via WebUI liegt die Konfiguration unter `/app/data/config.json`
> und bleibt auch nach Redeploy erhalten. Gleicher Ordner enthält `proxy.log` und `.hindsight_memory/`.

> **💡 Tipp — LiteLLM-Modellnamen ohne Provider-Prefix:**
> In Coolify-Envs `LITELLM_CLOUD_MODEL` **ohne** `deepseek/`-Prefix setzen, da der Proxy
> direkt per HTTPX (ohne LiteLLM-Library) an die DeepSeek-API sendet:
> - ✅ `LITELLM_CLOUD_MODEL=deepseek-v4-pro`
> - ❌ `LITELLM_CLOUD_MODEL=deepseek/deepseek-v4-pro`
>
> Die LiteLLM-Library benötigt den Prefix (`deepseek/`), der HTTPX-Direktaufruf nicht.

### docker-compose.yml (für Qdrant + Proxy)

```yaml
version: "3.8"
services:
  localproxy:
    build: .
    ports:
      - "9001:9001"
    environment:
      - PROXY_AUTH_ENABLED=false
      - VLLM_API_URL=http://192.168.1.100:8000/v1/chat/completions
      - MODEL_NAME=Qwen/Qwen3-Next-80B-Chat-mxfp4
      - HINDSIGHT_USE_QDRANT=true
      - QDRANT_URL=http://qdrant:6333
      # LiteLLM ohne Provider-Prefix (HTTPX-Direktmodus)
      - LITELLM_CLOUD_MODEL=deepseek-v4-pro
      - LITELLM_CLOUD_API_KEY=sk-...
      - LITELLM_CLOUD_API_URL=https://api.deepseek.com/v1/chat/completions
    volumes:
      # Daten persistent machen (config.json, Logs, Hindsight Memory)
      - localproxy_data:/app/data
    depends_on:
      - qdrant

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage

volumes:
  qdrant_storage:
  localproxy_data:
```
