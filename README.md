# LocalProxy v3.0 — Single-Model Pass-Through Proxy

Ein OpenAI-kompatibler **Single-Model-Pass-Through-Proxy** fur VS Code Copilot.
Der Proxy leitet Anfragen 1:1 an genau EIN konfiguriertes Modell weiter und streamed
die Antwort zuruck. **Keine** internen Agent-Loops, **keine** Planner-Worker-Architektur.

## Architektur

```
VS Code Copilot  --POST /v1/chat/completions-->  FastAPI Gateway
  |  (tools, messages von VS Code Host)            |
  |                                                 |
  |  1. Auth (PROXY_API_KEY)                        |
  |  2. Prompt-Flag extrahieren (--light etc.)      |
  |  3. Kategorie auswahlen (oder WebUI-Default)    |
  |  4. Hindsight Recall (System-Message-Prafix)    |
  |  5. Payload bauen + transparente Mods:           |
  |     - image_url strippen (wenn is_vision=false)  |
  |     - Moonshot-Patch (nur bei moonshot-ai URL)  |
  |     - Tool-Result-Capping (Token-Bombing)       |
  |  6. Request ans Modell --> Stream zuruck        |
  |  7. Background: Hindsight Retain                |
  v                                                 v
```

## 5 Modell-Kategorien

| Flag | Kategorie | Typischer Use-Case |
|------|-----------|-------------------|
| `--local` | Lokales Modell (vLLM/Qwen) | Offline, low-latency |
| `--coworker` | Co-Worker (2. lokales Modell, separate Hardware) | Delegations-Ziel via `ask_coworker`-Tool |
| `--light` | Schnelles Cloud-Modell (GPT-4.1-mini) | **Default**, schnelle Tasks |
| `--strong` | Leistungsstarkes Modell (Claude Sonnet) | Komplexe Architektur, grose Rewrites |
| `--vision` | Multimodales Modell (GPT-4o) | Bilder, Screenshots, Diagrams |

Ohne Flag wird die in der **WebUI** konfigurierte Default-Kategorie verwendet.

Jede Kategorie ist vollstandig konfigurierbar:
- `api_url` — OpenAI-kompatibler Endpoint
- `api_key` — API-Key (Bearer-Token)
- `model_name` — Name des Modells fur den Endpoint
- `max_tokens` — max. Antwort-Tokens
- `is_vision` — wenn `true`, werden image_url-Parts nicht gestrippt
- `timeout_seconds` — Timeout pro Request

## Co-Worker-Delegation (ask_coworker)

Bei aktiver Kategorie `--local` injiziert der Proxy ein synthetisches Tool
`ask_coworker(task, context)` in den Payload — **nur wenn** der Health-Check
das Co-Worker-Modell als erreichbar meldet (Hauptrechner an). Ruft das
Hauptmodell das Tool auf, arbeitet der Proxy den Call intern an das
Co-Worker-Modell ab (frische, minimale Session — keine VS-Code-History, kein
Thinking-Leak, keine VS-Code-Tools) und ruft das Hauptmodell mit dem Ergebnis
erneut auf. Fuer VS Code bleibt der gesamte Umweg unsichtbar; während der
Delegation laufen die SSE-Keepalives weiter (kein Timeout).

- Konfiguration: Kategorie `coworker` (single-dict, wie `local`) in der WebUI
- Einstellungen unter `tokens.coworker`: `enabled`, `max_delegations_per_request`,
  `task_cap_chars`, `result_cap_chars`, `health_interval_seconds`,
  `probe_timeout_seconds`, `system_prompt`
- Health-Check: periodischer Ping (Intervall konfigurierbar); bei Ausfall wird
das Tool nicht mehr injiziert, ein laufender Call liefert einen Fehlertext als
tool-result (Hauptmodell antwortet trotzdem)
- Gemischte Turns (ask_coworker + VS-Code-Tool gleichzeitig) werden mit einem
Hinweis an das Modell zurueckgegeben (keine fragile History-Rekonstruktion)
- Env-Vars: `COWORKER_API_URL/KEY/MODEL_NAME/...`, `COWORKER_ENABLED`,
  `COWORKER_MAX_DELEGATIONS`, `COWORKER_TASK_CAP`, `COWORKER_RESULT_CAP`,
  `COWORKER_HEALTH_INTERVAL`, `COWORKER_PROBE_TIMEOUT`, `COWORKER_SYSTEM_PROMPT`

## Quickstart

```powershell
# 1. Python-Environment
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Konfiguration via WebUI
uvicorn proxy:app --host 0.0.0.0 --port 9001
# Offne http://localhost:9001/webui/
# Login: SPARK_AUTH_USERNAME / SPARK_AUTH_PASSWORD (default: admin / auto-generated)

# 3. Modell-Kategorien in der WebUI konfigurieren:
#    - light: api_url + api_key eintragen
#    - vision: api_url + api_key + is_vision=true
#    - ...
#    Speichern --> Proxy-Config wird neu geladen

# 4. VS Code Copilot konfigurieren
#    Endpoint: http://localhost:9001/v1
#    API-Key:  <aus Proxy-Startup-Log oder WebUI>
```

## Prompt-Flags

Ans Ende einer User-Nachricht einfugen (wird vom Proxy entfernt bevor der Payload
ans Modell geht):

```
Schreibe ein Python-Script --light
Erklare diesen Screenshot --vision
Refactore die Architektur --strong
```

## Features

- **Hindsight Memory**: Qdrant/JSONL-basiertes persistentes Gedachtnis.
  Recall-Ergebnisse werden als `[HINDSIGHT MEMORY CONTEXT]` System-Message
  jedem Request vorangestellt.
- **Transparente Modifikationen**: image_url-Sanitizer fur text-only Modelle,
  Moonshot-Parameter-Patch (temp=1.0, top_p=0.95, penalties=0) nur bei
  moonshot-ai URL.
- **Tool-Result-Capping**: Verhindert Token-Bombing durch grosse grep/read-Results.
- **Read-Loop-Detection**: Erkennt wenn ein Modell dieselbe Datei mit denselben
  Zeilen >N mal hintereinander liest (Default: >3) und injiziert eine
  Interventions-Message ("STOP LOOPING!..."). Konfigurierbar via `READ_LOOP_THRESHOLD`
  und `READ_LOOP_INTERVENTION` (Env oder WebUI).
- **WebUI**: Login-gesichertes Dashboard, 4 Modell-Karten mit Test-Endpunkt,
  Live-Config-Reload via `_apply_config_file()`.
- **Debug-Endpoints**: `/debug/files`, `/debug/file/{id}`, `/debug/ring`,
  `/debug/active`, `/debug/cleanup` fur Payload-Inspection und
  Diagnose hangender Calls.

## Endpoints

| Endpoint | Beschreibung |
|----------|-------------|
| `POST /v1/chat/completions` | OpenAI-kompatibler Chat-Endpoint (Auth) |
| `GET /v1/models` | Liste der 4 Modell-Kategorien |
| `GET /healthz` | Proxy-Status |
| `GET /logs`, `GET /v1/logs` | Letzte Log-Zeilen |
| `GET /debug/*` | Payload-Dumps + Diagnostics |
| `/webui/` | Konfigurations-Dashboard |

## Migration von v2.x

- Das Config-Schema hat sich grundlegend geandert (keine `cloud`, `litellm`,
  `verify`, `mcp`-Sektionen mehr; stattdessen `model_categories`).
- Beim ersten Start mit alter `config.json`: Default-Config wird verwendet.
- Empfehlung: alte `config.json` sichern, Proxy starten, WebUI offnen und
  manuell die 4 Kategorien konfigurieren.

## Environment Variables (optional)

Alle 4 Kategorien konnen via Env-Vars konfiguriert werden (Prefix je Kategorie):

- `LOCAL_API_URL`, `LOCAL_API_KEY`, `LOCAL_MODEL_NAME`, `LOCAL_MAX_TOKENS`,
  `LOCAL_IS_VISION`, `LOCAL_TIMEOUT_SECONDS`
- `LIGHT_*`, `STRONG_*`, `VISION_*` (analog)
- `DEFAULT_CATEGORY` — `light` (default)
- `PROXY_PORT`, `PROXY_AUTH_ENABLED`, `PROXY_API_KEY`
- `HINDSIGHT_ENABLED`, `QDRANT_URL`, ...

## Lizenz

MIT
