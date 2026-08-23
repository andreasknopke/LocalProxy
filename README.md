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
das Co-Worker-Modell als erreichbar meldet (Hauptrechner an). Mit aktivem
Fork-Join (siehe unten) kommen `dispatch_coworker` / `collect_coworker` hinzu.
Ruft das Hauptmodell das Tool auf, arbeitet der Proxy den Call intern an das
Co-Worker-Modell ab (frische, minimale Session — keine VS-Code-History, kein
Thinking-Leak, keine VS-Code-Tools) und ruft das Hauptmodell mit dem Ergebnis
erneut auf.

### Live-Streaming + Stream-Inject (Kategorie `local`)

Anders als die Cloud-Kategorien (2-Pass: erst komplett rechnen, dann senden)
streamt der Proxy bei `--local` das Backend live mit `stream=True`:

- **Thinking (`reasoning_content`) und Antwort fliessen live an VS Code durch**
  — der User sieht sofort, dass das Modell arbeitet (wichtig bei langsamen
  lokalen Modellen, keine leeren Phasen, kein Timeout-Gefuehl).
- **Reasoning-Mapping**: Reasoning wird einheitlich als `reasoning_content` an
  VS Code gemappt — egal in welcher Struktur das Backend liefert:
  `reasoning_content` (DeepSeek/vLLM-Qwen3-Parser), `reasoning` (Ollama/LM
  Studio, auch als Liste/Objekt), `thinking` oder `<think>...</think>`-Tags im
  content (vLLM Qwen3 mit `preserve_thinking`).
- **Co-Worker-Delegation wird per Stream-Inject sichtbar gemacht:**
  1. `[Proxy] Delegation an Co-Worker: <task>…` — sobald übergeben wird
  2. das **Reasoning des Co-Workers streamt live als eigener Reasoning-Context**
     an VS Code durch — man sieht, worüber der Co-Worker nachdenkt
  3. die **Co-Worker-Antwort streamt live token-für-token** an VS Code durch
     (mit `[Proxy] Co-Worker-Antwort:`-Header) — kein Text-Burst am Ende,
     keine toten Phasen während der Co-Worker-Arbeit, kein Timeout-Gefühl
  4. danach streamt das Hauptmodell live weiter — man sieht, was es mit der
     Antwort macht
- Keepalives halten die Verbindung auch bei langen Denkpausen und während der
  Co-Worker-Arbeit am Leben (kein 300s-Timeout).
- VS-Code-Tools (`read_file` etc.) werden unverändert an Copilot durchgereicht.

- Konfiguration: Kategorie `coworker` (single-dict, wie `local`) in der WebUI
- Einstellungen unter `tokens.coworker`: `enabled`, `max_delegations_per_request`,
  `task_cap_chars`, `result_cap_chars`, `files_cap_chars`, `health_interval_seconds`,
  `probe_timeout_seconds`, `system_prompt` (Fork-Join-Keys siehe unten)
- **Automatischer Datei-Kontext**: Bei jedem `ask_coworker`-Call haengt der Proxy
  die Dateiinhalte aus dem Chat automatisch an die Co-Worker-Session an
  (VS-Code-Attachments + `read_file`/Search-Tool-Ergebnisse, dedupliziert nach
  Pfad/Inhalt, gedeckelt durch `files_cap_chars`, default 60000). Damit bekommt
  der Co-Worker auch bei komplexen Fragen IMMER die relevanten Dateiinhalte —
  selbst wenn das Hauptmodell sie nicht in `task`/`context` uebernommen hat.
  0 = deaktiviert.
- Health-Check: periodischer Ping (Intervall konfigurierbar); bei Ausfall wird
das Tool nicht mehr injiziert, ein laufender Call liefert einen Fehlertext als
tool-result (Hauptmodell antwortet trotzdem)
- Gemischte Turns (ask_coworker + VS-Code-Tool gleichzeitig) werden mit einem
Hinweis an das Modell zurueckgegeben (keine fragile History-Rekonstruktion)
- Env-Vars: `COWORKER_API_URL/KEY/MODEL_NAME/...`, `COWORKER_ENABLED`,
  `COWORKER_MAX_DELEGATIONS`, `COWORKER_TASK_CAP`, `COWORKER_RESULT_CAP`,
  `COWORKER_FILES_CAP`, `COWORKER_HEALTH_INTERVAL`, `COWORKER_PROBE_TIMEOUT`,
  `COWORKER_SYSTEM_PROMPT`

### Prefill-Progress-Polling (llama.cpp)

Waehrend ein llama.cpp-Server den Prompt verarbeitet (Prefill) sendet er
**keine** Tokens — ein OpenAI-Client wie VS Code zeigt in dieser Phase nur
"Reasoning" und man weiss nicht, wie lange der Prefill noch dauert. Der Proxy
pollt deshalb den `/slots`-Endpoint des Servers und streamt den Fortschritt
als `reasoning_content` an VS Code (sichtbar im "Reasoning"-Panel):

```
⏳ Prefill läuft…
⏳ Prefill 10% · 1234/12340 Tokens · ~18s verbleibend
⏳ Prefill 40% · 4936/12340 Tokens · ~12s verbleibend
⏳ Prefill 100% · 8.1s
```

- **Erkennung**: standardmaessig ueber **Port 8082** in der `api_url`
  (`http://host:8082/...`) — das Erkennungsmerkmal des lokalen llama.cpp.
  Ueberschreibbar per Modell-Flag `prefill_progress` (bool) in `config.json`.
- **Mechanik**: paralleler Poller neben dem SSE-Stream; stoppt automatisch,
  sobald das erste Token eintrifft (= Prefill fertig). Kein Einfluss auf den
  Antwort-/Reasoning-Inhalt (Progress zaehlt nicht gegen das Reasoning-Cap).
- **Neues /slots-Schema**: moderne llama.cpp-Builds liefern in `/slots` keine
  `state`/`prompt.progress`-Felder mehr, sondern `is_processing` +
  `n_prompt_tokens(_processed/_cache)`. Der Proxy unterstuetzt BEIDE Schemata.
  Da das neue Schema die **Gesamt-Tokens nicht** enthaelt, wird das Total
  einmalig per `POST /tokenize` (Messages + Tools) geschaetzt — fuer die
  Prozent-Anzeige. Die absolute Token-Zahl aus `/slots` ist exakt und wird
  immer mit angezeigt (bei fehlender Schaetzung: `Prefill: 57369 Tokens · 462 t/s`).
- **Env-Vars**: `PREFILL_PROGRESS_ENABLED` (default `true`),
  `PREFILL_PROGRESS_PORTS` (default `8082`, kommasepariert),
  `PREFILL_POLL_INTERVAL` (default `1.0`s), `PREFILL_POLL_TIMEOUT` (default
  `2.0`s), `PREFILL_PROGRESS_STEP` (default `10`%), `PREFILL_TOKEN_EMIT_STEP`
  (default `2000` Tokens, Fallback-Raster ohne bekannte Gesamt-Tokens).
- Hinweis: `/slots` korreliert bei mehreren gleichzeitigen Requests nicht
  eindeutig mit dem eigenen Slot — fuer den typischen Ein-Nutzer-Betrieb
  (ein Request zur Zeit) reicht die Heuristik "aktivster Prefill-Slot".

### Fork-Join Fabric (v3.2): `dispatch_coworker` / `collect_coworker`

Zusaetzlich zum blockierenden `ask_coworker` stehen drei Tools fuer parallele
Hintergrundarbeit zur Verfuegung — **Fork-Join** statt sequenzieller Delegation:

```mermaid
flowchart LR
    U[User fragt Hauptmodell] --> M["Hauptmodell (--local)"]
    M -->|dispatch_coworker task| P[Proxy: Task-Store]
    P -->|task_id sofort| M
    P -->|asyncio + Semaphore| CW[Co-Worker auf DGX Spark]
    M -->|weiterarbeiten oder VS-Code-Tools] M
    M -->|collect_coworker task_ids| J[Proxy: Join]
    J -->|Ergebnisse als tool-result| M
    M --> F[Finale Antwort]
```

**Tool-Semantik:**

| Tool | Verhalten |
|------|-----------|
| `ask_coworker(task, context)` | **Blockierend** (wie bisher): Ergebnis als tool-result, danach Hauptmodell-Weiterverarbeitung |
| `dispatch_coworker(task)` | **Fork**: registriert Hintergrund-Task im Store, liefert sofort `{"task_id": "cw_…", "status": "dispatched"}` als mini tool-result. Hauptmodell arbeitet sofort weiter (oder ruft VS-Code-Tools im selben Turn auf — mixed-turn-safe) |
| `collect_coworker(task_ids?, timeout_seconds?=600)` | **Join**: blockt bis Ergebnisse fertig sind oder Timeout, liefert alle summaries als JSON tool-result. Unbekannte/abgelaufene IDs werden als `unknown`/`expired` gemeldet |

**Status-Injection:** Sind undone Hintergrund-Tasks im Store, startet der
Proxy die Folgekonversation mit einem Status-Block:

```
[Proxy] 2 Co-Worker Hintergrund-Task(s) aktiv:
- ⏳ cw_abc12345: Implement parser tests
- ✅ cw_def67890: Review config.json — Rufe collect_coworker auf, um Ergebnisse abzuholen.
```

Der Co-Worker-Worker-Pool laeuft unter einem globalen Semaphore
(`max_parallel`, default 8) — weitere dispatches queuedn automatisch, bis ein
Slot frei wird.

**Konfiguration** (`tokens.coworker` in `data/config.json` bzw. WebUI):

| Key | Default | Bedeutung |
|-----|---------|-----------|
| `enable_fork_join` | `true` | Schalter fuer dispatch/collect (false = nur `ask_coworker`) |
| `max_parallel` | `8` | Gleichzeitige Hintergrund-Tasks (Semaphore) |
| `dispatch_cap_per_request` | `12` | Max. dispatches pro User-Request (Kreislaufschutz) |
| `bg_ttl_seconds` | `1800` | Task-Lebensdauer; laufende Tasks > TTL werden expired + gecancelt |

- Env-Vars: `COWORKER_FORK_JOIN`, `COWORKER_MAX_PARALLEL`, `COWORKER_DISPATCH_CAP`, `COWORKER_BG_TTL` (zu den bestehenden Co-Worker-Vars hinzu)

**Voraussetzung:** `coworker`-Kategorie (single-dict wie `local`) muss auf den
DGX-Spark-Endpoint zeigen (`api_url` + `model_name` setzen) — ohne Konfiguration
bleibt das Feature deaktiviert (Health-Check greift nicht) und das Tool wird
nicht injiziert.

**Tests:** `tests/test_bg_store.py` (Store, TTL, Semaphore, Status-Zeile,
Delegation-Loop) + Fork-Join-Faelle in `tests/test_stream_inject.py`
(dispatch/collect-Streaming, mixed-turn).

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
- **Transparente Modifikationen**: image_url-Sanitizer fur text-only Modelle.
- **Tool-Result-Capping**: Verhindert Token-Bombing durch grosse grep/read-Results.
- **Read-Loop-Detection**: Erkennt wenn ein Modell dieselbe Datei mit denselben
  Zeilen >N mal hintereinander liest (Default: >3) und injiziert eine
  Interventions-Message ("STOP LOOPING!..."). Konfigurierbar via `READ_LOOP_THRESHOLD`
  und `READ_LOOP_INTERVENTION` (Env oder WebUI).
- **WebUI**: Login-gesichertes Dashboard, 4 Modell-Karten mit Test-Endpunkt,
  Live-Config-Reload via `_apply_config_file()`.
- **Fork-Join Fabric** (v3.2): `dispatch_coworker` / `collect_coworker` fuer
  nicht-blockierende, parallele Co-Worker-Hintergrund-Tasks mit globalem
  Task-Store (Semaphore, TTL, Dispatch-Cap, Status-Injection) — Details im
  Abschnitt Fork-Join Fabric.
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
| `GET /debug/streams` | I/O-Trace-Index: alle Turns mit Live-Analyse (s. unten) |
| `GET /debug/streams/{turn_id}` | Vollständiges I/O eines Client-Turns (meta + events) |
| `DELETE /debug/streams` | Alle Trace-Turns löschen (Rotation erzwingen) |
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

### I/O-Stream-Tracing (`data/io_traces/`)

Vollständiges Full-Duplex-Logging pro Client-Turn — die empirische Basis für
 die Frage *„Sieht das Modell die Co-Worker-Tools überhaupt / ruft es sie
auf?“*. Pro Turn ein Verzeichnis:

```
data/io_traces/turn_<ts>_<uuid>/
  meta.json     Turn-Metadaten + Analyse (coworker_tools_on_wire,
                guidance_in_system, coworker_calls_seen, backend_error, ...)
  events.jsonl  Append-Only: inbound → outbound → backend_resp →
                client_sse → final → bg_result (komplette Bodies, ungekürzt)
```

- Via `ContextVar` bleiben auch BG-Co-Worker-Tasks dem Client-Turn zugeordnet.
- Fail-still: Tracing darf den Proxy-Betrieb nie beeinflussen.
- Env-Vars: `IO_TRACE_DIR` (default ./data/io_traces), `IO_TRACE_ENABLED`
  (1), `IO_TRACE_TTL_HOURS` (24), `IO_TRACE_MAX_BYTES` (209715200),
  `IO_TRACE_MAX_TURNS` (500), `IO_TRACE_SECONDS` (0 = unbegrenzt).

**Diagnose-Workflow** (rein per HTTP-API gegen das deployed Proxy — kein SSH nötig):

```powershell
# 0. Einmalig: API-Key lokal ablegen (gitignored, nie im Repo)
#    .env.proxy-status.local:
#      PROXY_STATUS_API_KEY=<PROXY_API_KEY aus den Coolify-Env-Vars>
#      PROXY_STATUS_HOST=192.168.188.134
#      PROXY_STATUS_PORT=9001
# 1. Statusübersicht: pro Turn Tools-on-Wire / Calls / Guidance / Errors
.\proxy-status.ps1 -Streams 10
# 2. Verdächtigen Turn komplett ziehen (in inbound stehen die ungekürzten
#    Tool-Defs, die VS Code geschickt hat)
.\proxy-status.ps1 -Turn turn_20250101_120000_000001_ab12cd
# 3. Trace-Turns aufräumen (Rotation erzwingen)
.\proxy-status.ps1 -ClearStreams
```

Ist `TOOLS-am-Backend` ✅ aber `CALLS=0` über viele Turns, sieht das Modell die
Tools und ruft sie trotzdem nicht — dann liegt es am Prompt/Guidance, nicht
am Proxy-Transport. Fehlt `TOOLS-am-Backend`, wurde die Injection nie
ausgelöst (Kategorie/Flag-Fehler).

## Lizenz

MIT
