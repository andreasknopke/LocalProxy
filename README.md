# LocalProxy

Ein OpenAI-kompatibler FastAPI-Proxy für lokale vLLM-/Qwen-Coder-Modelle auf dem DX Spark.

Der Proxy nimmt Chat-Completion-Anfragen entgegen, verteilt jede Anfrage parallel auf fünf spezialisierte Sub-Agenten und gibt die gesammelten Ergebnisse wieder als OpenAI-kompatible Chat-Completion-Antwort zurück. Dadurch können Workloads wie Planung, Open-Source-Recherche, Coding, Review und Performance-Optimierung parallel laufen und die parallele Batch-Verarbeitung von vLLM besser ausnutzen.

## Architektur

Bei jeder Anfrage an `/v1/chat/completions` werden bis zu fünf parallele Sub-Agenten gestartet:

1. **Architekt & Denker** – analysiert die Aufgabe, plant Architektur, Logik und Randfälle.
2. **Open-Source-Scout & Dependency-Manager** – sucht nach etablierten Bibliotheken, Standards oder bestehenden Lösungen, damit nicht unnötig neu entwickelt wird.
3. **Entwickler** – erstellt produktionsnahen Code unter Berücksichtigung der Scout-Ergebnisse.
4. **Reviewer & Security** – prüft Bugs, Sicherheitsrisiken, Race Conditions, Memory Leaks und Edge Cases.
5. **Performance & Refactoring** – optimiert Laufzeit, Speicherverbrauch und Struktur.

Optional kann ein sechster **Cloud Final Reviewer** aktiviert werden. Dieser läuft nicht auf dem lokalen Qwen-Modell, sondern gegen ein konfigurierbares Cloud-Modell und bewertet die gesamte lokale Multi-Agent-Antwort wie ein Senior Developer. Bei kritischen Mängeln kann er die Gesamtumsetzung zurückweisen.

Der ursprüngliche User-Prompt bleibt vorne erhalten. Nur die letzte User-Nachricht erhält einen agentenspezifischen Suffix. Dadurch kann vLLM Prefix Caching besser nutzen.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Starten

```bash
VLLM_API_URL=http://localhost:8000/v1/chat/completions \
MODEL_NAME="Qwen/Qwen3-Next-80B-Chat-mxfp4" \
python proxy.py
```

Mit optionalem Cloud Final Reviewer:

```bash
VLLM_API_URL=http://localhost:8000/v1/chat/completions \
MODEL_NAME="Qwen/Qwen3-Next-80B-Chat-mxfp4" \
CLOUD_REVIEW_ENABLED=true \
CLOUD_REVIEW_API_KEY="sk-..." \
CLOUD_REVIEW_MODEL="gpt-4.1-mini" \
python proxy.py
```

Der Proxy läuft standardmäßig auf:

```text
http://0.0.0.0:5001
```

## Konfiguration

| Variable | Standard | Beschreibung |
| --- | --- | --- |
| `VLLM_API_URL` | `http://localhost:8000/v1/chat/completions` | Ziel-API des lokalen vLLM-Servers |
| `VLLM_MODELS_URL` | `http://localhost:8000/v1/models` | Optionaler Modelle-Endpoint |
| `MODEL_NAME` | `Qwen/Qwen3-Next-80B-Chat-mxfp4` | Modellname für die Proxy-Antwort |
| `PROXY_PORT` | `5001` | Port des Proxy-Servers |
| `SUB_AGENT_MAX_TOKENS` | `2048` | Maximale Tokens pro Sub-Agent |
| `SUB_AGENT_TIMEOUT_SECONDS` | `60` | Timeout pro Sub-Agent-Anfrage |
| `SUB_AGENT_CONCURRENCY` | `5` | Konfigurierbarer Wert für die angestrebte Parallelität |
| `CLOUD_REVIEW_ENABLED` | `false` | Aktiviert den Cloud Final Reviewer |
| `CLOUD_REVIEW_API_URL` | `https://api.openai.com/v1/chat/completions` | OpenAI-kompatible Cloud-Review-API |
| `CLOUD_REVIEW_API_KEY` | leer | API-Key für den Cloud Final Reviewer |
| `CLOUD_REVIEW_MODEL` | `gpt-4.1-mini` | Cloud-Modell für den Final Review |
| `CLOUD_REVIEW_MAX_TOKENS` | `2048` | Maximale Tokens für den Cloud Final Review |
| `CLOUD_REVIEW_TIMEOUT_SECONDS` | `90` | Timeout für den Cloud Final Review |

## OpenAI-kompatibler Testaufruf

```bash
curl http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Next-80B-Chat-mxfp4",
    "messages": [
      {
        "role": "user",
        "content": "Erstelle eine robuste Python-Funktion, die große CSV-Dateien speicherschonend verarbeitet."
      }
    ]
  }'
```

## Nutzung mit VS Code / Copilot

Konfiguriere VS Code bzw. deine verwendete Extension so, dass der OpenAI-Endpoint auf den lokalen Proxy zeigt:

```text
Base URL: http://localhost:5001/v1
Model: Qwen/Qwen3-Next-80B-Chat-mxfp4
```

## Gesundheitscheck

```bash
curl http://localhost:5001/healthz
```

Antwort:

```json
{"status":"ok","agents":5}
```

## Hinweis zur Performance

Die Parallelität nutzt vLLM-Batching und Prefix Caching. Die lokalen Qwen-Agenten werden parallel ausgeführt. Der optionale Cloud Final Reviewer läuft danach sequenziell, damit er die komplette lokale Gesamtumsetzung bewerten und ggf. zurückweisen kann.

