# LocalProxy v3.0 — Coolify/Docker Deployment
# Coolify erkennt dieses Dockerfile automatisch (Nixpacks-Fallback deaktiviert)

FROM python:3.13-slim

# ── UTF-8 Locale (verhindert UnicodeEncodeError in Docker) ──────────────
ENV PYTHONIOENCODING=utf-8
ENV PYTHONUTF8=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

WORKDIR /app

# ── System-Dependencies ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python-Dependencies ──────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── App-Code ─────────────────────────────────────────────────────────────
COPY proxy.py webui.py ./
COPY data/ ./data/

# ── Datenverzeichnisse (Volume-Mounts via Coolify) ───────────────────────
# /app/data        — config.json, Logs, Hindsight-Speicher, Debug-Dumps
# /app/data/plans  — verwaistes Verzeichnis aus v2.x (bleibt leer)
RUN mkdir -p /app/data/plans /app/data/debug /app/data/profiles

# ── Coolify Volume-Hinweis ───────────────────────────────────────────────
# In Coolify unter "Storages" ein Volume auf /app/data mounten,
# damit config.json, Logs und Hindsight persistent bleiben.

# ── Port ─────────────────────────────────────────────────────────────────
EXPOSE 9001

# ── Healthcheck ──────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:9001/healthz || exit 1

# ── Startup ──────────────────────────────────────────────────────────────
# proxy.py erkennt automatisch ob webui.py vorhanden ist.
# config.json wird beim ersten Start aus Defaults + Env-Vars erzeugt,
# falls sie nicht existiert.
CMD ["python", "proxy.py"]
