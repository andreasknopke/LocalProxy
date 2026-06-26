# LocalProxy v2.1 — Coolify/Docker Deployment
FROM python:3.13-slim

WORKDIR /app

# Abhängigkeiten installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App-Code kopieren
COPY proxy.py webui.py config.json* ./

# Port für Coolify freigeben
EXPOSE 9001

# Healthcheck für Coolify
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import http.client; c=http.client.HTTPConnection('localhost',9001); c.request('GET','/healthz'); r=c.getresponse(); exit(0 if r.status==200 else 1)"

# Start — proxy.py mountet webui.py automatisch
CMD ["python", "proxy.py"]
