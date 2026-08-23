# -*- coding: utf-8 -*-
"""Agent-Runner (lokal auf dem User-PC).

Verbindet sich per Long-Polling mit dem Proxy (/agent/tools/claim) und
fuehrt Tool-Calls des Co-Worker-Subagents LOKAL aus — der Proxy-Container
(Coolify) hat keinen Zugriff auf die Workspace-Files.

Unterstuetzte Tools:
  read_file / write_file / list_dir  — Dateizugriff relativ zu --workspace
  web_search                          — DuckDuckGo-HTML-Suche (kein API-Key)

Start:
  python tools/agent_runner.py \
      --proxy https://proxy.abigailrook.de \
      --token <AGENT_RUNNER_TOKEN oder PROXY_API_KEY> \
      --workspace D:\GitHub

Der Runner pollt endlos; Abbruch mit Strg+C. Er ist bewusst abhaengig-
frei (nur requests + beautifulsoup4 optional).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

CLAIM_TIMEOUT = 30          # Sekunden Long-Poll pro Claim-Request
POLL_RETRY_DELAY = 3        # Sekunden Wartezeit nach Verbindungsfehlern


def _resolve(workspace: Path, path: str) -> Path:
    """Loest einen Pfad gegen das Workspace-Root auf und verhindert Escape."""
    p = Path(path)
    if not p.is_absolute():
        p = workspace / p
    p = p.resolve()
    ws = workspace.resolve()
    if not (str(p).lower().startswith(str(ws).lower() + os.sep)
            or str(p).lower() == str(ws).lower()):
        raise PermissionError(f"Pfad ausserhalb des Workspace: {path}")
    return p


def exec_tool(name: str, args: dict, workspace: Path) -> str:
    """Fuehrt EINEN Tool-Call lokal aus und liefert Text zurueck."""
    if name == "read_file":
        p = _resolve(workspace, str(args.get("path", "")))
        return p.read_text(encoding="utf-8", errors="replace")

    if name == "write_file":
        p = _resolve(workspace, str(args.get("path", "")))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(args.get("content", "")), encoding="utf-8")
        return f"OK: {p} ({len(args.get('content', ''))} chars geschrieben)"

    if name == "list_dir":
        p = _resolve(workspace, str(args.get("path", ".")))
        entries = []
        for e in sorted(p.iterdir()):
            kind = "dir" if e.is_dir() else "file"
            size = e.stat().st_size if e.is_file() else ""
            entries.append(f"{kind}\t{size}\t{e.name}")
        return "\n".join(entries) or "(leer)"

    if name == "web_search":
        query = str(args.get("query", ""))
        r = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (agent-runner)"},
            timeout=20,
        )
        r.raise_for_status()
        # Mini-Parsing ohne bs4: Links + Titel grob extrahieren
        import re
        results = re.findall(
            r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            r.text, re.S)
        out = []
        for url, title in results[:8]:
            title = re.sub(r"<[^>]+>", "", title).strip()
            out.append(f"- {title}\n  {url}")
        return "\n".join(out) or "(keine Treffer)"

    return f"[Unbekanntes Tool: {name}]"


def main() -> None:
    ap = argparse.ArgumentParser(description="Lokaler Agent-Runner fuer den MyProxy-Co-Worker")
    ap.add_argument("--proxy", default=os.getenv("PROXY_URL", "https://proxy.abigailrook.de"))
    ap.add_argument("--token", default=os.getenv("AGENT_RUNNER_TOKEN") or os.getenv("PROXY_API_KEY", ""))
    ap.add_argument("--workspace", default=os.getenv("AGENT_WORKSPACE", "."),
                    help="Workspace-Root, relativ dazu duerfen Subagent-Tools arbeiten")
    ns = ap.parse_args()

    workspace = Path(ns.workspace).expanduser().resolve()
    base = ns.proxy.rstrip("/")
    headers = {"Authorization": f"Bearer {ns.token}"}
    print(f"[runner] proxy={base} workspace={workspace}")

    session = requests.Session()
    session.headers.update(headers)

    while True:
        try:
            r = session.get(f"{base}/agent/tools/claim",
                            params={"wait": CLAIM_TIMEOUT - 5}, timeout=CLAIM_TIMEOUT)
            if r.status_code == 204:
                continue  # keine Jobs — weiter pollen
            r.raise_for_status()
            job = r.json()
        except requests.RequestException as exc:
            print(f"[runner] Verbindungsfehler: {exc} — retry in {POLL_RETRY_DELAY}s")
            time.sleep(POLL_RETRY_DELAY)
            continue

        call_id = job.get("call_id")
        name = job.get("name", "")
        args = job.get("arguments") or {}
        print(f"[runner] exec {name} {json.dumps(args, ensure_ascii=False)[:120]}")
        result, error = None, None
        try:
            result = exec_tool(name, args, workspace)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        try:
            session.post(f"{base}/agent/tools/result",
                         json={"call_id": call_id, "result": result, "error": error},
                         timeout=15)
        except requests.RequestException as exc:
            print(f"[runner] Ergebnis-Post fehlgeschlagen: {exc}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[runner] beendet")
        sys.exit(0)
