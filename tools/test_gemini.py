"""
Test-Script für Google Gemini OpenAI-kompatiblen Endpunkt.
Nutzung:
    python tools/test_gemini.py DEIN_API_KEY
    python tools/test_gemini.py                          # liest aus Env-Var GEMINI_API_KEY
"""

import json
import os
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
MODEL = "gemini-3.5-flash"


def get_api_key() -> str:
    if len(sys.argv) >= 2:
        return sys.argv[1]
    key = os.getenv("GEMINI_API_KEY", "")
    if key:
        return key
    print("❌ Kein API-Key. Entweder als Argument oder GEMINI_API_KEY env var setzen.")
    sys.exit(1)


def main():
    api_key = get_api_key()
    print(f"API-Key: {api_key[:8]}... (Länge {len(api_key)})")
    print(f"URL:     {URL}")
    print(f"Model:   {MODEL}")
    print()

    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hallo! Welches Modell bist du?"}],
        "max_tokens": 50,
    }).encode("utf-8")

    req = Request(URL, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    started = time.monotonic()
    try:
        with urlopen(req, timeout=15) as resp:
            duration = time.monotonic() - started
            body = json.loads(resp.read().decode("utf-8"))
            print(f"✅ HTTP {resp.status} ({duration:.1f}s)")
            print(f"   Model: {body.get('model', '?')}")
            choice = body.get("choices", [{}])[0]
            msg = choice.get("message", {})
            content = msg.get("content", "")
            print(f"   Content: {content[:500]}")
    except HTTPError as e:
        duration = time.monotonic() - started
        body = e.read().decode("utf-8", errors="replace")
        print(f"❌ HTTP {e.code} ({duration:.1f}s)")
        print(f"   {body[:1000]}")
    except URLError as e:
        duration = time.monotonic() - started
        print(f"❌ Connection Error ({duration:.1f}s): {e.reason}")
    except Exception as e:
        duration = time.monotonic() - started
        print(f"❌ Error ({duration:.1f}s): {e}")


if __name__ == "__main__":
    main()
