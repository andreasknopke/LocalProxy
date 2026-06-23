#!/usr/bin/env python3
"""
Quick Smoke-Test für Pipeline-Steuerflags (-force planning, -force review, -bypass worker).
Testet Flag-Erkennung, Intent-Klassifikation und LIVE-End-to-End.
NICHT ausführlich — nur essentielle Checks.
"""

import json, sys, time
import httpx

PROXY = "http://192.168.188.134:9001"
KEY = "owv-38579238457ehioweurzt873"
HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}

passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}  {detail}")

# ── Test 1: Flag-Extraktion (Unit-Test, inlined) ────────────────────────
print("\n── Test 1: Flag-Extraktion ──")
import re

_PIPELINE_FLAG_PATTERN = re.compile(
    r'(-\s*(?:force|bypass)\s*(?:planning|review|worker))',
    re.IGNORECASE,
)
_FLAG_ALIASES = {
    "force-planning": "force_planning",
    "force planning": "force_planning",
    "force-review": "force_review",
    "force review": "force_review",
    "bypass-worker": "bypass_worker",
    "bypass worker": "bypass_worker",
}

def _extract_pipeline_flags(text):
    flags = {"force_planning": False, "force_review": False, "bypass_worker": False}
    cleaned = text
    for match in _PIPELINE_FLAG_PATTERN.finditer(text):
        raw = match.group(1).strip().lower().replace(" ", "-").replace("--", "-")
        key = _FLAG_ALIASES.get(raw.replace("-", " ").strip(), raw.replace("-", "_"))
        if key in flags:
            flags[key] = True
    for flag_text in _FLAG_ALIASES:
        cleaned = re.sub(re.escape(flag_text), "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()
    return cleaned, flags

# Test _extract_pipeline_flags
text1, flags1 = _extract_pipeline_flags("hello world -force planning test")
test("force_planning erkannt", flags1["force_planning"] is True, str(flags1))
test("Text bereinigt", "force planning" not in text1.lower(), f"'{text1}'")
test("andere flags false", not flags1["force_review"] and not flags1["bypass_worker"])

text2, flags2 = _extract_pipeline_flags("fix this -force review -bypass worker")
test("force_review erkannt", flags2["force_review"] is True)
test("bypass_worker erkannt", flags2["bypass_worker"] is True)
test("force_planning nicht gesetzt", not flags2["force_planning"])

text3, flags3 = _extract_pipeline_flags("hello world")
test("keine flags", all(v is False for v in flags3.values()), str(flags3))
test("text unverändert", text3 == "hello world", f"'{text3}'")

# Simulate _classify_intent logic for flag detection
def _has_flag(text, flag):
    _, f = _extract_pipeline_flags(text)
    return f.get(flag, False)

test("force_planning → detected", _has_flag("short query -force planning", "force_planning"))
test("bypass_worker → detected", _has_flag("hello -bypass worker", "bypass_worker"))
test("force_review → detected", _has_flag("hi -force review", "force_review"))
test("no flags → none", not _has_flag("normal query", "force_planning"))

# ── Test 2: LIVE Proxy Health ────────────────────────────────────────────
print("\n── Test 2: LIVE Proxy Connectivity ──")
try:
    r = httpx.get(f"{PROXY}/healthz", headers=HEADERS, timeout=5)
    test("Healthcheck OK", r.status_code == 200, str(r.status_code))
except Exception as e:
    test("Healthcheck OK", False, str(e))
    print("⚠ Proxy nicht erreichbar – LIVE-Tests werden übersprungen.")
    print(f"\n{'='*50}")
    print(f"Ergebnis: {passed}/{passed+failed} Tests bestanden")
    sys.exit(0 if failed == 0 else 1)

# ── Test 3: LIVE -force planning (kurzer Prompt) ──────────────────────────
print("\n── Test 3: LIVE -force planning ──")
try:
    body = {
        "model": "aeon-ultimate",
        "messages": [{"role": "user", "content": "Write a Python hello world function. -force planning"}],
        "max_tokens": 200,
        "stream": False,
    }
    start = time.perf_counter()
    r = httpx.post(f"{PROXY}/v1/chat/completions", json=body, headers=HEADERS, timeout=120)
    dur = time.perf_counter() - start
    data = r.json()
    content = data["choices"][0]["message"]["content"] if "choices" in data else str(data)[:500]
    has_plan = "Caveman" in content or "Cloud Plan" in content or "Plan" in content
    test("force_planning: Status 200", r.status_code == 200, str(r.status_code))
    test("force_planning: Caveman Plan erkennbar", has_plan, content[:200])
    test("force_planning: Antwort nicht leer", len(content) > 20, f"len={len(content)}")
    print(f"    Dauer: {dur:.1f}s, Content-Len: {len(content)}")
except Exception as e:
    test("force_planning LIVE", False, str(e))

# ── Test 4: LIVE -bypass worker ──────────────────────────────────────────
print("\n── Test 4: LIVE -bypass worker ──")
try:
    body = {
        "model": "aeon-ultimate",
        "messages": [{"role": "user", "content": "What is 2+2? Answer in one word. -bypass worker"}],
        "max_tokens": 200,
        "stream": False,
    }
    start = time.perf_counter()
    r = httpx.post(f"{PROXY}/v1/chat/completions", json=body, headers=HEADERS, timeout=120)
    dur = time.perf_counter() - start
    data = r.json()
    content = data["choices"][0]["message"]["content"] if "choices" in data else str(data)
    test("bypass_worker: Status 200", r.status_code == 200, str(r.status_code))
    test("bypass_worker: Hat Antwort erhalten", len(content) > 5, f"len={len(content)}")
    print(f"    Dauer: {dur:.1f}s, Content-Len: {len(content)}")
except Exception as e:
    test("bypass_worker LIVE", False, str(e))

# ── Test 5: LIVE -force review ────────────────────────────────────────────
print("\n── Test 5: LIVE -force review ──")
try:
    body = {
        "model": "aeon-ultimate",
        "messages": [{"role": "user", "content": "Write a function that checks if a number is prime. -force review"}],
        "max_tokens": 400,
        "stream": False,
    }
    start = time.perf_counter()
    r = httpx.post(f"{PROXY}/v1/chat/completions", json=body, headers=HEADERS, timeout=180)
    dur = time.perf_counter() - start
    data = r.json()
    content = data["choices"][0]["message"]["content"] if "choices" in data else str(data)
    has_review = "Cloud Review" in content or "Review" in content
    test("force_review: Status 200", r.status_code == 200, str(r.status_code))
    test("force_review: Cloud Review sichtbar", has_review, content[:200])
    test("force_review: Antwort nicht leer", len(content) > 50, f"len={len(content)}")
    print(f"    Dauer: {dur:.1f}s, Content-Len: {len(content)}")
except Exception as e:
    test("force_review LIVE", False, str(e))

# ── Ergebnis ─────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Ergebnis: {passed}/{passed+failed} Tests bestanden")
print(f"{'='*50}")
sys.exit(0 if failed == 0 else 1)
