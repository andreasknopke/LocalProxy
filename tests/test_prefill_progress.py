"""
Tests fuer das llama.cpp Prefill-Progress-Polling (Kategorie local/coworker).

Testet:
  1) _url_port / _slots_base_url (Port-Erkennung, Base-URL-Ableitung)
  2) _is_llama_cpp (Auto-Detect Port 8082, explizites prefill_progress-Flag,
     globaler Enable-Schalter)
  3) _parse_llama_slot_progress (/slots-Antwort → Fortschritt, Fallbacks)
  4) _prefill_progress_line (Textformat)
  5) _read_sse_with_prefill (paralleles Polling + finaler 100%-Event)

Konvention wie in den anderen Test-Files: plain def test_*(), innere
asyncio.run(...) fuer async Code. Kein pytest-asyncio, kein TestClient.
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import proxy  # noqa: E402


# ── 1. URL-Helfer ──────────────────────────────────────────────────────────

def test_url_port():
    assert proxy._url_port("http://localhost:8082/v1/chat/completions") == 8082
    assert proxy._url_port("http://127.0.0.1:8082") == 8082
    assert proxy._url_port("http://localhost:1234/v1/chat/completions") == 1234
    assert proxy._url_port("https://api.openai.com/v1/chat/completions") is None
    assert proxy._url_port("http://localhost/v1") is None


def test_slots_base_url():
    assert proxy._slots_base_url("http://localhost:8082/v1/chat/completions") == "http://localhost:8082"
    assert proxy._slots_base_url("http://127.0.0.1:8082/") == "http://127.0.0.1:8082"
    assert proxy._slots_base_url("not a url") is None


# ── 2. _is_llama_cpp ───────────────────────────────────────────────────────

def test_is_llama_cpp_autodetect_port_8082():
    assert proxy._is_llama_cpp({"api_url": "http://localhost:8082/v1/chat/completions"}) is True


def test_is_llama_cpp_other_port_not_llama():
    assert proxy._is_llama_cpp({"api_url": "http://localhost:1234/v1/chat/completions"}) is False
    assert proxy._is_llama_cpp({"api_url": "https://api.openai.com/v1/chat/completions"}) is False


def test_is_llama_cpp_explicit_flag_overrides():
    # Explizites Flag schlaegt die Port-Auto-Erkennung
    assert proxy._is_llama_cpp({"api_url": "http://localhost:1234/x", "prefill_progress": True}) is True
    assert proxy._is_llama_cpp({"api_url": "http://localhost:8082/x", "prefill_progress": False}) is False
    assert proxy._is_llama_cpp({"api_url": "http://localhost:8082/x", "prefill_progress": "true"}) is True


def test_is_llama_cpp_disabled_globally(monkeypatch):
    monkeypatch.setattr(proxy, "PREFILL_PROGRESS_ENABLED", False)
    assert proxy._is_llama_cpp({"api_url": "http://localhost:8082/x"}) is False
    assert proxy._is_llama_cpp({"api_url": "http://localhost:8082/x", "prefill_progress": True}) is False


# ── 3. _parse_llama_slot_progress ──────────────────────────────────────────

def _slot(state=1, progress=None, n_processed=None, tokens=None):
    prompt = {}
    if progress is not None:
        prompt["progress"] = progress
    if n_processed is not None:
        prompt["n_processed"] = n_processed
    if tokens is not None:
        prompt["tokens"] = tokens
    return {"id": 0, "state": state, "prompt": prompt}


def test_parse_slot_progress_active_with_progress_field():
    info = proxy._parse_llama_slot_progress([_slot(progress=0.5, n_processed=50, tokens=list(range(100)))])
    assert info == {"active": True, "percent": 50, "n": 50, "total": 100}


def test_parse_slot_progress_fallback_from_n_processed():
    # Kein progress-Feld → aus n_processed/total ableiten
    info = proxy._parse_llama_slot_progress([_slot(n_processed=25, tokens=list(range(100)))])
    assert info["percent"] == 25
    assert info["total"] == 100


def test_parse_slot_progress_idle_returns_none():
    assert proxy._parse_llama_slot_progress([_slot(state=0, progress=0.3)]) is None


def test_parse_slot_progress_done_returns_none():
    # progress >= 1.0 → Prefill fertig, Generierung laeuft
    assert proxy._parse_llama_slot_progress([_slot(progress=1.0)]) is None


def test_parse_slot_progress_picks_most_advanced():
    slots = [
        _slot(state=0, progress=0.9),          # idle → ignoriert
        _slot(progress=0.2, n_processed=20),   # aktiv, 20%
        _slot(progress=0.7, n_processed=70),   # aktiv, 70% → gewinnt
    ]
    info = proxy._parse_llama_slot_progress(slots)
    assert info["percent"] == 70
    assert info["n"] == 70


def test_parse_slot_progress_non_list():
    assert proxy._parse_llama_slot_progress({"not": "a list"}) is None
    assert proxy._parse_llama_slot_progress(None) is None


# ── 3b. _parse_llama_slot_progress — neues Schema ──────────────────────────

def _slot_new(is_processing=True, n_prompt_tokens=0, n_processed=0, n_cache=0):
    return {"id": 0, "is_processing": is_processing,
            "n_prompt_tokens": n_prompt_tokens,
            "n_prompt_tokens_processed": n_processed,
            "n_prompt_tokens_cache": n_cache}


def test_parse_slot_progress_new_schema():
    info = proxy._parse_llama_slot_progress([_slot_new(n_prompt_tokens=57369)])
    assert info["active"] is True
    assert info["n"] == 57369
    assert info["percent"] is None  # Gesamt-Tokens nicht in /slots
    assert info["total"] == 0


def test_parse_slot_progress_new_schema_idle():
    assert proxy._parse_llama_slot_progress([_slot_new(is_processing=False, n_prompt_tokens=74265)]) is None


def test_parse_slot_progress_new_schema_fallback_to_processed():
    # n_prompt_tokens fehlt → aus processed + cache ableiten
    info = proxy._parse_llama_slot_progress([_slot_new(n_prompt_tokens=0, n_processed=500, n_cache=100)])
    assert info["n"] == 600


# ── 4. _prefill_progress_line ──────────────────────────────────────────────

def test_prefill_progress_line_with_total():
    # n=1234, total=3080 → 40%, rate 462 t/s
    line = proxy._prefill_progress_line(1234, 3080, 462.0, 8.0)
    assert "Prefill 40%" in line
    assert "1234/3080" in line
    assert "462 t/s" in line
    assert line.endswith("\n")


def test_prefill_progress_line_without_total():
    line = proxy._prefill_progress_line(57369, None, 462.0, 124.0)
    assert "57369 Tokens" in line
    assert "462 t/s" in line
    assert "Prefill" in line
    assert line.endswith("\n")


# ── 5. _read_sse_with_prefill ──────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, lines, delay=0.0):
        self._lines = lines
        self._delay = delay

    async def aiter_lines(self):
        if self._delay:
            await asyncio.sleep(self._delay)
        for ln in self._lines:
            yield ln


class _FakeSlotResponse:
    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, progress_seq):
        self._seq = progress_seq
        self._calls = 0

    async def get(self, url, timeout=None):
        idx = min(self._calls, len(self._seq) - 1)
        self._calls += 1
        return _FakeSlotResponse(self._seq[idx])


def _slot_payload(progress):
    return [{"id": 0, "state": 1, "prompt": {
        "progress": progress,
        "n_processed": int(progress * 100),
        "tokens": list(range(100)),
    }}]


def _slot_payload_new(n_tokens):
    return [{"id": 0, "is_processing": True,
             "n_prompt_tokens": n_tokens,
             "n_prompt_tokens_processed": n_tokens,
             "n_prompt_tokens_cache": 0}]


def test_read_sse_with_prefill_emits_progress():
    async def run():
        resp = _FakeResponse([
            'data: {"choices":[{"delta":{"role":"assistant"},"index":0}],"id":"c1"}',
            'data: {"choices":[{"delta":{"content":"hi"},"index":0}],"id":"c1"}',
            'data: [DONE]',
        ], delay=0.05)
        client = _FakeClient([
            _slot_payload(0.0),
            _slot_payload(0.0),
            _slot_payload(0.12),
            _slot_payload(0.25),
        ])
        out = []
        async for kind, val in proxy._read_sse_with_prefill(
                resp, client, "http://localhost:8082/slots", 0.01, 10, 1.0):
            out.append((kind, val))
        return out

    out = asyncio.run(run())
    progress = [v for k, v in out if k == "progress"]
    lines = [v for k, v in out if k == "line"]
    assert progress, "erwartet mindestens einen progress-Event"
    assert any(v.get("percent", 0) < 100 for v in progress)
    assert progress[-1].get("percent") == 100
    assert "Prefill" in progress[-1].get("content", "")
    assert any(l.startswith("data:") for l in lines)


def test_read_sse_with_prefill_new_schema_percent():
    async def run():
        resp = _FakeResponse([
            'data: {"choices":[{"delta":{"content":"hi"},"index":0}],"id":"c1"}',
            'data: [DONE]',
        ], delay=0.05)
        client = _FakeClient([
            _slot_payload_new(0),
            _slot_payload_new(2000),
            _slot_payload_new(5000),
            _slot_payload_new(9000),
        ])
        out = []
        async for kind, val in proxy._read_sse_with_prefill(
                resp, client, "http://localhost:8082/slots", 0.01, 10, 1.0,
                estimated_total=10000):
            out.append((kind, val))
        return out

    out = asyncio.run(run())
    progress = [v for k, v in out if k == "progress"]
    assert progress, "erwartet Fortschritts-Events (neues Schema)"
    assert any("90%" in v.get("content", "") for v in progress)
    assert progress[-1].get("percent") == 100


def test_read_sse_with_prefill_no_poll_when_slots_none():
    async def run():
        resp = _FakeResponse([
            'data: {"choices":[{"delta":{"content":"x"},"index":0}],"id":"c1"}',
            'data: [DONE]',
        ])
        out = []
        async for kind, val in proxy._read_sse_with_prefill(
                resp, None, None, 0.01, 10, 1.0):
            out.append((kind, val))
        return out

    out = asyncio.run(run())
    kinds = {k for k, _ in out}
    assert "progress" not in kinds
    assert "line" in kinds
