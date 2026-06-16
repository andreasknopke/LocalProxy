from __future__ import annotations

import asyncio
import copy
import json
import os
import secrets
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(
    title="DX Spark vLLM Parallel Proxy",
    docs_url=None,
    openapi_url=None,
    redoc_url=None,
)

VLLM_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8000/v1/chat/completions")
VLLM_MODELS_URL = os.getenv("VLLM_MODELS_URL", "http://localhost:8000/v1/models")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-Next-80B-Chat-mxfp4")
PROXY_PORT = int(os.getenv("PROXY_PORT", "9001"))

PROXY_AUTH_ENABLED = os.getenv("PROXY_AUTH_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "")

if PROXY_AUTH_ENABLED and not PROXY_API_KEY:
    raise RuntimeError("PROXY_AUTH_ENABLED=true requires PROXY_API_KEY to be set.")


def _get_bearer_token(request: Request) -> str:
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    return ""


def _is_authorized(request: Request) -> bool:
    if not PROXY_AUTH_ENABLED:
        return True

    token = _get_bearer_token(request)
    expected_token = os.getenv("PROXY_API_KEY", "")
    return bool(expected_token and secrets.compare_digest(token, expected_token))


async def _auth_or_raise(request: Request) -> None:
    if not _is_authorized(request):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )

CHATTY_MODE = os.getenv("CHATTY_MODE", "true").lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
CHATTY_HEARTBEAT_SECONDS = float(os.getenv("CHATTY_HEARTBEAT_SECONDS", "15"))

CLOUD_REVIEW_ENABLED = os.getenv("CLOUD_REVIEW_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
CLOUD_REVIEW_API_URL = os.getenv(
    "CLOUD_REVIEW_API_URL",
    "https://api.openai.com/v1/chat/completions",
)
CLOUD_REVIEW_API_KEY = os.getenv("CLOUD_REVIEW_API_KEY", "")
CLOUD_REVIEW_MODEL = os.getenv(
    "CLOUD_REVIEW_MODEL",
    "gpt-4.1-mini",
)
CLOUD_REVIEW_MAX_TOKENS = int(os.getenv("CLOUD_REVIEW_MAX_TOKENS", "2048"))
CLOUD_REVIEW_TIMEOUT_SECONDS = float(os.getenv("CLOUD_REVIEW_TIMEOUT_SECONDS", "90"))

DEFAULT_SUB_AGENT_MAX_TOKENS = int(os.getenv("SUB_AGENT_MAX_TOKENS", "2048"))
SUB_AGENT_TIMEOUT_SECONDS = float(os.getenv("SUB_AGENT_TIMEOUT_SECONDS", "60"))
SUB_AGENT_CONCURRENCY = int(os.getenv("SUB_AGENT_CONCURRENCY", "5"))

SUB_AGENTS: Dict[str, str] = {
    "architect": (
        "\n\n[Agent 1: Architekt & Denker]\n"
        "Analysiere die obige Anfrage tiefgehend. Plane die Architektur, Logik, Randfälle "
        "und algorithmischen Ansätze. Antworte kurz, präzise und rein fokussiert auf das Konzept."
    ),
    "os_scout": (
        "\n\n[Agent 2: Open-Source-Scout & Dependency-Manager]\n"
        "Verhindere, dass das Rad neu erfunden wird! Identifiziere etablierte Open-Source-Bibliotheken "
        "(z. B. auf PyPI, npm, GitHub, NuGet) oder native Standard-Bibliotheken der Sprache, die diese "
        "Aufgabe oder Teile davon bereits robust und getestet lösen. Nenne konkrete Paketnamen "
        "und etablierte Best-Practice-Abstraktionen, statt alles selbst zu schreiben."
    ),
    "coder": (
        "\n\n[Agent 3: Entwickler]\n"
        "Schreibe den produktionsreifen Code für die obige Anfrage. Nutze – falls vom Open-Source-Scout "
        "sinnvoll vorgeschlagen – etablierte Bibliotheken. Achte auf Best Practices, Typsicherheit "
        "und saubere Formatierung. Gib primär Codeblöcke mit minimalem Text aus."
    ),
    "reviewer": (
        "\n\n[Agent 4: Reviewer & Security]\n"
        "Überprüfe die Anfrage auf potenzielle Bugs, Sicherheitsrisiken (Race Conditions, Memory Leaks) "
        "und edge cases. Biete konkrete Optimierungen für Stabilität an."
    ),
    "optimizer": (
        "\n\n[Agent 5: Performance & Refactoring]\n"
        "Fokussiere dich auf die Performance des Codes. Wie kann die Laufzeit- und Speicherkomplexität "
        "optimiert werden? Biete falls nötig eine refactorte, hocheffiziente Teillösung an."
    ),
}

DISPLAY_NAMES = {
    "architect": "Architekt & Denker",
    "os_scout": "Open-Source-Scout & Dependency-Manager",
    "coder": "Entwickler",
    "reviewer": "Reviewer & Security",
    "optimizer": "Performance & Refactoring",
}


async def call_sub_agent(
    client: httpx.AsyncClient,
    payload: Dict[str, Any],
    agent_key: str,
    agent_instruction: str,
) -> Dict[str, Any]:
    """Sendet eine parallele Anfrage an vLLM und bewahrt den gemeinsamen Prompt-Präfix."""
    agent_payload = copy.deepcopy(payload)

    if agent_payload.get("messages"):
        last_message = agent_payload["messages"][-1]
        if isinstance(last_message, dict) and last_message.get("role") == "user":
            content = last_message.get("content", "")
            if isinstance(content, str):
                last_message["content"] = f"{content}{agent_instruction}"
            else:
                # OpenAI-kompatible multimodale Nachrichten können eine Liste sein.
                # Für diesen Proxy wird der Agenten-Suffix als zusätzliche Text-Part ergänzt.
                content_parts = list(content) if isinstance(content, list) else []
                content_parts.append({"type": "text", "text": agent_instruction})
                last_message["content"] = content_parts

    agent_payload["model"] = payload.get("model", MODEL_NAME)
    agent_payload["max_tokens"] = int(
        payload.get("max_tokens", DEFAULT_SUB_AGENT_MAX_TOKENS)
    )
    agent_payload["stream"] = False

    started = time.perf_counter()
    try:
        response = await client.post(
            VLLM_API_URL,
            json=agent_payload,
            timeout=SUB_AGENT_TIMEOUT_SECONDS,
        )
        duration = time.perf_counter() - started

        if response.status_code == 200:
            result = response.json()
            choices = result.get("choices", [])
            if not choices:
                return {
                    "agent_key": agent_key,
                    "status": "failed",
                    "content": f"### ❌ {DISPLAY_NAMES[agent_key].upper()} FAILED: keine Antwort-Choices erhalten\n",
                    "duration_seconds": duration,
                    "usage": None,
                }

            content = choices[0].get("message", {}).get("content", "")
            usage = result.get("usage")
            return {
                "agent_key": agent_key,
                "status": "ok",
                "content": f"### 🛠️ {DISPLAY_NAMES[agent_key].upper()} ANALYSIS\n{content}\n",
                "duration_seconds": duration,
                "usage": usage,
            }

        return {
            "agent_key": agent_key,
            "status": "failed",
            "content": (
                f"### ❌ {DISPLAY_NAMES[agent_key].upper()} FAILED "
                f"(Status {response.status_code})\n{response.text}\n"
            ),
            "duration_seconds": duration,
            "usage": None,
        }

    except Exception as exc:
        duration = time.perf_counter() - started
        return {
            "agent_key": agent_key,
            "status": "error",
            "content": f"### ❌ {DISPLAY_NAMES[agent_key].upper()} ERROR: {exc}\n",
            "duration_seconds": duration,
            "usage": None,
        }


def _extract_choice_content(result: Dict[str, Any]) -> str:
    choices = result.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "")


def _is_rejected(content: str) -> bool:
    lowered = content.lower()
    rejection_markers = [
        "reject",
        "rejected",
        "ablehnen",
        "abgelehnt",
        "zurückweisen",
        "zurueckweisen",
        "nicht akzeptieren",
        "nicht akzeptierbar",
        "critical issue",
        "blocker",
    ]
    return any(marker in lowered for marker in rejection_markers)


async def call_cloud_final_reviewer(
    client: httpx.AsyncClient,
    original_payload: Dict[str, Any],
    local_results: List[Dict[str, Any]],
    combined_response_text: str,
) -> Dict[str, Any]:
    """Lässt ein konfigurierbares Cloud-Modell die lokale Qwen-Gesamtantwort bewerten."""
    if not CLOUD_REVIEW_ENABLED:
        return {
            "agent_key": "cloud_final_reviewer",
            "status": "skipped",
            "content": "### ℹ️ CLOUD FINAL REVIEWER SKIPPED\nCloud Review ist deaktiviert.\n",
            "duration_seconds": 0.0,
            "usage": None,
            "rejected": False,
        }

    if not CLOUD_REVIEW_API_KEY:
        return {
            "agent_key": "cloud_final_reviewer",
            "status": "failed",
            "content": "### ❌ CLOUD FINAL REVIEWER FAILED: CLOUD_REVIEW_API_KEY fehlt.\n",
            "duration_seconds": 0.0,
            "usage": None,
            "rejected": False,
        }

    local_sections = "\n\n".join(
        f"## {DISPLAY_NAMES.get(result.get('agent_key', 'unknown'), 'Unknown').title()}\n{result.get('content', '')}"
        for result in local_results
    )

    original_user_message = ""
    messages = original_payload.get("messages", [])
    if messages and isinstance(messages[-1], dict):
        content = messages[-1].get("content", "")
        original_user_message = content if isinstance(content, str) else str(content)

    review_prompt = (
        "Du bist ein erfahrener Senior Developer und technischer Final Reviewer.\n"
        "Bewerte die Gesamtumsetzung einer lokalen Qwen-Multi-Agent-Antwort.\n\n"
        "Originaler User-Prompt:\n"
        f"{original_user_message}\n\n"
        "Lokale Qwen-Sub-Agenten-Antworten:\n"
        f"{local_sections}\n\n"
        "Zusammengeführte Proxy-Antwort:\n"
        f"{combined_response_text}\n\n"
        "Bewerte als Senior Developer:\n"
        "1. Korrektheit und Vollständigkeit\n"
        "2. Architektur und Wartbarkeit\n"
        "3. Sicherheit und Robustheit\n"
        "4. Performance\n"
        "5. Ob etablierte Open-Source-Lösungen korrekt berücksichtigt wurden\n"
        "6. Ob die finale Umsetzung akzeptiert werden kann oder zurückgewiesen werden muss\n\n"
        "Antworte strikt in folgendem Format:\n\n"
        "DECISION: ACCEPT oder REJECT\n"
        "SUMMARY: Kurze Begründung\n"
        "BLOCKERS: Konkrete Blocker oder 'keine'\n"
        "RECOMMENDED_FIXES: Konkrete, priorisierte Verbesserungsvorschläge oder 'keine'\n"
    )

    review_payload = {
        "model": CLOUD_REVIEW_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Du bist ein strenger Senior Developer, Architect und Security Reviewer. "
                    "Du akzeptierst nur technisch saubere, sichere, wartbare und vollständige Lösungen. "
                    "Wenn kritische Fehler bestehen, weise die Umsetzung klar zurück."
                ),
            },
            {"role": "user", "content": review_prompt},
        ],
        "max_tokens": CLOUD_REVIEW_MAX_TOKENS,
        "stream": False,
    }

    started = time.perf_counter()
    headers = {"Authorization": f"Bearer {CLOUD_REVIEW_API_KEY}"}

    try:
        response = await client.post(
            CLOUD_REVIEW_API_URL,
            json=review_payload,
            headers=headers,
            timeout=CLOUD_REVIEW_TIMEOUT_SECONDS,
        )
        duration = time.perf_counter() - started

        if response.status_code == 200:
            result = response.json()
            content = _extract_choice_content(result)
            rejected = _is_rejected(content)
            prefix = "### 🧠 CLOUD FINAL REVIEWER: REJECTED" if rejected else "### 🧠 CLOUD FINAL REVIEWER: ACCEPTED"
            return {
                "agent_key": "cloud_final_reviewer",
                "status": "ok",
                "content": f"{prefix}\n{content}\n",
                "duration_seconds": duration,
                "usage": result.get("usage"),
                "rejected": rejected,
            }

        return {
            "agent_key": "cloud_final_reviewer",
            "status": "failed",
            "content": (
                f"### ❌ CLOUD FINAL REVIEWER FAILED "
                f"(Status {response.status_code})\n{response.text}\n"
            ),
            "duration_seconds": duration,
            "usage": None,
            "rejected": False,
        }

    except Exception as exc:
        duration = time.perf_counter() - started
        return {
            "agent_key": "cloud_final_reviewer",
            "status": "error",
            "content": f"### ❌ CLOUD FINAL REVIEWER ERROR: {exc}\n",
            "duration_seconds": duration,
            "usage": None,
            "rejected": False,
        }


def _sum_usage(results: List[Dict[str, Any]]) -> Dict[str, int]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    complete = True

    for result in results:
        usage = result.get("usage") or {}
        try:
            prompt_tokens += int(usage.get("prompt_tokens", 0))
            completion_tokens += int(usage.get("completion_tokens", 0))
            total_tokens += int(usage.get("total_tokens", 0))
        except (TypeError, ValueError):
            complete = False

    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens if complete else -1,
        "completion_tokens": completion_tokens if complete else -1,
        "total_tokens": total_tokens if complete else -1,
    }


def _format_status_event(stage: str, message: str, data: Optional[Dict[str, Any]] = None) -> str:
    payload = {
        "type": "localproxy.status",
        "stage": stage,
        "message": message,
        "timestamp": time.time(),
    }
    if data is not None:
        payload["data"] = data
    return json.dumps(payload, ensure_ascii=False)


def _format_sse_event(stage: str, message: str, data: Optional[Dict[str, Any]] = None) -> str:
    return f"data: {_format_status_event(stage, message, data)}\n\n"


def _format_chat_progress_message(stage: str, message: str, data: Optional[Dict[str, Any]] = None) -> str:
    payload = _format_status_event(stage, message, data)
    data_text = f"  \n`{json.dumps(data, ensure_ascii=False)}`" if data is not None else ""
    return f"**▸ STATUS [{stage}]:** {message}{data_text}\n\n"


def _format_openai_stream_chunk(
    model: str,
    content: str = "",
    finish_reason: Optional[str] = None,
    include_role: bool = False,
) -> str:
    delta: Dict[str, Any] = {"content": content}
    if include_role:
        delta["role"] = "assistant"

    payload = {
        "id": f"chatcmpl-spark-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _maybe_add_progress_message(
    content: str,
    stage: str,
    message: str,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    if not CHATTY_MODE:
        return content
    return content + _format_chat_progress_message(stage, message, data)


async def _stream_chat_completion(
    body: Dict[str, Any],
) -> Dict[str, Any]:
    start_time = time.perf_counter()
    progress_messages = [
        _format_chat_progress_message(
            "received",
            "Proxy-Anfrage empfangen und validiert. Starte lokale Qwen Sub-Agenten.",
            {"model": body.get("model", MODEL_NAME)},
        )
    ]

    async with httpx.AsyncClient() as client:
        tasks = [
            call_sub_agent(client, body, name, instruction)
            for name, instruction in SUB_AGENTS.items()
        ]
        progress_messages.append(
            _format_chat_progress_message(
                "local_agents_started",
                "5 lokale Qwen Sub-Agenten werden parallel ausgeführt.",
                {"agents": list(SUB_AGENTS.keys())},
            )
        )
        results = await asyncio.gather(*tasks)

    completed = sum(1 for result in results if result.get("status") == "ok")
    progress_messages.append(
        _format_chat_progress_message(
            "local_agents_finished",
            f"{completed}/{len(results)} lokale Qwen Sub-Agenten abgeschlossen.",
            {
                "agent_status": [
                    {
                        "agent": result.get("agent_key"),
                        "status": result.get("status"),
                        "duration_seconds": result.get("duration_seconds"),
                    }
                    for result in results
                ]
            },
        )
    )

    combined_response_text = (
        "## 🚀 DX Spark Parallel Multi-Agent Response\n\n"
        f"_5 Sub-Agenten parallel ausgeführt, Dauer: {time.perf_counter() - start_time:.2f}s._\n\n"
        "\n---\n".join(result["content"] for result in results)
    )

    progress_messages.append(
        _format_chat_progress_message(
            "cloud_review_prepared",
            "Cloud Final Reviewer wird vorbereitet.",
            {
                "enabled": CLOUD_REVIEW_ENABLED,
                "model": CLOUD_REVIEW_MODEL,
            },
        )
    )

    async with httpx.AsyncClient() as cloud_client:
        cloud_review_result = await call_cloud_final_reviewer(
            cloud_client,
            body,
            results,
            combined_response_text,
        )

    progress_messages.append(
        _format_chat_progress_message(
            "cloud_review_finished",
            "Cloud Final Reviewer abgeschlossen.",
            {
                "status": cloud_review_result.get("status"),
                "rejected": cloud_review_result.get("rejected"),
                "duration_seconds": cloud_review_result.get("duration_seconds"),
            },
        )
    )

    all_results = [*results, cloud_review_result]

    if cloud_review_result.get("rejected"):
        combined_response_text = (
            "## 🚫 CLOUD FINAL REVIEWER REJECTED THE LOCAL Qwen OUTPUT\n\n"
            "Die lokale Qwen-Gesamtumsetzung wurde vom konfigurierbaren Cloud-Final-Reviewer zurückgewiesen. "
            "Bitte die Blocker und empfohlenen Fixes beachten:\n\n"
            f"{cloud_review_result['content']}\n\n"
            "## Ursprüngliche lokale Qwen-Antworten\n\n"
            "\n---\n".join(result["content"] for result in results)
        )
    else:
        combined_response_text = (
            "## 🚀 DX Spark Parallel Multi-Agent Response\n\n"
            f"_5 Sub-Agenten parallel ausgeführt, Dauer: {time.perf_counter() - start_time:.2f}s._\n\n"
            "\n---\n".join(result["content"] for result in all_results)
        )

    progress_messages.append(
        _format_chat_progress_message(
            "completed",
            "Proxy-Antwort fertiggestellt.",
            {
                "duration_seconds": time.perf_counter() - start_time,
                "rejected": cloud_review_result.get("rejected"),
            },
        )
    )

    return {
        "combined_response_text": "".join(progress_messages) + combined_response_text,
        "results": all_results,
        "duration_seconds": time.perf_counter() - start_time,
    }


async def _stream_events(
    request: Request,
    body: Dict[str, Any],
):
    model = body.get("model", MODEL_NAME)
    start_time = time.perf_counter()
    yield _format_openai_stream_chunk(
        model,
        _format_chat_progress_message(
            "received",
            "Proxy-Anfrage empfangen und validiert. Starte lokale Qwen Sub-Agenten.",
            {"model": model},
        ),
        include_role=True,
    )
    yield _format_sse_event(
        "received",
        "Proxy-Anfrage empfangen und validiert. Starte lokale Qwen Sub-Agenten.",
        {"model": model},
    )

    async with httpx.AsyncClient() as client:
        tasks = [
            call_sub_agent(client, body, name, instruction)
            for name, instruction in SUB_AGENTS.items()
        ]
        yield _format_sse_event(
            "local_agents_started",
            "5 lokale Qwen Sub-Agenten werden parallel ausgeführt.",
            {"agents": list(SUB_AGENTS.keys())},
        )
        results = await asyncio.gather(*tasks)

    completed = sum(1 for result in results if result.get("status") == "ok")
    yield _format_sse_event(
        "local_agents_finished",
        f"{completed}/{len(results)} lokale Qwen Sub-Agenten abgeschlossen.",
        {
            "agent_status": [
                {
                    "agent": result.get("agent_key"),
                    "status": result.get("status"),
                    "duration_seconds": result.get("duration_seconds"),
                }
                for result in results
            ]
        },
    )

    combined_response_text = (
        "## 🚀 DX Spark Parallel Multi-Agent Response\n\n"
        f"_5 Sub-Agenten parallel ausgeführt, Dauer: {time.perf_counter() - start_time:.2f}s._\n\n"
        "\n---\n".join(result["content"] for result in results)
    )

    yield _format_sse_event(
        "cloud_review_prepared",
        "Cloud Final Reviewer wird vorbereitet.",
        {
            "enabled": CLOUD_REVIEW_ENABLED,
            "model": CLOUD_REVIEW_MODEL,
        },
    )

    async with httpx.AsyncClient() as cloud_client:
        cloud_review_result = await call_cloud_final_reviewer(
            cloud_client,
            body,
            results,
            combined_response_text,
        )

    yield _format_sse_event(
        "cloud_review_finished",
        "Cloud Final Reviewer abgeschlossen.",
        {
            "status": cloud_review_result.get("status"),
            "rejected": cloud_review_result.get("rejected"),
            "duration_seconds": cloud_review_result.get("duration_seconds"),
        },
    )

    all_results = [*results, cloud_review_result]

    if cloud_review_result.get("rejected"):
        combined_response_text = (
            "## 🚫 CLOUD FINAL REVIEWER REJECTED THE LOCAL Qwen OUTPUT\n\n"
            "Die lokale Qwen-Gesamtumsetzung wurde vom konfigurierbaren Cloud-Final-Reviewer zurückgewiesen. "
            "Bitte die Blocker und empfohlenen Fixes beachten:\n\n"
            f"{cloud_review_result['content']}\n\n"
            "## Ursprüngliche lokale Qwen-Antworten\n\n"
            "\n---\n".join(result["content"] for result in results)
        )
    else:
        combined_response_text = (
            "## 🚀 DX Spark Parallel Multi-Agent Response\n\n"
            f"_5 Sub-Agenten parallel ausgeführt, Dauer: {time.perf_counter() - start_time:.2f}s._\n\n"
            "\n---\n".join(result["content"] for result in all_results)
        )

    yield _format_sse_event(
        "completed",
        "Proxy-Antwort fertiggestellt.",
        {
            "duration_seconds": time.perf_counter() - start_time,
            "rejected": cloud_review_result.get("rejected"),
        },
    )


def _build_response_payload(
    body: Dict[str, Any],
    combined_response_text: str,
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-spark-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", MODEL_NAME),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": combined_response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": _sum_usage(results),
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    await _auth_or_raise(request)

    body = await request.json()

    if "messages" not in body:
        raise HTTPException(status_code=400, detail="Invalid OpenAI payload: 'messages' required.")

    if body.get("stream"):
        return StreamingResponse(
            _stream_events(request, body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    streamed = await _stream_chat_completion(body)
    response_payload = _build_response_payload(
        body,
        streamed["combined_response_text"],
        streamed["results"],
    )

    return JSONResponse(content=response_payload)


@app.get("/v1/models")
async def list_models(request: Request):
    await _auth_or_raise(request)

    """Proxy-Modelliste. Falls vLLM erreichbar ist, werden dessen Modelle ergänzt."""
    models = [{"id": MODEL_NAME, "object": "model", "owned_by": "vllm"}]

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(VLLM_MODELS_URL)
            if response.status_code == 200:
                data = response.json()
                for model in data.get("data", []):
                    if model.get("id") not in {item.get("id") for item in models}:
                        models.append(model)
    except Exception:
        pass

    return JSONResponse(content={"object": "list", "data": models})


@app.get("/healthz")
async def healthz(request: Request):
    return JSONResponse(
        content={
            "status": "ok",
            "agents": len(SUB_AGENTS),
            "proxy_auth_enabled": PROXY_AUTH_ENABLED,
            "cloud_review_enabled": CLOUD_REVIEW_ENABLED,
            "cloud_review_model": CLOUD_REVIEW_MODEL,
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)
