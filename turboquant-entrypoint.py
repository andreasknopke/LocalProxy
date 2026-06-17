#!/usr/bin/env python3
"""
TurboQuant vLLM Entrypoint for DGX Spark.
Patches vLLM with TurboQuant KV-Cache compression BEFORE model loads,
then starts the vLLM OpenAI API server with all passed arguments.

Usage (in Docker):
    docker run ... avarok/dgx-vllm-nvfp4-kernel:turboquant serve --host 0.0.0.0 ...
"""
import sys
import os

# ─── 1. Patch TurboQuant BEFORE vLLM loads the model ───
print("[TurboQuant] Patching vLLM attention backend...", flush=True)
try:
    from turboquant.vllm_attn_backend import enable_no_alloc

    # 3-bit keys, 2-bit values, 128 token buffer
    enable_no_alloc(key_bits=3, value_bits=2, buffer_size=128)
    print("[TurboQuant] ✓ KV-Cache compression active (keys=3bit, values=2bit)", flush=True)
except ImportError:
    print("[TurboQuant] ⚠ turboquant not installed, falling back to default KV-cache", flush=True)
except Exception as e:
    print(f"[TurboQuant] ⚠ Failed to patch: {e}, falling back to default", flush=True)

# ─── 2. Start vLLM OpenAI API server ───
# The docker CMD is passed as sys.argv[1:]; we need to prepend "api_server"
# because vLLM's entrypoints.openai.api_server.main() expects that format.
from vllm.entrypoints.openai.api_server import main as vllm_main

if __name__ == "__main__":
    # If args start with 'serve' (vllm serve CLI style), convert to api_server style
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        sys.argv = ["api_server"] + sys.argv[2:]
    elif len(sys.argv) == 1:
        # Default safe fallback – should not happen with our run script
        sys.argv = ["api_server", "--port", "8000", "--host", "0.0.0.0"]

    print(f"[TurboQuant] Starting vLLM with args: {' '.join(sys.argv)}", flush=True)
    vllm_main(sys.argv)
